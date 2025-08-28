#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import cv2

# bring in helpers from your progressive script
from scripts.select_lines_progressive import (
    set_global_seed, preprocess_image, ProgressiveSelector,
    load_line_cache, save_canvas_triplet, export_svg, _norm01,
    draw_line_mask_roi, edge_endpoints_from_coords, angle_diff_deg, edge_angle_deg
)
def _grad_maps_from_rgb01(rgb01: np.ndarray):
    """Return (grad_angle_deg [H,W], grad_mag01 [H,W]) computed from luminance."""
    # luminance
    y = (0.2126*rgb01[...,0] + 0.7152*rgb01[...,1] + 0.0722*rgb01[...,2]).astype(np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    ang = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
    mag = np.hypot(gx, gy)
    mmax = float(mag.max()) or 1.0
    mag01 = (mag / mmax).astype(np.float32)
    return ang.astype(np.float32), mag01

def _scale_lines(lines_xyxy, s: float):
    if s == 1.0:
        return lines_xyxy
    out = []
    for (x1,y1,x2,y2) in lines_xyxy:
        out.append((int(round(x1*s)), int(round(y1*s)),
                    int(round(x2*s)), int(round(y2*s))))
    return out

def _write_recipe(out_dir, image_path, size_hw, layout_args, params, sequence, seed, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    recipe = dict(
        image=os.path.basename(image_path),
        size=[int(size_hw[0]), int(size_hw[1])],
        layout=layout_args,
        params=params,
        sequence=sequence,
        seed=int(seed),
    )
    if extra:
        recipe.update(extra)
    with open(os.path.join(out_dir, 'recipe.json'), 'w', encoding='utf-8') as f:
        json.dump(recipe, f, indent=2)

def _load_color_resized(path, size_hw):
    H, W = int(size_hw[0]), int(size_hw[1])
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)  # (W,H)
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)

# -------------------------------
# Joint RGB Selector (single pass)
# -------------------------------
class JointRGBSelector:
    """
    One-pass color selection. Each candidate line is scored by the *sum* of
    ΔSSE across channels (R,G,B), plus all your usual penalties (cooldown,
    small-angle, angle histogram, density, etc.).
    """
    def __init__(self, target_rgb01, cache, params):
        # target_rgb01: float32 RGB in [0,1], where 1 = light, 0 = dark
        self.H, self.W = int(target_rgb01.shape[0]), int(target_rgb01.shape[1])
        self.cache = cache
        self.P = dict(params or {})
        # desired darkness per channel (same convention as ProgressiveSelector)
        self.T = 1.0 - target_rgb01  # [H,W,3]
        self.D = np.zeros_like(self.T, np.float32)  # accumulated darkness per channel
                # orientation maps from luminance
        self.grad_angle_deg, self.grad_mag01 = _grad_maps_from_rgb01(1.0 - self.T)  # T=1 - rgb; so 1-T = rgb
        self.color_orient_weight = float(self.P.get('color_orient_weight', 0.0))

        # per-line opacity strategy
        self.opacity_mode   = str(self.P.get('opacity_mode', 'static'))
        self.kappa_min      = float(self.P.get('k_opacity_min', 0.02))
        self.kappa_max      = float(self.P.get('k_opacity_max', 0.08))

        self.used = np.zeros(len(cache['pairs']), dtype=bool)
        self.chosen_pairs = []
        self.canvas_lines_xyxy = []

        # angle stats for penalties
        self.angle_hist = np.zeros(36, np.float32)
        self.last_line_angle = None
        self.last_line_endpoints = None
        self.nail_recent = []

        # base heuristic (length norm like in ProgressiveSelector)
        base = cache['lengths'].astype(np.float32)
        self.base_scores = (base - base.min()) / max(1e-6, float(base.max() - base.min()))

        # quick lookup
        self._pair_to_index = {}
        for idx, (a, b) in enumerate(cache['pairs']):
            a, b = int(a), int(b)
            self._pair_to_index[(a, b)] = idx
            self._pair_to_index[(b, a)] = idx

        # optional: random subset of candidates per step (for speed/variation)
        self.rand_subset_frac = float(self.P.get('rand_subset_frac', 0.0))
        self.rand_subset_min  = int(self.P.get('rand_subset_min', 4096))

        # fast knobs
        self.k_opacity = float(self.P.get('k_opacity', 0.05))
        self.thickness_px = int(self.P.get('thickness_px', 1))
        self.shortlist_k = int(self.P.get('shortlist_k', 1024))
        self.base_top_keep = int(self.P.get('base_top_keep', 4096))

    def _cooldown_set(self) -> set:
        K = int(self.P.get('cooldown_steps', 0))
        return set(self.nail_recent[-K:]) if K > 0 else set()

    def _candidate_pool(self):
        mask = ~self.used
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return idxs
        # optional random subsample for speed
        if self.rand_subset_frac > 0.0:
            want = max(self.rand_subset_min, int(np.ceil(len(idxs) * (1.0 - self.rand_subset_frac))))
            if want < len(idxs):
                idxs = np.random.choice(idxs, size=want, replace=False)

        # keep by base heuristic first
        topN = min(self.base_top_keep, len(idxs))
        base_vals = self.base_scores[idxs]
        part = np.argpartition(-base_vals, topN - 1)[:topN]
        return idxs[part]

    @staticmethod
    def _delta_sse_gain(R_roi, M, alpha):
        # sum_c sum( r^2 - (r - aM)^2 ) = sum_c sum(2*aM*r - (aM)^2)
        aM = alpha * M
        return float(np.sum(2.0 * aM[..., None] * R_roi - (aM[..., None] ** 2)))

    def _roi_for_idx(self, k_idx: int):
        # draw AA mask in a small ROI (like ProgressiveSelector)
        ys = self.cache['ys'][k_idx]; xs = self.cache['xs'][k_idx]
        x1, y1, x2, y2 = edge_endpoints_from_coords(ys, xs)
        ysl, xsl, M = draw_line_mask_roi(self.H, self.W, x1, y1, x2, y2,
                                         thickness=self.thickness_px, aa=True)
        return (ysl, xsl, M.astype(np.float32)), (x1, y1, x2, y2)

    def _opacity_for_line(self, base_kappa: float, R_roi: np.ndarray, M: np.ndarray, k_idx: int) -> float:
        """
        Choose per-line opacity based on self.opacity_mode.
        R_roi: residual [h,w,3]; M: mask [h,w]
        Returns scalar kappa in [kappa_min, kappa_max].
        """
        mode = self.opacity_mode
        if mode == 'static':
            return base_kappa

        if mode == 'residual':
            w = float(np.sum(M)) + 1e-6
            lum = 0.2126*R_roi[...,0] + 0.7152*R_roi[...,1] + 0.0722*R_roi[...,2]
            rbar = float(np.sum(lum * M) / w)
            k = self.kappa_min + (self.kappa_max - self.kappa_min) * np.clip(rbar, 0.0, 1.0)
            return float(k)

        if mode == 'residual_clipped':
            w = float(np.sum(M)) + 1e-6
            lum = 0.2126*R_roi[...,0] + 0.7152*R_roi[...,1] + 0.0722*R_roi[...,2]
            rbar = float(np.sum(lum * M) / w)
            rbar = np.power(np.clip(rbar, 0.0, 1.0), 0.6)
            k = self.kappa_min + (self.kappa_max - self.kappa_min) * rbar
            return float(k)

        if mode == 'length':
            L = float(self.cache['lengths'][k_idx])
            Lmin, Lmax = 1.0, float(self.cache['lengths'].max() + 1e-6)
            t = (L - Lmin) / (Lmax - Lmin)
            k = self.kappa_min + (self.kappa_max - self.kappa_min) * np.clip(t, 0.0, 1.0)
            return float(k)

        if mode == 'residual_sqrt':
            # like 'residual' but with sqrt compression to emphasize mid–high residuals
            w = float(np.sum(M)) + 1e-6
            lum = 0.2126*R_roi[...,0] + 0.7152*R_roi[...,1] + 0.0722*R_roi[...,2]
            rbar = float(np.sum(lum * M) / w)
            rbar = np.sqrt(np.clip(rbar, 0.0, 1.0))
            k = self.kappa_min + (self.kappa_max - self.kappa_min) * rbar
            return float(k)

        return base_kappa


    def run(self, num_lines: int):
        kappa = self.k_opacity
        density_w = float(self.P.get('density_weight', 0.0))
        cool_pen  = float(self.P.get('cooldown_penalty', 0.35))
        min_angle = float(self.P.get('min_angle_deg', 12.0))
        small_ang = float(self.P.get('small_angle_penalty', 0.25))
        angle_hist_pen = float(self.P.get('angle_penalty', 0.0))

        early_eps  = float(self.P.get('min_gain_eps', 1e-5))
        patience   = int(self.P.get('no_gain_patience', 3))
        no_gain    = 0

        for _ in range(int(num_lines)):
            idxs = self._candidate_pool()
            if idxs.size == 0:
                break

            # shortlist by residual need (sum RGB)
            # compute approx line need using sampled coordinates (cheap pass)
            T = self.T; D = self.D
            need_full = np.clip(T - D, 0.0, 1.0)   # [H,W,3]
            pres = []
            for k in idxs:
                ys, xs = self.cache['ys'][k], self.cache['xs'][k]
                # sum residual need over the path in all channels
                pres.append(float(np.sum(need_full[ys, xs, :])))
            pres = np.asarray(pres, np.float32)
            take = min(self.shortlist_k, len(pres))
            cand = idxs[np.argpartition(-pres, take-1)[:take]]

            # exact scoring (ΔSSE across RGB) + penalties
            best_idx = -1
            best_score = -1e30
            best_gain = 0.0
            cool = self._cooldown_set()

            for k_idx in cand:
                (ysl, xsl, M), (x1,y1,x2,y2) = self._roi_for_idx(k_idx)
                D_roi = D[ysl, xsl, :]         # [h,w,3]
                T_roi = T[ysl, xsl, :]
                R_pre = np.clip(T_roi - D_roi, 0.0, 1.0)

                          # choose per-line opacity (may depend on residual/length)
                k_line = self._opacity_for_line(kappa, R_pre, M, k_idx)

                # ΔSSE across channels with per-line k
                gain = self._delta_sse_gain(R_pre, M, k_line)

                # orientation alignment bonus from luminance gradient
                orient_bonus = 0.0
                if self.color_orient_weight > 0.0:
                    ga = self.grad_angle_deg[ysl, xsl]   # [h,w]
                    gm = self.grad_mag01[ysl, xsl]       # [h,w] in [0,1]
                    # line angle:
                    ang = edge_angle_deg(x1, y1, x2, y2)
                    # difference of tangent direction: preferred when line is tangent to edges
                    diff = np.abs((((ga + 90.0) - ang + 180.0) % 360.0) - 180.0)
                    align = 1.0 - np.clip(diff / 90.0, 0.0, 1.0)  # 1 best, 0 worst
                    # use luminance residual to weight useful improvement
                    lum_res = 0.2126*R_pre[...,0] + 0.7152*R_pre[...,1] + 0.0722*R_pre[...,2]
                    orient_bonus = float(np.sum(lum_res * gm * align * M))

                dens_term = float(np.sum(self.D[ysl, xsl, 0] * M)) if density_w > 0.0 else 0.0

                score = (gain
                         + self.color_orient_weight * orient_bonus
                         - density_w * dens_term)

                # cooldown / small-angle / histogram penalties
                i, jn = int(self.cache['pairs'][k_idx][0]), int(self.cache['pairs'][k_idx][1])
                if i in cool or jn in cool:
                    score -= cool_pen

                ang = edge_angle_deg(x1, y1, x2, y2)
                if self.last_line_endpoints is not None and self.chosen_pairs:
                    last_i, last_j = self.chosen_pairs[-1]
                    shares = (i == last_i or i == last_j or jn == last_i or jn == last_j)
                    if shares:
                        diffang = angle_diff_deg(ang, self.last_line_angle if self.last_line_angle is not None else ang)
                        if diffang < min_angle:
                            score -= small_ang * (1.0 - diffang / max(1e-6, min_angle))

                if angle_hist_pen > 0.0:
                    bin_ = int(np.round(ang / 10.0)) % 36
                    crowd = self.angle_hist[bin_]
                    score -= angle_hist_pen * (crowd / (len(self.chosen_pairs) + 1.0))

                if score > best_score:
                    best_score = score
                    best_idx   = k_idx
                    best_gain  = gain

            # early stop based on ΔSSE gain
            if best_gain <= early_eps:
                no_gain += 1
                if no_gain >= patience:
                    break
            else:
                no_gain = 0

            if best_idx < 0:
                break

            # commit best line using the SAME per-line k as used to score
            (ysl, xsl, M), (x1,y1,x2,y2) = self._roi_for_idx(best_idx)
            D_roi = self.D[ysl, xsl, :]
            T_roi = self.T[ysl, xsl, :]
            R_pre = np.clip(T_roi - D_roi, 0.0, 1.0)
            k_line_commit = self._opacity_for_line(kappa, R_pre, M, best_idx)

            self.D[ysl, xsl, :] = self.D[ysl, xsl, :] + k_line_commit * M[..., None] * (1.0 - self.D[ysl, xsl, :])
            self.D = np.clip(self.D, 0.0, 1.0)
            self.used[best_idx] = True

            ij = (int(self.cache['pairs'][best_idx][0]), int(self.cache['pairs'][best_idx][1]))
            self.chosen_pairs.append(ij)
            self.canvas_lines_xyxy.append((x1, y1, x2, y2))
            self.last_line_endpoints = (x1, y1, x2, y2)
            self.last_line_angle = edge_angle_deg(x1, y1, x2, y2)
            self.angle_hist[int(np.round(self.last_line_angle / 10.0)) % 36] += 1.0
            self.nail_recent.extend(list(ij))


        return self.chosen_pairs, self.canvas_lines_xyxy, self.D  # D is [H,W,3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='gray', choices=['gray','color'],
                    help='gray: single-pass grayscale; color: color pipeline (perchannel or joint)')
    ap.add_argument('--color_strategy', default='joint', choices=['perchannel','joint'],
                    help='In color mode: three separate passes (perchannel) or single joint RGB scorer (joint).')

    # I/O
    ap.add_argument('--image', required=True)
    ap.add_argument('--out_dir', default='outputs_unified')
    ap.add_argument('--export_svg', action='store_true')
    ap.add_argument('--svg_stroke', type=float, default=1.0)
    ap.add_argument('--svg_stroke_scaled', action='store_true')

    # geometry / cache
    ap.add_argument('--nail_shape', default='circle')
    ap.add_argument('--num_nails', type=int, default=360)
    ap.add_argument('--num_sectors', type=int, default=12)
    ap.add_argument('--min_dist', type=int, default=30)

    # preprocess / size
    ap.add_argument('--size', type=int, nargs=2, default=[400, 400], help='H W')
    ap.add_argument('--gamma', type=float, default=1.9)
    ap.add_argument('--clahe_clip', type=float, default=2.0)
    ap.add_argument('--clahe_grid', type=int, default=8)
    ap.add_argument('--bilateral_d', type=int, default=0)
    ap.add_argument('--sharpen', type=float, default=0.0)
    ap.add_argument('--invert', action='store_true')
    ap.add_argument('--circle_mask', action='store_true')

    # selection / scoring
    ap.add_argument('--num_lines', type=int, default=500)
    ap.add_argument('--k_opacity', type=float, default=0.05)
    ap.add_argument('--thickness_px', type=int, default=1)
    ap.add_argument('--shortlist_mode', default='sector_need',
                    choices=['need','sector_need','base','random','sector_random'])
    ap.add_argument('--shortlist_k', type=int, default=1024)
    ap.add_argument('--base_top_keep', type=int, default=4096)
    ap.add_argument('--cooldown_steps', type=int, default=6)
    ap.add_argument('--cooldown_penalty', type=float, default=0.35)
    ap.add_argument('--min_angle_deg', type=float, default=12.0)
    ap.add_argument('--small_angle_penalty', type=float, default=0.25)
    ap.add_argument('--angle_penalty', type=float, default=0.0)  # optionally used by joint selector

    # progressive-only knobs (ignored by joint scorer but kept for parity)
    ap.add_argument('--scorer', default='hybrid', choices=['fast','delta_sse','hybrid'])
    ap.add_argument('--hybrid_topk', type=int, default=256)
    ap.add_argument('--need_gamma', type=float, default=1.0)
    ap.add_argument('--need_weight', type=float, default=1.0)
    ap.add_argument('--density_weight', type=float, default=0.0)
    ap.add_argument('--orient_weight', type=float, default=0.0)
    ap.add_argument('--early_stop_window', type=int, default=50)
    ap.add_argument('--early_stop_improve', type=float, default=1e-3)
    ap.add_argument('--min_gain_eps', type=float, default=1e-5)
    ap.add_argument('--no_gain_patience', type=int, default=3)

    # render scaling
    ap.add_argument('--render_scale', type=float, default=1.0)
    ap.add_argument('--render_thickness', type=int, default=1)

    # randomness/subsample for joint scorer
    ap.add_argument('--rand_subset_frac', type=float, default=0.0)
    ap.add_argument('--rand_subset_min', type=int, default=4096)

    ap.add_argument('--seed', type=int, default=123)
    # orientation for joint color
    ap.add_argument('--color_orient_weight', type=float, default=0.0,
                    help='Joint color: weight for orientation alignment bonus (0=off)')

    # per-line opacity strategy for joint color
    ap.add_argument(
    '--opacity_mode',
    choices=['static','residual','length','residual_clipped','residual_sqrt'],
    default='static',
    help='Opacity mode for line rendering'
)

    ap.add_argument('--k_opacity_min', type=float, default=0.02,
                    help='Joint color: min per-line opacity when adaptive modes are used.')
    ap.add_argument('--k_opacity_max', type=float, default=0.08,
                    help='Joint color: max per-line opacity when adaptive modes are used.')


    args = ap.parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    H, W = int(args.size[0]), int(args.size[1])

    # prepare cache
    cache, _ = load_line_cache(
        (H, W), args.nail_shape, args.num_nails, args.min_dist, args.num_sectors
    )
    if cache is None:
        raise FileNotFoundError('No line cache found. Run scripts.build_line_cache first!')

    params = dict(
        k_opacity=args.k_opacity,
        thickness_px=args.thickness_px,
        shortlist_k=args.shortlist_k,
        base_top_keep=args.base_top_keep,
        cooldown_steps=args.cooldown_steps,
        cooldown_penalty=args.cooldown_penalty,
        min_angle_deg=args.min_angle_deg,
        small_angle_penalty=args.small_angle_penalty,
        need_gamma=args.need_gamma,
        need_weight=args.need_weight,
        density_weight=args.density_weight,
        orient_weight=args.orient_weight,
        shortlist_mode=args.shortlist_mode,
        scorer=args.scorer,
        hybrid_topk=args.hybrid_topk,
        early_stop_window=args.early_stop_window,
        early_stop_improve=args.early_stop_improve,
        min_gain_eps=args.min_gain_eps,
        no_gain_patience=args.no_gain_patience,
        angle_penalty=args.angle_penalty,
        rand_subset_frac=getattr(args, 'rand_subset_frac', 0.0),
        rand_subset_min=getattr(args, 'rand_subset_min', 4096),
        color_orient_weight= args.color_orient_weight,
        opacity_mode= args.opacity_mode,
        k_opacity_min= args.k_opacity_min,
        k_opacity_max= args.k_opacity_max,

    )

    layout_args = dict(
        shape=args.nail_shape, num_nails=args.num_nails,
        num_sectors=args.num_sectors, min_dist=args.min_dist
    )

    # ---------- GRAY ----------
    if args.mode == 'gray':
        raw = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(args.image)
        target = preprocess_image(
            raw,
            size=(H, W),
            gamma=args.gamma, clahe_clip=args.clahe_clip, clahe_grid=args.clahe_grid,
            bilateral_d=args.bilateral_d, sharpen=args.sharpen, invert=args.invert,
            apply_circle=(args.circle_mask and args.nail_shape == 'circle'),
        )

        selector = ProgressiveSelector(target_img=target, cache=cache, params=params)
        pairs, lines, dark = selector.run(num_lines=args.num_lines)

        s = max(1.0, float(args.render_scale))
        H2, W2 = int(round(H*s)), int(round(W*s))
        scaled_lines = _scale_lines(lines, s)
        save_canvas_triplet(args.out_dir, scaled_lines,
                            cv2.resize(dark, (W2, H2), interpolation=cv2.INTER_AREA),
                            step=None, thickness=max(1, int(round(args.render_thickness * s))))

        if args.export_svg:
            stroke = args.svg_stroke * (s if args.svg_stroke_scaled else 1.0)
            export_svg(os.path.join(args.out_dir, 'lines.svg'),
                       scaled_lines, w=W2, h=H2, stroke_px=stroke)

        _write_recipe(args.out_dir, args.image, (H, W), layout_args, params, pairs, args.seed,
                      extra=dict(mode='gray'))
        print(f'✅ Grayscale output saved in: {args.out_dir}')
        return

    # ---------- COLOR ----------
    if args.color_strategy == 'perchannel':
        rgb = _load_color_resized(args.image, (H, W))
        targets = {'R': 1.0 - rgb[..., 0], 'G': 1.0 - rgb[..., 1], 'B': 1.0 - rgb[..., 2]}
        results = {}
        for ch in ['R','G','B']:
            sel = ProgressiveSelector(target_img=(1.0 - targets[ch]), cache=cache, params=params)
            pairs, lines, dark = sel.run(num_lines=args.num_lines)
            results[ch] = dict(pairs=pairs, lines=lines, dark=dark)

            ch_dir = os.path.join(args.out_dir, f'chan_{ch}')
            os.makedirs(ch_dir, exist_ok=True)
            save_canvas_triplet(ch_dir, lines, dark, step=None, thickness=args.thickness_px)
            with open(os.path.join(ch_dir, 'recipe.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(channel=ch, pairs=pairs, size=[H,W], params=params), f, indent=2)

        # combined preview as black lines
        s = max(1.0, float(args.render_scale))
        H2, W2 = int(round(H*s)), int(round(W*s))
        canvas = np.ones((H2, W2, 3), np.uint8) * 255
        tpx = max(1, int(round(args.thickness_px * s)))
        for ch in ['R','G','B']:
            for (x1,y1,x2,y2) in results[ch]['lines']:
                p1 = (int(round(x1*s)), int(round(y1*s)))
                p2 = (int(round(x2*s)), int(round(y2*s)))
                cv2.line(canvas, p1, p2, (0,0,0), thickness=tpx, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out_dir, 'color_lines_preview.png'), canvas)

        if args.export_svg:
            def export_svg_color(path, lines_by_ch, w, h, stroke_px=1.0):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
                parts = [header, '  <g fill="none" stroke-linecap="round" stroke-linejoin="round">\n']
                color_map = {'R':'#ff0000','G':'#00cc00','B':'#0066ff'}
                for ch in ['R','G','B']:
                    parts.append(f'    <g stroke="{color_map[ch]}" stroke-width="{stroke_px}">\n')
                    for (x1,y1,x2,y2) in lines_by_ch[ch]:
                        parts.append(f'      <line x1="{int(round(x1*s))}" y1="{int(round(y1*s))}" x2="{int(round(x2*s))}" y2="{int(round(y2*s))}" />\n')
                    parts.append('    </g>\n')
                parts.append('  </g>\n</svg>\n')
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(parts)
            stroke = args.svg_stroke * (s if args.svg_stroke_scaled else 1.0)
            export_svg_color(os.path.join(args.out_dir, 'color_lines.svg'),
                             {ch:results[ch]['lines'] for ch in ['R','G','B']},
                             W2, H2, stroke_px=stroke)

        _write_recipe(args.out_dir, args.image, (H, W), layout_args, params, [],
                      args.seed, extra=dict(mode='color', strategy='perchannel', channels=['R','G','B']))
        print(f'✅ Color (per-channel) output saved in: {args.out_dir}')
        return

    # --- joint RGB ---
    rgb = _load_color_resized(args.image, (H, W))
    selector = JointRGBSelector(target_rgb01=rgb, cache=cache, params=params)
    pairs, lines, D_rgb = selector.run(num_lines=args.num_lines)

    # save previews
    s = max(1.0, float(args.render_scale))
    H2, W2 = int(round(H*s)), int(round(W*s))
    scaled_lines = _scale_lines(lines, s)

    # accumulate “darkness” proxy for preview (use luminance)
    dark_luma = 0.2126*D_rgb[...,0] + 0.7152*D_rgb[...,1] + 0.0722*D_rgb[...,2]
    save_canvas_triplet(args.out_dir, scaled_lines,
                        cv2.resize(dark_luma, (W2, H2), interpolation=cv2.INTER_AREA),
                        step=None, thickness=max(1, int(round(args.render_thickness * s))))

    if args.export_svg:
        stroke = args.svg_stroke * (s if args.svg_stroke_scaled else 1.0)
        export_svg(os.path.join(args.out_dir, 'lines.svg'),
                   scaled_lines, w=W2, h=H2, stroke_px=stroke)

    _write_recipe(args.out_dir, args.image, (H, W), layout_args, params, pairs, args.seed,
                  extra=dict(mode='color', strategy='joint'))
    print(f'✅ Color (joint RGB) output saved in: {args.out_dir}')

if __name__ == '__main__':
    main()
