#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, glob, time, csv
import numpy as np
import cv2

# Reuse your progressive pipeline pieces
from scripts.select_lines_progressive import (
    set_global_seed, preprocess_image, ProgressiveSelector,
    export_svg, save_canvas_triplet, scale_lines
)
from utils.line_cache import load_line_cache
from scripts.select_lines_progressive import load_edge_ranker  # for checkpoints

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _load_gray(path, size_hw, gamma, clahe_clip, clahe_grid, bilateral_d, sharpen, invert, circle_mask, nail_shape):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    return preprocess_image(
        raw,
        size=(int(size_hw[0]), int(size_hw[1])),
        gamma=gamma, clahe_clip=clahe_clip, clahe_grid=clahe_grid,
        bilateral_d=bilateral_d, sharpen=sharpen, invert=invert,
        apply_circle=(circle_mask and nail_shape == 'circle'),
    )

def _mse(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def _psnr(a, b, data_range=1.0):
    mse = _mse(a, b)
    if mse <= 1e-12: return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))

def _render_and_save(out_dir, lines_xyxy, darkness_map, render_scale=1.0, render_thickness=1, svg_stroke=1.0, svg_stroke_scaled=False):
    H, W = darkness_map.shape
    s = max(1.0, float(render_scale))
    H2, W2 = int(round(H*s)), int(round(W*s))
    scaled_lines = scale_lines(lines_xyxy, s)
    dark_scaled  = cv2.resize(darkness_map, (W2, H2), interpolation=cv2.INTER_AREA)
    _ensure_dir(out_dir)
    save_canvas_triplet(out_dir, scaled_lines, dark_scaled, step=None, thickness=max(1, int(round(render_thickness*s))))
    svg_path = os.path.join(out_dir, 'lines.svg')
    if svg_stroke_scaled:
        svg_stroke *= s
    export_svg(svg_path, scaled_lines, w=W2, h=H2, stroke_px=svg_stroke)
    return os.path.join(out_dir, 'simulated_from_darkness.png'), os.path.join(out_dir, 'simulated_result_lines.png')

def _make_collage(items, out_path, label_height=28, gap=8):
    """
    items: list of (title, path_to_image_png) — images must be same size
    """
    imgs = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for title, p in items:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: continue
        H, W, _ = img.shape
        banner = np.ones((label_height, W, 3), np.uint8) * 255
        cv2.putText(banner, title, (8, int(label_height*0.7)), font, 0.6, (0,0,0), 1, cv2.LINE_AA)
        stacked = np.vstack([banner, img])
        imgs.append(stacked)
    if not imgs: return
    H = max(i.shape[0] for i in imgs)
    # pad heights
    imgs = [np.pad(i, ((0, H - i.shape[0]), (0,0), (0,0)), constant_values=255) for i in imgs]
    spacer = np.ones((H, gap, 3), np.uint8) * 255
    strip = []
    for k, im in enumerate(imgs):
        strip.append(im)
        if k < len(imgs) - 1:
            strip.append(spacer.copy())
    out = np.hstack(strip)
    cv2.imwrite(out_path, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='+', required=True,
                    help='Images or glob(s), e.g. data\\A*.png')
    ap.add_argument('--rankers', nargs='+', required=True,
                    help='List like: none ckpts\\edge_ranker_pairwise.pt ckpts\\edge_ranker_reg.pt ckpts\\edge_ranker_listwise.pt')
    ap.add_argument('--out_root', default='benchmarks')
    ap.add_argument('--num_lines', type=int, default=500)

    # geometry / cache
    ap.add_argument('--nail_shape', default='circle')
    ap.add_argument('--num_nails', type=int, default=360)
    ap.add_argument('--num_sectors', type=int, default=12)
    ap.add_argument('--min_dist', type=int, default=30)

    # preprocess
    ap.add_argument('--size', type=int, nargs=2, default=[400, 400], help='H W')
    ap.add_argument('--gamma', type=float, default=1.9)
    ap.add_argument('--clahe_clip', type=float, default=2.0)
    ap.add_argument('--clahe_grid', type=int, default=8)
    ap.add_argument('--bilateral_d', type=int, default=0)
    ap.add_argument('--sharpen', type=float, default=0.0)
    ap.add_argument('--invert', action='store_true')
    ap.add_argument('--circle_mask', action='store_true')

    # selection (shared across rankers)
    ap.add_argument('--scorer', default='hybrid', choices=['fast','delta_sse','hybrid'])
    ap.add_argument('--hybrid_topk', type=int, default=256)
    ap.add_argument('--nn_weight', type=float, default=0.8)
    ap.add_argument('--need_weight', type=float, default=0.6)
    ap.add_argument('--need_gamma', type=float, default=1.0)
    ap.add_argument('--density_weight', type=float, default=0.0)
    ap.add_argument('--k_opacity', type=float, default=0.05)
    ap.add_argument('--thickness_px', type=int, default=1)
    ap.add_argument('--shortlist_mode', default='need', choices=['need','sector_need','base','random','sector_random'])
    ap.add_argument('--shortlist_k', type=int, default=512)
    ap.add_argument('--base_top_keep', type=int, default=4096)
    ap.add_argument('--cooldown_steps', type=int, default=6)
    ap.add_argument('--cooldown_penalty', type=float, default=0.35)
    ap.add_argument('--min_angle_deg', type=float, default=12.0)
    ap.add_argument('--small_angle_penalty', type=float, default=0.25)
    ap.add_argument('--angle_penalty', type=float, default=0.0)
    ap.add_argument('--center_relief', type=float, default=0.0)
    ap.add_argument('--seed_long_chords', type=int, default=24)
    ap.add_argument('--seed_length_q', type=float, default=0.88)
    ap.add_argument('--early_stop_window', type=int, default=50)
    ap.add_argument('--early_stop_improve', type=float, default=1e-3)
    ap.add_argument('--min_gain_eps', type=float, default=1e-5)
    ap.add_argument('--no_gain_patience', type=int, default=3)

    # rendering
    ap.add_argument('--render_scale', type=float, default=1.0)
    ap.add_argument('--render_thickness', type=int, default=1)
    ap.add_argument('--svg_stroke', type=float, default=1.0)
    ap.add_argument('--svg_stroke_scaled', action='store_true')

    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    set_global_seed(args.seed)

    # expand globs
    img_list = []
    for pat in args.images:
        if any(ch in pat for ch in '*?['):
            img_list += glob.glob(pat)
        else:
            img_list.append(pat)
    img_list = sorted(list(dict.fromkeys(img_list)))  # unique, stable order
    if not img_list:
        raise RuntimeError('No images found for provided patterns.')

    H, W = int(args.size[0]), int(args.size[1])
    cache, _ = load_line_cache((H, W), args.nail_shape, args.num_nails, args.min_dist, args.num_sectors)
    if cache is None:
        raise FileNotFoundError('No line cache found. Run scripts.build_line_cache first!')

    # CSV metrics
    _ensure_dir(args.out_root)
    csv_path = os.path.join(args.out_root, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['image','ranker','lines','seconds','mse','psnr'])

        for img_path in img_list:
            target = _load_gray(
                img_path, (H, W),
                args.gamma, args.clahe_clip, args.clahe_grid,
                args.bilateral_d, args.sharpen, args.invert,
                args.circle_mask, args.nail_shape
            )
            desired_dark = 1.0 - target

            # prepare collage items per image
            collage_items = []

            for ranker_path in args.rankers:
                tag = ('none' if ranker_path.lower() == 'none' else os.path.splitext(os.path.basename(ranker_path))[0])
                out_dir = _ensure_dir(os.path.join(args.out_root, tag, os.path.splitext(os.path.basename(img_path))[0]))

                # load edge ranker (optional)
                edge_ranker = None; norm_mean = None; norm_std = None
                if ranker_path.lower() != 'none':
                    edge_ranker, norm_mean, norm_std, _ = load_edge_ranker(ranker_path)

                params = dict(
                    k_opacity=args.k_opacity,
                    thickness_px=args.thickness_px,
                    shortlist_k=args.shortlist_k,
                    base_top_keep=args.base_top_keep,
                    need_gamma=args.need_gamma,
                    need_weight=args.need_weight,
                    density_weight=args.density_weight,
                    cooldown_steps=args.cooldown_steps,
                    cooldown_penalty=args.cooldown_penalty,
                    min_angle_deg=args.min_angle_deg,
                    small_angle_penalty=args.small_angle_penalty,
                    seed_long_chords=args.seed_long_chords,
                    seed_length_q=args.seed_length_q,
                    early_stop_window=args.early_stop_window,
                    early_stop_improve=args.early_stop_improve,
                    render_thickness=args.render_thickness,
                    snapshots_every=0,
                    fast_score=False,
                    nn_weight=args.nn_weight,
                    score_norm=True,
                    dyn_len_norm=True,
                    center_relief=args.center_relief,
                    angle_penalty=args.angle_penalty,
                    orient_weight=0.0,
                    shortlist_mode=args.shortlist_mode,
                    scorer=args.scorer,
                    hybrid_topk=args.hybrid_topk,
                    min_gain_eps=args.min_gain_eps,
                    no_gain_patience=args.no_gain_patience,
                )

                selector = ProgressiveSelector(
                    target_img=target,
                    cache=cache,
                    edge_ranker=edge_ranker,
                    device='cpu',
                    params=params,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                )

                t0 = time.time()
                pairs, lines_xyxy, dark = selector.run(num_lines=args.num_lines, out_dir=None, snapshots_every=0)
                dt = time.time() - t0

                # Save previews/SVG and compute metrics against desired darkness
                recon_path, lines_img_path = _render_and_save(
                    out_dir, lines_xyxy, dark,
                    render_scale=args.render_scale,
                    render_thickness=args.render_thickness,
                    svg_stroke=args.svg_stroke,
                    svg_stroke_scaled=args.svg_stroke_scaled
                )

                # compare the *darkness* map to desired darkness (both 0..1)
                Hs, Ws = dark.shape
                mse = _mse(dark, desired_dark)
                psnr = _psnr(dark, desired_dark, data_range=1.0)
                w.writerow([img_path, tag, len(pairs), f'{dt:.2f}', f'{mse:.6f}', f'{psnr:.3f}'])

                collage_items.append((tag, lines_img_path))  # use the lines preview for visual comparison

            # collage for this image
            coll_path = os.path.join(args.out_root, f'collage_{os.path.splitext(os.path.basename(img_path))[0]}.png')
            _make_collage(collage_items, coll_path)
            print(f'✅ Collage: {coll_path}')

    print(f'📊 Wrote metrics: {csv_path}')
    print(f'📁 Bench results under: {args.out_root}')

if __name__ == '__main__':
    main()
