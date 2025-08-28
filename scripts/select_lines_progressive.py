#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
StrArt – Progressive line selection with perceptual preprocessing, residual-aware scoring,
soft thread attenuation, AA line masks, nail cooldown, angle penalties, shortlisting, seeding,
early stop, recipe export, and SVG export.

Compatible with your existing line cache produced by scripts.build_line_cache.
'''
import os
os.environ.setdefault('OMP_NUM_THREADS', str(max(1, os.cpu_count()//2)))

import csv
import json
import math
import time
import cv2
import torch
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))

import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from utils.line_cache import load_line_cache
from torch.serialization import safe_globals
# ---------------------------
# utils
# --------------------------
def save_or_append_npz(path, feats, gains, groups):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        X0 = z.get('X'); G0 = z.get('gains'); R0 = z.get('groups')
        X = np.concatenate([X0, feats], axis=0) if X0 is not None else feats
        G = np.concatenate([G0, gains], axis=0) if G0 is not None else gains
        R = np.concatenate([R0, groups], axis=0) if R0 is not None else groups
    else:
        X, G, R = feats, gains, groups
    np.savez_compressed(path, X=X, gains=G, groups=R)

def scale_lines(lines_xyxy, s: float):
    if s == 1.0:
        return lines_xyxy
    out = []
    for (x1, y1, x2, y2) in lines_xyxy:
        out.append((
            int(round(x1 * s)), int(round(y1 * s)),
            int(round(x2 * s)), int(round(y2 * s))
        ))
    return out

def _norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-8:
        return np.zeros_like(a, np.float32)
    return (a - mn) / (mx - mn)

def set_global_seed(seed: Optional[int] = None):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

def circle_mask(h: int, w: int, margin: float = 0.0) -> np.ndarray:
    '''1 inside circle, 0 outside.'''
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = min(cx, cy) - margin
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)

def preprocess_image(
    img_gray_u8: np.ndarray,
    size: Tuple[int, int] = (400, 400),
    gamma: float = 1.9,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    bilateral_d: int = 0,
    sharpen: float = 0.0,
    invert: bool = False,
    apply_circle: bool = False,
) -> np.ndarray:
    '''
    Returns float32 in [0,1], perceptually nicer for string art.
    '''
    assert img_gray_u8.ndim == 2, 'Expect grayscale uint8'
    img = cv2.resize(img_gray_u8, size, interpolation=cv2.INTER_AREA)

    if bilateral_d and bilateral_d > 0:
        img = cv2.bilateralFilter(img, d=bilateral_d, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=max(0.01, clahe_clip), tileGridSize=(clahe_grid, clahe_grid))
    img = clahe.apply(img)

    img_f = img.astype(np.float32) / 255.0
    img_f = np.clip(np.power(img_f, 1.0 / max(1e-6, gamma)), 0.0, 1.0)

    if sharpen > 0.0:
        blur = cv2.GaussianBlur((img_f * 255.0).astype(np.uint8), (0, 0), 1.0)
        sharp = cv2.addWeighted((img_f * 255.0).astype(np.uint8), 1.0 + sharpen, blur, -sharpen, 0)
        img_f = sharp.astype(np.float32) / 255.0

    if invert:
        img_f = 1.0 - img_f

    if apply_circle:
        mask = circle_mask(img_f.shape[0], img_f.shape[1], margin=1.0)
        img_f = img_f * mask + 1.0 * (1.0 - mask)

    return np.clip(img_f, 0.0, 1.0).astype(np.float32)

# ---------------------------
# model / features
# ---------------------------

FEATURE_KEYS = [
    'sum','mean','max','var','dens','length','sector','d_ang','cool_i','cool_j','sec_gap'
]

class EdgeRanker(nn.Module):
    def __init__(self, input_dim, hidden=64, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden), nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden, hidden), nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)



def load_edge_ranker(path: str, device: str='cpu'):
    try:
        with safe_globals([np.core.multiarray._reconstruct]):
            ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get('model', ckpt)

    # infer input dim
    in_dim = None
    if ckpt.get('norm_mean', None) is not None:
        in_dim = int(len(ckpt['norm_mean']))
    else:
        # look for first linear weight
        for k, v in sd.items():
            if k.endswith('0.weight') or k.endswith('net.0.weight'):
                in_dim = int(v.shape[1]); break
    if in_dim is None:
        raise RuntimeError('Could not infer input dimension from checkpoint')

    hidden  = int(ckpt.get('hidden', 128))
    dropout = float(ckpt.get('dropout', 0.0))
    model = EdgeRanker(input_dim=in_dim, hidden=hidden, dropout=dropout).to(device)


    if not any(k.startswith('net.') for k in sd.keys()):
        sd = {f'net.{k}': v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'⚠️ load_edge_ranker: missing={missing} unexpected={unexpected}')
    model.eval()

    norm_mean = ckpt.get('norm_mean', None)
    norm_std  = ckpt.get('norm_std', None)
    meta = {
        'feature_keys': ckpt.get('feature_keys', None),
        'gain_norm': ckpt.get('gain_norm', 'zscore'),
        'temperature': ckpt.get('temperature', 1.0),
        'group_topk': ckpt.get('group_topk', None),
    }
    return model, norm_mean, norm_std, meta



def extract_features_for_edges(cache: Dict[str, Any], darkness_map: np.ndarray) -> np.ndarray:
    '''
    Minimal placeholder; extend to match your training.
    '''
    num_edges = len(cache['pairs'])
    feats = np.zeros((num_edges, len(FEATURE_KEYS)), np.float32)
    feats[:, 0] = cache['lengths'].astype(np.float32)  # length
    feats[:, 1] = cache['sectors'].astype(np.float32)  # sector
    return feats

# ---------------------------
# geometry helpers
# ---------------------------

def edge_endpoints_from_coords(ys: np.ndarray, xs: np.ndarray) -> Tuple[int, int, int, int]:
    '''Infer (x1,y1,x2,y2) from sampled line coordinates (first/last).'''
    y1, x1 = int(ys[0]), int(xs[0])
    y2, x2 = int(ys[-1]), int(xs[-1])
    return x1, y1, x2, y2

def edge_angle_deg(x1, y1, x2, y2) -> float:
    vx, vy = (x2 - x1), (y2 - y1)
    return math.degrees(math.atan2(vy, vx)) % 360.0

def angle_diff_deg(a: float, b: float) -> float:
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d

def draw_line_mask_roi(
    h: int, w: int, x1: int, y1: int, x2: int, y2: int,
    thickness: int = 1, aa: bool = True, pad: int = 2
) -> Tuple[slice, slice, np.ndarray]:
    '''
    Create a small float mask ROI with an AA line drawn (0..1).
    '''
    x_min = max(0, min(x1, x2) - thickness - pad)
    x_max = min(w - 1, max(x1, x2) + thickness + pad)
    y_min = max(0, min(y1, y2) - thickness - pad)
    y_max = min(h - 1, max(y1, y2) + thickness + pad)
    roi = np.zeros((y_max - y_min + 1, x_max - x_min + 1), np.float32)
    p1 = (x1 - x_min, y1 - y_min)
    p2 = (x2 - x_min, y2 - y_min)
    line_type = cv2.LINE_AA if aa else cv2.LINE_8
    cv2.line(roi, p1, p2, color=1.0, thickness=max(1, int(thickness)), lineType=line_type)
    return slice(y_min, y_max + 1), slice(x_min, x_max + 1), roi

def soft_attenuation_update(dst_dark: np.ndarray, mask: np.ndarray, k: float):
    '''
    Thread stacking model: D_new = 1 - (1 - D_old) * (1 - k * mask)
    dst_dark[...] modified in-place.
    '''
    np.multiply(1.0 - dst_dark, (1.0 - k * mask), out=dst_dark)
    np.subtract(1.0, dst_dark, out=dst_dark)
    np.clip(dst_dark, 0.0, 1.0, out=dst_dark)

# ---------------------------
# rendering & export
# ---------------------------

def draw_lines_preview(canvas_lines_xyxy: List[Tuple[int,int,int,int]], hw: Tuple[int,int], thickness: int = 1):
    '''Render lines onto white canvas for preview (AA).'''
    H, W = int(hw[0]), int(hw[1])
    canvas = np.full((H, W, 3), 255, np.uint8)
    for (x1, y1, x2, y2) in canvas_lines_xyxy:
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), thickness=max(1, int(thickness)), lineType=cv2.LINE_AA)
    return canvas

def save_canvas_triplet(out_dir, canvas_lines_xyxy, darkness_map, step=None, thickness: int = 1):
    '''Save intermediate or final images.'''
    os.makedirs(out_dir, exist_ok=True)
    H, W = darkness_map.shape

    if step is None:
        lines_img = draw_lines_preview(canvas_lines_xyxy, (H, W), thickness)
        cv2.imwrite(os.path.join(out_dir, 'simulated_result_lines.png'), lines_img)

        dark_u8 = (255.0 * _norm01(darkness_map)).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, 'accumulated_darkness.png'), dark_u8)

        recon = (255.0 * (1.0 - np.clip(darkness_map, 0, 1))).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, 'simulated_from_darkness.png'), recon)
    else:
        lines_img = draw_lines_preview(canvas_lines_xyxy, (H, W), thickness)
        cv2.imwrite(os.path.join(out_dir, f'sim_{int(step):04d}.png'), lines_img)
        dark_u8 = (255.0 * _norm01(darkness_map)).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'dark_{int(step):04d}.png'), dark_u8)

def export_svg(out_path: str, lines_xyxy: List[Tuple[int,int,int,int]], w: int, h: int, stroke_px: float = 1.0):
    '''
    Minimal, dependency-free SVG export of the line sequence at pixel coordinates.
    '''
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
    style  = f'  <g fill="none" stroke="black" stroke-width="{stroke_px}" stroke-linecap="round" stroke-linejoin="round">\n'
    parts = [header, style]
    for (x1, y1, x2, y2) in lines_xyxy:
        parts.append(f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />\n')
    parts.append('  </g>\n</svg>\n')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(parts)

# ---------------------------
# selection core
# ---------------------------

class ProgressiveSelector:
    def __init__(self, target_img, cache, edge_ranker=None, device='cpu', params=None,
                 norm_mean=None, norm_std=None):
        self.target = np.clip(target_img.astype(np.float32), 0.0, 1.0)
        self.H, self.W = self.target.shape
        self.cache = cache
        self.edge_ranker = edge_ranker
        self.device = device
        self.norm_mean = norm_mean
        self.norm_std  = norm_std
                # ---- params dict (must exist before we read it) ----
        # In __init__, ensure defaults include density_weight:
        P = dict(
            nn_weight=0.8, need_weight=0.58, need_gamma=1.30,
            fast_score=True, score_norm=True, dyn_len_norm=True,
            k_opacity=0.050, thickness_px=1,
            shortlist_k=256, base_top_keep=2048,
            seed_long_chords=4, cooldown_steps=20, min_angle_deg=24,
            small_angle_penalty=0.25, cooldown_penalty=0.35,
            center_relief=0.35, angle_penalty=0.18,
            early_stop_window=50, early_stop_improve=1e-3,
            render_thickness=1, snapshots_every=0,
            density_weight=0.0,     
            scorer='hybrid',
            hybrid_topk=256,
            min_gain_eps=1e-5,
            no_gain_patience=3,
            num_sectors=int(cache.get('num_sectors', 12)),
            # ← add this
        )
        if params:
            P.update(params)
        self.P = P
        # --- orientation maps (for image-aware scoring) ---
        gx = cv2.Sobel(self.target, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.target, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy).astype(np.float32)
        theta = np.arctan2(gy, gx).astype(np.float32)  # radians, [-pi, pi]

        # normalize magnitude to [0,1] for stable weighting
        mmax = float(mag.max()) or 1.0
        self.grad_mag = (mag / mmax).astype(np.float32)
        self.grad_theta = theta

        # Precompute image gradients (orientation + strength)

        self.grad_angle = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0  # 0..360
        gm = np.hypot(gx, gy)

        # convenience flag
        self.fast_score = bool(self.P.get('fast_score', False))

        self.desired_darkness = 1.0 - self.target
        cr = float(self.P.get('center_relief', 0.0))
        if cr > 0:
            yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            cy, cx = (self.H-1)/2.0, (self.W-1)/2.0
            r = np.sqrt((yy-cy)**2 + (xx-cx)**2); r /= (r.max() + 1e-6)
            w = (1.0 - r*r)                      # smaller weight at center
            self.desired_darkness *= (1.0 - cr) + cr*w

        self.dark = np.zeros_like(self.target, np.float32)
        self.used = np.zeros(len(cache['pairs']), dtype=bool)
        self.chosen_pairs = []
        self.canvas_lines_xyxy = []

        self.nail_recent = []
        self.last_line_angle = None
        self.last_line_endpoints = None
        self.log_feats  = []   # list of [K, D] arrays
        self.log_gains  = []   # list of [K] arrays
        self.log_groups = []   # list of [K] arrays (same step id)
        self._cur_step  = 0


        self.angle_hist = np.zeros(36, np.float32)   # 10° bins
        self.angle_penalty = float(self.P.get('angle_penalty', 0.12))

        # base heuristic (do NOT call NN here; dims won’t match)
        base = cache['lengths'].astype(np.float32)
        self.base_scores = (base - base.min()) / max(1e-6, float(base.max() - base.min()))

        # (pair -> idx) lookup
        self._pair_to_index = {}
        for idx, (a, b) in enumerate(cache['pairs']):
            a, b = int(a), int(b)
            self._pair_to_index[(a, b)] = idx
            self._pair_to_index[(b, a)] = idx



    def _cooldown_set(self) -> set:
        K = self.P['cooldown_steps']
        return set(self.nail_recent[-K:]) if K > 0 else set()

    def _build_dyn_features(self, k: int, need_full: np.ndarray) -> np.ndarray:
        if self.fast_score:
            s,m,mx,v,dens,L,sec = self._line_stats_fast(need_full, k)
            x1,y1,x2,y2 = edge_endpoints_from_coords(self.cache['ys'][k], self.cache['xs'][k])
        else:
            ys_k = self.cache['ys'][k]; xs_k = self.cache['xs'][k]
            x1, y1, x2, y2 = edge_endpoints_from_coords(ys_k, xs_k)
            ysli, xsli, mask = self._get_roi_mask(k)
            vals = need_full[ysli, xsli] * mask
            s  = float(vals.sum())
            m  = float(vals.sum() / (mask.sum() + 1e-6))
            mx = float(vals.max()) if vals.size else 0.0
            v  = float(((vals - m)**2).sum() / (mask.sum() + 1e-6))
            dens = float(((1.0 - need_full[ysli, xsli]) * mask).sum() / (mask.sum() + 1e-6))
            L  = float(self.cache['lengths'][k]); sec = float(self.cache['sectors'][k])


        # dynamics (same as before)
        ang_cur = edge_angle_deg(x1, y1, x2, y2)
        if self.last_line_endpoints is None:
            d_ang = 0.0; sec_gap = 0.0
        else:
            d_ang = angle_diff_deg(ang_cur, self.last_line_angle if self.last_line_angle is not None else ang_cur)
            last_k = self.chosen_pairs[-1]
            last_sec = float(self.cache['sectors'][ self._pair_to_index.get(last_k, 0) ]) if hasattr(self, '_pair_to_index') else 0.0
            sec_gap = float(abs(sec - last_sec))
        i, jn = int(self.cache['pairs'][k][0]), int(self.cache['pairs'][k][1])
        cool_set = self._cooldown_set()
        cool_i = 1.0 if i in cool_set else 0.0
        cool_j = 1.0 if jn in cool_set else 0.0

        return np.array([s,m,mx,v,dens,L,sec,d_ang,cool_i,cool_j,sec_gap], np.float32)


    def _line_stats_fast(self, need_full: np.ndarray, k: int):
        ys = self.cache['ys'][k]; xs = self.cache['xs'][k]
        vals = need_full[ys, xs]
        s  = float(vals.sum())
        m  = float(vals.mean() if vals.size else 0.0)
        mx = float(vals.max() if vals.size else 0.0)
        v  = float(vals.var() if vals.size else 0.0)
        dens = float((1.0 - vals).mean() if vals.size else 0.0)
        L  = float(self.cache['lengths'][k])
        sec= float(self.cache['sectors'][k])
        return s, m, mx, v, dens, L, sec

    def _candidate_pool(self) -> np.ndarray:
        mask = ~self.used
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return idxs

        # NEW: global random subsample BEFORE base_top_keep
        frac = float(self.P.get('rand_subset_frac', 0.0) or 0.0)
        kmin = int(self.P.get('rand_subset_min', 4096))
        if frac > 0.0:
            keep = max(kmin, int(len(idxs) * min(1.0, max(0.0, frac))))
            if keep < len(idxs):
                idxs = np.random.choice(idxs, size=keep, replace=False)

        topN = min(self.P['base_top_keep'], len(idxs))
        base_vals = self.base_scores[idxs]
        part = np.argpartition(-base_vals, topN - 1)[:topN]
        return idxs[part]


    def _prefilter_by_need(self, need_full, keep):
        idxs = np.where(~self.used)[0]
        if keep <= 0 or keep >= len(idxs):
            return idxs
        # approximate dynamic gain along sampled coords
        gains = np.fromiter(
            (float(need_full[self.cache['ys'][k], self.cache['xs'][k]].sum()) for k in idxs),
            dtype=np.float32, count=len(idxs)
        )
        # optional: add a touch of orientation to the prefilter
        if self.P.get('orient_weight', 0.0) > 0.0:
            def orient_boost(k):
                ys, xs = self.cache['ys'][k], self.cache['xs'][k]
                ang = edge_angle_deg(*edge_endpoints_from_coords(ys, xs))
                ga  = self.grad_angle[ys, xs]; gm = self.grad_mag[ys, xs]
                diff = np.abs((((ga + 90.0) - ang + 180.0) % 360.0) - 180.0)
                align = 1.0 - np.clip(diff / 90.0, 0.0, 1.0)
                return float((need_full[ys, xs] * gm * align).sum())
            boosts = np.fromiter((orient_boost(k) for k in idxs), dtype=np.float32, count=len(idxs))
            gains += boosts
        part = np.argpartition(-gains, min(keep, len(gains))-1)[:keep]
        return idxs[part]

    def _score_candidates_dynamic(self, idxs: np.ndarray) -> Tuple[int, float]:
        """
        Pick the candidate that maximizes the exact reduction in residual error.

        Thread stacking update per pixel:
          D_new = D_old + k * M * (1 - D_old)
          R_pre_lin  = clip(T - D_old, 0, 1)
          R_post_lin = clip(T - D_new, 0, 1)
          d_lin      = R_pre_lin - R_post_lin   (>= 0)

        If need_gamma != 1:
          d_eff = (R_pre_lin**g - R_post_lin**g)

        Score:
          S = nn_weight*base + need_weight*(sum(d_eff)/L)
              + orient_weight * sum(d_eff * gm * align)
              - density_weight * sum(D_old * M)
              - cooldown/small-angle/angle-hist penalties
        """
        if len(idxs) == 0:
            return -1, 0.0

        # Common arrays/flags
        D = self.dark
        T = self.desired_darkness
        k = float(self.P['k_opacity'])
        g = float(self.P.get('need_gamma', 1.0))
        dyn_len_norm = bool(self.P.get('dyn_len_norm', True))
        ow = float(self.P.get('orient_weight', 0.0))
        density_w = float(self.P.get('density_weight', 0.0))
        cool = self._cooldown_set()

        # Residual need once
        need_full = np.clip(T - D, 0.0, 1.0)

          # ---------- shortlist (choose pool candidates)
        shortlist = min(self.P['shortlist_k'], len(idxs))
        mode = self.P.get('shortlist_mode', 'need')  # 'need' | 'sector_need' | 'base' | 'random' | 'sector_random'

        if mode == 'base':
            base_subset = self.base_scores[idxs]
            part = np.argpartition(-base_subset, shortlist - 1)[:shortlist]
            cand = idxs[part]

        elif mode == 'random':
            cand = np.random.choice(idxs, size=shortlist, replace=False)

        elif mode == 'sector_random':
            cand = self._sector_random_shortlist(idxs, shortlist, int(self.P.get('num_sectors', 12)))

        elif mode == 'sector_need':
            sectors = self.cache['sectors']
            nsec = int(self.cache.get('num_sectors', int(sectors.max()) + 1 if sectors.size else 1))
            per = max(1, shortlist // max(1, nsec))
            chosen = []
            for s in range(nsec):
                pool = idxs[sectors[idxs] == s]
                if pool.size == 0:
                    continue
                prescore = np.array(
                    [need_full[self.cache['ys'][k], self.cache['xs'][k]].sum() for k in pool],
                    dtype=np.float32
                )
                take = min(per, len(pool))
                part = np.argpartition(-prescore, take - 1)[:take]
                chosen.extend(pool[part].tolist())
            if len(chosen) < shortlist:
                remaining = np.setdiff1d(idxs, np.asarray(chosen, dtype=np.int32), assume_unique=False)
                if remaining.size:
                    prescore = np.array(
                        [need_full[self.cache['ys'][k], self.cache['xs'][k]].sum() for k in remaining],
                        dtype=np.float32
                    )
                    take = min(shortlist - len(chosen), remaining.size)
                    part = np.argpartition(-prescore, take - 1)[:take]
                    chosen.extend(remaining[part].tolist())
            cand = np.asarray(chosen[:shortlist], dtype=np.int32)

        else:  # 'need'
            prescore = np.array(
                [need_full[self.cache['ys'][k], self.cache['xs'][k]].sum() for k in idxs],
                dtype=np.float32
            )
            part = np.argpartition(-prescore, shortlist - 1)[:shortlist]
            cand = idxs[part]
        scorer_mode = str(self.P.get('scorer', 'hybrid'))
        hybrid_topk = int(self.P.get('hybrid_topk', 0))
        # In hybrid mode, reduce cand to top-K by ΔSSE-only (fast pass)
        if scorer_mode == 'hybrid' and hybrid_topk > 0 and len(cand) > hybrid_topk:
            cand = self._re_rank_by_delta_sse(cand, need_full, hybrid_topk)

        # ---------- features/NN on current residual
        feats_cand = np.stack([self._build_dyn_features(k_idx, need_full) for k_idx in cand]).astype(np.float32)

        nn_scores = None
        if self.edge_ranker is not None:
            feats = feats_cand
            if self.norm_mean is not None and self.norm_std is not None:
                feats = (feats - self.norm_mean) / (self.norm_std + 1e-6)
            with torch.no_grad():
                t = torch.from_numpy(feats).float().to(self.device)
                nn_scores = self.edge_ranker(t).cpu().numpy().astype(np.float32)
            if self.P.get('score_norm', True):
                mu = float(nn_scores.mean()); sd = float(nn_scores.std() + 1e-6)
                nn_scores = (nn_scores - mu) / sd

        # ---------- candidate scoring (exact per-pixel simulation)
        gains_cand = np.empty(len(cand), np.float32)
        best_idx = -1
        best_score = -1e30
        best_gain = 0.0

        for j, k_idx in enumerate(cand):
            ys_k = self.cache['ys'][k_idx]; xs_k = self.cache['xs'][k_idx]
            x1, y1, x2, y2 = edge_endpoints_from_coords(ys_k, xs_k)
            ysli, xsli, M = self._get_roi_mask(k_idx)

            D_old = D[ysli, xsli]
            T_roi = T[ysli, xsli]
            R_pre_lin = np.clip(T_roi - D_old, 0.0, 1.0)

            # simulate one step
            D_new = D_old + k * M * (1.0 - D_old)
            R_post_lin = np.clip(T_roi - D_new, 0.0, 1.0)
            d_lin = (R_pre_lin - R_post_lin)
            d_lin_sum = float(d_lin.sum())

            # ✅ ALWAYS compute ΔSSE once
            delta_sse_gain = self._delta_sse_gain_roi(R_pre_lin, M, k)

            # ✅ Log gain consistently
            if scorer_mode == 'delta_sse':
                gains_cand[j] = delta_sse_gain
            else:
                gains_cand[j] = d_lin_sum

            # scorer selection
            use_delta_only = (scorer_mode == 'delta_sse')
            if use_delta_only:
                base = 0.0
                d_eff_sum = delta_sse_gain
                L = float(self.cache['lengths'][k_idx]) if dyn_len_norm else 1.0
            else:
                if g != 1.0:
                    d_eff = np.power(R_pre_lin, g, dtype=np.float32) - np.power(R_post_lin, g, dtype=np.float32)
                else:
                    d_eff = d_lin
                d_eff_sum = float(d_eff.sum())
                L = float(self.cache['lengths'][k_idx]) if dyn_len_norm else 1.0
                base = nn_scores[j] if nn_scores is not None else self.base_scores[k_idx]

            # orientation reward on *effective improvement*
            orient_term = 0.0
            if ow > 0.0:
                line_ang = edge_angle_deg(x1, y1, x2, y2)
                ga = self.grad_angle[ysli, xsli]
                gm = self.grad_mag[ysli, xsli]
                diff = np.abs((((ga + 90.0) - line_ang + 180.0) % 360.0) - 180.0)
                align = 1.0 - np.clip(diff / 90.0, 0.0, 1.0)
                # weight by improvement proxy: d_eff (fast/hybrid) or R_pre*M (delta-only)
                improv = d_eff if not use_delta_only else (R_pre_lin * M)
                orient_term = float((improv * gm * align).sum())

            density_term = float((D_old * M).sum()) if density_w > 0.0 else 0.0

            score = ( (0.0 if use_delta_only else (self.P['nn_weight'] * base))
                    + self.P['need_weight'] * (d_eff_sum / (L + 1e-6))
                    + ow * orient_term
                    - density_w * density_term)

            # cooldown + small-angle + angle-hist
            i, jn = int(self.cache['pairs'][k_idx][0]), int(self.cache['pairs'][k_idx][1])
            if i in cool or jn in cool:
                score -= self.P['cooldown_penalty']

            ang = edge_angle_deg(x1, y1, x2, y2)
            if self.last_line_endpoints is not None and self.chosen_pairs:
                last_i, last_j = self.chosen_pairs[-1]
                shares = (i == last_i or i == last_j or jn == last_i or jn == last_j)
                if shares:
                    diffang = angle_diff_deg(ang, self.last_line_angle if self.last_line_angle is not None else ang)
                    if diffang < self.P['min_angle_deg']:
                        score -= self.P['small_angle_penalty'] * (1.0 - diffang / max(1e-6, self.P['min_angle_deg']))

            if getattr(self, 'angle_penalty', 0.0) > 0.0 and getattr(self, 'angle_hist', None) is not None:
                bin_ = int(np.round(ang / 10.0)) % 36
                crowd = self.angle_hist[bin_]
                score -= self.angle_penalty * (crowd / (len(self.chosen_pairs) + 1.0))

            if score > best_score:
                best_score = score
                best_idx = k_idx
                best_gain = (delta_sse_gain if use_delta_only else d_lin_sum)

        # logs once per step
        self.log_feats.append(feats_cand)
        self.log_gains.append(gains_cand)
        self.log_groups.append(np.full(len(cand), self._cur_step, np.int32))

        return best_idx, best_gain

    def _commit_edge(self, k_idx: int):
        ys_k = self.cache['ys'][k_idx]
        xs_k = self.cache['xs'][k_idx]
        x1, y1, x2, y2 = edge_endpoints_from_coords(ys_k, xs_k)
        ysli, xsli, mask = self._get_roi_mask(k_idx)
        soft_attenuation_update(self.dark[ysli, xsli], mask, self.P['k_opacity'])
        self.used[k_idx] = True
        ij = (int(self.cache['pairs'][k_idx][0]), int(self.cache['pairs'][k_idx][1]))
        self.chosen_pairs.append(ij)
        self.canvas_lines_xyxy.append((x1, y1, x2, y2))
        self.last_line_endpoints = (x1, y1, x2, y2)
        self.last_line_angle = edge_angle_deg(x1, y1, x2, y2)
        bin_ = int(np.round(self.last_line_angle / 10.0)) % 36
        self.angle_hist[bin_] += 1.0

        self.nail_recent.extend(list(ij))

    def _delta_sse_gain_roi(self, R_roi: np.ndarray, M: np.ndarray, alpha: float) -> float:
        """
        Exact ΔSSE over ROI: sum( r^2 - (r - alpha*M)^2 ) = sum( 2*alpha*M*r - (alpha*M)^2 ).
        R_roi: current residual (T - D) clipped to [0,1] at ROI
        M: AA line mask in [0,1]
        """
        aM = alpha * M
        return float(np.sum(2.0 * aM * R_roi - aM * aM))

    def _re_rank_by_delta_sse(self, cand: np.ndarray, need_full: np.ndarray, K: int) -> np.ndarray:
        D = self.dark; T = self.desired_darkness; alpha = float(self.P['k_opacity'])
        gains = np.empty(len(cand), np.float32)
        for j, k_idx in enumerate(cand):
            ysli, xsli, M = self._get_roi_mask(k_idx)
            R_roi = np.clip(T[ysli, xsli] - D[ysli, xsli], 0.0, 1.0)
            gains[j] = self._delta_sse_gain_roi(R_roi, M, alpha)
        if K >= len(cand):
            return cand
        part = np.argpartition(-gains, K-1)[:K]
        return cand[part]




    def _sector_random_shortlist(self, idxs: np.ndarray, k_shortlist: int, n_sectors: int) -> np.ndarray:
        """Ensure angular coverage by sampling ~k/S per sector."""
        if idxs.size <= k_shortlist:
            return idxs
        sectors = self.cache['sectors'][idxs]
        per = max(1, k_shortlist // max(1, n_sectors))
        picks = []
        for s in range(n_sectors):
            pool = idxs[sectors == s]
            if pool.size:
                take = min(per, pool.size)
                sel = np.random.choice(pool, size=take, replace=False)
                picks.append(sel)
        if not picks:
            return np.random.choice(idxs, size=k_shortlist, replace=False)
        sel = np.unique(np.concatenate(picks))
        if sel.size < k_shortlist:
            # fill remaining from leftovers
            left = np.setdiff1d(idxs, sel, assume_unique=False)
            if left.size:
                extra = np.random.choice(left, size=min(k_shortlist - sel.size, left.size), replace=False)
                sel = np.concatenate([sel, extra])
        # trim if we overshot
        if sel.size > k_shortlist:
            sel = np.random.choice(sel, size=k_shortlist, replace=False)
        return sel

    # def _scale_lines(lines_xyxy: List[Tuple[int,int,int,int]], s: float) -> List[Tuple[int,int,int,int]]:
    #     if s == 1.0:
    #         return lines_xyxy
    #     out = []
    #     for (x1,y1,x2,y2) in lines_xyxy:
    #         out.append((int(round(x1*s)), int(round(y1*s)),
    #                     int(round(x2*s)), int(round(y2*s))))
    #     return out

    def _get_roi_mask(self, k_idx: int):
        if all(key in self.cache for key in ('roi_y0','roi_y1','roi_x0','roi_x1','masks')):
            y0 = int(self.cache['roi_y0'][k_idx]); y1 = int(self.cache['roi_y1'][k_idx])
            x0 = int(self.cache['roi_x0'][k_idx]); x1 = int(self.cache['roi_x1'][k_idx])
            ysli = slice(y0, y1 + 1); xsli = slice(x0, x1 + 1)
            # masks are uint8 0..255 in shards
            M = np.asarray(self.cache['masks'][k_idx], dtype=np.float32) / 255.0
            return ysli, xsli, M
        # fallback: draw on the fly
        ys_k = self.cache['ys'][k_idx]; xs_k = self.cache['xs'][k_idx]
        x1p, y1p, x2p, y2p = edge_endpoints_from_coords(ys_k, xs_k)
        return draw_line_mask_roi(self.H, self.W, x1p, y1p, x2p, y2p,
                                  thickness=self.P['thickness_px'], aa=True)





    def _seed_structure(self, n_seed: int, q: float):
        if n_seed <= 0:
            return
        L = self.cache['lengths'].astype(np.float32)
        thr = np.quantile(L, q)
        candidates = np.where((~self.used) & (L >= thr))[0]
        if len(candidates) == 0:
            return
        sectors = self.cache['sectors']
        by_sec: Dict[int, List[int]] = {}
        for idx in candidates:
            s = int(sectors[idx])
            by_sec.setdefault(s, []).append(idx)
        picked = 0
        s_keys = sorted(by_sec.keys())
        ptr = {s: 0 for s in s_keys}
        while picked < n_seed:
            progressed = False
            for s in s_keys:
                lst = by_sec[s]
                if ptr[s] < len(lst):
                    k_idx = lst[ptr[s]]; ptr[s] += 1
                    if self.used[k_idx]: continue
                    self._commit_edge(k_idx)
                    picked += 1; progressed = True
                    if picked >= n_seed: break
            if not progressed:
                break
        self.last_line_angle = None
        self.last_line_endpoints = None
    def run(self, num_lines: int, out_dir: Optional[str] = None, snapshots_every: int = 0):
        # --- safety: ensure params dict exists ---
        if not hasattr(self, 'P') or self.P is None:
            self.P = dict(
                nn_weight=0.8, k_opacity=0.08, thickness_px=1,
                shortlist_k=512, base_top_keep=4096,
                need_gamma=1.0, need_weight=1.0, density_weight=0.0,
                cooldown_steps=6, cooldown_penalty=0.35,
                min_angle_deg=12.0, small_angle_penalty=0.25,
                seed_long_chords=24, seed_length_q=0.88,
                early_stop_window=50, early_stop_improve=1e-3,
                snapshots_every=0, render_thickness=1
            )

        # cap seeding so we never exceed num_lines
        n_total = int(num_lines)
        n_seed = int(min(self.P.get('seed_long_chords', 0), n_total))
        if n_seed > 0:
            self._seed_structure(n_seed, self.P.get('seed_length_q', 0.88))

        gains = []
        no_gain = 0
        remaining = n_total - len(self.chosen_pairs)
        for _ in range(remaining):
            pool = self._candidate_pool()
            if len(pool) == 0:
                break
            self._cur_step = len(self.chosen_pairs)
            best_idx, gain = self._score_candidates_dynamic(pool)
            # Early stop based on min_gain_eps / no_gain_patience
            if gain <= float(self.P.get('min_gain_eps', 1e-5)):
                no_gain += 1
                if no_gain >= int(self.P.get('no_gain_patience', 3)):
                    break
            else:
                no_gain = 0


            
            if best_idx < 0:
                break

            self._commit_edge(best_idx)
            gains.append(float(gain))

            if snapshots_every > 0 and out_dir is not None and ((len(self.chosen_pairs)) % snapshots_every == 0):
                save_canvas_triplet(out_dir, self.canvas_lines_xyxy, self.dark, step=len(self.chosen_pairs),
                                    thickness=self.P['render_thickness'])

            W = self.P.get('early_stop_window', 0)
            if W > 0 and len(gains) >= W:
                recent = gains[-W:]
                if np.mean(recent) < self.P.get('early_stop_improve', 1e-3):
                    break

        return self.chosen_pairs, self.canvas_lines_xyxy, self.dark

    # ---------------------------
# image IO
# ---------------------------

def load_image_gray(path: str) -> np.ndarray:
    tried = [path]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None and not os.path.isabs(path):
        for p in [os.path.join("data", path), os.path.join(os.getcwd(), path)]:
            if p not in tried:
                tried.append(p)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                path = p
                break

    if img is None:
        raise FileNotFoundError(
            "❌ Could not load image. Tried: " + " | ".join(tried) + f"\nCWD={os.getcwd()}"
        )
    return img

def _resolve_image_from_recipe(recipe_path: str, recipe_img: str) -> Optional[str]:
    base = os.path.dirname(os.path.abspath(recipe_path))
    cands = []
    if os.path.isabs(recipe_img):
        cands.append(recipe_img)
    else:
        cands += [
            os.path.join(base, recipe_img),
            os.path.join(base, "data", recipe_img),
            os.path.join("data", recipe_img),
            recipe_img,  # as-is (CWD)
        ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def _apply_recipe(args):
    if not args.recipe:
        return args
    with open(args.recipe, 'r', encoding='utf-8') as f:
        R = json.load(f)

    # IMAGE: CLI wins if it exists; otherwise resolve from recipe
    if not (args.image and os.path.exists(args.image)):
        img_from_recipe = R.get('image', None)
        if img_from_recipe:
            resolved = _resolve_image_from_recipe(args.recipe, img_from_recipe)
            if resolved:
                args.image = resolved

    # Size / preprocess
    if 'size' in R: args.size = [int(R['size'][0]), int(R['size'][1])]
    pp = R.get('preprocess', {})
    args.gamma        = float(pp.get('gamma', args.gamma))
    args.clahe_clip   = float(pp.get('clahe_clip', args.clahe_clip))
    args.clahe_grid   = int(pp.get('clahe_grid', args.clahe_grid))
    args.bilateral_d  = int(pp.get('bilateral_d', args.bilateral_d))
    args.sharpen      = float(pp.get('sharpen', args.sharpen))
    args.invert       = bool(pp.get('invert', args.invert))
    circle_mask_ok    = bool(pp.get('circle_mask', False))

    # Layout
    lay = R.get('layout', {})
    args.nail_shape   = lay.get('shape', args.nail_shape)
    args.num_nails    = int(lay.get('num_nails', args.num_nails))
    args.num_sectors  = int(lay.get('num_sectors', args.num_sectors))
    args.min_dist     = int(lay.get('min_dist', args.min_dist))

    # Params
    P = R.get('params', {})
    args.k_opacity         = float(P.get('k_opacity', args.k_opacity))
    args.thickness_px      = int(P.get('thickness_px', args.thickness_px))
    args.shortlist_k       = int(P.get('shortlist_k', args.shortlist_k))
    args.base_top_keep     = int(P.get('base_top_keep', args.base_top_keep))
    args.need_gamma        = float(P.get('need_gamma', args.need_gamma))
    args.need_weight       = float(P.get('need_weight', args.need_weight))
    args.density_weight    = float(P.get('density_weight', args.density_weight))
    args.cooldown_steps    = int(P.get('cooldown_steps', args.cooldown_steps))
    args.cooldown_penalty  = float(P.get('cooldown_penalty', args.cooldown_penalty))
    args.min_angle_deg     = float(P.get('min_angle_deg', args.min_angle_deg))
    args.small_angle_penalty = float(P.get('small_angle_penalty', args.small_angle_penalty))
    args.seed_long_chords  = int(P.get('seed_long_chords', args.seed_long_chords))
    args.seed_length_q     = float(P.get('seed_length_q', args.seed_length_q))
    args.early_stop_window = int(P.get('early_stop_window', args.early_stop_window))
    args.early_stop_improve= float(P.get('early_stop_improve', args.early_stop_improve))
    args.render_thickness  = int(P.get('render_thickness', args.render_thickness))
    args.fast_score        = bool(P.get('fast_score', args.fast_score))
    args.nn_weight         = float(P.get('nn_weight', args.nn_weight if hasattr(args,'nn_weight') else 0.8))
    args.center_relief     = float(P.get('center_relief', getattr(args, 'center_relief', 0.0)))
    args.angle_penalty     = float(P.get('angle_penalty', getattr(args, 'angle_penalty', 0.0)))
    args.orient_weight     = float(P.get('orient_weight', getattr(args, 'orient_weight', 0.0)))
    args.shortlist_mode    = str(P.get('shortlist_mode', args.shortlist_mode))
    args.scorer            = str(P.get('scorer', args.scorer))
    args.hybrid_topk       = int(P.get('hybrid_topk', args.hybrid_topk))
    args.min_gain_eps      = float(P.get('min_gain_eps', args.min_gain_eps))
    args.no_gain_patience  = int(P.get('no_gain_patience', args.no_gain_patience))
    args.rand_subset_frac  = float(P.get('rand_subset_frac', getattr(args,'rand_subset_frac', 0.0)))
    args.rand_subset_min   = int(P.get('rand_subset_min', getattr(args,'rand_subset_min', 4096)))

    if circle_mask_ok and args.nail_shape == 'circle':
        args.circle_mask = True

    args.num_lines = int(R.get('stats', {}).get('lines_drawn', R.get('num_lines', args.num_lines)))
    if 'seed' in R: args.seed = int(R['seed'])

    return args

# ---------------------------
# main
# ---------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=False, default=None)
    parser.add_argument('--recipe', type=str, default=None,
        help='Path to a recipe.json to replay a run; overrides CLI args except out_dir/export flags.')

    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--num_lines', type=int, default=500)
    parser.add_argument('--edge_ranker', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=0)

    parser.add_argument('--no_score_norm', action='store_true', help='Disable per-shortlist z-norm of NN scores')
    parser.add_argument('--no_dyn_len_norm', action='store_true', help='Disable length-normalization for dynamic need')

    parser.add_argument('--center_relief', type=float, default=0.0)
    parser.add_argument('--shortlist_mode', type=str, default='need',
    choices=['need','sector_need','base','random','sector_random'])

    # layout / cache
    parser.add_argument('--nail_shape', type=str, default='circle')
    parser.add_argument('--num_nails', type=int, default=360)
    parser.add_argument('--num_sectors', type=int, default=12)
    parser.add_argument('--min_dist', type=int, default=30)


    parser.add_argument('--orient_weight', type=float, default=0.0,
                        help='Weight for orientation alignment bonus (0=off)')
    # preprocessing
    parser.add_argument('--size', type=int, nargs=2, default=[400, 400], help='H W')
    parser.add_argument('--gamma', type=float, default=1.9)
    parser.add_argument('--clahe_clip', type=float, default=2.0)
    parser.add_argument('--clahe_grid', type=int, default=8)
    parser.add_argument('--bilateral_d', type=int, default=0)
    parser.add_argument('--sharpen', type=float, default=0.0)
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--circle_mask', action='store_true')

    # thread model + scoring
    parser.add_argument('--k_opacity', type=float, default=0.08)
    parser.add_argument('--thickness_px', type=int, default=1)
    parser.add_argument('--need_gamma', type=float, default=1.0)
    parser.add_argument('--need_weight', type=float, default=1.0)
    parser.add_argument('--density_weight', type=float, default=0.0)

    # constraints / speed
    parser.add_argument('--shortlist_k', type=int, default=512)
    parser.add_argument('--base_top_keep', type=int, default=4096)
    parser.add_argument('--cooldown_steps', type=int, default=6)
    parser.add_argument('--cooldown_penalty', type=float, default=0.35)
    parser.add_argument('--min_angle_deg', type=float, default=12.0)
    parser.add_argument('--small_angle_penalty', type=float, default=0.25)

    # seeding + early stop
    parser.add_argument('--seed_long_chords', type=int, default=24)
    parser.add_argument('--seed_length_q', type=float, default=0.88)
    parser.add_argument('--early_stop_window', type=int, default=50)
    parser.add_argument('--early_stop_improve', type=float, default=1e-3)

    # rendering / reproducibility / export
    parser.add_argument('--render_thickness', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--export_svg', action='store_true')
    parser.add_argument('--svg_stroke', type=float, default=1.0)

    parser.add_argument('--fast_score', action='store_true')
    parser.add_argument('--angle_penalty', type=float, default=0.12)
    parser.add_argument('--log_ranker_npz', type=str, default=None,
        help='If set, save (append) shortlist features/gains/groups to an NPZ for training')
    parser.add_argument('--nn_weight', type=float, default=0.8)
    # --- SCORER / HYBRID / EARLY-STOP ---
    parser.add_argument('--scorer', default='hybrid', choices=['fast', 'delta_sse', 'hybrid'],
                        help="fast: your current scorer; delta_sse: exact ΔSSE; hybrid: shortlist->re-rank by ΔSSE")
    parser.add_argument('--hybrid_topk', type=int, default=256,
                        help="In hybrid mode, re-rank this many candidates by ΔSSE (+penalties).")
    parser.add_argument('--min_gain_eps', type=float, default=1e-5,
                        help="If best ΔSSE gain <= eps, count as 'no-gain' for early stop.")
    parser.add_argument('--no_gain_patience', type=int, default=3,
                        help="Stop after this many consecutive no-gain steps.")
    # in argparse, near other constraints/speed flags
    parser.add_argument('--rand_subset_frac', type=float, default=0.0,
                        help='If > 0, randomly subsample this fraction of unused edges each step before shortlist (0..1].')
    parser.add_argument('--rand_subset_min', type=int, default=4096,
                        help='Minimum pool size to keep after random subsampling.')
    parser.add_argument('--render_scale', type=float, default=1.0,
                        help='Scale factor for *rendering only* (images & SVG). 1.0 = no scale.')
    parser.add_argument('--svg_stroke_scaled', action='store_true',
                        help='If set, multiply svg_stroke by render_scale.')

    args = parser.parse_args()
    set_global_seed(args.seed)
    args = _apply_recipe(args)
    print(f"[replay] using image: {args.image}")



    

    # load + preprocess
    raw = load_image_gray(args.image)
    target = preprocess_image(
        raw,
        size=(int(args.size[0]), int(args.size[1])),
        gamma=args.gamma,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        bilateral_d=args.bilateral_d,
        sharpen=args.sharpen,
        invert=args.invert,
        apply_circle=(args.circle_mask and args.nail_shape == 'circle'),
    )

    # load cache
    cache, _ = load_line_cache(
        (target.shape[0], target.shape[1]),
        args.nail_shape, args.num_nails,
        args.min_dist, args.num_sectors
    )
    if cache is None:
        raise FileNotFoundError('❌ No line cache found. Run scripts.build_line_cache first!')

    # ranker
    ranker = None; norm_mean = None; norm_std = None
    if args.edge_ranker:
        ranker, norm_mean, norm_std, meta = load_edge_ranker(args.edge_ranker)

        print(f'🤖 Loaded NN ranker from {args.edge_ranker}')

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
        snapshots_every=args.save_every,
        fast_score=bool(args.fast_score),
        nn_weight=args.nn_weight if hasattr(args, 'nn_weight') else 0.8,
        score_norm=not args.no_score_norm if hasattr(args,'no_score_norm') else True,
        dyn_len_norm=not args.no_dyn_len_norm if hasattr(args,'no_dyn_len_norm') else True,
        center_relief=getattr(args, 'center_relief', 0.0),   
        angle_penalty=getattr(args, 'angle_penalty', 0.0), 
        orient_weight=float(getattr(args, 'orient_weight', 0.0)),
        shortlist_mode=args.shortlist_mode,
        # NEW:
        scorer=args.scorer,
        hybrid_topk=args.hybrid_topk,
        min_gain_eps=args.min_gain_eps,
        no_gain_patience=args.no_gain_patience,
        rand_subset_frac=getattr(args, 'rand_subset_frac', 0.0),
        rand_subset_min=getattr(args, 'rand_subset_min', 4096),

    )

    selector = ProgressiveSelector(
    target_img=target,
    cache=cache,
    edge_ranker=ranker,
    device='cpu',
    params=params,
    norm_mean=norm_mean,
    norm_std=norm_std,
)

    t0 = time.time()
    chosen_pairs, canvas_lines_xyxy, darkness_map = selector.run(
        num_lines=args.num_lines, out_dir=args.out_dir, snapshots_every=args.save_every
    )
    dt = time.time() - t0

    # --- render scaling (compute once, reuse everywhere) ---
    s = max(1.0, float(getattr(args, 'render_scale', 1.0)))
    H, W = target.shape
    H2, W2 = int(round(H * s)), int(round(W * s))
    scaled_lines = scale_lines(canvas_lines_xyxy, s)
    dark_scaled  = cv2.resize(darkness_map, (W2, H2), interpolation=cv2.INTER_AREA)

    # Save previews
    os.makedirs(args.out_dir, exist_ok=True)
    save_canvas_triplet(
        args.out_dir, scaled_lines, dark_scaled,
        step=None, thickness=max(1, int(round(args.render_thickness * s)))
    )

    # SVG (scaled)
    if args.export_svg:
        svg_path = os.path.join(args.out_dir, 'lines.svg')
        stroke = args.svg_stroke * (s if args.svg_stroke_scaled else 1.0)
        export_svg(svg_path, scaled_lines, w=W2, h=H2, stroke_px=stroke)
        print(f'🖨️  SVG: {svg_path}')

    # CSV log
    try:
        csv_path = os.path.join(args.out_dir, 'chosen_lines.csv')
        pairs   = cache.get('pairs')
        sectors = cache.get('sectors')
        lengths = cache.get('lengths')

        pair_to_idx = {}
        if pairs is not None:
            for k, p in enumerate(pairs):
                a, b = int(p[0]), int(p[1])
                pair_to_idx[(a, b)] = k
                pair_to_idx[(b, a)] = k

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['t', 'index', 'i', 'j', 'sector', 'length'])
            for t, (i, j) in enumerate(chosen_pairs, start=1):
                idx = pair_to_idx.get((i, j), -1)
                sec = int(sectors[idx]) if (sectors is not None and 0 <= idx < len(sectors)) else -1
                L   = int(lengths[idx]) if (lengths is not None and 0 <= idx < len(lengths)) else -1
                w.writerow([t, idx, i, j, sec, L])
        print(f'📝 log: {csv_path}')
    except Exception as e:
        print(f'⚠️ could not write chosen_lines.csv: {e}')

    # Recipe export
    try:
        recipe_path = os.path.join(args.out_dir, 'recipe.json')
        recipe_dir  = os.path.dirname(os.path.abspath(recipe_path))
        img_for_recipe = args.image if os.path.isabs(args.image) else os.path.relpath(args.image, start=recipe_dir)

        recipe = dict(
            image=img_for_recipe,          # keep this
            size=[int(args.size[0]), int(args.size[1])],
            preprocess=dict(
                gamma=args.gamma, clahe_clip=args.clahe_clip, clahe_grid=args.clahe_grid,
                bilateral_d=args.bilateral_d, sharpen=args.sharpen, invert=bool(args.invert),
                circle_mask=bool(args.circle_mask)
            ),
            layout=dict(shape=args.nail_shape, num_nails=args.num_nails,
                        num_sectors=args.num_sectors, min_dist=args.min_dist),
            params=params,
            sequence=chosen_pairs,
            seed=args.seed,
            stats=dict(lines_drawn=len(chosen_pairs), seconds=dt)
        )

        with open(os.path.join(args.out_dir, 'recipe.json'), 'w', encoding='utf-8') as f:
            json.dump(recipe, f, indent=2)
        print(f'📦 recipe: {os.path.join(args.out_dir, "recipe.json")}')
    except Exception as e:
        print(f'⚠️ could not write recipe.json: {e}')

    # Append training NPZ (if requested)
    if args.log_ranker_npz:
        X = np.concatenate(selector.log_feats,  axis=0) if selector.log_feats else np.zeros((0, 11), np.float32)
        G = np.concatenate(selector.log_gains,  axis=0) if selector.log_gains else np.zeros((0,),  np.float32)
        R = np.concatenate(selector.log_groups, axis=0) if selector.log_groups else np.zeros((0,),  np.int32)
        save_or_append_npz(args.log_ranker_npz, X, G, R)
        print(f'🧾 appended dataset: {args.log_ranker_npz}  (X={X.shape}, groups={R.max()+1 if R.size else 0})')

    print(f'✅ lines drawn: {len(chosen_pairs)} in {dt:.2f}s')
    print(f'🖼️ saved results to {args.out_dir}')

if __name__ == '__main__':
    main()
