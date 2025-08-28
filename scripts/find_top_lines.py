#!/usr/bin/env python3
import os
import csv
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from utils.preprocess_image import preprocess_image
from importance.build_map import build_importance_map
from importance.infer_map import infer_importance
from utils.line_cache import load_line_cache

# -------------------------- helpers --------------------------

def normalize01(a):
    a = a.astype(np.float32, copy=False)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)

def maybe_blur01(M01, sigma):
    if sigma is None or sigma <= 0:
        return M01
    k = max(1, int(2 * round(3 * float(sigma)) + 1))
    M8 = np.clip(M01 * 255.0, 0, 255).astype(np.uint8)
    M8 = cv2.GaussianBlur(M8, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return M8.astype(np.float32) / 255.0

def resolve_foreground_mask(letter_mask, prefer_smaller_fg=True):
    """Return boolean FG mask (True=letter), even if input polarity is flipped."""
    if letter_mask is None:
        return None
    mask_pos = (letter_mask.astype(np.uint8) > 0)
    mask_neg = ~mask_pos
    if prefer_smaller_fg:
        return mask_pos if mask_pos.sum() <= mask_neg.sum() else mask_neg
    return mask_pos

def score_nails_from_local(M, nails, radius_px=6, fg_mask=None, mask_only=False, reduce="sum"):
    """Local importance around each nail in a disk radius."""
    H, W = M.shape
    yy, xx = np.mgrid[0:H, 0:W]
    M_eff = M if not (mask_only and fg_mask is not None) else (M * fg_mask.astype(np.float32))
    r2 = float(radius_px * radius_px)
    scores = np.zeros(len(nails), dtype=np.float32)
    for i, (x, y) in enumerate(nails):
        dx = xx - int(x)
        dy = yy - int(y)
        disk = (dx*dx + dy*dy) <= r2
        vals = M_eff[disk]
        s = 0.0 if vals.size == 0 else (vals.sum() if reduce == "sum" else vals.mean())
        scores[i] = s
    return scores

def load_nail_scores_from_csv(path, num_nails):
    """Read a CSV like top_nails.csv and return scores array aligned to nail index."""
    scores = np.zeros(num_nails, dtype=np.float32)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # Expect columns: rank,nail_index,x,y,score
        for row in r:
            idx = int(row.get("nail_index", -1))
            sc  = float(row.get("score", "0"))
            if 0 <= idx < num_nails:
                scores[idx] = sc
    return scores

# ------------------------ line scoring -----------------------

def score_lines(cache, M, fg_mask=None, normalize_by_length=False, mask_only=False,
                endpoint_bias=None, endpoint_bias_alpha=0.0):
    """
    Base line score = sum of M over line pixels (optionally masked & length-normalized).
    Optional endpoint bias: multiply by (1 + alpha * mean(endpoint_nail_scores_norm)).
    """
    ys_arr = cache["ys"]        # object array, each entry: np.ndarray (Li,)
    xs_arr = cache["xs"]        # object array, each entry: np.ndarray (Li,)
    lengths = cache["lengths"]  # int32 (NUM_LINES,)
    pairs = cache["pairs"]      # (NUM_LINES, 2) int32

    scores = np.zeros(len(lengths), dtype=np.float32)
    M32 = M.astype(np.float32, copy=False)

    if mask_only and fg_mask is not None:
        W = fg_mask.astype(np.float32)
    else:
        W = None

    # normalize endpoint bias to [0,1]
    if endpoint_bias is not None and endpoint_bias.size > 0:
        b = endpoint_bias.astype(np.float32)
        b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    else:
        b = None

    for idx in tqdm(range(len(lengths)), desc="Scoring lines"):
        ys = ys_arr[idx]
        xs = xs_arr[idx]
        s = (M32[ys, xs].sum() if W is None else (M32[ys, xs] * W[ys, xs]).sum())
        if normalize_by_length and lengths[idx] > 0:
            s /= float(lengths[idx])

        # endpoint bias
        if b is not None and endpoint_bias_alpha != 0.0:
            i, j = int(pairs[idx][0]), int(pairs[idx][1])
            mean_ep = 0.5 * (b[i] + b[j])   # in [0,1]
            s *= (1.0 + endpoint_bias_alpha * mean_ep)

        scores[idx] = s

    return scores

def post_select_with_caps(sorted_idxs, pairs, sectors, topk, max_per_sector=None, max_lines_per_nail=None):
    """
    Take indices sorted by score (desc) and enforce per-sector / per-nail caps.
    Returns a filtered list with length <= topk.
    """
    chosen = []
    used_sector = {}
    used_nail = {}
    for idx in sorted_idxs:
        if len(chosen) >= topk:
            break
        i, j = int(pairs[idx][0]), int(pairs[idx][1])
        sec = int(sectors[idx])

        if (max_per_sector is not None) and (used_sector.get(sec, 0) >= max_per_sector):
            continue
        if (max_lines_per_nail is not None) and (
            used_nail.get(i, 0) >= max_lines_per_nail or used_nail.get(j, 0) >= max_lines_per_nail
        ):
            continue

        chosen.append(idx)
        used_sector[sec] = used_sector.get(sec, 0) + 1
        used_nail[i] = used_nail.get(i, 0) + 1
        used_nail[j] = used_nail.get(j, 0) + 1

    return np.array(chosen, dtype=np.int32)

def draw_lines(canvas_h, canvas_w, nails, pairs, top_idxs, color=(0,0,0), thickness=1):
    """Render selected lines on a white canvas."""
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    for idx in top_idxs:
        i, j = pairs[idx]
        cv2.line(canvas,
                 (int(nails[i][0]), int(nails[i][1])),
                 (int(nails[j][0]), int(nails[j][1])),
                 color, thickness, cv2.LINE_AA)
    return canvas

# ----------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Find top-K important lines from cache using an importance map (+ optional nail bias & caps).")

    # I/O + canvas/cache
    ap.add_argument("--image", default="data/A.png")
    ap.add_argument("--out_dir", default="outputs_toplines")
    ap.add_argument("--size", type=str, default="400,400", help="HxW, e.g. 400,400")
    ap.add_argument("--nail_shape", choices=["circle","rectangle"], default="circle")
    ap.add_argument("--num_nails", type=int, default=360)
    ap.add_argument("--num_sectors", type=int, default=12)
    ap.add_argument("--min_dist", type=int, default=50)

    # Importance source + postproc
    ap.add_argument("--use_learned", action="store_true", help="Use learned importance map if checkpoint exists")
    ap.add_argument("--ckpt", default="checkpoints/pattern_cnn.pth", help="Checkpoint path for learned importance")
    ap.add_argument("--invert_importance", action="store_true", help="Force invert importance after normalization")
    ap.add_argument("--no_auto_invert", action="store_true", help="Disable auto-invert (FG darker than BG)")
    ap.add_argument("--m_blur_sigma", type=float, default=0.0, help="Gaussian blur sigma for importance map")

    # Scoring options
    ap.add_argument("--topk", type=int, default=2000, help="How many top lines to export (before caps)")
    ap.add_argument("--normalize_by_length", action="store_true", help="Divide line score by pixel length")
    ap.add_argument("--mask_only", action="store_true", help="Only count importance inside the foreground mask")

    # Endpoint nail bias (optional)
    ap.add_argument("--endpoint_bias_alpha", type=float, default=0.0, help="Multiply score by (1 + alpha * mean(endpoint_bias))")
    ap.add_argument("--use_nail_csv", default="", help="Path to top_nails.csv to use as endpoint bias (optional)")
    ap.add_argument("--nail_radius_px", type=int, default=6, help="If no CSV, compute local nail scores with this radius")
    ap.add_argument("--nail_reduce", choices=["sum","mean"], default="sum")

    # Caps (post-filter)
    ap.add_argument("--max_per_sector", type=int, default=0, help="0=off; else cap lines per sector")
    ap.add_argument("--max_lines_per_nail", type=int, default=0, help="0=off; else cap lines per nail")

    # Preview
    ap.add_argument("--save_preview", action="store_true", help="Save a preview image with selected lines")
    ap.add_argument("--preview_thickness", type=int, default=1)

    args = ap.parse_args()

    H, W = map(int, args.size.split(","))
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "top_lines.csv")
    preview_path = os.path.join(args.out_dir, "top_lines_preview.png")
    dbg_mask_path = os.path.join(args.out_dir, "debug_letter_mask.png")
    dbg_imp_raw   = os.path.join(args.out_dir, "debug_importance_raw.png")
    dbg_imp_path  = os.path.join(args.out_dir, "debug_importance.png")

    # 1) Image → gray (both uint8 and float)
    gray = preprocess_image(args.image, size=(H, W)).astype(np.float32)
    gray01 = normalize01(gray)
    gray_u8 = (gray01 * 255.0).astype(np.uint8)

    # 2) Importance RAW + mask
    baseline_M, letter_mask = build_importance_map(gray_u8)
    if args.use_learned and os.path.exists(args.ckpt):
        M_raw = infer_importance(gray_u8, ckpt=args.ckpt)
        print(f"🧠 using learned importance map: {args.ckpt}")
    else:
        M_raw = baseline_M
        print("ℹ️ using baseline importance map")

    fg_mask = resolve_foreground_mask(letter_mask)
    cv2.imwrite(dbg_mask_path, (fg_mask.astype(np.uint8) * 255) if fg_mask is not None else letter_mask)
    cv2.imwrite(dbg_imp_raw, (normalize01(M_raw) * 255).astype("uint8"))

    # 3) Importance postproc: normalize → auto/manual invert → blur → normalize
    M = normalize01(M_raw)
    if fg_mask is not None and not args.no_auto_invert:
        fg_mean = float(M[fg_mask].mean()) if fg_mask.any() else 0.0
        bg_mean = float(M[~fg_mask].mean()) if (~fg_mask).any() else 0.0
        inv = fg_mean < bg_mean
        if inv:
            M = 1.0 - M
        print(f"[auto-invert check] fg_mean={fg_mean:.4f}, bg_mean={bg_mean:.4f}, inverted={inv}")
    if args.invert_importance:
        M = 1.0 - M
        print("[manual invert] applied")
    M = maybe_blur01(M, args.m_blur_sigma)
    M = normalize01(M)
    cv2.imwrite(dbg_imp_path, (M * 255).astype("uint8"))

    # 4) Load line cache
    cache, cache_path = load_line_cache((H, W), args.nail_shape, args.num_nails, args.min_dist, args.num_sectors)
    if cache is None:
        print("❗ No line cache found for these settings.")
        print("   Build it: python -m scripts.build_line_cache")
        return
    print(f"📦 Loaded cache: {cache_path}")

    nails   = cache["nails"]
    pairs   = cache["pairs"]
    sectors = cache["sectors"]
    lengths = cache["lengths"]

    # 5) Endpoint bias (optional)
    endpoint_bias = None
    if args.endpoint_bias_alpha != 0.0:
        if args.use_nail_csv:
            endpoint_bias = load_nail_scores_from_csv(args.use_nail_csv, len(nails))
            if endpoint_bias is None:
                print(f"⚠️ Nail CSV not found or invalid: {args.use_nail_csv} — falling back to local scoring")
        if endpoint_bias is None:
            endpoint_bias = score_nails_from_local(
                M, nails,
                radius_px=args.nail_radius_px,
                fg_mask=fg_mask,
                mask_only=args.mask_only,
                reduce=args.nail_reduce
            )

    # 6) Score all lines
    scores = score_lines(
        cache=cache,
        M=M,
        fg_mask=fg_mask,
        normalize_by_length=args.normalize_by_length,
        mask_only=args.mask_only,
        endpoint_bias=endpoint_bias,
        endpoint_bias_alpha=args.endpoint_bias_alpha
    )

    # 7) Rank + (optional) caps
    K = min(args.topk, len(scores))
    top_idxs = np.argpartition(scores, -K)[-K:]
    top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]  # sorted desc

    max_per_sector     = args.max_per_sector if args.max_per_sector > 0 else None
    max_lines_per_nail = args.max_lines_per_nail if args.max_lines_per_nail > 0 else None
    if (max_per_sector is not None) or (max_lines_per_nail is not None):
        filtered = post_select_with_caps(top_idxs, pairs, sectors, K,
                                         max_per_sector=max_per_sector,
                                         max_lines_per_nail=max_lines_per_nail)
        top_idxs = filtered

    # 8) Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank","index","i","j","sector","length","score"])
        for rank, idx in enumerate(top_idxs, start=1):
            i, j = int(pairs[idx][0]), int(pairs[idx][1])
            sec   = int(sectors[idx])
            L     = int(lengths[idx])
            sc    = float(scores[idx])
            writer.writerow([rank, int(idx), i, j, sec, L, f"{sc:.6f}"])
    print(f"✅ saved: {csv_path}")

    # 9) Optional preview
    if args.save_preview:
        canvas = draw_lines(H, W, nails, pairs, top_idxs,
                            color=(0,0,0), thickness=args.preview_thickness)
        cv2.imwrite(preview_path, canvas)
        print(f"🖼️ preview saved: {preview_path}")

if __name__ == "__main__":
    main()
