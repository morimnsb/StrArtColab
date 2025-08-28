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
    """
    Return a boolean mask FG where True = foreground (the letter).
    Works even if the input has 255 on background (or vice versa).
    If prefer_smaller_fg, choose the polarity with fewer True pixels (typical for glyphs).
    """
    if letter_mask is None:
        return None
    mask_pos = (letter_mask.astype(np.uint8) > 0)
    mask_neg = ~mask_pos
    if prefer_smaller_fg:
        return mask_pos if mask_pos.sum() <= mask_neg.sum() else mask_neg
    # fallback: pick the one with higher mean intensity drop from the image (not needed here)
    return mask_pos

def overlay_points_on_gray(gray01, points_yx, radius=2):
    """Return BGR image: gray background with red dots."""
    g8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    for (y, x) in points_yx:
        cv2.circle(img, (int(x), int(y)), radius, (0,0,255), thickness=-1, lineType=cv2.LINE_AA)
    return img

# ----------------------- point ranking -----------------------

def topk_pixels(M, K, fg_mask=None, mask_only=False):
    """Return top-K pixel coords by importance. Returns (N,3) [y,x,score]."""
    M_eff = M
    if mask_only and fg_mask is not None:
        M_eff = M * fg_mask.astype(np.float32)
    flat = M_eff.reshape(-1)
    K = int(min(K, flat.size))
    if K <= 0:
        return np.empty((0, 3), dtype=np.int32)
    idxs = np.argpartition(flat, -K)[-K:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]
    ys, xs = np.divmod(idxs, M.shape[1])
    scores = flat[idxs]
    return np.stack([ys, xs, scores], axis=1)

def score_nails_from_local(M, nails, radius_px=6, fg_mask=None, mask_only=False, reduce="sum"):
    H, W = M.shape
    yy, xx = np.mgrid[0:H, 0:W]
    M_eff = M
    if mask_only and fg_mask is not None:
        M_eff = M * fg_mask.astype(np.float32)
    scores = np.zeros(len(nails), dtype=np.float32)
    r2 = float(radius_px * radius_px)
    for i, (x, y) in enumerate(nails):
        dx = xx - int(x)
        dy = yy - int(y)
        disk = (dx*dx + dy*dy) <= r2
        vals = M_eff[disk]
        s = 0.0 if vals.size == 0 else (vals.sum() if reduce == "sum" else vals.mean())
        scores[i] = s
    return scores

def draw_point_markers_rgb(img_h, img_w, points_yx, radius=2, color=(0,0,255)):
    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    for (y, x) in points_yx:
        cv2.circle(canvas, (int(x), int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas

# --------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Find top important POINTS (pixels and nails).")
    # I/O + canvas
    ap.add_argument("--image", default="data/A.png")
    ap.add_argument("--out_dir", default="outputs_toppoints")
    ap.add_argument("--size", type=str, default="400,400", help="HxW, e.g. 400,400")
    ap.add_argument("--nail_shape", choices=["circle","rectangle"], default="circle")
    ap.add_argument("--num_nails", type=int, default=360)
    ap.add_argument("--num_sectors", type=int, default=12)
    ap.add_argument("--min_dist", type=int, default=50)

    # Importance source + postproc
    ap.add_argument("--use_learned", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/pattern_cnn.pth")
    ap.add_argument("--invert_importance", action="store_true",
                    help="Force invert importance after normalization")
    ap.add_argument("--no_auto_invert", action="store_true",
                    help="Disable auto-invert based on FG/BG means")
    ap.add_argument("--m_blur_sigma", type=float, default=0.0)

    # Pixel points
    ap.add_argument("--topk_pixels", type=int, default=2000)
    ap.add_argument("--mask_only", action="store_true")
    ap.add_argument("--pixel_preview", action="store_true")
    ap.add_argument("--pixel_overlay", action="store_true",
                    help="Overlay points on the grayscale image (not on white)")
    ap.add_argument("--pixel_dot_radius", type=int, default=2)

    # Nail points
    ap.add_argument("--nail_radius_px", type=int, default=6)
    ap.add_argument("--nail_reduce", choices=["sum","mean"], default="sum")
    ap.add_argument("--nail_topk", type=int, default=200)
    ap.add_argument("--nail_preview", action="store_true")

    args = ap.parse_args()

    H, W = map(int, args.size.split(","))
    os.makedirs(args.out_dir, exist_ok=True)

    # Debug outputs
    dbg_mask_path     = os.path.join(args.out_dir, "debug_letter_mask.png")
    dbg_imp_raw_path  = os.path.join(args.out_dir, "debug_importance_raw.png")
    dbg_imp_path      = os.path.join(args.out_dir, "debug_importance.png")

    # 1) image → gray
    gray = preprocess_image(args.image, size=(H, W)).astype(np.float32)

    # keep both normalized float and uint8 versions
    gray01 = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)  # [0,1]
    gray_u8 = (gray01 * 255.0).astype(np.uint8)                      # uint8

    # 2) Importance (baseline or learned) — RAW
    # build_importance_map needs uint8 because it uses Otsu threshold inside OpenCV
    baseline_M, letter_mask = build_importance_map(gray_u8)

    # If your infer_importance expects 0–1 or 0–255, pick accordingly:
    # - If it was written for uint8: pass gray_u8
    # - If it expects float [0,1]: pass gray01
    # I’ll assume uint8 here; switch to gray01 if your model expects floats.
    if args.use_learned and os.path.exists(args.ckpt):
        M_raw = infer_importance(gray_u8, ckpt=args.ckpt)
        print(f"🧠 using learned importance map: {args.ckpt}")
    else:
        M_raw = baseline_M
        print("ℹ️ using baseline importance map")


    # Resolve a reliable FG mask (True=letter)
    fg_mask = resolve_foreground_mask(letter_mask)

    # save raw artifacts
    cv2.imwrite(dbg_mask_path, (fg_mask.astype(np.uint8) * 255) if fg_mask is not None else letter_mask)
    cv2.imwrite(dbg_imp_raw_path, (normalize01(M_raw) * 255).astype("uint8"))

    # 3) post-proc: normalize → auto-invert (using FG) → manual invert → blur → normalize
    M = normalize01(M_raw)

    if fg_mask is not None and not args.no_auto_invert:
        fg_mean = float(M[fg_mask].mean()) if fg_mask.any() else 0.0
        bg_mean = float(M[~fg_mask].mean()) if (~fg_mask).any() else 0.0
        # if FG is darker, flip so brighter=more important
        if fg_mean < bg_mean:
            M = 1.0 - M
        print(f"[auto-invert check] fg_mean={fg_mean:.4f}, bg_mean={bg_mean:.4f}, inverted={fg_mean < bg_mean}")

    if args.invert_importance:
        M = 1.0 - M
        print("[manual invert] applied")

    M = maybe_blur01(M, args.m_blur_sigma)
    M = normalize01(M)
    cv2.imwrite(dbg_imp_path, (M * 255).astype("uint8"))

    # 4) cache (optional, for nails)
    cache, cache_path = load_line_cache((H, W), args.nail_shape, args.num_nails, args.min_dist, args.num_sectors)
    nails = None
    if cache is None:
        print("ℹ️ No line cache found — pixel points OK; nail points need nails from cache.")
    else:
        print(f"📦 Loaded cache: {cache_path}")
        nails = cache["nails"]

    # ---------- PIXEL POINTS ----------
    if args.topk_pixels > 0:
        top_pix = topk_pixels(M, args.topk_pixels, fg_mask=fg_mask, mask_only=args.mask_only)
        csv_path = os.path.join(args.out_dir, "top_pixels.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["rank","y","x","score"])
            for rank, (y, x, s) in enumerate(top_pix, start=1):
                wcsv.writerow([rank, int(y), int(x), f"{float(s):.6f}"])
        print(f"✅ saved: {csv_path}")

        pts = [(int(y), int(x)) for (y, x, _) in top_pix]
        if args.pixel_preview:
            # white background preview
            white = draw_point_markers_rgb(H, W, pts, radius=args.pixel_dot_radius, color=(0,0,255))
            cv2.imwrite(os.path.join(args.out_dir, "top_pixels_preview.png"), white)
        if args.pixel_overlay:
            # overlay on grayscale (helps verify alignment)
            over = overlay_points_on_gray(gray01, pts, radius=args.pixel_dot_radius)
            cv2.imwrite(os.path.join(args.out_dir, "top_pixels_overlay.png"), over)

    # ---------- NAIL POINTS ----------
    if nails is not None:
        nail_scores = score_nails_from_local(
            M, nails,
            radius_px=args.nail_radius_px,
            fg_mask=fg_mask,
            mask_only=args.mask_only,
            reduce=args.nail_reduce
        )
        order = np.argsort(nail_scores)[::-1]
        csv_path = os.path.join(args.out_dir, "top_nails.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["rank","nail_index","x","y","score"])
            for rank, idx in enumerate(order, start=1):
                x, y = nails[idx]
                wcsv.writerow([rank, int(idx), int(x), int(y), f"{float(nail_scores[idx]):.6f}"])
        print(f"✅ saved: {csv_path}")

        if args.nail_preview:
            K = min(args.nail_topk, len(nails))
            top_idxs = order[:K]
            pts = [(int(nails[i][1]), int(nails[i][0])) for i in top_idxs]
            img = draw_point_markers_rgb(H, W, pts, radius=max(2, args.nail_radius_px//2), color=(0,0,255))
            cv2.imwrite(os.path.join(args.out_dir, "top_nails_preview.png"), img)

if __name__ == "__main__":
    main()
