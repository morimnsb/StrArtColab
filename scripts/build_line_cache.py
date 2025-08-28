#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, math, json, argparse
import numpy as np
from tqdm import tqdm
from utils.generate_nails import generate_nail_positions
from scoring.line_integral import line_pixels
import cv2

def sector_index(pt, center, num_sectors):
    dx, dy = pt[0] - center[0], pt[1] - center[1]
    ang = (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
    return int((ang / (2*math.pi)) * num_sectors)

def draw_line_mask_roi(h, w, x1, y1, x2, y2, thickness=1, aa=True, pad=2):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, required=True, help="H W")
    ap.add_argument("--nail_shape", type=str, default="circle", choices=["circle","rectangle"])
    ap.add_argument("--num_nails", type=int, default=360)
    ap.add_argument("--num_sectors", type=int, default=12)
    ap.add_argument("--min_dist", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="cache")
    ap.add_argument("--thickness_px", type=int, default=1)
    ap.add_argument("--aa", action="store_true", help="Precompute AA masks (writes sharded files).")
    ap.add_argument("--aa_shard_size", type=int, default=20000, help="Lines per shard file.")
    args = ap.parse_args()

    h, w = int(args.size[0]), int(args.size[1])
    os.makedirs(args.out_dir, exist_ok=True)

    nails = generate_nail_positions((h, w), count=args.num_nails, shape=args.nail_shape)
    center = (w/2.0, h/2.0)

    pairs, sectors, lengths = [], [], []
    ys_list, xs_list = [], []

    print(f"🧷 Precomputing lines for {len(nails)} nails ({args.nail_shape}, {w}x{h})…")
    for i in tqdm(range(len(nails))):
        for j in range(len(nails)):
            if i == j:
                continue
            p1, p2 = nails[i], nails[j]
            if math.hypot(p1[0]-p2[0], p1[1]-p2[1]) < args.min_dist:
                continue
            ys, xs = line_pixels(p1, p2, (h, w))
            pairs.append((i, j))
            sectors.append(sector_index(p1, center, args.num_sectors))
            lengths.append(len(xs))
            ys_list.append(ys.astype(np.int32))
            xs_list.append(xs.astype(np.int32))

    base_name = f"lines_{args.nail_shape}_{args.num_nails}_{w}x{h}_d{args.min_dist}_s{args.num_sectors}"
    core_path = os.path.join(args.out_dir, base_name + ".npz")

    meta = {
        "shape_hw": [h, w],
        "nails": nails,
        "pairs": pairs,
        "sectors": sectors,
        "lengths": lengths,
        "min_dist": args.min_dist,
        "num_sectors": args.num_sectors,
        "nail_shape": args.nail_shape,
        "num_nails": args.num_nails,
    }

    # core cache (safe size)
    np.savez_compressed(
        core_path,
        ys=np.array(ys_list, dtype=object),
        xs=np.array(xs_list, dtype=object),
        pairs=np.array(pairs, dtype=np.int32),
        sectors=np.array(sectors, dtype=np.int32),
        lengths=np.array(lengths, dtype=np.int32),
        nails=np.array(nails, dtype=np.int32),
        meta=json.dumps(meta),
    )
    print(f"✅ Saved core cache: {core_path}")

    # Optional: AA masks as shards
    if args.aa:
        N = len(pairs)
        shard_size = max(1, int(args.aa_shard_size))
        num_shards = (N + shard_size - 1) // shard_size

        # pre-allocate bound arrays
        roi_y0 = np.empty(N, np.int32)
        roi_y1 = np.empty(N, np.int32)
        roi_x0 = np.empty(N, np.int32)
        roi_x1 = np.empty(N, np.int32)

        # We will write masks per shard to keep files small
        for s in range(num_shards):
            a = s * shard_size
            b = min(N, (s+1) * shard_size)
            masks = []
            for k in tqdm(range(a, b), desc=f"Shard {s+1}/{num_shards}", leave=False):
                ys = ys_list[k]; xs = xs_list[k]
                y1, x1 = int(ys[0]), int(xs[0])
                y2, x2 = int(ys[-1]), int(xs[-1])
                ysli, xsli, M = draw_line_mask_roi(h, w, x1, y1, x2, y2,
                                                   thickness=args.thickness_px, aa=True)
                roi_y0[k], roi_y1[k] = ysli.start, ysli.stop - 1
                roi_x0[k], roi_x1[k] = xsli.start, xsli.stop - 1
                # store compact as uint8 (0..255)
                masks.append((M * 255.0 + 0.5).astype(np.uint8))

            shard_path = os.path.join(args.out_dir, f"{base_name}_masks_{s:03d}.npz")
            np.savez_compressed(
                shard_path,
                masks=np.array(masks, dtype=object),
                lo=a, hi=b
            )
            print(f"📦 Saved masks shard {s+1}/{num_shards}: {shard_path}")

        # write bounds sidecar once
        bounds_path = os.path.join(args.out_dir, f"{base_name}_bounds.npz")
        np.savez_compressed(
            bounds_path,
            roi_y0=roi_y0, roi_y1=roi_y1, roi_x0=roi_x0, roi_x1=roi_x1,
            num_shards=num_shards
        )
        print(f"🧭 Saved mask bounds: {bounds_path}")

if __name__ == "__main__":
    main()
