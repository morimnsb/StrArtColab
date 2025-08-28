#!/usr/bin/env python3
import os, sys, csv, argparse
import numpy as np
import cv2

def read_img(path):
    if not os.path.exists(path): return None
    return cv2.imread(path, cv2.IMREAD_COLOR)

def safe_resize(img, size):
    if img is None: return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def pad_row(images, target_h):
    pieces = []
    for im in images:
        if im is None:
            im = np.full((target_h, target_h, 3), 255, np.uint8)  # white square filler
        pieces.append(im)
    return np.hstack(pieces)

def annotate_top(img, label):
    if img is None: return None
    h, w = img.shape[:2]
    bar_h = max(32, h // 18)
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (w, bar_h), (255,255,255), -1)
    out = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    x = (w - size[0]) // 2
    y = bar_h - 10
    cv2.putText(out, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    return out

def load_run(run_dir):
    return {
        "lines": os.path.join(run_dir, "simulated_result_lines.png"),
        "dark":  os.path.join(run_dir, "simulated_from_darkness.png"),
        "heat":  os.path.join(run_dir, "accumulated_darkness.png"),
        "csv":   os.path.join(run_dir, "chosen_lines.csv"),
    }

def undirected_pairs_from_csv(csv_path):
    if not os.path.exists(csv_path): return set(), 0
    pairs = set()
    n = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n += 1
            i = int(row["i"]); j = int(row["j"])
            if i > j: i, j = j, i
            pairs.add((i, j))
    return pairs, n

def compare_csv(csvA, csvB):
    A, nA = undirected_pairs_from_csv(csvA)
    B, nB = undirected_pairs_from_csv(csvB)
    if nA == 0 or nB == 0:
        return "CSV not found or empty for overlap."
    overlap = len(A & B)
    pctA = overlap / max(1, len(A))
    pctB = overlap / max(1, len(B))
    return f"Baseline lines: {len(A)} | NN lines: {len(B)} | Overlap: {overlap}  ({pctA:0.1%} of baseline, {pctB:0.1%} of NN)"

def main():
    ap = argparse.ArgumentParser("Compare two runs (Baseline vs NN) and build a collage.")
    ap.add_argument("--baseline_dir", required=True, help="e.g. outputs_progressive")
    ap.add_argument("--nn_dir", required=True, help="e.g. outputs_nn")
    ap.add_argument("--out", default="compare_collage.png")
    ap.add_argument("--width", type=int, default=1200, help="final collage width")
    args = ap.parse_args()

    base = load_run(args.baseline_dir)
    nn   = load_run(args.nn_dir)

    # Load images
    linesA = read_img(base["lines"]);   linesB = read_img(nn["lines"])
    darkA  = read_img(base["dark"]);    darkB  = read_img(nn["dark"])
    heatA  = read_img(base["heat"]);    heatB  = read_img(nn["heat"])

    # Pick a reference size
    ref = None
    for im in [linesB, linesA, darkB, darkA, heatB, heatA]:
        if im is not None:
            ref = (im.shape[1], im.shape[0])  # (w,h)
            break
    if ref is None:
        print("❌ No images found to compare.")
        sys.exit(1)
    w0, h0 = ref

    # Resize & annotate
    linesA = annotate_top(safe_resize(linesA, (w0,h0)), "Baseline – Lines")
    linesB = annotate_top(safe_resize(linesB, (w0,h0)), "NN – Lines")
    darkA  = annotate_top(safe_resize(darkA,  (w0,h0)), "Baseline – Reconstructed")
    darkB  = annotate_top(safe_resize(darkB,  (w0,h0)), "NN – Reconstructed")
    heatA  = annotate_top(safe_resize(heatA,  (w0,h0)), "Baseline – Heatmap")
    heatB  = annotate_top(safe_resize(heatB,  (w0,h0)), "NN – Heatmap")

    row1 = pad_row([linesA, linesB], h0)
    row2 = pad_row([darkA,  darkB ], h0)
    row3 = pad_row([heatA,  heatB ], h0)

    collage = np.vstack([row1, row2, row3])

    # Title & footer
    title = f"String-Art: Baseline vs NN\n{args.baseline_dir}  ⇄  {args.nn_dir}"
    cv2.putText(collage, title, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)

    summary = compare_csv(base["csv"], nn["csv"])
    cv2.putText(collage, summary, (15, collage.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,10), 2, cv2.LINE_AA)

    # Resize final width if requested
    if args.width and collage.shape[1] != args.width:
        scale = args.width / collage.shape[1]
        collage = cv2.resize(collage, (args.width, int(collage.shape[0]*scale)), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, collage)
    print(f"✅ Collage saved: {args.out}")
    print(f"ℹ️ {summary}")

if __name__ == "__main__":
    main()
