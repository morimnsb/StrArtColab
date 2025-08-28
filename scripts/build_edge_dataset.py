#!/usr/bin/env python3
import os, argparse, json, glob
import numpy as np
import pandas as pd

# MUST match what your simulator logs in "features"
FEATURE_KEYS = [
    "length","sector","sum_imp","mean_imp","mean_dark","sat_frac",
    "avail_frac","mean_need","endpoint_score_i","endpoint_score_j",
    "sector_load","deg_i","deg_j","angle",
]

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def collect_from_jsonl(runs_root):
    pattern = os.path.join(runs_root, "runs", "*", "step_*.jsonl")
    files = sorted(glob.glob(pattern))
    X, y = [], []
    kept, skipped = 0, 0

    for jp in files:
        for row in iter_jsonl(jp):
            feats = row.get("features", {})
            lab = row.get("label", None)
            if lab is None:
                skipped += 1
                continue
            try:
                X.append([feats.get(k, 0.0) for k in FEATURE_KEYS])
                y.append(float(lab))
                kept += 1
            except Exception:
                skipped += 1

    return np.asarray(X, np.float32), np.asarray(y, np.float32), kept, skipped, len(files)

def collect_from_csv(runs_root):
    X, y = [], []
    kept, skipped = 0, 0
    for root, _, files in os.walk(runs_root):
        for f in files:
            if not f.endswith(".csv"): 
                continue
            try:
                df = pd.read_csv(os.path.join(root, f))
            except Exception:
                continue
            if "label" not in df.columns or any(k not in df.columns for k in FEATURE_KEYS):
                skipped += len(df)
                continue
            X.append(df[FEATURE_KEYS].values.astype(np.float32))
            y.append(df["label"].values.astype(np.float32))
            kept += len(df)

    if X:
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
    else:
        X = np.zeros((0, len(FEATURE_KEYS)), np.float32)
        y = np.zeros((0,), np.float32)
    return X, y, kept, skipped

def main():
    ap = argparse.ArgumentParser(description="Build NPZ edge ranking dataset from JSONL / CSV logs.")
    ap.add_argument("--log_dir", type=str, default="dataset", help="root folder that contains runs/*/step_*.jsonl")
    ap.add_argument("--out", type=str, default="dataset/edges_train.npz")
    args = ap.parse_args()

    # 1) Prefer JSONL (StepLogger output)
    X, y, kept, skipped, nfiles = collect_from_jsonl(args.log_dir)
    if X.size > 0:
        print(f"📄 JSONL: files={nfiles}  kept={kept}  skipped={skipped}")
    else:
        print("⚠️ No JSONL rows found; falling back to CSV scan…")
        X, y, kept, skipped = collect_from_csv(args.log_dir)
        print(f"📄 CSV: kept={kept}  skipped={skipped}")

    if X.size == 0:
        print("❌ No usable data found. Make sure you ran the simulator with --log_runs_root and it wrote runs/*/step_*.jsonl containing 'features' and 'label'.")
        return

    # For now, geom_feats are placeholders (shape [N,7]); you can extend later.
    geom_feats = np.zeros((X.shape[0], 7), dtype=np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, edge_feats=X, geom_feats=geom_feats, labels=y)
    print(f"✅ Saved dataset: {args.out} | rows={len(y)} | dims(edge)={X.shape[1]} | dims(geom)={geom_feats.shape[1]}")

if __name__ == "__main__":
    main()
