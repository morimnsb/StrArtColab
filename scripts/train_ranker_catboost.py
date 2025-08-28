#!/usr/bin/env python3
import os, json, glob, argparse, random, csv
import numpy as np
from catboost import CatBoostRanker, Pool


FEATURE_KEYS = [
    "length","sector","sum_imp","mean_imp","mean_dark","sat_frac",
    "avail_frac","mean_need","endpoint_score_i","endpoint_score_j",
    "sector_load","deg_i","deg_j","angle",
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def score_for_hard_neg(feats: dict) -> float:
    # simple hardness heuristic
    return (
        float(feats.get("sum_imp", 0.0)) * 0.6 +
        float(feats.get("mean_need", 0.0)) * 0.4 -
        float(feats.get("mean_dark", 0.0)) * 0.2
    )

def collect_streaming(runs_root, files_glob, neg_topk, neg_rand, limit_groups=None, log_every=200):
    """
    Yields per-group (Xg, yg, fabricated_flag) where each group is a step_*.jsonl file.
    Negatives are subsampled (hard + random). If no positive exists, we fabricate one.
    """
    pattern = os.path.join(runs_root, "runs", "*", files_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ No files matched {pattern}")
        return

    print(f"📂 Found {len(files)} step files.")
    groups_seen = 0

    for fi, jp in enumerate(files, start=1):
        rows = list(load_jsonl(jp))
        if not rows:
            continue

        pos = [r for r in rows if int(r.get("label", 0)) == 1]
        neg = [r for r in rows if int(r.get("label", 0)) == 0]

        fabricated = False
        if not pos:
            # fallback: promote hardest negative to positive
            if neg:
                neg_sorted = sorted(neg, key=lambda r: score_for_hard_neg(r["features"]), reverse=True)
                pos = [neg_sorted[0]]
                neg = neg_sorted[1:]
                fabricated = True
            else:
                continue

        chosen = [pos[0]]
        if neg_topk > 0 and neg:
            neg_sorted = sorted(neg, key=lambda r: score_for_hard_neg(r["features"]), reverse=True)
            chosen.extend(neg_sorted[:min(neg_topk, len(neg_sorted))])

        remain = [r for r in neg if r not in chosen]
        if neg_rand > 0 and remain:
            k = min(neg_rand, len(remain))
            chosen.extend(random.sample(remain, k))

        Xg, yg = [], []
        for r in chosen:
            feats = r["features"]
            Xg.append([feats.get(k, 0.0) for k in FEATURE_KEYS])
            lab = 1.0 if r in pos else 0.0
            yg.append(lab)

        Xg = np.asarray(Xg, np.float32)
        yg = np.asarray(yg, np.float32)

        yield Xg, yg, fabricated

        groups_seen += 1
        if limit_groups is not None and groups_seen >= limit_groups:
            print(f"⏹️  Stopping early at {groups_seen} groups due to --limit_groups.")
            break
        if fi % log_every == 0:
            print(f"  … processed {fi}/{len(files)} files")

def stack_groups(groups):
    """Stack list of (Xg, yg) into X, y and build per-row group_id vector."""
    X_blocks, y_blocks, gid_blocks = [], [], []
    gid = 0
    for Xg, yg in groups:
        X_blocks.append(Xg)
        y_blocks.append(yg)
        gid_blocks.append(np.full(Xg.shape[0], gid, dtype=np.int32))
        gid += 1
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    group_id = np.concatenate(gid_blocks)
    return X, y, group_id

def save_eval_csv(evals_result: dict, out_path: str):
    """Write CatBoost evals_result to CSV with columns: iteration, <metric>_train, <metric>_valid"""
    if not evals_result:
        return
    learn = evals_result.get('learn', {})
    valid = evals_result.get('validation', {})
    metrics = sorted(set(learn.keys()) | set(valid.keys()))
    iters = len(next(iter(learn.values()))) if learn else len(next(iter(valid.values())))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["iteration"]
        for m in metrics:
            header += [f"{m}_train", f"{m}_valid"]
        w.writerow(header)
        for i in range(iters):
            row = [i]
            for m in metrics:
                tr = learn.get(m, [None]*iters)[i] if learn else None
                va = valid.get(m, [None]*iters)[i] if valid else None
                row += [tr, va]
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Train CatBoost ranking model on step JSONL logs (memory-friendly, group-wise split).")
    ap.add_argument("--runs_root", default="dataset")
    ap.add_argument("--files_glob", default="step_*.jsonl")
    ap.add_argument("--model_out", default="checkpoints/ltr_cat.cbm")
    # subsampling
    ap.add_argument("--neg_topk", type=int, default=120)
    ap.add_argument("--neg_rand", type=int, default=80)
    ap.add_argument("--limit_groups", type=int, default=None, help="Cap total groups to load for training")
    # split
    ap.add_argument("--val_groups", type=int, default=200, help="How many groups to hold out for validation (at the end)")
    # catboost params
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l2_leaf_reg", type=float, default=3.0)
    ap.add_argument("--random_strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    # logging
    ap.add_argument("--metrics_csv", default="checkpoints/ltr_cat_metrics.csv")
    ap.add_argument("--featimp_csv", default="checkpoints/ltr_cat_featimp.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.featimp_csv), exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed)

    print("🔍 Streaming dataset with negative subsampling…")
    groups_data = []
    fabricated_groups = 0
    total_rows = 0
    ltr = None
    ltr = getattr(args, "ltr_model", None)
    if ltr and os.path.exists(ltr):

        try:
            ltr = CatBoostRanker()
            ltr.load_model(args.ltr_model)
            print(f"🤖 Loaded LTR model: {args.ltr_model}")
        except Exception as e:
            print(f"⚠️ Failed to load LTR model: {e}")
            ltr = None

    for Xg, yg, fabricated in collect_streaming(
        args.runs_root, args.files_glob, args.neg_topk, args.neg_rand,
        limit_groups=args.limit_groups, log_every=200
    ):
        groups_data.append((Xg, yg))
        total_rows += len(yg)
        if fabricated:
            fabricated_groups += 1
        if len(groups_data) % 200 == 0:
            print(f"  ✅ groups={len(groups_data)}, rows={total_rows}, fabricated={fabricated_groups}")

    if not groups_data:
        print("❌ No data collected (still no positives and no negatives).")
        return

    total_groups = len(groups_data)
    print(f"📦 Collected groups={total_groups}, rows={total_rows}, fabricated positives in {fabricated_groups} groups")

    # ----- group-wise split (last val_groups held out) -----
    val_g = min(args.val_groups, max(0, total_groups // 10)) if args.val_groups > 0 else 0
    if val_g > 0:
        train_groups = groups_data[:-val_g]
        valid_groups = groups_data[-val_g:]
    else:
        train_groups = groups_data
        valid_groups = []

    X_tr, y_tr, gid_tr = stack_groups(train_groups)
    print(f"🧪 Train rows={X_tr.shape[0]}, groups={len(train_groups)}")

    if valid_groups:
        X_va, y_va, gid_va = stack_groups(valid_groups)
        print(f"🧪 Valid rows={X_va.shape[0]}, groups={len(valid_groups)}")
        eval_set = Pool(data=X_va, label=y_va, group_id=gid_va)
    else:
        eval_set = None
        print("ℹ️ No validation set (val_groups=0).")

    # CatBoost pools
    train_pool = Pool(data=X_tr, label=y_tr, group_id=gid_tr)
    def features_for_idx(idx, cache, M_eff, darkness_map, cap, used_per_sector, used_per_nail):
        nails   = cache["nails"]; pairs = cache["pairs"]; sectors = cache["sectors"]; lengths = cache["lengths"]
        ys_arr  = cache["ys"];    xs_arr = cache["xs"]

        i, j = int(pairs[idx][0]), int(pairs[idx][1])
        sec  = int(sectors[idx])
        L    = max(1, int(lengths[idx]))
        ys, xs = ys_arr[idx], xs_arr[idx]

        imp_vals  = M_eff[ys, xs]
        dark_vals = darkness_map[ys, xs]
        need      = np.maximum(0.0, cap - dark_vals)

        # simple state features
        avail_frac = float((need > 0).mean())
        sat_frac   = float(1.0 - np.clip(need / cap, 0.0, 1.0).mean())
        angle = float(np.arctan2(nails[j][1]-nails[i][1], nails[j][0]-nails[i][0]))

        feats = {
            "length": float(L),
            "sector": int(sec),
            "sum_imp": float(imp_vals.sum()),
            "mean_imp": float(imp_vals.mean()),
            "mean_dark": float(dark_vals.mean()),
            "sat_frac": float(sat_frac),
            "avail_frac": float(avail_frac),
            "mean_need": float(need.mean()),
            "endpoint_score_i": 0.0,  # fill if you have endpoint bias precomputed
            "endpoint_score_j": 0.0,
            "sector_load": float(used_per_sector.get(sec, 0)),
            "deg_i": int(used_per_nail.get(i, 0)),
            "deg_j": int(used_per_nail.get(j, 0)),
            "angle": angle,
        }
        return np.array([feats[k] for k in FEATURE_KEYS], dtype=np.float32)

    print("🚀 Training CatBoost (YetiRank)…")
    model = CatBoostRanker(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        loss_function="YetiRank",
        eval_metric="NDCG:top=1",                         # focus on picking the best line
        custom_metric=["NDCG:top=1","NDCG:top=5","NDCG:top=10"],
        od_type="Iter", od_wait=200,                      # early stopping
        random_seed=args.seed,
        verbose=100
    )

    if eval_set is not None:
        model.fit(train_pool, eval_set=eval_set, verbose=100, use_best_model=True)
    else:
        model.fit(train_pool, verbose=100)

    print("🏁 Training completed.")

    # Save model
    model.save_model(args.model_out)
    print(f"💾 Model saved: {args.model_out}")

    # Save metrics per-iteration
    try:
        evals = model.get_evals_result()
        save_eval_csv(evals, args.metrics_csv)
        print(f"📈 Metrics CSV saved: {args.metrics_csv}")
    except Exception as e:
        print(f"⚠️ Could not save metrics CSV: {e}")

    # Save feature importance
    try:
        fi = model.get_feature_importance(type='FeatureImportance')
        with open(args.featimp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature","importance"])
            for k, v in zip(FEATURE_KEYS, fi):
                w.writerow([k, float(v)])
        print(f"🌟 Feature importance saved: {args.featimp_csv}")
    except Exception as e:
        print(f"⚠️ Could not save feature importance: {e}")

if __name__ == "__main__":
    main()
