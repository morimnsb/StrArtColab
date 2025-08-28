# scripts/generate_dataset.py
import os, glob, json, random, argparse
import numpy as np
from tqdm import tqdm

# We'll call your progressive selector as a module function for speed
from scripts.select_lines_progressive import (
    normalize01, maybe_blur01, resolve_foreground_mask,
    score_nails_local, load_nail_scores_csv,
    select_lines_progressive
)
from utils.preprocess_image import preprocess_image
from importance.build_map import build_importance_map
from importance.infer_map import infer_importance
from utils.line_cache import load_line_cache
from scripts.dataset_logging import StepLogger
import json

def relabel_steps_with_chosen(run_dir: str, chosen_indices: list[int]) -> None:
    """
    For each step file step_XXXX.jsonl in run_dir, set label=1 for the row whose
    candidate.idx matches chosen_indices[t-1]; others remain 0.
    """
    for t, idx in enumerate(chosen_indices, start=1):
        jp = os.path.join(run_dir, f"step_{t:04d}.jsonl")
        if not os.path.exists(jp):
            continue
        # read all rows
        with open(jp, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        # relabel
        for r in rows:
            r["label"] = 1 if r.get("candidate", {}).get("idx", None) == int(idx) else 0
        # write back
        with open(jp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

def nails_id_str(nail_shape, num_nails, min_dist, num_sectors, size_hw):
    H, W = size_hw
    return f"{nail_shape}_{num_nails}_{W}x{H}_d{min_dist}_s{num_sectors}"

def sample_config():
    # tweak ranges as you like
    return dict(
        lambda_overdraw=random.choice([2.0, 3.0, 4.0, 5.0]),
        darkness_per_line=random.choice([6.0, 8.0, 10.0]),
        max_hits_per_pixel=random.choice([8, 10, 12]),
        guard_band=random.choice([12.0, 16.0, 20.0, 24.0]),
        need_gamma=random.choice([1.0, 1.3, 1.5, 1.7]),
        topk_candidates=random.choice([2000, 3000, 3500]),
        normalize_by_length=True,
        endpoint_bias_alpha=random.choice([0.2, 0.3, 0.4, 0.5]),
        max_per_sector=random.choice([50, 60, 70]),
        max_lines_per_nail=random.choice([5, 6, 7]),
        num_lines=random.choice([300, 400, 500]),
        mask_only=True,
        m_blur_sigma=random.choice([0.0, 0.8, 1.2]),
    )

def main():
    ap = argparse.ArgumentParser(description="Generate string-art LTR dataset by sweeping configs.")
    ap.add_argument("--images_dir", default="data")
    ap.add_argument("--out_root", default="dataset")
    ap.add_argument("--size", default="400,400")
    ap.add_argument("--nail_shape", choices=["circle","rectangle"], default="circle")
    ap.add_argument("--num_nails", type=int, default=360)
    ap.add_argument("--num_sectors", type=int, default=12)
    ap.add_argument("--min_dist", type=int, default=30)
    ap.add_argument("--use_learned", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/pattern_cnn.pth")
    ap.add_argument("--runs_per_image", type=int, default=3)
    args = ap.parse_args()

    H, W = map(int, args.size.split(","))
    os.makedirs(args.out_root, exist_ok=True)

    # enumerate images
    imgs = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp"):
        imgs += glob.glob(os.path.join(args.images_dir, ext))
    imgs = sorted(imgs)
    if not imgs:
        print("No images found.")
        return

    # load line cache once per size/layout
    cache, cache_path = load_line_cache((H, W), args.nail_shape, args.num_nails, args.min_dist, args.num_sectors)
    if cache is None:
        print("❗ No cache found. Build it first: python -m scripts.build_line_cache")
        return
    nails_id = nails_id_str(args.nail_shape, args.num_nails, args.min_dist, args.num_sectors, (H, W))
    print(f"📦 cache: {cache_path} | nails_id={nails_id}")

    for img_path in tqdm(imgs, desc="Images"):
        img_id = os.path.splitext(os.path.basename(img_path))[0] + f"_{W}x{H}"
        gray = preprocess_image(img_path, size=(H, W)).astype(np.float32)
        gray01 = normalize01(gray)
        gray_u8 = (gray01 * 255.0).astype(np.uint8)

        baseline_M, letter_mask = build_importance_map(gray_u8)
        if args.use_learned and os.path.exists(args.ckpt):
            M_raw = infer_importance(gray_u8, ckpt=args.ckpt)
        else:
            M_raw = baseline_M
        fg_mask = resolve_foreground_mask(letter_mask)
        M = normalize01(M_raw)
        # auto invert
        if fg_mask is not None:
            fg_mean = float(M[fg_mask].mean()) if fg_mask.any() else 0.0
            bg_mean = float(M[~fg_mask].mean()) if (~fg_mask).any() else 0.0
            if fg_mean < bg_mean:
                M = 1.0 - M
        M = maybe_blur01(M, 0.8)
        M = normalize01(M)

        for run_idx in range(args.runs_per_image):
            cfg = sample_config()
            params = dict(
                image=os.path.basename(img_path),
                size=f"{H},{W}",
                nail_shape=args.nail_shape,
                num_nails=args.num_nails,
                num_sectors=args.num_sectors,
                min_dist=args.min_dist,
                **cfg
            )

            # --- set up logger
            logger = StepLogger(args.out_root, img_id=img_id, params=params, nails_id=nails_id)

            # --- wrap the progressive selector to log each step
            # We'll monkey-patch a tiny callback closure capturing local state
            nails   = cache["nails"]
            pairs   = cache["pairs"]
            sectors = cache["sectors"]
            lengths = cache["lengths"]
            ys_arr  = cache["ys"]
            xs_arr  = cache["xs"]

            # state mirrors your selector’s inner state
            cap = float(cfg["darkness_per_line"] * cfg["max_hits_per_pixel"])
            darkness = np.zeros((H, W), dtype=np.float32)
            used_per_sector = {}
            used_per_nail = {}

            # static prescore to get order (same as inside selector)
            # we only need it here to define the candidate shortlist each step
            # real scoring happens inside select_lines_progressive again
            M_eff = M.copy()  # with mask_only already set in cfg
            base_scores = np.zeros(len(lengths), dtype=np.float32)
            for idx in range(len(lengths)):
                ys = ys_arr[idx]; xs = xs_arr[idx]
                s = M_eff[ys, xs].sum()
                if cfg["normalize_by_length"] and lengths[idx] > 0:
                    s /= float(lengths[idx])
                base_scores[idx] = s
            global_order = np.argsort(base_scores)[::-1]

            # we'll hook into select_lines_progressive by copying its call and
            # logging just before it commits; simpler: re-call selection and compute
            # the same features here per step
            from scripts.select_lines_progressive import select_lines_progressive as _selector

            # local wrapper around logger to build candidate JSON rows
            def make_candidate(idx, t, chosen_idx=None):
                i, j = int(pairs[idx][0]), int(pairs[idx][1])
                sec  = int(sectors[idx])
                ys = ys_arr[idx]; xs = xs_arr[idx]
                L  = max(1, int(lengths[idx]))

                imp_vals = M_eff[ys, xs]
                dark_vals = darkness[ys, xs]
                need = np.maximum(0.0, cap - dark_vals)
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
                    "endpoint_score_i": 0.0,   # fill later if you compute them
                    "endpoint_score_j": 0.0,
                    "sector_load": float(used_per_sector.get(sec, 0)),
                    "deg_i": int(used_per_nail.get(i, 0)),
                    "deg_j": int(used_per_nail.get(j, 0)),
                    "angle": angle,
                }
                row = {
                    "img_id": img_id,
                    "run_id": logger.run_id,
                    "t": int(t),
                    "candidate": {"idx": int(idx), "i": i, "j": j, "sector": sec, "length": L},
                    "features": feats,
                    "label": 1 if (chosen_idx is not None and idx == chosen_idx) else 0
                }
                return row

            # run the real selector, but intercept per-step via its save_progress callback:
            # we re-create candidates list here from the same shortlist
            chosen_indices = []

            def _save_progress(step, _canvas_img):
                # build candidate rows from current shortlist BEFORE commit
                topk = cfg["topk_candidates"]
                cand = global_order[:topk]  # same shortlist as selector
                rows = [make_candidate(idx, step, None) for idx in cand]
                # mark the last actually chosen (if any) as label=1
                if chosen_indices:
                    rows = [dict(r, label=(1 if r["candidate"]["idx"] == chosen_indices[-1] else 0)) for r in rows]
                logger.log_step(step, rows, M_eff, darkness)

            # call the selector (this will update darkness internally; we’ll update our mirror afterwards)
            chosen, canvas, darkness_map = _selector(
                cache=cache, M=M, fg_mask=None, canvas_hw=(H, W),
                num_lines=cfg["num_lines"],
                lambda_overdraw=cfg["lambda_overdraw"],
                darkness_per_line=cfg["darkness_per_line"],
                max_hits_per_pixel=cfg["max_hits_per_pixel"],
                guard_band=cfg["guard_band"],
                need_gamma=cfg["need_gamma"],
                topk_candidates=cfg["topk_candidates"],
                normalize_by_length=cfg["normalize_by_length"],
                mask_only=True,
                endpoint_bias_alpha=0.0,
                endpoint_bias=None,
                max_per_sector=cfg["max_per_sector"],
                max_lines_per_nail=cfg["max_lines_per_nail"],
                save_progress=_save_progress,
            )

            # save manifest row (optional: aggregate later)
            # print(json.dumps(logger.manifest_row()))
        
        
            # write a small summary
            steps_done = int(len(chosen))
            with open(os.path.join(logger.run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"steps_done": steps_done}, f, indent=2)

            # 🔁 Fix labels: mark 1 positive per step based on chosen indices
            relabel_steps_with_chosen(logger.run_dir, [int(x) for x in chosen])
            print(f"✓ Relabeled {steps_done} step files with chosen positives in {logger.run_dir}")


    print("✅ dataset generation finished.")

if __name__ == "__main__":
    main()
