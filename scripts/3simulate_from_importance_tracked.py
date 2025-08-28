import os
import csv
import cv2
import argparse

from utils.preprocess_image import preprocess_image
from importance.build_map import build_importance_map
from importance.infer_map import infer_importance
from utils.line_cache import load_line_cache
from selection.greedy import pick_lines_with_cache

def save_progress_factory(progress_dir, save_every):
    os.makedirs(progress_dir, exist_ok=True)
    def save_progress(step, canvas):
        if save_every and (step % save_every != 0) and step != 1:
            return
        outp = os.path.join(progress_dir, f"progress_{step:03d}.png")
        cv2.imwrite(outp, canvas)
    return save_progress

def main():
    ap = argparse.ArgumentParser(description="Learned-importance string-art simulation with tracking (uses line cache).")
    ap.add_argument("--image", default="data/A.png")
    ap.add_argument("--out_dir", default="outputs_importance_tracked")
    ap.add_argument("--size", type=str, default="400,400", help="HxW, e.g. 400,400")
    ap.add_argument("--nail_shape", choices=["circle","rectangle"], default="circle")
    ap.add_argument("--num_nails", type=int, default=360)
    ap.add_argument("--num_lines", type=int, default=300)
    ap.add_argument("--num_sectors", type=int, default=12)
    ap.add_argument("--min_dist", type=int, default=50)
    ap.add_argument("--max_lines_per_nail", type=int, default=6)
    ap.add_argument("--lambda_overdraw", type=float, default=2.0)
    ap.add_argument("--topk_per_step", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--overlap_px", type=int, default=6)
    ap.add_argument("--overlap_ratio", type=float, default=0.12)
    ap.add_argument("--use_learned", action="store_true",
                    help="Use learned importance map if checkpoint exists")
    ap.add_argument("--ckpt", default="checkpoints/pattern_cnn.pth",
                    help="Checkpoint path for PatchCNN")
    args = ap.parse_args()

    H, W = map(int, args.size.split(","))
    OUT_DIR = args.out_dir
    PROGRESS_DIR = os.path.join(OUT_DIR, "progress_frames")
    OUTPUT_IMG = os.path.join(OUT_DIR, "simulated_result.png")
    OUTPUT_CSV = os.path.join(OUT_DIR, "line_log.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)

    print("🚀 simulate_from_importance_tracked (cached)")
    gray = preprocess_image(args.image, size=(H, W))

    baseline_M, letter_mask = build_importance_map(gray)
    if args.use_learned and os.path.exists(args.ckpt):
        M = infer_importance(gray, ckpt=args.ckpt)
        print(f"🧠 using learned importance map: {args.ckpt}")
    else:
        M = baseline_M
        print("ℹ️ using baseline importance map")

    cv2.imwrite(os.path.join(OUT_DIR, "debug_letter_mask.png"), letter_mask)
    cv2.imwrite(os.path.join(OUT_DIR, "debug_importance.png"), (M * 255).astype("uint8"))

    cache, path = load_line_cache((H, W), args.nail_shape, args.num_nails, args.min_dist, args.num_sectors)
    if cache is None:
        print("❗ No line cache found for these settings.")
        print("   Build it: python -m scripts.build_line_cache")
        return
    print(f"📦 Loaded cache: {path}")

    f = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=[
        "t","i","j","sector","length","overlap_px","penalty",
        "base_score","final_score","used_per_nail_i","used_per_nail_j","used_per_sector_sec"
    ])
    writer.writeheader()

    def on_step(step_idx, info):
        print(f"  • t={info['t']:03d} | {info['i']}→{info['j']} | "
              f"sec={info['sector']} | len={info['length']} | "
              f"overlap={info['overlap_px']} | pen={info['penalty']} | "
              f"base={info['base_score']:.1f} | final={info['final_score']:.1f}")
        writer.writerow(info)

    save_progress = save_progress_factory(PROGRESS_DIR, args.save_every)
    max_per_sector = max(1, args.num_lines // args.num_sectors)

    chosen, canvas, _ = pick_lines_with_cache(
        cache=cache,
        importance_map=M,
        letter_mask=letter_mask,
        num_lines=args.num_lines,
        max_per_sector=max_per_sector,
        max_lines_per_nail=args.max_lines_per_nail,
        lambda_overdraw=args.lambda_overdraw,
        topk_per_step=args.topk_per_step,
        save_progress=save_progress,
        on_step=on_step,
    )

    f.close()
    cv2.imwrite(OUTPUT_IMG, canvas)
    print(f"✅ saved image: {OUTPUT_IMG}")
    print(f"✅ saved log:   {OUTPUT_CSV}")
    print(f"🧵 lines drawn: {len(chosen)}")

if __name__ == "__main__":
    main()
