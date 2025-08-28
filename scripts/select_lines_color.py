#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, cv2, json, argparse
import numpy as np

# import your class directly from the script file
from scripts.select_lines_progressive import (
    set_global_seed, preprocess_image, ProgressiveSelector,
    load_line_cache, save_canvas_triplet, export_svg, _norm01
)

def to_float01(img_bgr_u8):
    return np.clip(img_bgr_u8.astype(np.float32) / 255.0, 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out_dir', default='outputs_color')
    ap.add_argument('--num_lines', type=int, default=600)
    ap.add_argument('--k_opacity', type=float, default=0.05)
    ap.add_argument('--thickness_px', type=int, default=1)
    ap.add_argument('--size', type=int, nargs=2, default=[400, 400])  # H W
    ap.add_argument('--render_scale', type=float, default=1.0)
    ap.add_argument('--export_svg', action='store_true')
    ap.add_argument('--svg_stroke', type=float, default=0.8)
    ap.add_argument('--svg_stroke_scaled', action='store_true')
    ap.add_argument('--shortlist_mode', default='sector_need',
                    choices=['need','sector_need','base','random','sector_random'])
    ap.add_argument('--shortlist_k', type=int, default=1024)
    ap.add_argument('--base_top_keep', type=int, default=4096)
    ap.add_argument('--cooldown_steps', type=int, default=6)
    ap.add_argument('--min_dist', type=int, default=30)
    ap.add_argument('--nail_shape', default='circle')
    ap.add_argument('--num_nails', type=int, default=360)
    ap.add_argument('--num_sectors', type=int, default=12)
    ap.add_argument('--scorer', default='hybrid', choices=['fast','delta_sse','hybrid'])
    ap.add_argument('--hybrid_topk', type=int, default=256)
    ap.add_argument('--seed', type=int, default=123)

    args = ap.parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load color image (BGR) and split
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    H, W = args.size
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # note cv2 uses (W,H)
    bgr = to_float01(img)
    # convert to RGB float for nicer mapping
    rgb = bgr[..., ::-1].copy()

    # 2) per-channel target darkness (1 - channel)
    targets = {
        'R': 1.0 - rgb[..., 0],
        'G': 1.0 - rgb[..., 1],
        'B': 1.0 - rgb[..., 2],
    }

    # 3) load shared line cache once
    cache, _ = load_line_cache((H, W), args.nail_shape, args.num_nails, args.min_dist, args.num_sectors)
    if cache is None:
        raise FileNotFoundError('Run scripts.build_line_cache first for the requested layout.')

    # Common params
    common = dict(
        k_opacity=args.k_opacity, thickness_px=args.thickness_px,
        shortlist_k=args.shortlist_k, base_top_keep=args.base_top_keep,
        cooldown_steps=args.cooldown_steps, shortlist_mode=args.shortlist_mode,
        scorer=args.scorer, hybrid_topk=args.hybrid_topk,
        # you can add orient/density weights here if you like
    )

    results = {}
    for ch in ['R','G','B']:
        sel = ProgressiveSelector(target_img=(1.0 - targets[ch]), cache=cache, params=common)
        pairs, lines_xyxy, dark = sel.run(num_lines=args.num_lines)
        results[ch] = dict(pairs=pairs, lines=lines_xyxy, dark=dark)

        ch_dir = os.path.join(args.out_dir, f'chan_{ch}')
        os.makedirs(ch_dir, exist_ok=True)
        # save channel triplet
        save_canvas_triplet(ch_dir, lines_xyxy, dark, step=None, thickness=args.thickness_px)

        # also stash recipe
        with open(os.path.join(ch_dir, 'recipe.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(channel=ch, pairs=pairs, size=[H,W]), f, indent=2)

    # 4) Merge: color rendering
    s = max(1.0, float(args.render_scale))
    H2, W2 = int(round(H*s)), int(round(W*s))
    canvas = np.ones((H2, W2, 3), np.float32)  # white
    for ch_idx, ch in enumerate(['R','G','B']):
        color = [0.0, 0.0, 0.0]
        color[ch_idx] = 0.0  # drawing black “ink” subtracts darkness on that channel (digital model)
        for (x1,y1,x2,y2) in results[ch]['lines']:
            p1 = (int(round(x1*s)), int(round(y1*s)))
            p2 = (int(round(x2*s)), int(round(y2*s)))
            # draw a gray line on the channel: stronger k_opacity -> darker line
            cv2.line(canvas, p1, p2, (0,0,0), thickness=max(1,int(round(args.thickness_px*s))), lineType=cv2.LINE_AA)
    out_png = os.path.join(args.out_dir, 'color_lines_preview.png')
    cv2.imwrite(out_png, (canvas*255.0).astype(np.uint8)[:, :, ::-1])  # save BGR

    # 5) Optional color SVG: assign per-channel stroke color
    if args.export_svg:
        def export_svg_color(path, lines_by_ch, w, h, stroke_px=1.0):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            parts = [header, '  <g fill="none" stroke-linecap="round" stroke-linejoin="round">\n']
            color_map = {'R':'#ff0000','G':'#00cc00','B':'#0066ff'}
            for ch in ['R','G','B']:
                parts.append(f'    <g stroke="{color_map[ch]}" stroke-width="{stroke_px}">\n')
                for (x1,y1,x2,y2) in lines_by_ch[ch]:
                    parts.append(f'      <line x1="{int(round(x1*s))}" y1="{int(round(y1*s))}" x2="{int(round(x2*s))}" y2="{int(round(y2*s))}" />\n')
                parts.append('    </g>\n')
            parts.append('  </g>\n</svg>\n')
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(parts)
        stroke = args.svg_stroke * (s if args.svg_stroke_scaled else 1.0)
        export_svg_color(os.path.join(args.out_dir, 'color_lines.svg'),
                         {ch:results[ch]['lines'] for ch in ['R','G','B']},
                         W2, H2, stroke_px=stroke)

    print(f'✅ Color output saved in: {args.out_dir}')

if __name__ == '__main__':
    main()
