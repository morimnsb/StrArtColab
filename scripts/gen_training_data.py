#!/usr/bin/env python3
import os, json, math, argparse, random
import numpy as np
import cv2
from tqdm import tqdm
from utils.line_cache import load_line_cache  # your existing cache loader
from collections import deque
import math
# ---- AA line ROI (fast + local) ----
def roi_line_mask(h, w, x1,y1,x2,y2, thickness=1, pad=2):
    xmn, xmx = max(0, min(x1,x2)-thickness-pad), min(w-1, max(x1,x2)+thickness+pad)
    ymn, ymx = max(0, min(y1,y2)-thickness-pad), min(h-1, max(y1,y2)+thickness+pad)
    roi = np.zeros((ymx-ymn+1, xmx-xmn+1), np.float32)
    p1, p2 = (x1-xmn, y1-ymn), (x2-xmn, y2-ymn)
    cv2.line(roi, p1, p2, 1.0, thickness=max(1,int(thickness)), lineType=cv2.LINE_AA)
    return slice(ymn, ymx+1), slice(xmn, xmx+1), roi

def endpoints_from_samples(ys, xs):
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])

def preprocess_u8(img_u8, size=(400,400), gamma=1.9, clahe=2.0, grid=8, invert=False):
    img = cv2.resize(img_u8, size, interpolation=cv2.INTER_AREA)
    clahe_obj = cv2.createCLAHE(clipLimit=max(0.01, clahe), tileGridSize=(grid,grid))
    img = clahe_obj.apply(img)
    f = (img.astype(np.float32)/255.0)
    f = np.power(f, 1.0/max(1e-6, gamma))
    if invert: f = 1.0 - f
    return np.clip(f, 0, 1).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--out', default='data/line_ranker_dataset.npz')
    ap.add_argument('--nail_shape', default='circle')
    ap.add_argument('--num_nails', type=int, default=360)
    ap.add_argument('--num_sectors', type=int, default=12)
    ap.add_argument('--min_dist', type=int, default=30)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--shortlist_k', type=int, default=512)
    ap.add_argument('--thickness', type=int, default=1)
    ap.add_argument('--gamma', type=float, default=1.9)
    ap.add_argument('--clahe', type=float, default=2.0)
    ap.add_argument('--grid', type=int, default=8)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # collect images
    fnames = [os.path.join(args.images_dir,f) for f in os.listdir(args.images_dir)
              if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    X, y = [], []  # features, gains
    meta = []
    recent_nails = deque(maxlen=8)   # cooldown horizon used for labels
    last_endpoints = None
    last_sector = 0
    for fp in tqdm(fnames, desc='gen'):
        raw = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if raw is None: continue
        target = preprocess_u8(raw, size=(400,400), gamma=args.gamma, clahe=args.clahe, grid=args.grid, invert=True)
        H,W = target.shape
        desired = 1.0 - target
        dark = np.zeros_like(target, np.float32)

        cache, _ = load_line_cache((H,W), args.nail_shape, args.num_nails, args.min_dist, args.num_sectors)
        if cache is None: continue

        pairs = cache['pairs']; ys_arr = cache['ys']; xs_arr = cache['xs']
        lengths = cache['lengths']; sectors = cache['sectors']

        # ---- curriculum: start at later steps more often (structured residual) ----
        steps = args.max_steps
        used = np.zeros(len(pairs), bool)
        for t in range(steps):
            need = np.clip(desired - dark, 0.0, 1.0)

            # shortlist by simple heuristic (longer + darker)
            base = lengths.astype(np.float32)
            idxs = np.where(~used)[0]
            if len(idxs)==0: break
            topN = min(args.shortlist_k, len(idxs))
            part = np.argpartition(-base[idxs], topN-1)[:topN]
            cand = idxs[part]

            # compute true gains + simple features for the shortlist
            gains = np.zeros(len(cand), np.float32)
            feats = np.zeros((len(cand), 11), np.float32)  # [sum,mean,max,var,dens,L,sector]
            for j, k in enumerate(cand):
                ys, xs = ys_arr[k], xs_arr[k]
                x1,y1,x2,y2 = endpoints_from_samples(ys, xs)
                ysli, xsli, mask = roi_line_mask(H,W,x1,y1,x2,y2, thickness=args.thickness)
                vals = need[ysli, xsli] * mask
                gains[j] = float(vals.sum())

                # base stats
                linevals = need[ysli, xsli]
                s = float(linevals.sum()); m = float(linevals.mean()); mx=float(linevals.max()); v=float(linevals.var()+1e-8)
                dens = float((1.0 - need[ysli, xsli]).mean())
                L = float(lengths[k]); sec = float(sectors[k])

                # dynamic + geometry
                ang_cur = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0
                if last_endpoints is None:
                    d_ang = 0.0
                else:
                    x1p,y1p,x2p,y2p = last_endpoints
                    ang_prev = math.degrees(math.atan2(y2p - y1p, x2p - x1p)) % 360.0
                    d = abs((ang_cur - ang_prev + 180.0) % 360.0 - 180.0)
                    d_ang = float(d)

                i, jn = int(pairs[k][0]), int(pairs[k][1])
                cool_i = 1.0 if i in recent_nails else 0.0
                cool_j = 1.0 if jn in recent_nails else 0.0
                sec_gap = float(abs(sec - last_sector)) if last_endpoints is not None else 0.0

                feats[j] = [s,m,mx,v,dens,L,sec,d_ang,cool_i,cool_j,sec_gap]

            # store all pairs (feats, gains)
            X.append(feats); y.append(gains)
            meta.append(dict(file=os.path.basename(fp), step=t, num=len(cand)))

            # take the best (teacher) to advance residual for next step
            if gains.size==0: break
            k_best = cand[int(np.argmax(gains))]
            recent_nails.extend([int(pairs[k_best][0]), int(pairs[k_best][1])])
            last_endpoints = (x1, y1, x2, y2)
            last_sector = float(sectors[k_best])
            ys, xs = ys_arr[k_best], xs_arr[k_best]
            x1,y1,x2,y2 = endpoints_from_samples(ys, xs)
            ysli, xsli, mask = roi_line_mask(H,W,x1,y1,x2,y2, thickness=args.thickness)
            # soft attenuation with k≈0.08 (thread opacity)
            dark_sub = dark[ysli, xsli]
            dark_sub[:] = 1.0 - (1.0 - dark_sub) * (1.0 - 0.08 * mask)
            used[k_best] = True

    if not X:
        raise SystemExit('No data generated')
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, meta=json.dumps(meta))
    print('Saved', args.out, X.shape, y.shape)

if __name__ == '__main__':
    main()
