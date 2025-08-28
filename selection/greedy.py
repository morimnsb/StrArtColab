import numpy as np
from scoring.line_integral import line_pixels
# selection/greedy.py
import numpy as np

def pick_lines_with_cache(
    cache,
    importance_map,             # float32 [0,1]
    letter_mask,                # uint8 0/255
    num_lines=300,
    max_per_sector=None,
    max_lines_per_nail=4,
    lambda_overdraw=2.0,
    topk_per_step=2000,
    min_overlap_px=6,
    min_overlap_ratio=0.12,
    save_progress=None,         # callable(step_idx, canvas) | None
    on_step=None,               # callable(step_idx, info_dict) | None
):
    h, w = importance_map.shape
    nails   = cache["nails"]
    pairs   = cache["pairs"]
    sectors = cache["sectors"]
    ys_arr  = cache["ys"]
    xs_arr  = cache["xs"]
    num_sectors = int(cache["meta"]["num_sectors"])

    if max_per_sector is None:
        max_per_sector = max(1, num_lines // num_sectors)

    # precompute base scores once
    base_scores = np.zeros(len(pairs), dtype=np.float32)
    for k in range(len(pairs)):
        ys = ys_arr[k]; xs = xs_arr[k]
        base_scores[k] = importance_map[ys, xs].sum()

    order = np.argsort(base_scores)[::-1]

    used_per_nail   = np.zeros(len(nails), np.int32)
    used_per_sector = np.zeros(num_sectors, np.int32)
    dark   = np.zeros((h, w), np.uint8)
    canvas = np.full((h, w), 255, np.uint8)
    chosen = []
    log    = []

    def can_use(i, j, sec):
        return (used_per_nail[i] < max_lines_per_nail and
                used_per_nail[j] < max_lines_per_nail and
                used_per_sector[sec] < max_per_sector)

    for t in range(1, num_lines + 1):
        best_idx = -1
        best_val = -1e18
        best     = None

        for k in order[:topk_per_step]:
            i, j = pairs[k]
            sec  = sectors[k]
            if not can_use(i, j, sec):
                continue

            ys = ys_arr[k]; xs = xs_arr[k]
            L = len(xs)
            if L == 0:
                continue

            overlap = (letter_mask[ys, xs] > 0).sum()
            if overlap < min_overlap_px or (overlap / max(1, L)) < min_overlap_ratio:
                continue

            penalty = ((dark[ys, xs] > 0) & (letter_mask[ys, xs] > 0)).sum()
            val = float(base_scores[k]) - lambda_overdraw * penalty

            if val > best_val:
                best_val = val
                best_idx = k
                best = (i, j, sec, ys, xs, L, overlap, penalty, float(base_scores[k]), float(val))

        if best_idx < 0:
            # no feasible candidates left
            break

        i, j, sec, ys, xs, L, overlap, penalty, base, final = best

        # draw and update state
        canvas[ys, xs] = 0
        dark[ys, xs]   = 1
        used_per_nail[i]  += 1
        used_per_nail[j]  += 1
        used_per_sector[sec] += 1
        chosen.append((i, j))

        # build info dict for this step
        info = {
            "t": t,
            "i": int(i), "j": int(j),
            "sector": int(sec),
            "length": int(L),
            "overlap_px": int(overlap),
            "penalty": int(penalty),
            "base_score": base,
            "final_score": final,
            "used_per_nail_i": int(used_per_nail[i]),
            "used_per_nail_j": int(used_per_nail[j]),
            "used_per_sector_sec": int(used_per_sector[sec]),
        }
        log.append(info)

        if on_step:
            on_step(t, info)
        if save_progress and (t % 1 == 0 or t == num_lines):  # save every line if you want
            save_progress(t, canvas)

    return chosen, canvas, log
