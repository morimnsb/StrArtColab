import numpy as np

def line_pixels(p1, p2, shape_hw):
    h, w = shape_hw
    x1, y1 = p1; x2, y2 = p2
    n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.linspace(x1, x2, n).astype(int).clip(0, w - 1)
    ys = np.linspace(y1, y2, n).astype(int).clip(0, h - 1)
    return ys, xs  # row, col

def score_line_from_map(M, p1, p2):
    ys, xs = line_pixels(p1, p2, M.shape)
    return float(M[ys, xs].sum())
