import cv2, numpy as np

def build_importance_map(gray, wT=.4, wE=.25, wC=.15, wK=.1, wI=.1):
    g = gray.astype(np.float32) / 255.0

    # tone
    T = 1.0 - g

    # sobel edges
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    S  = np.sqrt(sx*sx + sy*sy); S /= (S.max() + 1e-6)

    # harris
    H = cv2.cornerHarris((g*255).astype(np.uint8), 2, 3, 0.04)
    H = cv2.dilate(H, None); H = np.maximum(0, H); H /= (H.max()+1e-6)

    # skeleton of Otsu foreground
    _, raw = cv2.threshold((g*255).astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ensure foreground (letter) is white
    letter_mask = raw if cv2.countNonZero(raw) < (g.size/2) else cv2.bitwise_not(raw)
    try:
        K = cv2.ximgproc.thinning(letter_mask)
    except Exception:
        K = letter_mask
    K = (K>0).astype(np.float32)

    # LoG-ish compactness
    G1 = cv2.GaussianBlur(T, (0,0), 1.0)
    G2 = cv2.GaussianBlur(T, (0,0), 2.0)
    I  = np.abs(G1 - G2); I /= (I.max()+1e-6)

    M = wT*T + wE*S + wC*H + wK*K + wI*I
    M = M / (M.max() + 1e-6)
    return (M.astype(np.float32), letter_mask)
