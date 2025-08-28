import numpy as np
import torch
import cv2
from ml.pattern_model import PatchCNN

def infer_importance(gray, ckpt="checkpoints/pattern_cnn.pth", patch=32, stride=4, batch_size=256):
    """
    Slide a small CNN over the image to predict a dense importance map in [0,1].
    Falls back gracefully if CUDA is unavailable.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    H, W = gray.shape
    ph = patch // 2
    out = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)

    with torch.no_grad():
        batch, locs = [], []
        for y in range(ph, H - ph, stride):
            for x in range(ph, W - ph, stride):
                patch_img = gray[y - ph:y + ph, x - ph:x + ph].astype(np.float32) / 255.0
                batch.append(patch_img[None, None, ...])
                locs.append((y, x))

                if len(batch) == batch_size:
                    b = torch.from_numpy(np.concatenate(batch, 0)).to(device)
                    preds = model(b).cpu().numpy()
                    for (yy, xx), v in zip(locs, preds):
                        out[yy - 2:yy + 3, xx - 2:xx + 3] += v
                        cnt[yy - 2:yy + 3, xx - 2:xx + 3] += 1
                    batch, locs = [], []

        if batch:
            b = torch.from_numpy(np.concatenate(batch, 0)).to(device)
            preds = model(b).cpu().numpy()
            for (yy, xx), v in zip(locs, preds):
                out[yy - 2:yy + 3, xx - 2:xx + 3] += v
                cnt[yy - 2:yy + 3, xx - 2:xx + 3] += 1

    out = out / np.maximum(cnt, 1e-6)
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out.astype(np.float32)
