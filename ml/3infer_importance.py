# ml/infer_importance.py
import torch, numpy as np, cv2
from ml.pattern_model import PatchCNN

def infer_importance_full(gray, ckpt="checkpoints/pattern_cnn.pth", patch=32, stride=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    H, W = gray.shape
    ph = patch//2
    out = np.zeros((H, W), np.float32)
    counts = np.zeros((H, W), np.float32)

    with torch.no_grad():
        for y in range(ph, H-ph, stride):
            batch = []
            locs = []
            for x in range(ph, W-ph, stride):
                patch_img = gray[y-ph:y+ph, x-ph:x+ph].astype(np.float32)/255.0
                batch.append(patch_img[None, None, ...])
                locs.append((y, x))
                if len(batch) == 256:
                    b = torch.from_numpy(np.concatenate(batch,0)).to(device)
                    preds = model(b).cpu().numpy()
                    for (yy, xx), val in zip(locs, preds):
                        out[yy-2:yy+3, xx-2:xx+3] += val # smear tiny kernel
                        counts[yy-2:yy+3, xx-2:xx+3] += 1
                    batch, locs = [], []
            if batch:
                b = torch.from_numpy(np.concatenate(batch,0)).to(device)
                preds = model(b).cpu().numpy()
                for (yy, xx), val in zip(locs, preds):
                    out[yy-2:yy+3, xx-2:xx+3] += val
                    counts[yy-2:yy+3, xx-2:xx+3] += 1

    out = np.divide(out, np.maximum(counts, 1e-6))
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out
