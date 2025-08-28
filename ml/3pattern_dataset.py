# ml/pattern_dataset.py
import cv2, numpy as np, glob, torch
from torch.utils.data import Dataset
from importance.build_map import build_importance_map

class PatchDataset(Dataset):
    def __init__(self, img_paths, size=(400,400), patch=32, samples_per_img=2000):
        self.items = []
        self.patch = patch
        ph = patch // 2
        for p in img_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, size)
            M = build_importance_map(g)  # target
            H, W = g.shape
            ys = np.random.randint(ph, H-ph, size=samples_per_img)
            xs = np.random.randint(ph, W-ph, size=samples_per_img)
            for x,y in zip(xs,ys):
                patch_img = g[y-ph:y+ph, x-ph:x+ph].astype(np.float32)/255.0
                # target = mean importance around center 9x9
                t = M[y-4:y+5, x-4:x+5].mean().astype(np.float32)
                self.items.append((patch_img, t))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        patch_img, t = self.items[idx]
        return torch.from_numpy(patch_img)[None, ...], torch.tensor([t], dtype=torch.float32)
