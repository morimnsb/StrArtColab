import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class LineScoreDataset(Dataset):
    def __init__(self, image_dir, score_dir, transform=None, max_samples_per_image=5000):
        self.image_dir = image_dir
        self.score_dir = score_dir
        self.transform = transform
        self.samples = []

        for fname in os.listdir(score_dir):
            if not fname.endswith(".npy"):
                continue
            img_path = os.path.join(image_dir, fname.replace(".npy", ".png"))
            score_path = os.path.join(score_dir, fname)

            scores = np.load(score_path)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            h, w = image.shape
            nails = self.generate_nails((h, w))

            for _ in range(min(max_samples_per_image, len(nails)**2)):
                i, j = np.random.randint(0, len(nails), size=2)
                if i == j: continue

                score = scores[i, j]
                x1, y1 = nails[i]
                x2, y2 = nails[j]
                self.samples.append((img_path, (x1, y1, x2, y2), score))

        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, coords, score = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, 0)  # [1, H, W]
        coords = np.array(coords, dtype=np.float32) / 600  # normalize
        return torch.tensor(image), torch.tensor(coords), torch.tensor(score, dtype=torch.float32)

    def generate_nails(self, shape, count=360):
        h, w = shape
        center = (w // 2, h // 2)
        radius = int(min(w, h) * 0.45)
        nails = []
        for i in range(count):
            angle = 2 * np.pi * i / count
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            nails.append((x, y))
        return nails
