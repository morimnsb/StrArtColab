import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from ml.model import LineScoreModel
from ml.dataset import LineScoreDataset

# === CONFIG ===
IMAGE_DIR = "dataset/images"
SCORE_DIR = "dataset/line_scores"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/line_score_model.pth"

# === PREPARE DATA ===
dataset = LineScoreDataset(IMAGE_DIR, SCORE_DIR, max_samples_per_image=2000)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === INIT MODEL ===
model = LineScoreModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LR)

# === TRAINING LOOP ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, coords, targets in progress:
        images = images.to(DEVICE)
        coords = coords.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(images, coords)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"📉 Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"💾 Model saved to {CHECKPOINT_PATH}")
