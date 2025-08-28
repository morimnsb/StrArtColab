#!/usr/bin/env python3
import os, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# same feature keys as in select_lines_progressive.py
FEATURE_KEYS = [
    "length","sector","sum_imp","mean_imp","mean_dark","sat_frac",
    "avail_frac","mean_need","endpoint_score_i","endpoint_score_j",
    "sector_load","deg_i","deg_j","angle",
]

# ---------------- model ----------------
class EdgeRankerMLP(nn.Module):
    def __init__(self, in_edge_feats: int, in_geom_feats: int = 7, hidden: int = 256):
        super().__init__()
        d = in_edge_feats + in_geom_feats
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    def forward(self, edge_feats, geom_feats):
        x = torch.cat([edge_feats, geom_feats], dim=-1)
        return self.net(x).squeeze(-1)

# ---------------- dataset ----------------
class EdgeDataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.edge_feats = np.stack([d for d in data["edge_feats"]], axis=0).astype(np.float32)
        self.geom_feats = np.stack([d for d in data["geom_feats"]], axis=0).astype(np.float32)
        self.labels     = np.array(data["labels"], dtype=np.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.edge_feats[idx], self.geom_feats[idx], self.labels[idx]

# ---------------- train ----------------
def train(args):
    dataset = EdgeDataset(args.dataset)
    N = len(dataset)
    idxs = np.arange(N); np.random.shuffle(idxs)
    split = int(0.8*N)
    train_idx, val_idx = idxs[:split], idxs[split:]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeRankerMLP(len(FEATURE_KEYS), 7, hidden=args.hidden).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    crit  = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs+1):
        model.train(); total_loss=0
        for xf, xg, y in train_loader:
            xf, xg, y = xf.to(device), xg.to(device), y.to(device)
            opt.zero_grad()
            out = model(xf, xg)
            loss = crit(out, y)
            loss.backward(); opt.step()
            total_loss += loss.item()*len(y)
        train_loss = total_loss/len(train_set)

        # val
        model.eval(); val_loss=0
        with torch.no_grad():
            for xf, xg, y in val_loader:
                xf, xg, y = xf.to(device), xg.to(device), y.to(device)
                out = model(xf, xg)
                loss = crit(out, y)
                val_loss += loss.item()*len(y)
        val_loss /= len(val_set)

        print(f"Epoch {epoch}/{args.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    # save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "hidden": args.hidden,
        "feature_keys": FEATURE_KEYS,
    }, args.out)
    print(f"✅ Saved NN ranker: {args.out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset with edge_feats, geom_feats, labels")
    ap.add_argument("--out", type=str, default="checkpoints/edge_ranker.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()
    train(args)

if __name__=="__main__":
    main()
