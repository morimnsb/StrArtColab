#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
from torch.amp import GradScaler, autocast

# ---------------------------
# utils
# ---------------------------

def set_seed(seed: int = 123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(arg: str = "auto") -> torch.device:
    """Return a torch.device based on user arg and availability, and print it."""
    arg = (arg or "auto").lower()
    if arg == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif arg in ("cuda", "gpu"):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            print("[WARN] CUDA requested but not available; falling back to CPU.")
    elif arg == "cpu":
        dev = "cpu"
    else:
        print(f"[WARN] Unrecognized --device '{arg}', using auto.")
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = torch.device(dev)
    print(f"[INFO] Using device: {d} | cuda_available={torch.cuda.is_available()}")
    if d.type == "cuda":
        try:
            print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA capability: {torch.cuda.get_device_capability(0)}")
        except Exception:
            pass
    return d

# --------- Model (same shape as at inference) ----------
class EdgeRanker(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        if dropout > 0: layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0: layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# --------- Data loading from NPZ logs ----------
def load_npzs(npz_paths):
    Xs, Gs, Rs = [], [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        Xs.append(z['X'].astype(np.float32))
        Gs.append(z['gains'].astype(np.float32))
        Rs.append(z['groups'].astype(np.int64))
    X = np.concatenate(Xs, 0); gains = np.concatenate(Gs, 0); groups = np.concatenate(Rs, 0)
    return X, gains, groups

class PairwiseDataset(Dataset):
    """Builds random positive/negative pairs from groups (for your existing 'pairwise')."""
    def __init__(self, X, gains, groups, eps=1e-3):
        self.X = X; self.g = gains; self.grp = groups
        # index groups -> indices
        self.by: Dict[int, List[int]] = {}
        for i, gid in enumerate(groups):
            self.by.setdefault(int(gid), []).append(i)
        # use only groups with >=2 items
        self.keys = [k for k,v in self.by.items() if len(v) >= 2]
        self.eps = float(eps)

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        gid = self.keys[idx]
        idxs = self.by[gid]
        vals = self.g[idxs]
        # choose a better and worse example
        hi = int(idxs[int(np.argmax(vals))])
        lo = int(idxs[int(np.argmin(vals))])
        # fall back if flat
        if abs(self.g[hi] - self.g[lo]) < self.eps:
            lo = int(idxs[0]); hi = int(idxs[-1])
        return self.X[hi], self.X[lo]

class RegDataset(Dataset):
    def __init__(self, X, gains, groups):
        self.X = X; self.y = gains
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class ListwiseGroupDataset(Dataset):
    """Returns one whole shortlist (variable length) for 'listwise'."""
    def __init__(self, X, gains, groups):
        self.X = X; self.g = gains; self.grp = groups
        self.by: Dict[int, List[int]] = {}
        for i, gid in enumerate(groups):
            self.by.setdefault(int(gid), []).append(i)
        # keep only groups with at least 2 items
        self.groups = [k for k,v in self.by.items() if len(v) >= 2]

    def __len__(self): return len(self.groups)

    def __getitem__(self, i):
        gid = self.groups[i]
        idxs = np.asarray(self.by[gid], dtype=np.int64)
        return torch.from_numpy(self.X[idxs]), torch.from_numpy(self.g[idxs])

def listwise_collate(batch):
    # batch is a list of (X_g [Ng,D], y_g [Ng]) of variable Ng
    return batch  # train loop will iterate per group inside the batch

# --------- Losses ----------
def pairwise_loss(scores_pos, scores_neg):
    # hinge-like: max(0, 1 - (s_pos - s_neg))
    return torch.clamp(1.0 - (scores_pos - scores_neg), min=0.0).mean()

def reg_loss(scores, targets):
    return nn.functional.mse_loss(scores, targets)

def listnet_loss(scores_g, gains_g, temperature=1.0):
    """
    ListNet: cross-entropy between softmax(gains/τ) and softmax(scores).
    scores_g: [N], gains_g: [N]
    """
    P = torch.softmax(gains_g / temperature, dim=0)
    Q = torch.log_softmax(scores_g, dim=0)
    return -(P * Q).sum()

# --------- Main training ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True, help='One or more NPZ logs (X,gains,groups)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--loss', choices=['pairwise','reg','listwise'], default='pairwise')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=4096, help='Batch size: items for reg/pairwise; groups per batch for listwise')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-6)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--pair_eps', type=float, default=1e-3)
    ap.add_argument('--list_temp', type=float, default=1.0, help='Temperature for ListNet softmax on gains')
    # NEW:
    ap.add_argument('--device', type=str, default='auto', help='auto|cuda|cpu')
    ap.add_argument('--amp', action='store_true', help='Enable mixed precision on CUDA')

    args = ap.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    X, gains, groups = load_npzs(args.npz)
    in_dim = int(X.shape[1])

    # split by groups for validation (more stable)
    uniq = np.unique(groups)
    np.random.shuffle(uniq)
    n_val = max(1, int(len(uniq) * args.val_frac))
    val_groups = set(uniq[:n_val])
    train_mask = ~np.isin(groups, list(val_groups))
    val_mask   =  np.isin(groups, list(val_groups))

    Xtr, gtr, rtr = X[train_mask], gains[train_mask], groups[train_mask]
    Xva, gva, rva = X[val_mask],   gains[val_mask],   groups[val_mask]

    model = EdgeRanker(in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))


    # datasets
    pin = (device.type == "cuda")
    if args.loss == 'pairwise':
        ds_tr = PairwiseDataset(Xtr, gtr, rtr, eps=args.pair_eps)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=2, pin_memory=pin)
    elif args.loss == 'reg':
        ds_tr = RegDataset(Xtr, gtr, rtr)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=2, pin_memory=pin)
    else:  # listwise
        ds_tr = ListwiseGroupDataset(Xtr, gtr, rtr)
        dl_tr = DataLoader(ds_tr, batch_size=max(1, args.bs), shuffle=True,
                           collate_fn=listwise_collate, drop_last=False, num_workers=2, pin_memory=pin)

    # simple validation pair-accuracy (same as before)
    def val_pair_acc():
        by: Dict[int, List[int]] = {}
        for i, gid in enumerate(rva): by.setdefault(int(gid), []).append(i)
        correct = 0; total = 0
        model.eval()
        with torch.no_grad():
            for gid, idxs in by.items():
                if len(idxs) < 2: continue
                feat = torch.from_numpy(Xva[idxs]).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                    s = model(feat).detach().cpu().numpy()
                y = gva[idxs]
                # compare best vs worst
                i_hi, i_lo = int(np.argmax(y)), int(np.argmin(y))
                total += 1
                correct += 1 if (s[i_hi] > s[i_lo]) else 0
        return correct / max(1,total)

    # training loop
    best_va = -1.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)
            if args.loss == 'pairwise':
                xp, xn = batch
                xp = torch.from_numpy(xp).to(device, non_blocking=True) if isinstance(xp, np.ndarray) else xp.to(device, non_blocking=True)
                xn = torch.from_numpy(xn).to(device, non_blocking=True) if isinstance(xn, np.ndarray) else xn.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                    sp = model(xp); sn = model(xn)
                    loss = pairwise_loss(sp, sn)
            elif args.loss == 'reg':
                x, y = batch
                x = torch.from_numpy(x).to(device, non_blocking=True) if isinstance(x, np.ndarray) else x.to(device, non_blocking=True)
                y = torch.from_numpy(y).to(device, non_blocking=True) if isinstance(y, np.ndarray) else y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                    s = model(x)
                    loss = reg_loss(s, y)
            else:  # listwise
                total = 0.0
                for Xg, yg in batch:
                    Xg = Xg.to(device, non_blocking=True); yg = yg.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                        sg = model(Xg)
                        total = total + listnet_loss(sg, yg, temperature=args.list_temp)
                loss = total / max(1, len(batch))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            running += float(loss.item())

        va = val_pair_acc()
        print(f"[{ep}/{args.epochs}] train_loss_sum={running:.3f}  val_pair_acc={va:.3f}")

        if va > best_va:
            best_va = va
            ckpt = dict(
                model=model.state_dict(),
                hidden=args.hidden, dropout=args.dropout,
                norm_mean=None, norm_std=None,   # fill if you later add feature normalization
                feature_keys=None,
                gain_norm='zscore',
                temperature=1.0,
                group_topk=None,
            )
            torch.save(ckpt, args.out)
            print(f"  ↳ saved checkpoint: {args.out}")

    print(f"✅ best val_pair_acc={best_va:.3f}  saved to: {args.out}")

if __name__ == '__main__':
    main()
