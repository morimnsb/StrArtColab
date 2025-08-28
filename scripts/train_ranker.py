#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset

# ------------- Model -------------
class LineRankerMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # [N,F] -> [N]
        return self.net(x).squeeze(-1)

# ------------- Utils -------------
def set_seed(seed: int):
    if seed is None: return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ndcg_at_k(scores: torch.Tensor, gains: torch.Tensor, k: int = 10) -> float:
    """Compute NDCG@k for one group (scores/gains: [G])."""
    G = scores.numel()
    if G == 0: return 0.0
    k = min(k, G)
    # rank by model score
    _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
    top_g = gains[idx]
    # discounted gain
    denom = torch.log2(torch.arange(2, k + 2, device=gains.device, dtype=gains.dtype))
    dcg = (top_g / denom).sum()

    # ideal
    _, idx_star = torch.topk(gains, k=k, largest=True, sorted=True)
    top_star = gains[idx_star]
    idcg = (top_star / denom).sum() + 1e-12
    return float((dcg / idcg).item())

def soft_targets(yg_t: torch.Tensor, norm='zscore', temperature=0.5) -> torch.Tensor:
    """Listwise soft targets from teacher gains (yg_t: [G])."""
    if norm == 'zscore':
        m = yg_t.mean()
        s = yg_t.std().clamp_min(1e-6)
        yg_t = (yg_t - m) / s
    elif norm == 'minmax':
        mn, mx = yg_t.min(), yg_t.max()
        yg_t = (yg_t - mn) / (mx - mn + 1e-6)
    # else: none
    return torch.softmax(yg_t / max(1e-6, temperature), dim=0)

def listwise_ce(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(scores, dim=0)
    return (-targets * logp).sum()

def pairwise_hinge(scores: torch.Tensor, gains: torch.Tensor, margin=0.1, num_pairs=32) -> torch.Tensor:
    """Sampled pairwise hinge loss inside a group to sharpen ordering."""
    G = scores.numel()
    if G < 2 or num_pairs <= 0: 
        return scores.new_tensor(0.0)
    # sample positives (high gain) vs negatives (low gain)
    # pick by quantiles for stability
    q_hi = torch.quantile(gains, 0.75)
    q_lo = torch.quantile(gains, 0.25)
    pos = torch.nonzero(gains >= q_hi, as_tuple=False).flatten()
    neg = torch.nonzero(gains <= q_lo, as_tuple=False).flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_tensor(0.0)

    idx_pos = pos[torch.randint(0, pos.numel(), (num_pairs,), device=scores.device)]
    idx_neg = neg[torch.randint(0, neg.numel(), (num_pairs,), device=scores.device)]
    s_pos = scores[idx_pos]
    s_neg = scores[idx_neg]
    return torch.relu(margin - (s_pos - s_neg)).mean()

# ------------- Dataset -------------
class GroupDataset(Dataset):
    """
    NPZ must contain:
      - features: X | feats | features  (float32)  [N, D]
      - targets : y | gains             (float32)  [N]
      - groups  : groups | group | grp  (int64)    [N]
    Training iterates by GROUP (listwise).
    """
    def __init__(self, npz_path: str, group_topk: int = 128, topk_strategy: str = "by_gain",
                 min_group: int = 2):
        z = np.load(npz_path, allow_pickle=True)

        def get_any(keys):
            for k in keys:
                if k in z.files: return z[k]
            return None

        X = get_any(['X', 'feats', 'features'])
        y = get_any(['y', 'gains'])
        g = get_any(['groups', 'group', 'grp'])

        if X is None or y is None or g is None:
            raise KeyError(f"Dataset must have X + (y/gains) + (groups). Found: {list(z.files)}")

        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        g = g.astype(np.int64, copy=False)

        # remap group ids to 0..G-1
        uniq = np.unique(g)
        id_map = {int(u): i for i, u in enumerate(uniq)}
        g2 = np.fromiter((id_map[int(v)] for v in g), dtype=np.int64, count=g.size)

        # collect indices per group
        all_by_group = [np.where(g2 == i)[0] for i in range(len(uniq))]
        # keep groups with enough samples
        self.group_topk = int(group_topk) if group_topk is not None else 0
        self.topk_strategy = topk_strategy
        kept = []
        for idxs in all_by_group:
            if idxs.size >= max(min_group, 1):
                kept.append(idxs)
        if not kept:
            raise ValueError("No groups satisfy minimal size.")

        # (optional) impose group_topk by best gains
        self.idxs_by_group: List[np.ndarray] = []
        if self.group_topk > 0:
            for idxs in kept:
                if idxs.size <= self.group_topk:
                    self.idxs_by_group.append(idxs)
                else:
                    if topk_strategy == "by_gain":
                        order = np.argsort(self.y[idxs])[::-1]  # high gain first
                        self.idxs_by_group.append(idxs[order[:self.group_topk]])
                    elif topk_strategy == "random":
                        sel = np.random.choice(idxs, size=self.group_topk, replace=False)
                        self.idxs_by_group.append(sel)
                    else:
                        self.idxs_by_group.append(idxs[:self.group_topk])
        else:
            self.idxs_by_group = kept

        # global stats (over kept samples)
        all_idxs = np.concatenate(self.idxs_by_group, axis=0)
        Xk = self.X[all_idxs]
        self.mean = Xk.mean(axis=0).astype(np.float32)
        self.std  = (Xk.std(axis=0) + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.idxs_by_group)

    def __getitem__(self, gid: int) -> Tuple[np.ndarray, np.ndarray]:
        idxs = self.idxs_by_group[gid]
        return self.X[idxs], self.y[idxs]

def collate_groups(batch):
    return batch

# ------------- Train/Eval loops -------------
@dataclass
class TrainState:
    epoch: int = 0
    best_val: float = float('inf')
    best_path: Optional[str] = None

def run_epoch(model: nn.Module,
              dl: DataLoader,
              opt: Optional[optim.Optimizer],
              device: torch.device,
              temperature: float,
              gain_norm: str,
              feat_mean: Optional[np.ndarray],
              feat_std: Optional[np.ndarray],
              use_amp: bool = True,
              aux_pairwise_w: float = 0.0,
              clip_grad: float = 1.0,
              train: bool = True,
              log_k: int = 10) -> Tuple[float, float, float]:

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    if train: model.train()
    else: model.eval()

    total = 0.0; denom = 0
    acc_n = 0; acc_d = 0
    ndcg_sum = 0.0; ndcg_d = 0

    mean_t = torch.from_numpy(feat_mean).to(device) if feat_mean is not None else None
    std_t  = torch.from_numpy(feat_std ).to(device) if feat_std  is not None else None

    for batch in dl:
        loss_batch = 0.0
        if train: 
            opt.zero_grad(set_to_none=True)

        for Xg_np, yg_np in batch:
            Xg = torch.from_numpy(Xg_np).to(device)
            yg = torch.from_numpy(yg_np).to(device)

            if mean_t is not None and std_t is not None:
                Xg = (Xg - mean_t) / std_t

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                scores = model(Xg)  # [G]
                tgt = soft_targets(yg, norm=gain_norm, temperature=temperature)
                loss = listwise_ce(scores, tgt)
                if aux_pairwise_w > 0.0:
                    loss = loss + aux_pairwise_w * pairwise_hinge(scores, yg)

            if train:
                scaler.scale(loss).backward()
            loss_batch += float(loss.detach().item())

            # metrics
            pred_top = int(torch.argmax(scores).item())
            true_top = int(torch.argmax(yg).item())
            acc_n += int(pred_top == true_top); acc_d += 1
            ndcg_sum += ndcg_at_k(scores.detach(), yg.detach(), k=log_k); ndcg_d += 1

        if train:
            if clip_grad is not None and clip_grad > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(opt)
            scaler.update()

        total += loss_batch / max(1, len(batch))
        denom += 1

    avg_loss = total / max(1, denom)
    top1 = acc_n / max(1, acc_d)
    ndcg = ndcg_sum / max(1, ndcg_d)
    return avg_loss, top1, ndcg

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--groups_per_batch', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.0)

    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--warmup_epochs', type=int, default=1)
    ap.add_argument('--cosine', action='store_true', help='Use cosine decay after warmup')

    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--gain_norm', choices=['zscore','minmax','none'], default='zscore')
    ap.add_argument('--feat_norm', choices=['zscore','none'], default='zscore')

    ap.add_argument('--group_topk', type=int, default=128)
    ap.add_argument('--topk_strategy', choices=['by_gain','random','first'], default='by_gain')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--aux_pairwise_w', type=float, default=0.0, help='>0 to enable pairwise hinge aux loss')
    ap.add_argument('--clip_grad', type=float, default=1.0)
    ap.add_argument('--no_amp', action='store_true')

    ap.add_argument('--out', default='checkpoints/edge_ranker.pt')
    ap.add_argument('--resume', default=None, help='Path to .pt to resume (optimizer+epoch if present)')
    ap.add_argument('--early_patience', type=int, default=5, help='Early stop on val loss')

    args = ap.parse_args()
    set_seed(args.seed)

    # Dataset
    ds = GroupDataset(args.data, group_topk=args.group_topk, topk_strategy=args.topk_strategy)
    num_groups = len(ds)
    if num_groups <= 1:
        raise ValueError(f"Not enough groups to split: num_groups={num_groups}")

    # Stable group-wise split
    rng = np.random.default_rng(123)
    perm = rng.permutation(num_groups)
    n_val = max(1, int(round(args.val_frac * num_groups)))
    va_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    print(f"[dataset] groups kept: {num_groups} (train={len(tr_idx)}, val={len(va_idx)})")
    print(f"[dataset] feature_dim={ds.X.shape[1]}  mean/std computed over kept samples")

    tr = DataLoader(Subset(ds, tr_idx),
                    batch_size=args.groups_per_batch,
                    shuffle=True, drop_last=False,
                    num_workers=0, collate_fn=collate_groups)
    va = DataLoader(Subset(ds, va_idx),
                    batch_size=args.groups_per_batch,
                    shuffle=False, drop_last=False,
                    num_workers=0, collate_fn=collate_groups)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LineRankerMLP(in_dim=ds.X.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Scheduler: warmup + cosine
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return max(1e-3, (epoch + 1) / max(1, args.warmup_epochs))
        if args.cosine:
            t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
        return 1.0
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    use_amp = (not args.no_amp)
    use_feat_norm = (args.feat_norm == 'zscore')
    fmean, fstd = (ds.mean, ds.std) if use_feat_norm else (None, None)

    # Resume (optional)
    start_ep = 1
    best_val = float('inf')
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        if 'model' in ck:
            model.load_state_dict(ck['model'])
        if 'opt' in ck:
            opt.load_state_dict(ck['opt'])
        if 'sched' in ck:
            sched.load_state_dict(ck['sched'])
        start_ep = int(ck.get('epoch', 0)) + 1
        best_val = float(ck.get('best_val', best_val))
        print(f"[resume] from {args.resume} at epoch {start_ep-1}  best_val={best_val:.4f}")

    # Train
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_path = args.out
    patience = args.early_patience
    bad = 0

    for ep in range(start_ep, args.epochs + 1):
        tr_loss, tr_acc, tr_ndcg = run_epoch(
            model, tr, opt, device,
            temperature=args.temperature,
            gain_norm=args.gain_norm,
            feat_mean=fmean, feat_std=fstd,
            use_amp=use_amp,
            aux_pairwise_w=args.aux_pairwise_w,
            clip_grad=args.clip_grad,
            train=True, log_k=10
        )
        sched.step()

        va_loss, va_acc, va_ndcg = run_epoch(
            model, va, None, device,
            temperature=args.temperature,
            gain_norm=args.gain_norm,
            feat_mean=fmean, feat_std=fstd,
            use_amp=use_amp,
            aux_pairwise_w=0.0,
            clip_grad=0.0,
            train=False, log_k=10
        )

        lr_now = opt.param_groups[0]['lr']
        print(f"Epoch {ep:03d} | lr={lr_now:.2e} | "
              f"train: loss={tr_loss:.4f}, top1={tr_acc*100:.1f}%, ndcg@10={tr_ndcg:.3f} | "
              f"val: loss={va_loss:.4f}, top1={va_acc*100:.1f}%, ndcg@10={va_ndcg:.3f}")

        # Save last
        last_path = os.path.splitext(args.out)[0] + "_last.pt"
        torch.save({
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'epoch': ep,
            'best_val': best_val,
        }, last_path)

        # Early stopping on val loss
        if va_loss + 1e-8 < best_val:
            best_val = va_loss
            bad = 0
            # Save best in the format the selector expects
            torch.save({
                'model': model.state_dict(),
                'hidden': args.hidden,
                'feature_keys': ['sum','mean','max','var','dens','length','sector','d_ang','cool_i','cool_j','sec_gap'],
                'norm_mean': (ds.mean if use_feat_norm else None),
                'norm_std':  (ds.std  if use_feat_norm else None),
                'loss': 'listwise_ce',
                'temperature': args.temperature,
                'gain_norm': args.gain_norm,
                'group_topk': args.group_topk,
                'dropout': args.dropout,
            }, args.out)
            best_path = args.out
            print(f"✔ Saved BEST -> {best_path}  (val_loss={best_val:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"⏹ Early stop at epoch {ep} (no improvement for {patience} epochs).")
                break

    # Save a small training report
    report = dict(
        data=os.path.abspath(args.data),
        epochs=ep,
        best_val=float(best_val),
        out=os.path.abspath(best_path),
        feature_keys=['sum','mean','max','var','dens','length','sector','d_ang','cool_i','cool_j','sec_gap'],
        feat_norm=args.feat_norm,
        gain_norm=args.gain_norm,
        temperature=args.temperature,
        group_topk=args.group_topk,
        hidden=args.hidden,
        seed=args.seed,
    )
    with open(os.path.splitext(best_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Report ->", os.path.splitext(best_path)[0] + ".json")

if __name__ == '__main__':
    main()
