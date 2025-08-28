# utils/line_cache.py
import os, json, glob
import numpy as np

_STRUCT_KEYS = {"ys","xs","pairs","sectors","lengths","nails","shape_hw","nail_shape","num_nails","min_dist","num_sectors"}

def _to_ndarray(obj, dtype=None):
    """Coerce to ndarray (object dtype allowed)."""
    arr = np.asarray(obj, dtype=dtype) if dtype is not None else np.asarray(obj)
    return arr

def load_line_cache(shape_hw, nail_shape, num_nails, min_dist, num_sectors, cache_dir="cache"):
    h, w = int(shape_hw[0]), int(shape_hw[1])
    base = f"lines_{nail_shape}_{num_nails}_{w}x{h}_d{min_dist}_s{num_sectors}"
    core_path = os.path.join(cache_dir, base + ".npz")
    if not os.path.exists(core_path):
        return None, None

    z = np.load(core_path, allow_pickle=True)

    # Core structural arrays from the NPZ (authoritative)
    ys       = z["ys"]       # object ndarray (ragged)
    xs       = z["xs"]       # object ndarray (ragged)
    pairs    = z["pairs"]    # int32 ndarray, shape [N,2]
    sectors  = z["sectors"]  # int32 ndarray, shape [N]
    lengths  = z["lengths"]  # int32 ndarray, shape [N]
    nails    = z["nails"]    # int32 ndarray, shape [num_nails,2]

    # Build cache dict (do NOT let meta overwrite these arrays)
    cache = {
        "ys": ys,
        "xs": xs,
        "pairs": pairs,
        "sectors": sectors,
        "lengths": lengths,
        "nails": nails,
        "shape_hw": (h, w),
        "nail_shape": nail_shape,
        "num_nails": int(num_nails),
        "min_dist": int(min_dist),
        "num_sectors": int(num_sectors),
    }

    # Merge selected metadata keys only (ignore structural ones if present)
    try:
        meta = json.loads(str(z["meta"]))
        if isinstance(meta, dict):
            for k, v in meta.items():
                if k in _STRUCT_KEYS:
                    continue  # don't overwrite arrays/struct
                cache[k] = v
    except Exception:
        pass

    # Normalize types in case something slipped in:
    if isinstance(cache["pairs"], list):    cache["pairs"] = _to_ndarray(cache["pairs"], dtype=np.int32)
    if isinstance(cache["sectors"], list):  cache["sectors"] = _to_ndarray(cache["sectors"], dtype=np.int32)
    if isinstance(cache["lengths"], list):  cache["lengths"] = _to_ndarray(cache["lengths"], dtype=np.int32)
    if isinstance(cache["nails"], list):    cache["nails"] = _to_ndarray(cache["nails"], dtype=np.int32)

    total = int(cache["pairs"].shape[0]) if hasattr(cache["pairs"], "shape") else len(cache["pairs"])

    # ---- Optional AA mask bounds + shards ----
    bounds_path = os.path.join(cache_dir, f"{base}_bounds.npz")
    if os.path.exists(bounds_path):
        b = np.load(bounds_path, allow_pickle=True)
        cache["roi_y0"] = b["roi_y0"].astype(np.int32)
        cache["roi_y1"] = b["roi_y1"].astype(np.int32)
        cache["roi_x0"] = b["roi_x0"].astype(np.int32)
        cache["roi_x1"] = b["roi_x1"].astype(np.int32)
        num_shards = int(b["num_shards"])

        # Map each line index -> (shard id, position in shard)
        shard_files = sorted(glob.glob(os.path.join(cache_dir, f"{base}_masks_*.npz")))
        if len(shard_files) != num_shards:
            # Shards incomplete; skip masks to avoid crashes
            return cache, core_path

        idx2shard = np.empty(total, np.int32)
        pos_in_shard = np.empty(total, np.int32)

        shards = []
        for s, spath in enumerate(shard_files):
            zsh = np.load(spath, allow_pickle=True)
            lo = int(zsh["lo"]); hi = int(zsh["hi"])
            shards.append({"path": spath, "lo": lo, "hi": hi})
            idx2shard[lo:hi] = s
            pos_in_shard[lo:hi] = np.arange(0, hi - lo, dtype=np.int32)

        class _MaskProxy:
            """Lazy loader for sharded uint8 masks."""
            def __init__(self, shards, idx2shard, pos_in_shard):
                self.shards = shards
                self.idx2shard = idx2shard
                self.pos_in_shard = pos_in_shard
                self._cache = {}  # shard_id -> object ndarray of uint8 masks

            def __getitem__(self, k):
                s = int(self.idx2shard[k])
                if s not in self._cache:
                    zsh = np.load(self.shards[s]["path"], allow_pickle=True)
                    self._cache[s] = zsh["masks"]  # object-array of uint8 2D arrays
                return np.asarray(self._cache[s][int(self.pos_in_shard[k])], dtype=np.uint8)

            def __len__(self):
                return len(self.idx2shard)

        class _MasksView:
            """Simple adapter so caller can index like cache['masks'][k]."""
            def __init__(self, proxy): self.proxy = proxy
            def __getitem__(self, k):   return self.proxy[k]
            def __len__(self):          return len(self.proxy)

        cache["masks_u8_proxy"] = _MaskProxy(shards, idx2shard, pos_in_shard)
        cache["masks"] = _MasksView(cache["masks_u8_proxy"])

    return cache, core_path
