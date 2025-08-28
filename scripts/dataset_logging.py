# scripts/dataset_logging.py
import os, json, time, hashlib
import numpy as np
import cv2

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def resize01(x: np.ndarray, hw=(100,100), interp=cv2.INTER_AREA):
    x = x.astype(np.float32)
    H, W = hw
    if x.ndim == 2:
        return cv2.resize(x, (W, H), interpolation=interp)
    elif x.ndim == 3:
        return cv2.resize(x, (W, H), interpolation=interp)
    else:
        raise ValueError("resize01 expects 2D or 3D array")

class StepLogger:
    """
    Create one per run. At each step, call .log_step(...) with:
    - list of candidate dicts (features+metadata). Mark chosen with label=1, others 0.
    - pre-commit maps (importance, darkness), we save downsampled .npy for each step.

    Files:
      runs/<run_id>/
        config.json
        step_0001.jsonl
        step_0001_dark.npy
        step_0001_imp.npy
        ...
    """
    def __init__(self, root_dir: str, img_id: str, params: dict, nails_id: str,
                 small_hw=(100,100)):
        self.root_dir = root_dir
        self.img_id   = img_id
        self.params   = params or {}
        self.nails_id = nails_id
        self.small_hw = small_hw

        stamp = time.strftime("%Y%m%d_%H%M%S")
        phash = _hash(json.dumps(self.params, sort_keys=True))
        self.run_id = f"{img_id}_{nails_id}_{stamp}_{phash}"
        self.run_dir = os.path.join(root_dir, "runs", self.run_id)
        _ensure_dir(self.run_dir)

        cfg = {
            "img_id": img_id,
            "nails_id": nails_id,
            "params": self.params,
            "small_hw": list(self.small_hw),
            "created": stamp,
            "run_id": self.run_id,
        }
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        self._opened = {}

    def _step_path(self, t: int, kind: str):
        base = f"step_{t:04d}"
        if kind == "jsonl":
            return os.path.join(self.run_dir, f"{base}.jsonl")
        elif kind == "dark":
            return os.path.join(self.run_dir, f"{base}_dark.npy")
        elif kind == "imp":
            return os.path.join(self.run_dir, f"{base}_imp.npy")
        else:
            raise ValueError(kind)

    def log_step(self, t: int, candidates: list, M_eff: np.ndarray, darkness: np.ndarray):
        """
        candidates: list of dicts, each MUST include keys:
          - 'img_id','run_id','t','candidate':{...}, 'features':{...}, 'label' (0/1)
          - Optional: 'gt_gain'
        M_eff: current importance map (float32, HxW, 0..1 recommended)
        darkness: current darkness map BEFORE committing the chosen line (float32, HxW)
        """
        # write candidate rows
        jpath = self._step_path(t, "jsonl")
        with open(jpath, "w", encoding="utf-8") as f:
            for row in candidates:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # write small arrays
        small_imp  = resize01(M_eff, self.small_hw)
        small_dark = resize01(darkness, self.small_hw)
        if small_imp is not None and small_imp.size > 0:
            np.save(self._step_path(t, "imp"), small_imp.astype(np.float32))
        else:
            # Save a placeholder so downstream doesn’t crash
            np.save(self._step_path(t, "imp"), np.zeros((1,), dtype=np.float32))

        np.save(self._step_path(t, "dark"), small_dark.astype(np.float32))

    def manifest_row(self):
        return {
            "run_id": self.run_id,
            "img_id": self.img_id,
            "nails_id": self.nails_id,
            "root": self.run_dir,
            "params_hash": _hash(json.dumps(self.params, sort_keys=True)),
        }
