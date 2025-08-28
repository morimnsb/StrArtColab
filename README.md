# 🎨 StrArt – AI-Assisted String Art Simulation

StrArt is a **string art simulation framework** that reconstructs images by drawing lines between nails on a frame.  
It combines:
- **Rule-based importance maps** (edges, contrast, letter masking)
- **Learned importance maps** (PatchCNN trained on patches)
- **Greedy line selection** with penalties for overlap and coverage saturation
- **Precomputed line caches** for speed

![Example Output](docs/example_result.png)
![Progress GIF](docs/example_progress.gif)

---

## 📂 Project Structure

```
StrArt/
├─ data/                   # input images + training images
├─ cache/                  # precomputed line pixel caches
├─ checkpoints/            # trained ML models
├─ outputs_baseline/       # results from rule-based sim
├─ outputs_importance_tracked/ # learned-importance tracked results
├─ utils/                  # nails, preprocessing, cache utils
├─ scoring/                # line pixelization (Bresenham)
├─ selection/              # greedy selector
├─ importance/             # build/infer importance maps
├─ ml/                     # PatchCNN model
├─ scripts/                # runnable simulation + training scripts
└─ archive/                # (optional) old files moved by cleanup
```

---

## 🚀 Installation

```bash
git clone https://github.com/yourname/StrArt.git
cd StrArt
python -m venv env
source env/bin/activate  # on Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## ⚙️ Build Line Cache (Once per Layout)

Before simulating, precompute pixel coordinates for all nail→nail lines.

```bash
python -m scripts.build_line_cache
```

You can configure in `scripts/build_line_cache.py`:
- `NAIL_SHAPE` = `"circle"` or `"rectangle"`
- `NUM_NAILS` = number of nails
- `IMAGE_SIZE` = `(400, 400)`
- `MIN_DIST` = minimum nail distance
- `NUM_SECTORS` = number of angular sectors

---

## 🧵 Run Simulations

### 1. **Fast Baseline** (Rule-Based Importance Map)
```bash
python -m scripts.simulate_baseline
```
Outputs to:
```
outputs_baseline/simulated_result.png
```

---

### 2. **Tracked Baseline** (Per-Line CSV + Frames)
```bash
python -m scripts.simulate_baseline_tracked   --image data/A.png   --num_lines 300   --save_every 1
```
Saves:
- `outputs_baseline_tracked/simulated_result.png`
- `outputs_baseline_tracked/line_log.csv`
- `outputs_baseline_tracked/progress_frames/`

---

### 3. **Train PatchCNN** (Learned Importance Map)
```bash
python -m scripts.train_patterns
```
This uses `data/train_images/` to learn patch → importance mappings.

---

### 4. **Simulate with Learned Importance (Tracked)**
```bash
python -m scripts.simulate_from_importance_tracked   --use_learned   --ckpt checkpoints/pattern_cnn.pth   --num_lines 300   --save_every 1
```

---
> python -m scripts.select_lines_progressive `
>>   --image data/A.png --size 400,400 `
>>   --nail_shape circle --num_nails 360 --num_sectors 12 --min_dist 30 `
>>   --mask_only --normalize_by_length `
>>   --num_lines 300 `
>>   --lambda_overdraw 2.0 `
>>   --topk_candidates 3000 `
>>   --darkness_per_line 25 --max_hits_per_pixel 10 `
>>   --endpoint_bias_alpha 0.5 `
>>   --use_nail_csv outputs_toppoints/top_nails.csv `
>>   --nail_radius_px 6 --nail_reduce sum `
>>   --lambda_overdraw 5.0 `                     
>>   --topk_candidates 3000 `
>>   --darkness_per_line 25 --max_hits_per_pixel 10 `
>>   --endpoint_bias_alpha 0.5 `
>>   --use_nail_csv outputs_toppoints/top_nails.csv `
>>   --nail_radius_px 6 --nail_reduce sum `
>>   --num_lines 3 `
>>   --lambda_overdraw 2.0 `
>>   --topk_candidates 3000 `
>>   --darkness_per_line 25 --max_hits_per_pixel 10 `
>>   --endpoint_bias_alpha 0.5 `
>>   --use_nail_csv outputs_toppoints/top_nails.csv `
>>   --nail_radius_px 6 --nail_reduce sum `
>>   --num_lines 300 `
>>   --lambda_overdraw 2.0 `
>>   --topk_candidates 3000 `
>>   --darkness_per_line 25 --max_hits_per_pixel 10 `
>>   --endpoint_bias_alpha 0.5 `
>>   --use_nail_csv outputs_toppoints/top_nails.csv `
>>   --nail_radius_px 6 --nail_reduce sum `
>>   --max_per_sector 60 --max_lines_per_nail 6 `
>>   --save_every 10
## 📊 Analysis

### Make a GIF from Progress Frames
```bash
python -m scripts.make_gif
```

### Analyze Line Log
```bash
python -m scripts.analyze_log
```

---

## 🖼 Example Outputs

| Stage | Image |
|-------|-------|
| Initial | ![Initial Image](docs/example_initial.png) |
| Halfway | ![Halfway Image](docs/example_halfway.png) |
| Final | ![Final Image](docs/example_result.png) |

---

## 🔮 Roadmap
- [ ] Add **sector-based line caching CLI**
- [ ] Improve PatchCNN training with **more augmentations**
- [ ] Implement **force-fill staged relaxation**
- [ ] Web UI for interactive simulation

---

## 📜 License
MIT License © 2025 Morimn
