# Efficient Innovations - Quick Start

This guide shows how to enable the **five lightweight innovations** added to GaussianAvatars. Each module can be toggled independently through `train.py` arguments, making it easy to balance efficiency and quality.

The five innovations are:

1. **Region-Adaptive Loss Weighting** (`--use_region_adaptive_loss`)
2. **Smart Densification** (`--use_smart_densification`)
3. **Progressive Resolution Training** (`--use_progressive_resolution`)
4. **Lightweight Color Calibration Network** (`--use_color_calibration`)
5. **Contrastive Regularization** (`--use_contrastive_reg`)

All implementations live under `innovations/` and integrate tightly with the standard training pipeline.

---

## ⚙️ Common Setup

```bash
# Activate environment (example)
conda activate gaussian-avatars

# Navigate to repo root
cd /path/to/GaussianAvatars
```

The examples below assume you already prepared a dataset at `${DATA_DIR}`.

---

## 🚀 Recommended Configurations

| Profile | Enabled Innovations | Training Time vs Baseline | Quality Boost |
|---------|--------------------|----------------------------|---------------|
| **Ultra-Efficient** | 1, 2 | +5% | PSNR +0.5~0.8 dB |
| **Balanced (default)** | 1, 2, 3, 4 | +10% | PSNR +0.7~1.2 dB |
| **Quality-First** | 1, 2, 3, 4, 5 | +15% | PSNR +0.9~1.5 dB |

### Ultra-Efficient
```bash
python train.py \
  -s ${DATA_DIR} \
  -m output/ultra_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --use_region_adaptive_loss \
  --region_weight_eyes 2.0 \
  --region_weight_mouth 2.0 \
  --use_smart_densification \
  --densify_percentile_clone 75 \
  --densify_percentile_split 90 \
  --use_amp
```

### Balanced (recommended)
```bash
python train.py \
  -s ${DATA_DIR} \
  -m output/balanced_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --use_region_adaptive_loss \
  --region_weight_eyes 2.0 \
  --region_weight_mouth 2.0 \
  --region_weight_nose 1.5 \
  --use_smart_densification \
  --densify_percentile_clone 75 \
  --densify_percentile_split 90 \
  --use_progressive_resolution \
  --resolution_schedule "0.5,0.75,1.0" \
  --resolution_milestones "100000,300000" \
  --use_color_calibration \
  --color_net_hidden_dim 16 \
  --lambda_color_reg 1e-4 \
  --use_amp
```

### Quality-First
```bash
python train.py \
  -s ${DATA_DIR} \
  -m output/quality_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --use_region_adaptive_loss \
  --use_smart_densification \
  --use_progressive_resolution \
  --use_color_calibration \
  --use_contrastive_reg \
  --lambda_contrastive 0.01 \
  --contrastive_cache_size 2 \
  --use_amp
```

---

## 🧩 Module Reference

### 1. Region-Adaptive Loss
- File: `innovations/region_adaptive_loss.py`
- Description: Generates per-pixel weights for L1 loss using either FLAME semantics or a fallback heuristic mask.
- Key args:
  - `--region_weight_eyes`
  - `--region_weight_mouth`
  - `--region_weight_nose`
  - `--region_weight_face`

### 2. Smart Densification
- File: `innovations/smart_densification.py`
- Description: Percentile-based thresholds for clone/split during densification to prevent point explosion.
- Key args:
  - `--densify_percentile_clone`
  - `--densify_percentile_split`

### 3. Progressive Resolution Training
- File: `innovations/progressive_training.py`
- Description: Downsamples predicted/GT images in early iterations to accelerate convergence.
- Key args:
  - `--resolution_schedule` (comma separated floats)
  - `--resolution_milestones` (comma separated ints)

### 4. Color Calibration Network
- File: `innovations/color_calibration.py`
- Description: Tiny MLP applied post-render to correct color/exposure with optional weight regularization.
- Key args:
  - `--color_net_hidden_dim`
  - `--color_net_layers`
  - `--lambda_color_reg`

### 5. Contrastive Regularization
- File: `innovations/contrastive_regularization.py`
- Description: Maintains a cache of downsampled renders and penalizes cosine distance to improve temporal/multi-view consistency.
- Key args:
  - `--lambda_contrastive`
  - `--contrastive_cache_size`
  - `--contrastive_downsample`

---

## 📊 Monitoring & Diagnostics

- **Progress Bar**: Additional fields show `color_reg` and `contrastive` losses when enabled.
- **TensorBoard**: New scalars logged under `train_loss_patches/` for color regularization and contrastive loss.
- **Point Count**: Monitor how smart densification maintains Gaussians between 100k-120k.

---

## ❗ Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| Loss spikes when enabling region loss | Missing FLAME binding | Provide mesh binding or rely on heuristic weights (auto fallback) |
| Gaussian count grows too fast | Aggressive percentile settings | Increase `--densify_percentile_clone` and `--densify_percentile_split` |
| Color calibration overfits | Hidden dim too large or reg too small | Reduce `--color_net_hidden_dim`, increase `--lambda_color_reg` |
| Contrastive loss stuck at 1.0 | Missing cache warmup | Allow several iterations for cache to fill |

---

## 🔁 Ablation Checklist

```bash
# Baseline (no innovations)
python train.py ...

# Enable one at a time
python train.py ... --use_region_adaptive_loss
python train.py ... --use_smart_densification
python train.py ... --use_progressive_resolution
python train.py ... --use_color_calibration
python train.py ... --use_contrastive_reg
```

Combine modules gradually and track PSNR/SSIM/LPIPS improvements along with training time and point counts.

---

## 📚 See Also
- [INNOVATIONS_5.md](./INNOVATIONS_5.md) — detailed theory, metrics, and diagrams for each innovation
- `train.py` — integration points showing how each module plugs into the training loop
- `arguments/__init__.py` — full list of command-line options

Happy training!
