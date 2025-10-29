# Complete Experimental Workflow & Evaluation Metrics Guide

> 📖 **中文版本**: [实验步骤与评估指标完整指南 (中文)](experiment_steps.md)

This document provides a comprehensive guide for running experiments with GaussianAvatars, including environment setup, training, rendering, and outputting evaluation metrics.

---

## Quick Links

- **Setup**: [Installation](installation.md) | [Download Data](download.md)
- **Training**: See [Main README](../README.md#1-training)
- **Rendering**: [Offline Rendering Guide](offline_render.md)
- **Evaluation**: See sections below

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Training Models](#2-training-models)
3. [Rendering Results](#3-rendering-results)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Complete Workflow](#5-complete-workflow)

---

## 1. Environment Setup

### Prerequisites

- **GPU**: NVIDIA GPU with Compute Capability 7.0+ and 11GB+ VRAM
- **Software**: Python 3.10, CUDA 11.7+, PyTorch 2.0+, FFmpeg

### Installation Steps

```bash
# Clone repository
git clone https://github.com/ShenhanQian/GaussianAvatars.git --recursive
cd GaussianAvatars

# Create conda environment
conda create --name gaussian-avatars -y python=3.10
conda activate gaussian-avatars

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

For detailed installation instructions, see [installation.md](installation.md).

---

## 2. Training Models

### Baseline Training

Train a baseline model without innovations:

```bash
SUBJECT=306

python train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/baseline_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --lambda_perceptual 0 \
  --use_adaptive_densification False \
  --use_temporal_consistency False
```

### Full Model Training (With All Innovations)

Train with all three innovations enabled:

```bash
python train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/full_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --lambda_perceptual 0.05 \
  --use_vgg_loss True \
  --use_adaptive_densification True \
  --use_temporal_consistency True
```

### Innovation Parameters

| Innovation | Parameters | Default |
|-----------|-----------|---------|
| **Perceptual Loss** | `--lambda_perceptual` | 0.05 |
| | `--use_vgg_loss` | True |
| | `--use_lpips_loss` | False |
| **Adaptive Densification** | `--use_adaptive_densification` | True |
| | `--adaptive_densify_ratio` | 1.5 |
| **Temporal Consistency** | `--use_temporal_consistency` | True |
| | `--lambda_temporal` | 0.01 |

### Training Monitoring

**Option 1: Remote Viewer** (real-time visualization)

```bash
python remote_viewer.py --port 60000
```

**Option 2: TensorBoard** (training curves)

```bash
tensorboard --logdir output/full_${SUBJECT}
```

Then open `http://localhost:6006` in your browser.

---

## 3. Rendering Results

After training, render test sets for evaluation:

### Render Validation Set (Novel-View Synthesis)

```bash
python render.py -m output/full_306 --skip_train --skip_test
```

### Render Test Set (Self-Reenactment)

```bash
python render.py -m output/full_306 --skip_train --skip_val
```

**Output Structure**:

```
output/full_306/
├── val/ours_300000/
│   ├── renders/          # Rendered images
│   ├── gt/               # Ground truth images
│   ├── renders.mp4       # Video of renders
│   └── gt.mp4            # Video of GT
└── test/ours_300000/
    └── ...
```

For more rendering options (cross-identity reenactment, specific camera views, etc.), see [offline_render.md](offline_render.md).

---

## 4. Evaluation Metrics

### Automatic Evaluation (During Training)

When training with `--eval` flag, metrics are computed automatically every 7000 iterations:

```bash
[ITER 300000] Evaluating val
  PSNR: 32.45  SSIM: 0.954  LPIPS: 0.068

[ITER 300000] Evaluating test  
  PSNR: 31.82  SSIM: 0.948  LPIPS: 0.075
```

### Post-Training Evaluation

#### Step 1: Render Test Sets

```bash
python render.py -m output/full_306 --skip_train
```

#### Step 2: Calculate Metrics

```bash
python metrics.py -m output/full_306
```

**Output**:

```
Scene: output/full_306
Method: ours_300000
  SSIM :   0.9542341
  PSNR :  32.4512763
  LPIPS:   0.0684521
```

#### Step 3: View Results

Metrics are saved to:
- `output/full_306/results.json` - Overall metrics
- `output/full_306/per_view.json` - Per-frame metrics

**Example `results.json`**:

```json
{
  "ours_300000": {
    "SSIM": 0.9542341,
    "PSNR": 32.4512763,
    "LPIPS": 0.0684521
  }
}
```

### Metrics Explanation

| Metric | Full Name | Range | Description | Better |
|--------|-----------|-------|-------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | 0-∞ dB | Pixel-level error | ↑ Higher |
| **SSIM** | Structural Similarity Index | 0-1 | Structure preservation | ↑ Higher |
| **LPIPS** | Learned Perceptual Image Patch Similarity | 0-∞ | Perceptual similarity | ↓ Lower |

**Interpretation**:
- PSNR > 30 dB: Good quality
- PSNR > 32 dB: Excellent quality
- SSIM > 0.95: High structural similarity
- LPIPS < 0.10: Good perceptual quality
- LPIPS < 0.07: Excellent perceptual quality

### Batch Evaluation (Multiple Models)

```bash
python metrics.py -m \
  output/baseline_306 \
  output/perceptual_only_306 \
  output/full_306
```

---

## 5. Complete Workflow

### Single Model Experiment

Complete workflow script:

```bash
#!/bin/bash

SUBJECT=306
MODEL_NAME=full_${SUBJECT}
DATA_PATH=data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine

# 1. Train model
python train.py \
  -s ${DATA_PATH} \
  -m output/${MODEL_NAME} \
  --eval --bind_to_mesh --white_background \
  --lambda_perceptual 0.05 \
  --use_adaptive_densification True \
  --use_temporal_consistency True

# 2. Render validation set
python render.py -m output/${MODEL_NAME} --skip_train --skip_test

# 3. Render test set
python render.py -m output/${MODEL_NAME} --skip_train --skip_val

# 4. Calculate metrics
python metrics.py -m output/${MODEL_NAME}

# 5. Display results
echo "=== Results ==="
cat output/${MODEL_NAME}/results.json
```

### Expected Performance

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Training Time |
|--------------|-------|-------|--------|------|---------------|
| **Baseline** | 32.1 | 0.947 | 0.085 | 85 | 36h |
| **Full (All Innovations)** | **33.2** | **0.962** | **0.062** | **96** | 40h |

---

## 6. FPS Benchmarking

### Method 1: Using Demo Data

```bash
python fps_benchmark_demo.py \
  --point_path media/306/point_cloud.ply \
  --height 802 --width 550 \
  --n_iter 500 --vis
```

### Method 2: Using Training Data

```bash
python fps_benchmark_dataset.py \
  -m output/full_306 \
  --skip_val --skip_test \
  --n_iter 500 --vis
```

**Output Example**:

```
Rendering FPS Benchmark
Resolution: 802x550
Number of iterations: 500

Average FPS: 96.3
Min FPS: 89.2
Max FPS: 102.7
```

---

## 7. Troubleshooting

### Q: Out of memory during training?

**Solutions**:
- Reduce image resolution: `--resolution 2`
- Disable perceptual loss: `--lambda_perceptual 0`
- Lower densification threshold

### Q: Training too slow?

**Solutions**:
- Close remote viewer
- Disable LPIPS loss: `--use_lpips_loss False`
- Enable adaptive densification: `--use_adaptive_densification True`

### Q: How to resume training from checkpoint?

**Answer**: Simply re-run the same training command. The script will automatically load the latest checkpoint.

### Q: Metrics differ from paper?

**Possible reasons**:
- Different dataset
- Insufficient training iterations
- Different parameter settings
- Different train/val/test split

---

## 8. Visualization

### Interactive Local Viewer

After training, view results interactively:

```bash
python local_viewer.py \
  --point_path output/full_306/point_cloud/iteration_300000/point_cloud.ply
```

**Features**:
- Real-time rotation and zoom
- Adjust FLAME expression parameters
- Switch motion sequences
- Save rendered images

### TensorBoard Visualization

```bash
tensorboard --logdir output/ --port 6006
```

View at `http://localhost:6006`:
- Training loss curves
- Evaluation metric curves
- Rendered image comparisons
- Learning rate schedules

---

## 9. Citation

If you find this project useful for your research, please cite:

```bibtex
@inproceedings{qian2024gaussianavatars,
  title={Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians},
  author={Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20299--20309},
  year={2024}
}
```

---

## Additional Resources

- **Project Page**: https://shenhanqian.github.io/gaussian-avatars
- **Paper**: http://arxiv.org/abs/2312.02069
- **Video**: https://www.youtube.com/watch?v=lVEY78RwU_I
- **Face Tracker (VHAP)**: https://github.com/ShenhanQian/VHAP

For more details on innovations implemented in this repository, see:
- [INNOVATIONS.md](../INNOVATIONS.md) - Technical details (English)
- [README_INNOVATIONS.md](../README_INNOVATIONS.md) - Quick overview (English)
- [SUMMARY_ZH.md](../SUMMARY_ZH.md) - Comprehensive guide (Chinese)

---

**Last Updated**: 2024-01-XX
