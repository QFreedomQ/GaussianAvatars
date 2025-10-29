#!/bin/bash

# =============================================================================
# Corrected Training Commands for GaussianAvatars Innovation Experiments
# =============================================================================
# 
# IMPORTANT: 
# - All innovations are now OFF by default
# - Each command explicitly enables only the intended innovation(s)
# - This ensures fair comparison between baseline and innovation experiments
#
# Performance optimization flags are included for faster training.
# Adjust --dataloader_workers based on your CPU core count.
#
# =============================================================================

SUBJECT=306
DATA_DIR="data/${SUBJECT}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
PORT=60000

# Performance settings (adjust based on your hardware)
WORKERS=16        # Set to: min(CPU_cores - 4, 16)
PREFETCH=3        # Prefetch factor

# Common arguments
COMMON="--eval --bind_to_mesh --white_background --port ${PORT}"
PERF="--dataloader_workers ${WORKERS} --prefetch_factor ${PREFETCH}"

echo "========================================"
echo "GaussianAvatars Training Experiments"
echo "Subject: ${SUBJECT}"
echo "========================================"
echo ""

# =============================================================================
# 1. Baseline (基线实验)
# =============================================================================
# Innovations: NONE
# Purpose: Baseline for comparison
# Expected behavior:
#   - No perceptual loss (lambda=0)
#   - No adaptive densification
#   - No temporal consistency
# Verification:
#   - Progress bar: only shows l1, ssim, xyz, scale
#   - No "percep" or "temp" in progress bar
#   - No [Innovation] logs
# =============================================================================

echo "[1/5] Training Baseline (No Innovations)..."
echo "Innovations: NONE"
echo "----------------------------------------"

python train.py \
  -s ${DATA_DIR} \
  -m output/baseline_${SUBJECT} \
  ${COMMON} ${PERF} \
  --lambda_perceptual 0

echo ""
echo ""

# =============================================================================
# 2. Full Innovations (全部创新)
# =============================================================================
# Innovations: ALL (Perceptual + Adaptive + Temporal)
# Purpose: Best quality with all improvements
# Expected behavior:
#   - VGG perceptual loss active
#   - Adaptive densification for facial regions
#   - Temporal smoothness regularization
# Verification:
#   - Progress bar shows: percep: 0.xxx, temp: 0.xxx
#   - Log: "[Innovation 2] Enabled adaptive densification with ratio 1.5"
#   - TensorBoard: perceptual_loss and temporal_loss curves
# =============================================================================

echo "[2/5] Training with All Innovations..."
echo "Innovations: Perceptual + Adaptive + Temporal"
echo "----------------------------------------"

python train.py \
  -s ${DATA_DIR} \
  -m output/full_${SUBJECT} \
  ${COMMON} ${PERF} \
  --lambda_perceptual 0.05 \
  --use_vgg_loss \
  --use_adaptive_densification \
  --adaptive_densify_ratio 1.5 \
  --use_temporal_consistency \
  --lambda_temporal 0.01

echo ""
echo ""

# =============================================================================
# 3. Perceptual Loss Only (仅感知损失)
# =============================================================================
# Innovations: Perceptual Loss ONLY
# Purpose: Evaluate perceptual loss contribution
# Expected behavior:
#   - VGG perceptual loss active
#   - Standard densification (not adaptive)
#   - No temporal regularization
# Verification:
#   - Progress bar shows: percep: 0.xxx
#   - No "temp" in progress bar
#   - No adaptive densification logs
# =============================================================================

echo "[3/5] Training with Perceptual Loss Only..."
echo "Innovations: Perceptual Loss"
echo "----------------------------------------"

python train.py \
  -s ${DATA_DIR} \
  -m output/perceptual_only_${SUBJECT} \
  ${COMMON} ${PERF} \
  --lambda_perceptual 0.05 \
  --use_vgg_loss

echo ""
echo ""

# =============================================================================
# 4. Adaptive Densification Only (仅自适应密集化)
# =============================================================================
# Innovations: Adaptive Densification ONLY
# Purpose: Evaluate adaptive densification contribution
# Expected behavior:
#   - No perceptual loss
#   - Adaptive densification for facial regions
#   - No temporal regularization
# Verification:
#   - Log: "[Innovation 2] Enabled adaptive densification"
#   - Log: "[Adaptive Densification] High-importance faces: N"
#   - No "percep" or "temp" in progress bar
# =============================================================================

echo "[4/5] Training with Adaptive Densification Only..."
echo "Innovations: Adaptive Densification"
echo "----------------------------------------"

python train.py \
  -s ${DATA_DIR} \
  -m output/adaptive_only_${SUBJECT} \
  ${COMMON} ${PERF} \
  --lambda_perceptual 0 \
  --use_adaptive_densification \
  --adaptive_densify_ratio 1.5

echo ""
echo ""

# =============================================================================
# 5. Temporal Consistency Only (仅时序一致性)
# =============================================================================
# Innovations: Temporal Consistency ONLY
# Purpose: Evaluate temporal consistency contribution
# Expected behavior:
#   - No perceptual loss
#   - Standard densification (not adaptive)
#   - Temporal smoothness regularization active
# Verification:
#   - Progress bar shows: temp: 0.xxx
#   - No "percep" in progress bar
#   - No adaptive densification logs
# =============================================================================

echo "[5/5] Training with Temporal Consistency Only..."
echo "Innovations: Temporal Consistency"
echo "----------------------------------------"

python train.py \
  -s ${DATA_DIR} \
  -m output/temporal_only_${SUBJECT} \
  ${COMMON} ${PERF} \
  --lambda_perceptual 0 \
  --use_temporal_consistency \
  --lambda_temporal 0.01

echo ""
echo ""

# =============================================================================
# Training Complete
# =============================================================================

echo "========================================"
echo "All 5 experiments completed!"
echo "========================================"
echo ""
echo "Output directories:"
echo "  - output/baseline_${SUBJECT}/"
echo "  - output/full_${SUBJECT}/"
echo "  - output/perceptual_only_${SUBJECT}/"
echo "  - output/adaptive_only_${SUBJECT}/"
echo "  - output/temporal_only_${SUBJECT}/"
echo ""
echo "Next steps:"
echo "  1. Check TensorBoard: tensorboard --logdir output/"
echo "  2. Compare metrics: PSNR, SSIM, LPIPS"
echo "  3. Render videos for visual comparison"
echo "  4. See TRAINING_GUIDE.md for detailed analysis"
echo ""
echo "For performance monitoring:"
echo "  - GPU: watch -n 1 nvidia-smi"
echo "  - CPU: htop"
echo "  - Expect: GPU >85%, training speed 5-8 iter/s"
echo ""

# =============================================================================
# Quick Verification Test (Optional)
# =============================================================================
# 
# Run this to quickly test all innovations with short training (1000 iters):
#
# python train.py \
#   -s ${DATA_DIR} \
#   -m output/test_quick_${SUBJECT} \
#   ${COMMON} \
#   --iterations 1000 \
#   --interval 500 \
#   --lambda_perceptual 0.05 --use_vgg_loss \
#   --use_adaptive_densification --adaptive_densify_ratio 1.5 \
#   --use_temporal_consistency --lambda_temporal 0.01 \
#   --dataloader_workers 8
#
# Check logs for all innovation activation messages within 2-3 minutes.
# =============================================================================
