#!/bin/bash

# GaussianAvatars Enhancement Training Script
# 本脚本展示如何使用各个增强模块进行训练

# 设置环境变量
export SUBJECT=306
export DATA_DIR="data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"

# 通用训练参数
COMMON_ARGS="--bind_to_mesh --white_background --eval --iterations 600000 --interval 60000"

echo "========================================"
echo "GaussianAvatars Enhancement Training"
echo "========================================"
echo "Subject: ${SUBJECT}"
echo "Data Directory: ${DATA_DIR}"
echo ""

# ========================================
# 1. Baseline (原始方法)
# ========================================
echo "[1/2] Training Baseline..."
python train.py \
  -s ${DATA_DIR} \
  -m output/baseline_${SUBJECT} \
  ${COMMON_ARGS} \
  --lambda_perceptual 0.0

echo ""
echo "Baseline training completed. Results saved to: output/baseline_${SUBJECT}"
echo ""

# ========================================
# 2. Perceptual Loss Enhancement (感知损失增强)
# ========================================
echo "[2/2] Training with Perceptual Loss Enhancement..."
python train.py \
  -s ${DATA_DIR} \
  -m output/perceptual_${SUBJECT} \
  ${COMMON_ARGS} \
  --lambda_perceptual 0.05 \
  --use_vgg_loss

echo ""
echo "Perceptual Loss training completed. Results saved to: output/perceptual_${SUBJECT}"
echo ""

# ========================================
# 评估所有模型
# ========================================
echo "========================================"
echo "Evaluation Phase"
echo "========================================"

# 渲染并评估每个模型
MODELS=(
  "baseline_${SUBJECT}"
  "perceptual_${SUBJECT}"
)

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "Evaluating: ${MODEL}"
  
  # 渲染
  python render.py -m output/${MODEL}
  
  # 计算指标
  python metrics.py -m output/${MODEL}
  
  echo "${MODEL} evaluation completed."
done

echo ""
echo "========================================"
echo "All Training and Evaluation Completed!"
echo "========================================"
echo ""
echo "Results saved in output/ directory:"
echo "  - Baseline: output/baseline_${SUBJECT}"
echo "  - Perceptual Loss: output/perceptual_${SUBJECT}"
echo ""
echo "To compare results, check the following files in each model directory:"
echo "  - val_results.json: Validation metrics (PSNR, SSIM, LPIPS)"
echo "  - test_results.json: Test metrics"
echo "  - val/ours_600000/renders.mp4: Rendered video"
echo ""
