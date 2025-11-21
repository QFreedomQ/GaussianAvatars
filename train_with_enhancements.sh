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
echo "[1/7] Training Baseline..."
python train.py \
  -s ${DATA_DIR} \
  -m output/baseline_${SUBJECT} \
  ${COMMON_ARGS} \
  --lambda_perceptual 0.0

echo ""
echo "Baseline training completed. Results saved to: output/baseline_${SUBJECT}"
echo ""

# ========================================
# 2. Innovation 1: 感知损失增强
# ========================================
echo "[2/7] Training with Perceptual Loss Enhancement..."
python train.py \
  -s ${DATA_DIR} \
  -m output/innovation1_perceptual_${SUBJECT} \
  ${COMMON_ARGS} \
  --lambda_perceptual 0.05 \
  --use_vgg_loss

echo ""
echo "Perceptual Loss training completed. Results saved to: output/innovation1_perceptual_${SUBJECT}"
echo ""

# ========================================
# 3. Innovation 2: 表达式自适应着色
# ========================================
echo "[3/7] Training with Expression-Adaptive Appearance..."
python train.py \
  -s ${DATA_DIR} \
  -m output/innovation2_expr_color_${SUBJECT} \
  ${COMMON_ARGS} \
  --use_expr_adaptive_color \
  --lambda_expr_color 0.01 \
  --expr_color_lr 1e-4

echo ""
echo "Expression-Adaptive training completed. Results saved to: output/innovation2_expr_color_${SUBJECT}"
echo ""

# ========================================
# 4. Innovation 3: 法线约束与曲率正则
# ========================================
echo "[4/7] Training with Normal Regularization..."
python train.py \
  -s ${DATA_DIR} \
  -m output/innovation3_normal_reg_${SUBJECT} \
  ${COMMON_ARGS} \
  --use_normal_regularization \
  --lambda_normal_align 0.01 \
  --lambda_laplacian_smooth 0.001 \
  --lambda_normal_consistency 0.005

echo ""
echo "Normal Regularization training completed. Results saved to: output/innovation3_normal_reg_${SUBJECT}"
echo ""

# ========================================
# 5. Innovation 4: 动态密度过滤
# ========================================
echo "[5/7] Training with Adaptive Densification..."
python train.py \
  -s ${DATA_DIR} \
  -m output/innovation4_adaptive_densify_${SUBJECT} \
  ${COMMON_ARGS} \
  --use_adaptive_densification \
  --adaptive_densify_grad_threshold_min 0.0001 \
  --adaptive_densify_grad_threshold_max 0.0005 \
  --adaptive_coverage_factor 0.5

echo ""
echo "Adaptive Densification training completed. Results saved to: output/innovation4_adaptive_densify_${SUBJECT}"
echo ""

# ========================================
# 6. Innovation 5: 面部区域注意力
# ========================================
echo "[6/7] Training with Facial ROI Attention..."
python train.py \
  -s ${DATA_DIR} \
  -m output/innovation5_roi_attention_${SUBJECT} \
  ${COMMON_ARGS} \
  --use_facial_roi_attention \
  --lambda_roi 0.05

echo ""
echo "Facial ROI Attention training completed. Results saved to: output/innovation5_roi_attention_${SUBJECT}"
echo ""

# ========================================
# 7. 最佳组合：感知损失 + 法线正则 + 动态密度
# ========================================
echo "[7/7] Training with Best Combination (Perceptual + Normal + Adaptive)..."
python train.py \
  -s ${DATA_DIR} \
  -m output/best_combination_${SUBJECT} \
  ${COMMON_ARGS} \
  --lambda_perceptual 0.05 \
  --use_vgg_loss \
  --use_normal_regularization \
  --lambda_normal_align 0.01 \
  --lambda_laplacian_smooth 0.001 \
  --use_adaptive_densification \
  --adaptive_coverage_factor 0.5

echo ""
echo "Best Combination training completed. Results saved to: output/best_combination_${SUBJECT}"
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
  "innovation1_perceptual_${SUBJECT}"
  "innovation2_expr_color_${SUBJECT}"
  "innovation3_normal_reg_${SUBJECT}"
  "innovation4_adaptive_densify_${SUBJECT}"
  "innovation5_roi_attention_${SUBJECT}"
  "best_combination_${SUBJECT}"
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
echo "  - Innovation 1 (Perceptual): output/innovation1_perceptual_${SUBJECT}"
echo "  - Innovation 2 (ExprColor): output/innovation2_expr_color_${SUBJECT}"
echo "  - Innovation 3 (NormalReg): output/innovation3_normal_reg_${SUBJECT}"
echo "  - Innovation 4 (AdaptiveDensify): output/innovation4_adaptive_densify_${SUBJECT}"
echo "  - Innovation 5 (ROI Attention): output/innovation5_roi_attention_${SUBJECT}"
echo "  - Best Combination: output/best_combination_${SUBJECT}"
echo ""
echo "To compare results, check the following files in each model directory:"
echo "  - val_results.json: Validation metrics (PSNR, SSIM, LPIPS)"
echo "  - test_results.json: Test metrics"
echo "  - val/ours_600000/renders.mp4: Rendered video"
echo ""
