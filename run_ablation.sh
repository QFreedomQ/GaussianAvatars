#!/bin/bash
# Ablation Study Script for Five Innovations
# Usage: ./run_ablation.sh <subject_id> <data_dir>

set -e

SUBJECT=${1:-306}
DATA_DIR=${2:-"data/${SUBJECT}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"}
OUTPUT_BASE="output/ablation_${SUBJECT}"
PORT=60000
INTERVAL=60000
COMMON_FLAGS="--eval --bind_to_mesh --white_background --port ${PORT} --interval ${INTERVAL}"

echo "=========================================="
echo "Ablation Study for Subject ${SUBJECT}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_BASE}"
echo "=========================================="

# Baseline
echo "Running Baseline..."
python train.py -s ${DATA_DIR} -m ${OUTPUT_BASE}/baseline ${COMMON_FLAGS}

# Innovation 1: Region-Adaptive Loss
echo "Running Innovation 1 (Region-Adaptive Loss)..."
python train.py -s ${DATA_DIR} -m ${OUTPUT_BASE}/innov1_region_adaptive \
  ${COMMON_FLAGS} \
  --use_region_adaptive_loss \
  --region_weight_eyes 2.0 \
  --region_weight_mouth 2.0

# Innovation 2: Smart Densification
echo "Running Innovation 2 (Smart Densification)..."
python train.py -s ${DATA_DIR} -m ${OUTPUT_BASE}/innov2_smart_densify \
  ${COMMON_FLAGS} \
  --use_smart_densification \
  --densify_percentile_clone 75.0 \
  --densify_percentile_split 90.0

# Innovation 3: Progressive Resolution
echo "Running Innovation 3 (Progressive Resolution)..."
python train.py -s ${DATA_DIR} -m ${OUTPUT_BASE}/innov3_progressive \
  ${COMMON_FLAGS} \
  --use_progressive_resolution \
  --resolution_schedule "0.5,0.75,1.0" \
  --resolution_milestones "100000,300000"

# Innovation 4: Color Calibration
echo "Running Innovation 4 (Color Calibration)..."
python train.py -s ${DATA_DIR} -m ${OUTPUT_BASE}/innov4_color_calib \
  ${COMMON_FLAGS} \
  --use_color_calibration \
  --color_net_hidden_dim 16 \
  --lambda_color_reg 1e-4

# Innovation 5: Contrastive Regularization
echo "Running Innovation 5 (Contrastive Regularization)..."
python train.py -s ${DATA_DIR} -m ${OUTPUT_BASE}/innov5_contrastive \
  ${COMMON_FLAGS} \
  --use_contrastive_reg \
  --lambda_contrastive 0.01

# Combined: 1+2
echo "Running Combined 1+2..."
python train.py -s ${DATA_DIR} -m ${OUTPUT_BASE}/combo_1_2 \
  ${COMMON_FLAGS} \
  --use_region_adaptive_loss \
  --use_smart_densification

# Combined: All
echo "Running All Innovations..."
python train.py -s ${DATA_DIR} -m ${OUTPUT_BASE}/all_innovations \
  ${COMMON_FLAGS} \
  --use_region_adaptive_loss \
  --use_smart_densification \
  --use_progressive_resolution \
  --use_color_calibration \
  --use_contrastive_reg \
  --use_amp

echo "=========================================="
echo "Ablation study completed!"
echo "Running evaluation..."
echo "=========================================="

# Evaluate all experiments
for exp in baseline innov1_region_adaptive innov2_smart_densify innov3_progressive \
           innov4_color_calib innov5_contrastive combo_1_2 all_innovations; do
    echo "Evaluating ${exp}..."
    python metrics.py -m ${OUTPUT_BASE}/${exp}
done

echo "All experiments completed!"
