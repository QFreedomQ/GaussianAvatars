# GaussianAvatars 完整实验流程指南

本文档提供从环境配置到最终评估的完整实验流程，涵盖 5 个高效创新点的使用说明。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [训练配置选择](#3-训练配置选择)
4. [训练执行](#4-训练执行)
5. [结果评估](#5-结果评估)
6. [常见问题处理](#6-常见问题处理)
7. [性能优化建议](#7-性能优化建议)

---

## 1. 环境准备

### 1.1 系统要求

- **操作系统**：Linux (Ubuntu 18.04/20.04/22.04 推荐)
- **GPU**：NVIDIA GPU，CUDA >= 11.3，建议显存 >= 24GB
- **Python**：3.8/3.9/3.10
- **磁盘空间**：数据集 + 输出约需 50-100GB

### 1.2 安装依赖

详见 [doc/installation.md](./installation.md)。关键步骤：

```bash
# 1. 创建虚拟环境
conda create -n gaussian-avatars python=3.10
conda activate gaussian-avatars

# 2. 安装 PyTorch (根据 CUDA 版本调整)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装自定义 CUDA 扩展
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# 4. 安装其它依赖
pip install -r requirements.txt
```

### 1.3 验证安装

```bash
python -c "
from innovations import (
    RegionAdaptiveLoss,
    SmartDensificationMixin,
    ProgressiveResolutionScheduler,
    ColorCalibrationNetwork,
    ContrastiveRegularization
)
print('✅ 所有创新模块加载成功')
"
```

---

## 2. 数据准备

### 2.1 下载数据集

详见 [doc/download.md](./download.md)。以 UNION10 数据集为例：

```bash
cd data
# 下载并解压到 data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/
```

### 2.2 数据集结构

确保数据集包含以下内容：

```
data/UNION10_${SUBJECT}/
├── images/                  # 训练图像
├── canonical_flame_param.npz  # FLAME 参数
├── flame_param/              # 动态 FLAME 参数
├── transforms_train.json     # 相机参数（如使用 Blender）
└── ...
```

### 2.3 数据验证

```bash
# 检查图像数量
ls data/UNION10_306_*/images/*.png | wc -l
# 应输出足够的图像数量（例如 500-2000）
```

---

## 3. 训练配置选择

根据时间预算和质量需求选择合适的配置：

### 3.1 配置对比表

| 配置名称 | 训练时长 | PSNR 提升 | 高斯点数 | 适用场景 |
|----------|----------|-----------|----------|----------|
| **Baseline** | ~6 小时 | 基线 | ~92k | 快速原型/对比实验 |
| **Ultra-Efficient** | ~5.25 小时 | +0.6 dB | ~105k | 追求速度 |
| **Balanced** ⭐ | ~5.5 小时 | +1.0 dB | ~115k | **推荐：质量/速度平衡** |
| **Quality-First** | ~6.6 小时 | +1.2 dB | ~120k | 追求最高质量 |

### 3.2 Baseline 配置

不使用任何创新点，作为对比基线：

```bash
python train.py \
  -s data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/baseline_306 \
  --eval --bind_to_mesh --white_background
```

### 3.3 Ultra-Efficient 配置

仅启用关键创新点，最小开销：

```bash
python train.py \
  -s data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/ultra_306 \
  --eval --bind_to_mesh --white_background \
  --use_region_adaptive_loss \
  --region_weight_eyes 2.0 \
  --region_weight_mouth 2.0 \
  --use_smart_densification \
  --densify_percentile_clone 75 \
  --densify_percentile_split 90 \
  --use_amp
```

### 3.4 Balanced 配置 ⭐（推荐）

启用四个创新点，平衡质量与速度：

```bash
python train.py \
  -s data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/balanced_306 \
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

### 3.5 Quality-First 配置

启用全部五个创新点，追求最高质量：

```bash
python train.py \
  -s data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/quality_306 \
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
  --use_contrastive_reg \
  --lambda_contrastive 0.01 \
  --contrastive_cache_size 2 \
  --use_amp
```

---

## 4. 训练执行

### 4.1 启动训练

选择合适的配置后，执行训练命令。推荐使用 `nohup` 或 `screen` 保证长时间运行：

```bash
# 方式 1：使用 nohup
nohup python train.py \
  -s data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/balanced_306 \
  --eval --bind_to_mesh --white_background \
  --use_region_adaptive_loss \
  --use_smart_densification \
  --use_progressive_resolution \
  --use_color_calibration \
  --use_amp \
  > train.log 2>&1 &

# 方式 2：使用 screen
screen -S gaussian_train
python train.py -s ... -m ...
# Ctrl+A, D 挂后台
```

### 4.2 监控训练进度

#### 4.2.1 实时日志

```bash
tail -f train.log
```

关键输出：

```
[Innovation] Progressive resolution schedule 0.5,0.75,1.0 with milestones [100000, 300000]
[Innovation] Region-adaptive loss enabled
[Innovation] Smart densification enabled (clone=75.0%, split=90.0%)
[Innovation] Color calibration network enabled
[AMP] Automatic Mixed Precision enabled
Loading Training Cameras
Loading Training Cameras
Loading Training Cameras
Training progress: 100%|████████| 600000/600000 [05:30:00, 30.15it/s]
```

#### 4.2.2 TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir output/balanced_306 --port 6006

# 在浏览器打开 http://localhost:6006
```

监控指标：
- **train_loss_patches/total_loss**：总损失曲线
- **train_loss_patches/l1_loss**：L1 损失
- **train_loss_patches/ssim_loss**：SSIM 损失
- **iter_time**：每迭代耗时（ms）
- **total_points**：高斯点数变化

#### 4.2.3 关键 Milestone

观察以下关键节点：

- **[ITER 100000]**: 切换到 0.75 分辨率（如启用渐进式训练）
- **[ITER 300000]**: 切换到 1.0 分辨率
- **[ITER 60000, 120000, ...]**: 评估测试集指标

### 4.3 中断与恢复

训练会自动保存 checkpoint：

```bash
output/balanced_306/
└── chkpnt60000.pth
└── chkpnt120000.pth
└── ...
```

恢复训练：

```bash
python train.py \
  -s data/... \
  -m output/balanced_306 \
  --start_checkpoint output/balanced_306/chkpnt300000.pth \
  --eval --bind_to_mesh --white_background \
  ... (其它参数保持一致)
```

---

## 5. 结果评估

### 5.1 查看训练指标

训练完成后，TensorBoard 中可看到：

```
[ITER 600000] Evaluating test: L1 0.0123 PSNR 33.2 SSIM 0.962 LPIPS 0.062
```

### 5.2 渲染评估视频

```bash
python render.py \
  -m output/balanced_306 \
  --iteration 600000 \
  --skip_train \
  --skip_val
```

输出：`output/balanced_306/test/ours_600000/renders/*.png`

### 5.3 离线评估完整指标

```bash
# 计算 PSNR/SSIM/LPIPS
python metrics.py \
  -m output/balanced_306
```

### 5.4 可视化对比

```bash
# 启动本地查看器
python local_viewer.py \
  --point_path output/balanced_306/point_cloud/iteration_600000/point_cloud.ply
```

---

## 6. 常见问题处理

### 6.1 CUDA Out of Memory

**症状**：训练中途报 `CUDA out of memory` 错误。

**解决方案**：

1. **启用 AMP**（如未启用）：
   ```bash
   --use_amp
   ```

2. **调整密集化参数**，减少高斯点数：
   ```bash
   --densify_percentile_split 95  # 从 90 提高到 95
   ```

3. **降低批量大小**（通过减少 num_workers）：
   ```python
   # 修改 train.py 中的 DataLoader
   num_workers=4  # 从 8 改为 4
   ```

### 6.2 训练速度过慢

**症状**：`iter_time` 超过 100ms，预计耗时超过 10 小时。

**诊断**：

1. **检查是否启用 AMP**：
   ```bash
   # 日志中应看到
   [AMP] Automatic Mixed Precision enabled
   ```

2. **检查渐进式分辨率**是否生效：
   ```bash
   # 应看到
   [ITER 100000] Switching to resolution scale 0.75
   ```

3. **检查 GPU 利用率**：
   ```bash
   nvidia-smi
   # GPU 利用率应接近 100%
   ```

**解决方案**：

- 确保使用本次修复后的 `train.py`（已移除 `_downsample_if_needed`）
- 启用 `--use_amp`
- 减少不必要的评估频率：
  ```bash
  --interval 120000  # 从 60000 增加到 120000
  ```

### 6.3 点数爆炸

**症状**：高斯点数超过 200k，训练变慢，显存不足。

**解决方案**：

调整密集化百分位数：

```bash
--densify_percentile_clone 80  # 从 75 提高
--densify_percentile_split 95   # 从 90 提高
```

### 6.4 质量不达预期

**症状**：PSNR 提升小于 +0.5 dB。

**诊断**：

1. 检查创新点是否启用：
   ```bash
   # 日志中应看到
   [Innovation] Region-adaptive loss enabled
   [Innovation] Smart densification enabled
   ...
   ```

2. 检查数据集质量（图像分辨率、FLAME 标注等）

**解决方案**：

- 尝试更高质量的配置（Quality-First）
- 增加训练迭代数：
  ```bash
  python arguments/__init__.py  # 修改 iterations = 800_000
  ```

---

## 7. 性能优化建议

### 7.1 单卡训练优化

- **必选**：`--use_amp`（30-40% 加速）
- **推荐**：合理的密集化参数（避免点数爆炸）
- **可选**：减少评估频率（`--interval 120000`）

### 7.2 多卡训练（实验性）

当前代码不支持原生多卡训练，但可并行训练多个主体：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py -s data/UNION10_306_... -m output/306 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python train.py -s data/UNION10_307_... -m output/307 &
```

### 7.3 存储优化

- 定期清理中间 checkpoint（仅保留最终模型）：
  ```bash
  rm output/balanced_306/chkpnt*.pth
  ```

- 压缩渲染输出：
  ```bash
  ffmpeg -framerate 30 -i output/balanced_306/test/ours_600000/renders/%05d.png \
    -c:v libx264 -pix_fmt yuv420p output_video.mp4
  ```

---

## 8. 完整实验检查清单

- [ ] 环境安装完成，所有依赖可正常导入
- [ ] 数据集下载并验证结构正确
- [ ] 选择合适的训练配置（推荐 Balanced）
- [ ] 启动训练，确认看到创新点启用日志
- [ ] TensorBoard 启动并监控损失曲线
- [ ] 训练过程中监控 GPU 利用率
- [ ] 观察分辨率切换日志（如启用渐进式训练）
- [ ] 训练完成后查看最终指标（PSNR/SSIM/LPIPS）
- [ ] 渲染评估集并可视化结果
- [ ] 保存最终模型和实验日志

---

## 9. 进阶实验

### 9.1 消融实验

逐个启用创新点，观察各自贡献：

```bash
# Baseline
python train.py ... -m output/baseline

# +Region Loss
python train.py ... -m output/ablation_1 --use_region_adaptive_loss

# +Smart Densification
python train.py ... -m output/ablation_2 --use_region_adaptive_loss --use_smart_densification

# +Progressive Resolution
python train.py ... -m output/ablation_3 --use_region_adaptive_loss --use_smart_densification --use_progressive_resolution

# +Color Calibration
python train.py ... -m output/ablation_4 --use_region_adaptive_loss --use_smart_densification --use_progressive_resolution --use_color_calibration

# +Contrastive Regularization
python train.py ... -m output/ablation_5 --use_region_adaptive_loss --use_smart_densification --use_progressive_resolution --use_color_calibration --use_contrastive_reg
```

### 9.2 参数敏感性分析

测试不同超参数组合：

```bash
# 测试不同区域权重
for weight in 1.5 2.0 2.5 3.0; do
  python train.py ... \
    --region_weight_eyes $weight \
    --region_weight_mouth $weight \
    -m output/region_weight_${weight}
done

# 测试不同密集化阈值
for percentile in 70 75 80 85; do
  python train.py ... \
    --densify_percentile_clone $percentile \
    -m output/densify_clone_${percentile}
done
```

---

## 10. 参考文档

- [QUICK_START.md](../QUICK_START.md) - 快速开始
- [INNOVATIONS_5.md](../INNOVATIONS_5.md) - 五个创新点详细说明
- [EFFICIENT_INNOVATIONS_README.md](../EFFICIENT_INNOVATIONS_README.md) - 高效创新点用法
- [training_time_regression_analysis.md](./training_time_regression_analysis.md) - 训练时长问题分析
- [installation.md](./installation.md) - 环境安装
- [download.md](./download.md) - 数据集下载
- [offline_render.md](./offline_render.md) - 离线渲染

---

## 11. 引用

如果本项目对您的研究有帮助，请引用原论文：

```bibtex
@article{qian2023gaussianavatars,
  title={GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians},
  author={Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2312.02069},
  year={2023}
}
```

---

**祝实验顺利！如有问题，请参考 [常见问题处理](#6-常见问题处理) 或提交 Issue。**
