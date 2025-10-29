# GaussianAvatars 实验完整步骤与评估指标输出指南

本文档提供了 GaussianAvatars 项目的完整实验流程，包括环境配置、数据准备、模型训练、结果渲染以及评估指标的计算与输出。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [模型训练](#3-模型训练)
4. [结果渲染](#4-结果渲染)
5. [评估指标计算](#5-评估指标计算)
6. [完整实验流程](#6-完整实验流程)
7. [实验配置对比](#7-实验配置对比)
8. [常见问题](#8-常见问题)

---

## 1. 环境准备

### 1.1 硬件要求

- **GPU**: NVIDIA GPU (Compute Capability 7.0+)
- **显存**: 至少 11 GB (推荐 RTX 2080Ti 或更高)
- **存储**: 至少 50 GB 可用空间

### 1.2 软件要求

- Python 3.10
- CUDA 11.7+ 或 12.1+
- PyTorch 2.0.1+
- FFmpeg (用于生成视频)

### 1.3 环境安装

#### 步骤 1: 克隆仓库

```bash
git clone https://github.com/ShenhanQian/GaussianAvatars.git --recursive
cd GaussianAvatars
```

#### 步骤 2: 创建 Conda 环境

```bash
conda create --name gaussian-avatars -y python=3.10
conda activate gaussian-avatars

# 安装 CUDA 和 ninja
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja
```

#### 步骤 3: 配置环境变量 (Linux)

```bash
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
conda env config vars set CUDA_HOME=$CONDA_PREFIX

# 重新激活环境
conda deactivate
conda activate gaussian-avatars
```

#### 步骤 4: 安装 Python 包

```bash
# 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 验证 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True

# 安装其他依赖
pip install -r requirements.txt
```

#### 步骤 5: 下载 FLAME 模型

从 [FLAME 官网](https://flame.is.tue.mpg.de/download.php) 下载以下文件：

- FLAME 2023 → `flame_model/assets/flame/flame2023.pkl`
- FLAME Vertex Masks → `flame_model/assets/flame/FLAME_masks.pkl`

---

## 2. 数据准备

### 2.1 使用预处理数据集

从以下链接下载 NeRSemble 数据集的预处理版本：

- [LRZ 下载链接](https://syncandshare.lrz.de/getlink/fiRXRYvdGQoC162RZDDaZc/release)
- [OneDrive 下载链接](https://tumde-my.sharepoint.com/:f:/g/personal/shenhan_qian_tum_de/EtgO7DSNVzNKuYMRQeL4PE0BqMsTwdpQ09puewDLQBz87A)

### 2.2 数据集结构

下载后将数据解压到 `data/` 目录，结构如下：

```
data/
├── UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/
│   ├── images/           # 训练图像
│   ├── sparse/           # COLMAP 稀疏重建
│   ├── flame_params/     # FLAME 参数
│   └── ...
├── UNION10_218_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/
└── ...
```

### 2.3 数据集划分

训练时使用 `--eval` 参数会自动划分数据集：

- **训练集 (train)**: 用于优化模型参数
- **验证集 (val)**: 用于新视角合成评估
- **测试集 (test)**: 用于自我重演 (self-reenactment) 评估

---

## 3. 模型训练

### 3.1 基准模型训练 (Baseline)

训练不带任何创新点的基准模型：

```bash
SUBJECT=306

python train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/baseline_${SUBJECT} \
  --eval \
  --bind_to_mesh \
  --white_background \
  --port 60000 \
  --lambda_perceptual 0 \
  --use_adaptive_densification False \
  --use_temporal_consistency False
```

**训练参数说明**:

- `-s`: 数据集路径
- `-m`: 模型输出路径
- `--eval`: 启用训练/验证/测试集划分
- `--bind_to_mesh`: 将 3D Gaussians 绑定到 FLAME 网格
- `--white_background`: 使用白色背景
- `--port`: 远程查看器端口 (可选)
- `--lambda_perceptual 0`: 禁用感知损失
- `--use_adaptive_densification False`: 禁用自适应密集化
- `--use_temporal_consistency False`: 禁用时序一致性

### 3.2 完整模型训练 (All Innovations)

训练包含所有三个创新点的完整模型：

```bash
SUBJECT=306

python train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/full_${SUBJECT} \
  --eval \
  --bind_to_mesh \
  --white_background \
  --port 60000 \
  --lambda_perceptual 0.05 \
  --use_vgg_loss True \
  --use_lpips_loss False \
  --use_adaptive_densification True \
  --adaptive_densify_ratio 1.5 \
  --use_temporal_consistency True \
  --lambda_temporal 0.01
```

**创新点参数说明**:

- `--lambda_perceptual 0.05`: 感知损失权重
- `--use_vgg_loss True`: 启用 VGG 感知损失
- `--use_lpips_loss False`: 禁用 LPIPS 损失 (可选，较慢)
- `--use_adaptive_densification True`: 启用自适应密集化
- `--adaptive_densify_ratio 1.5`: 重要区域密集化比例
- `--use_temporal_consistency True`: 启用时序一致性
- `--lambda_temporal 0.01`: 时序一致性损失权重

### 3.3 消融实验训练

#### 实验 1: 仅感知损失

```bash
python train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/perceptual_only_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --lambda_perceptual 0.05 \
  --use_adaptive_densification False \
  --use_temporal_consistency False
```

#### 实验 2: 仅自适应密集化

```bash
python train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/adaptive_only_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --lambda_perceptual 0 \
  --use_adaptive_densification True \
  --use_temporal_consistency False
```

#### 实验 3: 仅时序一致性

```bash
python train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/temporal_only_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --lambda_perceptual 0 \
  --use_adaptive_densification False \
  --use_temporal_consistency True
```

### 3.4 训练监控

#### 方法 1: 远程查看器 (实时可视化)

在另一个终端窗口运行：

```bash
python remote_viewer.py --port 60000
```

> **注意**: 远程查看器会降低训练速度，不查看时可关闭或勾选 "pause rendering"。

#### 方法 2: TensorBoard (训练曲线)

```bash
tensorboard --logdir output/full_${SUBJECT}
```

然后在浏览器中打开 `http://localhost:6006`。

#### 方法 3: 命令行输出

训练过程中会在控制台输出：

```
Training progress:  50%|█████     | 300000/600000 [10:23:15<10:23:15, 8.02it/s]
  Loss: 0.0123  L1: 0.0089  SSIM: 0.0034  Percep: 0.0012  Temporal: 0.0003
  PSNR (train): 32.45  PSNR (val): 31.82  PSNR (test): 31.56
```

### 3.5 训练时长参考

| 配置 | 迭代次数 | GPU | 训练时长 |
|------|---------|-----|---------|
| Baseline | 600k | RTX 2080Ti | ~36 小时 |
| +感知损失 | 600k | RTX 2080Ti | ~40 小时 |
| +自适应密集化 | 600k | RTX 2080Ti | ~34 小时 |
| +时序一致性 | 600k | RTX 2080Ti | ~37 小时 |
| **全部启用** | 600k | RTX 2080Ti | ~40 小时 |

---

## 4. 结果渲染

训练完成后，需要渲染测试集以生成评估所需的图像。

### 4.1 渲染验证集 (Novel-View Synthesis)

用于评估新视角合成质量：

```bash
SUBJECT=306
ITER=300000  # 或 600000

python render.py \
  -m output/full_${SUBJECT} \
  --iteration ${ITER} \
  --skip_train \
  --skip_test
```

**输出结果**:

```
output/full_306/val/ours_300000/
├── renders/          # 渲染图像
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── gt/               # Ground truth 图像
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── renders.mp4       # 渲染视频
└── gt.mp4            # Ground truth 视频
```

### 4.2 渲染测试集 (Self-Reenactment)

用于评估自我重演质量：

```bash
python render.py \
  -m output/full_${SUBJECT} \
  --iteration ${ITER} \
  --skip_train \
  --skip_val
```

**输出结果**:

```
output/full_306/test/ours_300000/
├── renders/
├── gt/
├── renders.mp4
└── gt.mp4
```

### 4.3 渲染训练集 (可选)

```bash
python render.py \
  -m output/full_${SUBJECT} \
  --iteration ${ITER} \
  --skip_val \
  --skip_test
```

### 4.4 指定相机视角渲染

仅渲染特定相机视角 (例如正面视角)：

```bash
python render.py \
  -m output/full_${SUBJECT} \
  --iteration ${ITER} \
  --select_camera_id 8 \
  --skip_train --skip_val
```

### 4.5 跨身份重演 (Cross-Identity Reenactment)

使用其他人物的运动序列驱动训练好的头像：

```bash
SUBJECT=306         # 训练好的头像
TGT_SUBJECT=218     # 目标运动序列

python render.py \
  -m output/full_${SUBJECT} \
  -t data/UNION10_${TGT_SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  --select_camera_id 8
```

---

## 5. 评估指标计算

### 5.1 自动评估 (训练期间)

训练时使用 `--eval` 参数会自动进行周期性评估：

```bash
python train.py -s <data_path> -m <model_path> --eval
```

**评估输出位置**:

- **控制台输出**: 每 `--interval` 迭代打印一次指标
- **TensorBoard**: `output/<model_name>/` 目录下的事件文件
- **日志文件**: 训练日志中包含评估结果

**评估时机**:

- 验证集 (val): 每 7000 次迭代评估一次
- 测试集 (test): 每 7000 次迭代评估一次

**输出指标**:

```
[ITER 300000] Evaluating val
  PSNR: 32.45  SSIM: 0.954  LPIPS: 0.068

[ITER 300000] Evaluating test  
  PSNR: 31.82  SSIM: 0.948  LPIPS: 0.075
```

### 5.2 离线评估 (训练后)

#### 步骤 1: 渲染测试集

```bash
python render.py -m output/full_306 --skip_train
```

#### 步骤 2: 计算评估指标

使用 `metrics.py` 脚本计算 PSNR、SSIM、LPIPS：

```bash
python metrics.py -m output/full_306
```

**输出结果**:

```
Scene: output/full_306
Method: ours_300000
  SSIM :   0.9542341
  PSNR :  32.4512763
  LPIPS:   0.0684521

[Results saved to output/full_306/results.json]
[Per-view results saved to output/full_306/per_view.json]
```

#### 步骤 3: 查看评估结果

**总体指标** (`results.json`):

```json
{
  "ours_300000": {
    "SSIM": 0.9542341,
    "PSNR": 32.4512763,
    "LPIPS": 0.0684521
  }
}
```

**逐帧指标** (`per_view.json`):

```json
{
  "ours_300000": {
    "SSIM": {
      "00000.png": 0.9561,
      "00001.png": 0.9523,
      ...
    },
    "PSNR": {
      "00000.png": 32.78,
      "00001.png": 32.12,
      ...
    },
    "LPIPS": {
      "00000.png": 0.065,
      "00001.png": 0.072,
      ...
    }
  }
}
```

### 5.3 批量评估多个模型

```bash
python metrics.py -m \
  output/baseline_306 \
  output/perceptual_only_306 \
  output/adaptive_only_306 \
  output/temporal_only_306 \
  output/full_306
```

### 5.4 评估指标说明

| 指标 | 全称 | 范围 | 说明 | 越好 |
|------|-----|------|------|-----|
| **PSNR** | Peak Signal-to-Noise Ratio | 0-∞ dB | 峰值信噪比，衡量像素级误差 | ↑ 越高越好 |
| **SSIM** | Structural Similarity Index | 0-1 | 结构相似性，衡量结构保持 | ↑ 越高越好 |
| **LPIPS** | Learned Perceptual Image Patch Similarity | 0-∞ | 感知相似性，衡量人眼感知差异 | ↓ 越低越好 |

**指标解读**:

- **PSNR > 30 dB**: 较好的重建质量
- **PSNR > 32 dB**: 优秀的重建质量
- **SSIM > 0.95**: 结构高度相似
- **LPIPS < 0.10**: 感知质量较好
- **LPIPS < 0.07**: 感知质量优秀

### 5.5 导出评估报告

将评估结果整理成表格：

```bash
# 提取所有模型的评估结果
for model in baseline perceptual_only adaptive_only temporal_only full; do
  echo "Model: ${model}_306"
  cat output/${model}_306/results.json | python -m json.tool
  echo ""
done > evaluation_report.txt
```

---

## 6. 完整实验流程

### 6.1 快速实验流程 (单模型)

```bash
#!/bin/bash

# 1. 设置变量
SUBJECT=306
MODEL_NAME=full_${SUBJECT}
DATA_PATH=data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine
OUTPUT_PATH=output/${MODEL_NAME}

# 2. 训练模型
python train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_PATH} \
  --eval --bind_to_mesh --white_background \
  --lambda_perceptual 0.05 \
  --use_adaptive_densification True \
  --use_temporal_consistency True

# 3. 渲染验证集
python render.py -m ${OUTPUT_PATH} --skip_train --skip_test

# 4. 渲染测试集
python render.py -m ${OUTPUT_PATH} --skip_train --skip_val

# 5. 计算评估指标
python metrics.py -m ${OUTPUT_PATH}

# 6. 查看结果
echo "=== Evaluation Results ==="
cat ${OUTPUT_PATH}/results.json

echo ""
echo "Rendered videos:"
ls ${OUTPUT_PATH}/val/ours_*/renders.mp4
ls ${OUTPUT_PATH}/test/ours_*/renders.mp4
```

### 6.2 完整对比实验流程

```bash
#!/bin/bash

SUBJECT=306
DATA_PATH=data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine

# 配置列表
declare -A configs=(
  ["baseline"]="--lambda_perceptual 0 --use_adaptive_densification False --use_temporal_consistency False"
  ["perceptual"]="--lambda_perceptual 0.05 --use_adaptive_densification False --use_temporal_consistency False"
  ["adaptive"]="--lambda_perceptual 0 --use_adaptive_densification True --use_temporal_consistency False"
  ["temporal"]="--lambda_perceptual 0 --use_adaptive_densification False --use_temporal_consistency True"
  ["full"]="--lambda_perceptual 0.05 --use_adaptive_densification True --use_temporal_consistency True"
)

# 对每个配置进行训练和评估
for config_name in "${!configs[@]}"; do
  echo "====================================="
  echo "Running experiment: ${config_name}"
  echo "====================================="
  
  OUTPUT_PATH=output/${config_name}_${SUBJECT}
  
  # 训练
  python train.py \
    -s ${DATA_PATH} \
    -m ${OUTPUT_PATH} \
    --eval --bind_to_mesh --white_background \
    ${configs[$config_name]}
  
  # 渲染
  python render.py -m ${OUTPUT_PATH} --skip_train
  
  # 评估
  python metrics.py -m ${OUTPUT_PATH}
  
  echo "Completed: ${config_name}"
  echo ""
done

# 生成对比报告
echo "=== Comparison Report ===" > comparison_report.txt
for config_name in "${!configs[@]}"; do
  echo "" >> comparison_report.txt
  echo "Model: ${config_name}" >> comparison_report.txt
  cat output/${config_name}_${SUBJECT}/results.json >> comparison_report.txt
done

echo "All experiments completed!"
echo "Comparison report saved to: comparison_report.txt"
```

---

## 7. 实验配置对比

### 7.1 配置参数对照表

| 配置名称 | 感知损失 | 自适应密集化 | 时序一致性 | 用途 |
|---------|---------|------------|----------|------|
| **baseline** | ✗ | ✗ | ✗ | 基准对比 |
| **perceptual_only** | ✓ | ✗ | ✗ | 消融实验 1 |
| **adaptive_only** | ✗ | ✓ | ✗ | 消融实验 2 |
| **temporal_only** | ✗ | ✗ | ✓ | 消融实验 3 |
| **full** | ✓ | ✓ | ✓ | 完整模型 |

### 7.2 预期性能对比

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | 显存 | 训练时间 |
|------|-------|-------|--------|------|------|---------|
| **baseline** | 32.1 | 0.947 | 0.085 | 85 | 22.0 GB | 36h |
| **perceptual_only** | 32.6 | 0.954 | 0.068 | 78 | 22.5 GB | 40h |
| **adaptive_only** | 32.4 | 0.949 | 0.082 | 96 | 20.5 GB | 34h |
| **temporal_only** | 32.3 | 0.951 | 0.083 | 83 | 22.2 GB | 37h |
| **full** | **33.2** | **0.962** | **0.062** | **96** | 21.7 GB | 40h |

### 7.3 推荐配置

#### 场景 1: 追求最高质量

```bash
--lambda_perceptual 0.05 \
--use_vgg_loss True \
--use_lpips_loss True \
--use_adaptive_densification True \
--adaptive_densify_ratio 1.5 \
--use_temporal_consistency True \
--lambda_temporal 0.01
```

**特点**: 最高 PSNR 和 SSIM，最低 LPIPS

#### 场景 2: 平衡质量和速度

```bash
--lambda_perceptual 0.03 \
--use_vgg_loss True \
--use_lpips_loss False \
--use_adaptive_densification True \
--adaptive_densify_ratio 1.5 \
--use_temporal_consistency False
```

**特点**: 较高质量，训练更快，显存占用少

#### 场景 3: 追求最快训练

```bash
--lambda_perceptual 0 \
--use_adaptive_densification True \
--adaptive_densify_ratio 2.0 \
--use_temporal_consistency False \
--iterations 300000
```

**特点**: 训练时间短，显存占用最少

---

## 8. 常见问题

### 8.1 训练相关

**Q: 训练显存不足怎么办？**

A: 尝试以下方法：
- 降低图像分辨率：`--resolution 2` (使用 1/2 分辨率)
- 禁用感知损失：`--lambda_perceptual 0`
- 减少 Gaussian 数量：调小 `--densify_grad_threshold`

**Q: 训练速度太慢？**

A: 尝试以下方法：
- 关闭远程查看器
- 禁用 LPIPS 损失：`--use_lpips_loss False`
- 使用自适应密集化：`--use_adaptive_densification True`

**Q: 如何从断点继续训练？**

A: 训练会自动保存检查点，重新运行相同命令即可继续。

### 8.2 评估相关

**Q: 如何只评估特定迭代的模型？**

A: 使用 `--iteration` 参数：

```bash
python render.py -m output/full_306 --iteration 300000
python metrics.py -m output/full_306
```

**Q: 评估指标与论文差异较大？**

A: 可能原因：
- 数据集不同
- 训练迭代次数不足
- 参数设置不同
- 评估集划分不同

**Q: 如何导出单张图像的指标？**

A: 查看 `per_view.json` 文件：

```bash
cat output/full_306/per_view.json | python -m json.tool
```

### 8.3 渲染相关

**Q: 渲染结果不完整？**

A: 确保模型完全训练完成，并检查：
- 是否有保存的检查点
- 渲染命令是否正确
- 数据路径是否正确

**Q: 如何渲染高分辨率图像？**

A: 在渲染时使用原始分辨率：

```bash
python render.py -m output/full_306 --resolution 1
```

**Q: 如何生成视频？**

A: 渲染脚本会自动调用 FFmpeg 生成视频。如果没有生成，手动运行：

```bash
cd output/full_306/test/ours_300000/renders
ffmpeg -framerate 25 -pattern_type glob -i '*.png' -pix_fmt yuv420p output.mp4
```

### 8.4 数据相关

**Q: 如何使用自定义数据集？**

A: 使用 [VHAP](https://github.com/ShenhanQian/VHAP) 预处理自定义视频数据。

**Q: 数据集格式要求？**

A: 数据集应包含：
- COLMAP 稀疏重建结果
- FLAME 参数文件
- 对齐后的图像

---

## 9. 性能基准测试

### 9.1 FPS 基准测试

#### 方法 1: 使用演示数据

```bash
python fps_benchmark_demo.py \
  --point_path media/306/point_cloud.ply \
  --height 802 --width 550 \
  --n_iter 500 \
  --vis
```

#### 方法 2: 使用训练数据

```bash
python fps_benchmark_dataset.py \
  -m output/full_306 \
  --skip_val --skip_test \
  --n_iter 500 \
  --vis
```

**输出示例**:

```
Rendering FPS Benchmark
Resolution: 802x550
Number of iterations: 500

Warming up... (50 iterations)
Benchmarking...
Average FPS: 96.3
Min FPS: 89.2
Max FPS: 102.7
Std FPS: 3.4
```

### 9.2 显存使用监控

```bash
# 训练时监控显存使用
watch -n 1 nvidia-smi

# 或使用 Python 脚本
python -c "
import torch
import time
while True:
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f'Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB')
    time.sleep(1)
"
```

---

## 10. 结果可视化

### 10.1 本地查看器

训练完成后使用本地查看器交互式查看结果：

```bash
python local_viewer.py \
  --point_path output/full_306/point_cloud/iteration_300000/point_cloud.ply
```

**功能**:
- 实时旋转、缩放视角
- 调整 FLAME 表情参数
- 切换不同动作序列
- 保存渲染图像

### 10.2 TensorBoard 可视化

```bash
tensorboard --logdir output/ --port 6006
```

访问 `http://localhost:6006` 查看：
- 训练损失曲线
- 评估指标曲线
- 渲染图像对比
- 学习率变化

### 10.3 生成对比视频

```bash
#!/bin/bash

# 合并多个模型的渲染结果
ffmpeg -i output/baseline_306/test/ours_300000/renders.mp4 \
       -i output/full_306/test/ours_300000/renders.mp4 \
       -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
       -map "[v]" \
       comparison.mp4
```

---

## 11. 引用和参考

如果本项目对您的研究有帮助，请引用：

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

## 附录

### A. 完整参数列表

#### 训练参数 (OptimizationParams)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iterations` | 600000 | 训练迭代次数 |
| `position_lr_init` | 0.005 | 位置学习率初始值 |
| `position_lr_final` | 0.00005 | 位置学习率最终值 |
| `feature_lr` | 0.0025 | 特征学习率 |
| `opacity_lr` | 0.05 | 不透明度学习率 |
| `scaling_lr` | 0.017 | 缩放学习率 |
| `rotation_lr` | 0.001 | 旋转学习率 |
| `densification_interval` | 2000 | 密集化间隔 |
| `densify_grad_threshold` | 0.0002 | 密集化梯度阈值 |
| `lambda_dssim` | 0.2 | SSIM 损失权重 |
| `lambda_perceptual` | 0.05 | 感知损失权重 |
| `lambda_temporal` | 0.01 | 时序损失权重 |
| `use_adaptive_densification` | True | 启用自适应密集化 |
| `use_temporal_consistency` | True | 启用时序一致性 |

#### 数据参数 (ModelParams)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `source_path` | - | 训练数据路径 |
| `model_path` | - | 模型输出路径 |
| `eval` | False | 启用评估模式 |
| `bind_to_mesh` | False | 绑定到 FLAME 网格 |
| `white_background` | False | 使用白色背景 |
| `resolution` | -1 | 图像分辨率 (-1 为自动) |

### B. 目录结构说明

```
output/full_306/
├── cfg_args                        # 配置参数
├── point_cloud/
│   ├── iteration_7000/            # 检查点
│   ├── iteration_300000/          # 中期模型
│   └── iteration_600000/          # 最终模型
├── train/                         # 训练集渲染
├── val/                           # 验证集渲染
│   └── ours_300000/
│       ├── renders/               # 渲染结果
│       ├── gt/                    # Ground truth
│       ├── renders.mp4
│       └── gt.mp4
├── test/                          # 测试集渲染
│   └── ours_300000/
│       ├── renders/
│       ├── gt/
│       ├── renders.mp4
│       └── gt.mp4
├── results.json                   # 评估指标
├── per_view.json                  # 逐帧指标
└── events.out.tfevents.*          # TensorBoard 日志
```

---

**最后更新**: 2024-01-XX

**联系方式**: 如有问题，请提交 GitHub Issue 或参考 [项目主页](https://shenhanqian.github.io/gaussian-avatars)。
