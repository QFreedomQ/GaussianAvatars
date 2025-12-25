# GaussianAvatars 原论文方法重构与强化方案

> ✅ **目标**：重新从 GaussianAvatars 官方源码出发，严格在原始数据集（UNION10_*）上复现论文结果，并为后续论文撰写/创新模块集成提供一份端到端的实验、评估与可视化攻略。

## 1. 研究策略总览
- **阶段A：方法重构** —— 使用官方脚本 `train.py`、`render.py` 等重新跑通原论文管线，确保 *“相同数据集、相同配置、完全可复现”*。
- **阶段B：模块验证** —— 在已验证的baseline之上，引入感知损失等增益（或其它参考论文中的高级技术），用同一评估协议量化收益。
- **阶段C：撰写与可视化** —— 统一整理实验日志、指标JSON、视频/对比图，用于论文图表与附录。

## 2. 方法论拆解（代码定位 + 关键机制）
| 模块 | 对应文件 | 说明 |
|------|----------|------|
| 数据/场景加载 | `scene/dataset_readers.py`, `scene/__init__.py` | 解析 COLMAP 相机、FLAME 参数、掩膜等；`Scene` 类负责 train/val/test 划分。 |
| 3D 高斯模型 | `scene/gaussian_model.py`, `scene/flame_gaussian_model.py` | `GaussianModel` 负责通用 Gaussian 表达；`FlameGaussianModel` 绑定 FLAME 网格并随表情驱动。 |
| 渲染器 | `gaussian_renderer/__init__.py`, `mesh_renderer/__init__.py` | 高斯溅射主渲染 + nvdiffrast 网格渲染；训练/离线渲染共用。 |
| 损失 | `utils/loss_utils.py`, `utils/perceptual_loss.py`, `lpipsPyTorch/` | L1 + SSIM 为原论文主损失；`CombinedPerceptualLoss` 封装 VGG/LPIPS。 |
| CLI 参数 | `arguments/__init__.py` | `ModelParams`/`PipelineParams`/`OptimizationParams` 定义全部 flags。 |
| 训练主管道 | `train.py` | 数据加载、GUI通讯、损失汇总、优化器、checkpoint。 |
| 评估 | `render.py`, `metrics.py`, `evaluate_cross_identity.py` | 离线渲染、PSNR/SSIM/LPIPS 计算、BRISQUE 跨身份评估。 |
| 可视化 | `remote_viewer.py`, `local_viewer.py`, `doc/offline_render.md` | 远程/本地 GUI，离线渲染说明。 |

### 2.1 原论文优化目标
1. **外观**：L1 + SSIM（`train.py` 第178行附近）。
2. **密度自适应**：`gaussians.training_setup()` 内部 densify/prune 策略。
3. **FLAME 绑定**：每帧根据 `view.timestep` 在 `FlameGaussianModel.select_mesh_by_timestep()` 中切换驱动表情。
4. **可选增强（创新1）**：感知损失（`utils/perceptual_loss.py`）在 `train.py` 第55-75 行初始化，`lambda_perceptual` 控权重。

### 2.2 需保证的“原汁原味”设置
- **相同数据集**：`data/UNION10_${SUBJECT}_...`；训练命令中 `-s` 必须指向该目录。
- **白色背景**：`--white_background`，保证与官方实验一致。
- **迭代数**：600k（`--iterations 600000`）。
- **Mesh 绑定**：`--bind_to_mesh`，否则无法复现论文中的表情驱动。

## 3. 实验工作流（环境 → 数据 → 训练）
### 3.1 环境搭建
```bash
# 1. 克隆并进入仓库
git clone --recursive https://github.com/ShenhanQian/GaussianAvatars.git
cd GaussianAvatars

# 2. 创建 Python 环境
conda create -n gaussian-avatars python=3.10 -y
conda activate gaussian-avatars

# 3. 安装依赖
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install lpipsPyTorch piq  # 感知损失 + BRISQUE
```
> 位置参考：`doc/installation.md`、`requirements.txt`。

### 3.2 数据准备（与原文一致）
```bash
# 以主体 306 为例，保持与论文相同数据
export SUBJECT=306
export DATA_DIR="data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
```
- 数据结构需包含 `images/`, `masks/`, `cameras.json`, `flame_params.npz`。
- 下载方式与备注详见 `doc/download.md`。

### 3.3 训练阶段
#### (1) 原论文 Baseline（用于重构验证）
```bash
python train.py \
  -s ${DATA_DIR} \
  -m output/baseline_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --iterations 600000 \
  --lambda_perceptual 0.0 \
  --interval 60000
```
- 关键代码：`train.py` → `training()`；`arguments/__init__.py` → `OptimizationParams.iterations`。
- 输出：`output/baseline_${SUBJECT}/point_cloud/iteration_600000/point_cloud.ply` 等。

#### (2) 感知损失增强（创新验证）
```bash
python train.py \
  -s ${DATA_DIR} \
  -m output/perceptual_${SUBJECT} \
  --eval --bind_to_mesh --white_background \
  --iterations 600000 \
  --lambda_perceptual 0.05 \
  --use_vgg_loss \
  --interval 60000
```
- 若需加入 LPIPS：追加 `--use_lpips_loss`（默认 false）。
- 可在 `arguments/__init__.py` > `OptimizationParams` 中调整默认值，便于批量脚本化。

#### (3) 训练监控
| 工具 | 命令 | 说明 |
|------|------|------|
| TensorBoard | `tensorboard --logdir output --port 6006` | 指标路径写在 `train.py` 的 `prepare_output_and_logger()`。|
| 远程查看器 | 训练端保留 `--port 60000`；另开终端运行 `python remote_viewer.py --port 60000` | 文件：`remote_viewer.py`。|
| 本地查看器 | `python local_viewer.py --point_path output/.../point_cloud.ply` | 文件：`local_viewer.py`。|

> 若训练中断，可通过 `--checkpoint` 指向 `output/.../chkpnt` 恢复，逻辑在 `train.py` 第50-54 行。

## 4. 评估与实验管线（Val/Test + Cross-ID）
### 4.1 渲染各拆分
```bash
# 渲染 val + test（默认 train/val/test 全部）
python render.py -m output/perceptual_${SUBJECT}

# 仅渲染 val
python render.py -m output/perceptual_${SUBJECT} --skip_train --skip_test

# 仅渲染 test
python render.py -m output/perceptual_${SUBJECT} --skip_train --skip_val
```
- 关键函数：`render.py` → `render_sets()`（第111-156 行）。
- 结果：`output/.../val/ours_600000/{renders,gt}/` + `renders.mp4`。

### 4.2 Novel-View & Self-Reenactment 指标
```bash
python metrics.py -m output/perceptual_${SUBJECT}
```
- `metrics.py`（第59-193 行）会遍历 `val/` 与 `test/`，生成 `val_results.json`, `test_results.json`, `*_per_view.json`。
- 指标：PSNR、SSIM、LPIPS；LPIPS 由 `lpipsPyTorch.LPIPS` 在 GPU 预加载。

### 4.3 Cross-Identity Reenactment + BRISQUE
```bash
export SOURCE_MODEL=output/perceptual_${SUBJECT}
export TARGET_SUBJECT=218
export TARGET_DATA="data/UNION10_${TARGET_SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"

# 渲染跨身份序列
python render.py -m ${SOURCE_MODEL} --target_path ${TARGET_DATA} --iteration 600000

# 计算 BRISQUE
python evaluate_cross_identity.py \
  -m ${SOURCE_MODEL} \
  -t UNION10_${TARGET_SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine
```
- `render.py` 第123-147 行：`--target_path` 会将目标主体 FLAME 动作套用到源主体高斯上。
- `evaluate_cross_identity.py`：`compute_brisque_scores()` 生成 `cross_identity_metrics.json`（均值/方差/极值）。
- 若画质欠佳，可用 `--cross_identity_quality_mode balanced`，增强逻辑在 `utils/image_enhancement.py`。

### 4.4 指标可视化
- **逐帧 LPIPS 曲线**：读取 `val_per_view.json`，示例脚本见 `All.md` §5.4，可放至 `notebooks/`。
- **误差热力图**：可将示例函数加入 `utils/image_utils.py` 或独立脚本，公式依据 L1。

## 5. 可视化与图像资产
### 5.1 自动视频
- `render.py` 默认调用 `ffmpeg` 生成 `renders.mp4`/`gt.mp4`；如缺失需 `sudo apt-get install ffmpeg`。

### 5.2 并排对比命令
```bash
ffmpeg -i output/baseline_${SUBJECT}/val/ours_600000/renders.mp4 \
       -i output/perceptual_${SUBJECT}/val/ours_600000/renders.mp4 \
       -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" \
       comparison_baseline_vs_perceptual.mp4
```
- 三视图（Baseline | Perceptual | GT）可将 `inputs=3`。

### 5.3 误差热力图示例
```python
from utils.image_utils import psnr  # 可重用同目录
# 建议新增函数 visualize_error_map(render_path, gt_path, output_path)
```
- 代码位置说明：若添加至 `utils/image_utils.py`，需在 `__all__` 中暴露供外部脚本调用。

### 5.4 GUI 可视化
| 工具 | 场景 | 命令 | 说明 |
|------|------|------|------|
| `remote_viewer.py` | 训练中远程监控 | `python remote_viewer.py --port 60000` | 支持网格叠加、相机控制。 |
| `local_viewer.py` | 训练后离线预览 | `python local_viewer.py --point_path output/.../point_cloud.ply` | 支持加载不同 motion (`--motion_path`). |

## 6. 进阶增强与参考论文
在确认 baseline 完整可复现后，可按下述方向迭代（保持数据/评估一致）：
1. **感知损失调度**（已提供）：`--lambda_perceptual`。可探索 `0.02~0.1`，或仅启用 `LPIPS`。
2. **表达式自适应着色**：参考 *Neural Head Avatars (CVPR 2023)*，可在 `scene/flame_gaussian_model.py` 中为不同 FLAME 三角面添加表达式条件颜色。

> 每个新模块务必在 `New.md` 或 `All.md` 旁另开小节，记录新增参数与代码位置，保持论文复现实验与创新实验之间的清晰对照。

## 7. 全流程命令速查（复现实验脚本草案）
```bash
# 0. 环境
conda activate gaussian-avatars

# 1. 数据环境变量
export SUBJECT=306
export DATA_DIR="data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"

# 2. Baseline 训练
python train.py -s ${DATA_DIR} -m output/baseline_${SUBJECT} --eval --bind_to_mesh --white_background --iterations 600000 --lambda_perceptual 0.0

# 3. 感知损失版本
python train.py -s ${DATA_DIR} -m output/perceptual_${SUBJECT} --eval --bind_to_mesh --white_background --iterations 600000 --lambda_perceptual 0.05 --use_vgg_loss

# 4. 离线渲染 + 指标
python render.py -m output/perceptual_${SUBJECT}
python metrics.py -m output/perceptual_${SUBJECT}

# 5. 跨身份
export TARGET_SUBJECT=218
export TARGET_DATA="data/UNION10_${TARGET_SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
python render.py -m output/perceptual_${SUBJECT} --target_path ${TARGET_DATA}
python evaluate_cross_identity.py -m output/perceptual_${SUBJECT} -t UNION10_${TARGET_SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine

# 6. 结果整理
ls output/perceptual_${SUBJECT}/{val,test}  # renders/mp4/json
```

## 8. 质量保证与注意事项
1. **一致性验证**：在引入新模块前，先记录 Baseline 指标（PSNR/SSIM/LPIPS/BRISQUE），保证后续对比公平。
2. **显存管理**：若 `train.py` 报 OOM，可在 CLI 中添加 `--resolution 1` 或编辑 `train.py` `DataLoader` 的 `num_workers`（默认 8）。
3. **LPIPS 权重下载**：首次运行 `metrics.py` 需网络；如受限，可预先下载至 `~/.cache/torch/hub/checkpoints`。
4. **日志归档**：推荐将 `output/`、`New.md`、`All.md` 及生成脚本一并纳入论文附录，方便审稿复现。

---
借助本方案，可以：
- **确认原论文方法** 在完全一致的数据与配置下可无缝运行；
- **系统化记录** 每一步的代码入口与命令行；
- **快速对比** 不同创新模块（含失败尝试），为撰写新论文提供详实素材。
