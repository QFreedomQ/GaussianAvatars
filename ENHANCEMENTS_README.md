# GaussianAvatars 增强模块使用指南

本仓库在原始 GaussianAvatars 基础上新增了 **感知损失增强模块**，用于提升头部重建的质量和纹理细节。

---

## 🆕 新增功能概览

| 模块编号 | 名称 | 功能 | 启用参数 |
|---------|------|------|---------|
| **1** | 感知损失增强 | VGG/LPIPS 特征损失 | `--lambda_perceptual 0.05 --use_vgg_loss` |

---

## 📂 文件结构

```
GaussianAvatars/
├── utils/
│   └── perceptual_loss.py              # 模块 1：感知损失
│
├── arguments/__init__.py               # 新增命令行参数
├── train.py                            # 集成感知损失
│
├── New.md                              # 原论文重构方案（中文）
├── Change.md                           # 增强模块详解（中文）
├── SUMMARY.md                          # 实现总结
├── train_with_enhancements.sh         # 一键训练脚本
└── ENHANCEMENTS_README.md             # 本文档
```

---

## 🚀 快速开始

### 方法 1：使用一键脚本（推荐）

```bash
# 1. 设置数据路径
export SUBJECT=306
export DATA_DIR="data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"

# 2. 运行一键训练脚本（包含 Baseline + 感知损失增强）
chmod +x train_with_enhancements.sh
./train_with_enhancements.sh
```

脚本将依次训练：
1. Baseline（无增强）
2. Perceptual Loss（感知损失增强）

所有结果保存在 `output/` 目录，自动生成 JSON 指标和 MP4 视频。

---

### 方法 2：手动训练单个模块

#### Baseline（用于对比）
```bash
python train.py \
  -s ${DATA_DIR} \
  -m output/baseline \
  --bind_to_mesh --white_background --eval --iterations 600000
```

#### 模块 1：感知损失增强（推荐首选）
```bash
python train.py \
  -s ${DATA_DIR} \
  -m output/perceptual_loss \
  --bind_to_mesh --white_background --eval --iterations 600000 \
  --lambda_perceptual 0.05 \
  --use_vgg_loss
```

**预期效果**：LPIPS ↓10-20%，纹理更清晰，头发/皮肤细节改善。

---

## 📊 评估与可视化

### 渲染并计算指标
```bash
# 渲染 val 和 test 集
python render.py -m output/your_model

# 计算 PSNR/SSIM/LPIPS
python metrics.py -m output/your_model

# 跨身份重演（可选）
python render.py -m output/your_model --target_path ${TARGET_DATA}
python evaluate_cross_identity.py -m output/your_model -t ${TARGET_SUBJECT_NAME}
```

### 生成对比视频
```bash
# Baseline vs Innovation1
ffmpeg -i output/baseline/val/ours_600000/renders.mp4 \
       -i output/perceptual_loss/val/ours_600000/renders.mp4 \
       -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" \
       comparison.mp4
```

---

## 📖 详细文档

| 文档 | 内容 | 字数/行数 |
|------|------|----------|
| **New.md** | 原论文重构方案（中文）| 174 行 |
|  | - 环境搭建、数据准备 | |
|  | - 训练流程、评估管线 | |
|  | - 可视化方法、命令速查 | |
| **Change.md** | 增强模块详解（中文）| 237 行 |
|  | - 感知损失模块的原理、实现、使用 | |
|  | - 参数调优指南 | |
|  | - 预期效果对比 | |
|  | - 论文撰写建议 | |
| **SUMMARY.md** | 实现总结 | 完整清单 |

**推荐阅读顺序**：
1. 先读 `SUMMARY.md`（5 分钟）了解全貌
2. 再读 `New.md`（15 分钟）掌握实验流程
3. 最后读 `Change.md`（30 分钟）理解每个模块细节

---

## 🔧 参数调优技巧

### 感知损失权重
| λ_perceptual | 效果 | 适用场景 |
|--------------|------|----------|
| 0.02-0.03 | 轻微增强 | 已有较好纹理 |
| **0.05** | **推荐** | 大多数场景 |
| 0.08-0.1 | 强增强 | 极度模糊数据 |

**常见问题**：若出现颜色偏移，降低权重到 0.03。

---

## ❓ 常见问题

### Q1: 如何禁用某个模块？
A: 不传对应的 `--use_xxx` 参数即可（默认全部禁用）。

### Q2: 训练时间会增加多少？
A: 感知损失：+5-10%

### Q3: 如何验证模块生效？
A: 检查 TensorBoard：
    - 感知损失：`train_loss_patches/perceptual_loss` 曲线应下降

### Q4: 论文中应该报告哪些指标？
A: - **必须**：PSNR、SSIM、LPIPS（val + test）
   - **推荐**：BRISQUE（跨身份）、训练时间、高斯数量
   - **可选**：用户研究 MOS 评分

---

## 📚 参考论文

### 感知损失增强
- Johnson et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution." ECCV 2016.
- Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." CVPR 2018.

---

## 🎓 引用

如果您使用了这些增强模块，请引用原始 GaussianAvatars 论文：

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

## 📧 联系方式

- **问题反馈**：请在仓库中提 Issue
- **论文合作**：请查阅 `Change.md` 第 8 节的论文撰写建议
- **代码贡献**：欢迎 Pull Request

---

**版本**: 1.1  
**最后更新**: 2024-11-21  
**许可证**: 与原 GaussianAvatars 保持一致（CC-BY-NC-SA-4.0）

---

**快速链接**：
- [New.md - 原论文重构方案](New.md)
- [Change.md - 增强模块详解](Change.md)
- [SUMMARY.md - 实现总结](SUMMARY.md)
- [train_with_enhancements.sh - 一键训练脚本](train_with_enhancements.sh)
