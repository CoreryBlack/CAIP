# RAICOM 2026 — 智海算法调优赛：参考文献

> 全部 PDF 下载自顶会/顶刊官方来源（CVF Open Access / arXiv / PMLR / OpenReview）
> 下载脚本: `download_papers.py`

---

## 索引

| # | 论文 | 发表 | 对应代码模块 | 核心结论 |
|---|------|------|-------------|---------|
| 01 | **EfficientNet** — Rethinking Model Scaling for CNNs | ICML 2019 | `model.py` | depth/width/resolution 联合缩放，B3+300 是合理选择 |
| 02 | **ConvNeXt** — A ConvNet for the 2020s | CVPR 2022 | `model.py` | 现代 recipe 下 CNN 依然很强，可做 ConvNeXt-Tiny 对比 |
| 03 | **AutoAugment** — Learning Augmentation Policies | CVPR 2019 | `augmentations.py` | 自动搜索增强策略可迁移 |
| 04 | **RandAugment** — Practical Automated Data Augmentation | NeurIPS 2020 | `augmentations.py` | 比 AutoAugment 更简单，调两参数即可 |
| 05 | **Mixup** — Beyond Empirical Risk Minimization | ICLR 2018 | `train_utils.py` | 样本/标签线性插值，提升泛化 |
| 06 | **CutMix** — Regularization with Localizable Features | ICCV 2019 | `train_utils.py` | patch 级混合，比 Mixup 更强 |
| 07 | **Random Erasing** — Data Augmentation | AAAI 2020 | `augmentations.py` (CoarseDropout) | 遮挡鲁棒性，已有近似实现 |
| 08 | **Label Smoothing** — When Does It Help? | NeurIPS 2019 | `train_utils.py` | 提升泛化与校准，已在用 (ε=0.1) |
| 09 | **Focal Loss** — for Dense Object Detection | ICCV 2017 | `train_utils.py` | 适合极端不平衡，4 分类不一定需要 |
| 10 | **Class-Balanced Loss** — Effective Number of Samples | CVPR 2019 | `train_utils.py` (待接入) | 比倒数加权更合理，备选 |
| 11 | **Balanced Softmax** — for Long-Tailed Recognition | NeurIPS 2020 | `train_utils.py` (待接入) | 长尾分类首选，优于 sampler |
| 12 | **Noisy Student** — Self-Training | CVPR 2020 | `model.py` | 预训练 + 伪标签 + 噪声增强 |
| 13 | **Quantization** — Integer-Arithmetic-Only Inference | CVPR 2018 | `export_onnx.py`, `infer.py` | INT8 量化可降延迟至 1/4 |
| 14 | **Knowledge Distillation** — Distilling the Knowledge | 2015 | `export_onnx.py` | 大模型 Teacher → 小模型 Student |

---

## 代码优先接入优先级

| 优先级 | 文献 | 改动量 | 预期收益 |
|--------|------|--------|---------|
| P0 | 修 EarlyStopping bug | 1 行 | 修复逻辑反向 |
| P1 | Mixup (05) | 中等 | 泛化 +0.5~2% F1 |
| P2 | RandAugment (04) vs 当前增强 A/B | 低 | 更稳的增强基线 |
| P3 | Balanced Softmax (11) | 低 | 处理类别不平衡 |
| P4 | 分层学习率 (从 01/02 推论) | 低 | 微调更稳 |
| P5 | INT8 量化 (13) | 低 | 提速抢同分排名 |
| P6 | 知识蒸馏 (14) | 中等 | 大 Teacher 教小 Student |
