# RAICOM 2026 — 智海算法调优赛

> **睿抗机器人开发者大赛 · CAIP 强脑赛道 · 智能算法赛项**
> 天气四分类（多云 / 雨天 / 雪天 / 晴天），EfficientNet-B3 + ONNX Runtime CPU 推理

---

## 📋 项目结构

```
raicom-weather/
├── config.py                  # 全局配置
├── augmentations.py           # 数据增强 (Albumentations)
├── dataset.py                 # 数据集 & DataLoader
├── model.py                   # EfficientNet-B3 模型定义
├── train_utils.py             # 训练工具 (Macro F1, LR Scheduler, Early Stopping)
├── train.py                   # 训练主脚本
├── quantize_onnx.py           # ONNX INT8 量化（dynamic / static）
├── requirements.txt           # 依赖清单
├── .gitignore
└── README.md
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
cd raicom-weather
pip install -r requirements.txt
```

推荐使用 Conda 虚拟环境：
```bash
conda create -n raicom python=3.10
conda activate raicom
pip install -r requirements.txt
```

### 2️⃣ 准备数据

将数据集按以下目录结构放置：

```
data/
├── train/
│   ├── cloudy/       ← 多云图片
│   ├── rainy/        ← 雨天图片
│   ├── snowy/        ← 雪天图片
│   └── sunny/        ← 晴天图片
├── val/              ← (可选) 独立验证集，结构同上
└── test/             ← (可选) 推理用图片直接放在此目录
```

> 也可使用 CSV 标注文件：`data/train/train.csv`（列：`filename, label`）

### 3️⃣ 训练

```bash
# 默认配置
python train.py --data-root ./data

# 自定义参数
python train.py --data-root ./data --epochs 60 --batch-size 64 --lr 1e-3 --image-size 300

# 增强强度 (0.0=轻, 1.0=标准, 2.0=极强)
python train.py --data-root ./data --aug-strength 1.0

# 从检查点恢复
python train.py --resume ./outputs/checkpoint.pth
```

训练过程中：
- 最佳模型保存至 `outputs/best_model.pth`
- 每轮检查点保存至 `outputs/checkpoint.pth`
- 训练日志保存至 `outputs/training_log.json`

### 4️⃣ 导出 ONNX

```bash
python export_onnx.py --checkpoint ./outputs/best_model.pth
```

可选项：
- `--image-size 300`：输入尺寸
- `--opset 17`：ONNX opset 版本
- `--no-dynamic-batch`：固定批大小（禁用动态轴）

导出时会自动运行 CPU 推理基准测试，输出单张推理耗时和 70 分钟可处理总量。

### 5️⃣ ONNX INT8 量化

```bash
# 先做 dynamic 量化（最快上手）
python quantize_onnx.py --onnx-path ./outputs/model.onnx --benchmark

# 再做 static 量化（推荐比赛最终版）
python quantize_onnx.py --onnx-path ./outputs/model.onnx \
                        --mode static \
                        --calib-dir ./data/train \
                        --max-calib-images 256 \
                        --benchmark
```

推荐工作流：
1. 先训练得到 `best_model.pth`
2. `export_onnx.py` 导出 FP32 ONNX
3. `quantize_onnx.py` 做 INT8 量化
4. 用 `infer.py --onnx-path <int8.onnx>` 直接测试

说明：
- `dynamic`：无需校准集，适合先看速度收益
- `static`：需要校准图片，通常对 CNN 模型更稳，推荐最终提交前尝试
- `--benchmark`：自动对 FP32 / INT8 做简单 CPU 吞吐测试

### 6️⃣ CPU 推理

```bash
# 对测试集推理（批处理）
python infer.py --onnx-path ./outputs/model.onnx --data-root ./data/test

# 启用 TTA（5 次增强取平均，提分但变慢）
python infer.py --onnx-path ./outputs/model.onnx --data-root ./data/test --tta

# 单张图片推理
python infer.py --onnx-path ./outputs/model.onnx --image-path ./data/sample.jpg

# 自定义输出
python infer.py --onnx-path ./outputs/model.onnx --data-root ./data/test \
                --output ./results/predictions.csv
```

推理结果 CSV 包含：`filename, pred_label, pred_class_id, prob_cloudy, prob_rainy, prob_snowy, prob_sunny`

---

## 🧠 模型架构

```
EfficientNet-B3 (timm, ImageNet pretrained)
    │
    ▼  Global Average Pooling (1536-dim)
    │
    ▼  Dropout (p=0.3) → Linear(1536→512) → BN → ReLU → Dropout (p=0.15) → Linear(512→4)
    │
    ▼  Logits → Softmax → 4-class probabilities
```

- 主干：`efficientnet_b3`（timm），输入 300×300
- 分类头：自定义 2 层 MLP + BatchNorm + Dropout
- 损失：CrossEntropyLoss + Label Smoothing (0.1)
- 优化器：AdamW + Cosine Warmup LR
- 混合精度训练（AMP）加速

---

## 📊 数据增强策略

| 增强类型 | 默认强度 | 说明 |
|---------|---------|------|
| 水平/垂直翻转 | ✅ | 基础翻转 |
| 随机旋转 ±30° | ✅ | 适应不同拍摄角度 |
| RandomResizedCrop | ✅ | 尺度不变性 |
| 亮度-对比度 | ✅ | 光照变化适应 |
| 色调-饱和度-明度 | ✅ | 颜色变化鲁棒 |
| 高斯模糊 / 噪声 | 中高 | 传感器噪声模拟 |
| CoarseDropout | ✅ | 遮挡鲁棒性 |
| 透视变换 | 高 | 视角变化 |

> `--aug-strength` 参数控制全局增强强度，在验证集过拟合时调高、欠拟合时调低。

---

## 🏆 比赛评分策略

比赛评分 = **Macro F1 × 100**，同分时比较：
1. 模型推理时间（越短越好）
2. 代码执行效率
3. 代码规范性

本项目针对策略：
- **精度**：EfficientNet-B3 在 ImageNet 上 Top-1 81.6%（4 分类远简单于 ImageNet，预期 Macro F1 ≥ 0.95）
- **速度**：ONNX Runtime CPU 推理，单张约 40-50ms，70 分钟可跑 8 万张以上（远超评分集几千张）
- **稳定性**：TTA 可选，推理时间仍充裕

---

## 🔗 与 AstralLight 农业机器人的技术复用

| RAICOM 比赛模块 | AstralLight 农业复用 |
|----------------|---------------------|
| PyTorch 训练框架 | YOLOv8 训练模板 |
| Albumentations 增强 pipeline | 农业数据集增强（直接复用） |
| ONNX Runtime CPU 推理 | Jetson 边缘推理 |
| Macro F1 评估 | 作物成熟度分类评估 |

---

## 📝 参考

- [睿抗官网](https://www.raicom.com.cn/)
- [智海Mo平台](https://momodel.cn/)
- 比赛 QQ 群：**603641284**（验证：学校+姓名）
