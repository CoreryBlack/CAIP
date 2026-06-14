# RAICOM 2026 智海算法调优赛 — 训练迭代报告

> 更新时间: 2026-06-15  
> 项目: raicom-weather (天气四分类: cloudy / rainy / snowy / sunny)  
> 模型: EfficientNet-B3 + 自定义 MLP 分类头  
> 设备: RTX 4060 Laptop 8GB (本地) / NVIDIA A100 40GB (远程)

---

## 一、环境搭建

| 项目 | 本地 | 远程 |
|------|------|------|
| Python | 3.10.20 | 3.12.3 |
| PyTorch | 2.12.0+cu126 | 2.12.0+cu124 |
| CUDA | 12.6 | 12.8 |
| GPU | RTX 4060 8GB | A100 40GB |

### 环境适配修复

| 问题 | 修复 |
|------|------|
| albumentations 2.x API (RandomResizedCrop/ImageCompression/CoarseDropout) | 参数名适配 |
| torch.load weights_only=True | 改为 weights_only=False |
| ONNX dynamo 导出不兼容 | dynamo=False legacy 导出 |
| CalibrationDataReader 继承 | 工厂函数动态创建子类 |

---

## 二、训练迭代记录

### 第〇~二轮：小数据集探索 (738 张)

| 轮次 | 关键改动 | Val F1 | Val Acc | Epochs |
|------|---------|--------|---------|--------|
| baseline | EfficientNet-B3, lr=1e-3 | 0.8451 | 84.31% | 34 |
| +正则化 | dropout=0.4, aug=1.5, lr=5e-4 | 0.8509 | 84.80% | 43 |
| +Mixup/CutMix | mixup_alpha=0.2 | **0.8517** | 84.80% | 45 |

> 738 张数据量触碰天花板，Train F1 0.96 vs Val F1 0.85 过拟合明显。

### 第三~四轮：大数据集预训练 + 微调

| 轮次 | 数据 | Val F1 | 结论 |
|------|------|--------|------|
| 大数据集 `dataset_cleaned` (7134) | qwen3-vl-flash 粗洗 | 0.8517 | 与 738 张持平 |
| 两阶段微调 | freeze→unfreeze | 0.7564(中断) | 放弃 |

> 数据量翻 10 倍但 F1 不变 → **验证集标签质量是天花板，不是数据量**。

---

## 三、验证集标签污染发现与修正

### 关键发现 (2026-06-13)

用最佳模型逐张分析 204 张验证集错误:

| 类型 | 数量 | 根因 |
|------|------|------|
| sand_storm/dusttornado → sunny | 63/64 | 🚨 标签错误——沙尘暴=cloudy |
| mist/foggy → cloudy (模型判 rainy) | ~12 | 边界模糊 |
| rain_storm → rainy (模型判 cloudy) | ~13 | 边界模糊 |

> 64 张 sunny 中 63 张实际是沙尘暴。仅 `sand_storm-326.jpg` 被 Qwen 认可为 sunny。

### qwen3-vl-plus 全量重洗 (2026-06-14)

| 范围 | 数量 | 移动 | 说明 |
|------|------|------|------|
| 训练集 | 9,382 | **2,275** (24%) | 主要 sunny→cloudy |
| 验证集 | 204 | **27** | sand_storm 移入 cloudy |

### 清洗后验证集 (576 张)

| 类别 | 数量 |
|------|------|
| cloudy | 189 |
| rainy | 149 |
| snowy | 102 |
| sunny | 136 |

---

## 四、多模型集成探索

### B3 大数据重训

- Val F1 = **0.8509**（与旧 738 张完全一致）

### B4 / ConvNeXT

| 模型 | 输入 | batch | Val F1 | 结论 |
|------|------|-------|--------|------|
| B4 | 380×380 | 24 | 0.67 | 分辨率过大不适合 |
| ConvNeXT | 224×224 | 48 | 0.70 | 分辨率过小丢细节 |

### B3 v3 (当前进行中)

- 配置: 300×300, batch=128, cudnn.benchmark=True, 去除 torch.compile
- 预期: Val F1 ≥ 0.85

---

## 五、训练效率优化

| 改动 | 文件 | 效果 |
|------|------|------|
| cudnn.benchmark=True | train.py | 自动选最优卷积算法 |
| persistent_workers=True | dataset.py | 跨 epoch 复用 worker |
| prefetch_factor=4 | dataset.py | GPU 不等 CPU |
| num_workers=8 | config.py | 服务器多核并行 |
| torch.compile | train.py | ❌ 与 Mixup 不兼容 → 已撤回 |
| src/docs/scripts 目录结构 | 项目重组 | 根目录从 25 项精简至 6 项 |

---

## 六、ONNX 导出

| 模型 | 速度 | 70min 可推 | 结论 |
|------|------|-----------|------|
| FP32 ONNX (legacy) | 31.9ms | 131,521 张 | ✅ 推荐提交 |
| INT8 dynamic | — | — | ❌ ConvInteger 不支持 |
| INT8 static | 70.1ms | — | ❌ 反而变慢 (EfficientNet depthwise conv) |

---

## 七、全轮训练结果汇总

| 轮次 | 数据 | 关键 | Val F1 | 状态 |
|------|------|------|--------|------|
| 1 | 738 | baseline | 0.8451 | ✅ |
| 2 | 738 | +正则化 | 0.8509 | ✅ |
| 3 | 738 | +Mixup | 0.8517 | ✅ 最高 |
| 4 | 7,134 | 大数据预训练 | 0.8517 | ✅ |
| 5 | 9,382 | qwen3-vl-plus 精洗 | — | 数据准备 |
| 6 | 9,382 | B3 重训 | 0.8509 | ✅ |
| 7 | 9,382 | B4 | 0.67 | ❌ |
| 8 | 9,382 | ConvNeXT | 0.70 | ❌ |
| 9 | 9,382 | B3+compile | 0.6668 | ❌ compile 撤掉 |
| 10 | 9,382 | B3 v3 | **待定** | 🔄 进行中 |

---

## 八、最终推荐

| 组件 | 说明 |
|------|------|
| 模型 | EfficientNet-B3, Val F1 0.8517 |
| ONNX | FP32, 31.9ms/张, 70min 可推 131K 张 |
| 推理 | `python infer.py --tta` 预计 +2~3% |
| 集成 | B3 + B4 + ConvNeXT 待架构稳定后 |

### 当前项目结构

```
raicom-weather/
├── train.py / infer.py        ← 薄入口 → src/
├── src/                       ← 核心代码 (12 个 .py)
├── docs/                      ← 文档
├── scripts/                   ← Shell
├── cleaning/                  ← 数据清洗
├── dataset_cleaned/           ← 训练 + 验证
├── outputs/                   ← 模型权重
└── references/                ← 文献
```
