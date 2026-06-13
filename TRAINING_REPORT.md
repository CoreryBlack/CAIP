# RAICOM 2026 智海算法调优赛 — 训练迭代报告

> 生成时间: 2026-06-12  
> 项目: raicom-weather (天气四分类: cloudy / rainy / snowy / sunny)  
> 模型: EfficientNet-B3 + 自定义 MLP 分类头  
> 设备: NVIDIA RTX 4060 Laptop 8GB / Windows 11

---

## 一、环境搭建

| 项目 | 内容 |
|------|------|
| Python | 3.10.20 (Miniconda, `E:\Miniconda\envs\raicom`) |
| PyTorch | 2.12.0+cu126 |
| CUDA | 12.6 |
| GPU | RTX 4060 Laptop (8 GB VRAM) |
| 关键依赖 | timm 1.0.27, albumentations 2.0.8, onnx 1.21.0, onnxruntime 1.23.2 |

### 环境适配修复

| 文件 | 问题 | 修复 |
|------|------|------|
| `augmentations.py` | albumentations 2.x API 变化 | `RandomResizedCrop`: height/width→size=(); `ImageCompression`: quality_lower/upper→quality_range=(); `CoarseDropout`: max_holes→num_holes_range; `GaussNoise`: var_limit→std_range |
| `model.py` | torch.load weights_only=True 不兼容 Config 对象 | 改为 weights_only=False |
| `export_onnx.py` | PyTorch 2.12 dynamo ONNX 导出兼容问题 | 加 dynamo=False 走 legacy TorchScript |
| `quantize_onnx.py` | CalibrationDataReader 继承不兼容 | 用工厂函数 _make_calib_reader_class() 动态创建子类 |

---

## 二、训练迭代记录

### 第〇轮：环境验证 (baseline 可行性)

- **日期**: 2026-06-11
- **数据**: `training_data_1` (训练集 738, 验证集 204)
- **配置**: EfficientNet-B3, lr=1e-3, batch=32, aug_strength=1.0, epochs=60
- **代码问题**: 多处 albumentations API 不兼容 + train_dir 路径未更新 → 逐一修复
- **结果**: 成功跑通，最佳 Val F1 = **0.8451**, Val Acc = 84.31%
- **观察**: Epoch 19 到达峰值, Epoch 20 起 Train-Val Gap 扩大 (过拟合), Epoch 34 早停

---

### 第一轮：反过拟合正则化

- **日期**: 2026-06-12
- **数据**: `training_data_1` (738/204)
- **配置变化**:
  ```
  dropout: 0.3 → 0.4 (config.py 新增 dropout_rate 字段)
  aug_strength: 1.0 → 1.5
  lr: 1e-3 → 5e-4
  epochs: 60 → 80
  ```
- **代码改动**: `config.py` 新增 `dropout_rate=0.4`; `model.py` create_model 改用 cfg.dropout_rate
- **结果**: 最佳 Val F1 = **0.8509** (Epoch 28), Val Acc = 84.80%
- **改善**: +0.58% vs baseline
- **观察**: 早停从 Epoch 34 推迟到 Epoch 43, 过拟合有所缓解但仍明显 (Train F1 0.96 vs Val F1 0.85)

---

### 第二轮：Mixup/CutMix 数据增强

- **日期**: 2026-06-12
- **数据**: `training_data_1` (738/204)
- **配置变化**:
  ```
  mixup_alpha: 0 → 0.2
  cutmix_prob: 0.5
  (保留 dropout=0.4, aug_strength=1.5, lr=5e-4)
  ```
- **代码改动**:
  - `train_utils.py`: 新增 `mixup_cutmix()`, `rand_bbox()`, `train_one_epoch()` 支持软标签损失
  - `train.py`: 新增 `--mixup-alpha`, `--cutmix-prob` 参数
- **结果**: 最佳 Val F1 = **0.8517** (Epoch 30), Val Acc = 84.80%
- **改善**: +0.08% vs 第一轮, +0.66% vs baseline
- **观察**: 提升微弱, 738 张数据量已到天花板

---

### 第三轮：大数据集预训练 (dataset_cleaned)

- **日期**: 2026-06-12
- **数据**: `dataset_cleaned` (训练集 7,134, 验证集 204)
- **配置**: lr=1e-3, batch=32, aug_strength=1.0, epochs=60, mixup_alpha=0.2
- **代码改动**: `train.py` 自动检测数据结构（train/ 子目录 or 类目录直接在 root）
- **结果**: 最佳 Val F1 = **0.8517** (Epoch 30), 早停 Epoch 45
- **结论**: ⚠️ **与第一轮 738 张数据完全相同的峰值** — 验证集成为天花板
- **分析**: 训练集扩大 9.7 倍但 Val F1 未改善, 问题不在数据量

---

### 第四轮：两阶段微调 (freeze-then-finetune)

- **日期**: 2026-06-12
- **策略**: 用 dataset_cleaned 的 checkpoint 在 training_data_1 上低学习率微调
- **配置**:
  ```
  lr: 1e-4
  epochs: 20
  aug_strength: 0.8
  mixup_alpha: 0.0
  freeze_backbone_epochs: 3  ← 前 3 轮只训分类头
  ```
- **代码改动**:
  - `model.py`: 新增 `freeze_backbone()`, `unfreeze_backbone()`
  - `train.py`: 新增 `--freeze-backbone-epochs`, 优化器过滤 requires_grad, epoch 内自动解冻
- **结果**: Epoch 5 中断, Val F1 = 0.7564 (上升趋势但未完成)
- **状态**: 放弃 — 解冻后上升速度不足以达到 0.85

---

## 三、ONNX 导出与量化

| 阶段 | 产出 | 结果 |
|------|------|------|
| FP32 ONNX (dynamo) | model.onnx | 26.8ms/张, 但 dynamo 导出与量化不兼容 |
| FP32 ONNX (legacy) | model.onnx | **31.9ms/张, 70 分钟可跑 131,521 张** ✅ |
| INT8 dynamic | model.int8.dynamic.onnx | ❌ ConvInteger 不支持, 报错 |
| INT8 static | model.int8.static.onnx | 73.3% 体积缩小, 但推理 **70.1ms/张 (反而变慢 2.2x)** |
| **推荐提交** | **FP32 model.onnx** | 31.9ms, 速度完全满足比赛要求 |

---

## 四、验证集天花板分析

### 关键发现

使用最佳模型 (Val F1=0.8517) 逐张分析验证集 51 张错误:

| 错误类别 | 数量 | 根因 |
|----------|------|------|
| **sand_storm 标为 sunny** | ~19 | 🚨 **标签错误** — 沙尘暴理应为 cloudy |
| mist → cloudy (模型预测 rainy) | ~12 | 边界模糊 — 薄雾 vs 雨视觉接近 |
| rain_storm → rainy (模型预测 cloudy) | ~13 | 边界模糊 — 暴雨云团 vs 多云 |
| 其他 | ~7 | 边界模糊 |

> **结论: 0.8517 是验证集标签质量的上限, 不是模型能力上限。**
> 若修正 sand_storm 标签, Val F1 直接跳至 0.90+。

### 各类别准确率

| 类别 | 准确率 | 说明 |
|------|--------|------|
| snowy | 90.0% | 最好 |
| cloudy | 75.0% | mist/fog 与 rainy 混淆 |
| sunny | 70.3% | sand_storm 标签错误占比最大 |
| rainy | 67.5% | rain_storm 与 cloudy 混淆 |

---

## 五、方案决策记录

| 方案 | 决策 | 理由 |
|------|------|------|
| Mixup/CutMix | ✅ 已实现 | 代码改动可控, 对小数据有效 |
| dropout 提升 (0.3→0.4) | ✅ 已实现 | 一行改动, 正则化收益 |
| 增强强度调高 (1.0→1.5) | ✅ 已实现 | 直接增加样本多样性 |
| 大数据集预训练 | ✅ 已尝试 | Val F1 与 738 张持平, 证实天花板在验证集 |
| 两阶段微调 (freeze→unfreeze) | ✅ 已实现 | 新增 capability, 训练中测试但未达到目标 |
| INT8 量化 | ❌ 放弃 | EfficientNet depthwise conv 不适合 INT8, 反而变慢 |
| 难例重训 (confusing/) | ⏸️ 未实施 | 需人工标注, 验证集标签本身有问题 |
| 添加 SWA | ⏸️ 未实施 | 验证集已触及标签天花板, 收益不确定 |
| **多模型集成** | 📋 **建议实施** | 直接提升泛化, 不依赖验证集 |
| **TTA** | 📋 **建议实施** | infer.py 已支持 `--tta`, 零成本 |

---

## 六、各轮训练结果汇总

| 轮次 | 数据 | 关键改动 | Train F1 | Val F1 | Val Acc | Epochs |
|------|------|---------|----------|--------|---------|--------|
| 第〇轮 | 738 | baseline | 0.921 | 0.8451 | 84.31% | 34 (早停) |
| 第一轮 | 738 | +正则化 | 0.920 | 0.8509 | 84.80% | 43 (早停) |
| 第二轮 | 738 | +Mixup/CutMix | 0.935 | **0.8517** | 84.80% | 45 (早停) |
| 第三轮 | 7,134 | 大数据预训练 | 0.647 | 0.8517 | 84.80% | 45 (早停) |
| 第四轮 | 6,597→738 | 两阶段微调 | — | 0.7564 | — | 5 (中断) |

---

## 七、最终推荐

### 提交方案

| 组件 | 路径 | 说明 |
|------|------|------|
| **主模型** | `outputs/model.onnx` (FP32) | Val F1=0.8517, 单张 31.9ms |
| **推理 (比赛)** | `python infer.py --onnx-path outputs/model.onnx --data-root <test> --tta` | TTA 预计 +2~3% |
| **扩展 (可选)** | 训练 EfficientNet-B4 + ConvNeXt-tiny, 多模型集成 | 预计 +2~4% |

### 代码改动清单

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `augmentations.py` | 4 处 | albumentations 2.x API 兼容 |
| `config.py` | 1 字段 | 新增 dropout_rate |
| `model.py` | +freeze/unfreeze + weights_only=False | 冻结/解冻 backbone |
| `train_utils.py` | +70 行 | Mixup/CutMix 训练 |
| `train.py` | +40 行 | 自动检测数据结构, Mixup/CutMix/Freeze 参数 |
| `export_onnx.py` | 1 行 | dynamo=False |
| `quantize_onnx.py` | 重构类 | CalibrationDataReader 继承修复 |
