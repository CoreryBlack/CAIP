# 结合文献，对 RAICOM 天气分类项目的具体建议

> 基于 CVPR 2019 (RandAugment), ICLR 2018 (Mixup), NeurIPS 2020 (Balanced Softmax) 的实验结论与工程实践

---

## 1. 数据增强：从“手动调参”转向“自动化搜索”

**现状：** 
当前 `augmentations.py` 使用了大量手工设定的增强（翻转、旋转、抖动、模糊等），参数由人工经验设定，难以保证最优。

**文献建议 (RandAugment - CVPR 2019):**
- **策略:** 不再手动调，使用 `RandAugment`。它只定义两个超参数：`N`（应用几次变换）和 `M`（变换的强度，0-10）。
- **收益:** 在多个任务上超越手动调参和 AutoAugment，且搜索空间极小，训练时只需几分钟即可找到最优 `N` 和 `M`。
- **实施建议:**
  ```python
  # 使用 torchvision 自带的 RandAugment，替换手工增强链
  from torchvision import transforms

  transform_train = transforms.Compose([
      transforms.RandomResizedCrop(300),
      transforms.RandomHorizontalFlip(),
      transforms.RandAugment(num_ops=2, magnitude=9), # 关键参数：N=2, M=9
      transforms.ToTensor(),
      transforms.Normalize(...)
  ])
  ```
  - 建议先用 `N=2, M=9` 开始，然后在 `[0, 30]` 范围内微调 `M`。

---

## 2. 损失函数：替换过时的“加权采样”为“无偏估计”

**现状：** 
当前 `dataset.py` 使用 `WeightedRandomSampler` 平衡样本，`train_utils.py` 使用 `CrossEntropyLoss`。
**问题:** 传统 Softmax 在长尾数据（类别不平衡）下会导致梯度偏差，少样本类别的权重更新不足，模型容易“偏向”多样本类别。

**文献建议 (Balanced Softmax - NeurIPS 2020):**
- **策略:** 替换输出层的 Softmax 为 Balanced Softmax。它直接在损失函数中引入类别频率的修正项，使梯度估计无偏。
- **收益:** 比单独的重采样（Sampler）或 Focal Loss 更有效，能显著提升少数类别的准确率。
- **实施建议:**
  在 `train_utils.py` 中添加 Balanced Softmax 损失：
  ```python
  class BalancedSoftmaxLoss(nn.Module):
      def __init__(self, cls_num_list):
          super().__init__()
          cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
          cls_num_list = cls_num_list / cls_num_list.sum()
          self.weight = torch.log(cls_num_list) # 关键修正项
      
      def forward(self, logits, targets):
          # 核心：在 logits 上加上类别频率的 log 值
          adjusted_logits = logits + self.weight.to(logits.device)
          loss = F.cross_entropy(adjusted_logits, targets)
          return loss
  ```
  - **注意:** 使用此方法时，建议去掉 `WeightedRandomSampler`，或将其强度降低，避免过度补偿。

---

## 3. 正则化：引入 Mixup / CutMix 提升泛化

**现状：** 
当前训练使用标准 ERM (经验风险最小化)。

**文献建议 (Mixup - ICLR 2018):**
- **策略:** 在每批样本中，随机挑选两张图片进行线性插值：$x_{new} = \lambda x_i + (1-\lambda) x_j$，标签也做相应插值。
- **收益:** 让模型学习更平滑的边界，减少对抗样本的敏感度，防止过拟合。
- **实施建议:**
  在 `train.py` 的循环中加入 Mixup：
  ```python
  alpha = 0.2  # 推荐值：0.2
  lam = np.random.beta(alpha, alpha)
  indices = torch.randperm(images.size(0))
  
  mixed_images = lam * images + (1 - lam) * images[indices]
  # 对于分类任务，通常还是计算硬标签的 CrossEntropy，或者使用软标签计算 KL 散度
  ```
  - **CutMix** 是 Mixup 的进阶版（切块拼贴），建议在 Mixup 基础上再尝试。

---

## 4. 迁移学习：解冻策略与学习率分离

**现状：** 
当前直接对整个模型进行训练。

**文献建议 (EfficientNet - ICML 2019):**
- **策略:** 预训练模型应采取“小 LR + 深层冻结”策略。
- **实施建议:**
  ```python
  # 冻结 backbone 前几层
  for param in list(model.backbone.parameters())[:-20]: # 只解冻最后几层
      param.requires_grad = False
      
  # 分层学习率
  optimizer = torch.optim.AdamW([
      {'params': model.backbone.parameters(), 'lr': 1e-4},  # 主干小 LR
      {'params': model.classifier.parameters(), 'lr': 1e-3}, # 头部大 LR
  ])
  ```

---

## 5. 速度优化：INT8 量化与图优化

**现状：** 
导出 ONNX 后直接使用 `onnxruntime`。

**文献建议 (Quantization - CVPR 2018):**
- **策略:** 使用 ONNX Runtime 的静态/动态量化将模型权重从 FP32 压缩到 INT8。
- **收益:** 模型大小减小 75%，CPU 推理速度提升 2-4 倍。
- **实施建议:**
  在 `export_onnx.py` 导出后，使用 `onnxruntime.quantization.quantize` 进行量化。

---

## 总结：下一步实验清单

| 优先级 | 任务 | 预期收益 | 风险 |
|--------|------|---------|------|
| P0 | 替换增强为 `RandAugment` | 提升 1-2% 准确率，减少过拟合 | 几乎无 |
| P1 | 损失函数换为 `Balanced Softmax` | 解决类间不平衡，提升 macro F1 | 需调整类别权重 |
| P2 | 加入 `Mixup` | 提升模型鲁棒性 | 训练速度稍慢 |
| P3 | 分层学习率 | 训练更稳，避免灾难性遗忘 | 无 |
| P4 | ONNX INT8 量化 | 提速 2-4 倍 | 精度微跌（通常 <0.5%） |
