"""
RAICOM 2026 — 智海算法调优赛
训练工具函数

包含：
- Macro F1 评分（与比赛一致）
- 学习率调度器（Cosine + Warmup）
- 早停
- 指标追踪
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════
# 评分指标
# ════════════════════════════════════════════════════════

def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    计算 Macro F1 分数（比赛评分标准）

    对应比赛评分细则：最终得分 = F1分数 × 100
    这里的 F1 就是 macro-averaged F1
    """
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    计算所有评估指标

    返回:
        {
            "macro_f1": float,
            "accuracy": float,
            "per_class_f1": np.ndarray,
            "report": str (文本报告，可选)
        }
    """
    metrics = {
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": f1_score(y_true, y_pred, average=None),
    }

    if class_names:
        metrics["report"] = classification_report(
            y_true, y_pred, target_names=class_names, digits=4
        )

    return metrics


# ════════════════════════════════════════════════════════
# 损失函数
# ════════════════════════════════════════════════════════

def create_criterion(
    num_classes: int,
    label_smoothing: float = 0.1,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    创建损失函数

    使用 Label Smoothing Cross Entropy
    如果提供了 class_weights，则加权
    """
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


# ════════════════════════════════════════════════════════
# Mixup / CutMix
# ════════════════════════════════════════════════════════

def rand_bbox(size, lam):
    """CutMix: 随机裁剪区域"""
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix(images, labels, num_classes, alpha=0.2, cutmix_prob=0.5):
    """
    对一个 batch 随机应用 Mixup 或 CutMix

    Args:
        images: [B, C, H, W]
        labels: [B] (整数标签)
        num_classes: 类别数
        alpha: Beta 分布参数（越大混合越强）
        cutmix_prob: 使用 CutMix 的概率（否则用 Mixup）

    Returns:
        mixed_images, soft_labels (one-hot, float)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    # one-hot 标签
    labels_onehot = F.one_hot(labels, num_classes).float()

    if np.random.rand() < cutmix_prob:
        # CutMix
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images_mixed = images.clone()
        images_mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        # 按面积调整 lambda
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        soft_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
    else:
        # Mixup
        images_mixed = lam * images + (1 - lam) * images[index]
        soft_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]

    return images_mixed, soft_labels


# ════════════════════════════════════════════════════════
# 学习率调度器
# ════════════════════════════════════════════════════════

class CosineWarmupLR(_LRScheduler):
    """
    Cosine Annealing + Linear Warmup

    前 warmup_epochs 轮线性增长到 lr
    然后 cosine 衰减到 lr_min
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int = 5,
        lr_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        # Warmup 阶段：线性上升
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]

        # Cosine 衰减阶段
        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.lr_min + (base_lr - self.lr_min) * cosine_decay
            for base_lr in self.base_lrs
        ]


# ════════════════════════════════════════════════════════
# 早停
# ════════════════════════════════════════════════════════

class EarlyStopping:
    """
    早停：监控验证集指标，连续 patience 轮不改善则停止

    支持两种模式：
    - "max"：指标越大越好（如 F1, accuracy）
    - "min"：指标越小越好（如 loss）
    """

    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 1e-4,
        verbose: bool = True,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "max":
            self.best_score = float("-inf")
            self.improve_fn = lambda curr, best: curr > best + min_delta
        elif mode == "min":
            self.best_score = float("inf")
            self.improve_fn = lambda curr, best: curr < best - min_delta
        else:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")

    def __call__(self, score: float) -> bool:
        """返回 True 表示应停止"""
        if self.improve_fn(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


# ════════════════════════════════════════════════════════
# 训练步骤
# ════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    log_freq: int = 50,
    monitor=None,
    mixup_alpha: float = 0.0,
    cutmix_prob: float = 0.5,
    num_classes: int = 4,
) -> Dict:
    """训练一个 epoch（支持 Mixup/CutMix）"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = loader
    if log_freq > 0:
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"Epoch {epoch:2d} [Train]", leave=False)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup / CutMix
        use_mix = mixup_alpha > 0 and np.random.rand() < 0.8  # 80% 概率启用
        if use_mix:
            images, soft_labels = mixup_cutmix(
                images, labels, num_classes,
                alpha=mixup_alpha, cutmix_prob=cutmix_prob,
            )

        # 混合精度前向
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(images)
            if use_mix:
                # 软标签交叉熵
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(soft_labels * log_probs).sum(dim=1).mean()
            else:
                loss = criterion(logits, labels)

        # 反向
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # 统计
        total_loss += loss.item()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels_np.tolist())

        if log_freq > 0 and isinstance(pbar, object):
            try:
                postfix = {"loss": f"{loss.item():.4f}"}
                if monitor is not None and (step % max(1, log_freq) == 0):
                    snap = monitor.snapshot()
                    postfix["cpu"] = f"{snap.cpu_percent:.0f}%"
                    postfix["rss"] = f"{snap.process_rss_gb:.2f}G"
                    if snap.torch_reserved_mb is not None:
                        postfix["cuda"] = f"{snap.torch_reserved_mb:.0f}M"
                pbar.set_postfix(postfix)
            except Exception:
                pass

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds)
    )

    return {
        "loss": avg_loss,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    monitor=None,
) -> Dict:
    """验证"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = loader
    from tqdm import tqdm
    pbar = tqdm(loader, desc=f"Epoch {epoch:2d} [Val]  ", leave=False)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        if monitor is not None and step % 20 == 0:
            try:
                snap = monitor.snapshot()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "cpu": f"{snap.cpu_percent:.0f}%",
                    "rss": f"{snap.process_rss_gb:.2f}G",
                })
            except Exception:
                pass

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds)
    )

    return {
        "loss": avg_loss,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "per_class_f1": metrics["per_class_f1"],
        "report": metrics.get("report", ""),
    }
