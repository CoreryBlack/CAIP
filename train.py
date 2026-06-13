"""
RAICOM 2026 — 智海算法调优赛
训练主脚本

用法:
    # 默认配置训练
    python train.py

    # 指定数据路径
    python train.py --data-root ./data --epochs 60 --batch-size 64

    # 从检查点恢复
    python train.py --resume ./outputs/best_model.pth
"""

import os
import sys
import argparse
import json
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import cfg, Config
from model import create_model, load_model
from dataset import build_data_loaders
from augmentations import get_train_transforms, get_val_transforms
from train_utils import (
    CosineWarmupLR,
    EarlyStopping,
    create_criterion,
    train_one_epoch,
    validate,
)
from monitor import ResourceMonitor, ResourceLogger


def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict,
    is_best: bool,
    output_dir: str,
    filename: str = "checkpoint.pth",
):
    """保存检查点"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(output_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"🏆 保存最佳模型: {best_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAICOM 2026 — 智海算法调优赛 训练脚本"
    )
    parser.add_argument("--data-root", type=str, default=cfg.data_root,
                        help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default=cfg.output_dir,
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=cfg.epochs,
                        help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=cfg.lr,
                        help="学习率")
    parser.add_argument("--image-size", type=int, default=cfg.image_size,
                        help="输入图像尺寸")
    parser.add_argument("--aug-strength", type=float, default=cfg.aug_strength,
                        help="数据增强强度 (0.0 - 2.0)")
    parser.add_argument("--device", type=str, default=cfg.device,
                        help="训练设备 (cuda/cpu)")
    parser.add_argument("--resume", type=str, default="",
                        help="恢复训练用的检查点路径")
    parser.add_argument("--seed", type=int, default=cfg.seed,
                        help="随机种子")
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用混合精度训练")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Mixup/CutMix alpha（0=禁用，0.2=推荐）")
    parser.add_argument("--cutmix-prob", type=float, default=0.5,
                        help="CutMix 概率（其余为 Mixup）")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                        help="前 N 个 epoch 冻结 backbone，仅训练分类头")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 更新配置 ──
    cfg.data_root = args.data_root
    # 自动检测数据结构：如果 data_root/train 不存在但 class 目录存在，则 train=data_root
    potential_train = os.path.join(cfg.data_root, "train")
    if os.path.isdir(potential_train):
        cfg.train_dir = potential_train
    elif any(os.path.isdir(os.path.join(cfg.data_root, c)) for c in cfg.classes):
        cfg.train_dir = cfg.data_root
    else:
        cfg.train_dir = potential_train
    cfg.val_dir = os.path.join(cfg.data_root, "val")
    cfg.output_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.image_size = args.image_size
    cfg.aug_strength = args.aug_strength
    cfg.device = args.device
    cfg.seed = args.seed
    if args.no_amp:
        cfg.use_amp = False

    set_seed(cfg.seed)

    # ── 设备 ──
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，回退到 CPU")
        device = torch.device("cpu")
    print(f"🔧 使用设备: {device}")

    monitor = ResourceMonitor(device=device.type)
    resource_logger = ResourceLogger(output_dir=cfg.output_dir, run_name="train")
    boot_snap = monitor.snapshot()
    monitor.print_snapshot(prefix="[BOOT]")
    resource_logger.log(boot_snap, stage="boot")

    # ── 数据增强 ──
    train_transform = get_train_transforms(
        image_size=cfg.image_size,
        strength=cfg.aug_strength,
    )
    val_transform = get_val_transforms(image_size=cfg.image_size)

    # ── DataLoader ──
    print("📂 加载数据...")
    train_loader, val_loader, test_loader = build_data_loaders(
        cfg, train_transform, val_transform,
    )
    print(f"   训练集: {len(train_loader.dataset)} 张")
    print(f"   验证集: {len(val_loader.dataset)} 张")
    if test_loader:
        print(f"   测试集: {len(test_loader.dataset)} 张")

    # ── 模型 ──
    print("🧠 创建模型...")
    if args.resume and os.path.isfile(args.resume):
        print(f"   从检查点恢复: {args.resume}")
        model = load_model(args.resume, cfg)
        start_epoch = 0
        best_f1 = 0.0
    else:
        model = create_model(cfg)
        start_epoch = 0
        best_f1 = 0.0
    model = model.to(device)

    # ── 损失函数 ──
    criterion = create_criterion(
        num_classes=cfg.num_classes,
        label_smoothing=cfg.label_smoothing,
    )

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ── 学习率调度器 ──
    scheduler = CosineWarmupLR(
        optimizer,
        total_epochs=cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
        lr_min=cfg.lr_min,
    )

    # ── 混合精度 ──
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    # ── 早停 ──
    early_stopping = EarlyStopping(
        patience=cfg.early_stop_patience,
        mode="max",
    )

    # ── 日志 ──
    log_path = os.path.join(cfg.output_dir, "training_log.json")
    history = []
    print(f"📝 日志保存至: {cfg.output_dir}")
    print(f"📈 资源日志(JSONL): {resource_logger.jsonl_path}")
    print(f"📈 资源日志(CSV):   {resource_logger.csv_path}")

    # ── 训练循环 ──
    print(f"\n{'='*60}")
    print(f"开始训练  |  模型: {cfg.model_name}  |  类别: {cfg.num_classes}")
    if args.mixup_alpha > 0:
        print(f"Mixup/CutMix  |  alpha={args.mixup_alpha}  |  cutmix_prob={args.cutmix_prob}")
    print(f"{'='*60}")
    print(f"[RESOURCE] {monitor.compact()}")

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()

        # ── 冻结/解冻 backbone ──
        if args.freeze_backbone_epochs > 0:
            if epoch < args.freeze_backbone_epochs:
                if hasattr(model, 'freeze_backbone'):
                    model.freeze_backbone()
            elif epoch == args.freeze_backbone_epochs:
                if hasattr(model, 'unfreeze_backbone'):
                    model.unfreeze_backbone()
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
                scheduler = CosineWarmupLR(
                    optimizer,
                    total_epochs=cfg.epochs,
                    warmup_epochs=cfg.warmup_epochs,
                    lr_min=cfg.lr_min,
                )
                print(f"🔓 已在 epoch {epoch+1} 解冻 backbone，切换为全量训练")

        # ── 训练 ──
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, cfg.log_freq, monitor=monitor,
            mixup_alpha=args.mixup_alpha, cutmix_prob=args.cutmix_prob,
            num_classes=cfg.num_classes,
        )

        # ── 验证 ──
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, monitor=monitor,
        )

        # ── 学习率 ──
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ── 日志 ──
        epoch_time = time.time() - epoch_start
        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["macro_f1"]

        snap = monitor.snapshot()
        resource_logger.log(
            snap,
            stage="epoch_end",
            epoch=epoch + 1,
            extra={
                "epoch_time_sec": round(epoch_time, 3),
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["macro_f1"],
            },
        )
        log_entry = {
            "epoch": epoch + 1,
            "time": round(epoch_time, 1),
            "lr": round(current_lr, 8),
            "train_loss": round(train_metrics["loss"], 4),
            "train_f1": round(train_metrics["macro_f1"], 4),
            "train_acc": round(train_metrics["accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_f1": round(val_metrics["macro_f1"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "resource": snap.to_dict(),
        }
        history.append(log_entry)

        print(
            f"Epoch {epoch+1:2d}/{cfg.epochs} | "
            f"LR {current_lr:.2e} | "
            f"Train F1 {train_metrics['macro_f1']:.4f} | "
            f"Val F1 {val_metrics['macro_f1']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Best {best_f1:.4f} | "
            f"{epoch_time:.0f}s | {monitor.compact(snap)}"
        )

        # ── 保存检查点 ──
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "config": cfg,
            },
            is_best=is_best,
            output_dir=cfg.output_dir,
        )

        # ── 早停检查 ──
        if early_stopping(val_metrics["macro_f1"]):
            print(f"\n🛑 早停触发！连续 {cfg.early_stop_patience} 轮未改善。")
            break

    # ── 保存日志 ──
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    monitor.close()

    print(f"\n{'='*60}")
    print(f"🎉 训练完成！最佳 Macro F1: {best_f1:.4f}")
    print(f"   模型路径: {os.path.join(cfg.output_dir, 'best_model.pth')}")
    print(f"   日志路径: {log_path}")
    print(f"   资源日志(JSONL): {resource_logger.jsonl_path}")
    print(f"   资源日志(CSV):   {resource_logger.csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
