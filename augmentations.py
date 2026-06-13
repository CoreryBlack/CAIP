"""
RAICOM 2026 — 智海算法调优赛
数据增强模块 (Albumentations)

增强策略说明：
- 训练时使用强增强（颜色抖动 + 几何变换 + 噪声）
- 验证/推理时仅用必要的 resize + normalize
- 增强强度通过 aug_strength 全局控制
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Optional


def get_train_transforms(image_size: int = 300, strength: float = 1.0):
    """
    训练增强 pipeline
    strength=0.0 → 轻增强（仅翻转+轻微颜色）
    strength=1.0 → 标准增强（适合大多数场景）
    strength=2.0 → 极强增强（小数据集 / 严重过拟合）
    """
    s = max(0.0, min(2.0, strength))  # 钳制到 [0, 2]

    transforms = []

    # ── 几何变换 ──
    # 水平翻转（基础，始终开启）
    transforms.append(A.HorizontalFlip(p=0.5))

    # 垂直翻转（强度 ≥ 0.3 时开启）
    if s >= 0.3:
        transforms.append(A.VerticalFlip(p=0.15 * s))

    # 随机旋转（强度 ≥ 0.2 时开启）
    if s >= 0.2:
        angle = 30 * s
        transforms.append(A.Rotate(limit=angle, p=0.5, border_mode=0))

    # 随机缩放 + 裁切（强度 ≥ 0.4 时开启）
    if s >= 0.4:
        scale_range = (1.0 - 0.15 * s, 1.0 + 0.10 * s)
        transforms.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0) if s <= 1.0 else (0.60, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            )
        )

    # 透视变换（高强度时开启）
    if s >= 0.8:
        transforms.append(
            A.Perspective(scale=(0.03 * s, 0.06 * s), keep_size=True, p=0.2 * s)
        )

    # ── 颜色 / 光照变换 ──
    # 亮度-对比度
    bc_limit = 0.15 * s
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=bc_limit, contrast_limit=bc_limit, p=0.5
        )
    )

    # 色调-饱和度-明度
    if s >= 0.3:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=int(10 * s),
                sat_shift_limit=int(20 * s),
                val_shift_limit=int(15 * s),
                p=0.4,
            )
        )

    # 随机 Gamma 校正（强度 ≥ 0.5）
    if s >= 0.5:
        transforms.append(A.RandomGamma(gamma_limit=(70, 130), p=0.3 * s))

    # ── 模糊 / 噪声（高强度时开启） ──
    if s >= 0.6:
        # 高斯模糊
        transforms.append(
            A.GaussianBlur(blur_limit=(3, 5), p=0.15 * s)
        )
        # 高斯噪声
        transforms.append(
            A.GaussNoise(std_range=(0.02 * s, 0.06 * s), p=0.15 * s)
        )

    # ── 图像质量退化（高强度） ──
    if s >= 0.8:
        transforms.append(A.ImageCompression(quality_range=(60, 95), p=0.1 * s))

    # ── 随机擦除 / Cutout ──
    if s >= 0.5:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, int(4 * s)),
                hole_height_range=(int(0.05 * image_size), int(0.1 * image_size)),
                hole_width_range=(int(0.05 * image_size), int(0.1 * image_size)),
                fill=0,
                p=0.3 * s,
            )
        )

    # ── 最终处理 ──
    # 先 Resize（如果 RandomResizedCrop 没生效）
    transforms.append(A.Resize(image_size, image_size))
    # 归一化（EfficientNet 标准均值标准差）
    transforms.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_val_transforms(image_size: int = 300):
    """验证 / 推理增强（仅 resize + 归一化）"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 300):
    """
    Test-Time Augmentation (TTA) 增强列表
    返回多个增强配置，推理时对每张图依次推理后取平均
    """
    # TTA 变换列表（不含 Normalize，由外部统一处理）
    tta_list = [
        # 原始
        A.Compose([A.Resize(image_size, image_size)]),
        # 水平翻转
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0)]),
        # 垂直翻转
        A.Compose([A.Resize(image_size, image_size), A.VerticalFlip(p=1.0)]),
        # 旋转 ±5°
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=5, p=1.0)]),
        # 旋转 -5°
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=-5, p=1.0)]),
    ]
    return tta_list
