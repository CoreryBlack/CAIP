"""
RAICOM 2026 — 智海算法调优赛
数据集模块

支持两种模式：
1. 按文件夹结构加载：data/train/cloudy/, data/train/rainy/, ...
2. CSV 标注文件加载：filename, label
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Callable, Optional, List, Tuple
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path


class WeatherDataset(Dataset):
    """
    天气分类数据集

    期望目录结构：
        data_root/
        ├── train/
        │   ├── cloudy/     (多云)
        │   ├── rainy/      (雨天)
        │   ├── snowy/      (雪天)
        │   └── sunny/      (晴天)
        ├── val/            (可选)
        │   └── ... (同上)
        └── test/           (可选，推理用)
            └── ... (图片直接放在 test/ 下)

    或 CSV 格式：
        data_root/train.csv 包含列: filename, label (label 为类别名称)
        data_root/test.csv  包含列: filename
    """

    def __init__(
        self,
        root_dir: str,
        class_to_idx: dict,
        transform: Optional[Callable] = None,
        is_test: bool = False,
        image_size: int = 300,
    ):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.is_test = is_test
        self.image_size = image_size
        self.samples: List[Tuple[str, int]] = []  # (image_path, label)
        self._load_samples()

    def _load_samples(self):
        """自动检测目录结构或 CSV 文件加载"""
        # 优先尝试 CSV
        csv_path = os.path.join(self.root_dir, "train.csv")
        if os.path.exists(csv_path):
            self._load_from_csv(csv_path)
            return

        # 尝试 test.csv
        csv_path = os.path.join(self.root_dir, "test.csv")
        if self.is_test and os.path.exists(csv_path):
            self._load_test_from_csv(csv_path)
            return

        # 按文件夹结构加载
        self._load_from_folders()

    def _load_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(os.path.dirname(csv_path), row["filename"])
            label_name = str(row["label"]).strip()
            if label_name in self.class_to_idx:
                self.samples.append((img_path, self.class_to_idx[label_name]))

    def _load_test_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        base_dir = os.path.dirname(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(base_dir, row["filename"])
            self.samples.append((img_path, -1))  # label = -1 表示测试集

    def _load_from_folders(self):
        """按 class_name/ 子文件夹结构遍历"""
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.root_dir}")

        for class_name in sorted(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # 标注未知类别时跳过（用于测试集）
            label = self.class_to_idx.get(class_name, -1)
            if label == -1 and not self.is_test:
                # 如果不是测试集且类别未知，跳过
                continue

            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            for fname in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            # 降级到 PIL
            from PIL import Image
            img_pil = Image.open(img_path).convert("RGB")
            img = np.array(img_pil)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 应用增强
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label


def build_data_loaders(
    cfg,
    train_transform,
    val_transform,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    构建 train / val / (可选) test DataLoader

    返回: (train_loader, val_loader, test_loader)
    """
    class_to_idx = {name: i for i, name in enumerate(cfg.classes)}

    # ── 训练集 ──
    train_dataset = WeatherDataset(
        root_dir=cfg.train_dir,
        class_to_idx=class_to_idx,
        transform=train_transform,
        is_test=False,
    )

    # ── 从训练集切分验证集 ──
    # 如果有独立的 val 目录则直接使用
    if os.path.isdir(cfg.val_dir) and len(os.listdir(cfg.val_dir)) > 0:
        val_dataset = WeatherDataset(
            root_dir=cfg.val_dir,
            class_to_idx=class_to_idx,
            transform=val_transform,
            is_test=False,
        )
        # 训练集保持完整
        train_dataset_for_loader = train_dataset
    else:
        # 从训练集按比例切分
        labels = [s[1] for s in train_dataset.samples]
        train_indices, val_indices = train_test_split(
            np.arange(len(train_dataset)),
            test_size=cfg.val_split,
            stratify=labels,
            random_state=cfg.seed,
        )
        train_dataset_for_loader = _SubsetDataset(train_dataset, train_indices)
        val_dataset = _SubsetDataset(train_dataset, val_indices)

    # ── 加权采样（处理类别不平衡） ──
    sampler = None
    if cfg.use_weighted_sampler:
        train_labels = [s[1] for s in train_dataset_for_loader.samples]
        class_counts = np.bincount(train_labels, minlength=cfg.num_classes)
        weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    # ── DataLoader ──
    train_loader = DataLoader(
        train_dataset_for_loader,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=4,
    )

    # ── 测试集（可选） ──
    test_loader = None
    test_dir = os.path.join(cfg.data_root, "test")
    if os.path.isdir(test_dir):
        test_dataset = WeatherDataset(
            root_dir=test_dir,
            class_to_idx=class_to_idx,
            transform=val_transform,
            is_test=True,
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.cpu_inference_batch,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )

    return train_loader, val_loader, test_loader


class _SubsetDataset(Dataset):
    """工具类：从数据集取子集"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.samples = [dataset.samples[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
