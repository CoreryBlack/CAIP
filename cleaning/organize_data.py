"""
RAICOM 2026 - 数据整理与标注脚本

将 data_raw_weather4class/ 中的原始数据按类别映射整理到 data/ 目录，
同时按 8:2 划分训练集和验证集，排除 Pascal VOC / YOLO 等非图片文件。

类别映射:
    Fog  -> cloudy   (雾天 -> 多云)
    Rain -> rainy    (雨天 -> 雨天)
    Sand -> sunny    (沙尘 -> 晴天)
    Snow -> snowy    (雪天 -> 雪天)
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# 类别映射表（数据原始类名 -> 项目类名）
CLASS_MAPPING = {
    "Fog": "cloudy",
    "Rain": "rainy",
    "Sand": "sunny",
    "Snow": "snowy",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_image_files(source_dir: str) -> list:
    """获取目录下所有图片文件（排除子目录）"""
    images = []
    for fname in os.listdir(source_dir):
        fpath = os.path.join(source_dir, fname)
        # 跳过子目录
        if os.path.isdir(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in VALID_EXTS:
            images.append(fpath)
    return sorted(images)


def organize_data(raw_root: str, target_root: str, val_split: float = 0.2, seed: int = 42):
    """整理数据：按类别映射复制图片，并划分 train/val"""
    random.seed(seed)
    raw_root = Path(raw_root)
    target_root = Path(target_root)

    print("=" * 60)
    print(" [数据整理] 原始数据: %s" % raw_root)
    print(" [数据整理] 目标路径: %s" % target_root)
    print(" [数据整理] 验证集比例: %.1f" % val_split)
    print("=" * 60)

    total_train = 0
    total_val = 0

    for raw_class, target_class in CLASS_MAPPING.items():
        source_dir = raw_root / raw_class
        if not source_dir.is_dir():
            print("   原始文件夹不存在: %s，跳过" % source_dir)
            continue

        # 获取所有图片
        images = get_image_files(str(source_dir))
        print("\n  [%s -> %s]  找到 %d 张图片" % (raw_class, target_class, len(images)))

        if len(images) == 0:
            continue

        # 随机打乱并切分
        random.shuffle(images)
        val_count = max(1, int(len(images) * val_split))
        train_images = images[val_count:]
        val_images = images[:val_count]

        # 创建目标目录
        train_dir = target_root / "train" / target_class
        val_dir = target_root / "val" / target_class
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # 复制训练集
        for src in train_images:
            fname = os.path.basename(src)
            dst = train_dir / fname
            # 处理文件名冲突
            if dst.exists():
                stem, ext = os.path.splitext(fname)
                dst = train_dir / "%s_%s%s" % (stem, raw_class, ext)
            shutil.copy2(src, str(dst))

        # 复制验证集
        for src in val_images:
            fname = os.path.basename(src)
            dst = val_dir / fname
            if dst.exists():
                stem, ext = os.path.splitext(fname)
                dst = val_dir / "%s_%s%s" % (stem, raw_class, ext)
            shutil.copy2(src, str(dst))

        print("    -> train/%s/: %d 张" % (target_class, len(train_images)))
        print("    -> val/%s/:   %d 张" % (target_class, len(val_images)))

        total_train += len(train_images)
        total_val += len(val_images)

    print("\n" + "=" * 60)
    print(" 整理完成!")
    print("   训练集: %d 张" % total_train)
    print("   验证集: %d 张" % total_val)
    print("   总计:   %d 张" % (total_train + total_val))
    print("=" * 60)

    # 打印最终目录结构
    print("\n 生成的文件结构:")
    for split_name in ["train", "val"]:
        split_dir = target_root / split_name
        if split_dir.is_dir():
            print("  data/%s/" % split_name)
            for class_dir in sorted(os.listdir(str(split_dir))):
                class_path = split_dir / class_dir
                if class_path.is_dir():
                    count = len(list(class_path.glob("*.*")))
                    print("    +-- %s/  (%d 张)" % (class_dir, count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="整理天气分类数据集")
    parser.add_argument("--raw-root", type=str,
                        default="./data_raw_weather4class",
                        help="原始数据根目录")
    parser.add_argument("--target-root", type=str,
                        default="./data",
                        help="目标数据根目录")
    parser.add_argument("--val-split", type=float,
                        default=0.2,
                        help="验证集比例")
    parser.add_argument("--seed", type=int,
                        default=42,
                        help="随机种子")
    args = parser.parse_args()

    organize_data(
        raw_root=args.raw_root,
        target_root=args.target_root,
        val_split=args.val_split,
        seed=args.seed,
    )
