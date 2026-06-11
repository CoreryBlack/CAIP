"""
RAICOM 2026 - 数据集清洗工具

针对 Fog/Rain 等易混淆类别，将图片成对展示，
方便人工判断标签是否正确。
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from glob import glob


def collect_confusing_pairs(data_root: str, class_a: str, class_b: str,
                            num_pairs: int = 50, output_dir: str = "confusing_pairs"):
    """
    从两个易混淆的类别中，配对采样图片到一个目录，方便人类对比判断

    生成: confusing_pairs/
        ├── pair_001_a_classA.jpg
        ├── pair_001_b_classB.jpg
        ├── pair_002_a_classA.jpg
        └── pair_002_b_classB.jpg
    """
    dir_a = os.path.join(data_root, "train", class_a)
    dir_b = os.path.join(data_root, "train", class_b)

    imgs_a = sorted(glob(os.path.join(dir_a, "*.*")))
    imgs_b = sorted(glob(os.path.join(dir_b, "*.*")))

    random.seed(42)
    random.shuffle(imgs_a)
    random.shuffle(imgs_b)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(min(num_pairs, len(imgs_a), len(imgs_b))):
        fname_a = "pair_%03d_a_%s_%s" % (i + 1, class_a, os.path.basename(imgs_a[i]))
        fname_b = "pair_%03d_b_%s_%s" % (i + 1, class_b, os.path.basename(imgs_b[i]))
        shutil.copy2(imgs_a[i], str(out_dir / fname_a))
        shutil.copy2(imgs_b[i], str(out_dir / fname_b))
        count += 1

    print("已生成 %d 组对比对" % count)
    print("目录: %s/" % output_dir)
    print()
    print("使用方法:")
    print("  1. 打开 each_pair 文件夹")
    print("  2. 一张一张看: 左边是 [%s], 右边是 [%s]" % (class_a, class_b))
    print("  3. 问自己: 这两张图真的属于不同的类别吗？")
    print("  4. 如果觉得标错了, 记录文件名, 后续修正")


def remove_outliers(data_root: str, class_name: str, keep_ratio: float = 0.9,
                    output_bad: str = "data_bad_labels"):
    """
    简单离群检测: 提取特征并对每个类做聚类, 挑出离群图片
    (需要 torch + timm)
    """
    try:
        import torch
        import torch.nn as nn
        import timm
        from PIL import Image
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("需要安装: pip install torch timm sklearn Pillow")
        return

    class_dir = os.path.join(data_root, "train", class_name)
    if not os.path.isdir(class_dir):
        print("目录不存在: %s" % class_dir)
        return

    img_paths = sorted(glob(os.path.join(class_dir, "*.*")))
    print("正在提取 %s 类 %d 张图片的特征..." % (class_name, len(img_paths)))

    # 用 EfficientNet-B0 提取特征
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    model.eval().to(device)

    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    features = []
    valid_paths = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(x).cpu().numpy().flatten()
            features.append(feat)
            valid_paths.append(p)
        except Exception as e:
            print(" 跳过 %s: %s" % (p, e))

    features = np.array(features)

    # PCA 降维 + DBSCAN 聚类
    pca = PCA(n_components=min(50, len(features)))
    feat_low = pca.fit_transform(features)

    clustering = DBSCAN(eps=3, min_samples=3).fit(feat_low)
    labels = clustering.labels_

    # 离群样本 (label = -1)
    outlier_mask = labels == -1
    n_outliers = outlier_mask.sum()
    print("  聚类结果: %d 个簇, %d 个离群样本" % (len(set(labels)) - (1 if -1 in labels else 0), n_outliers))

    # 保存离群样本
    bad_dir = Path(output_bad) / class_name
    bad_dir.mkdir(parents=True, exist_ok=True)

    for i, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            shutil.copy2(valid_paths[i], str(bad_dir / ("%03d_" % i + os.path.basename(valid_paths[i]))))

    print("  离群样本已复制到: %s/" % bad_dir)
    print("  请人工检查这些图片, 判断是否标签错误, 然后移到正确类别或删除")


def merge_classes(data_root: str, class_a: str, class_b: str, new_class: str):
    """
    如果两个类实在分不清, 合并为一个类
    （放弃区分 Fog 和 Rain, 统一标为 "bad_weather"）
    """
    train_dir = os.path.join(data_root, "train")

    src_dirs = [os.path.join(train_dir, c) for c in [class_a, class_b]]
    dst_dir = os.path.join(train_dir, new_class)
    os.makedirs(dst_dir, exist_ok=True)

    count = 0
    for src in src_dirs:
        if not os.path.isdir(src):
            continue
        for fname in os.listdir(src):
            if os.path.splitext(fname)[1].lower() in {".jpg", ".jpeg", ".png"}:
                shutil.move(os.path.join(src, fname), os.path.join(dst_dir, fname))
                count += 1
        # 删除空文件夹
        try:
            os.rmdir(src)
        except:
            pass

    print("已将 %s 和 %s 合并为 %s, 共 %d 张图片" % (class_a, class_b, new_class, count))


def main():
    parser = argparse.ArgumentParser(description="数据清洗工具")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--action", choices=["pair", "outlier", "merge"], default="pair")
    parser.add_argument("--class-a", default="cloudy")
    parser.add_argument("--class-b", default="rainy")
    parser.add_argument("--new-class", default="bad_weather")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--keep-ratio", type=float, default=0.9)
    args = parser.parse_args()

    if args.action == "pair":
        collect_confusing_pairs(args.data_root, args.class_a, args.class_b, args.num_pairs)
    elif args.action == "outlier":
        remove_outliers(args.data_root, args.class_a, keep_ratio=args.keep_ratio)
    elif args.action == "merge":
        merge_classes(args.data_root, args.class_a, args.class_b, args.new_class)


if __name__ == "__main__":
    main()
