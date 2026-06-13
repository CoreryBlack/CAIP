"""
RAICOM 2026 - 标签质量验证工具

用法:
    python verify_labels.py --data-root ./data --mode grid
    python verify_labels.py --data-root ./data --mode browse

模式:
    grid  - 每个类别随机抽 N 张图片，拼成网格图保存
    browse - 逐张显示图片和对应标签，人工确认
"""

import os
import random
import argparse
import csv
from pathlib import Path


def sample_images(data_root, num_per_class=5):
    """每个类别随机抽 num_per_class 张图片"""
    train_dir = os.path.join(data_root, "train")
    samples = {}
    classes = sorted(os.listdir(train_dir))

    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        sampled = random.sample(imgs, min(num_per_class, len(imgs)))
        samples[cls] = [os.path.join(cls_dir, s) for s in sampled]

    return samples


def mode_grid(data_root, output="label_preview.png", num_per_class=5):
    """拼成网格图，保存到文件"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("需要安装 Pillow: pip install Pillow")
        return

    samples = sample_images(data_root, num_per_class)
    classes = list(samples.keys())
    n_cols = max(len(v) for v in samples.values())
    n_rows = len(classes)

    thumb_w, thumb_h = 256, 256
    label_h = 30
    cell_h = thumb_h + label_h
    cell_w = thumb_w

    canvas_w = n_cols * cell_w + 20
    canvas_h = n_rows * cell_h + 20
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for row, cls in enumerate(classes):
        for col, img_path in enumerate(samples[cls]):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
                x = col * cell_w + 10
                y = row * cell_h + 10
                canvas.paste(img, (x, y))
                draw.text((x + 5, y + thumb_h + 5), f"{cls}/{os.path.basename(img_path)}",
                         fill=(0, 0, 0))
            except Exception as e:
                print("  跳过 %s: %s" % (img_path, e))

    canvas.save(output, quality=95)
    print("标签预览图已保存: %s" % output)
    print("请打开此图片，逐行检查每个类别的图片是否标签正确。")
    print()
    print("如果发现标注错误，请记录并手动修正:")
    print("  1. 把错放的图片从当前文件夹移走")
    print("  2. 放到正确的类别文件夹中")


def mode_csv(data_root, output="label_review.csv"):
    """生成 CSV 文件，方便 Excel 逐行审查"""
    train_dir = os.path.join(data_root, "train")
    rows = []
    for cls in sorted(os.listdir(train_dir)):
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                rows.append({
                    "filename": fname,
                    "current_label": cls,
                    "image_path": os.path.join(cls_dir, fname),
                    "correct?_Y/N": "",
                    "correct_label": "",
                    "notes": "",
                })

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("标签审查 CSV 已保存: %s" % output)
    print("总行数: %d 张图片" % len(rows))
    print()
    print("使用方法:")
    print("  1. 用 Excel 打开此文件")
    print("  2. 对每张图片，在 'correct?_Y/N' 列填写 Y 或 N")
    print("  3. 如果是 N，在 'correct_label' 列写上正确的类别")
    print("  4. 保存后用下面的脚本批量修正")


def main():
    parser = argparse.ArgumentParser(description="标签质量验证")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--mode", choices=["grid", "csv"], default="grid")
    parser.add_argument("--num-per-class", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "grid":
        output = args.output or "label_preview.png"
        mode_grid(args.data_root, output, args.num_per_class)
    else:
        output = args.output or "label_review.csv"
        mode_csv(args.data_root, output)


if __name__ == "__main__":
    main()
