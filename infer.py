"""
RAICOM 2026 — 智海算法调优赛
CPU 推理脚本 (ONNX Runtime)

用于比赛最终提交：加载 ONNX 模型，对测试集图片做推理，输出预测结果。
支持 TTA (Test-Time Augmentation)。

用法:
    # 对测试集进行推理
    python infer.py --onnx-path ./outputs/model.onnx \
                    --data-root ./data/test \
                    --output ./outputs/predictions.csv

    # 使用 TTA（5 次增强取平均）
    python infer.py --onnx-path ./outputs/model.onnx \
                    --data-root ./data/test \
                    --tta

    # 单张图片推理
    python infer.py --onnx-path ./outputs/model.onnx \
                    --image-path ./data/test/sample.jpg
"""

import os
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import onnxruntime as ort

from config import cfg
from augmentations import get_val_transforms, get_tta_transforms


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAICOM 2026 — CPU 推理 (ONNX Runtime)"
    )
    parser.add_argument("--onnx-path", type=str, required=True,
                        help="ONNX 模型路径")
    parser.add_argument("--data-root", type=str, default="",
                        help="测试集目录（图片直接放在此目录下）")
    parser.add_argument("--image-path", type=str, default="",
                        help="单张图片路径")
    parser.add_argument("--output", type=str, default="./outputs/predictions.csv",
                        help="输出 CSV 路径")
    parser.add_argument("--batch-size", type=int, default=cfg.cpu_inference_batch,
                        help="推理批次大小")
    parser.add_argument("--image-size", type=int, default=cfg.image_size,
                        help="输入图像尺寸")
    parser.add_argument("--tta", action="store_true",
                        help="启用 Test-Time Augmentation")
    return parser.parse_args()


def load_image(image_path: str, image_size: int):
    """加载单张图片并返回 RGB numpy 数组"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_images(
    images: list,
    transforms,
    image_size: int,
):
    """
    对一批图像应用预处理

    Returns: numpy array [B, 3, H, W]
    """
    processed = []
    for img in images:
        augmented = transforms(image=img)
        processed.append(augmented["image"].numpy())
    return np.array(processed, dtype=np.float32)


def run_inference(
    session: ort.InferenceSession,
    batch: np.ndarray,
) -> np.ndarray:
    """
    运行 ONNX 推理

    Returns: softmax 概率 [B, num_classes]
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: batch})[0]
    # softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs


def run_inference_tta(
    session: ort.InferenceSession,
    images: list,
    image_size: int,
    val_transforms,
    tta_transforms_list: list,
) -> np.ndarray:
    """
    Test-Time Augmentation 推理

    对每张图片应用多个增强配置，对 softmax 概率取平均
    """
    all_probs = []

    for img in images:
        # 基础预测
        base = val_transforms(image=img)
        base_tensor = base["image"].numpy()[np.newaxis, ...]
        base_probs = run_inference(session, base_tensor)
        avg_probs = base_probs

        # TTA 变换
        for tta_tf in tta_transforms_list:
            tta_result = tta_tf(image=img)
            # 应用标准化（TTA 变换不含 Normalize）
            normalized = val_transforms(image=tta_result["image"])
            tta_tensor = normalized["image"].numpy()[np.newaxis, ...]
            tta_probs = run_inference(session, tta_tensor)
            avg_probs += tta_probs

        avg_probs /= (1 + len(tta_transforms_list))
        all_probs.append(avg_probs[0])

    return np.array(all_probs)


def infer_images(args):
    """对目录下所有图片进行推理"""
    print(f"🔍 推理模式: {'TTA' if args.tta else '标准'}")
    print(f"   ONNX 模型: {args.onnx_path}")
    print(f"   数据目录: {args.data_root}")

    # ── 加载 ONNX 模型 ──
    session = ort.InferenceSession(
        args.onnx_path,
        providers=["CPUExecutionProvider"],
    )
    print(f"   ONNX Runtime 提供者: CPUExecutionProvider")

    # ── 图片列表 ──
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    img_paths = sorted([
        os.path.join(args.data_root, f)
        for f in os.listdir(args.data_root)
        if os.path.splitext(f)[1].lower() in img_exts
    ])

    if not img_paths:
        print("⚠️  未找到图片文件")
        return

    print(f"   共找到 {len(img_paths)} 张图片")
    print(f"   批大小: {args.batch_size}")

    # ── 预处理 ──
    val_transforms = get_val_transforms(image_size=args.image_size)
    tta_transforms = get_tta_transforms(image_size=args.image_size) if args.tta else []

    # ── 推理 ──
    all_probs = []
    all_times = []

    if args.tta:
        # TTA 模式下逐张推理（单张增强取平均）
        for img_path in tqdm(img_paths, desc="TTA 推理"):
            img = load_image(img_path, args.image_size)
            probs = run_inference_tta(
                session, [img], args.image_size,
                val_transforms, tta_transforms,
            )
            all_probs.append(probs[0])
    else:
        # 批处理推理
        for i in tqdm(range(0, len(img_paths), args.batch_size), desc="推理"):
            batch_paths = img_paths[i:i + args.batch_size]
            batch_images = [load_image(p, args.image_size) for p in batch_paths]
            batch_tensor = preprocess_images(batch_images, val_transforms, args.image_size)

            start = time.perf_counter()
            probs = run_inference(session, batch_tensor)
            elapsed = time.perf_counter() - start
            all_times.append(elapsed)

            all_probs.extend(probs)

    # ── 结果 ──
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    # ── 统计 ──
    if all_times:
        total_time = sum(all_times)
        per_image = total_time / len(img_paths) * 1000
        print(f"\n⏱️  总耗时: {total_time:.2f}s")
        print(f"   单张平均: {per_image:.3f} ms")
        print(f"   {len(img_paths)} 张预计 70 分钟可推理 {int(70*60/per_image*1000):,} 轮")

    # ── 输出 CSV ──
    class_names = cfg.classes
    results = pd.DataFrame({
        "filename": [os.path.basename(p) for p in img_paths],
        "pred_label": [class_names[p] for p in all_preds],
        "pred_class_id": all_preds,
    })

    # 添加各类别概率
    for i, name in enumerate(class_names):
        results[f"prob_{name}"] = all_probs[:, i]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\n✅ 推理完成！结果保存至: {args.output}")
    print(f"\n预测分布:")
    print(results["pred_label"].value_counts().to_string())

    return results


def infer_single_image(args):
    """单张图片推理"""
    session = ort.InferenceSession(
        args.onnx_path,
        providers=["CPUExecutionProvider"],
    )

    img = load_image(args.image_path, args.image_size)
    val_transforms = get_val_transforms(image_size=args.image_size)
    processed = preprocess_images([img], val_transforms, args.image_size)

    probs = run_inference(session, processed)[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]

    print(f"\n📷 单张图片推理: {args.image_path}")
    print(f"   预测类别: {cfg.classes[pred_class]} (ID: {pred_class})")
    print(f"   置信度: {confidence:.4f}")
    print(f"\n类别概率:")
    for i, name in enumerate(cfg.classes):
        print(f"   {name:8s}: {probs[i]:.4f}")

    return pred_class


def main():
    args = parse_args()

    if args.image_path:
        infer_single_image(args)
    elif args.data_root:
        infer_images(args)
    else:
        print("请指定 --data-root 或 --image-path")


if __name__ == "__main__":
    main()
