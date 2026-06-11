"""
RAICOM 2026 — 智海算法调优赛
ONNX 模型导出脚本

将训练好的 PyTorch 模型导出为 ONNX 格式，用于 CPU 推理加速。
支持动态批处理尺寸。

用法:
    # 导出最佳模型
    python export_onnx.py --checkpoint ./outputs/best_model.pth

    # 指定输入尺寸和 opset
    python export_onnx.py --checkpoint ./outputs/best_model.pth \
                          --image-size 300 --opset 17
"""

import os
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from config import cfg
from model import create_model, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="导出 ONNX 模型"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="PyTorch 检查点路径")
    parser.add_argument("--output", type=str, default="",
                        help="输出 .onnx 路径（默认在 checkpoint 同目录）")
    parser.add_argument("--image-size", type=int, default=cfg.image_size,
                        help="输入图像尺寸")
    parser.add_argument("--opset", type=int, default=cfg.onnx_opset,
                        help="ONNX opset 版本")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="导出时的固定批大小（-1 表示动态）")
    parser.add_argument("--no-dynamic-batch", action="store_true",
                        help="禁止动态批处理（固定批大小）")
    return parser.parse_args()


def export_to_onnx(args):
    print(f"📦 导出 ONNX 模型")
    print(f"   检查点: {args.checkpoint}")

    # ── 加载模型 ──
    model = load_model(args.checkpoint, cfg)
    model.eval()

    # ── 确定输出路径 ──
    if args.output:
        onnx_path = args.output
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        onnx_path = os.path.join(ckpt_dir, "model.onnx")

    # ── 准备输入 ──
    batch_size = args.batch_size if not args.no_dynamic_batch else 1
    dummy_input = torch.randn(batch_size, 3, args.image_size, args.image_size)

    # ── 动态轴 ──
    dynamic_axes = None
    if not args.no_dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
        print("   动态批处理: ✅ 启用")
    else:
        print(f"   固定批大小: {batch_size}")

    # ── 导出 ──
    print(f"   输入尺寸: {3}x{args.image_size}x{args.image_size}")
    print(f"   ONNX opset: {args.opset}")
    print(f"   输出路径: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # ── 验证 ONNX 模型 ──
    print("\n🔍 验证 ONNX 模型...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("   ✅ ONNX 模型结构验证通过")

    # ── ONNX Runtime 推理测试 ──
    print("\n⚡ 测试 ONNX Runtime 推理...")
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    # 测试推理
    test_input = np.random.randn(1, 3, args.image_size, args.image_size).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: test_input})

    probabilities = torch.softmax(torch.from_numpy(output[0]), dim=1)
    pred_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities.max().item()

    print(f"   ✅ 推理测试通过")
    print(f"   预测类别: {pred_class} | 置信度: {confidence:.4f}")

    # ── 性能基准 ──
    print("\n⏱️  CPU 推理基准测试...")
    import time

    # warmup
    for _ in range(10):
        session.run([output_name], {input_name: test_input})

    # 基准
    batch = np.random.randn(128, 3, args.image_size, args.image_size).astype(np.float32)
    n_iter = 10
    start = time.perf_counter()
    for _ in range(n_iter):
        session.run([output_name], {input_name: batch})
    elapsed = time.perf_counter() - start

    avg_time_ms = elapsed / n_iter * 1000
    per_image_ms = avg_time_ms / 128
    fps = 1000 / per_image_ms

    print(f"   批大小 128 × {n_iter} 次: {avg_time_ms:.1f} ms")
    print(f"   单张推理: {per_image_ms:.3f} ms")
    print(f"   等效 FPS: {fps:.1f}")

    # ── 估算 70 分钟可处理张数 ──
    total_in_70min = int((70 * 60 * 1000) / per_image_ms)
    print(f"\n📊  70 分钟内可推理约 {total_in_70min:,} 张图片")
    print(f"    评分集（几千张）预计耗时约 {per_image_ms * 5000 / 1000:.0f} 秒")

    print(f"\n✅  ONNX 导出完成: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(args)
