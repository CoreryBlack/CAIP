"""
RAICOM 2026 — 智海算法调优赛
ONNX INT8 量化脚本

支持两种模式：
1. dynamic 量化：无需校准数据，最快上手，适合先做速度测试
2. static 量化：需要校准图片，通常对 CNN 更友好，速度收益更稳定

示例：
    # 动态量化（推荐先试）
    python quantize_onnx.py --onnx-path ./outputs/model.onnx

    # 静态量化（推荐比赛最终版）
    python quantize_onnx.py --onnx-path ./outputs/model.onnx \
                            --mode static \
                            --calib-dir ./data/train

    # 指定输出文件
    python quantize_onnx.py --onnx-path ./outputs/model.onnx \
                            --output ./outputs/model.int8.onnx
"""

import argparse
import os
import time
from pathlib import Path
from typing import Iterator, List

import numpy as np

from config import cfg


def get_val_transform_lazy(image_size: int):
    try:
        from augmentations import get_val_transforms
    except ImportError as e:
        raise SystemExit(
            "缺少预处理依赖，请先执行: pip install albumentations opencv-python"
        ) from e
    return get_val_transforms(image_size=image_size)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def require_quant_deps():
    try:
        import cv2  # type: ignore
        import onnxruntime as ort  # type: ignore
        from onnxruntime.quantization import (  # type: ignore
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except ImportError as e:
        raise SystemExit(
            "缺少量化依赖，请先执行: pip install onnx onnxruntime opencv-python"
        ) from e
    return cv2, ort, CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX INT8 量化脚本")
    parser.add_argument("--onnx-path", type=str, required=True, help="原始 FP32 ONNX 模型路径")
    parser.add_argument("--output", type=str, default="", help="量化后模型输出路径")
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="量化模式：dynamic / static",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default="",
        help="static 量化时使用的校准图片目录；可直接给图片目录，或给 train 根目录（递归扫描）",
    )
    parser.add_argument("--image-size", type=int, default=cfg.image_size, help="模型输入尺寸")
    parser.add_argument("--batch-size", type=int, default=32, help="校准 / 基准测试批大小")
    parser.add_argument("--max-calib-images", type=int, default=256, help="最多使用多少张图片做静态量化校准")
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="minmax",
        choices=["minmax", "entropy", "percentile"],
        help="静态量化校准算法",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="启用 per-channel 量化（通常更准，但不一定所有平台都更快）",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="量化完成后对 FP32 / INT8 做简单 CPU 吞吐测试",
    )
    return parser.parse_args()


def _make_calib_reader_class():
    """动态创建继承自 CalibrationDataReader 的类"""
    from onnxruntime.quantization import CalibrationDataReader

    class ImageCalibrationDataReader(CalibrationDataReader):
        """ONNX Runtime 静态量化校准数据读取器"""

        def __init__(self, onnx_path: str, image_paths: List[str], image_size: int, batch_size: int):
            import cv2
            import onnxruntime as ort
            self.cv2 = cv2
            self.image_paths = image_paths
            self.image_size = image_size
            self.batch_size = batch_size
            self.transform = get_val_transform_lazy(image_size=image_size)

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.input_name = session.get_inputs()[0].name
            self._iter = None

        def _preprocess(self, image_path: str) -> np.ndarray:
            img = self.cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"无法读取校准图片: {image_path}")
            img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
            tensor = self.transform(image=img)["image"].numpy().astype(np.float32)
            return tensor

        def get_next(self):
            if self._iter is None:
                self._iter = self._batch_iterator()
            return next(self._iter, None)

        def _batch_iterator(self) -> Iterator[dict]:
            for i in range(0, len(self.image_paths), self.batch_size):
                batch_paths = self.image_paths[i:i + self.batch_size]
                batch = np.stack([self._preprocess(p) for p in batch_paths], axis=0)
                yield {self.input_name: batch}

    return ImageCalibrationDataReader


def collect_images(root: str, max_images: int) -> List[str]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"校准目录不存在: {root}")

    image_paths: List[str] = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            image_paths.append(str(path))
            if len(image_paths) >= max_images:
                break

    if not image_paths:
        raise RuntimeError(f"未在 {root} 中找到可用图片")

    return image_paths


def get_output_path(args) -> str:
    if args.output:
        return args.output
    base = os.path.splitext(args.onnx_path)[0]
    suffix = "int8.dynamic.onnx" if args.mode == "dynamic" else "int8.static.onnx"
    return f"{base}.{suffix}"


def get_calibration_method(method: str):
    _, _, _, CalibrationMethod, *_ = require_quant_deps()
    mapping = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    return mapping[method]


def benchmark_model(onnx_path: str, image_size: int, batch_size: int = 128):
    _, ort, *_ = require_quant_deps()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    test_input = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)

    for _ in range(5):
        session.run([output_name], {input_name: test_input})

    n_iter = 10
    start = time.perf_counter()
    for _ in range(n_iter):
        session.run([output_name], {input_name: test_input})
    elapsed = time.perf_counter() - start

    avg_time_ms = elapsed / n_iter * 1000
    per_image_ms = avg_time_ms / batch_size
    return {
        "batch_time_ms": avg_time_ms,
        "per_image_ms": per_image_ms,
        "fps": 1000.0 / per_image_ms,
    }


def quantize_dynamic_model(args, output_path: str):
    _, _, _, _, _, QuantType, quantize_dynamic, _ = require_quant_deps()
    quantize_dynamic(
        model_input=args.onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        reduce_range=False,
    )


def quantize_static_model(args, output_path: str):
    _, _, _, _, QuantFormat, QuantType, _, quantize_static = require_quant_deps()
    if not args.calib_dir:
        raise ValueError("static 量化必须提供 --calib-dir")

    image_paths = collect_images(args.calib_dir, args.max_calib_images)
    print(f"📸 校准图片: {len(image_paths)} 张")

    ImageCalibrationDataReader = _make_calib_reader_class()
    reader = ImageCalibrationDataReader(
        onnx_path=args.onnx_path,
        image_paths=image_paths,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    quantize_static(
        model_input=args.onnx_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        calibrate_method=get_calibration_method(args.calibration_method),
    )


def verify_model(onnx_path: str, image_size: int):
    _, ort, *_ = require_quant_deps()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    output = session.run([output_name], {input_name: test_input})[0]
    return output.shape


def main():
    args = parse_args()
    output_path = get_output_path(args)

    print("=" * 70)
    print("RAICOM 2026 — ONNX INT8 量化")
    print(f"原始模型: {args.onnx_path}")
    print(f"量化模式: {args.mode}")
    print(f"输出模型: {output_path}")
    print("=" * 70)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if args.mode == "dynamic":
        print("\n⚙️  开始 dynamic INT8 量化...")
        quantize_dynamic_model(args, output_path)
    else:
        print("\n⚙️  开始 static INT8 量化...")
        quantize_static_model(args, output_path)

    print("\n🔍 验证量化模型...")
    output_shape = verify_model(output_path, args.image_size)
    print(f"✅ 验证通过，输出 shape: {output_shape}")

    src_size = os.path.getsize(args.onnx_path)
    dst_size = os.path.getsize(output_path)
    shrink = 100 * (1 - dst_size / src_size)
    print(f"📦 大小变化: {src_size/1024/1024:.2f} MB -> {dst_size/1024/1024:.2f} MB ({shrink:.1f}% smaller)")

    if args.benchmark:
        print("\n⏱️  CPU 基准测试...")
        fp32_stats = benchmark_model(args.onnx_path, args.image_size)
        int8_stats = benchmark_model(output_path, args.image_size)
        speedup = fp32_stats["per_image_ms"] / int8_stats["per_image_ms"]

        print(f"FP32: {fp32_stats['per_image_ms']:.3f} ms / image | FPS {fp32_stats['fps']:.1f}")
        print(f"INT8: {int8_stats['per_image_ms']:.3f} ms / image | FPS {int8_stats['fps']:.1f}")
        print(f"🚀 加速比: {speedup:.2f}x")

    print("\n✅ 量化完成")
    print("推荐：先用 dynamic 跑通，再用 static + 200~500 张校准图做最终版。")


if __name__ == "__main__":
    main()
