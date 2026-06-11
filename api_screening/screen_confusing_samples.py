#!/usr/bin/env python
"""
RAICOM 2026 — 混淆样本批量筛查

通过千问视觉模型对天气图片做"easy/confusing"初筛。
支持断点续跑，避免中途中断后重来。

用法：
    # 默认：data_raw_weather4class 全部四类
    python api_screening/screen_confusing_samples.py

    # 只扫 Fog 和 Rain（推荐先做）
    python api_screening/screen_confusing_samples.py --classes Fog Rain

    # 指定 API key / 模型
    python api_screening/screen_confusing_samples.py --api-key sk-xxx --model qwen3-vl-flash

    # 从上次中断处续跑
    python api_screening/screen_confusing_samples.py --resume
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

from providers.qwen_provider import QwenVisionProvider

PROMPTS_DIR = Path(__file__).parent / "prompts"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
DATA_ROOT = Path(__file__).parent.parent / "data_raw_weather4class"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="千问视觉混淆样本批量筛查")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_ROOT),
        help="数据集根目录 (默认: data_raw_weather4class)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=["Fog", "Rain", "Sand", "Snow"],
        help="要筛查的类别子目录名 (默认: Fog Rain Sand Snow)",
    )
    parser.add_argument("--model", type=str, default="qwen3-vl-flash", help="千问模型名")
    parser.add_argument("--api-key", type=str, default="", help="DASHSCOPE_API_KEY")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR), help="输出目录")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续跑模式：跳过 outputs/results.jsonl 中已处理过的图片",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="最多处理多少张图（0=全部，用于小批量测试）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="每张图之间的间隔秒数 (避免 API 限流)",
    )
    return parser.parse_args()


def collect_images(data_dir: str, classes: list) -> list:
    images = []
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] 类别目录不存在，跳过: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTS:
                images.append({
                    "path": os.path.join(cls_dir, fname),
                    "class": cls_name,
                    "filename": fname,
                })
    return images


def load_processed(results_path: str) -> set:
    processed = set()
    if not os.path.exists(results_path):
        return processed
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                processed.add(record.get("image_path", ""))
            except json.JSONDecodeError:
                continue
    return processed


def build_prompt(original_label: str) -> str:
    from prompts.confusion_prompt import SCREENING_USER_PROMPT
    return SCREENING_USER_PROMPT.replace("当前已有的粗标类别", original_label)


def write_result_jsonl(results_path: str, record: dict):
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_csv(
    output_dir: str,
    total: int,
    confusing_count: int,
    easy_count: int,
    confusion_pairs: Counter,
    class_stats: dict,
):
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "数值"])
        writer.writerow(["总图片数", total])
        writer.writerow(["easy", easy_count])
        writer.writerow(["confusing", confusing_count])
        writer.writerow(["confusing 占比", f"{confusing_count/max(total,1)*100:.1f}%"])
        writer.writerow([])
        writer.writerow(["按类别统计"])
        writer.writerow(["类别", "总数", "easy", "confusing", "confusing%"])
        for cls_name in sorted(class_stats.keys()):
            s = class_stats[cls_name]
            c_pct = f"{s['confusing']/max(s['total'],1)*100:.1f}%"
            writer.writerow([cls_name, s["total"], s["easy"], s["confusing"], c_pct])
        writer.writerow([])
        writer.writerow(["最常混淆的标签对"])
        writer.writerow(["primary_label", "secondary_label", "次数"])
        for (p, s), cnt in confusion_pairs.most_common(10):
            writer.writerow([p, s, cnt])
    print(f"[OUT] 统计摘要: {summary_path}")


def write_confusing_only(output_dir: str, results: list):
    confusing_path = os.path.join(output_dir, "confusing_only.jsonl")
    count = 0
    with open(confusing_path, "w", encoding="utf-8") as f:
        for r in results:
            if r.get("result", {}).get("easy_confusing") == "confusing":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1
    print(f"[OUT] 混淆样本: {confusing_path} ({count} 条)")


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.jsonl")
    summary_csv_path = os.path.join(output_dir, "summary.csv")

    # 收集图片
    print(f"[SCAN] 扫描数据集: {args.data_dir}")
    all_images = collect_images(args.data_dir, args.classes)
    print(f"       共找到 {len(all_images)} 张图片")

    if args.max_images > 0:
        all_images = all_images[: args.max_images]
        print(f"       (限制为 {args.max_images} 张)")

    # 续跑
    processed = set()
    if args.resume:
        processed = load_processed(results_path)
        print(f"       [RESUME] 已处理 {len(processed)} 张，剩余 {len(all_images) - len(processed)} 张")

    # 初始化 provider
    provider = QwenVisionProvider(
        model=args.model,
        api_key=args.api_key if args.api_key else None,
    )
    print(f"[MODEL] 模型: {args.model}")
    print(f"[OUT] 结果保存: {results_path}")
    print()

    # 逐张筛查
    results = []
    stats = {
        "total": 0,
        "easy": 0,
        "confusing": 0,
        "error": 0,
    }
    class_stats: dict = {}
    confusion_pairs: Counter = Counter()

    for i, img in enumerate(all_images):
        img_path = img["path"]

        if img_path in processed:
            print(f"   [{i+1}/{len(all_images)}] [SKIP] 跳过 (已处理): {img['filename']}")
            continue

        print(f"   [{i+1}/{len(all_images)}] [SCREEN] {img['class']}/{img['filename']} ...", end="", flush=True)

        result = provider.screen_image(
            image_path=img_path,
            user_prompt=build_prompt(img["class"]),
        )

        record = {
            "image_path": img_path,
            "class": img["class"],
            "filename": img["filename"],
            "timestamp": datetime.now().isoformat(),
            "result": result,
        }

        write_result_jsonl(results_path, record)
        results.append(record)

        # 统计
        stats["total"] += 1
        cls_name = img["class"]
        if cls_name not in class_stats:
            class_stats[cls_name] = {"total": 0, "easy": 0, "confusing": 0}
        class_stats[cls_name]["total"] += 1

        if result.get("status") == "error":
            stats["error"] += 1
            print(f" [ERROR] {result.get('error', '')[:60]}")
        elif result.get("status") == "parse_error":
            stats["confusing"] += 1
            class_stats[cls_name]["confusing"] += 1
            print(f" [PARSE_ERROR] JSON 解析失败")
        else:
            ec = result.get("easy_confusing", "unknown")
            if ec == "confusing":
                stats["confusing"] += 1
                class_stats[cls_name]["confusing"] += 1
            else:
                stats["easy"] += 1
                class_stats[cls_name]["easy"] += 1

            primary = result.get("primary_label", "unknown")
            secondary = result.get("secondary_label", "none")
            conf = result.get("confidence", 0)
            confusion_pairs[(primary, secondary)] += 1
            print(f" {ec:10s} | 置信度 {conf:.2f}")

        # 延时，避免限流
        time.sleep(args.delay)

    # 生成报告
    print(f"\n{'='*60}")
    print(f"[DONE] 筛查完成")
    print(f"   总图片: {stats['total']}")
    print(f"   easy:  {stats['easy']}")
    print(f"   confusing: {stats['confusing']}")
    print(f"   失败:   {stats['error']}")
    print(f"结果文件: {results_path}")
    write_summary_csv(
        output_dir,
        stats["total"],
        stats["confusing"],
        stats["easy"],
        confusion_pairs,
        class_stats,
    )
    write_confusing_only(
        output_dir,
        results,
    )


if __name__ == "__main__":
    main()
