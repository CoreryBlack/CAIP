#!/usr/bin/env python3
"""
RAICOM 2026 — 验证集修正 + confusing 样本加入验证集
1. 用 Qwen API 对 196 张 confusing 图片逐张分类
2. 修正验证集中 sand_storm / dusttornado 标签 (sunny→cloudy)
3. 把 confusing 图片按 Qwen 分类结果加入验证集
"""
import os, sys, json, shutil
from pathlib import Path
from collections import Counter

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR / "api_screening"))
from providers.qwen_provider import QwenVisionProvider

VAL_DIR = SRC_DIR.parent / "training_data_1" / "val"
CONFUSING_DIR = SRC_DIR.parent / "data_raw_weather4class" / "confusing"
NEW_VAL_DIR = SRC_DIR.parent / "dataset_cleaned" / "val"
CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

SYSTEM_PROMPT = "你是一个专业的气象观测专家，请对每张天气图片做准确的四分类判断。"

USER_PROMPT = """请判断这张图片的天气类型，只能从以下四类中选择：

- cloudy: 多云/阴天/雾霾/沙尘暴/扬尘/烟霾，能见度低但无降水
- rainy: 可见雨滴/雨丝/积水反光/车窗水珠/湿润路面
- snowy: 雪花/积雪/冰霜覆盖
- sunny: 晴朗阳光充足，天空明亮清澈

返回 JSON (只输出 JSON，不要 markdown):
{"label": "cloudy/rainy/snowy/sunny", "confidence": 0.0-1.0}"""


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--delay", type=float, default=0.2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", type=str, default=str(NEW_VAL_DIR))
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 修正原验证集：sand_storm/dusttornado sunny→cloudy
    print("=" * 60)
    print("Step 1: 修正 sand_storm/dusttornado 标签 (sunny → cloudy)")
    print("=" * 60)
    fixes = 0
    sunny_dir = VAL_DIR / "sunny"
    # sand_storm-326.jpg 是唯一 Qwen 也判为 sunny 的，保留不修正
    KEEP_IN_SUNNY = {"sand_storm-326.jpg"}
    for fname in sorted(os.listdir(sunny_dir)):
        base = fname.lower()
        if (base.startswith("sand_storm") or base.startswith("dusttornado")) and fname not in KEEP_IN_SUNNY:
            src = sunny_dir / fname
            dst = VAL_DIR / "cloudy" / fname
            if args.dry_run:
                print(f"  [DRY] {fname}: sunny → cloudy")
            else:
                shutil.move(str(src), str(dst))
                print(f"  ✅ {fname}: sunny → cloudy")
            fixes += 1
    print(f"  共修正: {fixes} 张\n")

    # 2. Qwen 分类 confusing 样本
    print("=" * 60)
    print(f"Step 2: Qwen API 分类 confusing 样本 ({len(list(CONFUSING_DIR.glob('*')))} 张)")
    print("=" * 60)
    print(f"  源目录: {CONFUSING_DIR}")

    confusing_images = sorted([
        p for p in CONFUSING_DIR.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])
    print(f"  图片数: {len(confusing_images)}")

    if args.dry_run:
        print("  [DRY RUN] 不调用 API")
        return

    provider = QwenVisionProvider(model="qwen3-vl-flash", api_key=args.api_key or None)
    results = []
    for i, img_path in enumerate(confusing_images):
        result = provider.screen_image(
            image_path=str(img_path),
            user_prompt=USER_PROMPT,
        )
        provider.system_prompt = SYSTEM_PROMPT
        label = result.get("label", "?")
        conf = result.get("confidence", 0)
        # 容错：将不在四类中的统一归为 cloudy
        if label not in CLASSES:
            label = "cloudy"
        results.append({"filename": img_path.name, "label": label, "confidence": conf})
        pct = (i + 1) / len(confusing_images) * 100
        print(f"  [{i+1:3d}/{len(confusing_images)}] {label:8s} conf={conf:.2f}  "
              f"{img_path.name:40s}  {pct:.0f}%")

    # 统计
    counter = Counter(r["label"] for r in results)
    print(f"\n  分类分布: {dict(counter)}")

    # 3. 构建新验证集
    print(f"\n{'='*60}")
    print(f"Step 3: 构建新验证集 → {args.output_dir}")
    print("=" * 60)

    out_dir = Path(args.output_dir)
    for cls in CLASSES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    # 先复制原验证集（已修正的）
    old_counts = {}
    for cls in CLASSES:
        src_cls = VAL_DIR / cls
        dst_cls = out_dir / cls
        n = 0
        for fname in os.listdir(src_cls):
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                if not args.dry_run:
                    shutil.copy2(str(src_cls / fname), str(dst_cls / fname))
                n += 1
        old_counts[cls] = n
        print(f"  原验证集 {cls}: {n}")

    # 再把 confusing 样本加入
    conf_counts = Counter()
    for r in results:
        label = r["label"]
        src = CONFUSING_DIR / r["filename"]
        dst = out_dir / label / r["filename"]
        if src.exists() and not args.dry_run:
            shutil.copy2(str(src), str(dst))
        conf_counts[label] += 1

    print()
    print("  confusing 加入:")
    for cls in CLASSES:
        print(f"    {cls}: +{conf_counts.get(cls, 0)}")

    print()
    print("  新验证集汇总:")
    total = 0
    for cls in CLASSES:
        n = old_counts[cls] + conf_counts.get(cls, 0)
        print(f"    {cls:8s}: {n:>4d}")
        total += n
    print(f"    总计:    {total:>4d}")

    print(f"\n✅ 新验证集: {out_dir}")
    if args.dry_run:
        print("  [DRY RUN] 未实际写入")


if __name__ == "__main__":
    main()
