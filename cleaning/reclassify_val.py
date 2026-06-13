#!/usr/bin/env python3
"""
RAICOM 2026 — 验证集重分类脚本
使用千问 Qwen3-VL-Flash 对 204 张验证集图片逐张重分类，
对比原标签，输出修正建议。
"""
import os, sys, json, csv, time, argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR / "api_screening"))
from providers.qwen_provider import QwenVisionProvider

VAL_DIR = Path(__file__).parent.parent / "training_data_1" / "val"
CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# === 精简分类 Prompt（仅做四分类，不判 confusing） ===
SYSTEM_PROMPT = """你是一个专业的气象观测专家。请对每张天气图片做出准确的四分类判断。"""

USER_PROMPT_TEMPLATE = """请判断这张图片的天气类型。只能从以下四类中选择：

- cloudy: 多云/阴天/雾霾/沙尘暴/扬尘/烟霾，能见度降低但无降水
- rainy: 有可见雨滴、雨丝、积水反光、车窗水珠、湿润路面
- snowy: 有雪花、积雪、冰霜覆盖，白色冰晶特征
- sunny: 晴朗天气，阳光充足，天空明亮清澈

请返回 JSON（只输出 JSON，不要 markdown）：
{
  "label": "cloudy/rainy/snowy/sunny",
  "confidence": 0.0到1.0之间的数,
  "evidence": "20-80字的判断依据"
}"""


def parse_args():
    p = argparse.ArgumentParser(description="验证集重分类")
    p.add_argument("--val-dir", type=str, default=str(VAL_DIR))
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--model", type=str, default="qwen3-vl-flash")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--delay", type=float, default=0.3)
    p.add_argument("--dry-run", action="store_true", help="只统计不调 API")
    return p.parse_args()


def collect_val_images(val_dir):
    images = []
    for cls_name in sorted(os.listdir(val_dir)):
        cls_dir = os.path.join(val_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                images.append({
                    "path": os.path.join(cls_dir, fname),
                    "current_label": cls_name,
                    "filename": f"{cls_name}/{fname}",
                })
    return images


def load_done(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["filename"])
            except:
                pass
    return done


def main():
    args = parse_args()
    images = collect_val_images(args.val_dir)
    print(f"[VAL] 共 {len(images)} 张验证集图片")

    if args.dry_run:
        print("[DRY RUN] 不调用 API")
        for cls in CLASSES:
            n = sum(1 for i in images if i["current_label"] == cls)
            print(f"  {cls}: {n}")
        return

    # 输出路径
    out_dir = os.path.join(args.val_dir, "_reclassify_output")
    os.makedirs(out_dir, exist_ok=True)
    results_path = args.output or os.path.join(out_dir, "results.jsonl")

    # 断点续跑
    done = load_done(results_path) if args.resume else set()
    pending = [i for i in images if i["filename"] not in done]
    print(f"[RESUME] 已完成 {len(done)}, 待处理 {len(pending)}")

    if not pending:
        print("[DONE] 全部完成")
        return

    # 初始化 provider
    provider = QwenVisionProvider(model=args.model, api_key=args.api_key or None)
    print(f"[MODEL] {args.model}")

    completed = len(done)
    total = len(images)
    corrections = []  # 记录需要修正的
    per_class = {cls: {"correct": 0, "wrong": 0, "total": 0} for cls in CLASSES}

    with open(results_path, "a", encoding="utf-8") as out_f:
        for i, img in enumerate(pending):
            # 替换分类 prompt
            provider.system_prompt = SYSTEM_PROMPT
            result = provider.screen_image(
                image_path=img["path"],
                user_prompt=USER_PROMPT_TEMPLATE,
            )

            record = {
                "filename": img["filename"],
                "current_label": img["current_label"],
                "image_path": img["path"],
                "timestamp": datetime.now().isoformat(),
                "api_result": result,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # 解析
            api_label = result.get("label", "?")
            conf = result.get("confidence", 0)
            match = "✅" if api_label == img["current_label"] else "❌ DIFF"

            completed += 1
            pct = completed / total * 100
            eta = (time.time() - (time.time() - 0)) if completed == 0 else 0
            print(f"  [{completed:3d}/{total}] {match}  "
                  f"orig={img['current_label']:8s} api={api_label:8s} conf={conf:.2f}  "
                  f"{img['filename']:40s}  {pct:.0f}%")

            per_class[img["current_label"]]["total"] += 1
            if api_label == img["current_label"]:
                per_class[img["current_label"]]["correct"] += 1
            else:
                per_class[img["current_label"]]["wrong"] += 1
                corrections.append(record)

            if args.delay > 0:
                time.sleep(args.delay)

    # === 汇总 ===
    print()
    print("=" * 60)
    print("重分类结果汇总")
    print("=" * 60)
    total_correct = sum(p["correct"] for p in per_class.values())
    total_wrong = sum(p["wrong"] for p in per_class.values())
    print(f"一致: {total_correct}  不一致: {total_wrong}  准确率: {total_correct/204*100:.1f}%")
    print()
    for cls in CLASSES:
        p = per_class[cls]
        acc = p["correct"] / max(p["total"], 1) * 100
        print(f"  {cls:8s}: {p['correct']}/{p['total']} 一致, {p['wrong']} 不一致 ({acc:.0f}%)")
    print()
    if corrections:
        print(f"=== 需要修正的 {len(corrections)} 张 ===")
        for c in corrections:
            r = c["api_result"]
            print(f"  {c['filename']:40s} {c['current_label']:8s} → {r.get('label','?'):8s}  conf={r.get('confidence',0):.2f}")
        # 输出修正清单
        correction_path = os.path.join(out_dir, "corrections.csv")
        with open(correction_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "current_label", "api_label", "confidence", "evidence"])
            for c in corrections:
                r = c["api_result"]
                w.writerow([c["filename"], c["current_label"], r.get("label", "?"),
                            r.get("confidence", 0), r.get("evidence", "")])
        print(f"\n修正清单: {correction_path}")
    print(f"\n完整结果: {results_path}")


if __name__ == "__main__":
    main()
