#!/usr/bin/env python3
"""
RAICOM 2026 — 用 qwen3-vl-plus 并清洗验证集/训练集（并发版）
用法: python3 clean_val_fast.py --target val     (清洗验证集)
     python3 clean_val_fast.py --target train   (清洗训练集 sand_storm)
     python3 clean_val_fast.py --target train --max 2000 --workers 10
"""
import os, sys, json, shutil, argparse, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR / "api_screening"))
from providers.qwen_provider import QwenVisionProvider

CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SYSTEM_PROMPT = "你是气象专家，对天气图片做准确四分类。"
USER_PROMPT = '{"label":"cloudy/rainy/snowy/sunny","confidence":0.0-1.0}。cloudy=多云/阴天/雾霾/沙尘暴, rainy=下雨/雨滴/积水, snowy=下雪/积雪/冰霜, sunny=晴朗/阳光。只返回JSON。'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["val", "train"], default="val")
    p.add_argument("--val-dir", default=str(SRC_DIR.parent / "training_data_1" / "val"))
    p.add_argument("--train-dir", default=str(SRC_DIR.parent / "dataset_cleaned"))
    p.add_argument("--max", type=int, default=0, help="最多处理图片数（0=全部）")
    p.add_argument("--workers", type=int, default=8, help="并发数")
    p.add_argument("--delay", type=float, default=0.0)
    return p.parse_args()


def collect_images(root_dir, max_n):
    images = []
    for cls in CLASSES:
        d = os.path.join(root_dir, cls)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if os.path.splitext(fname)[1].lower() in EXTS:
                images.append((os.path.join(d, fname), cls, fname))
    if max_n > 0:
        images = images[:max_n]
    return images


def classify_one(provider, path, orig_label, fname):
    r = provider.screen_image(image_path=path, user_prompt=USER_PROMPT)
    label = r.get("label", "?")
    if label not in CLASSES:
        label = "cloudy"
    conf = r.get("confidence", 0)
    return {"file": fname, "path": path, "orig": orig_label, "api": label, "conf": conf}


def main():
    args = parse_args()

    if args.target == "val":
        root = args.val_dir
    else:
        root = args.train_dir

    images = collect_images(root, args.max)
    print(f"[SCAN] {root} 共 {len(images)} 张, workers={args.workers}")

    # 并发分类
    results = []
    lock = threading.Lock()
    completed = [0]
    start = time.time()

    def process_one(img):
        path, orig_label, fname = img
        provider = QwenVisionProvider(model="qwen3-vl-plus")
        provider.system_prompt = SYSTEM_PROMPT
        if args.delay > 0:
            time.sleep(args.delay)
        rec = classify_one(provider, path, orig_label, fname)
        with lock:
            completed[0] += 1
            pct = completed[0] / len(images) * 100
            diff = "DIFF" if rec["orig"] != rec["api"] else "OK"
            print(f"  [{completed[0]:4d}/{len(images)}] {diff:5s} {rec['orig']:8s} -> {rec['api']:8s} conf={rec['conf']:.2f}  {fname}")
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, img): img for img in images}
        for f in as_completed(futures):
            results.append(f.result())

    elapsed = time.time() - start
    print(f"\n[DONE] {elapsed:.0f}s ({elapsed/len(images):.1f}s/img)")

    # 移动
    moved = 0
    for r in results:
        if r["orig"] != r["api"]:
            src = r["path"]
            dst_dir = os.path.join(os.path.dirname(os.path.dirname(src)), r["api"])
            dst = os.path.join(dst_dir, os.path.basename(src))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, dst)
            moved += 1

    # 汇总
    print(f"\n{'='*60}")
    print(f"  qwen3-vl-plus 清洗完成 ({args.target})")
    print(f"  总计: {len(results)}  移动: {moved}  保留: {len(results) - moved}")
    for cls in CLASSES:
        d = os.path.join(root, cls)
        n = len([f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in EXTS]) if os.path.isdir(d) else 0
        print(f"    {cls}: {n}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
