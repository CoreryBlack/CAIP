#!/usr/bin/env python
"""
RAICOM 2026 — 混淆样本批量筛查（并发版）
"""
import os, sys, json, csv, time, argparse, threading
from datetime import datetime
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")
from providers.qwen_provider import QwenVisionProvider

PROMPTS_DIR = Path(__file__).parent / "prompts"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
DATA_ROOT = Path(__file__).parent.parent.parent / "data_raw_weather4class"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(DATA_ROOT))
    p.add_argument("--classes", type=str, nargs="*", default=["Fog", "Rain", "Sand", "Snow"])
    p.add_argument("--model", type=str, default="qwen3-vl-flash")
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--delay", type=float, default=0)
    return p.parse_args()


def collect_images(data_dir, classes):
    images = []
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] 跳过: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                images.append({"path": os.path.join(cls_dir, fname), "class": cls_name, "filename": fname})
    return images


def load_processed(path):
    if not os.path.exists(path):
        return set()
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                s.add(json.loads(line).get("image_path", ""))
            except:
                pass
    return s


def build_prompt(original_label):
    from prompts.confusion_prompt import SCREENING_USER_PROMPT
    return SCREENING_USER_PROMPT.replace("当前已有的粗标类别", original_label)


class ThreadSafeWriter:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
    def write(self, record):
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def screen_one(provider, img, writer, results, lock):
    prompt = build_prompt(img["class"])
    result = provider.screen_image(image_path=img["path"], user_prompt=prompt)
    record = {"image_path": img["path"], "class": img["class"], "filename": img["filename"],
              "timestamp": datetime.now().isoformat(), "result": result}
    writer.write(record)
    with lock:
        results.append(record)
    return record


def write_summary(output_dir, total, easy_c, confusing_c, pairs, class_stats):
    sp = os.path.join(output_dir, "summary.csv")
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["指标", "数值"]); w.writerow(["总图片数", total])
        w.writerow(["easy", easy_c]); w.writerow(["confusing", confusing_c])
        w.writerow(["confusing%", f"{confusing_c/max(total,1)*100:.1f}%"])
        w.writerow([]); w.writerow(["按类别", "总数", "easy", "confusing", "confusing%"])
        for c in sorted(class_stats):
            s = class_stats[c]; wp = f"{s['confusing']/max(s['total'],1)*100:.1f}%"
            w.writerow([c, s["total"], s["easy"], s["confusing"], wp])
        w.writerow([]); w.writerow(["混淆对", "次数"])
        for (p, s), cnt in pairs.most_common(10):
            w.writerow([f"{p}->{s}", cnt])
    print(f"[OUT] 统计摘要: {sp}")


def write_confusing(output_dir, results):
    cp = os.path.join(output_dir, "confusing_only.jsonl")
    cnt = 0
    with open(cp, "w", encoding="utf-8") as f:
        for r in results:
            if r.get("result", {}).get("easy_confusing") == "confusing":
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); cnt += 1
    print(f"[OUT] 混淆样本: {cp} ({cnt} 条)")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.jsonl")

    print(f"[SCAN] {args.data_dir}")
    all_images = collect_images(args.data_dir, args.classes)
    print(f"       共 {len(all_images)} 张")
    if args.max_images > 0:
        all_images = all_images[:args.max_images]
        print(f"       (限制 {args.max_images} 张)")

    if args.resume:
        processed = load_processed(results_path)
        pending = [i for i in all_images if i["path"] not in processed]
        print(f"       [RESUME] 已处理 {len(processed)} 张，剩余 {len(pending)} 张")
    else:
        pending = list(all_images)

    if not pending:
        print("[DONE] 无待处理图片"); return

    provider = QwenVisionProvider(model=args.model, api_key=args.api_key or None)
    print(f"[MODEL] {args.model} | [WORKERS] {args.workers}")

    writer = ThreadSafeWriter(results_path)
    rlock = threading.Lock()
    results = []
    total = len(pending)
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        batch_size = args.workers * 4
        for bstart in range(0, len(pending), batch_size):
            batch = pending[bstart:bstart+batch_size]
            futures = {executor.submit(screen_one, provider, i, writer, results, rlock): i for i in batch}
            for f in as_completed(futures):
                img = futures[f]
                completed += 1
                try:
                    rec = f.result()
                    r = rec["result"]
                    if r.get("status") == "error": st = "ERROR"
                    elif r.get("status") == "parse_error": st = "PARSE_ERR"
                    else: st = f"{r.get('easy_confusing','?'):10s} {r.get('confidence',0):.2f}"
                except Exception as e:
                    st = f"EXCEPT {str(e)[:30]}"
                pct = completed / total
                bar_len = 40
                filled = int(bar_len * pct)
                bar = "#" * filled + "-" * (bar_len - filled)
                eta = (time.time() - start_time) / pct - (time.time() - start_time) if pct > 0 else 0
                print(f"  [{completed:4d}/{total}] [{bar}] {pct*100:5.1f}% | {img['class']:12s}/{img['filename']:30s} | {st} | ETA {eta:.0f}s")
            if args.delay > 0:
                time.sleep(args.delay)

    easy_c = confusing_c = err_c = 0
    cs = {}
    pairs = Counter()
    for r in results:
        cls = r["class"]
        if cls not in cs: cs[cls] = {"total": 0, "easy": 0, "confusing": 0}
        cs[cls]["total"] += 1
        res = r.get("result", {})
        if res.get("status") == "error": err_c += 1
        elif res.get("status") == "parse_error": confusing_c += 1; cs[cls]["confusing"] += 1
        else:
            ec = res.get("easy_confusing", "")
            if ec == "confusing": confusing_c += 1; cs[cls]["confusing"] += 1
            else: easy_c += 1; cs[cls]["easy"] += 1
            pairs[(res.get("primary_label","?"), res.get("secondary_label","?"))] += 1

    print(f"\n[DONE] {time.time()-start_time:.0f}s | easy {easy_c} | confusing {confusing_c} | error {err_c}")
    write_summary(args.output_dir, len(results), easy_c, confusing_c, pairs, cs)
    write_confusing(args.output_dir, results)


if __name__ == "__main__":
    main()
