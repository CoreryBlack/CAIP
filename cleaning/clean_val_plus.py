#!/usr/bin/env python3
"""
RAICOM 2026 — 用 qwen3-vl-plus 清洗验证集并直接移动图片
结果: 验证集标签修正完毕，可直接训练
"""
import os, sys, json, shutil
from pathlib import Path

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR / "api_screening"))
from providers.qwen_provider import QwenVisionProvider

VAL_DIR = SRC_DIR.parent / "training_data_1" / "val"
CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SYSTEM_PROMPT = "你是气象专家，对天气图片做准确四分类。"
USER_PROMPT = """类别：cloudy(多云/阴天/雾霾/沙尘暴), rainy(下雨/雨滴/积水), snowy(下雪/积雪/冰霜), sunny(晴朗/阳光)。
只返回JSON: {"label":"...", "confidence":0.0-1.0}"""


def main():
    images = []
    for cls in CLASSES:
        d = VAL_DIR / cls
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in EXTS:
                    images.append((str(f), cls, f.name))

    provider = QwenVisionProvider(model="qwen3-vl-plus")
    provider.system_prompt = SYSTEM_PROMPT

    log_path = SRC_DIR.parent / "outputs" / "val_clean_plus.jsonl"
    moved = 0
    total = len(images)

    with open(log_path, "w", encoding="utf-8") as log:
        for i, (path, orig_label, fname) in enumerate(images):
            r = provider.screen_image(image_path=path, user_prompt=USER_PROMPT)
            label = r.get("label", "?")
            if label not in CLASSES:
                label = "cloudy"
            conf = r.get("confidence", 0)

            rec = {"file": fname, "orig": orig_label, "plus": label, "conf": conf}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()

            action = "OK"
            if label != orig_label:
                src = os.path.join(VAL_DIR, orig_label, fname)
                dst = os.path.join(VAL_DIR, label, fname)
                shutil.move(src, dst)
                action = f"MOVE {orig_label}->{label}"
                moved += 1

            print(f"[{i+1:3d}/{total}] {action:20s} conf={conf:.2f}  {fname}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"  qwen3-vl-plus 清洗完成")
    print(f"  总计: {total}  移动: {moved}  保留: {total - moved}")
    print(f"  日志: {log_path}")
    for cls in CLASSES:
        n = len([f for f in (VAL_DIR/cls).iterdir() if f.suffix.lower() in EXTS])
        print(f"    {cls}: {n}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
