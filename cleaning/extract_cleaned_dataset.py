import os, json, shutil, sys
sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "api_screening/outputs_hf/results.jsonl")
DST = os.path.join(os.path.dirname(SCRIPT_DIR), "training_data_3")

VALID_4 = {"cloudy", "rainy", "snowy", "sunny"}
MAPPING = {
    "fogsmog": "cloudy", "fog": "cloudy", "foggy": "cloudy",
    "frost": "cloudy", "rime": "cloudy",
    "dew": "rainy", "hail": "rainy", "lightning": "rainy",
    "sandstorm": "sunny",
}

for c in VALID_4:
    os.makedirs(f"{DST}/{c}", exist_ok=True)

total = 0
copied = 0
skipped = 0

with open(SRC, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        result = r.get("result", {})
        if result.get("easy_confusing") != "easy":
            continue

        pl = result.get("primary_label", "")
        total += 1

        src_path = r["image_path"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(src_path):
            parent = os.path.dirname(script_dir)
            src_path = os.path.join(parent, src_path)
        src_path = os.path.normpath(src_path)

        if not os.path.exists(src_path):
            skipped += 1
            continue

        if pl in VALID_4:
            dest_cls = pl
        elif pl in MAPPING:
            dest_cls = MAPPING[pl]
        else:
            skipped += 1
            continue

        fname = os.path.basename(src_path)
        dst_path = os.path.join(DST, dest_cls, f"{r['class']}_{fname}")
        shutil.copy2(src_path, dst_path)
        copied += 1

print(f"easy 总张数: {total}")
print(f"已复制: {copied}")
print(f"跳过: {skipped}")
print()
for c in VALID_4:
    n = len(os.listdir(f"{DST}/{c}"))
    print(f"  {c}: {n}")
