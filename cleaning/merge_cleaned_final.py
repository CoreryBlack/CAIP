import os, json, shutil, sys
sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)

VALID_4 = {"cloudy", "rainy", "snowy", "sunny"}
MAPPING = {
    "fogsmog": "cloudy", "fog": "cloudy", "foggy": "cloudy",
    "frost": "cloudy", "rime": "cloudy",
    "dew": "rainy", "hail": "rainy", "lightning": "rainy",
    "sandstorm": "sunny",
}

DST = os.path.join(BASE, "dataset_cleaned")
for c in VALID_4:
    os.makedirs(f"{DST}/{c}", exist_ok=True)

# 数据源：(jsonl 文件路径, 图片根目录, 前缀)
SOURCES = [
    ("cleaning/api_screening/outputs_hf/results.jsonl", ".", "hf"),
    ("cleaning/api_screening/outputs_kaggle/results.jsonl", ".", "kg"),
]

total = 0
for jsonl_path, img_base, prefix in SOURCES:
    path = os.path.join(BASE, jsonl_path)
    if not os.path.exists(path):
        print(f"[SKIP] {path} 不存在")
        continue

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result = r.get("result", {})
            if result.get("easy_confusing") != "easy":
                continue

            pl = result.get("primary_label", "")
            if pl in VALID_4:
                dest_cls = pl
            elif pl in MAPPING:
                dest_cls = MAPPING[pl]
            else:
                continue

            src_path = r["image_path"]
            if not os.path.isabs(src_path):
                src_path = os.path.join(BASE, src_path)
            src_path = os.path.normpath(src_path)

            if not os.path.exists(src_path):
                continue

            fname = os.path.basename(src_path)
            dst_path = os.path.join(DST, dest_cls, f"{prefix}_{r['class']}_{fname}")
            shutil.copy2(src_path, dst_path)
            total += 1

print(f"清洗后统一数据集: {total} 张")
for c in VALID_4:
    n = len(os.listdir(f"{DST}/{c}"))
    print(f"  {c}: {n}")
