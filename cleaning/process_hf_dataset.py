import os, sys, shutil, glob
sys.stdout.reconfigure(encoding="utf-8")

MAP = {
    "cloudy": ["fogsmog", "frost", "rime"],
    "rainy": ["rain", "dew", "hail", "glaze"],
    "snowy": ["snow"],
}
SKIP = ["lightning", "rainbow", "sandstorm"]

SRC = "raicom-weather/dataset"
DST = "raicom-weather/training_data_2/hf_processed"

for c in MAP:
    os.makedirs(f"{DST}/{c}", exist_ok=True)

for dest_cls, src_classes in MAP.items():
    for src_cls in src_classes:
        for f in glob.glob(f"{SRC}/{src_cls}/*"):
            shutil.copy2(f, f"{DST}/{dest_cls}/")

total = 0
for c in MAP:
    n = len(glob.glob(f"{DST}/{c}/*"))
    print(f"  {c}: {n}")
    total += n
print(f"  总计: {total}")

print(f"\n跳过: {', '.join(SKIP)}")
