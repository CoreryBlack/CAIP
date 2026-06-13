import os, sys, shutil, glob
sys.stdout.reconfigure(encoding="utf-8")

CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
DST = "dataset_unified"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)  # raicom-weather/

# 数据源配置：(源目录, 文件名前缀)
SOURCES_TRAIN = [
    (os.path.join(BASE, "training_data_1/train"), "t1"),
    (os.path.join(BASE, "training_data_2/kaggle_processed"), "t2"),
    (os.path.join(BASE, "training_data_3"), "t3"),
]
SOURCE_VAL = os.path.join(BASE, "training_data_1/val")

# 创建目标目录
for c in CLASSES:
    os.makedirs(f"{DST}/train/{c}", exist_ok=True)
    os.makedirs(f"{DST}/val/{c}", exist_ok=True)

total = 0
# 合并训练集
for src_dir, prefix in SOURCES_TRAIN:
    for c in CLASSES:
        cls_dir = os.path.join(src_dir, c)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]
        for f in files:
            src_path = os.path.join(cls_dir, f)
            fname = f"{prefix}_{f}"
            dst_path = f"{DST}/train/{c}/{fname}"
            shutil.copy2(src_path, dst_path)
            total += 1

# 复制验证集
val_count = 0
for c in CLASSES:
    cls_dir = os.path.join(SOURCE_VAL, c)
    if not os.path.isdir(cls_dir):
        continue
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]
    for f in files:
        src_path = os.path.join(cls_dir, f)
        dst_path = f"{DST}/val/{c}/{f}"
        shutil.copy2(src_path, dst_path)
        val_count += 1

print(f"训练集合并完成: {total} 张")
for c in CLASSES:
    n = len(os.listdir(f"{DST}/train/{c}"))
    print(f"  {c}: {n}")
print(f"\n验证集: {val_count} 张")
for c in CLASSES:
    n = len(os.listdir(f"{DST}/val/{c}"))
    print(f"  val/{c}: {n}")
