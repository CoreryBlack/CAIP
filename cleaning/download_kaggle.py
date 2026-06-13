"""
下载 Kaggle multiclass-weather-dataset 并映射到4类
"""
import kagglehub
import os, glob, shutil, sys
sys.stdout.reconfigure(encoding="utf-8")

path = kagglehub.dataset_download("vijaygiitk/multiclass-weather-dataset")
src = os.path.join(path, "dataset")
dst = "training_data_2/kaggle_processed"
for c in ["cloudy","rainy","snowy","sunny"]:
    os.makedirs(f"{dst}/{c}", exist_ok=True)

for f in glob.glob(f"{src}/cloudy/*") + glob.glob(f"{src}/foggy/*"):
    shutil.copy2(f, f"{dst}/cloudy/")
for f in glob.glob(f"{src}/rainy/*"):
    shutil.copy2(f, f"{dst}/rainy/")
for f in glob.glob(f"{src}/shine/*") + glob.glob(f"{src}/sunrise/*"):
    shutil.copy2(f, f"{dst}/sunny/")

for c in ["cloudy","rainy","snowy","sunny"]:
    n = len(glob.glob(f"{dst}/{c}/*"))
    print(f"  {c}: {n}")
