#!/usr/bin/env python3
"""用旧模型(Val F1=0.8517)对数据集逐张推理，移动预测不一致的图片"""
import os, sys, shutil
import torch, cv2, numpy as np, albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

SRC_DIR = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, SRC_DIR)
from model import WeatherClassifier

CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/best_model.pth"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "dataset_cleaned"
CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 加载模型
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
sd = {k.replace("module.", ""): v for k, v in sd.items()}
model = WeatherClassifier(num_classes=4)
model.load_state_dict(sd, strict=False)
model.cuda().eval()

transform = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def predict(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = transform(image=img)["image"].unsqueeze(0).cuda()
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1).cpu().numpy()[0]
    return CLASSES[int(np.argmax(probs))], float(np.max(probs))

# 遍历训练集 + 验证集
for scope in [".", "val"]:
    root = os.path.join(TARGET, scope) if scope != "." else TARGET
    print(f"\n=== {scope if scope != '.' else 'train'} ===")
    moved = 0
    kept = 0
    for cls in CLASSES:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if os.path.splitext(fname)[1].lower() not in EXTS:
                continue
            path = os.path.join(d, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"  [SKIP] 读取失败: {path}")
                continue
            pred, conf = predict(img)
            if pred != cls:
                dst_dir = os.path.join(root, pred)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(path, os.path.join(dst_dir, fname))
                moved += 1
                if conf > 0.7:
                    print(f"  [MOVE] {cls:8s} -> {pred:8s} conf={conf:.2f}  {fname}")
            else:
                kept += 1
    print(f"  moved={moved}  kept={kept}")

# 汇总
print(f"\n=== 清洗后分布 ===")
for scope in [".", "val"]:
    root = os.path.join(TARGET, scope) if scope != "." else TARGET
    label = "train" if scope == "." else "val"
    print(f"{label}:")
    for cls in CLASSES:
        d = os.path.join(root, cls)
        n = len([f for f in os.listdir(d) if f.endswith(tuple(EXTS))]) if os.path.isdir(d) else 0
        print(f"  {cls}: {n}")
