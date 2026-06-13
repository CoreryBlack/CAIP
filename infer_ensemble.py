#!/usr/bin/env python3
"""
RAICOM 2026 — 多模型集成推理
用法:
  python3 infer_ensemble.py --data-root ./test --models b3 b4 convnext
  python3 infer_ensemble.py --image-path ./image.jpg
"""
import os, sys, argparse
import numpy as np
import cv2

# 推迟导入 PyTorch，避免不必要的显存占用
_imported = False

MODEL_REGISTRY = {
    "b3": {
        "model_name": "efficientnet_b3",
        "image_size": 300,
        "checkpoint": "outputs_efficientnet_b3/best_model.pth",
    },
    "b4": {
        "model_name": "efficientnet_b4",
        "image_size": 380,
        "checkpoint": "outputs_efficientnet_b4/best_model.pth",
    },
    "convnext": {
        "model_name": "convnext_tiny",
        "image_size": 224,
        "checkpoint": "outputs_convnext_tiny/best_model.pth",
    },
}

CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="多模型集成推理")
    p.add_argument("--data-root", type=str, default="", help="测试图片目录（扁平）")
    p.add_argument("--image-path", type=str, default="", help="单张图片路径")
    p.add_argument("--models", nargs="+", default=["b3", "b4", "convnext"],
                   help="参与集成的模型 (b3 b4 convnext)")
    p.add_argument("--output", type=str, default="outputs_ensemble/predictions.csv",
                   help="输出 CSV")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def _lazy_import():
    global torch, F, A, ToTensorV2, WeatherClassifier, _imported
    if not _imported:
        import torch
        import torch.nn.functional as F
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from model import WeatherClassifier
        _imported = True


def build_transform(image_size):
    _lazy_import()
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_ensemble(model_keys, device="cuda"):
    _lazy_import()
    models = {}
    transforms = {}
    for key in model_keys:
        cfg = MODEL_REGISTRY[key]
        ckpt_path = os.path.join(os.path.dirname(__file__), cfg["checkpoint"])
        if not os.path.exists(ckpt_path):
            print(f"[WARN] 未找到检查点: {ckpt_path}，跳过 {key}")
            continue
        model = WeatherClassifier(num_classes=4, model_name=cfg["model_name"])
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        models[key] = model
        transforms[key] = build_transform(cfg["image_size"])
        print(f"[LOAD] {key}: {cfg['model_name']} ({cfg['image_size']}x{cfg['image_size']})")
    return models, transforms


def predict_one(img_bgr, models, transforms):
    """返回 (label, prob_dict)"""
    all_probs = []
    for key, model in models.items():
        t = transforms[key]
        augmented = t(image=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        tensor = augmented["image"].unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    if not all_probs:
        return "unknown", {}
    avg = np.mean(all_probs, axis=0)
    pred = CLASSES[int(np.argmax(avg))]
    prob_dict = {CLASSES[i]: round(float(avg[i]), 6) for i in range(len(CLASSES))}
    return pred, prob_dict


def predict_batch(paths, models, transforms):
    results = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            results.append({"filename": os.path.basename(path), "pred_label": "error",
                            "prob_cloudy": 0, "prob_rainy": 0, "prob_snowy": 0, "prob_sunny": 0})
            continue
        label, probs = predict_one(img, models, transforms)
        results.append({
            "filename": os.path.basename(path),
            "pred_label": label,
            "pred_class_id": CLASSES.index(label) if label in CLASSES else -1,
            "prob_cloudy": probs.get("cloudy", 0),
            "prob_rainy": probs.get("rainy", 0),
            "prob_snowy": probs.get("snowy", 0),
            "prob_sunny": probs.get("sunny", 0),
        })
    return results


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 加载模型
    _lazy_import()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, transforms = load_ensemble(args.models, device)
    if not models:
        print("[ERR] 无可用模型")
        return
    print(f"[INFO] {len(models)} 个模型已加载，设备: {device}")

    # 收集图片
    if args.image_path:
        paths = [args.image_path]
    elif args.data_root:
        paths = sorted([
            os.path.join(args.data_root, f) for f in os.listdir(args.data_root)
            if os.path.splitext(f)[1].lower() in EXTS and os.path.isfile(os.path.join(args.data_root, f))
        ])
    else:
        print("[ERR] 需要 --data-root 或 --image-path")
        return

    print(f"[INFO] 共 {len(paths)} 张图片")

    # 推理
    results = predict_batch(paths, models, transforms)

    # 保存
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[OUT] {args.output}")

    # 统计
    counts = {}
    for r in results:
        counts[r["pred_label"]] = counts.get(r["pred_label"], 0) + 1
    print("[DIST]", counts)


if __name__ == "__main__":
    main()
