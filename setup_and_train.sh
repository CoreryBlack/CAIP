#!/usr/bin/env bash
# RAICOM 2026 — 服务器端一键准备 + 训练
# 用法: bash setup_and_train.sh [--dry-run]
set -e

export HF_ENDPOINT=https://hf-mirror.com
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-***REDACTED_API_KEY***}"

DRY=""
if [[ "$1" == "--dry-run" ]]; then
    DRY="--dry-run"
    echo "[DRY RUN] 不写入文件"
fi

# ================================================================
# Step 1: 修正验证集 sand_storm/dusttornado sunny→cloudy
# ================================================================
echo ""
echo "============================================================"
echo " Step 1: 修正 sand_storm/dusttornado (sunny → cloudy)"
echo "============================================================"
python3 -c "
import os, shutil
val = 'training_data_1/val'
keep_in_sunny = {'sand_storm-326.jpg'}
fixes = 0
for fname in os.listdir(f'{val}/sunny'):
    base = fname.lower()
    if (base.startswith('sand_storm') or base.startswith('dusttornado')) and fname not in keep_in_sunny:
        src = f'{val}/sunny/{fname}'
        dst = f'{val}/cloudy/{fname}'
        if '$DRY' != '':
            print(f'  [DRY] {fname}: sunny → cloudy')
        else:
            shutil.move(src, dst)
            print(f'  ✅ {fname}: sunny → cloudy')
        fixes += 1
print(f'  共修正: {fixes} 张')
for cls in ['cloudy','rainy','snowy','sunny']:
    n = len([f for f in os.listdir(f'{val}/{cls}') if f.endswith('.jpg')])
    print(f'    {cls}: {n}')
"

# ================================================================
# Step 2: Qwen API 分类 confusing 样本
# ================================================================
echo ""
echo "============================================================"
echo " Step 2: Qwen API 分类 confusing 样本 (196 张)"
echo "============================================================"
if [[ -n "$DRY" ]]; then
    echo "[DRY RUN] 跳过 API 调用"
else
    cd cleaning
    python3 -u fix_val_add_confusing.py --delay 0.2
    cd ..
fi

# ================================================================
# Step 3: 补充 sunny 和 snowy 使验证集平衡
# ================================================================
echo ""
echo "============================================================"
echo " Step 3: 验证集平衡 (sunny + snowy)"
echo "============================================================"
python3 -c "
import os, random, shutil
random.seed(42)
base = 'dataset_cleaned'
if '$DRY' != '':
    print('[DRY] sunny: 训练集→验证集')
    print('[DRY] snowy: 训练集→验证集')
else:
    # sunny: 补充到 100
    train_s = [f for f in os.listdir(f'{base}/sunny') if f.lower().endswith(('.jpg','.jpeg')) and not f.lower().startswith('sand_storm')]
    val_s = len([f for f in os.listdir(f'{base}/val/sunny') if f.endswith('.jpg')])
    need_s = max(0, 100 - val_s)
    for f in random.sample(train_s, min(need_s, len(train_s))):
        shutil.move(os.path.join(f'{base}/sunny', f), os.path.join(f'{base}/val/sunny', f))
    print(f'sunny: {val_s} → {val_s + need_s}')

    # snowy: 补充到 100
    train_w = [f for f in os.listdir(f'{base}/snowy') if f.lower().endswith(('.jpg','.jpeg'))]
    val_w = len([f for f in os.listdir(f'{base}/val/snowy') if f.endswith('.jpg')])
    need_w = max(0, 100 - val_w)
    for f in random.sample(train_w, min(need_w, len(train_w))):
        shutil.move(os.path.join(f'{base}/snowy', f), os.path.join(f'{base}/val/snowy', f))
    print(f'snowy: {val_w} → {val_w + need_w}')

# 汇总
total = 0
for c in ['cloudy','rainy','snowy','sunny']:
    n = len([f for f in os.listdir(f'{base}/val/{c}') if f.endswith('.jpg')])
    print(f'  {c}: {n}')
    total += n
print(f'  总计: {total}')
"

# ================================================================
# Step 4: 训练
# ================================================================
echo ""
echo "============================================================"
echo " Step 4: 启动训练"
echo "============================================================"
if [[ -n "$DRY" ]]; then
    echo "[DRY RUN] 完整流程完成，未训练"
    exit 0
fi

echo "数据集: dataset_cleaned (训练集 ~7000 / 验证集 534)"
echo ""

python3 train.py \
    --data-root ./dataset_cleaned \
    --batch-size 128 \
    --epochs 60 \
    --lr 1e-3 \
    --aug-strength 1.0 \
    --mixup-alpha 0.2 \
    --cutmix-prob 0.5
