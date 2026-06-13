#!/usr/bin/env bash
# RAICOM 2026 — Linux 训练启动脚本
# 用法: bash train.sh [--large] [--tiny]

set -e

# 默认数据目录
DATA_ROOT="${DATA_ROOT:-./dataset_cleaned}"
BATCH_SIZE=32
EPOCHS=60
MIXUP="--mixup-alpha 0.2 --cutmix-prob 0.5"
AUG=1.0
LR=1e-3
RESUME=""
FREEZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --large)
            DATA_ROOT="./dataset_cleaned"
            MIXUP="--mixup-alpha 0.2 --cutmix-prob 0.5"
            ;;
        --tiny)
            DATA_ROOT="./training_data_1"
            AUG=0.8
            LR=5e-4
            MIXUP="--mixup-alpha 0.0"
            EPOCHS=30
            ;;
        --resume)
            RESUME="--resume $2"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash train.sh [--large|--tiny] [--resume ./outputs/checkpoint.pth]"
            exit 1
            ;;
    esac
    shift
done

echo "============================================"
echo " RAICOM 2026 天气分类 — 训练脚本"
echo "============================================"
echo "数据目录:  $DATA_ROOT"
echo "Batch:     $BATCH_SIZE"
echo "Epochs:    $EPOCHS"
echo "LR:        $LR"
echo "Aug:       $AUG"
echo "Mixup:     $MIXUP"
echo "Resume:    $RESUME"
echo "Freeze:    $FREEZE"
echo "============================================"
echo ""

python train.py \
    --data-root "$DATA_ROOT" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --aug-strength $AUG \
    $MIXUP \
    $RESUME \
    $FREEZE
