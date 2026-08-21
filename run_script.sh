#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/SSD2/pengzhipeng/anaconda3/envs/overlock/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------- User configurable parameters ----------------
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-6}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR=1e-4
TRAINSIZE=352
TRAIN_PATH="/HDD/pengzhipeng/dataset/TrainDataset"  # 数据集根目录（images/ 和 masks/ 所在路径）
SPLIT_DIR="./utils/TrainDataset"                     # train.txt / val.txt 所在目录
VALID_INTERVAL=1
PATIENCE="${PATIENCE:-10}"
# -------------------------------------------------------------

# -------- 1. 训练 --------
echo "Start training..."
"$PYTHON_BIN" MyTrain.py \
    --epoch "$EPOCHS" \
    --batchsize "$BATCH_SIZE" \
    --lr "$LR" \
    --trainsize "$TRAINSIZE" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --train-path "$TRAIN_PATH" \
    --split-dir "$SPLIT_DIR" \
    --valid-interval "$VALID_INTERVAL" \
    --patience "$PATIENCE" \
    --use-augmentation

echo "Training finished."

# -------- 2. 测试 --------
echo "Start testing..."
"$PYTHON_BIN" MyTest.py --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
echo "Testing finished."

# -------- 3. 评估 --------
echo "Start evaluation..."
"$PYTHON_BIN" MyEval.py
echo "Evaluation finished."

echo "All steps completed."
