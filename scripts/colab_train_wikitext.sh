#!/bin/bash
set -e

# Parse arguments
MAX_SAMPLES=""
EPOCHS=30
BATCH_SIZE=32
LR=""
LAYERS=1
CONTEXT_DIM=""
CONTEXT_STRATEGY=""
MAX_LENGTH=""
OUTPUT_DIR="checkpoints"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --context-dim) CONTEXT_DIM="$2"; shift 2 ;;
        --context-update-strategy) CONTEXT_STRATEGY="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 1. 最新版を取得
echo "========================================="
echo "📦 Fetching Latest Code"
echo "========================================="

if [ -d "/content/new-llm/.git" ]; then
    echo "✓ Repository exists, updating with git pull..."
    cd /content/new-llm
    git fetch origin
    git reset --hard origin/main
    git pull origin main
else
    echo "✓ Repository not found, cloning..."
    cd /content
    git clone https://github.com/rato-tokyo/new-llm
    cd new-llm
fi

echo ""

# 2. 依存関係インストール
pip install -q tokenizers datasets tqdm

# 3. 訓練コマンド構築
CMD="python train.py --epochs $EPOCHS --batch-size $BATCH_SIZE --layers $LAYERS --output-dir $OUTPUT_DIR --device cuda"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi
if [ -n "$LR" ]; then
    CMD="$CMD --lr $LR"
fi
if [ -n "$CONTEXT_DIM" ]; then
    CMD="$CMD --context-dim $CONTEXT_DIM"
fi
if [ -n "$CONTEXT_STRATEGY" ]; then
    CMD="$CMD --context-update-strategy $CONTEXT_STRATEGY"
fi
if [ -n "$MAX_LENGTH" ]; then
    CMD="$CMD --max-length $MAX_LENGTH"
fi
LOG_FILE="/content/wikitext_training.log"

# 4. 実験開始（バックグラウンド）
echo ""
echo "========================================="
echo "🚀 Starting Training (Reconstruction Learning)"
echo "========================================="
echo "Command: $CMD"
echo "Log file: $LOG_FILE"
echo ""

nohup $CMD > $LOG_FILE 2>&1 &
PID=$!

# 5. 初期状態表示
sleep 10
tail -30 $LOG_FILE

# 6. モニタリングコマンド表示
echo ""
echo "========================================="
echo "📊 Monitoring Commands"
echo "========================================="
echo "Watch progress:  !tail -20 $LOG_FILE"
echo "Kill training:   !kill -9 $PID"
echo ""
