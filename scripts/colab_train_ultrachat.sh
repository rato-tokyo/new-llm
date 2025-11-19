#!/bin/bash
set -e

# Parse arguments
MAX_SAMPLES=""
EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=""
LAYERS=""
OUTPUT_DIR="checkpoints/ultrachat"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 1. 最新版を取得（リポジトリの有無で自動判定）
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
pip install -q transformers tokenizers datasets tensorboard

# 3. 訓練コマンド構築
CMD="python train.py --dataset ultrachat --epochs $EPOCHS --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi
if [ -n "$LEARNING_RATE" ]; then
    CMD="$CMD --learning-rate $LEARNING_RATE"
fi
if [ -n "$LAYERS" ]; then
    CMD="$CMD --layers $LAYERS"
fi
LOG_FILE="/content/ultrachat_training.log"

# 4. 実験開始（バックグラウンド）
echo ""
echo "========================================="
echo "🚀 Starting Training"
echo "========================================="
echo "Command: $CMD"
echo "Log file: $LOG_FILE"
echo ""

nohup $CMD > $LOG_FILE 2>&1 &
PID=$!

# 6. 初期状態表示
sleep 10
tail -30 $LOG_FILE

# 7. モニタリングコマンド表示
echo ""
echo "========================================="
echo "📊 Monitoring Commands"
echo "========================================="
echo "Watch progress: tail -f $LOG_FILE"
echo "Kill training: kill -9 $PID"
echo ""
