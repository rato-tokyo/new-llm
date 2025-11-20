#!/bin/bash
set -e

# Parse arguments
TOP_K=20
START_ID=0
EPOCHS=5
REPETITIONS=50
BATCH_SIZE=4
LR=0.001
CONTEXT_DIM=256
EMBED_DIM=256
HIDDEN_DIM=512
LAYERS=2
MAX_LENGTH=512
OUTPUT_DIR="checkpoints/cvfpt_topk"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --top-k) TOP_K="$2"; shift 2 ;;
        --start-id) START_ID="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --repetitions) REPETITIONS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --context-dim) CONTEXT_DIM="$2"; shift 2 ;;
        --embed-dim) EMBED_DIM="$2"; shift 2 ;;
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
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
# -u: unbuffered output (ログが即座に書き込まれる)
CMD="python -u scripts/train_repetition_topk.py"
CMD="$CMD --top-k $TOP_K"
CMD="$CMD --start-id $START_ID"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --repetitions $REPETITIONS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --lr $LR"
CMD="$CMD --context-dim $CONTEXT_DIM"
CMD="$CMD --embed-dim $EMBED_DIM"
CMD="$CMD --hidden-dim $HIDDEN_DIM"
CMD="$CMD --layers $LAYERS"
CMD="$CMD --max-length $MAX_LENGTH"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --device $DEVICE"

LOG_FILE="/content/cvfpt_topk_training.log"

# 4. 実験開始（バックグラウンド）
echo ""
echo "========================================="
echo "🚀 Starting CVFPT Top-K Token Evaluation"
echo "========================================="
echo "Top-K Tokens: $TOP_K (starting from ID $START_ID)"
echo "Epochs per Token: $EPOCHS"
echo "Repetitions: $REPETITIONS"
echo ""
echo "Purpose: Measure CVFPT convergence performance on tokens"
echo ""
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
echo "Kill training:   !pkill -9 -f train_repetition_topk.py"
echo "View results:    !cat checkpoints/cvfpt_topk/cvfpt_performance_topk.json"
echo ""
