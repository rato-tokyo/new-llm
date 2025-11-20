#!/bin/bash
set -e

# Parse arguments
MAX_STAGE=3
EPOCHS_PER_STAGE=10
REPETITIONS=100
BATCH_SIZE=8
LR=0.001
CONTEXT_DIM=256
EMBED_DIM=256
HIDDEN_DIM=512
LAYERS=2
CONVERGENCE_WEIGHT=1.0
TOKEN_WEIGHT=0.0
MAX_LENGTH=512
OUTPUT_DIR="checkpoints"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-stage) MAX_STAGE="$2"; shift 2 ;;
        --epochs-per-stage) EPOCHS_PER_STAGE="$2"; shift 2 ;;
        --repetitions) REPETITIONS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --context-dim) CONTEXT_DIM="$2"; shift 2 ;;
        --embed-dim) EMBED_DIM="$2"; shift 2 ;;
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --convergence-weight) CONVERGENCE_WEIGHT="$2"; shift 2 ;;
        --token-weight) TOKEN_WEIGHT="$2"; shift 2 ;;
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
pip install -q tokenizers tqdm

# 3. 訓練コマンド構築
# -u: unbuffered output (ログが即座に書き込まれる)
CMD="python -u scripts/train_repetition.py"
CMD="$CMD --max-stage $MAX_STAGE"
CMD="$CMD --epochs-per-stage $EPOCHS_PER_STAGE"
CMD="$CMD --repetitions $REPETITIONS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --lr $LR"
CMD="$CMD --context-dim $CONTEXT_DIM"
CMD="$CMD --embed-dim $EMBED_DIM"
CMD="$CMD --hidden-dim $HIDDEN_DIM"
CMD="$CMD --layers $LAYERS"
CMD="$CMD --convergence-weight $CONVERGENCE_WEIGHT"
CMD="$CMD --token-weight $TOKEN_WEIGHT"
CMD="$CMD --max-length $MAX_LENGTH"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --device $DEVICE"

LOG_FILE="/content/repetition_training.log"

# 4. 実験開始（バックグラウンド）
echo ""
echo "========================================="
echo "🚀 Starting Repetition Training"
echo "========================================="
echo "Max Stage: $MAX_STAGE (1=single token, 2=two tokens, ...)"
echo "Epochs per Stage: $EPOCHS_PER_STAGE"
echo "Repetitions: $REPETITIONS"
echo "Convergence Weight: $CONVERGENCE_WEIGHT (context convergence loss)"
echo "Token Weight: $TOKEN_WEIGHT (token prediction loss, usually 0)"
echo ""
echo "Hypothesis: context(n) ≈ context(n+1) for repeated patterns"
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
echo "Kill training:   !pkill -9 -f train_repetition.py"
echo ""
