#!/bin/bash
# Colab one-line execution script for 2-layer repetition training
# Usage: !curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_repetition_2layer.sh | bash

set -e

# ========== Default Parameters ==========
MAX_STAGE=1
EPOCHS_PER_STAGE=10
REPETITIONS=10
BATCH_SIZE=16
LR=0.001
CONVERGENCE_WEIGHT=1.0
TOKEN_WEIGHT=0.01
CONTEXT_DIM=256
EMBED_DIM=256
HIDDEN_DIM=512

# ========== Parse Arguments ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-stage) MAX_STAGE="$2"; shift 2 ;;
        --epochs-per-stage) EPOCHS_PER_STAGE="$2"; shift 2 ;;
        --repetitions) REPETITIONS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --convergence-weight) CONVERGENCE_WEIGHT="$2"; shift 2 ;;
        --token-weight) TOKEN_WEIGHT="$2"; shift 2 ;;
        --context-dim) CONTEXT_DIM="$2"; shift 2 ;;
        --embed-dim) EMBED_DIM="$2"; shift 2 ;;
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo "2-Layer Repetition Training Setup"
echo "========================================="
echo "Max Stage: $MAX_STAGE"
echo "Epochs per Stage: $EPOCHS_PER_STAGE"
echo "Repetitions: $REPETITIONS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "========================================="

# ========== Clone Latest Code ==========
echo ""
echo "📥 Cloning latest code..."
cd /content
rm -rf new-llm
git clone https://github.com/rato-tokyo/new-llm
cd new-llm

# ========== Install Dependencies ==========
echo ""
echo "📦 Installing dependencies..."
pip install -q transformers datasets tqdm

# ========== Start Training in Background ==========
echo ""
echo "🚀 Starting 2-layer repetition training in background..."
nohup python3 scripts/train_repetition_2layer.py \
    --max-stage $MAX_STAGE \
    --epochs-per-stage $EPOCHS_PER_STAGE \
    --repetitions $REPETITIONS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --convergence-weight $CONVERGENCE_WEIGHT \
    --token-weight $TOKEN_WEIGHT \
    --context-dim $CONTEXT_DIM \
    --embed-dim $EMBED_DIM \
    --hidden-dim $HIDDEN_DIM \
    --device cuda \
    --output-dir /content/checkpoints \
    --max-length 512 \
    > /content/train_2layer.log 2>&1 &

# ========== Wait and Show Initial Log ==========
echo ""
echo "⏳ Waiting for training to start (10 seconds)..."
sleep 10

echo ""
echo "========================================="
echo "📊 Initial Training Log (last 30 lines)"
echo "========================================="
tail -30 /content/train_2layer.log

# ========== Instructions ==========
echo ""
echo "========================================="
echo "✅ Training Started Successfully!"
echo "========================================="
echo ""
echo "📋 Monitor training:"
echo "  !tail -30 /content/train_2layer.log"
echo ""
echo "🛑 Stop training:"
echo "  !pkill -9 -f train_repetition_2layer"
echo ""
echo "📁 Output directory:"
echo "  /content/checkpoints/"
echo ""
echo "🔍 Check GPU usage:"
echo "  !nvidia-smi"
echo ""
echo "========================================="
