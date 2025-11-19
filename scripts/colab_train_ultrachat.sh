#!/bin/bash
set -e

# ========================================
# UltraChat Training - One-Line Colab Script
# ========================================
# Usage:
#   curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash
#
# Or with parameters:
#   curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash -s -- --num_layers 4 --max_samples 100000
# ========================================

# Default parameters
NUM_LAYERS=1
MAX_SAMPLES=""  # Empty = full dataset

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "UltraChat Training - Colab Auto Setup"
echo "========================================="
echo "Parameters:"
echo "  Layers: $NUM_LAYERS"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Max samples: $MAX_SAMPLES"
else
    echo "  Max samples: Full dataset (1.5M+)"
fi
echo "========================================="

# 1. Navigate to /content
cd /content

# 2. Remove old repository
echo "🗑️  Removing old repository..."
rm -rf new-llm

# 3. Clone latest version
echo "📥 Cloning latest version from GitHub..."
git clone https://github.com/rato-tokyo/new-llm
cd new-llm

# 4. Install dependencies
echo "📦 Installing dependencies..."
pip install -q datasets tqdm

# 5. Build training command
CMD="python scripts/train_ultrachat.py --num_layers $NUM_LAYERS"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
    LOG_FILE="/content/ultrachat_layer${NUM_LAYERS}_${MAX_SAMPLES}.log"
else
    LOG_FILE="/content/ultrachat_layer${NUM_LAYERS}_full.log"
fi

# 6. Start training in background
echo "🚀 Starting training..."
echo "   Command: $CMD"
echo "   Log file: $LOG_FILE"
nohup $CMD > $LOG_FILE 2>&1 &

# 7. Wait for initialization
echo "⏳ Waiting for initialization (10 seconds)..."
sleep 10

# 8. Show initial log
echo ""
echo "========================================="
echo "📊 Initial Training Log"
echo "========================================="
tail -30 $LOG_FILE

# 9. Show monitoring commands
echo ""
echo "========================================="
echo "✅ Training Started!"
echo "========================================="
echo ""
echo "📋 Monitoring commands:"
echo "  !tail -20 $LOG_FILE              # View latest 20 lines"
echo "  !tail -f $LOG_FILE               # Real-time monitoring"
echo "  !nvidia-smi                      # GPU status"
echo "  !ps aux | grep train_ultrachat   # Process status"
echo ""
echo "🛑 Stop training:"
echo "  !pkill -9 -f train_ultrachat"
echo ""
if [ -n "$MAX_SAMPLES" ]; then
    echo "⏱️  Estimated time: 20-40 minutes (subset)"
else
    echo "⏱️  Estimated time: 2.5-3.5 hours (full dataset)"
fi
echo "========================================="
