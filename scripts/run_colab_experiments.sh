#!/bin/bash
# Google Colab実験自動実行スクリプト
# 実験1（FP16）と実験2（Context 1024）を同時実行

set -e

echo "=================================="
echo "Colab実験セットアップ開始"
echo "=================================="

# セットアップ
cd /content
rm -rf new-llm
git clone https://github.com/rato-tokyo/new-llm
cd new-llm
pip install -q datasets

echo ""
echo "=================================="
echo "実験1: FP16訓練（Baseline）開始"
echo "=================================="
nohup python3 scripts/train_wikitext_fp16.py > /content/fp16_log.txt 2>&1 &
FP16_PID=$!
echo "PID: $FP16_PID"

sleep 3

echo ""
echo "=================================="
echo "実験2: Context 1024実験準備"
echo "=================================="

# Context vector dimを1024に変更
sed -i 's/context_vector_dim = 512/context_vector_dim = 1024/' scripts/train_wikitext_advanced.py
echo "✓ Context vector dim = 1024に設定"

echo ""
echo "=================================="
echo "実験2: Context 1024訓練開始"
echo "=================================="
nohup python3 scripts/train_wikitext_advanced.py > /content/ctx1024_log.txt 2>&1 &
CTX1024_PID=$!
echo "PID: $CTX1024_PID"

sleep 5

echo ""
echo "=================================="
echo "実験開始完了！"
echo "=================================="
echo "実験1 (FP16): PID $FP16_PID"
echo "実験2 (Ctx1024): PID $CTX1024_PID"
echo ""
echo "進捗確認コマンド:"
echo "  tail -30 /content/fp16_log.txt"
echo "  tail -30 /content/ctx1024_log.txt"
echo ""
echo "GPU確認コマンド:"
echo "  nvidia-smi"
echo ""

# 初期ログ表示
echo "【実験1: FP16 - 初期ログ】"
tail -30 /content/fp16_log.txt 2>/dev/null || echo "ログ生成中..."

echo ""
echo "【実験2: Context 1024 - 初期ログ】"
tail -30 /content/ctx1024_log.txt 2>/dev/null || echo "ログ生成中..."

echo ""
echo "=================================="
echo "✓ 2つの実験が並行実行中"
echo "完了予定: 約30-40分後"
echo "=================================="
