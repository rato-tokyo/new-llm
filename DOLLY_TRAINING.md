# Dolly-15k Training for Dialog Model

## Overview

Dolly-15kは15,000件のインストラクション-応答ペアを含むデータセットです。
WikiText-2で訓練された言語モデルベースラインから、対話能力を獲得するための訓練を行います。

## Google Colab での実行方法

### 1. 環境セットアップ（最初に1回だけ）

```python
# GPU確認
!nvidia-smi

# リポジトリクローン
%cd /content
!rm -rf new-llm
!git clone https://github.com/rato-tokyo/new-llm
%cd new-llm

# 依存関係インストール
!pip install -q datasets
```

### 2. Dolly-15k訓練開始（Layer 1）

**バックグラウンドで実行**:

```bash
nohup python3 scripts/train_dolly.py --num_layers 1 > /content/dolly_layer1.log 2>&1 &
```

**ログ確認**:

```bash
!tail -20 /content/dolly_layer1.log
```

**GPU使用状況確認**:

```bash
!nvidia-smi
```

### 3. 進捗確認コマンド

**リアルタイムログ表示**:

```bash
!tail -f /content/dolly_layer1.log
```

**訓練プロセス確認**:

```bash
!ps aux | grep "python.*train_dolly" | grep -v grep
```

### 4. 訓練停止

```bash
!pkill -9 -f "python.*train_dolly"
```

---

## 実験設定

| パラメータ | 値 | 説明 |
|----------|-----|------|
| **Dataset** | Dolly-15k | 15,000 instruction-response pairs |
| **Num Layers** | 1 | Layer 1から開始 |
| **Max Seq Length** | 128 | WikiText-2の2倍（対話用） |
| **Batch Size** | 2048 | L4 GPU最適化 |
| **Learning Rate** | 0.0008 | Square Root Scaling Rule |
| **Epochs** | 100 | WikiText-2より多め |
| **Patience** | 20 | Early stopping |

---

## データセット形式

Dolly-15kの例:

```
Instruction: What is photosynthesis?
Context: (optional)
Response: Photosynthesis is the process by which green plants...
```

訓練データ形式:

```
"Instruction: ... Context: ... Response: ..."
```

---

## 期待される結果

| 指標 | 期待値 | 備考 |
|-----|--------|------|
| **訓練時間** | ~30-40分 | L4 GPU, 100 epochs |
| **Val PPL** | 15-25 | WikiText-2ベースライン（20.5）より低い可能性 |
| **Val Acc** | 35-45% | インストラクションデータで向上期待 |

---

## 次のステップ

1. **Layer 1完了後**: Layer 4でも試す（WikiText-2で最も性能が良かった）
2. **Context Expansion**: 文脈ベクトル256→512次元に拡張
3. **日本語対応**: Japanese Alpacaでファインチューニング

---

## トラブルシューティング

### GPU Out of Memory

batch_sizeを減らす:

```bash
# config.py を編集してbatch_size=1024に変更
!sed -i 's/batch_size = 2048/batch_size = 1024/' src/utils/config.py
```

### 訓練が進まない

学習率を調整:

```python
# train_dolly.pyのDollyTrainConfigクラスで
# learning_rate = 0.0008 → 0.0004 に変更
```

---

## 参考

- WikiText-2訓練結果: `experiments/layer_optimization_experiment_2025-11-18.md`
- Layer 1 WikiText-2: PPL 20.4-20.5 (Epoch 75時点)
- Layer 4 WikiText-2: PPL 20.1-20.2 (最も有望)
