# HH-RLHF Training Guide

Anthropic HH-RLHF（Human-Human RLHF）での訓練ガイド

---

## 📊 HH-RLHFとは

**高品質な人間同士の対話データ**

| 特徴 | 詳細 |
|-----|------|
| **データ量** | 85,000件（Helpful: 43k + Harmless: 42k） |
| **形式** | 複数ターン対話（Multi-turn） |
| **言語** | 英語のみ |
| **品質** | 人間フィードバック付き |
| **作成元** | Anthropic |

---

## 🎯 Dolly-15kとの比較

| 項目 | Dolly-15k | HH-RLHF |
|-----|-----------|---------|
| **データ量** | 15,000 | **85,000（5.7倍）** |
| **会話形式** | 単発Q&A | **複数ターン** |
| **文脈理解** | 不要 | **必須** |
| **品質** | 高い | **非常に高い** |
| **期待PPL** | 15.6（実測） | **17-20** |

**HH-RLHFの方が難しい**が、より実践的な対話能力を獲得できます。

---

## 🚀 Google Colabでの実行方法

### ステップ1: 環境セットアップ

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

### ステップ2: HH-RLHF訓練開始（Layer 1）

```bash
# バックグラウンドで実行
!nohup python3 scripts/train_hh_rlhf.py --num_layers 1 > /content/hh_rlhf_layer1.log 2>&1 &

# ログ確認
!tail -20 /content/hh_rlhf_layer1.log

# GPU使用状況
!nvidia-smi
```

### ステップ3: Layer 4で訓練（推奨）

```bash
# Layer 4はWikiText-2で最も性能が良かった
!nohup python3 scripts/train_hh_rlhf.py --num_layers 4 > /content/hh_rlhf_layer4.log 2>&1 &

# ログ確認
!tail -20 /content/hh_rlhf_layer4.log
```

---

## 📈 期待される結果

### Layer 1

| 指標 | 期待値 |
|-----|--------|
| **Val PPL** | **18-20** |
| **Val Acc** | **43-45%** |
| **訓練時間** | 20-30分（100 epochs） |

### Layer 4（推奨）

| 指標 | 期待値 |
|-----|--------|
| **Val PPL** | **16-18** |
| **Val Acc** | **45-47%** |
| **訓練時間** | 30-40分（100 epochs） |

**Dolly-15kとの比較**:
- Dolly Layer 1: PPL 15.6
- HH-RLHF Layer 1: PPL 18-20（+2-4ポイント難しい）

---

## ⚙️ 設定

### 自動設定（デフォルト）

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| **batch_size** | 2048 | L4 GPU最適化 |
| **learning_rate** | 0.0008 | Square Root Scaling |
| **max_seq_length** | 128 | 複数ターン対話用 |
| **epochs** | 100 | HH-RLHF用 |
| **context_dim** | 256 | 標準 |

### カスタマイズ

```bash
# より長いシーケンス
python scripts/train_hh_rlhf.py --num_layers 4 --max_seq_length 256

# Harmless subset（安全性重視）
python scripts/train_hh_rlhf.py --num_layers 4 --subset harmless
```

---

## 🔍 進捗モニタリング

### リアルタイムログ表示

```bash
# 最新20行
!tail -20 /content/hh_rlhf_layer1.log

# 継続的な監視
!tail -f /content/hh_rlhf_layer1.log
```

### ✓マークの意味

```
Epoch 50/100
  Training... 100% | 0.2min | Loss: 2.85
  Val: Loss=2.82 PPL=16.8 Acc=45.1% ✓
  [Checkpoint saved]
  ↑
  ✓ = このEpochがベストモデル
```

### GPU使用状況

```bash
!nvidia-smi
```

---

## 📊 結果の解釈

### 成功の指標

| PPL範囲 | 評価 | 次のステップ |
|---------|------|-------------|
| **< 18** | 🏆 **優秀** | Level 3（UltraChat）へ |
| **18-21** | ✅ **成功** | Layer 4で再挑戦 or Level 3へ |
| **> 21** | ⚠️ **要改善** | 設定調整 or Layer増やす |

### Dollyとの比較

**期待されるパターン**:
```
Dolly-15k (Layer 1): PPL 15.6 ← 構造化データで簡単
HH-RLHF (Layer 1): PPL 18-20 ← 複数ターンで難しい
```

**これは正常です！** HH-RLHFの方が難しいタスクです。

---

## 🎯 何を学習するか

### Dolly-15kで学んだこと

- 単発Q&Aの応答
- 基本的なInstruction理解
- 明確なパターン認識

### HH-RLHFで学ぶこと

- **複数ターン対話**: 前の発言を踏まえた応答
- **文脈保持**: 会話の流れを理解
- **高品質応答**: 人間が好む応答パターン
- **安全性**: 有害な応答を避ける

---

## ⚡ トラブルシューティング

### GPU Out of Memory

```bash
# batch_sizeを減らす
# src/utils/config.pyのNewLLML4Configを編集
batch_size = 1024  # 2048 → 1024
```

### 訓練が遅い

```bash
# Layer数を減らす
python scripts/train_hh_rlhf.py --num_layers 1  # 4 → 1
```

### PPLが下がらない

1. **Epochを増やす**: 100 → 150
2. **Learning rateを下げる**: 0.0008 → 0.0004
3. **Layer 4で試す**: より深いモデル

---

## 📁 保存されるファイル

| ファイル | 説明 |
|---------|------|
| `checkpoints/best_new_llm_hh_rlhf_layers1.pt` | ベストモデル |
| `new_llm_hh_rlhf_layers1_final.pt` | 最終Epoch |
| `/content/hh_rlhf_layer1.log` | 訓練ログ |

---

## 🚀 訓練完了後

### 結果の保存

```bash
# Colabからダウンロード
from google.colab import files
files.download('/content/new-llm/checkpoints/best_new_llm_hh_rlhf_layers1.pt')
files.download('/content/hh_rlhf_layer1.log')
```

### 次のステップ

**成功したら（PPL < 21）**:

1. **Layer 4で試す**: さらに性能向上
2. **Level 3へ進む**: UltraChat（大規模対話）
3. **Context Expansion**: 256→512次元

---

## 📖 関連ドキュメント

- `TRAINING_PROGRESSION.md` - データセット難易度順
- `experiments/dolly_dialog_experiment_2025-11-19.md` - Dolly結果
- `ARCHITECTURE.md` - New-LLMアーキテクチャ

---

**準備完了！** HH-RLHF訓練を開始してください。

```bash
python scripts/train_hh_rlhf.py --num_layers 1
```
