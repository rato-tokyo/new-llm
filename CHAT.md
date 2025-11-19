# Chat with New-LLM

対話機能の使い方ガイド

---

## 🚀 クイックスタート

### UltraChat訓練済みモデルで対話

```bash
python scripts/chat.py --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt
```

**これだけで対話開始！**

---

## 📋 使い方

### 基本的な対話

```
You: Hello, how are you?
Assistant: I'm doing well, thank you! How can I help you today?

You: What can you do?
Assistant: I can help you with various tasks...

You: exit
👋 Goodbye!
```

### コマンド

| コマンド | 説明 |
|---------|------|
| `exit` または `quit` | 対話を終了 |
| `reset` | 会話履歴をクリア |
| `settings` | 現在の設定を表示 |

---

## ⚙️ 設定オプション

### 温度（Temperature）

**応答のランダム性を制御**:

```bash
# 保守的な応答（決定論的）
python scripts/chat.py --checkpoint <path> --temperature 0.5

# バランス（デフォルト）
python scripts/chat.py --checkpoint <path> --temperature 0.8

# 創造的な応答（多様性）
python scripts/chat.py --checkpoint <path> --temperature 1.2
```

| Temperature | 特徴 | 用途 |
|-------------|------|------|
| **0.3-0.5** | 保守的、一貫性高い | 事実回答、要約 |
| **0.7-0.9** | バランス | 通常の対話 |
| **1.0-1.5** | 創造的、多様 | ブレインストーミング |

### 最大生成長（Max Length）

**応答の最大トークン数**:

```bash
# 短い応答
python scripts/chat.py --checkpoint <path> --max_length 50

# 長い応答
python scripts/chat.py --checkpoint <path> --max_length 200
```

### Top-p（Nucleus Sampling）

**応答の多様性を制御**:

```bash
# より集中した応答
python scripts/chat.py --checkpoint <path> --top_p 0.7

# より多様な応答
python scripts/chat.py --checkpoint <path> --top_p 0.95
```

---

## 📊 利用可能なモデル

### UltraChat訓練済み（推奨）

```bash
python scripts/chat.py --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt
```

**特徴**:
- ✅ 1.3M対話で訓練
- ✅ 多様なトピック対応
- ✅ 最高性能（PPL 10-14）

### 将来のモデル

```bash
# CodeAlpaca訓練済み（コード生成特化）
python scripts/chat.py --checkpoint checkpoints/best_new_llm_codealpaca_layers1.pt

# MATH訓練済み（数学的推論特化）
python scripts/chat.py --checkpoint checkpoints/best_new_llm_math_layers1.pt
```

---

## 🎯 推奨設定

### 一般的な対話

```bash
python scripts/chat.py \
  --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt \
  --temperature 0.8 \
  --top_p 0.9 \
  --max_length 100
```

### コード生成（将来）

```bash
python scripts/chat.py \
  --checkpoint checkpoints/best_new_llm_codealpaca_layers1.pt \
  --temperature 0.5 \
  --top_p 0.9 \
  --max_length 200
```

### 創造的な対話

```bash
python scripts/chat.py \
  --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt \
  --temperature 1.2 \
  --top_p 0.95 \
  --max_length 150
```

---

## 💡 使用例

### 質問応答

```
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly programmed...
```

### 複数ターン対話

```
You: Tell me about Python
Assistant: Python is a high-level programming language...

You: What are its main features?
Assistant: Python's main features include simple syntax, dynamic typing...

You: Can you give me an example?
Assistant: Sure! Here's a simple Python example...
```

### 文脈を保持した対話

**New-LLMは会話履歴を保持**:
- 前の発言を参照できる
- 一貫した対話が可能
- `reset`コマンドで履歴クリア

---

## 🔧 トラブルシューティング

### 問題: 応答が意味不明

**原因**: モデルが未訓練または訓練中

**解決策**:
- UltraChat訓練が完了するまで待つ
- Epoch 10以降で実用的な性能になる

### 問題: 応答が短すぎる

**解決策**: `--max_length`を増やす
```bash
python scripts/chat.py --checkpoint <path> --max_length 200
```

### 問題: 応答が繰り返す

**解決策**: Temperatureを上げる
```bash
python scripts/chat.py --checkpoint <path> --temperature 1.0
```

### 問題: GPU out of memory

**解決策**: CPUで実行
```bash
python scripts/chat.py --checkpoint <path> --device cpu
```

---

## 📈 性能指標

### 期待される性能（UltraChat訓練後）

| Epoch | PPL | 対話品質 |
|-------|-----|---------|
| 1 | 14.6 | 基本的な応答可能 |
| 10 | ~12 | 実用的な対話 |
| 20 | ~11 | 高品質な対話 |
| 50 | ~10 | 最高性能 |

---

## 🎓 技術詳細

### アーキテクチャ

New-LLMは**Context Vector Propagation**を使用：
- O(1)メモリ使用量
- 任意長のシーケンス処理
- 高速な推論

### サンプリング戦略

実装されているサンプリング:
- **Greedy decoding**: 最も確率の高いトークンを選択
- **Temperature sampling**: 確率分布を調整
- **Top-k sampling**: 上位kトークンから選択
- **Top-p (nucleus) sampling**: 累積確率pまでのトークンから選択

---

## 📚 関連ドキュメント

- `README.md` - プロジェクト概要
- `ULTRACHAT_TRAINING.md` - UltraChat訓練ガイド
- `ARCHITECTURE.md` - New-LLMアーキテクチャ詳細
- `TRAINING_PROGRESSION.md` - データセット難易度順

---

**準備完了！** UltraChat訓練が完了したら、すぐに対話を試せます。

```bash
python scripts/chat.py --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt
```
