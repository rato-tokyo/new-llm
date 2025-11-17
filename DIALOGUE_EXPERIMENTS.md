# Dialogue Training Experiments - Setup Guide

## 🎯 目標

New-LLMを会話レベルに引き上げる実験。

**実装する施策**:
1. WikiText-2での事前学習（実テキストで自然な文章の流れを学習）
2. DailyDialogでのファインチューニング（対話パターンを学習）
3. TinyGPT2との比較（Attention vs Context Vector）

## 📋 前提条件

### 必要なライブラリのインストール

```bash
cd /Users/sakajiritomoyoshi/Desktop/git/new-llm
pip install -r requirements.txt
```

新たに追加されたライブラリ:
- `datasets>=2.14.0` - HuggingFace Datasets（WikiText-2, DailyDialog）
- `transformers>=4.30.0` - GPT-2モデル

## 🚀 実験手順

### ステップ1: WikiText-2事前学習

**目的**: 実際のWikipedia記事で自然な文章の流れを学習

```bash
python scripts/train_wikitext.py
```

**実行内容**:
1. WikiText-2データセットを自動ダウンロード
2. New-LLMを訓練（100エポック）
3. TinyGPT2を訓練（比較用）
4. 両モデルのPerplexityを比較

**出力ファイル**:
- `checkpoints/best_new_llm_wikitext.pt` - New-LLMの事前学習済みモデル
- `checkpoints/best_tinygpt2_wikitext.pt` - TinyGPT2の事前学習済みモデル
- `checkpoints/new_llm_wikitext_training_curves.png` - 訓練曲線
- `checkpoints/tinygpt2_wikitext_training_curves.png` - 訓練曲線

**予想実行時間**: 1-2時間（CPUの場合）

**期待される結果**:
- Perplexity: ランダムデータ（21.41）→ WikiText（10-15程度）に改善

---

### ステップ2: DailyDialog対話ファインチューニング

**目的**: 日常会話のパターンを学習

```bash
python scripts/finetune_dialog.py
```

**実行内容**:
1. DailyDialogデータセット（13k対話）を自動ダウンロード
2. ステップ1の事前学習済みモデルをロード
3. 対話データでファインチューニング（50エポック）
4. 両モデルを比較

**出力ファイル**:
- `checkpoints/best_new_llm_dialog.pt` - 対話ファインチューニング済みモデル
- `checkpoints/best_tinygpt2_dialog.pt` - TinyGPT2対話モデル
- `checkpoints/new_llm_dialog_training_curves.png`
- `checkpoints/tinygpt2_dialog_training_curves.png`

**予想実行時間**: 30分-1時間（CPUの場合）

**期待される結果**:
- 対話タスクに特化したPerplexity改善
- New-LLMとTinyGPT2の性能差の把握

---

### ステップ3: 評価と比較

**目的**: 両モデルの生成品質を定量的・定性的に評価

```bash
python scripts/evaluate_comparison.py
```

**実行内容**:
1. Perplexity評価（テストセット）
2. サンプル対話生成（プロンプトからの応答）
3. 推論速度測定

**出力例**:
```
Prompt: 'hello how are you'
New-LLM:   hello how are you doing today
TinyGPT2:  hello how are you ? i am fine

Perplexity:
  New-LLM:   15.32
  TinyGPT2:  12.45
  Difference: +23.1%

Inference Speed:
  New-LLM:   45.23 ms/batch
  TinyGPT2:  52.18 ms/batch
  Difference: -13.3% (New-LLM is faster!)
```

**予想実行時間**: 5-10分

---

## 📊 評価指標

### 1. Perplexity（定量評価）
- 低いほど良い
- 言語モデルの予測精度を表す
- **目標**: WikiText事前学習でPerplexity 10-15に到達

### 2. 生成品質（定性評価）
- プロンプトに対する応答の自然さ
- 文法の正確性
- 文脈の一貫性

### 3. 推論速度
- ms/batch（ミリ秒/バッチ）
- New-LLMの方が軽量なので高速な可能性あり

---

## 🔬 実験設計

### New-LLM vs TinyGPT2 の公平な比較

| 項目 | New-LLM | TinyGPT2 |
|------|---------|----------|
| **アーキテクチャ** | Context Vector Propagation | Multi-head Attention |
| **パラメータ数** | ~4M | ~4M（New-LLMと揃えた） |
| **Attention** | なし | あり |
| **処理方式** | 逐次処理 | 並列処理 |
| **メモリ** | O(1) - 固定 | O(n²) - シーケンス長依存 |

### データセットの段階的拡張

| 段階 | データセット | サイズ | 語彙数 |
|------|-------------|--------|--------|
| **初期** | ランダム生成 | 500文 | 252語 |
| **ステップ1** | WikiText-2 | ~2MB | 数千語 |
| **ステップ2** | DailyDialog | 13k対話 | 数千語 |

---

## 🐛 トラブルシューティング

### エラー: `ImportError: No module named 'datasets'`

```bash
pip install datasets transformers
```

### エラー: メモリ不足

`batch_size`を小さくする:

```python
# scripts/train_wikitext.py の WikiTextConfig
batch_size = 8  # 16 → 8 に変更
```

### エラー: `FileNotFoundError: Checkpoint not found`

ステップ1（WikiText事前学習）を先に実行してください:

```bash
python scripts/train_wikitext.py
```

### WikiText/DailyDialogのダウンロードが遅い

初回のみHuggingFaceからダウンロードされます。
キャッシュされるので2回目以降は高速です。

---

## 📈 予想される実験結果

### Phase 1: WikiText事前学習

**仮説**: 実テキストで訓練することで、ランダム生成データより大幅改善

| モデル | Perplexity (before) | Perplexity (after) | 改善率 |
|--------|---------------------|--------------------| ------|
| New-LLM | 21.41 | 10-15（予想） | -30~50% |
| TinyGPT2 | - | 8-12（予想） | - |

**期待**: Attentionの有無で差が出るが、両者とも大幅改善

---

### Phase 2: 対話ファインチューニング

**仮説**: 対話データでさらに特化した性能向上

| モデル | 対話Perplexity | 生成品質 |
|--------|---------------|---------|
| New-LLM | 12-18（予想） | 短い応答が得意 |
| TinyGPT2 | 10-15（予想） | より流暢な可能性 |

**期待**: New-LLMは逐次処理なので対話との相性が良い可能性

---

## 🎯 成功基準

### Minimum Viable Product (MVP)

✅ WikiTextでPerplexity < 20に到達
✅ 対話ファインチューニングで簡単な応答生成
✅ TinyGPT2との差が±30%以内

### Stretch Goals

🎯 Perplexity < 15（WikiText）
🎯 自然な対話応答の生成
🎯 TinyGPT2と同等またはそれ以上の性能

---

## 📝 次のステップ（実験後）

実験結果に基づいて以下を検討:

1. **パラメータ拡大**: 4M → 10M → 25Mと段階的に増やす
2. **Knowledge Distillation**: GPT-2 Small (117M)から知識を蒸留
3. **アーキテクチャ改善**: Multi-scale context vectorsの導入
4. **Instruction Tuning**: Alpaca/Dollyデータでの指示応答訓練

---

## 📚 参考文献

- **WikiText**: [Merity et al., 2016] Pointer Sentinel Mixture Models
- **DailyDialog**: [Li et al., 2017] DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset
- **GPT-2**: [Radford et al., 2019] Language Models are Unsupervised Multitask Learners
- **Knowledge Distillation**: [Hinton et al., 2015] Distilling the Knowledge in a Neural Network

---

## 🤝 サポート

質問や問題がある場合:
1. GitHubのIssueを確認
2. エラーメッセージをコピーして検索
3. 新しいIssueを作成

---

**Let's make New-LLM conversational! 🚀**
