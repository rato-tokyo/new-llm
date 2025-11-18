# Advanced Experiments Guide - 拡張実験ガイド

## 📋 概要

`scripts/train_wikitext_advanced.py`を使った拡張実験のガイド。

**実験できること**:
1. コンテキストベクトル次元の拡張（256 → 512, 1024など）
2. レイヤー数の増減（6 → 12, 24など）
3. int8量子化（メモリ削減）

---

## 🚀 クイックスタート

### 1. 基本実験（デフォルト設定）

```bash
python scripts/train_wikitext_advanced.py
```

**デフォルト設定**:
- Context Vector Dim: 512（2倍）
- Layers: 12（2倍）
- Batch Size: 512（GPU RAM最大活用）
- Quantization: none

---

## ⚙️ 設定変更方法

### 実験設定の変更

`scripts/train_wikitext_advanced.py`の**27-37行目**を編集：

```python
# ========================================
# 実験パラメータ（ここを変更するだけ！）
# ========================================

# コンテキストベクトル次元（256, 512, 1024, 2048など）
context_vector_dim = 512  # ← ここを変更

# レイヤー数（6, 12, 24, 48など）
num_layers = 12  # ← ここを変更

# 量子化モード: 'none', 'int8'
quantization_mode = 'none'  # ← ここを変更
```

---

## 🧪 推奨実験パターン

### 実験1: コンテキストベクトル次元の影響

**目的**: 文脈表現力の向上がPerplexityに与える影響を測定

| 実験 | context_vector_dim | 期待効果 |
|------|-------------------|---------|
| ベースライン | 256 | 既存結果 |
| 2倍 | 512 | PPL 5-10%改善 |
| 4倍 | 1024 | PPL 10-15%改善 |

**手順**:
1. `context_vector_dim = 512`に設定
2. `python scripts/train_wikitext_advanced.py`
3. 結果を記録
4. 次は`context_vector_dim = 1024`で繰り返し

---

### 実験2: レイヤー数の影響

**目的**: モデルの深さが学習能力に与える影響を測定

| 実験 | num_layers | パラメータ数 | 期待効果 |
|------|-----------|-------------|---------|
| ベースライン | 6 | 2.74M | 既存結果 |
| 2倍 | 12 | 5-6M | PPL 10-15%改善 |
| 4倍 | 24 | 10-12M | PPL 20-25%改善 |

**手順**:
1. `num_layers = 12`に設定
2. `python scripts/train_wikitext_advanced.py`
3. 結果を記録

---

### 実験3: 量子化の効果

**目的**: メモリ削減と精度のトレードオフを測定

| 実験 | quantization_mode | メモリ削減 | 精度低下予想 |
|------|------------------|----------|------------|
| FP32 | 'none' | - | ベースライン |
| INT8 | 'int8' | 約4倍 | 1-3% PPL上昇 |

**手順**:
1. `quantization_mode = 'int8'`に設定
2. `python scripts/train_wikitext_advanced.py`
3. メモリ使用量と精度を比較

---

## 📊 実験名の自動生成

実験結果は自動的に以下の命名規則で保存されます：

```
new_llm_wikitext_ctx{次元数}_layers{レイヤー数}_{量子化モード}
```

**例**:
- `new_llm_wikitext_ctx512_layers12`: コンテキスト512、12層
- `new_llm_wikitext_ctx1024_layers24_int8`: コンテキスト1024、24層、int8量子化

**保存先**:
- モデル: `checkpoints/best_new_llm_wikitext_ctx512_layers12.pt`
- グラフ: `checkpoints/new_llm_wikitext_ctx512_layers12_training_curves.png`
- 進捗: `checkpoints/new_llm_wikitext_ctx512_layers12_progress.json`

---

## 🎯 推奨実験順序

### Phase 1: コンテキストベクトル次元の最適化

1. **実験1a**: `context_vector_dim = 512, num_layers = 6`
2. **実験1b**: `context_vector_dim = 1024, num_layers = 6`
3. **比較**: どの次元が最もコスパが良いか

### Phase 2: レイヤー数の最適化

1. **実験2a**: `context_vector_dim = 512, num_layers = 12`
2. **実験2b**: `context_vector_dim = 512, num_layers = 24`
3. **比較**: レイヤー数とPerplexity改善の関係

### Phase 3: 量子化の検証

1. **実験3a**: `context_vector_dim = 512, num_layers = 12, quantization_mode = 'int8'`
2. **比較**: メモリ削減と精度低下のバランス

---

## 💡 量子化について

### int8量子化の推奨理由

1. **PyTorchネイティブサポート**: 安定した実装
2. **メモリ削減**: 約4倍（fp32 → int8）
3. **精度低下**: 通常1-3%程度（許容範囲）
4. **推論高速化**: CPUでも高速

### 量子化の仕組み

```
fp32 (4バイト/パラメータ) → int8 (1バイト/パラメータ)
2.74M params × 4 bytes = 11 MB → 2.74M params × 1 byte = 2.75 MB
```

### 量子化の有効化

```python
quantization_mode = 'int8'
```

---

## 📈 期待される結果

### コンテキストベクトル次元の影響

| 次元 | パラメータ数 | 期待PPL | メモリ |
|------|------------|---------|--------|
| 256 | 2.74M | 24-25 | 11 MB |
| 512 | 3.5M | 22-23 | 14 MB |
| 1024 | 5M | 20-21 | 20 MB |

### レイヤー数の影響

| レイヤー数 | パラメータ数 | 期待PPL | 訓練時間 |
|-----------|------------|---------|---------|
| 6 | 2.74M | 24-25 | 7時間 |
| 12 | 5-6M | 22-23 | 10時間 |
| 24 | 10-12M | 20-21 | 15時間 |

---

## 🔧 トラブルシューティング

### GPU RAM最適化

**デフォルトbatch_size=512は15GB GPU（T4）向け**

| GPU | GPU RAM | 推奨batch_size |
|-----|---------|---------------|
| T4 (Colab無料) | 15GB | 512 |
| V100 | 16GB | 512 |
| A100 | 40GB | 1024-2048 |
| RTX 3060 | 12GB | 256-384 |

**調整方法**:
- メモリ不足 → batch_sizeを半分に（512 → 256 → 128）
- 余裕あり → batch_sizeを2倍に（512 → 1024）

### メモリ不足エラー

**エラー**: `RuntimeError: CUDA out of memory` または `Killed`

**対処法**:
1. `batch_size`を減らす（512 → 256 → 128）
2. `num_layers`を減らす（12 → 6）
3. `context_vector_dim`を減らす（512 → 256）

### 訓練が遅い

**対処法**:
1. `num_epochs`を減らす（50 → 30）
2. `batch_size`を増やす（32 → 64）
3. Google Colabを使用（GPU利用）

---

## 📝 実験結果の記録

### 推奨フォーマット

```markdown
## 実験1: コンテキストベクトル512

**設定**:
- context_vector_dim: 512
- num_layers: 6
- quantization_mode: none

**結果**:
- Best Val PPL: 22.5
- 訓練時間: 8時間
- パラメータ数: 3.5M

**考察**:
- ベースライン（PPL 24）より7%改善
- コストパフォーマンス良好
```

---

## 🚀 次のステップ

実験完了後:
1. 最良の設定をメインスクリプトに統合
2. DailyDialogでファインチューニング
3. TinyGPT2と性能比較
4. さらなる最適化（Learning Rate調整など）

---

**Happy Experimenting! 🧪**
