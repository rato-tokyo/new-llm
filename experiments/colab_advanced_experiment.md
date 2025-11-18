# Google Colab Advanced Experiment - 実験結果

**実験日**: 2025-01-18
**モデル**: New-LLM (Advanced)
**データセット**: WikiText-2
**デバイス**: Google Colab (Tesla T4, 15GB GPU)

---

## 📋 実験設定

### モデル構成（Advanced）

| パラメータ | Baseline | Advanced | 変化率 |
|-----------|---------|----------|-------|
| 総パラメータ数 | 2.74M | **4.84M** | **1.77倍** |
| Context Vector Dim | 256 | **512** | **2倍** |
| Num Layers | 6 | **12** | **2倍** |
| Embed Dim | 256 | 256 | - |
| Hidden Dim | 512 | 512 | - |

### 訓練設定

| パラメータ | 値 |
|-----------|-----|
| Batch Size | **512** (GPU最適化) |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Gradient Clip | Adaptive (0.5 → 1.0 → 2.0) |
| Max Epochs | 50 |
| Device | **CUDA (Tesla T4)** |
| 訓練速度 | **0.6-0.7分/epoch** |

---

## 📊 訓練結果

### 最終性能 (Epoch 50)

| 指標 | Train | Validation |
|------|-------|-----------|
| **Loss** | 3.596 | **3.596** ✓ |
| **Perplexity** | 36.5 | **36.45** ✓ |
| **Accuracy** | - | **31.4%** |

### 改善推移

| 指標 | 初期 (Epoch 1) | 最良 (Epoch 50) | 改善率 |
|------|---------------|----------------|-------|
| Val Loss | 4.27 | **3.60** | **-15.7%** |
| Val PPL | 71.3 | **36.45** | **-48.9%** |

### エポック別推移（抜粋）

| Epoch | Train Loss | Val Loss | Train PPL | Val PPL | 備考 |
|-------|-----------|----------|-----------|---------|------|
| 1 | 5.506 | 4.27 | - | 71.3 | 初期 |
| 5 | 4.217 | 4.32 | - | 75.0 | checkpoint |
| 10 | 4.203 | 4.25 | - | 70.0 | checkpoint |
| 15 | 4.198 | 4.21 | - | 67.3 | checkpoint |
| **17** | **4.123** | **4.10** | - | **60.2** | **改善加速開始** |
| 20 | 4.003 | 3.99 | - | 54.3 | checkpoint |
| 25 | 3.768 | 3.76 | - | 43.1 | checkpoint |
| 30 | 3.710 | 3.71 | - | 41.0 | checkpoint |
| 35 | 3.670 | 3.67 | - | 39.3 | checkpoint |
| 40 | 3.643 | 3.65 | - | 38.4 | checkpoint |
| 45 | 3.616 | 3.63 | - | 37.8 | checkpoint |
| **50** | **3.596** | **3.60** | **36.5** | **36.45** | **最終** ✓ |

---

## 📈 学習曲線の特徴

### 良好な学習の証拠

✅ **過学習なし**: Train/Val Lossが並行して低下
✅ **継続的改善**: Epoch 50まで Val Loss が改善継続
✅ **安定した学習**: 大きなスパイクなし
✅ **健全なギャップ**: Train/Val の差が小さい

### 観察されたパターン

**3つのフェーズ**:

1. **Phase 1 (Epoch 1-16)**: 緩やかな改善期
   - Val PPL: 71.3 → 66.7
   - 改善率: 6.5%

2. **Phase 2 (Epoch 17-30)**: **急速改善期** ⚡
   - Val PPL: 60.2 → 41.0
   - 改善率: **31.9%** (最大の改善)
   - **Epoch 17で何かブレークスルー発生**

3. **Phase 3 (Epoch 31-50)**: 収束期
   - Val PPL: 40.6 → 36.45
   - 改善率: 10.2%
   - 緩やかな改善が継続

### 重要な観察: Epoch 17のブレークスルー

**Epoch 16 → 17**:
- Train Loss: 4.195 → **4.123** (-1.7%)
- Val Loss: 4.20 → **4.10** (-2.4%)
- Val PPL: 66.7 → **60.2** (-9.7%)

**仮説**: モデルが言語パターンの"臨界質量"を学習し、理解が急速に深まった可能性

---

## ⚠️ 重大な問題点

### Baselineとの性能比較

| モデル | パラメータ数 | Val PPL | 訓練時間 | デバイス |
|-------|------------|---------|---------|---------|
| **Baseline** | 2.74M | **23.34** ✓ | ~27時間 | CPU |
| **Advanced** | 4.84M | **36.45** ✗ | ~35分 | GPU |

**驚くべき結果**: **小さいモデルの方が性能が良い** ❗

### 性能差の分析

- Baseline (2.74M): PPL **23.34**
- Advanced (4.84M): PPL **36.45**
- **差**: +56.2%（Advancedの方が悪い）

### 考えられる原因

#### 1. **訓練不足**（最有力）

**仮説**: 大きいモデルは収束に時間がかかる

証拠:
- Epoch 50でも Val Loss が改善継続中
- Early stoppingが発動していない
- Phase 3でも緩やかに改善している

**推奨**: **100エポックまで訓練延長**

#### 2. **Learning Rateの不一致**

**現状**: 両モデルとも LR = 0.0001

**問題**: より大きいモデルには**異なる学習率**が必要な可能性

**推奨対策**:
- LR = 0.0001 → 0.0005（5倍）で再実験
- または Learning Rate Scheduler導入

#### 3. **Batch Sizeの影響**

| モデル | Batch Size | 効果的batch更新数/epoch |
|-------|-----------|---------------------|
| Baseline | 32 | 1,595回 |
| Advanced | 512 | **100回** |

**問題**: Batch size 512では1エポックあたりの更新回数が**16分の1**

**推奨**: Batch size = 256 or 128 に削減

#### 4. **モデル容量の問題**

**仮説**: WikiText-2 (vocab=1000) には4.84Mは大きすぎる？

**反論**: GPT-2 (117M) も同様のデータで訓練されている

**結論**: 容量よりも**訓練方法**の問題の可能性が高い

---

## 🎯 改善提案

### 優先度A: 訓練延長

```python
# 現在: num_epochs = 50
# 推奨: num_epochs = 100

config.num_epochs = 100  # 2倍に延長
```

**理由**:
- Epoch 50でも改善継続中
- より大きいモデルには時間が必要
- GPU速度（0.7分/epoch）なら100エポックでも70分

### 優先度B: Batch Sizeの調整

```python
# 現在: batch_size = 512
# 推奨: batch_size = 256 or 128

config.batch_size = 256  # 更新頻度を2倍に
```

**期待効果**:
- 更新頻度の増加
- より細かい勾配更新
- 収束の加速

### 優先度C: Learning Rateの最適化

```python
# オプション1: 高いLR
config.learning_rate = 0.0005  # 5倍

# オプション2: LR Scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

---

## 💾 保存されたチェックポイント

| ファイル名 | Epoch | Val PPL | サイズ |
|-----------|-------|---------|--------|
| `best_new_llm_wikitext_ctx512_layers12.pt` | 50 | **36.45** | 推定50MB |
| `new_llm_wikitext_ctx512_layers12_epoch_5.pt` | 5 | 75.0 | - |
| `new_llm_wikitext_ctx512_layers12_epoch_10.pt` | 10 | 70.0 | - |
| ... | ... | ... | ... |
| `new_llm_wikitext_ctx512_layers12_epoch_50.pt` | 50 | 36.45 | - |

---

## 🔬 次の実験計画

### 実験1: 訓練延長（推奨）

```python
# Epoch 50のcheckpointから再開
trainer.train(
    num_epochs=100,
    resume_from="new_llm_wikitext_ctx512_layers12_epoch_50.pt"
)
```

**期待結果**: Val PPL < 25（Baselineを超える）

### 実験2: ハイパーパラメータ調整

**設定変更**:
- batch_size: 512 → 256
- learning_rate: 0.0001 → 0.0003
- num_epochs: 100

**期待結果**: より早い収束

### 実験3: Context Vector 1024実験

**設定**:
- context_vector_dim: 1024（4倍）
- num_layers: 12（維持）
- パラメータ数: 約7-8M

**条件**: 実験1で良い結果が出た場合のみ

---

## 📌 結論

### 成功点

✅ **安定した訓練**: 過学習なく50エポック完走
✅ **GPU高速化**: 0.7分/epoch（CPU比100倍速）
✅ **継続的改善**: 最後まで学習継続
✅ **再現性**: チェックポイント保存で再開可能

### 課題

❌ **Baselineより低性能**: PPL 36.45 vs 23.34
❌ **訓練不足の可能性**: 50エポックでは不十分
❌ **Batch size大きすぎ**: 更新頻度が少ない

### 次のステップ

**最優先**: Epoch 50から100まで訓練延長

**理由**:
1. 最も低コスト（追加35分のみ）
2. 改善の可能性が高い
3. 他の調整の必要性を判断できる

**実行コマンド**:
```python
# Google Colabで実行
trainer.train(
    num_epochs=100,
    resume_from="checkpoints/new_llm_wikitext_ctx512_layers12_epoch_50.pt"
)
```

**期待**: Val PPL 25以下を達成し、Baselineを超える性能

---

**総評**: 訓練は健全だが、**大きいモデルには十分な訓練時間が必要**。100エポックまで延長を強く推奨。
