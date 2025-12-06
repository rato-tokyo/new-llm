# Layer 0 Distillation + Long-Context Finetuning 実験結果 (v2)

**実験日時**: 2025-12-06
**スクリプトバージョン**: 新バージョン（Train/Val分割 + Early Stopping あり）

## 実験環境

| 項目 | 値 |
|------|-----|
| Device | NVIDIA L4 (23.8GB) |
| Base Model | EleutherAI/pythia-70m |
| ALiBi | False |

---

## Phase 1: Layer 0 Distillation

既存Pythia-70MのLayer 0の出力をInfini Layerに蒸留。

### 設定

| パラメータ | 値 |
|-----------|-----|
| Total Samples | 4,417 |
| Train Samples | 3,976 |
| Val Samples | 441 |
| Max Epochs | 50 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Early stopping patience | 2 |

### 訓練ログ（抜粋）

```
Epoch  1: train=0.032786 val=0.019092 (17.1s) *
Epoch 10: train=0.011443 val=0.011596 (17.0s) *
Epoch 20: train=0.010135 val=0.010575 (17.0s) *
Epoch 30: train=0.009512 val=0.010107 (17.0s) *
Epoch 40: train=0.009108 val=0.009840 (17.0s) *
Epoch 48: train=0.008875 val=0.009766 (17.0s)   ← Early stopping未発動
Epoch 49: train=0.008850 val=0.009725 (17.0s) *
Epoch 50: train=0.008822 val=0.009664 (17.0s) *
```

**観察**: 50 epoch全て実行。Val lossは継続的に改善（Early stopping未発動）。

### 結果

| Model | PPL |
|-------|-----|
| Original Layer 0 | 25.0 |
| Infini Layer (distilled) | 35.9 |
| **PPL difference** | **+10.9** |
| Distillation Loss (val) | 0.009664 |

**v1との比較**: PPL差が21.8 → 10.9に改善（約50%改善）

---

## Phase 2: Long-Context Finetuning

蒸留済みInfini LayerをLayer 0として使用し、長文データでEnd-to-end訓練。

### 設定

| パラメータ | 値 |
|-----------|-----|
| Total Documents | 100 |
| Train Documents | 90 |
| Val Documents | 10 |
| Tokens per document | 4,096 |
| Segment length | 256 |
| Max Epochs | 50 |
| Learning rate | 1e-5 |
| Early stopping patience | 2 |
| Training target | Infini Layer only |

### 訓練ログ

```
Long-context PPL (before): 19701.1

Epoch  1: train_ppl=1539.5 val_ppl=1111.0 (29.1s) *
Epoch  2: train_ppl=661.1 val_ppl=853.0 (29.1s) *
Epoch  3: train_ppl=463.0 val_ppl=719.6 (29.0s) *
Epoch  4: train_ppl=369.2 val_ppl=632.8 (29.0s) *
Epoch  5: train_ppl=308.6 val_ppl=599.3 (29.0s) *
Epoch  6: train_ppl=283.6 val_ppl=628.9 (28.8s)   ← 悪化
Epoch  7: train_ppl=269.1 val_ppl=587.6 (28.8s) * ← 改善
Epoch  8: train_ppl=245.8 val_ppl=592.7 (28.9s)   ← 悪化
Epoch  9: train_ppl=236.7 val_ppl=594.4 (28.8s)   ← 悪化
Early stopping at epoch 9 (patience=2)

Long-context PPL (after): 587.6
```

### 結果

| Metric | Value |
|--------|-------|
| PPL (before finetuning) | 19,701.1 |
| PPL (after finetuning) | 587.6 |
| **Improvement** | **+19,113.5** |
| Best epoch | 7 |

---

## v1 vs v2 比較

| Metric | v1 | v2 | 変化 |
|--------|-----|-----|------|
| Distillation epochs | 10 | 50 | +40 |
| Distillation loss | 0.0113 | 0.0097 | -14% |
| Distilled PPL | 46.8 | 35.9 | -23% |
| PPL gap (distillation) | +21.8 | +10.9 | -50% |
| Finetune epochs | 5 | 7 (early stop) | +2 |
| Final PPL | 568.3 | 587.6 | +3% |

**観察**:
- 蒸留は大幅に改善（PPL gap半減）
- しかし最終PPLはほぼ同じ（むしろ微増）
- Early stoppingが効いてepoch 9で終了

---

## 考察

### 蒸留フェーズの改善
- より長い訓練（50 epoch）で蒸留品質が向上
- PPL gap: 21.8 → 10.9（約50%改善）
- しかしまだ+10.9の差があり、完全な模倣は困難

### ファインチューニングの限界
- 蒸留が改善しても最終PPLはほぼ変わらず
- **ボトルネックは蒸留ではなくアーキテクチャ**
- Layer 0を完全に置き換える方式には本質的な限界がある

### 問題点
1. **RoPE → NoPE**: 位置情報の損失が回復不能
2. **Layer 0の役割**: 後続レイヤーが期待する表現を生成できない
3. **アーキテクチャミスマッチ**: Infini Layerの出力が後続レイヤーに適合しない

---

## 次のステップ: Adapter的並列挿入

**Layer 0置き換え方式の限界が明確になったため、新しいアプローチを試す。**

### 案2: Adapter的並列挿入（LoRA-like）

```
Input Embedding
      ↓
┌─────┴─────┐
│  Original │  Infini Layer
│  Layer 0  │  (Memory)
└─────┬─────┘      │
      ↓            ↓
    Output + α × Infini_Output
      ↓
Layer 1-5 (unchanged)
```

**利点**:
- 元のLayer 0を保持（RoPE維持）
- Infini Layerは補助的な役割
- αパラメータで影響度を調整可能
- 既存の短文性能を維持しやすい

---

## 所要時間

| Phase | v1 | v2 |
|-------|-----|-----|
| Distillation | ~3分 | ~14分 |
| Finetuning | ~2.5分 | ~4.5分 |
| **Total** | **~5.5分** | **~18.5分** |

---

## 使用コマンド

```bash
# Phase 1: Distillation (with early stopping)
python3 scripts/distill_layer0.py --samples 5000 --epochs 50 --patience 2

# Phase 2: Long-context Finetuning (with early stopping)
python3 scripts/finetune_long_context.py --distilled distilled_layer0.pt --epochs 50 --patience 2
```

---

## 結論

**Layer 0置き換え方式は限界がある。Adapter的並列挿入を試すべき。**
