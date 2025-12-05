# Position Encoding Experiment Results

位置エンコーディングの比較実験結果。

---

## 統一比較実験 - 2025-12-05

### 実験設定

| 項目 | 値 |
|------|-----|
| Device | NVIDIA L4 (23.8GB VRAM) |
| Samples | 5,000 |
| Sequence Length | 128 |
| Epochs | 30 (early stopping) |
| Learning Rate | 1e-4 |
| Model | UnifiedPythiaModel (70.4M params) |
| Reversal Pairs | 順方向文を訓練データに含む |

### 結果サマリー

| Position Encoding | PPL | Best Epoch | RoPE比 |
|-------------------|-----|------------|--------|
| **RoPE (25%)** | **510.3** | 4 | baseline |
| ALiBi | 517.8 | 4 | +1.5% |
| NoPE (None) | 559.9 | 4 | +9.7% |

### 学習曲線

**RoPE (25%)**
```
Epoch  1: train_ppl=854.2 val_ppl=764.9 *
Epoch  2: train_ppl=194.9 val_ppl=571.2 *
Epoch  3: train_ppl=98.0  val_ppl=528.3 *
Epoch  4: train_ppl=58.1  val_ppl=510.3 * (best)
Epoch  5: train_ppl=36.9  val_ppl=534.1
-> Early stop
```

**ALiBi**
```
Epoch  1: train_ppl=829.4 val_ppl=766.6 *
Epoch  2: train_ppl=189.9 val_ppl=577.9 *
Epoch  3: train_ppl=96.9  val_ppl=530.7 *
Epoch  4: train_ppl=57.6  val_ppl=517.8 * (best)
Epoch  5: train_ppl=36.4  val_ppl=538.9
-> Early stop
```

**NoPE (None)**
```
Epoch  1: train_ppl=874.7 val_ppl=786.8 *
Epoch  2: train_ppl=209.9 val_ppl=605.7 *
Epoch  3: train_ppl=111.2 val_ppl=568.6 *
Epoch  4: train_ppl=68.8  val_ppl=559.9 * (best)
Epoch  5: train_ppl=45.5  val_ppl=579.8
-> Early stop
```

### Position-wise PPL

| Position | RoPE (25%) | ALiBi | NoPE (None) |
|----------|------------|-------|-------------|
| 0-16 | 668.5 | 690.4 | 689.4 |
| 16-32 | 517.5 | 529.0 | 558.0 |
| 32-64 | 534.8 | 543.8 | 571.1 |
| 64-96 | 510.7 | 511.6 | 563.8 |
| 96-128 | 506.6 | 501.8 | 565.8 |

**観察**:
- RoPEとALiBiは位置が進むほどPPLが改善（長距離依存を活用）
- NoPEは位置による差が小さい（均一化傾向）
- 96-128位置ではALiBi (501.8) がRoPE (506.6) を上回る

### Reversal Curse評価

| Model | Forward PPL | Backward PPL | Ratio | Gap |
|-------|-------------|--------------|-------|-----|
| RoPE (25%) | 10646.2 | 6757.9 | 1.575 | -3888.3 |
| ALiBi | 9549.0 | 5183.9 | 1.842 | -4365.1 |
| NoPE (None) | 6783.7 | 5787.0 | **1.172** | -996.7 |

**観察**:
- Reversal Ratio が 1.0 に近いほど「逆転の呪い」が少ない
- **NoPE (1.172) が最もReversal Curseが少ない**
  - 位置情報がないため方向の区別がつきにくい
- RoPE/ALiBiは順方向を強く学習、逆方向は苦手

---

## 考察

### 位置エンコーディングの比較

| 位置エンコーディング | PPL | Reversal Ratio | 特徴 |
|---------------------|-----|----------------|------|
| RoPE (25%) | 510.3 | 1.575 | 最良PPL、相対位置を回転行列で表現 |
| ALiBi | 517.8 | 1.842 | +1.5%、距離に線形ペナルティ、MLA互換 |
| NoPE | 559.9 | 1.172 | +9.7%、Reversal Curse最小 |

### 発見

1. **RoPE vs ALiBi**: PPL差はわずか1.5%、ほぼ同等
2. **NoPEの劣化は約10%**: 位置情報は重要だが致命的ではない
3. **NoPEのReversal Curse耐性**: 位置情報がないため方向に依存しにくい
4. **長距離位置でのALiBi優位**: 96-128位置でALiBiがRoPEを上回る

### トレードオフ

```
位置情報あり（RoPE/ALiBi）:
  ✅ PPL良好
  ❌ Reversal Curse強い（方向依存）

位置情報なし（NoPE）:
  ❌ PPL劣化（+10%）
  ✅ Reversal Curse少ない
```

---

## 実行コマンド

```bash
# 全比較実験
python3 scripts/experiment_position.py --samples 5000 --epochs 30

# 特定の位置エンコーディングのみ
python3 scripts/experiment_position.py --pos-types rope alibi
python3 scripts/experiment_position.py --pos-types none
```

---

Last Updated: 2025-12-05
