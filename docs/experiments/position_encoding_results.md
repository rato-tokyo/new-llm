# Position Encoding Experiment Results

位置エンコーディングの比較実験結果。

---

## 統一比較実験（修正版）- 2025-12-05

**重要**: 以前の実験ではデータリークバグがあり、PPLが人工的に低く出ていた。本実験は修正後の結果。

### 実験設定

| 項目 | 値 |
|------|-----|
| Device | NVIDIA L4 (23.8GB VRAM) |
| Samples | 5,000 |
| Sequence Length | 128 |
| Epochs | 30 (early stopping, patience=3) |
| Learning Rate | 1e-4 |
| Model | UnifiedPythiaModel (70.4M params) |
| Train samples | 4,830 (Pile: 4,500 + Reversal: 330) |
| Val samples | 500 (Pile only, データリークなし) |

### 結果サマリー

| Position Encoding | PPL | Best Epoch | RoPE比 |
|-------------------|-----|------------|--------|
| **RoPE (25%)** | **107.4** | 6 | baseline |
| ALiBi | 114.9 | 6 | +7.0% |
| NoPE (None) | 131.5 | 6 | +22.4% |

### 学習曲線

**RoPE (25%)**
```
Epoch  1: train_ppl=566.2 val_ppl=363.9 *
Epoch  2: train_ppl=142.6 val_ppl=206.7 *
Epoch  3: train_ppl=74.9  val_ppl=153.0 *
Epoch  4: train_ppl=45.9  val_ppl=125.5 *
Epoch  5: train_ppl=30.0  val_ppl=112.9 *
Epoch  6: train_ppl=20.1  val_ppl=107.4 * (best)
Epoch  7: train_ppl=13.6  val_ppl=110.4
-> Early stop
```

**ALiBi**
```
Epoch  1: train_ppl=550.9 val_ppl=357.7 *
Epoch  2: train_ppl=139.7 val_ppl=206.8 *
Epoch  3: train_ppl=74.4  val_ppl=155.8 *
Epoch  4: train_ppl=45.7  val_ppl=130.2 *
Epoch  5: train_ppl=29.7  val_ppl=116.3 *
Epoch  6: train_ppl=19.7  val_ppl=114.9 * (best)
Epoch  7: train_ppl=13.0  val_ppl=116.7
-> Early stop
```

**NoPE (None)**
```
Epoch  1: train_ppl=571.8 val_ppl=380.3 *
Epoch  2: train_ppl=152.0 val_ppl=231.3 *
Epoch  3: train_ppl=83.9  val_ppl=178.0 *
Epoch  4: train_ppl=53.5  val_ppl=151.7 *
Epoch  5: train_ppl=36.3  val_ppl=136.1 *
Epoch  6: train_ppl=25.1  val_ppl=131.5 * (best)
Epoch  7: train_ppl=17.6  val_ppl=133.0
-> Early stop
```

### Position-wise PPL

| Position | RoPE (25%) | ALiBi | NoPE (None) |
|----------|------------|-------|-------------|
| 0-16 | 153.0 | 159.0 | 167.0 |
| 16-32 | 107.1 | 108.7 | 123.6 |
| 32-64 | 107.0 | 115.0 | 126.1 |
| 64-96 | 103.4 | 110.9 | 129.0 |
| 96-128 | 104.9 | 110.7 | 133.7 |

**観察**:
- RoPEとALiBiは位置が進むほどPPLが改善（長距離依存を活用）
- NoPEは位置による差が小さい（均一化傾向）
- RoPEが全位置で最良

### Reversal Curse評価

| Model | Forward PPL | Backward PPL | Ratio | Gap |
|-------|-------------|--------------|-------|-----|
| RoPE (25%) | 1.7 | 760.1 | 0.002 | +758.5 |
| ALiBi | 1.7 | 683.5 | 0.002 | +681.9 |
| NoPE (None) | 1.7 | 644.7 | 0.003 | +643.0 |

**観察**:
- **Forward PPLが1.7と非常に低い**: 順方向文を訓練データに10回繰り返して含めているため、完全に暗記
- **Backward PPLが高い**: 逆方向は訓練データに含まれていないため、正しくReversal Curseを測定
- **NoPE (644.7) が最もBackward PPLが低い**: 位置情報がないため方向の区別がつきにくい
- Ratioは低すぎて比較困難（Forward PPLが低すぎるため）

---

## 考察

### 位置エンコーディングの比較（修正後）

| 位置エンコーディング | PPL | RoPE比 | 特徴 |
|---------------------|-----|--------|------|
| RoPE (25%) | 107.4 | baseline | 最良PPL、相対位置を回転行列で表現 |
| ALiBi | 114.9 | +7.0% | 距離に線形ペナルティ、MLA互換 |
| NoPE | 131.5 | +22.4% | 位置情報なし、Reversal Curse耐性 |

### 発見

1. **RoPE vs ALiBi**: PPL差は約7%、RoPEが優位
2. **NoPEの劣化は約22%**: 位置情報は重要
3. **NoPEのReversal Curse耐性**: Backward PPLが最も低い（644.7 vs 760.1）
4. **学習効率**: 全モデルがepoch 6で収束

### 以前の実験との比較

| 実験 | RoPE PPL | ALiBi PPL | NoPE PPL | 備考 |
|------|----------|-----------|----------|------|
| 以前（バグあり） | 510.3 | 517.8 | 559.9 | データリーク |
| **今回（修正後）** | **107.4** | **114.9** | **131.5** | 正常 |

**差異の原因**: 以前はreversal pairsが検証データにも混入し、PPLが人工的に悪く見えていた。修正後は正常なPPL（107〜131）に改善。

### トレードオフ

```
位置情報あり（RoPE/ALiBi）:
  ✅ PPL良好
  ❌ Reversal Curse強い（方向依存）

位置情報なし（NoPE）:
  ❌ PPL劣化（+22%）
  ✅ Reversal Curse少ない（Backward PPL最小）
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

Last Updated: 2025-12-05 (データリークバグ修正後)
