# Position Encoding Experiment Results

位置エンコーディングの比較実験結果。

## 実験設定

| 項目 | 値 |
|------|-----|
| Device | NVIDIA L4 (23.8GB VRAM) |
| Samples | 10,000 |
| Sequence Length | 128 |
| Epochs | 30 (early stopping) |
| Learning Rate | 1e-4 |
| Model | UnifiedPythiaModel (70.4M params) |

---

## NoPE (No Position Encoding) 実験 - 2025-12-05

### 結果サマリー

| Model | Position Encoding | PPL | Best Epoch |
|-------|-------------------|-----|------------|
| NoPE-Pythia | None | 504.6 | 3 |
| Pythia (参考) | RoPE (25%) | ~424 | - |
| MLA-Pythia (参考) | ALiBi | ~455 | - |

### PPL劣化

- **NoPE vs RoPE**: +80.6 PPL (+19.0%)
- **NoPE vs ALiBi**: +49.6 PPL (+10.9%)

位置情報がないことで約19%のPPL劣化が確認された。

### 学習曲線

```
Epoch  1: train_ppl=639.7 val_ppl=694.9 *
Epoch  2: train_ppl=177.2 val_ppl=546.0 *
Epoch  3: train_ppl=96.6  val_ppl=504.6 *
Epoch  4: train_ppl=60.2  val_ppl=506.5
-> Early stop (epoch 3)
```

### Position-wise PPL

| Position Range | NoPE PPL |
|----------------|----------|
| 0-16 | 609.9 |
| 16-32 | 519.8 |
| 32-64 | 488.7 |
| 64-96 | 493.0 |
| 96-128 | 485.3 |

**観察**:
- 予想通り、位置による差が小さい（均一化傾向）
- ただし完全に均一ではない（causal maskによる暗黙的な順序情報）
- 先頭位置(0-16)のPPLが高いのは、コンテキストが少ないため

### Reversal Curse

| Metric | Value |
|--------|-------|
| Forward PPL | 6226.1 |
| Backward PPL | 4677.2 |
| Reversal Ratio | 1.331 |
| Reversal Gap | -1548.9 |

**観察**:
- Reversal Ratioが1.0より大きい（Backward < Forward）
- 位置情報がないため、方向の概念が薄い
- ただしcausal maskにより完全に対称ではない

---

## 考察

### 位置エンコーディングの重要性

1. **PPL劣化は約19%**: 位置情報は言語モデルにとって重要だが、致命的ではない
2. **Causal maskの暗黙的順序**: 位置エンコーディングなしでも、causal maskにより「過去→現在」の順序は保持される
3. **長距離依存の課題**: Position-wise PPLを見ると、NoPEは長い文脈でも改善が限定的

### RoPE vs ALiBi vs NoPE

| 位置エンコーディング | PPL | 特徴 |
|---------------------|-----|------|
| RoPE (25%) | ~424 | 最良。相対位置を回転行列で表現 |
| ALiBi | ~455 | +7%。距離に線形ペナルティ、MLA互換 |
| NoPE | ~505 | +19%。位置情報なし |

---

## 今後の実験候補

1. **シーケンス長の影響**: seq_len=256, 512での比較
2. **ALiBi slope調整**: 0.0625以外のslopeでの実験
3. **RoPE rotary_pct調整**: 25%以外の割合での実験
4. **組み合わせ**: RoPE + ALiBi等のハイブリッド

---

## 実行コマンド

```bash
# 実行したコマンド
python3 scripts/experiment_nope.py --samples 10000 --skip-baseline

# 全比較（推奨）
python3 scripts/experiment_position.py --samples 10000 --epochs 30
```

---

Last Updated: 2025-12-05
