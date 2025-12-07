# Selective Output LM 実験 v1

**日付**: 2025-12-07

## 概要

LLMが即座に出力せず、隠れ状態を追加処理してから出力する方式の検証。

## 実験設定

| 項目 | 値 |
|------|-----|
| Samples | 2,500 |
| Sequence length | 128 |
| Epochs | 30 (early stopping) |
| Position Encoding | NoPE |
| Reversal pairs | 10 samples (33文から連結) |

## モデル

- **Baseline (extra_passes=0)**: 即座に出力（追加処理なし）
- **Selective (extra_passes=1)**: 1回追加処理してから出力

## 結果

### Validation PPL

| Model | Extra Passes | Best PPL | Epoch |
|-------|--------------|----------|-------|
| baseline | 0 | **484.6** | 7 |
| selective | 1 | 516.8 | 7 |

### Reversal Curse

| Model | Forward PPL | Backward PPL | Gap |
|-------|-------------|--------------|-----|
| baseline | 12868.4 | 11069.4 | -1799.1 |
| selective | **9576.1** | 10690.2 | **+1114.1** |

### Position-wise PPL

| Position | Baseline | Selective |
|----------|----------|-----------|
| 0-16 | 644.1 | 669.8 |
| 16-32 | 515.2 | 561.6 |
| 32-64 | 405.1 | 429.8 |
| 64-96 | 483.0 | 525.0 |
| 96-128 | 493.6 | 519.5 |

## 考察

1. **Val PPL**: Baselineが優位（484.6 vs 516.8）
   - 追加処理がPile一般データには不利

2. **Reversal Curse**: Selectiveが優位
   - Gap: -1799 → +1114（正常方向に改善）
   - Forward PPL: 12868 → 9576（大幅改善）
   - 追加処理で知識の定着が良くなる可能性

3. **注意点**:
   - Forward/Backward PPLが両方とも非常に高い（9000-13000）
   - Reversal pairsが10サンプルしか追加されていない
   - サンプル数を増やす必要あり

## 問題点

- 現在のSelectiveは持ち越し後の隠れ状態のみで推論
- 持ち越し前の隠れ状態を使っていない
- 両方使うパターンを検証する必要あり

## 次のステップ

1. **両方の隠れ状態を使用**: 持ち越し前 + 持ち越し後で推論
2. **Reversal pairsを増やす**: 33文すべてを十分な回数使用
