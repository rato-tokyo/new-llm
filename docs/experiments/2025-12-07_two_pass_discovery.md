# 2-Pass Processing による Reversal Curse 改善の発見

**日付**: 2025-12-07

## 概要

SelectiveOutputLMの実装中に、意図せず「2回処理」方式がReversal Curseを改善することを発見した。

## 実装内容

```
baseline (1回処理):
  token_A → embed → layers → h1 → "B"予測

selective (2回処理):
  token_A → embed → layers → h1 → proj → layers → h2 → "B"予測
```

**注意**: これは当初「Continuous LM」と誤って呼んでいたが、正しいContinuousとは異なる。

## 実験結果

| Model | Val PPL | Forward PPL | Backward PPL | Gap |
|-------|---------|-------------|--------------|-----|
| baseline | **484.6** | 12868.4 | 11069.4 | -1799.1 |
| selective | 516.8 | **9576.1** | 10690.2 | **+1114.1** |

## 観察

1. **Val PPL**: Baselineが優位（484.6 vs 516.8）
   - 一般的なPileデータでは2回処理が不利

2. **Reversal Curse**: Selectiveが大幅改善
   - Gap: -1799 → +1114（正常方向に改善）
   - Forward PPL: 12868 → 9576（大幅改善）
   - 訓練データの記憶が強化されている可能性

## 考察

2回処理がReversal Curseを改善する理由の仮説：

1. **情報の精緻化**: 1回目の処理で得た表現を、2回目でさらに洗練
2. **非線形性の増加**: 2回Transformerを通ることで、より複雑なパターンを学習
3. **記憶の定着**: 繰り返し処理により、訓練データの記憶が強化

## 今後の検討事項

- なぜ2回処理で記憶が定着するのか？
- Val PPLが悪化するのは過学習か、それとも別の理由か？
- 3回以上の処理では効果が変わるか？

## 実験設定

| 項目 | 値 |
|------|-----|
| Samples | 2,500 |
| Sequence length | 128 |
| Epochs | 30 (early stopping) |
| Position Encoding | NoPE |
| Reversal pairs | 100 samples |

## 備考

この発見は意図せず得られたもので、正しいContinuous LM（離散化スキップ）とは異なる。
コードはメンテナンス性のため削除し、発見の記録のみ残す。
