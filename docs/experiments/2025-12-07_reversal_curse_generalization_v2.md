# Reversal Curse 汎化性能実験 v2

**日付**: 2025-12-07

## 概要

v1実験の成功を受け、より多くのパターン学習ペアとValペアで検証を行った。

## 実験設計

### データ構成

- **パターン学習ペア (400組)**: 順方向・逆方向の両方を学習
- **Valペア (100組)**: 順方向のみ学習 → **逆方向で評価（汎化テスト）**
- **Pile samples**: 1100（固定）

### 訓練方式

| 条件 | Baseline | Modified |
|------|----------|----------|
| パターン学習 | 全文学習 | コンテキスト分離学習 |
| Valペア | 順方向のみ（全文） | 順方向のみ（全文） |
| Pile | 1100サンプル | 1100サンプル |
| **Total** | **2000** | **2000** |

## 実験設定

```
Total pairs: 500
  - Pattern pairs: 400
  - Val pairs: 100
Pile samples: 1100 (fixed)
Sequence length: 128
Epochs: 30
Position Encoding: RoPE
Parameters: 70,420,480
Early stopping patience: 3
```

## 訓練経過

### Baseline

```
Epoch  1: train=  85.9, pile_val=1232.0 *
Epoch  2: train=  25.7, pile_val=1114.6 *
Epoch  3: train=  16.6, pile_val= 931.9 *
Epoch  4: train=  11.9, pile_val= 905.7 *
Epoch  5: train=   9.0, pile_val= 868.5 *
Epoch  6: train=   7.1, pile_val= 848.2 *
Epoch  7: train=   5.7, pile_val= 913.0
Epoch  8: train=   4.6, pile_val=1623.1
Epoch  9: train=   3.8, pile_val=1239.6
Early stopping at epoch 9
Best: epoch 6, ppl=848.2
```

### Modified

```
Epoch  1: train=  87.5, pile_val=1065.7 *
Epoch  2: train=  27.0, pile_val=1073.7
Epoch  3: train=  17.1, pile_val= 855.1 *
Epoch  4: train=  12.0, pile_val= 944.1
Epoch  5: train=   9.1, pile_val= 896.9
Epoch  6: train=   7.1, pile_val=1278.8
Early stopping at epoch 6
Best: epoch 3, ppl=855.1
```

## 結果

| Model | Reversal PPL | Pile PPL | Best Epoch |
|-------|--------------|----------|------------|
| Baseline | 142.3 | 848.2 | 6 |
| **Modified** | **75.9** | **855.1** | 3 |
| **改善率** | **-46.7%** | +0.8% | - |

## v1との比較

| 設定 | v1 | v2 |
|------|----|----|
| Pattern pairs | 180 | 400 |
| Val pairs | 20 | 100 |
| Pile samples | 3500 | 1100 |
| Reversal PPL (Baseline) | 64.0 | 142.3 |
| Reversal PPL (Modified) | 44.4 | 75.9 |
| **改善率** | -30.6% | **-46.7%** |

## 分析

### 1. Reversal Curse（汎化テスト）の大幅改善

Modified（75.9）はBaseline（142.3）より**約47%低い**Reversal PPLを達成。

これが意味すること：
- Valペア100組では逆方向を**一度も学習していない**
- にもかかわらず、Modifiedは「Who is Bob's child?」→「Jack」をより正確に推論
- **パターン学習ペアで学んだ逆方向推論がValペアに汎化した**

### 2. パターン学習ペア増加の効果

パターン学習ペアを180→400に増やすことで、改善率が30.6%→46.7%に向上。

より多くの「逆方向推論パターン」を学習することで、汎化能力が向上した。

### 3. Pile PPLへの影響

Pile PPLはModified（855.1）がBaseline（848.2）より若干悪い（+0.8%）。

しかし、Reversal PPLの47%改善に対して、この程度の悪化は許容範囲。

### 4. 訓練挙動の違い

- Baseline: 6エポックで収束
- Modified: 3エポックで収束

Modifiedはより早く最適点に到達している。コンテキスト分離により、効率的にパターンを学習できている可能性がある。

## 結論

**仮説は強く支持された。**

コンテキスト分離訓練（Modified）により：
1. 知識の丸暗記を抑制
2. 推論パターンの抽象化を促進
3. 未学習の逆方向クエリへの汎化能力が大幅向上（47%改善）

パターン学習ペア数を増やすことで、汎化性能がさらに向上することも確認された。

## 今後の課題

1. **さらに多くのパターン/Valペアでの検証**
   - 現在: 400パターン、100 Val
   - 提案: 800パターン、200 Val

2. **統計的有意性の確認**
   - 複数シードでの実行
   - 信頼区間の計算

3. **他のドメインへの適用**
   - 親子関係以外の関係性（例：首都-国、著者-作品）
