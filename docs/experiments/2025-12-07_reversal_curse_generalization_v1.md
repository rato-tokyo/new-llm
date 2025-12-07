# Reversal Curse 汎化性能実験 v1

**日付**: 2025-12-07

## 仮説

Reversal Curseの真の問題は「逆方向を推論できない」ことではなく、**汎化性能の低さ**である。

```
問題の本質:
  "Tom is Alice's parent" と "Alice is Tom's children" を学習しても、
  別のペア "Bob is Jack's parent" から "Who is Bob's children?" に答えられない。

  → 逆方向の「パターン」を学習できていない
  → 個別の事実を丸暗記しているだけ
```

## 実験設計

### データ構成

- **パターン学習ペア (180組)**: 順方向・逆方向の両方を学習
- **Valペア (20組)**: 順方向のみ学習 → **逆方向で評価（汎化テスト）**

### 訓練方式

| 条件 | Baseline | Modified |
|------|----------|----------|
| パターン学習 | 全文学習 | コンテキスト分離学習 |
| Valペア | 順方向のみ（全文） | 順方向のみ（全文） |
| Pile | 3500サンプル | 3500サンプル |
| **Total** | **3880** | **3880** |

**Baseline（全文学習）**:
```
学習対象: "Tom is Alice's parent. Who is Alice's parent? Tom"
→ 全文をloss計算対象
```

**Modified（コンテキスト分離）**:
```
初期context: "Tom is Alice's parent."  ← loss計算から除外
学習対象: "Who is Alice's parent? Tom"
→ 推論パターンのみを学習
```

### 評価

- **Reversal PPL**: `"Who is [parent]'s child?"` → 正解は子の名前
  - Valペアでは逆方向を**一度も学習していない**
  - 汎化能力の直接テスト
- **Pile PPL**: 一般的な言語能力の維持

## 実験結果

### 設定

```
Total pairs: 200
  - Pattern pairs: 180
  - Val pairs: 20
Pile samples: 3500 (fixed)
Sequence length: 128
Epochs: 30
Position Encoding: RoPE
Parameters: 70,420,480
```

### 訓練経過

**Baseline**:
```
Epoch  1: train=548.5, pile_val=585.8 *
Epoch  2: train=127.5, pile_val=457.8 *
Epoch  3: train= 66.6, pile_val=450.3 *
Epoch  4: train= 40.9, pile_val=466.5
Early stopping at epoch 4
Best: epoch 3, ppl=450.3
```

**Modified**:
```
Epoch  1: train=549.8, pile_val=553.4 *
Epoch  2: train=130.2, pile_val=462.9 *
Epoch  3: train= 67.8, pile_val=447.3 *
Epoch  4: train= 41.6, pile_val=436.7 *
Epoch  5: train= 27.3, pile_val=532.9
Early stopping at epoch 5
Best: epoch 4, ppl=436.7
```

### 最終結果

| Model | Reversal PPL | Pile PPL | Best Epoch |
|-------|--------------|----------|------------|
| Baseline | 64.0 | 450.3 | 3 |
| **Modified** | **44.4** | **436.7** | 4 |
| **改善率** | **-30.6%** | **-3.0%** | - |

## 分析

### 1. Reversal Curse（汎化テスト）の改善

Modified（44.4）はBaseline（64.0）より**約31%低い**Reversal PPLを達成。

これが意味すること：
- Valペアでは逆方向を**一度も学習していない**
- にもかかわらず、Modifiedは「Who is Bob's child?」→「Jack」をより正確に推論
- **パターン学習ペアで学んだ逆方向推論がValペアに汎化した**

### 2. 言語能力の維持

Pile PPLはModified（436.7）がBaseline（450.3）より若干良い。
コンテキスト分離訓練によって一般的な言語能力は損なわれていない。

### 3. 訓練挙動の違い

- Baseline: 3エポックで収束
- Modified: 4エポックで収束（より長く学習が進んだ）

コンテキスト分離により、より多くのエポックで学習が続いた可能性がある。

## 結論

**仮説は支持された。**

コンテキスト分離訓練（Modified）により：
1. 知識の丸暗記を抑制
2. 推論パターンの抽象化を促進
3. 未学習の逆方向クエリへの汎化能力が向上

## 今後の課題

1. **より多くのパターン/Valペアでの検証**
   - 現在: 180パターン、20 Val
   - 提案: 400パターン、100 Val

2. **統計的有意性の確認**
   - 複数シードでの実行
   - 信頼区間の計算

3. **他のドメインへの適用**
   - 親子関係以外の関係性（例：首都-国、著者-作品）
