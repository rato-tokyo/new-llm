# Selective Output LM with Learnable Gate - 実験結果（失敗）

**日付**: 2025-12-07
**結論**: 学習可能ゲート方式は複雑すぎて失敗。固定スキップパターンに簡素化。

---

## 実験概要

### 仮説

LLMの全出力がトークン化されるべきではない。
- 確信度が低いときは持ち越し（carry-over）、損失なし
- 確信度が高いときのみトークン出力、持ち越されたターゲットと比較

### 実験設定

| 項目 | 値 |
|------|-----|
| Device | NVIDIA L4 (23.8GB) |
| Samples | 5,000 |
| Sequence length | 128 |
| Epochs | 30 (early stopping) |
| Max skip | 1 |
| Threshold | 0.5 |
| Gate loss weight | 0.1 |
| Parameters | 70,748,929 |
| - output_gate | 65,793 |
| - hidden_proj | 262,656 |

---

## 訓練結果

```
Epoch  1: train=  754.2, val=  919.0, gate=0.331, carry=47.3% (61.8s) *
Epoch  2: train=  251.6, val=  467.1, gate=0.406, carry=45.1% (62.1s) *
Epoch  3: train=  149.9, val=  311.2, gate=0.441, carry=41.6% (62.4s) *
Epoch  4: train=  114.3, val=  256.2, gate=0.462, carry=38.8% (62.7s) *
Epoch  5: train=   97.8, val=  273.9, gate=0.479, carry=36.7% (63.0s)
Epoch  6: train=   84.4, val=  248.1, gate=0.493, carry=34.8% (63.2s) *
Epoch  7: train=   74.6, val=  180.8, gate=0.507, carry=33.0% (63.4s) *
Epoch  8: train=   64.2, val=  176.3, gate=0.520, carry=31.0% (63.6s) *
Epoch  9: train=   57.1, val=  172.2, gate=0.531, carry=28.8% (63.8s) *
Epoch 10: train=   51.0, val=  151.8, gate=0.542, carry=26.9% (64.1s) *
Epoch 11: train=   45.3, val=  188.3, gate=0.554, carry=24.3% (64.5s)
Epoch 12: train=   41.7, val=  165.3, gate=0.566, carry=21.8% (64.7s)
Early stopping

Best: epoch 10, ppl=151.8
```

### 訓練の観察

1. **gate_probの推移**: 0.331 → 0.566（徐々に増加）
2. **carry-over率の推移**: 47.3% → 21.8%（徐々に減少）
3. **収束**: val PPL 151.8 で early stopping

---

## 評価結果

### Gate Distribution

| 指標 | 値 |
|------|-----|
| Mean | 0.494 |
| Std | 0.105 |
| Range | [0.129, 0.979] |
| Gate-Entropy Corr | -0.861 |

**解釈**: gate_probとエントロピーに強い負の相関（-0.861）があり、低エントロピー（高確信度）時に出力する傾向を学習している。

### Selective Generation (threshold=0.5)

| 指標 | 値 |
|------|-----|
| Skip Ratio | 28.89% |
| Total Steps | 45 |

### Position-wise PPL (standard mode)

| 位置 | PPL |
|------|-----|
| 0-16 | 1169.7 |
| 16-32 | 1225.9 |
| 32-64 | 1224.7 |
| 64-96 | 1247.9 |
| 96-128 | 1178.4 |

**問題**: 通常モードでのPPLが非常に高い（1000+）。

### Reversal Curse

| 指標 | 値 |
|------|-----|
| Forward PPL | 586.0 |
| Backward PPL | 6150.5 |
| Gap | +5564.5 |

---

## 失敗の分析

### 問題点

1. **訓練-評価の不一致**
   - 訓練時: carry-over方式（出力位置のみ損失計算）
   - 評価時: 通常モード（全位置で損失計算）
   - 結果: 訓練val PPL=151.8 vs 評価PPL=1200+

2. **ゲート学習の不安定性**
   - gate_probがthreshold 0.5付近に収束
   - carry-over率が訓練中に大きく変動（47% → 22%）

3. **勾配の不連続性**
   - threshold判定（gate_prob > 0.5）で勾配が途切れる
   - ゲートが適切に最適化されない

4. **ターゲットアライメントの複雑さ**
   - 持ち越し時のターゲット計算が非自明
   - 正しく実装しても、学習が困難

### 試した方式と結果

| 方式 | 結果 |
|------|------|
| OutputGate + threshold | carry-over 96.3%、ほぼ学習されない |
| max_skip強制 + threshold | 収束が不安定、PPL改善せず |
| エントロピーベースgate_loss | ゲートが適切に学習されない |

---

## 教訓

1. **動的判断より固定パターン**
   - 学習可能ゲートは複雑すぎる
   - 固定スキップパターン（例: 2回に1回持ち越し）がシンプル

2. **訓練-評価一貫性**
   - 訓練時と評価時の条件を揃えることが必須
   - 離散的なthresholdは学習バイアスを生む

3. **追加パラメータのコスト**
   - OutputGate（65K params）に対して効果が見合わない
   - シンプルなアプローチが優先

---

## 結論

学習可能ゲート方式は削除し、固定スキップパターンに簡素化。

**変更内容**:
- `OutputGate`クラスを削除
- `max_skip` → `skip_interval`に変更
- `compute_carryover_loss` → `compute_fixed_skip_loss`に置き換え
- エントロピーベースの損失計算を削除

**新しい使い方**:
```python
# 固定スキップパターン（2回に1回出力）
model = create_model("selective", skip_interval=2)

# 訓練
logits, _ = model(input_ids)
loss, stats = model.compute_fixed_skip_loss(logits, labels)
```
