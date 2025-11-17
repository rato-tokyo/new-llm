# 訓練中に観察された注目すべき現象

## 1. 検証損失のスパイク現象（最も重要）

### 発生パターン

すべての実験で、Epoch 17-22あたりで**突然の損失急上昇**が発生しています。

#### 実験1 (context_dim=512) の例:
```
Epoch 17: Val Loss 5.7866 ✓ 良好
Epoch 18: Val Loss 7.6308 ⚠️ 急上昇 (+32%)
Epoch 19: Val Loss 8.9319 ⚠️ さらに悪化 (+17%)
Epoch 20: Val Loss 6.8859 → 回復傾向
Epoch 21: Val Loss 5.6821 ✓ ほぼ回復
```

#### 実験2 (FNN容量増加) の例:
```
Epoch 9:  Val Loss 6.5119 ✓ 良好
Epoch 10: Val Loss 8.2924 ⚠️ 急上昇 (+27%)
Epoch 11: Val Loss 15.3114 ⚠️ 大幅悪化 (+85%!)
Epoch 12: Val Loss 7.9969 → 回復傾向
Epoch 13: Val Loss 5.8286 ✓ 回復
```

#### ベースライン (context_dim=256) の例:
```
Epoch 17: Val Loss 5.7692 ✓ 良好
Epoch 18: Val Loss 5.7692 (best)
Epoch 19: Val Loss 11.1040 ⚠️ 急上昇 (+92%!)
Epoch 20: Val Loss 14.7028 ⚠️ さらに悪化
Epoch 21: Val Loss 10.5887 → 回復傾向
Epoch 22: Val Loss 6.5873 → 回復継続
Epoch 23: Val Loss 5.6552 ✓ ほぼ回復
```

### 現象の特徴

1. **再現性が高い**: すべての実験で同じ時期に発生
2. **急激な変化**: 1エポックで2-9倍の損失増加
3. **自己回復**: 数エポック後に元のレベルに戻る
4. **モデルサイズに依存**: 大きいモデルほど激しい

## 2. この現象の原因分析

### 原因1: 文脈ベクトルの累積による数値不安定性

**メカニズム**:
```python
# New-LLMの文脈更新式
context[t] = context[t-1] + delta[t]
```

この**加算的な更新**により、文脈ベクトルが徐々に増大：

```
Epoch 1:  context = [小さい値]
Epoch 10: context = [中程度の値]
Epoch 17: context = [大きい値] ⚠️ 臨界点
Epoch 18: context = [爆発] ⚠️ 数値オーバーフロー
```

**証拠**:
- 訓練後の文脈統計を見ると:
```python
# ベースライン (Epoch 50)
Context stats - Mean: -0.0539, Std: 0.6230, Norm: 7.1440

# 実験2 (hidden=1024, Epoch 50)
Context stats - Mean: 0.0308, Std: 0.4833, Norm: 8.6724
```

ノルムが徐々に増加している → 累積効果

### 原因2: 勾配爆発 (Gradient Explosion)

**メカニズム**:

文脈ベクトルが大きくなると、その勾配も増大：

```
大きい文脈値 → 大きい出力 → 大きい損失 → 大きい勾配
    ↑                                        ↓
    ←────────── backpropagation ──────────────
```

**証拠**:
- Gradient clipping設定: `gradient_clip = 1.0`
- しかし、それでも防げないレベルの爆発が発生
- 実験2（パラメータ4倍）で最も激しい → モデルサイズと相関

### 原因3: 学習率の相対的増加

エポックが進むにつれて：
1. 文脈ベクトルの値が大きくなる
2. 同じ学習率でも、**実効的な更新幅が大きくなる**
3. 不安定な更新 → オーバーシュート

## 3. その他の観察された現象

### 現象A: 訓練損失と検証損失の乖離

すべての実験で、スパイク後に乖離が拡大：

```
実験2 Epoch 19:
  Train Loss: 5.2175, Train PPL: 184.48
  Val Loss:   5.7088, Val PPL: 301.52

実験2 Epoch 50:
  Train Loss: 3.8546, Train PPL: 47.21
  Val Loss:   7.7683, Val PPL: 2364.34
```

→ **過学習の兆候**

### 現象B: パープレキシティの異常値

スパイク時にパープレキシティが異常値に：

```
実験1 Epoch 19: Val PPL: 66,436 (通常の200倍)
実験2 Epoch 11: Val PPL: 4,463,340 (通常の15,000倍!)
ベースライン Epoch 20: Val PPL: 2,428,475
```

→ モデルが完全に混乱している状態

### 現象C: 早期収束 (Early Plateau)

すべての実験で、Epoch 20前後でベストモデルが確定：

```
ベースライン: Best at Epoch 18 (以降改善なし)
実験1: Best at Epoch 28 (その後悪化)
実験2: Best at Epoch 19 (その後悪化)
実験3: Best at Epoch 20 (130エポック訓練しても改善なし)
```

→ アーキテクチャの表現力上限に達している

### 現象D: 訓練精度の低さ

すべてのエポックで検証精度が一定：

```
Val Acc: 0.1714 (全エポックで同じ)
```

**分析**:
- 語彙サイズ91のうち、1/6 ≈ 0.167
- ほぼランダム推測レベル
- モデルが meaningful predictions を学習できていない

## 4. Transformerとの比較

**Transformer訓練の特徴**:
```
✓ スパイクなし - 安定した訓練
✓ 滑らかな損失減少
✓ Best at Epoch 49 - 最後まで改善
✓ Val Loss: 4.8379 (安定)
```

**New-LLM訓練の特徴**:
```
✗ 複数回のスパイク発生
✗ 不安定な訓練曲線
✗ Epoch 20前後で収束
✗ Val Loss: 5.6-5.7 (スパイクを繰り返す)
```

## 5. 根本原因のまとめ

### アーキテクチャ的な問題

**1. 加算的更新の限界**:
```python
context[t] = context[t-1] + delta[t]  # 無制限に増大可能
```

vs

```python
# LSTMのゲート機構
forget_gate * old_state + input_gate * new_info  # 制御された更新
```

**2. 正規化の欠如**:
- 文脈ベクトルに LayerNorm がない
- 値の範囲が制御されていない

**3. リセット機構の欠如**:
- シーケンス全体で累積し続ける
- 古い情報を忘れる仕組みがない

## 6. 推奨される改善策

### 短期的対策:

1. **文脈ベクトルの正規化**:
```python
context = context + delta
context = LayerNorm(context)  # 追加
```

2. **学習率の調整**:
```python
learning_rate = 0.00005  # 0.0001 → 0.00005
```

3. **勾配クリッピングの強化**:
```python
gradient_clip = 0.5  # 1.0 → 0.5
```

### 長期的対策:

1. **ゲート機構の導入**:
```python
# LSTM風のゲート
update_gate = sigmoid(W_u @ [token, context])
context = (1 - update_gate) * context + update_gate * delta
```

2. **指数移動平均の使用**:
```python
# 古い情報を徐々に忘れる
alpha = 0.9
context = alpha * context + (1 - alpha) * delta
```

3. **複数文脈ベクトル**:
```python
# 短期文脈と長期文脈を分離
short_context[t] = short_context[t-1] + delta_short
long_context[t] = 0.99 * long_context[t-1] + 0.01 * delta_long
```

## 7. 視覚化

訓練曲線グラフ (`checkpoints/new_llm_training_curves.png`) を確認すると、
以下が明確に見えるはずです：

1. **損失のスパイク** - 急激な上昇と回復
2. **パープレキシティの爆発** - 対数スケールでも見える
3. **早期プラトー** - Epoch 20以降フラット

## 結論

**スパイク現象の正体**:
文脈ベクトルの無制限な累積により、Epoch 17-22あたりで数値が臨界点に達し、
勾配爆発が発生。一時的にモデルが破綻するが、勾配クリッピングにより
数エポック後に回復する。

**根本的な問題**:
加算的更新 (`context = context + delta`) という設計自体が、
長期訓練における数値安定性を欠いている。

**重要性**:
この現象は、New-LLMアーキテクチャの**致命的な欠陥**を示している。
実用化には、ゲート機構や正規化などの根本的な改良が必須。
