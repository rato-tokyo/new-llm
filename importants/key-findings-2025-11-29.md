# 重要な発見まとめ (2025-11-29)

## 1. token継ぎ足し（token_input_all_layers）が本質的

### 比較データ

| 設定 | token_input_all_layers | ER | Val PPL | Val Acc | データ |
|------|------------------------|-----|---------|---------|--------|
| 旧構造 | **True**（全レイヤー入力） | 76.3% | **334** | **18.9%** | 500サンプル |
| 等差減少 | False（最初のみ入力） | 8.6% | 536 | 15.4% | 500サンプル |

### 結論

- **token継ぎ足し（token_input_all_layers=True）がパフォーマンスに本質的**
- Effective Rank（ER）はtoken継ぎ足しの副産物であり、ER自体が直接パフォーマンスを決めるわけではない
- 同じ500サンプルで、token継ぎ足しありの方がPPL 38%改善、Acc 23%向上

---

## 2. 並列処理とシーケンシャル処理のcontext差は無視できる

### 実験結果

50サンプル（56,602トークン）での比較：

| 指標 | 値 |
|------|-----|
| コサイン類似度（平均） | **0.9969** |
| コサイン類似度（最小） | 0.9890 |
| MSE | 0.006156 |
| 相対誤差（平均） | 7.5% |

### 位置による差

| 位置 | MSE | CosSim |
|------|-----|--------|
| 最初100トークン | 0.0069 | 0.9966 |
| 中間100トークン | 0.0062 | 0.9969 |
| 最後100トークン | 0.0057 | 0.9972 |

### 結論

- **並列処理で得られるcontextは、シーケンシャル処理とほぼ同一（類似度99.7%）**
- 位置に関わらず一貫して高い類似度
- Phase 1の並列処理は正当化される（品質を犠牲にしていない）

---

## 3. キャッシュ収集の並列化が可能

### 背景

Phase 1で確定したcontextを使えば、全レイヤー出力の計算は並列化可能。

### 実装

```python
# シーケンシャル（旧）: 51秒
for i, token_embed in enumerate(token_embeds):
    context_outputs = model.forward_context_with_intermediates(context, token_embed)
    context = context_outputs[-1]

# 並列（新）: 数秒
shifted_contexts = [initial_context] + previous_contexts[:-1]
all_layer_outputs = model.forward_with_intermediates_batch(shifted_contexts, token_embeds)
```

### ポイント

- Phase 2では token i の処理に `previous_contexts[i-1]` を使用（1つずらし）
- token 0 は初期context（ゼロベクトル）を使用

---

## 4. スケーリング則: トークン数が支配的要因

### 実験結果

| Samples | Tokens | Val PPL | Val Acc | Val ER |
|---------|--------|---------|---------|--------|
| 50 | 56,602 | 1503.3 | 9.9% | 8.5% |
| 100 | 110,516 | 1036.3 | 13.7% | 7.8% |
| 200 | 216,119 | 757.7 | 17.6% | 8.4% |
| 500 | 529,173 | 536.0 | 15.4% | 8.6% |

### スケーリング係数

- **α = -0.459** (R² = 0.993)
- 2倍のデータでPPL約27%減少

### 結論

- **ERは7.8%〜9.0%でほぼ一定**（トークン数に依存しない）
- **トークン数がPPLの支配的要因**（強い相関 R²=0.993）
- ERの変動はパフォーマンスと相関しない

---

## 5. 500サンプル以上での早期停止問題

### 観察

- 200→500サンプルでAccuracyが17.6%→15.4%に低下（PPLは改善継続）
- 500サンプルではEpoch 3で早期停止（200サンプルはEpoch 5まで継続）

### 原因推定

- 訓練データが多いとモデルが急速にフィットし、早期停止が早すぎる
- train_pplの急激な低下（Epoch 1: 918 → Epoch 2: 306）

### 対策案

- `phase2_patience` を増やす
- 固定エポック数で訓練

---

## 今後の方針

1. **token_input_all_layers=True を標準設定として採用**
2. **並列キャッシュ収集を有効化**（51秒→数秒）
3. **大規模実験では早期停止パラメータを調整**

---

## 関連ファイル

- [scaling-experiment-analysis-2025-11-29.md](scaling-experiment-analysis-2025-11-29.md) - スケーリング実験の詳細分析
- [embedding-freeze-experiment-2025-11-27.md](embedding-freeze-experiment-2025-11-27.md) - Embedding凍結実験（token_input_all_layers=Trueでの結果）

---

Last Updated: 2025-11-29
