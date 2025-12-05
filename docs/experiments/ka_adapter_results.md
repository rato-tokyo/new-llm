# KA Cache Adapter Experiment Results (案1)

KAキャッシュ + Adapter方式によるKVキャッシュ代替実験の結果。

---

## 実験概要

### 目的

KVキャッシュの代わりにKA（Key + Attention Output）キャッシュを使用し、Adapterで精度を補完できるか検証。

### 方式

```
標準KVキャッシュ:
  cache: [K, V]
  inference: A = softmax(Q @ K^T) @ V

KAキャッシュ + Adapter (案1):
  cache: [K, A]  ← VではなくAttention Outputを保存
  inference: V' = Adapter(A_past)  ← Adapterで過去のAから次のVを推定
```

### Adapterアーキテクチャ

```
A (past attention output) → Linear(64→64) → GELU → Linear(64→64) → V'
                            [bottleneck]
```

---

## 実験設定

| 項目 | 値 |
|------|-----|
| Device | NVIDIA L4 (23.8GB VRAM) |
| Samples | 5,000 |
| Sequence Length | 128 |
| Base Epochs | 10 (early stopping, patience=1) |
| Adapter Epochs | 5 (early stopping, patience=1) |
| Learning Rate (base) | 1e-4 |
| Adapter Bottleneck | 64 |
| Model | KAAdapterPythiaModel |
| Total Parameters | 70,471,168 |
| Adapter Parameters | 50,688 (0.07%) |
| Train samples | 4,500 (Pile) |
| Val samples | 500 (Pile only) |

---

## 結果サマリー

| Method | PPL | Tokens/sec | KVキャッシュ比 |
|--------|-----|------------|----------------|
| **KV Cache (baseline)** | **109.0** | 1370.8 | baseline |
| KA Cache (no adapter) | 1240.3 | 1080.5 | +1037.4% |
| KA Cache (with adapter) | 439.2 | 1081.4 | +302.7% |

### Adapter効果

| 指標 | 値 |
|------|-----|
| KA (adapter未使用) | PPL 1240.3 |
| KA (adapter使用) | PPL 439.2 |
| **Adapter改善率** | **+64.6%** |

---

## 学習曲線

### Phase 1: ベースモデル学習

```
Epoch  1: train_ppl=890.5 val_ppl=368.0 *
Epoch  2: train_ppl=209.0 val_ppl=209.3 *
Epoch  3: train_ppl=106.0 val_ppl=153.3 *
Epoch  4: train_ppl=63.0  val_ppl=125.6 *
Epoch  5: train_ppl=40.1  val_ppl=112.0 *
Epoch  6: train_ppl=26.2  val_ppl=106.9 *
Epoch  7: train_ppl=17.4  val_ppl=106.8 * (best)
Epoch  8: train_ppl=11.6  val_ppl=113.3
-> Early stop
```

**Best**: Epoch 7, PPL=106.8

### Phase 3: Adapter学習（並列モード）

```
Epoch 1: batch 1/563 ppl=9.7 adapter_loss=1.0632 [0.2s]
Epoch 1: batch 100/563 ppl=11.3 adapter_loss=0.9549 [2.1s]
Epoch 1: batch 200/563 ppl=11.4 adapter_loss=0.8140 [4.0s]
Epoch 1: batch 300/563 ppl=11.3 adapter_loss=0.7014 [5.9s]
Epoch 1: batch 400/563 ppl=11.2 adapter_loss=0.6184 [7.8s]
Epoch 1: batch 500/563 ppl=11.2 adapter_loss=0.5573 [9.7s]
Epoch 1: done [10.9s] ppl=11.1 adapter=0.5274
Epoch 1: validating with KA cache...
Epoch  1: val_ppl=439.2 *

Epoch 2: done [10.9s] ppl=11.1 adapter=0.2745
Epoch  2: val_ppl=516.7
-> Early stop
```

**Best Adapter**: Epoch 1, PPL=439.2

**観察**: Adapter損失は0.27まで下がったが、val_pplは悪化。過学習の兆候。

---

## Reversal Curse評価

| 指標 | 値 |
|------|-----|
| Forward PPL | 14222.9 |
| Backward PPL | 7364.4 |
| Reversal Ratio | 1.9313 |
| Reversal Gap | -6858.5 |

**観察**:
- Forward/Backward PPLが極めて高い（14000+）
- 逆方向（Backward）の方がPPLが低い（異常）
- Reversal Ratio > 1.0は通常と逆
- **注意**: KAキャッシュ推論ではReversal Curse評価が意味をなさない可能性

---

## 考察

### 1. KAキャッシュの限界

| 方式 | PPL | 評価 |
|------|-----|------|
| KV Cache | 109.0 | baseline |
| KA (no adapter) | 1240.3 | **使用不可** (+1037%) |
| KA (with adapter) | 439.2 | **実用困難** (+303%) |

**結論**: Adapterで64.6%改善したが、依然としてKVキャッシュの4倍のPPL。

### 2. Adapter学習の問題点

1. **Epoch 1が最良**: 継続学習で悪化（過学習）
2. **Adapter損失との乖離**: adapter_loss↓ でも val_ppl↑
3. **訓練/検証の分布差**: 並列学習とautoregressive評価のギャップ

### 3. なぜKAキャッシュが困難か

```
KVキャッシュ:
  position t: A_t = softmax(Q_t @ K_cached^T) @ V_cached
  → 正確なValue vectorを使用

KAキャッシュ:
  position t: V' = Adapter(A_past)
  → Attention Outputから次のValueを予測する必要
  → 予測誤差が蓄積（error propagation）
```

**根本的問題**: Attention Output (A) と Value (V) は異なる情報を持つ。
- A: 過去の文脈を重み付け集約した結果
- V: 各トークンの意味表現

Adapterで変換するには情報損失が大きすぎる。

### 4. 推論速度

| Method | Tokens/sec | 相対速度 |
|--------|------------|----------|
| KV Cache | 1370.8 | 100% |
| KA Cache | 1080.5 | 79% |

KAキャッシュは約21%遅い。Adapter計算のオーバーヘッドが原因。

---

## 結論

### KAキャッシュ + Adapter（案1）の評価

| 観点 | 評価 | 詳細 |
|------|------|------|
| PPL精度 | **×** | KVキャッシュの4倍悪い |
| 改善効果 | △ | Adapterで64.6%改善するも不十分 |
| 推論速度 | △ | 21%遅い |
| パラメータ効率 | ○ | わずか0.07%追加 |
| 実用性 | **×** | 現状では使用不可 |

### 今後の検討事項

1. **案1の改良**
   - より大きなAdapterボトルネック
   - 複数層Adapter
   - Adapter損失関数の改善

2. **代替案の検討**
   - **案2**: KVキャッシュ量子化（4-bit等）
   - **案3**: GQA/MQAとの組み合わせ
   - **案4**: MLAの吸収モードをさらに活用

3. **評価方法の改善**
   - 並列学習とautoregressive評価のギャップ解消
   - 長文での評価（現在は128トークン）

---

## 実行コマンド

```bash
# 基本実験（自動キャッシュあり）
python3 scripts/experiment_ka_adapter.py --samples 5000 --epochs 10

# キャッシュから読み込み（2回目以降高速）
python3 scripts/experiment_ka_adapter.py --samples 5000 --epochs 10

# キャッシュなしで再学習
python3 scripts/experiment_ka_adapter.py --samples 5000 --epochs 10 --no-cache

# Adapter学習スキップ（比較用）
python3 scripts/experiment_ka_adapter.py --samples 5000 --skip-adapter-training
```

---

## 付録: 実装詳細

### Adapter学習（並列モード）

```python
# 従来: autoregressiveループ（128 forward/batch）
for t in range(seq_len):
    # 1トークンずつ生成

# 改良: 並列学習（1 forward/batch、約128倍高速）
logits, adapter_loss = model.forward_adapter_training(input_ids)
# adapter_loss = MSE(Adapter(A[:-1]), V[1:])
```

### 自動キャッシュ機能

```python
# パラメータハッシュに基づくキャッシュパス
cache_path = f"checkpoints/ka_adapter_base_{hash}.pt"

# 2回目以降は自動読み込み
if cache_path.exists():
    load_checkpoint(model, cache_path, device)
```

---

Last Updated: 2025-12-05
