# Context Mode比較実験結果 (2025-12-01)

## 実験概要

**目的**: A案（final context only）とE案（layerwise）の性能比較

**環境**:
- GPU: NVIDIA L4 (22.2GB)
- Samples: 2000
- Context dim: 500
- Layers: 2
- 検証データ: indices 50000-50019（22,723 tokens）
- 訓練データ: indices 0-1999（2,403,563 tokens）

**Context Mode**:
- **E案 (layerwise)**: TokenBlock Layer i は ContextBlock Layer i の出力を参照
- **A案 (final_only)**: 全TokenBlockレイヤーが ContextBlock の最終出力のみを参照

---

## 結果サマリー

| Config | Mode | Layers | Params | Phase1 Iter | Conv% | ER% | Val PPL | Val Acc | Total Time |
|--------|------|--------|--------|-------------|-------|-----|---------|---------|------------|
| **L2_E** | E案 | 2 | 41.8M | 14 | 93% | 79.7% | **128.1** | **24.9%** | **865.4s** |
| L2_A | A案 | 2 | 41.8M | 14 | 93% | 79.7% | 136.9 | 24.6% | 982.1s |

**Winner**: E案 (layerwise)

---

## 詳細比較

### Phase 1（共通）

両方式ともPhase 1は同一の処理（ContextBlockのみ学習）のため、結果は同じ：

| 指標 | 値 |
|------|-----|
| 収束イテレーション | 14 |
| 収束率 | 93% |
| Train ER | 81.5% |
| Val ER | 79.7% |
| Phase 1 時間 | ~177s |

```
Iter 10: conv=26%
Iter 11: conv=58%
Iter 12: conv=80%
Iter 13: conv=89%
Iter 14: conv=93% → Early stop
```

### Phase 2（差異あり）

| 指標 | E案 (layerwise) | A案 (final_only) | 差分 |
|------|-----------------|------------------|------|
| Best Epoch | 11 | 13 | +2 |
| Val PPL | **128.1** | 136.9 | +8.8 (+6.9%) |
| Val Acc | **24.9%** | 24.6% | -0.3% |
| Train PPL | 81.7 | 87.7 | +6.0 |
| Phase 2 時間 | ~688s | ~805s | +117s |

### 学習曲線比較

**E案 (layerwise)**:
```
Epoch 1:  train_ppl=210.9 val_ppl=176.4 acc=20.9%
Epoch 5:  train_ppl=130.9 val_ppl=144.7 acc=23.7%
Epoch 10: train_ppl=91.9  val_ppl=130.9 acc=24.6%
Epoch 11: train_ppl=88.5  val_ppl=128.1 acc=24.9% ★ Best
```

**A案 (final_only)**:
```
Epoch 1:  train_ppl=438.9 val_ppl=220.2 acc=20.3%
Epoch 5:  train_ppl=132.4 val_ppl=147.8 acc=23.6%
Epoch 10: train_ppl=97.8  val_ppl=137.8 acc=24.3%
Epoch 13: train_ppl=87.7  val_ppl=136.9 acc=24.6% ★ Best
```

---

## 分析

### 1. E案の優位性

**PPL**: E案が8.8ポイント良い（128.1 vs 136.9）
- 約6.9%の改善
- 中間レイヤー情報の活用が予測精度向上に寄与

**Acc**: E案が0.3%高い（24.9% vs 24.6%）
- 僅差だが一貫してE案が優位

**学習速度**: E案が2エポック早く収束
- 11エポック vs 13エポック
- 中間context情報により学習が効率化

### 2. A案の特徴

**初期PPLが高い**: Epoch 1で438.9（E案: 210.9）
- 最終contextのみでは初期の学習が困難
- 中間レイヤー情報がない分、TokenBlockの負担が大きい

**収束は遅いが安定**:
- Early Stopまで13エポック必要
- PPL改善は継続的

### 3. なぜE案が優れるか

```
E案: TokenBlock Layer i ← ContextBlock Layer i
     各TokenBlockレイヤーが対応する深さの文脈情報を受け取る
     → 階層的な文脈表現を活用

A案: TokenBlock Layer i ← ContextBlock 最終出力
     全TokenBlockレイヤーが同じ（最も抽象化された）文脈情報を受け取る
     → 中間レイヤーの文脈情報が失われる
```

**中間レイヤーの文脈情報の価値**:
- 浅いレイヤー: より局所的・具体的な文脈
- 深いレイヤー: より大域的・抽象的な文脈
- E案はこの階層構造を保持、A案は最終出力のみで情報が圧縮される

---

## 結論

### E案（layerwise）を推奨

| 観点 | E案 | A案 |
|------|-----|-----|
| PPL | ✅ 128.1 | 136.9 |
| Acc | ✅ 24.9% | 24.6% |
| 学習速度 | ✅ 11エポック | 13エポック |
| 総時間 | ✅ 865s | 982s |

**E案の利点**:
1. より良いPPL（-6.9%）
2. より高いAcc（+0.3%）
3. より速い収束（-2エポック）
4. 中間レイヤー情報の有効活用

**A案を選ぶ理由**:
- 特になし（E案が全指標で優位）
- メモリ効率はA案が若干良い（キャッシュが1/num_layers）が、性能差を正当化できない

### 推奨設定

```python
# デフォルト（E案）を使用
model = LLM(
    ...,
    use_final_context_only=False,  # E案（デフォルト）
)
```

---

## 生データ

### L2_E (E案 - 既存実験から)
```
PPL=128.1, Acc=24.9%, ER=79.7%, 14 iter, 865.4s
Phase 1: 14 iter, conv=93%
Phase 2: 11 epochs (early stop at 12)
```

### L2_A (A案 - 本実験)
```
PPL=136.9, Acc=24.6%, ER=79.7%, 14 iter, 982.1s
Phase 1: 14 iter, conv=93%
Phase 2: 13 epochs (early stop at 14)

Epoch-by-epoch:
  Epoch 1:  train_ppl=438.9 val_ppl=220.2 acc=20.3%
  Epoch 2:  train_ppl=210.1 val_ppl=178.2 acc=22.0%
  Epoch 3:  train_ppl=168.8 val_ppl=162.0 acc=22.7%
  Epoch 4:  train_ppl=146.9 val_ppl=153.3 acc=23.1%
  Epoch 5:  train_ppl=132.4 val_ppl=147.8 acc=23.6%
  Epoch 6:  train_ppl=122.0 val_ppl=143.9 acc=23.9%
  Epoch 7:  train_ppl=114.0 val_ppl=141.3 acc=24.0%
  Epoch 8:  train_ppl=107.5 val_ppl=139.7 acc=24.1%
  Epoch 9:  train_ppl=102.2 val_ppl=138.6 acc=24.2%
  Epoch 10: train_ppl=97.8  val_ppl=137.8 acc=24.3%
  Epoch 11: train_ppl=94.0  val_ppl=137.3 acc=24.4%
  Epoch 12: train_ppl=90.6  val_ppl=137.0 acc=24.6%
  Epoch 13: train_ppl=87.7  val_ppl=136.9 acc=24.6% ★ Best
  Epoch 14: → Early stop
```

---

*Last Updated: 2025-12-01*
*Experiment: context_mode_experiment.py with --layers 2*
