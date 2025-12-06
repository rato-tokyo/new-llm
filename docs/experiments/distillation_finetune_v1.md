# Layer 0 Distillation + Long-Context Finetuning 実験結果 (v1)

**実験日時**: 2025-12-06
**スクリプトバージョン**: 古いバージョン（Train/Val分割・Early Stopping なし）

## 実験環境

| 項目 | 値 |
|------|-----|
| Device | NVIDIA L4 (23.8GB) |
| Base Model | EleutherAI/pythia-70m |
| ALiBi | False |

---

## Phase 1: Layer 0 Distillation

既存Pythia-70MのLayer 0の出力をInfini Layerに蒸留。

### 設定

| パラメータ | 値 |
|-----------|-----|
| Samples | 4,417 (5,000リクエスト) |
| Epochs | 10 |
| Batch size | 8 |
| Learning rate | 1e-4 |

### 訓練ログ

```
Epoch  1: loss=0.031436 (18.7s) *
Epoch  2: loss=0.016727 (18.0s) *
Epoch  3: loss=0.014747 (18.0s) *
Epoch  4: loss=0.013683 (18.0s) *
Epoch  5: loss=0.012996 (18.1s) *
Epoch  6: loss=0.012489 (18.2s) *
Epoch  7: loss=0.012089 (18.2s) *
Epoch  8: loss=0.011757 (18.1s) *
Epoch  9: loss=0.011480 (18.1s) *
Epoch 10: loss=0.011252 (18.1s) *
```

### 結果

| Model | PPL |
|-------|-----|
| Original Layer 0 | 25.0 |
| Infini Layer (distilled) | 46.8 |
| **PPL difference** | **+21.8** |
| Distillation Loss | 0.011252 |

**評価**: PPL差が大きい。蒸留のみでは不十分。

---

## Phase 2: Long-Context Finetuning

蒸留済みInfini LayerをLayer 0として使用し、長文データでEnd-to-end訓練。

### 設定

| パラメータ | 値 |
|-----------|-----|
| Documents | 100 (Train: 90, Val: 10) |
| Tokens per document | 4,096 |
| Segment length | 256 |
| Epochs | 5 |
| Learning rate | 1e-5 |
| Training target | Infini Layer only |

### 訓練ログ

```
Long-context PPL (before): 18641.0

Epoch  1: ppl=1610.4 (27.7s) *
Epoch  2: ppl=627.9 (27.6s) *
Epoch  3: ppl=461.2 (27.5s) *
Epoch  4: ppl=378.6 (27.6s) *
Epoch  5: ppl=332.7 (27.6s) *

Long-context PPL (after): 568.3
```

### 結果

| Metric | Value |
|--------|-------|
| PPL (before finetuning) | 18,641.0 |
| PPL (after finetuning) | 568.3 |
| **Improvement** | **+18,072.7** |

---

## 考察

### 蒸留フェーズ
- 蒸留損失は順調に減少（0.031 → 0.011）
- しかしPPL差は+21.8と大きい
- Infini Layer（NoPE）がRoPEベースのLayer 0を完全に模倣するのは困難

### ファインチューニングフェーズ
- 劇的な改善: 18,641 → 568.3
- しかし568.3はまだ高い（実用レベルは100未満が目安）
- Epoch数を増やすことでさらなる改善の余地あり

### 課題
1. **蒸留の限界**: RoPE → NoPE変換で情報損失
2. **PPLまだ高い**: 568.3は実用には不十分
3. **Early Stopping未使用**: 過学習の可能性を検証できていない

### 次のステップ
1. Train/Val分割 + Early Stopping を導入した新スクリプトで再実験
2. Epoch数を増やす（50 epoch + patience=3）
3. より多くのドキュメントで訓練
4. ALiBi版も試す

---

## 所要時間

| Phase | Time |
|-------|------|
| Distillation (10 epochs) | ~3分 |
| Finetuning (5 epochs) | ~2.5分 |
| **Total** | **~5.5分** |

---

## 使用コマンド

```bash
# Phase 1: Distillation
python3 scripts/distill_layer0.py --samples 5000 --epochs 10

# Phase 2: Long-context Finetuning
python3 scripts/finetune_long_context.py --distilled distilled_layer0.pt --epochs 5
```
