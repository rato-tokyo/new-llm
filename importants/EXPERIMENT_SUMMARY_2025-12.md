# 実験結果サマリー (2025年12月)

## 概要

2025年12月1日〜2日に実施した実験の統合サマリー。

**共通環境**:
- GPU: NVIDIA L4 (22.2GB)
- Samples: 2000
- Context dim: 500
- 訓練データ: 2,403,563 tokens
- 検証データ: 22,723 tokens

---

## 1. レイヤー構成比較

### 結果

| Config | Layers | FFN | Params | Conv% | Val PPL | Val Acc | 収束速度 |
|--------|--------|-----|--------|-------|---------|---------|---------|
| L1_F1 | 1 | 1 | 40.2M | 92% | **127.2** | 24.7% | 遅い (30 iter) |
| **L2_F1** | 2 | 1 | 41.8M | 93% | 128.1 | **24.9%** | **速い (14 iter)** |
| L1_F2 | 1 | 2 | 43.4M | 91% | 156.9 | 23.0% | 最遅 (31 iter) |

### 結論

- **L2_F1（2層）推奨**: 最高Acc、最速収束、PPLもL1_F1とほぼ同等
- **L1_F2（FFN深化）非推奨**: ER低下（69%）、過学習傾向、PPL最悪

---

## 2. Early Stopping戦略

### 収束率90%閾値の効果

| 実験 | Early Stop条件 | Val PPL | Val Acc |
|------|---------------|---------|---------|
| 旧方式 | Val ER + 10iter | 189.1 | 22.2% |
| **新方式** | **conv 90%** | **127.2** | **24.7%** |
| 改善率 | - | **-32.8%** | **+11.3%** |

### 結論

**収束率90%でのEarly Stop**により大幅な性能改善。Phase 1を十分に収束させることが重要。

---

## 3. Context Mode比較（最重要）

### 4方式の比較

| Mode | 説明 | Val PPL | Val Acc | 備考 |
|------|------|---------|---------|------|
| **E案** | Layer i → Context Layer i | **128.1** | **24.9%** | 理論上最良 |
| **G案** | Layer1→prev, Layer2→current | 132.2 | 24.4% | **採用決定** |
| A案 | 全Layer → 最終context | 136.9 | 24.6% | 削除予定 |
| F案 | Layer1のみcontext注入 | 137.9 | 24.4% | 削除予定 |

### Context入力パターン

| Layer | E案 | G案 | A案 | F案 |
|-------|-----|-----|-----|-----|
| Layer 1 | ctx_layer1 | ctx_prev | ctx_final | ctx_final |
| Layer 2 | ctx_layer2 | ctx_current | ctx_final | none |

### G案採用の理由

E案が理論上最良だが、以下の理由でG案を採用：

1. **メモリ効率**
   - E案: `[num_layers, num_tokens, context_dim]` → レイヤー数倍
   - G案: `[num_tokens, context_dim]` → 固定

2. **拡張性**
   - G案は3層以上に自然拡張可能
   - Layer 1 ← context[i-2]、Layer 2 ← context[i-1]、Layer 3 ← context[i]

3. **精度差は許容範囲**
   - PPL: +4.1 (+3.2%)
   - Acc: -0.5%

4. **メンテナンス性**
   - 複数Mode維持のコストが高い

---

## 4. 推奨設定

```python
# config.py
num_layers = 2                      # 2層推奨（G案に必要）
early_stopping_threshold = 0.90     # 90%収束で停止
context_mode = "prev_and_current"   # G案
```

---

## 5. 主要な知見

### Phase 1
- **2層構成は収束が2倍以上速い**（14 iter vs 30 iter）
- **FFN深化は逆効果**（ER低下、過学習）
- **収束率90%での停止が最適**

### Phase 2
- **階層的context（E案）が理論上最良**だがメモリ効率に課題
- **時間的context（G案）**が実用的な代替案
- **同一context繰り返し（A案）**や**context削減（F案）**は効果薄

### トレードオフ
- PPL vs メモリ効率: G案を選択（精度を若干犠牲にメモリ効率を優先）
- 収束速度 vs 精度: 2層構成が両立

---

*Last Updated: 2025-12-02*
*詳細データは old/1202/ フォルダ参照*
