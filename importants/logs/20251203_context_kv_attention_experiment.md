# Context-KV Attention 実験結果

**実験日時**: 2025-12-03
**環境**: Google Colab (NVIDIA L4, 22.2GB)

---

## 実験設定

| パラメータ | 値 |
|-----------|-----|
| Samples | 100 |
| Context dim | 256 |
| Context interval | 20 |
| Num heads | 8 |
| Max contexts | 32 |
| Total parameters | 45,163,264 |
| Trainable (Phase 2) | 6,301,440 (14%) |

### Context Window 計算

```
interval=20, max_contexts=32
→ 32 × 20 = 640 トークン分の履歴を参照

theoretical_max = 122,794 / 20 + 1 = 6,140
→ max_contexts=32 で制限（OOM防止）
```

---

## 実験結果

### Phase 1: OACD多様性学習

| 指標 | 値 |
|------|-----|
| 処理時間 | 7.0s |
| 収束率 | 92% |
| Early stop | Iter 18 (conv >= 90%) |
| Effective Rank | 82.2% |

**収束過程**:
```
Iter 2: conv=0%  loss=5.37
Iter 7: conv=1%  loss=1.59
Iter 12: conv=22% loss=1.70
Iter 15: conv=55% loss=1.34
Iter 18: conv=92% loss=1.26 → Early stop
```

### Phase 2: Context-KV Attention学習

| Epoch | Train PPL | Val PPL | Val Acc | 備考 |
|-------|-----------|---------|---------|------|
| 1 | 3085.4 | 572.0 | 11.6% | * |
| 2 | 413.9 | 436.8 | 13.7% | * |
| 3 | 276.8 | 380.3 | 15.1% | * |
| 4 | 199.0 | 374.2 | 15.7% | * |
| **5** | **148.7** | **361.0** | **16.2%** | **Best** |
| 6 | 111.5 | 387.9 | 15.5% | |
| 7 | 81.9 | 424.1 | 16.1% | |
| 8 | 59.8 | 484.4 | 16.0% | Early stop |

### 最終結果

| 指標 | 値 |
|------|-----|
| **Val PPL** | **361.0** |
| **Val Acc** | **16.2%** |
| Best Epoch | 5 |
| Phase 2 Time | 89.2s |
| Total Time | 97.9s |

---

## 重要な観察

### 1. 過学習の兆候

```
Epoch 5: train_ppl=148.7, val_ppl=361.0  ← Best
Epoch 8: train_ppl=59.8,  val_ppl=484.4  ← 過学習

Train PPL: 148.7 → 59.8 (大幅改善)
Val PPL:   361.0 → 484.4 (悪化)
```

**Train/Val Gap**: 約2.4倍（361/148.7）→ 過学習傾向あり

### 2. 旧アーキテクチャとの比較

| 指標 | Context-KV | 2-Block Cascade |
|------|------------|-----------------|
| Val PPL (100 samples) | 361.0 | ~400-500 (推定) |
| Val Acc | 16.2% | ~15% (推定) |
| KVキャッシュサイズ | 32 contexts | 全トークン |

**Context-KV方式の利点**:
- KVキャッシュを大幅削減（32 vs 6140）
- メモリ効率が高い
- 長いシーケンスでもスケール可能

### 3. メモリ使用量

| 段階 | CPU | GPU |
|------|-----|-----|
| データロード後 | 0.8GB | 0.0GB |
| モデル初期化後 | 1.6GB | 0.2GB |
| Phase 1後 | 2.6GB | 0.2GB |
| Phase 2準備後 | 2.8GB | 0.2GB |

**キャッシュサイズ**:
- train_ctx: 120MB
- train_emb: 360MB
- val_ctx: 22MB
- val_emb: 67MB
- **Total: 568MB**

---

## Context-KV Attentionの興味深い特性

### 1. 位置に依存しないContext Window

通常のLLM:
```
Position i: 全ての過去トークン [0, 1, 2, ..., i-1] を参照
→ KVキャッシュが線形に増加
```

Context-KV方式:
```
Position i: [ctx[i], ctx[i-20], ctx[i-40], ...] (最大32個)
→ KVキャッシュは定数（max_contexts固定）
```

### 2. 階層的情報圧縮

```
Token Embedding (768-dim) → Context (256-dim) → KV (768-dim)
```

- 768次元のトークン情報を256次元に圧縮
- その後、Attentionで768次元に展開
- 情報のボトルネックが正則化効果を持つ可能性

### 3. スケーラビリティ

| シーケンス長 | 通常LLM KVサイズ | Context-KV KVサイズ |
|--------------|------------------|---------------------|
| 1,000 | 1,000 | 32 |
| 10,000 | 10,000 | 32 |
| 100,000 | 100,000 | 32 |

**100kトークンでも32コンテキストのみ** → 推論時のメモリ効率が極めて高い

---

## 今後の実験提案

1. **サンプル数増加**: 100 → 500, 1000で過学習が改善されるか
2. **interval調整**: 20 → 50, 100でPPLがどう変化するか
3. **max_contexts調整**: 32 → 16, 64での影響
4. **Regularization追加**: Dropout増加、Weight Decay調整

---

## 結論

Context-KV Attention方式は：
- **動作確認**: 正常に学習・推論が実行可能
- **メモリ効率**: 従来比で大幅なKVキャッシュ削減を実現
- **課題**: 100サンプルでは過学習傾向があり、データ増加が必要

---

*Last Updated: 2025-12-03*
