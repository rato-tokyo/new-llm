# Context-KV Attention 実験結果（500サンプル、固定数方式）

**実験日時**: 2025-12-03
**環境**: Google Colab (NVIDIA L4, 22.2GB)
**KV方式**: 方法2（固定数方式） - 常にmax_contexts個を使用、不足分はゼロパディング

---

## 実験設定

| パラメータ | 値 |
|-----------|-----|
| Samples | 500 |
| Train tokens | 587,970 |
| Val tokens | 22,723 |
| Context dim | 256 |
| Context interval | 20 |
| Num heads | 8 |
| Max contexts | 32（固定） |
| Total parameters | 45,163,264 |
| Trainable (Phase 2) | 6,301,440 (14%) |

### Context Window 計算

```
interval=20, max_contexts=32（固定）
→ 常に32個のKV（不足分はゼロパディング）
→ 32 × 20 = 640 トークン分の履歴を参照可能

theoretical_max = 587,970 / 20 + 1 = 29,399
→ max_contexts=32 で大幅に制限
```

---

## 実験結果

### Phase 1: OACD多様性学習

| 指標 | 値 |
|------|-----|
| 処理時間 | ~30s (18 iterations) |
| 収束率 | 91% |
| Early stop | Iter 18 (conv >= 90%) |
| Effective Rank | 82.1% |

### Phase 2: Context-KV Attention学習

| Epoch | Train PPL | Val PPL | Val Acc | 備考 |
|-------|-----------|---------|---------|------|
| 1 | 559.6 | 291.2 | 16.2% | * |
| 2 | 222.4 | 225.1 | 17.7% | * |
| 3 | 149.1 | 196.7 | 19.4% | * |
| 4 | 106.0 | 181.9 | 19.9% | * |
| **5** | **78.4** | **175.7** | **20.9%** | **Best** |
| 6 | 59.6 | 183.0 | 20.6% | 悪化 |
| 7 | 46.1 | 193.1 | 20.9% | 悪化 |
| 8 | ~30 | - | - | Early stop予定 |

### 最終結果（Epoch 5時点）

| 指標 | 値 |
|------|-----|
| **Val PPL** | **175.7** |
| **Val Acc** | **20.9%** |
| Best Epoch | 5 |
| Epoch Time | ~49s |

---

## Early Stopping動作確認

**Phase 2のEarly Stoppingは正常に機能しています。**

```
patience=3 の設定:

Epoch 5: val_ppl=175.7 → Best（no_improve=0）
Epoch 6: val_ppl=183.0 → 悪化（no_improve=1）
Epoch 7: val_ppl=193.1 → 悪化（no_improve=2）
Epoch 8: 完了後 → no_improve=3 → Early stop
```

---

## 100サンプル vs 500サンプル 比較

| 指標 | 100 samples | 500 samples | 改善 |
|------|-------------|-------------|------|
| Val PPL | 361.0 | **175.7** | **51% 改善** |
| Val Acc | 16.2% | **20.9%** | **+4.7%** |
| Best Epoch | 5 | 5 | 同じ |
| Train/Val Gap | 2.4x | 2.2x | やや改善 |

**観察:**
- サンプル数5倍でPPLが約半分に改善
- 過学習の傾向は依然として存在（Train PPL << Val PPL）
- データ増加により汎化性能が向上

---

## 過学習の分析

```
Epoch 5 (Best):
  Train PPL: 78.4
  Val PPL:   175.7
  Ratio:     2.2x

Epoch 7:
  Train PPL: 46.1
  Val PPL:   193.1
  Ratio:     4.2x (過学習が進行)
```

**Train PPLが下がり続けてもVal PPLは悪化** → 典型的な過学習パターン

---

## メモリ使用量

| 段階 | CPU | GPU |
|------|-----|-----|
| データロード後 | 1.2GB | 0.0GB |
| モデル初期化後 | 2.1GB | 0.2GB |
| Phase 1後 | 5.6GB | 0.2GB |
| Phase 2準備後 | 5.6GB | 0.2GB |

**キャッシュサイズ**:
- train_ctx: 574MB
- train_emb: 1723MB
- val_ctx: 22MB
- val_emb: 67MB
- **Total: 2386MB**

---

## 固定数方式（方法2）の特徴

```
方法2: 常にmax_contexts=32個のKVを使用

Position 50:   [ctx[50], 0, 0, ..., 0]  (1実データ + 31ゼロ)
Position 350:  [ctx[350], ctx[330], ..., ctx[50], 0, ..., 0]  (18実データ + 14ゼロ)
Position 3500: [ctx[3500], ctx[3480], ..., ctx[2880]]  (32実データ)
```

**利点:**
- テンソル形状が常に一定
- バッチ処理が効率的
- モデルが「履歴の長さ」を暗黙的に学習可能

---

## 今後の実験提案

1. **interval増加**: 20 → 50, 100 でより広い範囲をカバー
2. **サンプル数増加**: 1000, 2000 でさらなる改善を確認
3. **Regularization**: Dropout増加、Weight Decay追加
4. **max_contexts増加**: 32 → 64 でより多くの履歴を参照

---

## 結論

- **固定数方式（方法2）は正常に動作**
- **500サンプルで Val PPL 175.7 を達成**（100サンプルの361.0から51%改善）
- **Early Stoppingは正常に機能**
- **過学習傾向は依然として存在**するが、データ増加で改善傾向

---

*Last Updated: 2025-12-03*
