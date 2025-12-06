# 2段階訓練（蒸留 + Fine-tuning）実験結果

**日付**: 2025-12-06
**実験**: Pretrained Pythia-70mへのInfini-Attention導入（Layer 0置換）

## 概要

Pretrained LLMのLayer 0をInfini-Attention（Linear Attention + 圧縮メモリ）に置き換える際の問題を解決するため、2段階訓練アプローチを試行した。

## 実験設定

### アーキテクチャ

```
Stage 1: Knowledge Distillation
  オリジナルLayer 0 (RoPE) → ターゲット出力
  Infini Layer (NoPE)     → 予測出力
  Loss = MSE(予測, ターゲット)

Stage 2: Full Fine-tuning with Layer-wise LR
  Layer 0 (Infini): lr = base_lr × 2.0
  他のレイヤー:     lr = base_lr × 0.5
```

### ハイパーパラメータ

| パラメータ | 値 |
|------------|-----|
| Base Model | EleutherAI/pythia-70m |
| Distillation Epochs | 10 |
| Fine-tuning Epochs | 30 (early stopped at 7) |
| Distillation LR | 1e-4 |
| Fine-tuning Base LR | 1e-5 |
| Layer 0 LR Scale | 2.0x |
| Other Layers LR Scale | 0.5x |
| Segment Length | 256 |
| Memory Head Dim | 512 (single-head) |

### パラメータ数

| コンポーネント | パラメータ数 |
|----------------|-------------|
| Total | 73,576,961 |
| Infini Layer | 3,150,337 |

## 結果

### PPL推移

| Stage | PPL | 変化 |
|-------|-----|------|
| Baseline (Original Pythia) | 44.08 | - |
| After Distillation | 68.75 | +24.67 (56%劣化) |
| After Fine-tuning | 1,237.67 | +1,193.59 (**28倍劣化**) |

### Stage 1: Knowledge Distillation

```
Epoch  1: distill_loss=0.013433 val_ppl=15457.5
Epoch  2: distill_loss=0.010438 val_ppl=15694.9
...
Epoch 10: distill_loss=0.009237 val_ppl=16632.1
```

**観察**:
- Distillation lossは継続的に減少（0.0134 → 0.0092）
- しかしval_pplは逆に上昇（15,457 → 16,632）
- Sliding window評価後のPPL: 68.75（Baselineの44.08から55%劣化）

### Stage 2: Full Fine-tuning

```
Epoch 1: train_ppl=509.8 val_ppl=437.9
Epoch 2: train_ppl=313.6 val_ppl=385.7
...
Epoch 6: train_ppl=175.1 val_ppl=341.1 (best)
Epoch 7: train_ppl=155.7 val_ppl=345.2 (early stop)
```

**観察**:
- Train PPLは順調に減少（509 → 155）
- Val PPLも減少傾向だが、最終評価（sliding window）で大幅劣化
- 過学習の兆候：segment-based評価とsliding window評価の乖離

## 失敗の原因分析

### 1. Linear AttentionとRoPEの根本的非互換性

```
Layer 0 (Infini): Linear Attention (NoPE)
  ↓
Layer 1-5 (Pythia): Self-Attention (RoPE)
```

- Pretrained Pythia Layer 1-5は、**RoPE付きの入力分布**を前提に訓練されている
- Layer 0をLinear Attention (NoPE)に置き換えると、出力分布が変化
- 後続レイヤーが適応しきれない

### 2. 蒸留の限界

- MSE Lossで出力を模倣しても、**内部表現の特性**までは複製できない
- RoPEによる位置情報がLinear Attentionでは表現できない
- 蒸留が「成功」してもPPLは改善しない

### 3. Fine-tuning時の過学習

- Segment-based評価（訓練と同じ条件）では改善
- Sliding window評価（標準的な条件）では大幅劣化
- モデルがセグメント分割の特性に過適合

## 既存研究との比較

### Google Infini-Attention論文 (2024)

| 実験 | 方法 | 結果 |
|------|------|------|
| 1B LLM | 全レイヤーMHA→Infini置換 + 30K steps継続事前訓練 | Passkey 1M成功 |
| 8B LLM | 全レイヤーMHA→Infini置換 + 30K steps継続事前訓練 | BookSum SOTA |

**重要な違い**:
- Googleは**全レイヤー**を置換
- **継続事前訓練**（4Kトークン）を30Kステップ実施
- 我々は**1レイヤーのみ**置換、WikiText-2のみで訓練

### Hugging Face失敗実験 (2024)

> "Infini-attention's performance gets worse as we increase the number of times we compress the memory"

- Llama 3 8Bでの継続事前訓練で失敗
- Needle評価で早期セグメントの検索に失敗
- **結論**: Ring Attention、YaRN、RoPE scalingの方が信頼性が高い

## 教訓

### 1. 単一レイヤー置換は機能しない

Pretrained LLMの特定レイヤーのみを異なるアーキテクチャに置き換えることは、以下の理由で困難：

- **入出力分布の不整合**: 後続レイヤーが新しい出力分布に適応できない
- **位置情報の喪失**: RoPE→NoPEの変換で位置情報が失われる
- **蒸留の限界**: 出力を模倣しても内部表現は複製できない

### 2. 有効なアプローチ

| アプローチ | 説明 | 既存性能 |
|------------|------|----------|
| **全レイヤー置換** | 全MHAをInfiniに置換 + 継続事前訓練 | 要再構築 |
| **スクラッチ訓練** | 最初からInfiniアーキテクチャで訓練 | N/A |
| **Parallel Adapter** | 元のレイヤーを保持 + α×Infini出力を加算 | 維持可能 |
| **RoPE Scaling** | 既存RoPEを拡張（YaRN等） | 維持可能 |

### 3. Infini-Attentionの制約

- **圧縮回数の増加で性能低下**: メモリを繰り返し圧縮すると情報が失われる
- **既存LLMへの後付けが困難**: アーキテクチャ変更は継続事前訓練が必須
- **Linear Attentionの表現力**: Softmax Attentionより表現力が低い

## 次のステップ候補

1. **Full Infini Model**: 全レイヤーをInfini-Attentionにしてスクラッチ訓練
2. **継続事前訓練の規模拡大**: より大規模なデータ（PG19, Arxiv等）で訓練
3. **Hybrid方式**: Layer 0は元のまま、最終層にInfiniを追加
4. **RoPE Scaling**: Infiniを諦め、既存のRoPE拡張手法を採用

## 参考文献

- [Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143)
- [A failed experiment: Infini-Attention (Hugging Face)](https://huggingface.co/blog/infini-attention)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Advancing Transformer Architecture in Long-Context LLMs](https://arxiv.org/abs/2311.12351)
