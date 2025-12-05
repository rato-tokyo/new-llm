# PretrainedMKA V2 実験結果

## 実験日時
2025-12-04

## 用語

- **Attention Output (A)**: `attention_weights @ V` の結果
- **KA Attention**: Attention Output同士のAttention

## 実験概要

事前学習済みPythia-70Mを凍結し、KA Attention層のみを学習する方式の検証。

### アーキテクチャ

```
PretrainedMKA V2:
  Stage 1: Pretrained Pythia (Frozen) → A (Attention Output)
  Stage 2: KA Attention (Trainable) → Attention Output同士のattention

  Parameters:
    Total: 71,215,616
    Trainable: 788,992 (1.1%)
    Frozen: 70,426,624 (98.9%)
```

## 実験設定

| 項目 | 値 |
|------|-----|
| Samples | 10,000 |
| Sequence length | 128 |
| Epochs | 30 |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Early stopping | 3 epochs |

## 結果

### 全体PPL

| Model | val_ppl | Note |
|-------|---------|------|
| Pythia-70M (pretrained) | **26.6** | frozen baseline |
| PretrainedMKA V2 | **32.9** | epoch 29 |

**Difference: +6.3 ppl (23.7%悪化)**

### Position-wise PPL (長距離依存性評価)

| Position | Pythia | MKA | Diff | 悪化率 |
|----------|--------|-----|------|--------|
| 0-16 | 77.1 | 96.9 | +19.8 | 25.7% |
| 16-32 | 33.6 | 42.2 | +8.6 | 25.6% |
| 32-64 | 25.0 | 30.9 | +5.8 | 23.2% |
| 64-96 | 21.1 | 26.2 | +5.2 | 24.6% |
| 96-128 | 18.8 | 23.3 | +4.5 | 23.9% |

### 学習曲線

```
Epoch  1: train_ppl=438.5 val_ppl=316.8
Epoch 10: train_ppl=38.3  val_ppl=44.4
Epoch 20: train_ppl=28.1  val_ppl=34.1
Epoch 29: train_ppl=26.1  val_ppl=32.9 (best)
```

## 分析

### 1. Position-wise PPLの傾向

- **後ろの位置ほどPPLが低い**: これは正常（コンテキストが多いため予測しやすい）
- **悪化率は全位置でほぼ一定** (23-26%): KA attentionが一様にノイズを追加

### 2. 問題点

1. **KA attention層が有害**: Pythiaの出力に対して余計な処理を加えている
2. **長距離依存性の改善なし**: 後ろの位置でも悪化率は変わらない
3. **Residual connectionの効果不足**: `A + ka_output`でも元のPythia性能を回復できない

### 3. 根本的な問題

現在の設計では：
- Pythiaは既に最適化された出力Aを生成
- KA attentionは「A同士の関係」を学習するが、これは冗長
- 結果として、元の情報を壊すだけ

## 結論

**PretrainedMKA V2アプローチは失敗**

KA attentionを事後的に追加する方式では、事前学習済みモデルの性能を超えられない。

---

## 次の実験: V-DProj (V圧縮方式)

### 仮説

KVキャッシュのV（Value）に対してDProj（次元圧縮）を適用し、適切な逆行列で復元する方式。

```
V-DProj方式:
  V (512-dim) → DProj → V_compressed (320-dim) → InvProj → V_restored (512-dim)

  学習目標: V_restored ≈ V となるようにDProjとInvProjを学習
```

### 理論的根拠

1. **線形変換の可逆性**: DProjが行列Wなら、逆行列W^(-1)で復元可能
2. **情報保存**: 適切に学習すれば、圧縮しても重要な情報を保持
3. **KVキャッシュ削減**: 推論時はV_compressedのみ保存（37.5%削減）

### 期待される効果

- **精度低下の抑制**: 復元学習により情報損失を最小化
- **KVキャッシュ削減**: 512 → 320 dim (37.5%削減)
- **Pythia互換**: 既存のAttention構造を維持

### 実装予定

```python
class VDProjAttention:
    def __init__(self):
        self.v_proj = nn.Linear(512, 320)      # V圧縮
        self.v_inv_proj = nn.Linear(320, 512)  # V復元

    def forward(self, hidden_states):
        # Standard Q, K
        Q, K, V = self.qkv_proj(hidden_states)

        # V compression & restoration
        V_compressed = self.v_proj(V)
        V_restored = self.v_inv_proj(V_compressed)

        # Attention with restored V
        attn_output = attention(Q, K, V_restored)

        return attn_output
```

### 学習方式

1. **Reconstruction Loss**: `||V - V_restored||^2`を最小化
2. **End-to-end**: Language Modeling Lossと同時に学習
3. **Scheduled Training**: 最初はReconstruction重視、徐々にLM重視

---

## 実験履歴

| 日付 | 実験 | 結果 |
|------|------|------|
| 2025-12-04 | KA-Attention (V→A置換) | Pythia 442 vs KA 449 (+7 ppl) |
| 2025-12-04 | PretrainedMKA V2 | Pythia 26.6 vs MKA 32.9 (+6.3 ppl) |
| 2025-12-04 | V-DProj (予定) | - |
