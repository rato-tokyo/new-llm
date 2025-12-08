# HSA-UltraLong: Hierarchical Sparse Attention

**論文**: Every Token Counts: Generalizing 16M Ultra-Long Context in Large Language Models
**著者**: Xiang Hu, Zhanchao Zhou, et al. (Ant Group, Westlake University)
**arXiv**: 2511.23319v1 (2025-11-28)

---

## 概要

長期記憶（Long-term Memory）を「効率的な超長コンテキストモデリング」として捉える研究。
32Kトークンで訓練し、16Mトークンまで外挿可能なモデルを実現。

### 3つの必須要件

| 要件 | 説明 |
|------|------|
| **Sparsity** | 全てを活性化せず、必要な部分のみを取得 |
| **Random-Access Flexibility** | 遠くの情報にもアクセス可能 |
| **Length Generalization** | 短いコンテキストから長いコンテキストへの汎化 |

---

## 従来手法の問題点

| 手法 | 問題 |
|------|------|
| Mamba/Linear Attention | 固定次元の状態ベクトル → 情報ボトルネック |
| Sliding Window | 遠いトークンにアクセス不可 |
| NSA/MoBA | チャンク選択が不正確、外挿性能が劣化 |

---

## HSA (Hierarchical Sparse Attention) の核心

### MoEとの類似性

```
MoE:
  Input → Router → Top-K Expert選択 → 各Expertで処理 → 重み付き統合

HSA:
  Token → Retrieval Score → Top-K Chunk選択 → 各Chunkと個別Attention → 重み付き統合
```

**重要**: 「チャンクを選んでから連結してAttention」ではなく、
「各チャンクと個別にAttentionしてからスコアで統合」

### 数式

1. **チャンク分割**: シーケンスを長さS（デフォルト64）のチャンクに分割

2. **Retrieval Score計算**:
```
s_{t,i} = Q_slc^T @ K_slc_i / sqrt(d)   (i <= t/S の場合)
        = -∞                             (i > t/S の場合、因果性)

I_t = {i | rank(s_{t,i}) < K}  # Top-Kチャンクのインデックス
```

3. **チャンク内Attention（NoPE）**:
```
O_bar_{t,i} = Softmax(norm(Q_attn) @ norm(K_[i]^T) / sqrt(d_h)) @ V_[i]
```
※ Query-Key Normalization使用（大規模訓練の安定性に重要）

4. **加重統合**:
```
w_{t,i} = exp(s_{t,i}) / Σ_{k∈I_t} exp(s_{t,k})
O_t = Σ_{k∈I_t} w_{t,k} @ O_bar_{t,k}
```

---

## モデルアーキテクチャ

```
┌─────────────────────────────────────┐
│         Upper Decoder                │
│  ┌─────────────────────────────────┐ │
│  │  MLP/MoE + SWA (×R)              │ │  × G groups
│  │  MLP/MoE + HSA + SWA (×1)        │ │
│  └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│         Lower Decoder                │
│  MLP/MoE + SWA (×L/2)                │
└─────────────────────────────────────┘
          ↑
   Chunk Encoder (Bi-directional)
   H^{L/2} + [CLS] → E, Landmark
```

### 設計のポイント

1. **Lower Decoder**: 標準Transformer + Sliding Window Attention (SWA)
2. **Chunk Encoder**: 中間層出力 H^{L/2} を双方向エンコーダで処理
3. **共有KVキャッシュ**: 中間層のKVキャッシュを全HSAモジュールで共有
4. **RoPE for short, NoPE for long**: SWAはRoPE使用、HSAはNoPE

---

## 訓練戦略

### 5段階訓練

| 段階 | SWA窓 | HSA Top-K | Context長 | 目的 |
|------|-------|-----------|-----------|------|
| 1. Warm-up | 512 | Full | 16K | HSAの検索能力を学習 |
| 2. Pre-training | 4K | Sparse | 16K | 通常の言語モデリング |
| 3. Long-context | 4K | Full | 32K | 長いコンテキストへ拡張 |
| 4. Annealing | 4K | - | 32K | 高品質データで仕上げ |
| 5. SFT | 4K | - | 8K | 指示追従能力 |

### Warm-upの重要性

```
問題: SWA窓が大きい(4K)と、HSAが短距離依存を学習しない
      → 長距離への外挿能力が発達しない

解決: Warm-up段階でSWA窓を小さく(512)設定
      → HSAが短距離パターンを学習
      → そのパターンが長距離に汎化
```

---

## 実験結果

### Needle-in-a-Haystack (16Mトークン)

- 32Kまでの訓練で、16Mトークンまで90%+の精度を維持
- Long-context mid-training後に大幅改善

### ベンチマーク比較

| Model | Params | Tokens | BBH | MMLU | GSM8K | HumanEval+ |
|-------|--------|--------|-----|------|-------|------------|
| Qwen2.5 | 0.5B | 18T | 32.3 | 49.7 | 41.3 | 24.4 |
| Qwen3 | 0.6B | 36T | 41.3 | 54.4 | 60.9 | 26.8 |
| **HSA-UL** | 0.5B | 4T | 18.2 | 41.8 | 37.5 | 29.3 |
| HSA-UL-MoE | 8B(1B active) | 8T | 60.1 | 60.7 | 72.9 | 61.6 |

---

## 発見と教訓

### 1. 長さ汎化の3要素

効果的な長さ汎化には**全て**が必要:
- Chunk-wise Attention
- Retrieval Score-based Fusion
- NoPE (No Positional Encoding)

### 2. HSA/SWAシーソー問題

```
SWA窓が大きすぎる → HSAが短距離を学習しない → 長距離外挿が失敗
SWA窓が小さすぎる → 短距離性能が低下
```

**解決**: Warm-upで小さいSWA窓 → Pre-trainingで大きいSWA窓

### 3. 訓練データの実効コンテキスト長

- 16Kで訓練しても、データの実効コンテキスト長が短いと外挿しない
- 32K以上の実効コンテキストを持つデータで訓練が必要

---

## 現在の課題

1. **HSA/SWAシーソー**: SFT後に外挿能力が劣化することがある
2. **ヘッド比率制約**: Query:KVヘッド = 16:1 が必要（情報ボトルネック）
3. **短いシーケンスでの効率**: FlashAttention-3より遅い

---

## 圧縮メモリへの適用案

### Chunk → Compressed Memory

| HSA (論文) | 圧縮メモリ版 |
|-----------|-------------|
| Chunk (64 tokens) | Memory行列 `(d, d)` |
| Landmark K_slc | Memory要約ベクトル |
| Chunk内KVキャッシュ | Linear Attention Memory |
| Top-K Chunk選択 | Top-K Memory選択 |
| Softmax Attention | Linear Attention検索 |

### 実装のポイント

```python
class CompressedMemoryHSA:
    def forward(self, query):
        # 1. Landmark（要約）との内積でスコア計算
        scores = [dot(query, landmark_i) for i in range(num_memories)]

        # 2. Top-K選択
        top_k_indices = top_k(scores, k)

        # 3. 選ばれたメモリから個別にLinear Attention検索
        outputs = [linear_attn(query, memory_i) for i in top_k_indices]

        # 4. スコアで加重統合
        weights = softmax([scores[i] for i in top_k_indices])
        return sum(w * o for w, o in zip(weights, outputs))
```

### 現在の実装との差分

| 項目 | 現在のMultiMemoryLayer | HSA方式 |
|------|----------------------|---------|
| Landmark | なし（memory_normを流用） | 明示的な要約ベクトル |
| Top-K選択 | なし（全メモリ参照） | あり |
| 計算量 | O(N) 全メモリ | O(K) 選択メモリ |
| スケーラビリティ | メモリ数に線形 | 大量メモリでも効率的 |

---

## 参考リンク

- GitHub: https://github.com/ant-research/long-context-modeling
- 関連論文:
  - [18] Hardware-aligned HSA (NeurIPS 2025)
  - [19] Efficient length-generalizable attention (ICML 2025)
  - [23] Understanding length generalization in HSA (arXiv 2510.17196)

---

Last Updated: 2025-12-08
