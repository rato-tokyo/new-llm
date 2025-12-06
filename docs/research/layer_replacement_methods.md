# Pretrained LLMのレイヤー置換・修正に関する既存研究

**作成日**: 2025-12-06

## 概要

Pretrained LLMの特定レイヤーを置換・修正する研究は**比較的少ない**。多くの研究は以下のいずれかに分類される：

1. **全レイヤー修正**: 全アーキテクチャを変更して継続事前訓練
2. **アダプター挿入**: 元のレイヤーを保持しつつ追加モジュールを挿入
3. **パラメータ効率的微調整**: LoRA等で既存重みを低ランク更新

## 1. アダプター方式（レイヤー保持型）

### LoRA (Low-Rank Adaptation)

**論文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021)

```
元の重み W ∈ R^{d×k} を凍結
低ランク行列 B ∈ R^{d×r}, A ∈ R^{r×k} を追加
出力 = Wx + BAx
```

**特徴**:
- 元のレイヤーを**完全に保持**
- 並列に低ランク更新を加算
- 学習パラメータが大幅削減（GPT-3で10,000分の1）

**レイヤー選択オプション**:
- `layers_to_transform`: 特定レイヤーのみにLoRAを適用可能
- 通常はAttention層のみに適用（計算効率のため）
- QLoRA論文では全Linear層への適用を推奨

### Adapter Methods

**参考**: [AdapterHub Documentation](https://docs.adapterhub.ml/methods.html)

| 方式 | 説明 |
|------|------|
| **Houlsby Adapter** | Attention後とFFN後に挿入 |
| **Parallel Adapter** | 元のレイヤーと並列に配置 |
| **AdapterPlus** | MHA後にチャネルスケーリング付き |

**重要ポイント**: すべてのAdapter方式は**元のレイヤーを保持**し、追加モジュールの出力を加算する。

## 2. 全レイヤー置換型

### Infini-Attention (Google, 2024)

**論文**: [Leave No Context Behind](https://arxiv.org/abs/2404.07143)

**実験設定**:

| モデル | 方法 | 結果 |
|--------|------|------|
| 1B LLM | **全MHA**をInfini-attentionに置換 + 4K長で30K steps継続事前訓練 | 1M Passkey成功 |
| 8B LLM | **全MHA**をInfini-attentionに置換 + 8K長で30K steps継続事前訓練 | BookSum SOTA |

**重要**: 単一レイヤーではなく**全レイヤー**を置換している。

### MAMBA / State Space Models

**特徴**: Attention機構自体を廃止し、State Space Modelに置換

- Selectionメカニズムで入力依存のデータ選択
- MLPレイヤーとの統合で単一ブロック化
- 最初から新アーキテクチャとして訓練（後付け置換ではない）

### DyT (Dynamic Tanh, 2025)

**対象**: LayerNorm / RMSNormの置換

```
従来: LayerNorm(x) or RMSNorm(x)
DyT:  α * tanh(x / β)
```

**成功例**:
- LLaMA 7B〜70Bで同等性能
- 訓練時間8.2%削減、推論時間7.8%削減

**成功の理由**: 正規化層は**情報のボトルネックではない**ため、置換が容易。

## 3. 単一/少数レイヤー置換の困難さ

### 問題点

```
[元のLayer N-1] → [置換したLayer N] → [元のLayer N+1]
        ↓                ↓                    ↓
   期待される分布    異なる分布出力      分布不整合で性能低下
```

1. **入出力分布の不整合**: 後続レイヤーが新しい出力分布に適応できない
2. **位置情報の喪失**: RoPE→NoPEの変換で位置情報が失われる
3. **残差接続の前提崩壊**: 深層モデルは各レイヤーの「微小な修正」を前提

### Hugging Face Infini-Attention失敗実験

**参考**: [A failed experiment: Infini-Attention](https://huggingface.co/blog/infini-attention)

- Llama 3 8Bでの継続事前訓練を試行
- **失敗**: Needle評価で早期セグメントの検索に失敗
- **結論**: 圧縮回数が増えると性能が低下

## 4. 成功パターンの分析

### 成功しやすいケース

| パターン | 例 | 理由 |
|----------|-----|------|
| **並列追加** | LoRA, Parallel Adapter | 元のパスを保持 |
| **正規化層置換** | DyT | 情報ボトルネックでない |
| **全体置換 + 大規模再訓練** | Infini-Attention (Google) | 全体の整合性を再構築 |
| **スクラッチ訓練** | MAMBA, RWKV | 不整合が存在しない |

### 失敗しやすいケース

| パターン | 例 | 理由 |
|----------|-----|------|
| **単一Attention層置換** | 本実験 | 後続レイヤーとの不整合 |
| **部分的アーキテクチャ変更** | - | 残差接続の前提崩壊 |
| **位置エンコーディング変更** | RoPE→NoPE | 位置情報の喪失 |

## 5. 推奨アプローチ

### Pretrained LLMへの機能追加

1. **Parallel Adapter方式**（推奨）
   ```
   output = original_layer(x) + α * new_module(x)
   ```
   - 元の性能を保持
   - αを0から学習開始で安全

2. **LoRA方式**
   - 低ランク更新で既存重みを微調整
   - 特定レイヤーのみに適用可能

3. **継続事前訓練 + 全レイヤー修正**
   - 大規模データが必要（PG19, Arxiv等）
   - 30K+ stepsの訓練が必要

### 長文対応の代替手法

| 手法 | 説明 | 複雑さ |
|------|------|--------|
| **RoPE Scaling** | 既存RoPEの周波数を調整 | 低 |
| **YaRN** | NTK-aware interpolation | 中 |
| **Ring Attention** | 分散処理でKVキャッシュを共有 | 高 |
| **Sliding Window** | 固定長ウィンドウ + グローバルトークン | 中 |

## 6. 結論

### 単一レイヤー置換が少ない理由

1. **技術的困難**: 入出力分布の不整合が深刻
2. **代替手法の存在**: LoRA/Adapterが安全で効果的
3. **全体置換の方が効果的**: 部分修正より全体再構築の方が結果が良い

### 本プロジェクトへの示唆

| 選択肢 | 実現可能性 | 既存性能 |
|--------|------------|----------|
| Parallel Adapter（α学習） | ◎ | 維持 |
| 全レイヤーInfini + 継続事前訓練 | △（計算コスト大） | 要再構築 |
| スクラッチ訓練Full Infini | ○ | N/A |
| RoPE Scaling採用 | ◎ | 維持 |

## 参考文献

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [AdapterHub Methods](https://docs.adapterhub.ml/methods.html)
- [Infini-Attention](https://arxiv.org/abs/2404.07143)
- [HF Infini-Attention Failed Experiment](https://huggingface.co/blog/infini-attention)
- [Transformer Alternatives 2024](https://nebius.com/blog/posts/model-pre-training/transformer-alternatives-2024)
- [Advancing Long-Context LLMs Survey](https://arxiv.org/abs/2311.12351)
- [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms)
