# NEW-LLM 技術仕様書

## 概要

NEW-LLMは、**CVFP（Context Vector Fixed-Point）理論**に基づく新しい言語モデルアーキテクチャです。従来のTransformerとは異なり、**固定点学習**と**分離アーキテクチャ**を採用しています。

### 主な特徴

1. **2段階学習（Phase 1/Phase 2）**: 文脈学習とトークン予測を分離
2. **固定点学習**: イテレーションを重ねて安定した文脈表現に収束
3. **分離アーキテクチャ（E案）**: ContextBlockとTokenBlockを物理的に分離
4. **GPT-2埋め込み活用**: 事前学習済み768次元埋め込みを使用

---

## アーキテクチャ

### 全体構造

```
入力トークン
    ↓
Token Embedding (GPT-2 pretrained, 768次元, frozen)
    ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 1: ContextBlock（固定点学習）                      │
│  - 3層のContextLayer                                    │
│  - 入力: [context, token_embed]                          │
│  - 出力: 更新されたcontext                                │
│  - 損失: CVFP損失 + 多様性損失                            │
└─────────────────────────────────────────────────────────┘
    ↓ (Phase 1学習後、freezeして)
┌─────────────────────────────────────────────────────────┐
│  Phase 2: TokenBlock（トークン予測学習）                   │
│  - 3層のTokenLayer                                      │
│  - E案: Layer i は ContextBlock Layer i の出力を参照       │
│  - 損失: CrossEntropy（次トークン予測）                   │
└─────────────────────────────────────────────────────────┘
    ↓
token_output (Linear: 768 → 50257)
    ↓
次トークン予測
```

### E案：レイヤー対応アーキテクチャ

```
ContextBlock (Phase 1で学習、Phase 2でfreeze):
  Layer 1: [context_0, token_embed] → context_1
  Layer 2: [context_1, token_embed] → context_2
  Layer 3: [context_2, token_embed] → context_3 (= C*)

TokenBlock (Phase 2で学習):
  Layer 1: [context_1, token_embed] → token_1
  Layer 2: [context_2, token_1]     → token_2
  Layer 3: [context_3, token_2]     → token_3 (= token_out)
```

**重要**: TokenBlock Layer i は ContextBlock Layer i の出力を参照する

---

## コンポーネント詳細

### 1. ContextLayer（文脈処理レイヤー）

```python
class ContextLayer:
    入力: context [batch, 768], token_embed [batch, 768]
    処理:
        1. concat([context, token_embed]) → [batch, 1536]
        2. Linear(1536 → 768) + ReLU → delta_context
        3. LayerNorm(context + delta_context) → new_context
    出力: new_context [batch, 768]
```

- **目的**: 文脈ベクトルを更新（token_embedは参照のみ、更新されない）
- **残差接続**: 安定した学習のために使用
- **LayerNorm**: 数値安定性のために必須

### 2. TokenLayer（トークン処理レイヤー）

```python
class TokenLayer:
    入力: context [batch, 768], token [batch, 768]
    処理:
        1. concat([context, token]) → [batch, 1536]
        2. Linear(1536 → 768) + ReLU → delta_token
        3. LayerNorm(token + delta_token) → new_token
    出力: new_token [batch, 768]
```

- **目的**: contextを参照しながらトークン表現を更新
- **contextは読み取り専用**: contextは更新されず、参照のみ

### 3. LLMクラス（メインモデル）

```python
class LLM:
    - token_embedding: GPT-2事前学習済み (50257 × 768, frozen)
    - embed_norm: LayerNorm(768)
    - context_block: ContextBlock(3層)
    - token_block: TokenBlock(3層)
    - token_output: Linear(768 → 50257)
```

---

## 学習プロセス

### Phase 1: CVFP固定点学習

**目的**: 各トークンに対して安定した文脈ベクトル（固定点 C*）を学習

**CVFP理論**: `f(x) = x` となる固定点に収束させる

```python
# 処理フロー
Iteration 0: シーケンシャル処理（学習なし、previous_contextsを初期化）
Iteration 1+: 並列処理（前回contextを使用）

# 損失関数
CVFP損失 = MSE(contexts, previous_contexts)  # 前回との差
多様性損失 = -L2_norm(deviation_from_mean)   # 平均からの偏差を最大化
総損失 = (1 - dist_reg_weight) × CVFP損失 + dist_reg_weight × 多様性損失
```

**現在の設定**:
- `dist_reg_weight = 0.5`（50% CVFP, 50% Diversity）
- `max_iterations = 30`
- `convergence_threshold = 0.03`
- `learning_rate = 0.002`

**収束条件**: `MSE(context_t, context_{t-1}) < threshold`

**並列処理の仕組み**:
- Iteration 0: シーケンシャル処理で初期fixed-pointを確立
- Iteration 1+: 並列バッチ処理（23倍高速化）
- Token i には previous_contexts[i-1] を使用（1トークン分のずれ）

### Phase 2: トークン予測学習

**目的**: 固定点文脈を使って次トークンを予測

```python
# 処理フロー（E案）
1. ContextBlock(frozen): 各レイヤーの出力を取得 [context_1, ..., context_N]
2. TokenBlock(学習): Layer i は context_i を参照して token を更新
3. Prediction: token_out → logits → CrossEntropy

# 学習対象
- TokenBlock: 全パラメータ
- token_output: Linear層
- ContextBlock: frozen（勾配なし）
```

**現在の設定**:
- `learning_rate = 0.002`
- `epochs = 10`
- `batch_size = 512`
- `gradient_clip = 1.0`
- `patience = 2`（Early stopping）

---

## 評価指標

### 1. Effective Rank（多様性指標）

コンテキストベクトルが多様な次元を使用しているかを測定

```python
# 計算方法
1. 特異値分解 (SVD): contexts → U, S, V
2. エントロピー計算: H = -Σ(s_i/Σs) × log(s_i/Σs)
3. Effective Rank = exp(H)

# 目標値
- 訓練データ: ~74% (568/768次元)
- 検証データ: ~66.6% (511/768次元)

# 警告レベル
- < 30%: 多様性不足（次元が偏っている）
- Global attractor: 全トークンが同じcontextに収束
```

### 2. Identity Mapping Check（恒等写像チェック）

モデルが学習できているか確認

```python
# チェック項目
1. ゼロベクトルとの差分 > 0.1
2. トークン埋め込みとの類似度 < 0.95
3. コンテキストの多様性（平均からの偏差）

# 失敗パターン
- 学習後のcontextがゼロに近い → 学習していない
- contextがtoken_embedと同一 → 恒等写像
```

### 3. CVFP収束チェック

固定点学習が成功しているか確認

```python
# 指標
- 収束率 = 収束トークン数 / 総トークン数
- Final diff = 最終イテレーションでのMSE

# 目標
- final_diff < 0.001 (GOOD)
- イテレーション間の変化が減少傾向
```

---

## ハイパーパラメータ

```python
# モデルアーキテクチャ
context_dim = 768          # コンテキスト次元（GPT-2に合わせる）
embed_dim = 768            # 埋め込み次元
context_layers = 3         # ContextBlockのレイヤー数
token_layers = 3           # TokenBlockのレイヤー数
vocab_size = 50257         # GPT-2語彙サイズ
layernorm_mix = 0.0        # LayerNorm混合比率（0.0で無効）

# Phase 1: CVFP学習
phase1_learning_rate = 0.002
phase1_max_iterations = 30
phase1_convergence_threshold = 0.03
phase1_min_converged_ratio = 1.01  # Early stopping無効化（101%不可能）
dist_reg_weight = 0.5              # 多様性正則化の重み

# Phase 2: トークン予測
phase2_learning_rate = 0.002
phase2_epochs = 10
phase2_batch_size = 512
phase2_patience = 2
phase2_gradient_clip = 1.0
```

---

## データ仕様

```python
# 訓練データ
- ソース: UltraChat (HuggingFaceH4/ultrachat_200k)
- サンプル数: 500（設定可能）
- シーケンス長: 128トークン/サンプル
- 総トークン数: ~64000

# 検証データ
- ソース: 訓練データの最後20%から生成
- 必須条件: 全トークンが訓練データに存在すること
- 生成: 自動（データ生成スクリプトが作成）
```

---

## ファイル構造

```
new-llm/
├── config.py                  # 全設定値
├── test.py                    # Phase 1テスト
├── train.py                   # フル訓練
├── src/
│   ├── models/
│   │   └── llm.py             # LLM, ContextBlock, TokenBlock
│   ├── trainers/
│   │   ├── phase1.py          # Phase 1訓練（CVFP固定点学習）
│   │   └── phase2.py          # Phase 2訓練（トークン予測）
│   ├── evaluation/
│   │   └── metrics.py         # Effective Rank, Identity Check等
│   └── data/
│       └── loader.py          # データローダー
├── data/
│   └── ultrachat_*_val.txt    # 検証データ
├── cache/
│   └── ultrachat_*_*.pt       # キャッシュされた訓練データ
└── checkpoints/
    └── model_latest.pt        # 保存されたモデル
```

---

## 重要な設計判断

### 1. なぜ分離アーキテクチャか？

- **C*の保持**: Phase 1で学習した文脈表現がPhase 2で変質しない
- **段階的学習**: 複雑なタスクを2つの単純なタスクに分解
- **物理的分離**: ContextBlockとTokenBlockの役割が明確

### 2. なぜE案（レイヤー対応）か？

- **段階的文脈情報**: 浅いレイヤーでは浅い文脈、深いレイヤーでは深い文脈
- **Transformerとの類似性**: 各レイヤーで異なる深さの表現を参照
- **情報フローの自然さ**: Layer iの出力がLayer iに入力される

### 3. なぜ固定点学習か？

- **安定した文脈**: イテレーションを重ねても変化しない表現
- **理論的基盤**: `f(x) = x` という明確な学習目標
- **並列化可能**: 収束後は1回の順伝播でcontextを得られる

---

## 既知の問題と制約

### 1. コンテキスト引き継ぎの重要性

```python
# 致命的バグ（修正済み）
# ❌ 毎イテレーションでリセット → CVFP学習が機能しない
context = torch.zeros(...)

# ✅ 前イテレーションのcontextを引き継ぐ
context = previous_contexts[-1].unsqueeze(0).detach()
```

### 2. 正規化の禁止（CVFP損失計算時）

```python
# ❌ 正規化すると方向のみで値が収束しない
cvfp_loss = MSE(F.normalize(new), F.normalize(prev))

# ✅ 生のMSEで値の一致を確認
cvfp_loss = MSE(new, prev)
```

### 3. 検証データの制約

- 検証データは訓練データに含まれるトークンのみを使用
- `auto_split`モードは禁止（エラー発生）
- 訓練データから自動生成する必要がある

---

## 性能ベンチマーク

### CPU (Apple Silicon/Intel)

```
- Phase 1訓練: ~11秒（並列版、6400トークン）
- Phase 1訓練: ~265秒（シーケンシャル版、比較参照）
- 検証評価: ~4秒（1280トークン）
- 処理速度: 250-330 tok/s
```

### 期待される結果

```
# 並列版（dist_reg_weight=0.5）
- 訓練 Effective Rank: ~74% (568/768)
- 検証 Effective Rank: ~66.6% (511/768)
- 訓練 収束率: ~30%
- 検証 収束率: ~100%
```

---

## よくある質問

### Q: Transformerとの主な違いは？

A:
- **Attention機構なし**: 代わりに固定点学習で文脈を圧縮
- **2段階学習**: 文脈学習とトークン予測を分離
- **イテレーティブ処理**: 複数回の順伝播で固定点に収束

### Q: なぜGPT-2埋め込みを使う？

A:
- **学習済み意味表現**: 単語の意味関係がすでに学習されている
- **次元の一致**: 768次元で一貫したアーキテクチャ
- **学習の安定化**: ランダム初期化よりも安定

### Q: Effective Rankが低い場合は？

A:
- `dist_reg_weight`を上げる（多様性重視）
- LayerNormが正しく機能しているか確認
- コンテキスト引き継ぎが正しいか確認

### Q: Phase 2の精度が低い場合は？

A:
- Phase 1のEffective Rankを確認（低いと文脈情報が不足）
- 学習率を調整
- バッチサイズを調整
- ContextBlockが正しくfreezeされているか確認

---

## 変更履歴

- **2025-11-26**: E案アーキテクチャ採用（レイヤー対応版）
- **2025-11-25**: 並列処理版導入（23x高速化）
- **2025-11-24**: コンテキスト引き継ぎバグ修正
- **2025-11-24**: F.normalize()バグ修正
