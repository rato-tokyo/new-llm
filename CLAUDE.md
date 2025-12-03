# New-LLM Project Guidelines

## 🎯 Context-KV Attention Architecture (2025-12-03)

**KVキャッシュを大幅に削減するContext-KV Attention方式を採用。**

### アーキテクチャ概要

```
Context-KV Attention:
  - 等間隔（interval）でContextを取得
  - 常に「現在位置」を含めたcontextでAttention
  - ~99% KVキャッシュ削減
```

### 🚨 Context Interval方式（重要）

**Position i の予測には、現在位置から等間隔で過去のcontextを取得：**

```
interval = 100 の場合:

Position 350:
  KV Cache = [context[350], context[250], context[150], context[50]]
              ↑現在          ↑100前        ↑200前        ↑300前
           = 4 context vectors

Position 150:
  KV Cache = [context[150], context[50]]
              ↑現在          ↑100前
           = 2 context vectors

Position 50:
  KV Cache = [context[50]]
              ↑現在
           = 1 context vector
```

**ポイント：**
- 常に「現在位置のcontext」を含める（最新情報）
- 過去のcontextは等間隔（interval）で取得
- 古い「チャンク境界」方式ではなく、「現在位置基準」方式を使用

### 🚨 max_contexts（Context Window）設計方針

**通常LLMの「context window」と同様に、使用するcontext数に上限を設ける。**

```
通常LLM:
  - max_length で入力シーケンス長を制限
  - 古いトークンは切り捨て

Context-KV方式:
  - max_contexts で使用するcontext数を制限
  - 古いcontextは切り捨て
```

**例: interval=100, max_contexts=32 の場合**
```
Position 3500:
  理論上: [ctx[3500], ctx[3400], ..., ctx[0]] = 36個
  実際:   [ctx[3500], ctx[3400], ..., ctx[300]] = 32個（最新の32個のみ）

  → 古すぎるcontext（position 0〜200）は切り捨て
  → これにより 32 × 100 = 3200 トークン分の履歴を参照
```

**OOM防止の重要性：**
- max_contextsを設定しないと、長いシーケンスでAttention計算が爆発
- 通常LLMと同じ設計思想を維持することで、メモリ管理が容易

### 実験の実行

```bash
# Colab（GPU）: 200サンプル、interval=100
python3 scripts/experiment_context_kv.py -s 200 --chunk-size 100

# カスタムcontext次元
python3 scripts/experiment_context_kv.py -s 200 -c 256 --chunk-size 50
```

---

## 🎯 OACDアルゴリズム (Phase 1)

**Phase 1ではOACD (Origin-Anchored Centroid Dispersion) アルゴリズムを採用。**

```python
def oacd_loss(contexts, centroid_weight=0.1):
    # Term 1: 重心からの分散を最大化
    dispersion_loss = -||X - mean(X)|| / n

    # Term 2: 重心を原点に引き寄せる
    centroid_loss = ||mean(X)||²

    return dispersion_loss + centroid_weight * centroid_loss
```

---

## 🚨🚨 Phase 1学習では順次処理禁止 (CRITICAL) 🚨🚨

**Phase 1学習では、順次処理は厳禁。必ずshifted_prev_context方式で並列処理すること。**

```python
# ❌ 禁止: Phase 1学習で順次処理（非常に遅い）
for i in range(num_tokens):
    new_context = model.forward_context(prev_context, token_embed)
    prev_context = new_context

# ✅ 推奨: shifted_prev_context方式（並列処理）
for iteration in range(max_iterations):
    shifted_prev_context = torch.cat([zero_init, previous_contexts[:-1]], dim=0)
    new_contexts = model.forward_context(shifted_prev_context, input_embeds)
    previous_contexts = new_contexts
```

---

## 🚨 CPU/GPUテンソル管理

**大規模データでOOMを防ぐため、テンソルのデバイス管理を徹底。**

```python
# ❌ 修正前
batch_contexts = previous_contexts[start_idx:end_idx].detach()

# ✅ 修正後
batch_contexts = previous_contexts[start_idx:end_idx].detach().to(self.device)
```

---

## 🔧 開発環境のLint/Type Check

```bash
# Lint (ruff)
python3 -m ruff check src/

# Type check (mypy)
python3 -m mypy src/ --ignore-missing-imports

# 実験スクリプト
python3 -m ruff check scripts/experiment_context_kv.py
python3 -m mypy scripts/experiment_context_kv.py --ignore-missing-imports
```

---

## 🚨 CRITICAL: 後方互換性コード禁止

**古い機能を残すことは厳禁。後方互換性を意識したコードは絶対に書かない。**

---

## 📐 アーキテクチャ仕様

### Core Components

**1. ContextKVAttentionLLM**
- 複数のContextBlock（各1層固定）
- Context-KV Attention Layer
- Token Embedding: GPT-2 pretrained (768-dim, frozen)
- Weight Tying: token_output shares weights with token_embedding

**2. ContextBlock**
- 1層固定、Phase 1で学習、Phase 2でfreeze
- OACDアルゴリズムで多様性学習

**3. Context-KV Attention**
- ContextをK,Vに変換
- 等間隔（interval）でcontextを取得してAttention
- 常に現在位置のcontextを含める

### Phase 1: 多様性学習（OACD）

- **学習対象**: ContextBlockのみ
- **損失**: OACD（多様性損失）

### Phase 2: トークン予測

- **ContextBlock**: frozen（重み固定）
- **Context-KV Attention + FFN**: 学習
- **損失**: CrossEntropy（次トークン予測）

---

## Code Quality Standards

### Principles

1. **No Hardcoding**: All hyperparameters in config.py
2. **Single Responsibility**: Each module has one clear purpose
3. **Type Hints Required**: 関数・メソッドのパラメータには型注釈を必須

### 🚨🚨 ハードコード厳禁 - 全ての値はconfigから読み込む (CRITICAL) 🚨🚨

**実験スクリプトでパラメータをハードコードしない。全ての値はconfigから読み込む。**

**禁止事項:**
1. 関数のデフォルト引数に数値を直接書く
2. argparseのdefaultに数値を直接書く
3. コード内にマジックナンバーを書く

```python
# ❌ 禁止: 関数のデフォルト引数にハードコード
def train_phase2(..., num_epochs: int = 40, patience: int = 3):
    ...

# ❌ 禁止: argparseのdefaultにハードコード
parser.add_argument('--samples', type=int, default=200)

# ✅ 推奨: configから読み込み（関数）
def train_phase2(..., num_epochs: int, patience: int):  # デフォルト値なし
    ...

# ✅ 推奨: configから読み込み（argparse）
default_config = Config()
parser.add_argument('--samples', type=int, default=default_config.num_samples)

# ✅ 推奨: 呼び出し時にconfigから値を渡す
train_phase2(
    ...,
    num_epochs=base_config.phase2_epochs,
    patience=base_config.phase2_patience,
)
```

**Config ファイル構成:**
- `config/base.py` - モデルアーキテクチャ、データ設定、max_contexts、context_interval
- `config/phase1.py` - Phase 1学習パラメータ（max_iterations, early_stopping等）
- `config/phase2.py` - Phase 2学習パラメータ（epochs, patience, lr等）
- `config/__init__.py` - 統合Configクラス

**この方針の理由:**
- 設定変更はconfigファイルのみで完結
- 実験の再現性を保証
- パラメータの一元管理

---

## File Structure

**Main Scripts**:
- `scripts/experiment_context_kv.py` - Context-KV Attention実験スクリプト

**Core Implementation**:
- `src/models/context_kv.py` - ContextKVAttentionLLM
- `src/models/blocks.py` - ContextBlock（1層固定）
- `src/models/layers.py` - ContextLayer
- `src/trainers/phase1/memory.py` - Phase 1訓練ロジック
- `src/losses/diversity.py` - OACDアルゴリズム

---

Last Updated: 2025-12-03 (Context-KV Attention方式に完全移行)
