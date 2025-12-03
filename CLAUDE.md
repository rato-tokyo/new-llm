# New-LLM Project Guidelines

## 🎯 Context-KV Attention Architecture (2025-12-03)

**KVキャッシュを大幅に削減するContext-KV Attention方式を採用。**

### アーキテクチャ概要

```
Context-KV Attention:
  - 100トークンごとにContextを圧縮
  - 圧縮されたContextをKVキャッシュとして使用
  - ~99% KVキャッシュ削減

Position 350 の場合:
  KV Cache = [context_0-99, context_100-199, context_200-299, context_300-350]
           = 4 context vectors のみ
```

### 実験の実行

```bash
# Colab（GPU）: 200サンプル
python3 scripts/experiment_context_kv.py -s 200 --chunk-size 100

# カスタムcontext次元
python3 scripts/experiment_context_kv.py -s 200 -c 256 128 --chunk-size 50
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
- チャンク単位のcontextでAttention

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
