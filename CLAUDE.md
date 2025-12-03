# New-LLM Project Guidelines

## 🎯 Project Goal: Pythia-70M + Context-KV Attention (2025-12-03)

**Pythia-70MのLayer 0をContext-KV Attentionに置き換え、KVキャッシュメモリを50%削減する。**

### Target Architecture

```
Pythia-70M (6 layers, hidden_size=512, heads=8)
  ↓
Layer 0: Context-KV Attention（置き換え）
Layer 1-5: Original Pythia Attention（維持）
```

### Key Decisions

| 項目 | 決定 |
|------|------|
| **置き換え層** | Layer 0 のみから開始 |
| **Context次元** | 256（積極的な圧縮） |
| **学習データ** | Pile（Pythiaと同じ）、開発時は限定サンプル |
| **学習方法** | Phase 1（OACD）→ Phase 2（全体ファインチューニング） |
| **評価指標** | PPL + LAMBADA |
| **メモリ削減目標** | 50% |

---

## 🎯 Context-KV Attention Architecture

**KVキャッシュを大幅に削減するContext-KV Attention方式を採用。**

### アーキテクチャ概要

```
Context-KV Attention:
  - 等間隔（interval）でContextを取得
  - 常に「現在位置」を含めたcontextでAttention
  - ~50% KVキャッシュ削減（Layer 0のみ置き換え時）
```

### 🚨 Context Interval方式（重要）

**Position i の予測には、現在位置から等間隔で過去のcontextを取得：**

```
interval = 32 の場合:

Position 350:
  KV Cache = [context[350], context[318], context[286], ...]
              ↑現在          ↑32前         ↑64前
           = 11 context vectors + zero padding

Position 1000:
  KV Cache = [context[1000], context[968], ..., context[8]]
           = 32 context vectors (max_contexts)
```

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

### Pythia-70M Base Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Hidden Size | 512 |
| Attention Heads | 8 |
| Total Parameters | 70M |

### Context-KV Replacement (Layer 0)

**1. ContextBlock**
- 1層固定、Phase 1で学習、Phase 2でfreeze
- OACDアルゴリズムで多様性学習
- context_dim = 256

**2. Context-KV Attention**
- ContextをK,Vに変換
- 等間隔（interval）でcontextを取得してAttention
- 常に現在位置のcontextを含める

### Training Pipeline

**Phase 1: Context多様性学習（OACD）**
- **学習対象**: ContextBlockのみ
- **損失**: OACD（多様性損失）
- **データ**: Pile（開発時は限定サンプル）

**Phase 2: 全体ファインチューニング**
- **ContextBlock**: frozen（重み固定）
- **Context-KV Attention**: 学習
- **Pythia Layer 1-5**: ファインチューニング
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
- `scripts/experiment_context_kv.py` - Context-KV Attention実験スクリプト（現行）
- `scripts/experiment_pythia_context_kv.py` - Pythia統合実験スクリプト（予定）

**Core Implementation**:
- `src/models/context_kv.py` - ContextKVAttentionLLM
- `src/models/blocks.py` - ContextBlock（1層固定）
- `src/models/layers.py` - ContextLayer
- `src/trainers/phase1/memory.py` - Phase 1訓練ロジック
- `src/losses/diversity.py` - OACDアルゴリズム

---

## Evaluation Metrics

### Primary

| Metric | Purpose |
|--------|---------|
| **PPL (Perplexity)** | 言語モデリング品質 |
| **LAMBADA Accuracy** | 長距離依存性（最終単語予測） |
| **KV Cache Memory** | 実際のメモリ使用量 |

### Comparison Plan

```
Baseline: Pythia-70M (original)
Ours:     Pythia-70M + Context-KV (Layer 0 replaced)

Evaluate on:
- WikiText-2 PPL
- Pile test set PPL
- LAMBADA accuracy
- torch.cuda.max_memory_allocated()
```

---

## Related Work

- **DeepSeek MLA**: Low-rank KV compression (トークンごと)
- **本プロジェクト**: Context-based KV compression (interval間隔)

---

Last Updated: 2025-12-03 (Pythia-70M統合方針に移行)
