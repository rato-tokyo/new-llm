# New-LLM Project Guidelines

---

## ⚠️⚠️⚠️ CLAUDE AIへの重要な警告 ⚠️⚠️⚠️

**このセクションは絶対に削除しないでください。**

### 過去の問題

2025-12-04にPythia統合を試みた際、CLAUDE.mdの編集時にPhase 1の重要な仕様が誤って削除されました。
これにより、Phase 1の学習が正常に収束しなくなり、プロジェクトを以前の状態にリバートする必要がありました。

### ルール

1. **Phase 1仕様セクションは絶対に削除禁止**
2. **ContextBlock/ContextLayerの実装詳細は削除禁止**
3. **初期化方法（normal_ std=0.1）の記述は削除禁止**
4. CLAUDE.mdを編集する際は、既存の重要なセクションが残っていることを必ず確認すること

---

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

## 🚨🚨🚨 Phase 1 完全仕様（削除禁止）🚨🚨🚨

**このセクションは試行錯誤の末に確立された必須仕様です。絶対に削除しないでください。**

### Phase 1の目的

ContextBlockを使って、多様なcontext vectorを生成する。
OACD（Origin-Anchored Centroid Dispersion）損失で学習し、収束率90%以上を目指す。

### ContextBlock/ContextLayerの実装（削除禁止）

```python
# src/models/layers.py - ContextLayer
class ContextLayer(nn.Module):
    def __init__(self, context_input_dim, context_output_dim, token_input_dim):
        # FFN: Linear(input_dim → output_dim) + GELU
        input_dim = context_input_dim + token_input_dim
        self.fnn = FFN(input_dim, context_output_dim)

        # LayerNorm（必須：数値安定性のため）
        self.context_norm = nn.LayerNorm(context_output_dim)

        # 残差接続用の射影（次元が異なる場合のみ）
        if context_input_dim != context_output_dim:
            self.residual_proj = nn.Linear(context_input_dim, context_output_dim)

        # ⚠️ 重要: 初期化は normal_(std=0.1)
        init_linear_weights(self)  # weight: std=0.1, bias: std=0.01

    def forward(self, context, token_embeds):
        fnn_input = torch.cat([context, token_embeds], dim=-1)
        delta_context = self.fnn(fnn_input)

        # 残差接続 + LayerNorm
        new_context = self.context_norm(context + delta_context)
        return new_context
```

### 初期化方法（削除禁止）

```python
# src/utils/initialization.py
def init_linear_weights(module, weight_std=0.1, bias_std=0.01):
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.normal_(submodule.weight, mean=0.0, std=0.1)  # ⚠️ Xavier禁止
            if submodule.bias is not None:
                nn.init.normal_(submodule.bias, mean=0.0, std=0.01)
```

**⚠️ 警告**: Xavier初期化やKaiming初期化を使用すると、Phase 1が収束しません。
必ず `normal_(std=0.1)` を使用してください。

### OACD損失関数（削除禁止）

```python
# src/losses/diversity.py
def oacd_loss(contexts, centroid_weight=0.1):
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean

    # Term 1: 重心からの分散を最大化（負の損失で最大化）
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)

    # Term 2: 重心を原点に引き寄せる
    centroid_loss = torch.norm(context_mean, p=2) ** 2

    return dispersion_loss + centroid_weight * centroid_loss
```

### Phase 1 設定値（削除禁止）

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `max_iterations` | 60 | 最大イテレーション数 |
| `convergence_threshold` | 0.03 | 収束判定のMSE閾値 |
| `learning_rate` | 0.002 | 学習率 |
| `batch_size` | 5000 | バッチサイズ |
| `gradient_clip` | 2.0 | 勾配クリッピング値 |
| `context_noise` | 0.1 | ガウシアンノイズ（汎化性能向上） |
| `early_stopping_threshold` | 0.9 | 収束率90%で早期停止 |

### shifted_prev_context方式（並列処理）（削除禁止）

```python
# ❌ 禁止: 順次処理（非常に遅い）
for i in range(num_tokens):
    new_context = model.forward_context(prev_context, token_embed)
    prev_context = new_context

# ✅ 必須: shifted_prev_context方式（並列処理）
for iteration in range(max_iterations):
    # ゼロベクトルから開始
    init_ctx = torch.zeros(1, context_dim)
    shifted_prev_context = torch.cat([init_ctx, previous_contexts[:-1]], dim=0)

    # バッチ並列処理
    new_contexts = model.context_block(shifted_prev_context, token_embeds)
    previous_contexts = new_contexts
```

### 勾配累積（削除禁止）

```python
# バッチごとに勾配を計算・累積
optimizer.zero_grad()
for batch in batches:
    loss = oacd_loss(batch_output)
    scaled_loss = loss / num_batches  # バッチ数で割る
    scaled_loss.backward()  # 勾配累積

# 最後にまとめて更新
torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
optimizer.step()
```

### CPU/GPUメモリ分離（削除禁止）

```python
# token_embedsとprevious_contextsはCPUに保持
token_embeds = token_embeds_gpu.cpu()
previous_contexts = contexts.cpu()

# バッチごとにGPUに転送して処理
for start_idx in range(0, num_tokens, batch_size):
    batch_contexts = previous_contexts[start_idx:end_idx].to(device)
    batch_embeds = token_embeds[start_idx:end_idx].to(device)

    # 処理後は即座にCPUに戻す
    all_contexts_cpu.append(batch_output.detach().cpu())
```

### 収束率計算（削除禁止）

```python
def compute_convergence_rate(current, previous, threshold=0.03):
    token_losses = ((current - previous) ** 2).mean(dim=1)
    converged_count = (token_losses < threshold).sum()
    return converged_count / len(current)
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

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-04 | Phase 1仕様の詳細を追記（Pythia統合失敗からの復旧後） |
| 2025-12-03 | Context-KV Attention方式に完全移行 |

---

Last Updated: 2025-12-04
