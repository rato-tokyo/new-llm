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

## 🎯 Context-Pythia Architecture (2025-12-04)

**Pythia-70MをベースにKVキャッシュを50%削減するContext-Pythia方式を採用。**

### アーキテクチャ概要

```
Context-Pythia:
  Token Embedding (512-dim) ← Pythia-70M
       ↓
  ContextBlock: 512 → 256 (圧縮)
       ↓
  Layer 0-5: 全て context (256-dim) を入力
       ↓
  Output Head (vocab_size=50304)

KVキャッシュ削減: 50%
  元: hidden_size (512) × seq_len × num_layers (6)
  圧縮後: context_dim (256) × seq_len × num_layers (6)
```

### Pythia-70M仕様

| 項目 | 値 |
|------|-----|
| Hidden Size | 512 |
| Layers | 6 |
| Attention Heads | 8 |
| Intermediate Size | 2048 |
| Position Encoding | Rotary (RoPE, 25%) |
| Vocab Size | 50,304 |
| Training Data | Pile (~300B tokens) |

### 実験の実行

```bash
# Phase 1: ContextBlock OACD学習
python3 scripts/train_phase1_pythia.py --tokens 100000

# Phase 2: 比較実験（準備中）
python3 scripts/experiment_pythia_comparison.py --samples 10000 --epochs 10
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
| `max_iterations` | 100 | 最大イテレーション数 |
| `convergence_threshold` | 0.03 | 収束判定のMSE閾値 |
| `learning_rate` | 0.003 | 学習率 |
| `batch_size` | 5000 | バッチサイズ |
| `gradient_clip` | 2.0 | 勾配クリッピング値 |
| `context_noise` | 0.05 | ガウシアンノイズ（収束優先） |
| `early_stopping_threshold` | 0.9 | 収束率90%で早期停止 |

### embed_norm（埋め込み正規化）（削除禁止）

```python
# ⚠️ 重要: 埋め込み後の正規化が必須（Phase 1収束に必要）
self.embed_norm = nn.LayerNorm(hidden_size)

# 使用時
token_embeds = model.embed_in(token_ids)
token_embeds = model.embed_norm(token_embeds)  # ⚠️ 必須
```

**⚠️ 警告**: embed_normがないとPhase 1が収束しません。

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

## 🔧 開発環境

### Lint/Type Check

```bash
# Lint (ruff)
python3 -m ruff check src/

# Type check (mypy)
python3 -m mypy src/ --ignore-missing-imports
```

---

## 🚨 CRITICAL: コード品質

### 後方互換性コード禁止

**古い機能を残すことは厳禁。後方互換性を意識したコードは絶対に書かない。**

### ハードコード厳禁

**全ての値はconfigから読み込む。**

### ランダムデータ使用禁止（厳禁）

**実験でランダムデータ（torch.randint等）を使用することは絶対に禁止。**
必ず実データ（Pile）を使用すること。

---

## 📐 アーキテクチャ仕様

### Core Components

**1. ContextPythiaModel**
- Token Embedding: Pythia-70M (512-dim)
- ContextBlock: 512 → 256 (圧縮)
- 6 Context-based Transformer Layers
- Output Head: 512 → vocab_size

**2. ContextBlock**
- 1層固定、Phase 1で学習、Phase 2でfreeze
- OACDアルゴリズムで多様性学習
- 初期化: normal_(std=0.1)

**3. PythiaModel (Baseline)**
- 標準のPythia-70M再実装
- 比較用

### 学習フロー

```
Phase 1: OACD (ContextBlock多様性学習)
  ├─ ContextBlockのみを学習
  ├─ OACD損失で多様なcontext vectorを生成
  └─ 収束まで実行（~60 iterations, 90%+収束）
       ↓
Phase 2: Full Training (ContextBlock frozen)
  ├─ ContextBlockをfreeze
  ├─ Transformer Layers + Output Headを学習
  └─ Cross-entropy損失
```

---

## 📁 File Structure

```
new-llm/
├── checkpoints/
│   └── context_block_pythia_phase1.pt  # Phase 1 checkpoint
├── config/
│   ├── __init__.py
│   ├── phase1.py              # Phase 1設定
│   └── pythia.py              # PythiaConfig, ContextPythiaConfig
├── scripts/
│   ├── train_phase1_pythia.py         # Phase 1: ContextBlock OACD学習
│   └── experiment_pythia_comparison.py # Phase 2: Pythia vs Context-Pythia比較
├── src/
│   ├── models/
│   │   ├── pythia.py          # PythiaModel (baseline)
│   │   ├── context_pythia.py  # ContextPythiaModel (ours)
│   │   ├── blocks.py          # ContextBlock
│   │   └── layers.py          # ContextLayer
│   ├── losses/
│   │   └── diversity.py       # OACD algorithm
│   └── utils/
│       ├── data_pythia.py     # Pileデータローダー
│       └── initialization.py  # 重み初期化
├── CLAUDE.md
└── README.md
```

---

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-04 | Phase 2比較実験スクリプト追加、Phase 1パラメータ調整 |
| 2025-12-04 | embed_norm追加（Phase 1収束に必須） |
| 2025-12-04 | Pythia-70M統合（Context-Pythia方式に完全移行） |
| 2025-12-04 | Phase 1仕様の詳細を追記（Pythia統合失敗からの復旧後） |
| 2025-12-03 | Context-KV Attention方式（旧方式、削除済み） |

---

Last Updated: 2025-12-04
