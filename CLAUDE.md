# New-LLM Project Guidelines

---

## ⚠️⚠️⚠️ CLAUDE AIへの重要な警告 ⚠️⚠️⚠️

**このセクションは絶対に削除しないでください。**

### 過去の問題

2025-12-04にPythia統合を試みた際、CLAUDE.mdの編集時にDProj学習の重要な仕様が誤って削除されました。
これにより、DProj学習が正常に収束しなくなり、プロジェクトを以前の状態にリバートする必要がありました。

### ルール

1. **DProj学習仕様セクションは絶対に削除禁止**
2. **DiverseProjection/DiverseProjectionLayerの実装詳細は削除禁止**
3. **初期化方法（normal_ std=0.1）の記述は削除禁止**
4. CLAUDE.mdを編集する際は、既存の重要なセクションが残っていることを必ず確認すること

---

## 🎯 DProj-Pythia Architecture (2025-12-04)

**Pythia-70MをベースにKVキャッシュを削減するDProj-Pythia方式を採用。**

### ⚠️ 重要な設計方針（絶対に守ること）

**Baselineとの違いは「Token Embedding → DiverseProjection」の圧縮部分のみ。**
**PythiaLayer自体は同じ構造（RoPE含む）で、hidden_size=proj_dimで動作させる。**

```
Pythia (Baseline):                    DProj-Pythia (Ours):
Token Embedding (512-dim)             Token Embedding (512-dim)
       ↓                                     ↓
       │                              DiverseProjection (512 → 320)  ← ここだけ違う
       ↓                                     ↓
PythiaLayer × 6 (512-dim, RoPE)       PythiaLayer × 6 (320-dim, RoPE)
       ↓                                     ↓
Output Head (512 → vocab)             Output Head (320 → vocab)

KV Cache: 3072 KB                     KV Cache: 1920 KB (37.5%削減)
```

### 設定値

| 項目 | Baseline (Pythia) | DProj-Pythia |
|------|-------------------|--------------|
| embed_dim | 512 | 512 |
| hidden_size / proj_dim | 512 | 320 |
| Layers | 6 | 6 |
| Attention Heads | 8 | 8 |
| intermediate_size | 2048 | 1280 |
| Position Encoding | RoPE (25%) | RoPE (25%) |
| Vocab Size | 50,304 | 50,304 |

### 学習フロー

```
DProj Training: OACD (DiverseProjection多様性学習)
  ├─ DiverseProjectionのみを学習
  ├─ OACD損失で多様なprojection vectorを生成
  └─ 収束まで実行（~60 iterations, 90%+収束）
       ↓
Main Training: Full Training (DiverseProjection frozen)
  ├─ DiverseProjectionをfreeze
  ├─ PythiaLayer × 6 + Output Headを学習 (proj_dim=320で動作)
  └─ Cross-entropy損失
```

### 実験の実行

```bash
# DProj Training: DiverseProjection OACD学習
python3 scripts/train_dproj.py --samples 1000

# Main Training: 比較実験
python3 scripts/experiment_pythia_comparison.py --samples 10000 --epochs 10
```

### ⚠️ DProj Training コマンドラインオプションの制約

**DProj Trainingは`--samples`のみ使用可能。`--tokens`オプションは禁止。**

理由: サンプル数で指定することで、データサイズが直感的に理解しやすくなる。

### ⚠️ proj_dim の制約

**`proj_dim`は`num_attention_heads` (8) で割り切れる値が推奨。**

割り切れない場合は自動的に切り上げて調整される:
- 300 → 304 (304 / 8 = 38)
- 250 → 256 (256 / 8 = 32)

有効な値の例:
- 256 (256 / 8 = 32) ← 50%圧縮
- 320 (320 / 8 = 40) ← デフォルト、37.5%圧縮
- 384 (384 / 8 = 48) ← 25%圧縮

---

## 🚨🚨🚨 DProj Training 完全仕様（削除禁止）🚨🚨🚨

**このセクションは試行錯誤の末に確立された必須仕様です。絶対に削除しないでください。**

### DProj Trainingの目的

DiverseProjectionを使って、多様なprojection vectorを生成する。
OACD（Origin-Anchored Centroid Dispersion）損失で学習し、収束率90%以上を目指す。

### DiverseProjection/DiverseProjectionLayerの実装（削除禁止）

```python
# src/models/dproj.py - DiverseProjectionLayer
class DiverseProjectionLayer(nn.Module):
    def __init__(self, proj_input_dim, proj_output_dim, token_input_dim):
        # FFN: Linear(input_dim → output_dim) + GELU
        input_dim = proj_input_dim + token_input_dim
        self.ffn = FFN(input_dim, proj_output_dim)

        # LayerNorm（必須：数値安定性のため）
        self.proj_norm = nn.LayerNorm(proj_output_dim)

        # 残差接続用の射影（次元が異なる場合のみ）
        if proj_input_dim != proj_output_dim:
            self.residual_proj = nn.Linear(proj_input_dim, proj_output_dim)

        # ⚠️ 重要: 初期化は normal_(std=0.1)
        init_linear_weights(self)  # weight: std=0.1, bias: std=0.01

    def forward(self, prev_proj, token_embeds):
        ffn_input = torch.cat([prev_proj, token_embeds], dim=-1)
        delta = self.ffn(ffn_input)

        # 残差接続 + LayerNorm
        new_proj = self.proj_norm(prev_proj + delta)
        return new_proj
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

**⚠️ 警告**: Xavier初期化やKaiming初期化を使用すると、DProj学習が収束しません。
必ず `normal_(std=0.1)` を使用してください。

### OACD損失関数（削除禁止）

```python
# src/losses/diversity.py
def oacd_loss(projections, centroid_weight=0.1):
    proj_mean = projections.mean(dim=0)
    deviation = projections - proj_mean

    # Term 1: 重心からの分散を最大化（負の損失で最大化）
    dispersion_loss = -torch.norm(deviation, p=2) / len(projections)

    # Term 2: 重心を原点に引き寄せる
    centroid_loss = torch.norm(proj_mean, p=2) ** 2

    return dispersion_loss + centroid_weight * centroid_loss
```

### DProj Training 設定値（削除禁止）

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `max_iterations` | 100 | 最大イテレーション数 |
| `convergence_threshold` | 0.03 | 収束判定のMSE閾値 |
| `learning_rate` | 0.003 | 学習率 |
| `batch_size` | 5000 | バッチサイズ |
| `gradient_clip` | 2.0 | 勾配クリッピング値 |
| `proj_noise` | 0.05 | ガウシアンノイズ（収束優先） |
| `early_stopping_threshold` | 0.95 | 収束率95%で早期停止 |

### embed_norm（埋め込み正規化）（削除禁止）

```python
# ⚠️ 重要: 埋め込み後の正規化が必須（DProj学習収束に必要）
self.embed_norm = nn.LayerNorm(hidden_size)

# 使用時
token_embeds = model.embed_in(token_ids)
token_embeds = model.embed_norm(token_embeds)  # ⚠️ 必須
```

**⚠️ 警告**: embed_normがないとDProj学習が収束しません。

### shifted_prev_proj方式（並列処理）（削除禁止）

```python
# ❌ 禁止: 順次処理（非常に遅い）
for i in range(num_tokens):
    new_proj = model.dproj(prev_proj, token_embed)
    prev_proj = new_proj

# ✅ 必須: shifted_prev_proj方式（並列処理）
for iteration in range(max_iterations):
    # ゼロベクトルから開始
    init_proj = torch.zeros(1, proj_dim)
    shifted_prev_proj = torch.cat([init_proj, previous_projs[:-1]], dim=0)

    # バッチ並列処理
    new_projs = model.dproj(shifted_prev_proj, token_embeds)
    previous_projs = new_projs
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
# token_embedsとprevious_projsはCPUに保持
token_embeds = token_embeds_gpu.cpu()
previous_projs = projs.cpu()

# バッチごとにGPUに転送して処理
for start_idx in range(0, num_tokens, batch_size):
    batch_projs = previous_projs[start_idx:end_idx].to(device)
    batch_embeds = token_embeds[start_idx:end_idx].to(device)

    # 処理後は即座にCPUに戻す
    all_projs_cpu.append(batch_output.detach().cpu())
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

**1. DProjPythiaModel (Ours)**
- Token Embedding: vocab → embed_dim (512)
- embed_norm: LayerNorm（DProj学習収束に必須）
- DiverseProjection: embed_dim (512) → proj_dim (320)
- PythiaLayer × 6: hidden_size=proj_dim (320)、RoPE含む
- Output Head: proj_dim (320) → vocab_size

**2. DiverseProjection (DProj)**
- 1層固定、DProj Trainingで学習、Main Trainingでfreeze
- OACDアルゴリズムで多様性学習
- 初期化: normal_(std=0.1)

**3. PythiaModel (Baseline)**
- Token Embedding: vocab → hidden_size (512)
- PythiaLayer × 6: hidden_size (512)
- Output Head: hidden_size (512) → vocab_size

---

## 📁 File Structure

```
new-llm/
├── checkpoints/
│   └── dproj_pythia.pt           # DProj checkpoint
├── config/
│   ├── __init__.py
│   ├── dproj.py                  # DProj Training設定
│   └── pythia.py                 # PythiaConfig, DProjPythiaConfig
├── scripts/
│   ├── train_dproj.py            # DProj Training: DiverseProjection OACD学習
│   ├── experiment_pythia_comparison.py  # Pythia vs DProj-Pythia比較
│   └── experiment_ka_comparison.py      # KA-Attention実験
├── src/
│   ├── models/
│   │   ├── pythia.py             # PythiaModel (baseline)
│   │   ├── dproj_pythia.py       # DProjPythiaModel (ours)
│   │   ├── dproj.py              # DiverseProjection, DiverseProjectionLayer
│   │   └── ka_attention.py       # KA-Attention実験
│   ├── losses/
│   │   └── diversity.py          # OACD algorithm
│   └── utils/
│       ├── data_pythia.py        # Pileデータローダー
│       ├── training.py           # 共通学習ユーティリティ
│       └── initialization.py     # 重み初期化
├── CLAUDE.md
└── README.md
```

---

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-04 | **Rename**: Phase 1 → DProj Training, ContextBlock → DiverseProjection |
| 2025-12-04 | **KA-Attention**: V を A に置き換える実験実装 |
| 2025-12-04 | **重要**: PythiaLayerをproj_dim (320)で動作させる設計に変更 |
| 2025-12-04 | Main Training比較実験スクリプト追加、DProj Trainingパラメータ調整 |
| 2025-12-04 | embed_norm追加（DProj学習収束に必須） |
| 2025-12-04 | Pythia-70M統合（DProj-Pythia方式に完全移行） |
| 2025-12-04 | DProj学習仕様の詳細を追記（Pythia統合失敗からの復旧後） |
| 2025-12-03 | Context-KV Attention方式（旧方式、削除済み） |

---

Last Updated: 2025-12-04
