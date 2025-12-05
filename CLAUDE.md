# New-LLM Project Guidelines

---

## 🎯 MLA-Pythia Architecture (2025-12-05)

**Pythia-70MをベースにMLA（Multi-head Latent Attention）でKVキャッシュを大幅削減。**
**位置エンコーディングはALiBi（統一スロープ）を採用。**

### アーキテクチャ

```
Pythia (Baseline, RoPE):              MLA-Pythia (Ours, ALiBi):
Token Embedding (512-dim)             Token Embedding (512-dim)
       ↓                                     ↓
PythiaLayer × 6                       MLALayer × 6
  ├─ Attention (RoPE)                   ├─ MLA Attention (ALiBi)
  │    K: 512-dim                       │    c_kv: 128-dim (KV共通圧縮)
  │    V: 512-dim                       │    吸収モード
  └─ MLP                                └─ MLP
       ↓                                     ↓
Output Head (512 → vocab)             Output Head (512 → vocab)

KV Cache: K(512) + V(512) = 1024      KV Cache: c_kv(128) = 128
削減率: 0%                            削減率: 87.5%
```

### 設定値

| 項目 | Baseline (Pythia) | MLA-Pythia |
|------|-------------------|------------|
| hidden_size | 512 | 512 |
| kv_dim | - | 128 |
| Layers | 6 | 6 |
| Attention Heads | 8 | 8 |
| intermediate_size | 2048 | 2048 |
| Position Encoding | RoPE (25%) | ALiBi (統一スロープ) |
| KV Cache削減 | 0% | 87.5% |

### 実験の実行

```bash
# MLA実験: Pythia (RoPE) vs MLA-Pythia (ALiBi)
python3 scripts/experiment_mla.py --samples 10000 --epochs 30

# MLAのみ（baselineスキップ）
python3 scripts/experiment_mla.py --samples 10000 --skip-baseline

# kv_dim変更
python3 scripts/experiment_mla.py --kv-dim 256  # 75%削減
python3 scripts/experiment_mla.py --kv-dim 64   # 93.75%削減

# 位置エンコーディング比較実験（統一モデル使用）
python3 scripts/experiment_position.py --samples 10000 --epochs 30

# 特定の位置エンコーディングのみ
python3 scripts/experiment_position.py --pos-types rope alibi
python3 scripts/experiment_position.py --pos-types none  # NoPE
```

---

## 📚 DeepSeek MLA (Multi-head Latent Attention) 参考資料

### MLA概要

DeepSeek-V2で導入されたKVキャッシュ圧縮技術。K+Vを共通の低次元潜在ベクトルに圧縮し、「吸収」技法により復元せずにAttention計算を実現。

### 吸収モード（Absorbed Projection）の数式

```
標準MHA:
  scores = Q @ K^T
  output = softmax(scores) @ V

MLA（圧縮・復元あり）:
  c_kv = X @ W_DKV     # KV共通圧縮: (seq, 512) → (seq, 128)
  K = c_kv @ W_UK      # K復元
  V = c_kv @ W_UV      # V復元

MLA（吸収モード - 復元不要）:
  scores = Q @ K^T
        = Q @ (c_kv @ W_UK)^T
        = Q @ W_UK^T @ c_kv^T

  # KVキャッシュは c_kv のみ保存
```

### V処理（吸収モード）

```
output = softmax(scores) @ V
       = attn_weights @ (c_kv @ W_UV)
       = (attn_weights @ c_kv) @ W_UV  ← 結合法則
         ↑ 圧縮空間での計算    ↑ 最後に復元
```

### KVキャッシュの削減効果

| 方式 | キャッシュ内容 | 例（512-dim） | 削減率 |
|------|---------------|---------------|--------|
| 標準MHA | K(512) + V(512) | 1024 | 0% |
| MLA (kv_dim=128) | c_kv(128) | 128 | 87.5% |
| MLA (kv_dim=64) | c_kv(64) | 64 | 93.75% |

### 参考リンク

- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
- [MLA Explanation (HuggingFace)](https://huggingface.co/blog/NormalUhr/mla-explanation)
- [Understanding MLA](https://planetbanatt.net/articles/mla.html)

---

## 🎯 ALiBi (Attention with Linear Biases) 採用方針

### 採用決定事項

**本プロジェクトではRoPEの代わりにALiBiを採用する。**

理由:
- ALiBiはMLA（吸収モード）と完全に互換性がある
- RoPEは回転行列が位置依存のため、吸収モードで事前計算できない
- ALiBiは加算バイアスのため、吸収後のscoreに単純に加算可能

### ALiBi仕様

```
score = Q @ K^T - m * distance_matrix

distance_matrix[i][j] = |i - j|  # 位置間の距離
m = slope (全ヘッド統一)
```

### スロープ設定（重要）

**全ヘッドで統一スロープを使用する（ヘッドごとに異なるスロープは使用しない）**

理由:
- ヘッド分割は埋め込み次元を任意に分割したもの
- 異なる次元に異なるスロープを割り当てる理論的根拠が薄い
- シンプルな統一スロープで十分

```python
# ✅ 採用: 統一スロープ
slope = 0.0625  # デフォルト: 1/16
alibi_bias = -slope * distance_matrix

# ❌ 不採用: ヘッドごとに異なるスロープ
# slopes = 2 ** (-8 * torch.arange(1, num_heads + 1) / num_heads)
```

### ALiBi + MLA の組み合わせ

```
# MLA吸収モードとの互換性
score = Q @ W_UK^T @ c_kv^T - m * distance_matrix
        ↑ 事前計算可能      ↑ 加算バイアス（干渉なし）

# RoPEの場合（不可能）
score = (R_q @ Q) @ (R_k @ c_kv @ W_UK)^T
        ↑ 回転行列が位置依存のため事前計算不可
```

---

## 📊 Reversal Curse 評価

### 概要

Reversal Curseは「A is B」を学習したモデルが「B is A」も推論できるかを測定する指標。

### 正しい実験設計

```
訓練データ:
  - Pile（一般テキスト）
  - 順方向文のみ: "The capital of France is Paris"

評価データ:
  - 順方向: "The capital of France is Paris" → 低PPL期待
  - 逆方向: "Paris is the capital of France" → 高PPL（Reversal Curse）
```

### 指標

| 指標 | 定義 | 解釈 |
|------|------|------|
| Forward PPL | 順方向文のPPL | 訓練データに含まれるため低い |
| Backward PPL | 逆方向文のPPL | 訓練データに含まれないため高い |
| Reversal Ratio | Forward / Backward | 1.0に近いほど良い |
| Reversal Gap | Backward - Forward | 0に近いほど良い |

### 実装

- 訓練データ: `prepare_data_loaders(include_reversal_pairs=True)`
- 順方向文は10回繰り返して訓練データに追加
- 評価: `evaluate_reversal_curse(model, tokenizer, pairs, device)`

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

## ⚠️ 過去のバグと教訓

### 1. ALiBi因果マスクの行列方向バグ（2025-12-05）

**症状**: MLA-PythiaのPPLが異常に低い（1.5）、Pythiaは正常（424）

**原因**: `build_alibi_bias_causal`で行列の行と列が逆転していた

```python
# ❌ バグ: relative_pos[i][j] = j - i （未来が見えていた）
relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)

# ✅ 修正: relative_pos[i][j] = i - j （正しい因果マスク）
relative_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
```

**教訓**:
- PPL < 10 は異常。データ暗記または因果マスクのバグを疑う
- 行列演算では`unsqueeze`の順序（行/列）を必ず確認
- Attentionマスクは「queryが行、keyが列」が標準

### 2. PPL異常値の診断基準

| PPL | 状態 | 対処 |
|-----|------|------|
| < 5 | **異常** - データリーク/因果マスクバグ | コード点検必須 |
| 5-30 | **疑わしい** - 過学習の可能性 | データ量・分割を確認 |
| 30-100 | 正常（小規模データ） | - |
| 100-500 | 正常（スクラッチ訓練） | - |
| > 1000 | 学習不足 | epoch増加/lr調整 |

### 3. 因果マスクの検証方法

新しいAttention実装では必ず以下を確認：

```python
# テストコード
seq_len = 5
bias = build_alibi_bias_causal(seq_len, slope=0.0625)
print(bias)
# 期待出力: 上三角が-inf、下三角が負の値
# tensor([[  0., -inf, -inf, -inf, -inf],
#         [-0.0625,   0., -inf, -inf, -inf],
#         [-0.1250, -0.0625,   0., -inf, -inf],
#         ...])
```

---

## 📐 アーキテクチャ仕様

### Core Components

**1. UnifiedPythiaModel（統一モデル）**
- 位置エンコーディングを設定で切り替え可能
- RoPE, ALiBi, NoPE（なし）に対応
- 疎結合設計により拡張が容易

**2. PositionEncoding（位置エンコーディングモジュール）**
```python
# 使用例
from src.models import UnifiedPythiaModel, PositionEncodingConfig

# RoPE
model = UnifiedPythiaModel(pos_encoding=PositionEncodingConfig(type="rope"))

# ALiBi
model = UnifiedPythiaModel(pos_encoding=PositionEncodingConfig(type="alibi"))

# NoPE（位置情報なし）
model = UnifiedPythiaModel(pos_encoding=PositionEncodingConfig(type="none"))
```

**3. MLAPythiaModel（KVキャッシュ圧縮）**
- Token Embedding: vocab → hidden_size (512)
- MLALayer × 6: KV共通圧縮 (kv_dim=128)、ALiBi
- Output Head: hidden_size (512) → vocab_size

**4. PythiaModel (Baseline)**
- Token Embedding: vocab → hidden_size (512)
- PythiaLayer × 6: RoPE (25%)
- Output Head: hidden_size (512) → vocab_size

---

## 📁 File Structure

```
new-llm/
├── config/
│   ├── __init__.py
│   └── pythia.py                   # PythiaConfig
├── scripts/
│   ├── experiment_mla.py           # MLA実験: Pythia vs MLA-Pythia
│   └── experiment_position.py      # 位置エンコーディング比較実験
├── src/
│   ├── models/
│   │   ├── pythia.py               # PythiaModel (baseline, RoPE)
│   │   ├── mla_pythia.py           # MLAPythiaModel (ours, ALiBi)
│   │   ├── mla.py                  # MLAAttention, MLALayer
│   │   ├── alibi.py                # ALiBi実装
│   │   ├── position_encoding.py    # 位置エンコーディング統一モジュール
│   │   └── unified_pythia.py       # UnifiedPythiaModel（位置エンコ切替可能）
│   └── utils/
│       ├── training.py             # 共通学習ユーティリティ
│       ├── evaluation.py           # 評価関数
│       └── device.py               # デバイス管理
├── docs/
│   └── experiments/                # 実験結果
├── CLAUDE.md
└── README.md
```

---

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-05 | **位置エンコーディング統一化**: RoPE/ALiBi/NoPEを疎結合で切り替え可能に |
| 2025-12-05 | **ALiBi因果マスクバグ修正**: unsqueeze順序の修正、PPL異常の解消 |
| 2025-12-05 | **Reversal Curse評価追加**: 順方向/逆方向PPL比較機能 |
| 2025-12-05 | **MLA-Pythia実装**: V-DProjからMLA方式に移行、ALiBi採用 |
| 2025-12-05 | **ALiBi採用**: RoPEからALiBiに変更、統一スロープ方式 |
| 2025-12-04 | V-DProj実験（アーカイブ済み） |
| 2025-12-04 | DProj-Pythia実験（アーカイブ済み） |

---

## 📦 アーカイブ: DProj関連（参考用）

以下は過去の実験で使用した仕様です。現在はMLA方式に移行しています。

<details>
<summary>DProj Training 仕様（クリックで展開）</summary>

### DProj Trainingの目的

DiverseProjectionを使って、多様なprojection vectorを生成する。
OACD（Origin-Anchored Centroid Dispersion）損失で学習し、収束率90%以上を目指す。

### DiverseProjection/DiverseProjectionLayerの実装

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

### 初期化方法

```python
# src/utils/initialization.py
def init_linear_weights(module, weight_std=0.1, bias_std=0.01):
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.normal_(submodule.weight, mean=0.0, std=0.1)  # ⚠️ Xavier禁止
            if submodule.bias is not None:
                nn.init.normal_(submodule.bias, mean=0.0, std=0.01)
```

### OACD損失関数

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

### DProj Training 設定値

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `max_iterations` | 100 | 最大イテレーション数 |
| `convergence_threshold` | 0.03 | 収束判定のMSE閾値 |
| `learning_rate` | 0.003 | 学習率 |
| `batch_size` | 5000 | バッチサイズ |
| `gradient_clip` | 2.0 | 勾配クリッピング値 |
| `proj_noise` | 0.05 | ガウシアンノイズ |
| `early_stopping_threshold` | 0.95 | 収束率95%で早期停止 |

</details>

---

Last Updated: 2025-12-05
