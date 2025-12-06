# New-LLM Project Guidelines

---

## 🎯 Infini-Pythia Architecture (Memory-Only)

**Pythia-70Mベースに1層目Infini-Attention（Memory-Only）を導入。**

### アーキテクチャ

```
Infini-Pythia:
Token Embedding (512-dim)
       ↓
Layer 0: InfiniAttentionLayer (NoPE, Memory Only)
  └─ Memory Attention (linear attention)
       ↓
Layer 1-5: PythiaLayer (RoPE)
  ├─ Multi-Head Attention
  └─ MLP
       ↓
Output Head (512 → vocab)
```

### Infini-Attention (Memory-Only)

```
メモリ更新 (Delta Rule):
  M_s = M_{s-1} + σ(K)^T @ (V - retrieved_V)

メモリ取得:
  A_mem = σ(Q) @ M / (σ(Q) @ z)

σ(x) = ELU(x) + 1
```

### Multi-Memory Bank

```python
# 複数メモリバンクで情報混合を低減
model = InfiniPythiaModel(
    num_memory_banks=2,      # 2つのバンク
    segments_per_bank=4,     # 4セグメントで次バンクに切替
)
```

### ALiBi (Attention with Linear Biases)

線形化近似でALiBiを圧縮メモリに組み込む:

```
メモリ更新 (ALiBi重み付き):
  M_φ = Σ_i w_i * φ(K_i) * V_i^T
  z_φ = Σ_i w_i * φ(K_i)

  w_i = exp(-slope * segment_distance)  # 遠いほど小さい重み

メモリ取得:
  Output = φ(Q) @ M_φ / (φ(Q) @ z_φ)
```

```python
# ALiBi付きモデル
model = InfiniPythiaModel(
    use_alibi=True,      # ALiBi有効化
    alibi_scale=1.0,     # スロープスケール（大きいほど減衰が強い）
)
```

### 実験の実行

```bash
# 標準実験（両モデル比較）
python3 scripts/experiment_infini.py --samples 5000 --epochs 30

# Infiniのみ
python3 scripts/experiment_infini.py --skip-baseline

# Baselineのみ
python3 scripts/experiment_infini.py --skip-infini

# Multi-Memory Bank
python3 scripts/experiment_infini.py --num-memory-banks 2 --segments-per-bank 4

# ALiBi位置エンコーディング
python3 scripts/experiment_infini.py --alibi --skip-baseline

# ALiBi (強い減衰)
python3 scripts/experiment_infini.py --alibi --alibi-scale 2.0 --skip-baseline

# Long Context訓練・評価
python3 scripts/experiment_infini.py --long-context-train --long-context
```

### Multi-Memory Infini-Attention (Attention-based Selection)

複数の独立したメモリをAttention-based方式で動的に選択・混合。

```
Multi-Memory Infini-Pythia:
Token Embedding (512-dim)
       ↓
Layer 0: MultiMemoryInfiniAttentionLayer
  ├─ Memory 0, 1, 2, ... (独立したメモリ)
  ├─ 関連度: phi(Q) @ z_i
  └─ Softmax重み付け混合
       ↓
Layer 1-5: PythiaLayer (RoPE)
       ↓
Output Head (512 → vocab)
```

**特徴**:
- 各メモリは独立して更新（ラウンドロビン）
- クエリとメモリのz（正規化項）との内積で関連度計算
- Softmax重み付けで全メモリを混合
- 追加パラメータなし（学習が安定）

```bash
# Multi-Memory実験（4メモリ）
python3 scripts/experiment_multi_memory.py --num-memories 4

# 8メモリで実験
python3 scripts/experiment_multi_memory.py --num-memories 8 --samples 10000

# ベースラインスキップ
python3 scripts/experiment_multi_memory.py --skip-baseline --num-memories 4
```

```python
from src.models.multi_memory_pythia import MultiMemoryInfiniPythiaModel

model = MultiMemoryInfiniPythiaModel(
    vocab_size=50304,
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    num_memories=4,  # 独立したメモリ数
)
```

---

## 📊 Reversal Curse 評価

### 概要

Reversal Curseは「A is B」を学習したモデルが「B is A」も推論できるかを測定する指標。

### 指標

| 指標 | 定義 | 解釈 |
|------|------|------|
| Forward PPL | 順方向文のPPL | 訓練データに含まれるため低い |
| Backward PPL | 逆方向文のPPL | 訓練データに含まれないため高い |
| Reversal Ratio | Forward / Backward | 1.0に近いほど良い |
| Reversal Gap | Backward - Forward | 0に近いほど良い |

### 実装

```python
from src.utils.evaluation import evaluate_reversal_curse
from src.data.reversal_pairs import get_reversal_pairs

tokenizer = get_tokenizer(config.tokenizer_name)
reversal_pairs = get_reversal_pairs()
reversal_result = evaluate_reversal_curse(model, tokenizer, reversal_pairs, device)
```

---

## 🚨 CRITICAL: コード品質

### 後方互換性コード禁止

**古い機能を残すことは厳禁。後方互換性を意識したコードは絶対に書かない。**

### ハードコード厳禁

**全ての値はconfigから読み込む。**

### ランダムデータ使用禁止

**実験でランダムデータ（torch.randint等）を使用することは絶対に禁止。**
必ず実データ（Pile）を使用すること。

### Reversal Curse評価必須

**すべての実験スクリプトで、Reversal Curse評価を必ず実行すること。**

---

## ⚠️ 過去のバグと教訓

### 1. Infini-Attention メモリ勾配バグ

```python
# ❌ バグ: メモリ更新でグラフが残り、二重backwardエラー
self.memory = self.memory + memory_update

# ✅ 修正: detach()でグラフを切断
self.memory = (self.memory + memory_update).detach()
```

### 2. PPL異常値の診断基準

| PPL | 状態 | 対処 |
|-----|------|------|
| < 5 | **異常** - データリーク/因果マスクバグ | コード点検必須 |
| 5-30 | **疑わしい** - 過学習の可能性 | データ量・分割を確認 |
| 30-100 | 正常（小規模データ） | - |
| 100-500 | 正常（スクラッチ訓練） | - |
| > 1000 | 学習不足 | epoch増加/lr調整 |

### 3. Long Context評価でuntrained modelを使用

```python
# ❌ バグ: 新しいモデルを作成して評価
pythia_model = PythiaModel(...)  # 未訓練のランダム重み
result = evaluate_long_documents(pythia_model, ...)

# ✅ 修正: 訓練済み重みをロード
pythia_model = PythiaModel(...)
pythia_model.load_state_dict(results["pythia"]["model_state_dict"])
result = evaluate_long_documents(pythia_model, ...)
```

---

## 📁 File Structure

```
new-llm/
├── config/
│   └── pythia.py                   # PythiaConfig
├── scripts/
│   └── experiment_infini.py        # Infini-Attention実験
├── src/
│   ├── data/
│   │   └── reversal_pairs.py       # Reversal Curse評価データ
│   ├── models/
│   │   ├── pythia.py               # PythiaModel (RoPE)
│   │   ├── infini_attention.py     # InfiniAttention, InfiniAttentionLayer
│   │   ├── infini_pythia.py        # InfiniPythiaModel (1層Infini + RoPE)
│   │   ├── multi_memory_attention.py  # MultiMemoryInfiniAttention
│   │   └── multi_memory_pythia.py  # MultiMemoryInfiniPythiaModel
│   └── utils/
│       ├── training.py             # 共通学習ユーティリティ
│       ├── evaluation.py           # 評価関数
│       ├── device.py               # デバイス管理
│       ├── data_pythia.py          # Pileデータ読み込み
│       └── seed.py                 # シード設定
├── docs/
│   └── experiments/                # 実験結果
└── CLAUDE.md
```

---

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-06 | **Multi-Memory Attention追加**: Attention-based選択で複数メモリを動的混合 |
| 2025-12-06 | **ALiBi位置エンコーディング追加**: 線形化近似でALiBiをメモリに組み込み |
| 2025-12-05 | **Memory-Onlyに集中**: Local Attention削除、コード簡素化 |
| 2025-12-05 | **Multi-Memory Bank追加**: 複数バンクで情報混合低減 |
| 2025-12-05 | **Long Context評価バグ修正**: 訓練済み重みをロード |
| 2025-12-05 | **Infini-Pythia実装**: 1層目Infini + RoPE |

---

Last Updated: 2025-12-06
