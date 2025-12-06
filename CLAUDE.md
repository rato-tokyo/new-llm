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

---

## 🏭 モデルファクトリ

`create_model()`でシンプルにモデル作成。

```python
from src.models import create_model

# 基本的な使い方
model = create_model("pythia")       # 標準Pythia
model = create_model("infini")       # Infini-Pythia
model = create_model("multi_memory") # Multi-Memory
model = create_model("hierarchical") # Hierarchical

# オプション付き
model = create_model("infini", use_alibi=True, alibi_scale=1.0)
model = create_model("multi_memory", num_memories=8)
model = create_model("hierarchical", num_memories=4, use_delta_rule=False)

# カスタムconfig
from config.pythia import PythiaConfig
config = PythiaConfig()
model = create_model("infini", config, use_alibi=True)
```

### 利用可能なオプション

| オプション | 対象モデル | デフォルト | 説明 |
|------------|------------|------------|------|
| `use_delta_rule` | 全memory系 | `True` | Delta Rule使用 |
| `num_memories` | multi_memory, hierarchical | `4` | メモリ数 |
| `num_memory_banks` | infini | `1` | メモリバンク数 |
| `segments_per_bank` | infini | `4` | バンクあたりセグメント数 |
| `use_alibi` | infini | `False` | ALiBi有効化 |
| `alibi_scale` | infini | `1.0` | ALiBiスロープスケール |

---

## 💾 メモリ状態の保存・転送

圧縮メモリを別PCに転送可能。

```python
import torch
from src.models import create_model

# ===== PC A =====
model = create_model("infini")
model.reset_memory()

# テキスト処理でメモリを蓄積
for batch in data_loader:
    _ = model(batch, update_memory=True)

# メモリ状態を保存
state = model.get_memory_state()
torch.save(state, "memory.pt")

# ===== PC B =====
# メモリ状態を読み込み
state = torch.load("memory.pt")
model = create_model("infini")
model.set_memory_state(state)

# メモリが引き継がれた状態で推論
output = model(input_ids)
```

### メモリ状態のキー

| モデル | キー |
|--------|------|
| `InfiniPythiaModel` | `memories`, `memory_norms`, `current_bank`, `segment_counter` |
| `MultiMemoryInfiniPythiaModel` | `memories`, `memory_norms`, `current_memory_idx` |
| `HierarchicalMemoryPythiaModel` | `fine_memories`, `fine_memory_norms`, `current_memory_idx` |

### メモリサイズ

| モデル | サイズ |
|--------|--------|
| Infini (1 bank) | ~135 KB |
| Multi-Memory (4) | ~540 KB |
| Hierarchical (4) | ~540 KB |

---

## 🧪 統一実験スクリプト

全モデルを統一スクリプトで実験可能。

```bash
# 全モデル比較
python3 scripts/experiment.py --models pythia infini multi_memory hierarchical

# Infiniのみ
python3 scripts/experiment.py --models infini

# Multi-MemoryとHierarchical比較
python3 scripts/experiment.py --models multi_memory hierarchical --num-memories 4

# ALiBi付きInfini
python3 scripts/experiment.py --models infini --alibi

# 設定カスタマイズ
python3 scripts/experiment.py --models infini --samples 10000 --epochs 50 --lr 5e-5

# 8メモリで実験
python3 scripts/experiment.py --models hierarchical --num-memories 8
```

### モデルタイプ

| タイプ | 説明 |
|--------|------|
| `pythia` | 標準Pythia (RoPE) |
| `infini` | Infini-Pythia (1層目Infini + RoPE) |
| `multi_memory` | Multi-Memory (複数独立メモリ) |
| `hierarchical` | Hierarchical (階層的メモリ) |

### プログラムからの使用

```python
from src.utils.experiment_runner import (
    ExperimentConfig,
    ModelType,
    run_experiment,
)

# 設定
config = ExperimentConfig(
    num_samples=5000,
    seq_length=256,
    num_epochs=30,
    num_memories=4,
)

# 実験実行
results = run_experiment(
    model_types=[ModelType.INFINI, ModelType.HIERARCHICAL],
    exp_config=config,
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

### 4. PPL評価方法によるPPL異常値（重要）

**セグメント分割評価で異常に高いPPL（10,000+）が出る原因と対策。**

#### 問題の発見経緯

Pythia-70mの公式WikiText-2 PPLは約32-35だが、セグメント分割評価では14,000+になった。

#### 原因: コンテキストなしでの予測

```python
# ❌ 問題のある評価方法（セグメント分割、非重複）
for start in range(0, seq_len, segment_length):
    segment = tokens[start:end]
    input_ids = segment[:-1].unsqueeze(0)  # 各セグメントが独立
    labels = segment[1:].unsqueeze(0)
    outputs = model(input_ids, labels=labels)
    # → 各セグメントの最初のトークンを予測する際、コンテキストがない
    # → 「文書の途中」を「文書の先頭」として扱うため高PPL
```

#### 正しい評価方法: Sliding Window

```python
# ✅ 正しい評価方法（Sliding Window）
stride = 512
context_length = 2048

for start in range(0, seq_len - 1, stride):
    end = min(start + context_length, seq_len)
    input_ids = tokens[start:end].unsqueeze(0)

    # 最初のstride個はコンテキスト（loss計算しない）
    labels = input_ids.clone()
    labels[0, :stride] = -100  # -100はloss計算から除外

    outputs = model(input_ids, labels=labels)
```

#### PPL比較（WikiText-2での実測値）

| 評価方法 | PPL | 解釈 |
|----------|-----|------|
| Sliding window (stride=512) | **40.96** | ✓ 正常（公式値に近い） |
| Simple non-overlapping (2048) | 15,885 | ❌ コンテキストなし問題 |
| Segment-based (256 tokens) | 14,204 | ❌ コンテキストなし問題 |

#### 教訓

1. **PPL評価は必ずSliding window方式を使用**
2. **異常に高いPPL（1000+）が出たら評価方法を疑う**
3. **訓練時と評価時で異なる方法を使うと比較が不正確になる**
4. **Pythia-70mのWikiText-2 PPLは約32-35が正常**

---

## 🔧 Pretrained LLMへのInfini-Attention導入

### アプローチ比較

| 方式 | 説明 | 結果 |
|------|------|------|
| **Case C: Layer 0置き換え** | Layer 0をInfini Layerに完全置き換え | ❌ RoPE損失、後続レイヤー不適合 |
| **Parallel Adapter（推奨）** | Layer 0に並列でInfini Adapterを追加 | ✓ 既存性能維持しながらメモリ追加 |

### Parallel Adapter アーキテクチャ

```
Input Embedding
      ↓
┌─────┴─────┐
│  Original │  Infini Adapter
│  Layer 0  │  (Memory)
└─────┬─────┘      │
      ↓            ↓
    Output + α × Infini_Output
      ↓
Layer 1-5 (unchanged)
```

### 使用方法

```python
from src.models.infini_adapter import create_pythia_with_parallel_infini

# モデル作成
model = create_pythia_with_parallel_infini(
    model_name="EleutherAI/pythia-70m",
    use_delta_rule=True,
    use_alibi=False,
    initial_alpha=0.0,  # 0から学習開始
    freeze_base_model=True,  # ベースモデルはfreeze
)

# 訓練後
print(f"Learned alpha: {model.get_alpha()}")  # 学習されたalpha値

# メモリ操作
model.reset_memory()
state = model.get_memory_state()
model.set_memory_state(state)
```

### 訓練スクリプト

```bash
# Parallel Adapter訓練
python3 scripts/train_parallel_adapter.py --num-docs 100 --epochs 50

# WikiText-2での評価
python3 scripts/evaluate_wikitext.py --parallel-adapter parallel_adapter.pt
```

### 発見: alphaが負になる

実験で`alpha = -0.1561`に学習された。これは元のLayer 0出力から**減算**していることを意味し、「ノイズ除去」的な役割を学習した可能性がある。

---

## 📁 File Structure

```
new-llm/
├── config/
│   └── pythia.py                   # PythiaConfig
├── scripts/
│   └── experiment.py               # 統一実験スクリプト
├── src/
│   ├── data/
│   │   └── reversal_pairs.py       # Reversal Curse評価データ
│   ├── models/
│   │   ├── __init__.py             # create_model() ファクトリ
│   │   ├── pythia.py               # PythiaModel (RoPE)
│   │   ├── infini_attention.py     # InfiniAttention, InfiniAttentionLayer
│   │   ├── infini_pythia.py        # InfiniPythiaModel (1層Infini + RoPE)
│   │   ├── multi_memory_attention.py  # MultiMemoryInfiniAttention
│   │   ├── multi_memory_pythia.py  # MultiMemoryInfiniPythiaModel
│   │   ├── hierarchical_memory.py  # HierarchicalMemoryAttention
│   │   └── hierarchical_pythia.py  # HierarchicalMemoryPythiaModel
│   └── utils/
│       ├── experiment_runner.py    # 統一実験ランナー
│       ├── training.py             # 共通学習ユーティリティ
│       ├── evaluation.py           # 評価関数
│       ├── device.py               # デバイス管理
│       ├── data_pythia.py          # Pileデータ読み込み
│       └── seed.py                 # シード設定
├── docs/
│   └── experiments/                # 実験結果
├── CLAUDE.md
└── README.md
```

---

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-06 | **PPL評価方法の教訓追加**: Sliding window方式が正しい、セグメント分割は高PPLになる |
| 2025-12-06 | **Parallel Adapter実装**: Pretrained LLMにInfini-Attentionを並列挿入する方式 |
| 2025-12-06 | **WikiText-2評価スクリプト追加**: 標準ベンチマークでの正確なPPL評価 |
| 2025-12-06 | **メモリ転送API追加**: get_memory_state/set_memory_stateで圧縮メモリを別PCに転送可能 |
| 2025-12-06 | **モデルファクトリ追加**: create_model()でシンプルにモデル作成 |
| 2025-12-06 | **実験スクリプト統一**: experiment.pyに統合、experiment_runner.py追加 |
| 2025-12-06 | **Hierarchical Memory追加**: 学習可能な展開判断、Coarse-to-Fine検索 |
| 2025-12-06 | **Multi-Memory Attention追加**: Attention-based選択で複数メモリを動的混合 |
| 2025-12-06 | **ALiBi位置エンコーディング追加**: 線形化近似でALiBiをメモリに組み込み |
| 2025-12-05 | **Memory-Onlyに集中**: Local Attention削除、コード簡素化 |
| 2025-12-05 | **Multi-Memory Bank追加**: 複数バンクで情報混合低減 |
| 2025-12-05 | **Long Context評価バグ修正**: 訓練済み重みをロード |
| 2025-12-05 | **Infini-Pythia実装**: 1層目Infini + RoPE |

---

Last Updated: 2025-12-06
