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

### シングルヘッドメモリ（重要）

Linear Attentionはhead_dimが小さいと機能しない。
シングルヘッド（memory_head_dim=hidden_size=512）で最大の表現力を確保。

```python
# メモリ行列: [memory_head_dim, memory_head_dim] = [512, 512]
# Q, K, V: Linear(hidden_size → memory_head_dim)
# Output: Linear(memory_head_dim → hidden_size)
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
model = create_model("multi_memory", num_memories=8)
model = create_model("hierarchical", num_memories=4, use_delta_rule=False)

# カスタムconfig
from config.pythia import PythiaConfig
config = PythiaConfig()
model = create_model("infini", config)
```

### 利用可能なオプション

| オプション | 対象モデル | デフォルト | 説明 |
|------------|------------|------------|------|
| `use_delta_rule` | 全memory系 | `True` | Delta Rule使用 |
| `num_memories` | multi_memory, hierarchical | `4` | メモリ数 |
| `num_memory_banks` | infini | `1` | メモリバンク数 |
| `segments_per_bank` | infini | `4` | バンクあたりセグメント数 |

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

### 5. Linear Attentionのhead_dim次元数問題（重要）

**head_dimが小さいとLinear Attentionが機能しない。**

#### 問題

```python
# ❌ 問題: head_dim=64（hidden_size=512 / num_heads=8）
head_dim = hidden_size // num_heads  # = 64

# メモリ行列: [num_heads, head_dim, head_dim] = [8, 64, 64]
# → 64次元空間では異なるキーベクトルが直交しやすい
# → σ(Q) @ σ(K)^T ≈ 0 → メモリから何も取り出せない
```

#### 原因: キーベクトルの直交性

Linear Attentionでは `σ(K)` を使ってメモリに書き込み、`σ(Q)` で取り出す：

```
メモリ更新: M = M + σ(K)^T @ V
メモリ取得: output = σ(Q) @ M / (σ(Q) @ z)
```

**64次元空間だと**：
- 異なるトークンのKベクトルが高確率で直交
- `σ(Q) @ σ(K)^T` がほぼゼロ
- メモリから有用な情報が取り出せない
- モデルがメモリを無視する方向に学習 → alphaが小さいまま

#### 解決策: シングルヘッド（memory_head_dim=hidden_size）

```python
# ✅ 修正: memory_head_dim=hidden_size（シングルヘッド）
memory_head_dim = hidden_size  # = 512

# メモリ行列: [memory_head_dim, memory_head_dim] = [512, 512]
# → 512次元空間で十分な表現力を確保
# → キーベクトル間の類似度が計算可能
```

#### 教訓

1. **Linear Attentionにはhead_dim >= 256が必要**
2. **可能ならシングルヘッド（head_dim=hidden_size）を使用**
3. **alphaが小さいまま → head_dimを疑う**
4. **マルチヘッドは表現力を犠牲にする**

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

## 🔧 Pretrained LLMへのInfini-Attention導入（失敗）

**⚠️ Layer置き換え方式は全て失敗。詳細は `docs/experiments/2025-12-06_distill_finetune_failure.md` を参照。**

### 試行したアプローチと結果

| 方式 | 説明 | 結果 |
|------|------|------|
| Layer 0置き換え | Layer 0をInfini Layerに完全置き換え | ❌ RoPE損失、PPL大幅劣化 |
| 2段階訓練（蒸留+Fine-tune） | 蒸留後にFull Fine-tune | ❌ PPL 44→1237（28倍劣化） |
| Parallel Adapter | 並列でInfini追加 | ❌ alphaが学習されない |

### 失敗の根本原因

1. **RoPE情報の喪失**: Linear Attention (NoPE)は位置情報を持たない
2. **後続レイヤーとの不整合**: Layer 1-5はRoPE付き入力を前提に訓練されている
3. **蒸留の限界**: 出力を模倣しても内部表現は複製できない

### 既存研究の知見

- **Google Infini-Attention**: **全レイヤー**を置換 + 30K steps継続事前訓練で成功
- **Hugging Face実験**: Llama 3 8Bでの単一レイヤー置換は失敗
- **結論**: 単一レイヤー置換は技術的に困難。全レイヤー置換かスクラッチ訓練が必要

### 推奨アプローチ

| アプローチ | 説明 |
|------------|------|
| **スクラッチ訓練** | 最初からInfiniアーキテクチャで訓練（本プロジェクトのInfiniPythia） |
| **RoPE Scaling** | YaRN等の既存手法で長文対応 |
| **全レイヤー置換** | 大規模継続事前訓練が必要（計算コスト大） |

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
| 2025-12-06 | **Layer置き換え方式を削除**: 蒸留+Fine-tune等すべて失敗、スクラッチ訓練に集中 |
| 2025-12-06 | **シングルヘッドメモリ導入**: memory_head_dim=512でLinear Attentionの表現力を最大化、ALiBi版削除 |
| 2025-12-06 | **PPL評価方法の教訓追加**: Sliding window方式が正しい、セグメント分割は高PPLになる |
| 2025-12-06 | **WikiText-2評価スクリプト追加**: 標準ベンチマークでの正確なPPL評価 |
| 2025-12-06 | **メモリ転送API追加**: get_memory_state/set_memory_stateで圧縮メモリを別PCに転送可能 |
| 2025-12-06 | **モデルファクトリ追加**: create_model()でシンプルにモデル作成 |
| 2025-12-06 | **実験スクリプト統一**: experiment.pyに統合、experiment_runner.py追加 |
| 2025-12-06 | **Hierarchical Memory追加**: 学習可能な展開判断、Coarse-to-Fine検索 |
| 2025-12-06 | **Multi-Memory Attention追加**: Attention-based選択で複数メモリを動的混合 |
| 2025-12-05 | **Memory-Onlyに集中**: Local Attention削除、コード簡素化 |
| 2025-12-05 | **Multi-Memory Bank追加**: 複数バンクで情報混合低減 |
| 2025-12-05 | **Long Context評価バグ修正**: 訓練済み重みをロード |
| 2025-12-05 | **Infini-Pythia実装**: 1層目Infini + RoPE |

---

Last Updated: 2025-12-06
