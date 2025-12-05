# New-LLM Project Guidelines

---

## 🎯 Infini-Pythia Architecture (2025-12-05)

**Pythia-70Mベースに1層目Infini-Attention（圧縮メモリ）を導入。**

### アーキテクチャ

```
Infini-Pythia:
Token Embedding (512-dim)
       ↓
Layer 0: InfiniAttentionLayer (NoPE, 圧縮メモリ)
  ├─ Local Attention (dot-product)
  ├─ Memory Attention (linear attention)
  └─ Beta Gate (learned)
       ↓
Layer 1-5: PythiaLayer (RoPE)
  ├─ Multi-Head Attention
  └─ MLP
       ↓
Output Head (512 → vocab)
```

### Infini-Attention

```
メモリ更新 (Delta Rule):
  M_s = M_{s-1} + σ(K)^T @ (V - retrieved_V)

メモリ取得:
  A_mem = σ(Q) @ M / (σ(Q) @ z)

結合:
  A = sigmoid(β) * A_mem + (1 - sigmoid(β)) * A_local

σ(x) = ELU(x) + 1
```

### 実験の実行

```bash
# Infini実験（両モデル比較）
python3 scripts/experiment_infini.py --samples 5000 --epochs 30

# Infiniのみ
python3 scripts/experiment_infini.py --skip-baseline

# Baselineのみ
python3 scripts/experiment_infini.py --skip-infini

# 長いシーケンス（Infiniの強み）
python3 scripts/experiment_infini.py --seq-length 512
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
│   │   └── infini_pythia.py        # InfiniPythiaModel
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
| 2025-12-05 | **MLA関連コード削除**: Infini-Attentionに集中 |
| 2025-12-05 | **Infini-Pythia実装**: 1層目Infini + RoPE |
| 2025-12-05 | **Reversal Curse評価追加**: 順方向/逆方向PPL比較 |

---

Last Updated: 2025-12-05
