# New-LLM Project Guidelines

---

## 🎯 MLA-Pythia Architecture (2025-12-05)

**Pythia-70MをベースにMLA（Multi-head Latent Attention）でKVキャッシュを大幅削減。**
**位置エンコーディングはALiBi（統一スロープ）を採用。**

### アーキテクチャ

```
MLA-Pythia (ALiBi):
Token Embedding (512-dim)
       ↓
MLALayer × 6
  ├─ MLA Attention (ALiBi)
  │    c_kv: 128-dim (KV共通圧縮)
  │    吸収モード
  └─ MLP
       ↓
Output Head (512 → vocab)

KV Cache: c_kv(128) = 128
削減率: 87.5%
```

### 設定値

| 項目 | MLA-Pythia |
|------|------------|
| hidden_size | 512 |
| kv_dim | 128 |
| Layers | 6 |
| Attention Heads | 8 |
| intermediate_size | 2048 |
| Position Encoding | ALiBi (統一スロープ) |
| KV Cache削減 | 87.5% |

### 実験の実行

```bash
# MLA実験
python3 scripts/experiment_mla.py --samples 10000 --epochs 30

# kv_dim変更
python3 scripts/experiment_mla.py --kv-dim 256  # 75%削減
python3 scripts/experiment_mla.py --kv-dim 64   # 93.75%削減
```

---

## 🎯 ALiBi (Attention with Linear Biases)

### 仕様

```
score = Q @ K^T - m * distance_matrix

distance_matrix[i][j] = |i - j|  # 位置間の距離
m = slope (全ヘッド統一、デフォルト: 0.0625)
```

### 使用方法

```python
from src.models import ALiBiPositionEncoding

pos_enc = ALiBiPositionEncoding(slope=0.0625)
attn_scores = pos_enc.apply_to_scores(attn_scores, seq_len)
```

---

## 📚 DeepSeek MLA (Multi-head Latent Attention)

### 吸収モード（Absorbed Projection）

```
MLA（吸収モード - 復元不要）:
  c_kv = X @ W_DKV     # KV共通圧縮: (seq, 512) → (seq, 128)
  scores = Q @ W_UK^T @ c_kv^T

  # KVキャッシュは c_kv のみ保存（87.5%削減）
```

### 参考リンク

- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
- [MLA Explanation (HuggingFace)](https://huggingface.co/blog/NormalUhr/mla-explanation)

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

### 1. ALiBi因果マスクの行列方向バグ

```python
# ❌ バグ: relative_pos[i][j] = j - i （未来が見えていた）
relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)

# ✅ 修正: relative_pos[i][j] = i - j （正しい因果マスク）
relative_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
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
│   └── experiment_mla.py           # MLA実験
├── src/
│   ├── data/
│   │   └── reversal_pairs.py       # Reversal Curse評価データ
│   ├── models/
│   │   ├── mla_pythia.py           # MLAPythiaModel (ALiBi)
│   │   ├── mla.py                  # MLAAttention, MLALayer
│   │   ├── alibi.py                # ALiBi実装
│   │   └── position_encoding.py    # ALiBiPositionEncoding
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
| 2025-12-05 | **RoPE関連コード削除**: ALiBi一本化、シンプル化 |
| 2025-12-05 | **MLA-Pythia実装**: KVキャッシュ87.5%削減 |
| 2025-12-05 | **ALiBi採用**: 統一スロープ方式 |
| 2025-12-05 | **Reversal Curse評価追加**: 順方向/逆方向PPL比較 |

---

Last Updated: 2025-12-05
