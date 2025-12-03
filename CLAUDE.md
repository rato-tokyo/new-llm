# New-LLM Project Guidelines

## 🎯 Project Goal: Context-Pythia (2025-12-03)

**Pythia-70Mの全LayerをContext-based Attentionに置き換え、KVキャッシュメモリを50%削減する。**

### Target Architecture

```
Context-Pythia:
  Token Embedding (512-dim)
       ↓
  ContextBlock: 512 → 256 (圧縮)
       ↓
  Layer 0-5: 全て context (256-dim) を入力
       ↓
  Output Head (vocab_size)
```

### Key Decisions

| 項目 | 決定 |
|------|------|
| **置き換え層** | 全6Layer |
| **Context次元** | 256（50%圧縮） |
| **学習データ** | Pile（Pythiaと同じ）、開発時は限定サンプル |
| **学習方法** | Phase 1（OACD）→ Phase 2（全体学習） |
| **評価指標** | PPL + LAMBADA |
| **メモリ削減目標** | 50% |

---

## 🏆 Baseline: Pythia Scaling Suite

**我々のライバル。Pythiaモデルスイートの性能を上回ることが目標。**

### Pythia Model Suite

| Model | Params | Layers | Hidden | Heads | Training Data |
|-------|--------|--------|--------|-------|---------------|
| Pythia-70M | 70M | 6 | 512 | 8 | Pile (~300B tokens) |
| Pythia-160M | 160M | 12 | 768 | 12 | Pile (~300B tokens) |
| Pythia-410M | 410M | 24 | 1024 | 16 | Pile (~300B tokens) |
| Pythia-1B | 1B | 16 | 2048 | 8 | Pile (~300B tokens) |
| Pythia-1.4B | 1.4B | 24 | 2048 | 16 | Pile (~300B tokens) |

### Pythia-70M Specifications

- **Architecture**: GPT-NeoX (Transformer decoder)
- **Layers**: 6
- **Hidden Size**: 512
- **Attention Heads**: 8
- **Intermediate Size**: 2048
- **Position Encoding**: Rotary (RoPE, 25%)
- **Vocab Size**: 50,304
- **Training**: ~300B tokens on the Pile
- **Parallel Attention**: Yes (attention + MLP in parallel)

### Evaluation Benchmarks (from Pythia paper)

- **LAMBADA**: 長距離依存性（最終単語予測）
- **WikiText**: Perplexity
- **HellaSwag**: 常識推論
- **PIQA**: 物理的直感
- **ARC**: 推論

参考: [Pythia Paper](https://arxiv.org/abs/2304.01373), [GitHub](https://github.com/EleutherAI/pythia)

---

## 📐 Context-Pythia Architecture

### 新方式: Context次元圧縮

```
通常Pythia:
  KV Cache = hidden_size (512) × seq_len × num_layers (6)

Context-Pythia:
  KV Cache = context_dim (256) × seq_len × num_layers (6)

削減率 = 1 - (256/512) = 50%
```

### Components

**1. ContextBlock**
- 入力: prev_context (256) + token_embed (512)
- 出力: context (256)
- Phase 1でOACD学習、Phase 2でfreeze

**2. ContextPythiaLayer**
- 入力: context (256-dim)
- query_key_value: Linear(256 → 1536)
- 出力: hidden_states (512-dim)
- 6 Layers、全て同じ構造

**3. Output Head**
- Linear(512 → vocab_size)

---

## 🚨 CRITICAL: Phase 1学習は必須

**Phase 1（OACD）学習はContext-Pythiaの核心であり、絶対にスキップしてはならない。**

### 学習フロー（必須）

```
Phase 1: OACD (ContextBlock多様性学習)
  ├─ ContextBlockのみを学習
  ├─ OACD損失で多様なcontext vectorを生成
  └─ 収束まで実行（~60 iterations）
       ↓
Phase 2: Full Training (ContextBlock frozen)
  ├─ ContextBlockをfreeze
  ├─ Layers + Output Headを学習
  └─ Cross-entropy損失
```

### なぜPhase 1が必須か

1. **多様性確保**: Phase 1なしではcontext vectorが縮退し、情報が失われる
2. **学習安定性**: 多様なcontextがないとPhase 2の学習が不安定になる
3. **性能**: Phase 1を経ることで、圧縮後も表現力を維持できる

### Phase 1の実装（削除禁止）

```python
# src/losses/diversity.py - 削除禁止
def oacd_loss(contexts, centroid_weight=0.1):
    # Term 1: 重心からの分散を最大化
    dispersion_loss = -||X - mean(X)|| / n

    # Term 2: 重心を原点に引き寄せる
    centroid_loss = ||mean(X)||²

    return dispersion_loss + centroid_weight * centroid_loss
```

### 実験スクリプトの必須構造

```python
# scripts/experiment_pythia_comparison.py
# Phase 1は必ず実行すること

# Phase 1: OACD
phase1_loss = train_phase1_oacd(model, train_loader, device, config)

# Phase 2: Full Training (ContextBlock frozen)
model.freeze_context_block()
# ... CE loss training
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

### 実験の実行

```bash
python3 scripts/experiment_pythia_comparison.py --samples 10000 --seq-length 256 --epochs 10
```

---

## 🚨 CRITICAL: コード品質

### 後方互換性コード禁止

**古い機能を残すことは厳禁。後方互換性を意識したコードは絶対に書かない。**

### ハードコード厳禁

**全ての値はconfigから読み込む。**

### デフォルト値禁止（重要パラメータ）

**サンプル数、シーケンス長、エポック数は必須引数とする。デフォルト値は予期せぬ問題を引き起こすため禁止。**

### ランダムデータ使用禁止（厳禁）

**実験でランダムデータ（torch.randint等）を使用することは絶対に禁止。**

ランダムデータでは：
- 言語パターンがないため学習不可能
- PPLが理論値（log(vocab_size) ≈ 10.8）で収束し、改善しない
- 実験として無意味

```python
# ❌ 禁止: ランダムデータ
input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))

# ✅ 必須: 実データ（Pile）を使用
inputs, targets = load_pile_data(num_samples, seq_length, config, device)
```

```python
# ❌ 禁止: デフォルト値あり
parser.add_argument('--samples', type=int, default=200)

# ✅ 必須: required=True
parser.add_argument('--samples', type=int, required=True, help='(REQUIRED)')
parser.add_argument('--seq-length', type=int, required=True, help='(REQUIRED)')
parser.add_argument('--epochs', type=int, required=True, help='(REQUIRED)')
```

---

## 📁 File Structure

```
new-llm/
├── config/
│   ├── __init__.py
│   └── pythia.py              # PythiaConfig, ContextPythiaConfig
├── scripts/
│   └── experiment_pythia_comparison.py  # 比較実験
├── src/
│   ├── models/
│   │   ├── pythia.py          # PythiaModel (baseline)
│   │   └── context_pythia.py  # ContextPythiaModel (ours)
│   ├── losses/
│   │   └── diversity.py       # OACD algorithm
│   └── utils/
├── CLAUDE.md
└── README.md
```

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
Baseline: PythiaModel (our reproduction)
Ours:     ContextPythiaModel (50% KV reduction)

Evaluate on:
- WikiText-2 PPL
- LAMBADA accuracy
- torch.cuda.max_memory_allocated()
```

---

## Related Work

- **DeepSeek MLA**: Low-rank KV compression (トークンごと)
- **本プロジェクト**: Context-based dimension reduction (全Layer)

---

Last Updated: 2025-12-03 (全Layer置き換え方式に移行)
