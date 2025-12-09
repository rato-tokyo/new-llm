# Senri: Japanese LLM with Compressive Memory

Senri is a Japanese LLM with unified compressive memory for efficient long-context processing.

## Overview

Senri uses OpenCALM tokenizer with SenriLayer (unified compressive memory), enabling infinite context processing through learned memory. The architecture uses a simple layer-based design with just 2 layer types.

```
Architecture:
  Token Embedding (512-dim, vocab=52,000)
         |
  Layer 0: SenriLayer (NoPE, Memory Only)
    - Compressive Memory (linear attention)
    - Configurable: num_memories, memory_head_dim
         |
  Layer 1-5: PythiaLayer (RoPE)
    - Multi-Head Attention
    - MLP
         |
  Output Head (512 -> vocab)
```

## Key Features

- **Japanese LLM**: OpenCALM tokenizer (UNK-free, byte_fallback support)
- **Simple Design**: Only 2 layer types (PythiaLayer, SenriLayer)
- **Compressive Memory**: O(1) memory for infinite context via linear attention
- **Flexible Configuration**: num_memories, memory_head_dim parameters
- **Delta Rule**: Efficient memory update preventing information overwrite
- **Memory Transfer**: Save/load memory state across devices

## Quick Start

### Installation

```bash
pip install torch transformers datasets tqdm
```

### Create Models (Direct Layer Pattern)

```python
from src.models import SenriModel, SenriLayer, PythiaLayer

# Senri model (1 SenriLayer + 5 PythiaLayers) - 推奨
model = SenriModel([
    SenriLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
])

# Pythia-only baseline
model = SenriModel([PythiaLayer() for _ in range(6)])

# Multiple memories
model = SenriModel([
    SenriLayer(num_memories=4),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
])

# Or use presets
from src.config import SENRI_MODEL, PYTHIA_MODEL
model = SENRI_MODEL()
model = PYTHIA_MODEL()

# Forward pass with memory update
output = model(input_ids, update_memory=True)

# Reset memory
model.reset_memory()
```

### Training & Evaluation (Quick Start)

```bash
# Senriモデルの訓練 + 評価（日本語Wikipedia使用）
python3 scripts/quick_model.py --model senri --train --epochs 3

# Pythiaベースライン
python3 scripts/quick_model.py --model pythia --train --epochs 3

# トークン数・エポック数を指定
python3 scripts/quick_model.py --model senri --train --train-tokens 500000 --epochs 5
```

### Fine-tuning with Custom Knowledge

カスタム知識データでのファインチューニングは `senri-fine-tuner/` を参照してください。

```bash
cd senri-fine-tuner
python3 finetune.py --data data/example_knowledge.json --epochs 10
```

詳細: [senri-fine-tuner/README.md](senri-fine-tuner/README.md)

### Memory Transfer

```python
import torch
from src.models import SenriModel, SenriLayer, PythiaLayer

# On PC A: Save memory state
model = SenriModel([
    SenriLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
])
# ... training or processing ...
state = model.get_memory_state()
torch.save(state, "memory.pt")

# On PC B: Load memory state
state = torch.load("memory.pt")
model = SenriModel([
    SenriLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
])
model.set_memory_state(state)
# Memory is now restored!
```

## Dataset

### 推奨データセット

| 用途 | データセット | サイズ | 説明 |
|------|-------------|--------|------|
| **開発・実験** | `wikipedia` (ja) | ~3GB | 高品質、扱いやすい |
| **PPL評価** | `wikipedia` (ja) | ~3GB | 安定した評価 |
| **本格訓練** | `mc4` (ja) | ~330GB | 大規模・多様 |

### 使用方法

```python
from src.utils.data_utils import load_wiki_ja_tokens_cached
from src.config import OPEN_CALM_TOKENIZER

# 日本語Wikipediaからトークンをロード（キャッシュ付き）
tokens = load_wiki_ja_tokens_cached(
    num_tokens=100000,
    tokenizer_name=OPEN_CALM_TOKENIZER,
)
```

### データセット選択の理由

- **OpenCALMトークナイザー** = 日本語特化
- **Pile（英語）** = トークン効率が悪い → **非推奨**
- **日本語Wikipedia** = 高品質・日本語対応 → **推奨**

## Tokenizer

Using **CyberAgent OpenCALM** tokenizer for Japanese:

| Feature | Value |
|---------|-------|
| Vocab Size | 52,000 |
| UNK Token | None (byte_fallback) |
| Japanese | Full support |
| English | Full support (AI, API, GPU, etc.) |
| Emoji | Full support |

```python
from src.utils.tokenizer_utils import get_open_calm_tokenizer

tokenizer = get_open_calm_tokenizer()
# No UNK tokens for any input!
```

## Layer Types

| Layer | Description |
|-------|-------------|
| `PythiaLayer` | Standard Pythia (RoPE + Softmax Attention) |
| `SenriLayer` | Unified memory layer (Linear Attention, NoPE). Configurable: num_memories, memory_head_dim |

### SenriLayer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 512 | Hidden dimension |
| `num_heads` | 8 | Number of attention heads |
| `intermediate_size` | 2048 | MLP intermediate dimension |
| `num_memories` | 1 | Number of memory slots |
| `memory_head_dim` | None (=hidden_size) | Memory head dimension. None = single-head (512) |
| `use_delta_rule` | True | Use delta rule for memory update |

### PythiaLayer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 512 | Hidden dimension |
| `num_heads` | 8 | Number of attention heads |
| `intermediate_size` | 2048 | MLP intermediate dimension |
| `rotary_pct` | 0.25 | RoPE application ratio |

## Architecture Details

### Default Configuration

| Component | Value |
|-----------|-------|
| Hidden Size | 512 |
| Layers | 6 |
| Attention Heads | 8 |
| Intermediate Size | 2048 |
| Position Encoding | Rotary (RoPE, 25%) |
| Vocab Size | 52,000 |
| Tokenizer | cyberagent/open-calm-small |

### SenriAttention (Compressive Memory)

```
Memory Update (Delta Rule):
  M_s = M_{s-1} + sigma(K)^T @ (V - retrieved_V)

Memory Retrieval:
  A_mem = sigma(Q) @ M / (sigma(Q) @ z)

sigma(x) = ELU(x) + 1
```

### Memory Sizes

| Configuration | Memory Size |
|---------------|-------------|
| SenriLayer (num_memories=1) | ~135 KB |
| SenriLayer (num_memories=4) | ~540 KB |

## Project Structure

```
senri/
├── src/
│   ├── config/
│   │   ├── __init__.py           # Constants + Model presets
│   │   ├── constants.py          # OPEN_CALM_TOKENIZER, OPEN_CALM_VOCAB_SIZE
│   │   ├── models.py             # Model presets (SENRI_MODEL, etc.)
│   │   └── experiments/
│   │       └── base.py           # ExperimentConfig
│   ├── models/
│   │   ├── __init__.py           # Layer class exports
│   │   ├── layers/
│   │   │   ├── base.py           # BaseLayer
│   │   │   ├── pythia.py         # PythiaLayer
│   │   │   └── senri.py          # SenriLayer (unified memory layer)
│   │   ├── model.py              # SenriModel
│   │   ├── base_components.py    # PythiaMLP, init_weights
│   │   ├── memory_utils.py       # Linear attention utilities
│   │   └── position_encoding.py  # RoPE
│   └── utils/
│       ├── data_utils.py         # Japanese Wikipedia data loading
│       ├── tokenizer_utils.py    # OpenCALM tokenizer
│       ├── training.py           # Training utilities
│       └── evaluation.py         # Evaluation utilities
├── scripts/
│   ├── quick_model.py            # Quick training & evaluation
│   └── experiment_context_reasoning.py  # Reversal Curse experiment
├── senri-fine-tuner/             # Fine-tuning toolkit (separate)
│   ├── finetune.py               # Fine-tuning script
│   ├── data/                     # Training data
│   └── README.md
├── tests/
├── docs/
│   └── experiments/              # Experiment results
├── CLAUDE.md                     # Development guidelines
└── README.md
```

## Google Colab

```python
# セットアップ
!git clone https://github.com/rato-tokyo/new-llm.git
%cd new-llm
!pip install torch transformers datasets tqdm

# Senriモデルの訓練 + 評価
!python3 scripts/quick_model.py --model senri --train --epochs 3

# Pythiaベースライン
!python3 scripts/quick_model.py --model pythia --train --epochs 3

# より多くのトークンで訓練
!python3 scripts/quick_model.py --model senri --train --train-tokens 500000 --epochs 5
```

## Evaluation

### Reversal Curse

Measures if a model trained on "A is B" can also infer "B is A".

| Metric | Description |
|--------|-------------|
| Forward PPL | PPL on training direction |
| Backward PPL | PPL on reverse direction |
| Reversal Gap | Backward - Forward (lower is better) |

## Development

See `CLAUDE.md` for development guidelines including:
- Direct layer pattern (mandatory)
- SenriLayer configuration (num_memories, memory_head_dim)
- Memory transfer API
- Code quality rules
- Past bugs and lessons

## License

MIT
