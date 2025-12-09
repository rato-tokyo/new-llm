"""Senri Models

Senri: Japanese LLM with Compressive Memory

## Quick Start

```python
from src.models import SenriModel, SenriLayer, PythiaLayer

# Senri: 1 Senri + 5 Pythia
model = SenriModel([
    SenriLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
])

# Pythia ベースライン
model = SenriModel([PythiaLayer() for _ in range(6)])

# 複数メモリ構成
model = SenriModel([
    SenriLayer(num_memories=4),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
    PythiaLayer(),
])

# プリセットを使用
from src.config import SENRI_MODEL, PYTHIA_MODEL

model = SENRI_MODEL()
model = PYTHIA_MODEL()
```
"""

# Core models
from .model import SenriModel, TransformerLM  # TransformerLM is alias for SenriModel

# Layer types
from .layers import (
    BaseLayer,
    PythiaLayer,
    PythiaAttention,
    SenriLayer,
    SenriAttention,
)

# Building blocks
from .base_components import (
    PythiaMLP,
    init_weights,
    count_parameters,
)
from .memory_utils import (
    elu_plus_one,
    causal_linear_attention,
)
from .position_encoding import (
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)

__all__ = [
    # Core models
    "SenriModel",
    "TransformerLM",  # Alias for SenriModel
    # Layer types
    "BaseLayer",
    "SenriLayer",
    "SenriAttention",
    "PythiaLayer",
    "PythiaAttention",
    # Building blocks
    "PythiaMLP",
    "init_weights",
    "count_parameters",
    # Memory utilities
    "elu_plus_one",
    "causal_linear_attention",
    # Position encoding
    "RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
]
