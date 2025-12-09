"""Senri Models

Senri: Japanese LLM with Compressive Memory

## Quick Start

```python
from src.models import TransformerLM, SenriLayer, PythiaLayer

# Senri構成: 1 Senri + 5 Pythia
model = TransformerLM([
    SenriLayer(hidden_size=512, num_heads=8, intermediate_size=2048, num_memories=2, memory_head_dim=512, use_delta_rule=True),
    PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
    PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
    PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
    PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
    PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
])

# Pythia ベースライン
model = TransformerLM([PythiaLayer(...) for _ in range(6)])

# プリセットを使用
from src.config import SENRI_MODEL, PYTHIA_MODEL

model = SENRI_MODEL()
model = PYTHIA_MODEL()
```
"""

# Core models
from .model import TransformerLM

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
    "TransformerLM",
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
