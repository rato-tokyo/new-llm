"""Senri Models

Senri: Japanese LLM with Compressive Memory
Unified memory architecture with landmark-based selection.

## Quick Start

```python
from src.models import TransformerLM, senri_layers, pythia_layers

# Senriモデル（1 Senri + 5 Pythia）
model = TransformerLM(
    layers=senri_layers(1) + pythia_layers(5),
    vocab_size=52000,
)

# Pythiaのみ（ベースライン）
model = TransformerLM(
    layers=pythia_layers(6),
    vocab_size=52000,
)

# 複数メモリ構成
model = TransformerLM(
    layers=senri_layers(1, num_memories=4) + pythia_layers(5),
    vocab_size=52000,
)

# カスタム構成
from src.models import SenriLayer, PythiaLayer

model = TransformerLM(
    layers=[
        SenriLayer(hidden_size=512, num_heads=8, intermediate_size=2048, num_memories=4),
        PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
        PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    ],
    vocab_size=52000,
)
```
"""

from typing import Optional

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

# =============================================================================
# Layer Factory Functions
# =============================================================================

# Default layer parameters
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_INTERMEDIATE_SIZE = 2048


def senri_layers(
    n: int = 1,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_memories: int = 1,
    memory_head_dim: Optional[int] = None,
    use_delta_rule: bool = True,
) -> list[BaseLayer]:
    """Create n SenriLayer instances.

    Args:
        n: Number of layers
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension
        num_memories: Number of memory slots (1 = original Infini-Attention)
        memory_head_dim: Memory head dimension (None = hidden_size for single-head)
        use_delta_rule: Use delta rule for memory update

    Returns:
        List of SenriLayer instances

    Example:
        # Single memory (Infini-Attention equivalent)
        layers = senri_layers(1) + pythia_layers(5)

        # Multiple memories
        layers = senri_layers(1, num_memories=4) + pythia_layers(5)

        # Custom memory head dimension
        layers = senri_layers(1, memory_head_dim=256) + pythia_layers(5)
    """
    return [
        SenriLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_memories=num_memories,
            memory_head_dim=memory_head_dim,
            use_delta_rule=use_delta_rule,
        )
        for _ in range(n)
    ]


def pythia_layers(
    n: int = 6,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    rotary_pct: float = 0.25,
    max_position_embeddings: int = 2048,
) -> list[BaseLayer]:
    """Create n PythiaLayer instances.

    Args:
        n: Number of layers
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension
        rotary_pct: Rotary embedding percentage
        max_position_embeddings: Maximum sequence length

    Returns:
        List of PythiaLayer instances

    Example:
        # Pythia-only model
        model = TransformerLM(layers=pythia_layers(6), vocab_size=52000)

        # Mixed model
        layers = senri_layers(1) + pythia_layers(5)
        model = TransformerLM(layers=layers, vocab_size=52000)
    """
    return [
        PythiaLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
        )
        for _ in range(n)
    ]


__all__ = [
    # Layer factory functions
    "senri_layers",
    "pythia_layers",
    # Constants
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_INTERMEDIATE_SIZE",
    # Core models
    "TransformerLM",
    # Layer types
    "BaseLayer",
    "PythiaLayer",
    "PythiaAttention",
    "SenriLayer",
    "SenriAttention",
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
