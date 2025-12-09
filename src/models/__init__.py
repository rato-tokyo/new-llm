"""Senri Models

Senri: Japanese LLM with Compressive Memory
Infini-Attention for efficient long-context processing.

## Quick Start

```python
from src.models import TransformerLM, infini_layers, pythia_layers

# Senriモデル（1 Infini + 5 Pythia）
model = TransformerLM(
    layers=infini_layers(1) + pythia_layers(5),
    vocab_size=52000,
)

# Pythiaのみ（ベースライン）
model = TransformerLM(
    layers=pythia_layers(6),
    vocab_size=52000,
)

# カスタム構成
from src.models import InfiniLayer, PythiaLayer, MultiMemoryLayer

model = TransformerLM(
    layers=[
        MultiMemoryLayer(hidden_size=512, num_heads=8, intermediate_size=2048, num_memories=4),
        PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
        PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    ],
    vocab_size=52000,
)
```
"""

# Core models
from .model import TransformerLM

# Layer types
from .layers import (
    BaseLayer,
    PythiaLayer,
    PythiaAttention,
    InfiniLayer,
    InfiniAttention,
    MultiMemoryLayer,
    MultiMemoryAttention,
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
    MemoryStateMixin,
    create_memory_matrix,
    create_memory_norm,
    retrieve_from_memory,
    update_memory_delta_rule,
    update_memory_simple,
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


def infini_layers(
    n: int = 1,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_memory_banks: int = 1,
    segments_per_bank: int = 4,
    use_delta_rule: bool = True,
) -> list[BaseLayer]:
    """Create n InfiniLayer instances.

    Args:
        n: Number of layers
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension
        num_memory_banks: Number of memory banks
        segments_per_bank: Segments per bank
        use_delta_rule: Use delta rule for memory update

    Returns:
        List of InfiniLayer instances

    Example:
        layers = infini_layers(2) + pythia_layers(4)
        model = TransformerLM(layers=layers, vocab_size=52000)
    """
    return [
        InfiniLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_memory_banks=num_memory_banks,
            segments_per_bank=segments_per_bank,
            use_delta_rule=use_delta_rule,
        )
        for _ in range(n)
    ]


def multi_memory_layers(
    n: int = 1,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_memories: int = 4,
    use_delta_rule: bool = True,
) -> list[BaseLayer]:
    """Create n MultiMemoryLayer instances.

    Args:
        n: Number of layers
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension
        num_memories: Number of memory slots
        use_delta_rule: Use delta rule for memory update

    Returns:
        List of MultiMemoryLayer instances

    Example:
        layers = multi_memory_layers(1, num_memories=8) + pythia_layers(5)
        model = TransformerLM(layers=layers, vocab_size=52000)
    """
    return [
        MultiMemoryLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_memories=num_memories,
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
        layers = infini_layers(1) + pythia_layers(5)
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


def senri_layers(
    n_infini: int = 1,
    n_pythia: int = 5,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
) -> list[BaseLayer]:
    """Create default Senri layer configuration (Infini + Pythia).

    Args:
        n_infini: Number of InfiniLayers
        n_pythia: Number of PythiaLayers
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension

    Returns:
        List of layers (InfiniLayers followed by PythiaLayers)

    Example:
        # Default: 1 Infini + 5 Pythia
        model = TransformerLM(layers=senri_layers(), vocab_size=52000)

        # Custom: 2 Infini + 4 Pythia
        model = TransformerLM(layers=senri_layers(2, 4), vocab_size=52000)
    """
    return infini_layers(
        n_infini,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
    ) + pythia_layers(
        n_pythia,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
    )


__all__ = [
    # Layer factory functions
    "infini_layers",
    "multi_memory_layers",
    "pythia_layers",
    "senri_layers",
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
    "InfiniLayer",
    "InfiniAttention",
    "MultiMemoryLayer",
    "MultiMemoryAttention",
    # Building blocks
    "PythiaMLP",
    "init_weights",
    "count_parameters",
    # Memory utilities
    "elu_plus_one",
    "causal_linear_attention",
    "MemoryStateMixin",
    "create_memory_matrix",
    "create_memory_norm",
    "retrieve_from_memory",
    "update_memory_delta_rule",
    "update_memory_simple",
    # Position encoding
    "RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
]
