"""New-LLM Models

Pythia-70M based experimental architectures.
Infini-Attention for compressive memory.

## Quick Start

```python
from src.models import create_model

# 標準Pythia
model = create_model("pythia")

# Infini-Pythia (1層目Infini + 5層Pythia)
model = create_model("infini")

# Multi-Memory (4メモリ)
model = create_model("multi_memory", num_memories=8)

# Hierarchical Memory
model = create_model("hierarchical", num_memories=4)
```

## Layer-based Construction

```python
from src.models import TransformerLM
from src.models.layers import InfiniLayer, PythiaLayer

# カスタムアーキテクチャ
layers = [
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    *[PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048) for _ in range(4)]
]
model = TransformerLM(layers=layers)
```
"""

from typing import Literal, Optional

from config.pythia import PythiaConfig

# Core models
from .model import TransformerLM
from .continuous import ContinuousLM
from .selective import SelectiveOutputLM

# Layer types
from .layers import (
    BaseLayer,
    PythiaLayer,
    PythiaAttention,
    InfiniLayer,
    InfiniAttention,
    MultiMemoryLayer,
    MultiMemoryAttention,
    HierarchicalLayer,
    HierarchicalAttention,
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

# Type alias for model types
ModelTypeLiteral = Literal["pythia", "infini", "multi_memory", "hierarchical", "continuous", "selective"]


def create_model(
    model_type: ModelTypeLiteral,
    config: Optional[PythiaConfig] = None,
    *,
    # Memory settings
    use_delta_rule: bool = True,
    num_memories: int = 4,
    # Infini-specific settings
    num_memory_banks: int = 1,
    segments_per_bank: int = 4,
):
    """
    Create a model by type.

    Args:
        model_type: Model type ("pythia", "infini", "multi_memory", "hierarchical", "continuous", "selective")
        config: PythiaConfig (uses default if None)
        use_delta_rule: Use delta rule for memory update (memory models only)
        num_memories: Number of memories (multi_memory, hierarchical only)
        num_memory_banks: Number of memory banks (infini only)
        segments_per_bank: Segments per bank (infini only)

    Returns:
        TransformerLM, ContinuousLM, or SelectiveOutputLM instance

    Examples:
        # Standard Pythia
        model = create_model("pythia")

        # Continuous representation model
        model = create_model("continuous")

        # Selective output model (2-pass processing)
        model = create_model("selective")

        # Infini-Pythia
        model = create_model("infini")
    """
    if config is None:
        config = PythiaConfig()

    h = config.hidden_size
    n = config.num_attention_heads
    i = config.intermediate_size
    r = config.rotary_pct
    m = config.max_position_embeddings

    def pythia_layers(count: int) -> list[BaseLayer]:
        return [PythiaLayer(h, n, i, r, m) for _ in range(count)]

    if model_type == "pythia":
        layers = pythia_layers(config.num_layers)
        return TransformerLM(
            layers=layers,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

    elif model_type == "continuous":
        layers = pythia_layers(config.num_layers)
        return ContinuousLM(
            layers=layers,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

    elif model_type == "selective":
        layers = pythia_layers(config.num_layers)
        return SelectiveOutputLM(
            layers=layers,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

    elif model_type == "infini":
        layers = [
            InfiniLayer(h, n, i, num_memory_banks, segments_per_bank, use_delta_rule),
            *pythia_layers(config.num_layers - 1),
        ]

    elif model_type == "multi_memory":
        layers = [
            MultiMemoryLayer(h, n, i, num_memories, use_delta_rule),
            *pythia_layers(config.num_layers - 1),
        ]

    elif model_type == "hierarchical":
        layers = [
            HierarchicalLayer(h, n, i, num_memories, use_delta_rule),
            *pythia_layers(config.num_layers - 1),
        ]

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: pythia, infini, multi_memory, hierarchical, continuous, selective"
        )

    return TransformerLM(
        layers=layers,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
    )


__all__ = [
    # Factory function
    'create_model',
    'ModelTypeLiteral',

    # Core models
    'TransformerLM',
    'ContinuousLM',
    'SelectiveOutputLM',

    # Layer types
    'BaseLayer',
    'PythiaLayer',
    'PythiaAttention',
    'InfiniLayer',
    'InfiniAttention',
    'MultiMemoryLayer',
    'MultiMemoryAttention',
    'HierarchicalLayer',
    'HierarchicalAttention',

    # Building blocks
    'PythiaMLP',
    'init_weights',
    'count_parameters',

    # Memory utilities
    'elu_plus_one',
    'causal_linear_attention',
    'MemoryStateMixin',
    'create_memory_matrix',
    'create_memory_norm',
    'retrieve_from_memory',
    'update_memory_delta_rule',
    'update_memory_simple',

    # Position encoding
    'RotaryEmbedding',
    'rotate_half',
    'apply_rotary_pos_emb',
]
