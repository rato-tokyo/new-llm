"""Senri Models

Senri: Japanese LLM with Compressive Memory
Infini-Attention for efficient long-context processing.

## Quick Start

```python
from src.models import create_model
from src.config import SenriConfig

# 標準モデル（Senri設定）
model = create_model("pythia")

# Infiniモデル（デフォルト設定）
model = create_model("infini")

# Infiniモデル（カスタム設定）
config = SenriConfig(num_memory_banks=2, segments_per_bank=8)
model = create_model("infini", config)

# Multi-Memory（カスタム設定）
config = SenriConfig(num_memories=8, use_delta_rule=False)
model = create_model("multi_memory", config)
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

from typing import Optional

from src.config import SenriConfig, PythiaConfig, ModelTypeLiteral

# Core models
from .model import TransformerLM  # noqa: E402

# Layer types
from .layers import (  # noqa: E402
    BaseLayer,
    PythiaLayer,
    PythiaAttention,
    InfiniLayer,
    InfiniAttention,
    MultiMemoryLayer,
    MultiMemoryAttention,
)

# Building blocks
from .base_components import (  # noqa: E402
    PythiaMLP,
    init_weights,
    count_parameters,
)
from .memory_utils import (  # noqa: E402
    elu_plus_one,
    causal_linear_attention,
    MemoryStateMixin,
    create_memory_matrix,
    create_memory_norm,
    retrieve_from_memory,
    update_memory_delta_rule,
    update_memory_simple,
)
from .position_encoding import (  # noqa: E402
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)


def create_model(
    model_type: ModelTypeLiteral,
    config: Optional[SenriConfig] = None,
):
    """
    Create a model by type.

    Args:
        model_type: Model type ("pythia", "infini", "multi_memory")
        config: SenriConfig for model structure and memory settings.
            If None, uses default SenriConfig.

    Returns:
        TransformerLM instance

    Examples:
        from src.config import SenriConfig

        # Standard model (Senri config)
        model = create_model("pythia")

        # Infini model (default config)
        model = create_model("infini")

        # Infini model (custom config)
        config = SenriConfig(num_memory_banks=2, segments_per_bank=8)
        model = create_model("infini", config)

        # Multi-Memory (custom config)
        config = SenriConfig(num_memories=8, use_delta_rule=False)
        model = create_model("multi_memory", config)
    """
    if config is None:
        config = SenriConfig()

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

    elif model_type == "infini":
        layers = [
            InfiniLayer(
                h, n, i,
                config.num_memory_banks,
                config.segments_per_bank,
                config.use_delta_rule,
            ),
            *pythia_layers(config.num_layers - 1),
        ]

    elif model_type == "multi_memory":
        layers = [
            MultiMemoryLayer(h, n, i, config.num_memories, config.use_delta_rule),
            *pythia_layers(config.num_layers - 1),
        ]

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: pythia, infini, multi_memory"
        )

    return TransformerLM(
        layers=layers,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
    )


__all__ = [
    # Factory function
    'create_model',

    # Core models
    'TransformerLM',

    # Layer types
    'BaseLayer',
    'PythiaLayer',
    'PythiaAttention',
    'InfiniLayer',
    'InfiniAttention',
    'MultiMemoryLayer',
    'MultiMemoryAttention',

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
