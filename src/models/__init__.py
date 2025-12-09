"""New-LLM Models

OpenCALM (Japanese LLM) based experimental architectures.
Infini-Attention for compressive memory.

## Quick Start

```python
from src.models import create_model
from src.config import OpenCalmConfig, InfiniConfig, MultiMemoryConfig

# 標準モデル（OpenCALM設定）
model = create_model("pythia")

# Infiniモデル（デフォルト設定）
model = create_model("infini")

# Infiniモデル（カスタム設定）
config = InfiniConfig(num_memory_banks=2, segments_per_bank=8)
model = create_model("infini", model_config=config)

# Multi-Memory（デフォルト: 4メモリ）
model = create_model("multi_memory")

# Multi-Memory（カスタム設定）
config = MultiMemoryConfig(num_memories=8, use_delta_rule=False)
model = create_model("multi_memory", model_config=config)
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

from typing import Optional, Union

from src.config import (
    PythiaConfig,
    OpenCalmConfig,
    InfiniConfig,
    MultiMemoryConfig,
    ModelConfigType,
    ModelTypeLiteral,
)

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

# Base config type (supports both PythiaConfig and OpenCalmConfig)
BaseConfigType = Union[PythiaConfig, OpenCalmConfig]

def create_model(
    model_type: ModelTypeLiteral,
    base_config: Optional[BaseConfigType] = None,
    model_config: ModelConfigType = None,
):
    """
    Create a model by type.

    Args:
        model_type: Model type ("pythia", "infini", "multi_memory")
        base_config: OpenCalmConfig or PythiaConfig for base model structure
            (uses OpenCalmConfig if None)
        model_config: Model-specific config (InfiniConfig or MultiMemoryConfig)
            If None, uses default config for the model type.

    Returns:
        TransformerLM instance

    Examples:
        from src.config import OpenCalmConfig, InfiniConfig, MultiMemoryConfig

        # Standard model (OpenCALM config)
        model = create_model("pythia")

        # Infini model (default config)
        model = create_model("infini")

        # Infini model (custom config)
        config = InfiniConfig(num_memory_banks=2, segments_per_bank=8)
        model = create_model("infini", model_config=config)

        # Multi-Memory (custom config)
        config = MultiMemoryConfig(num_memories=8, use_delta_rule=False)
        model = create_model("multi_memory", model_config=config)
    """
    if base_config is None:
        base_config = OpenCalmConfig()

    h = base_config.hidden_size
    n = base_config.num_attention_heads
    i = base_config.intermediate_size
    r = base_config.rotary_pct
    m = base_config.max_position_embeddings

    def pythia_layers(count: int) -> list[BaseLayer]:
        return [PythiaLayer(h, n, i, r, m) for _ in range(count)]

    if model_type == "pythia":
        layers = pythia_layers(base_config.num_layers)
        return TransformerLM(
            layers=layers,
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
        )

    elif model_type == "infini":
        # Use InfiniConfig (default if not provided)
        infini_cfg = model_config if isinstance(model_config, InfiniConfig) else InfiniConfig()
        layers = [
            InfiniLayer(
                h, n, i,
                infini_cfg.num_memory_banks,
                infini_cfg.segments_per_bank,
                infini_cfg.use_delta_rule,
            ),
            *pythia_layers(base_config.num_layers - 1),
        ]

    elif model_type == "multi_memory":
        # Use MultiMemoryConfig (default if not provided)
        mm_cfg = model_config if isinstance(model_config, MultiMemoryConfig) else MultiMemoryConfig()
        layers = [
            MultiMemoryLayer(h, n, i, mm_cfg.num_memories, mm_cfg.use_delta_rule),
            *pythia_layers(base_config.num_layers - 1),
        ]

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: pythia, infini, multi_memory"
        )

    return TransformerLM(
        layers=layers,
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
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
