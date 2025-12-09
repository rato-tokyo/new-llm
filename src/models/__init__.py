"""Senri Models

Senri: Japanese LLM with Compressive Memory
Infini-Attention for efficient long-context processing.

## Quick Start

```python
from src.config import SenriLayerConfig, PythiaLayerConfig, default_senri_layers
from src.models import create_model

# デフォルト構成（1 Senri + 5 Pythia）
layers = default_senri_layers()
model = create_model(layers)

# カスタム構成
layers = [
    SenriLayerConfig(num_memories=8, use_multi_memory=True),
    PythiaLayerConfig(),
    PythiaLayerConfig(),
    PythiaLayerConfig(),
]
model = create_model(layers)

# 全層Pythia（ベースライン）
from src.config import default_pythia_layers
layers = default_pythia_layers(6)
model = create_model(layers)
```

## Direct Layer Construction

```python
from src.models import TransformerLM
from src.models.layers import InfiniLayer, PythiaLayer

# 直接レイヤーを構築
layers = [
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
]
model = TransformerLM(layers=layers, vocab_size=52000, hidden_size=512)
```
"""

from typing import Optional

from src.config import OPEN_CALM_VOCAB_SIZE
from src.config.layers import (
    BaseLayerConfig,
    PythiaLayerConfig,
    SenriLayerConfig,
    LayerConfigType,
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


def _build_layer(config: LayerConfigType) -> BaseLayer:
    """LayerConfigからレイヤーインスタンスを構築"""
    if isinstance(config, SenriLayerConfig):
        if config.use_multi_memory:
            return MultiMemoryLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                num_memories=config.num_memories,
                use_delta_rule=config.use_delta_rule,
            )
        else:
            return InfiniLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                num_memory_banks=config.num_memory_banks,
                segments_per_bank=config.segments_per_bank,
                use_delta_rule=config.use_delta_rule,
            )
    elif isinstance(config, PythiaLayerConfig):
        return PythiaLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            rotary_pct=config.rotary_pct,
            max_position_embeddings=config.max_position_embeddings,
        )
    else:
        raise ValueError(f"Unknown layer config type: {type(config)}")


def create_model(
    layer_configs: list[LayerConfigType],
    vocab_size: Optional[int] = None,
    hidden_size: Optional[int] = None,
) -> TransformerLM:
    """
    LayerConfigのリストからモデルを構築

    Args:
        layer_configs: レイヤー設定のリスト
        vocab_size: 語彙サイズ（デフォルト: OpenCALM 52,000）
        hidden_size: 隠れ層サイズ（デフォルト: 最初のレイヤーから取得）

    Returns:
        TransformerLM instance

    Examples:
        from src.config import SenriLayerConfig, PythiaLayerConfig, default_senri_layers

        # デフォルト構成
        model = create_model(default_senri_layers())

        # カスタム構成
        layers = [
            SenriLayerConfig(num_memories=8),
            PythiaLayerConfig(),
            PythiaLayerConfig(),
        ]
        model = create_model(layers)

        # カスタムvocab_size
        model = create_model(layers, vocab_size=50304)
    """
    if not layer_configs:
        raise ValueError("layer_configs cannot be empty")

    # デフォルト値の設定
    if vocab_size is None:
        vocab_size = OPEN_CALM_VOCAB_SIZE

    if hidden_size is None:
        hidden_size = layer_configs[0].hidden_size

    # レイヤーを構築
    layers = [_build_layer(config) for config in layer_configs]

    return TransformerLM(
        layers=layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
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
