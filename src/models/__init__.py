"""New-LLM Models

Pythia-70M based experimental architectures.
Infini-Attention for compressive memory.
"""

from typing import Literal, Optional

from config.pythia import PythiaConfig
from .pythia import PythiaModel
from .infini_attention import InfiniAttention, InfiniAttentionLayer
from .infini_pythia import InfiniPythiaModel
from .multi_memory_attention import MultiMemoryInfiniAttention, MultiMemoryInfiniAttentionLayer
from .multi_memory_pythia import MultiMemoryInfiniPythiaModel
from .hierarchical_memory import HierarchicalMemoryAttention, HierarchicalMemoryAttentionLayer
from .hierarchical_pythia import HierarchicalMemoryPythiaModel

# Type alias for model types
ModelTypeLiteral = Literal["pythia", "infini", "multi_memory", "hierarchical"]


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
        model_type: Model type ("pythia", "infini", "multi_memory", "hierarchical")
        config: PythiaConfig (uses default if None)
        use_delta_rule: Use delta rule for memory update (memory models only)
        num_memories: Number of memories (multi_memory, hierarchical only)
        num_memory_banks: Number of memory banks (infini only)
        segments_per_bank: Segments per bank (infini only)

    Returns:
        Model instance

    Examples:
        # Standard Pythia
        model = create_model("pythia")

        # Infini-Pythia
        model = create_model("infini")

        # Multi-Memory with 8 memories
        model = create_model("multi_memory", num_memories=8)

        # Hierarchical with custom config
        config = PythiaConfig()
        model = create_model("hierarchical", config, num_memories=4)
    """
    if config is None:
        config = PythiaConfig()

    if model_type == "pythia":
        return PythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
        )

    elif model_type == "infini":
        return InfiniPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            use_delta_rule=use_delta_rule,
            num_memory_banks=num_memory_banks,
            segments_per_bank=segments_per_bank,
        )

    elif model_type == "multi_memory":
        return MultiMemoryInfiniPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            use_delta_rule=use_delta_rule,
            num_memories=num_memories,
        )

    elif model_type == "hierarchical":
        return HierarchicalMemoryPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            use_delta_rule=use_delta_rule,
            num_fine_memories=num_memories,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: pythia, infini, multi_memory, hierarchical"
        )


__all__ = [
    # Factory function
    'create_model',
    'ModelTypeLiteral',
    # Core models
    'PythiaModel',
    'InfiniPythiaModel',
    'MultiMemoryInfiniPythiaModel',
    'HierarchicalMemoryPythiaModel',
    # Infini-Attention
    'InfiniAttention',
    'InfiniAttentionLayer',
    'MultiMemoryInfiniAttention',
    'MultiMemoryInfiniAttentionLayer',
    'HierarchicalMemoryAttention',
    'HierarchicalMemoryAttentionLayer',
]
