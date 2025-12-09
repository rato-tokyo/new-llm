"""
Model Configuration

モデル構成の一元管理。
レイヤー構成、パラメータ、プリセットをすべてここで定義。
"""

from dataclasses import dataclass, field
from typing import Optional

from .constants import OPEN_CALM_VOCAB_SIZE


@dataclass
class ModelConfig:
    """
    モデル構成の一元管理

    使用例:
        from src.config import PYTHIA_CONFIG, SENRI_CONFIG
        from src.config.models import create_model_from_config

        model = create_model_from_config(PYTHIA_CONFIG)
        model = create_model_from_config(SENRI_CONFIG)
    """

    # === レイヤー構成 ===
    num_layers: int = 6
    senri_layer_indices: tuple[int, ...] = (0,)  # どの層がSenriか (空=全てPythia)

    # === 共通パラメータ ===
    hidden_size: int = 512
    num_heads: int = 8
    intermediate_size: int = 2048
    vocab_size: int = OPEN_CALM_VOCAB_SIZE

    # === Senri固有パラメータ ===
    num_memories: int = 1
    memory_head_dim: Optional[int] = None  # None = hidden_size (シングルヘッド)
    use_delta_rule: bool = True

    # === Pythia固有パラメータ ===
    rotary_pct: float = 0.25
    max_position_embeddings: int = 2048

    def get_layer_type(self, layer_idx: int) -> str:
        """指定レイヤーのタイプを返す"""
        return "senri" if layer_idx in self.senri_layer_indices else "pythia"

    def describe(self) -> str:
        """モデル構成の説明文を返す"""
        if not self.senri_layer_indices:
            return f"Pythia ({self.num_layers} layers)"
        elif len(self.senri_layer_indices) == self.num_layers:
            return f"Senri-Only ({self.num_layers} Senri layers, {self.num_memories} memories)"
        else:
            senri_count = len(self.senri_layer_indices)
            pythia_count = self.num_layers - senri_count
            return f"Senri ({senri_count} Senri + {pythia_count} Pythia, {self.num_memories} memories)"


# =============================================================================
# プリセット設定
# =============================================================================

# Pythia ベースライン（全6層がPythiaLayer）
PYTHIA_CONFIG = ModelConfig(
    num_layers=6,
    senri_layer_indices=(),  # Senriレイヤーなし
)

# Senri 標準構成（Layer 0がSenri、Layer 1-5がPythia）
SENRI_CONFIG = ModelConfig(
    num_layers=6,
    senri_layer_indices=(0,),  # Layer 0のみSenri
    num_memories=1,
)

# Senri 複数メモリ構成
SENRI_MULTI_MEMORY_CONFIG = ModelConfig(
    num_layers=6,
    senri_layer_indices=(0,),
    num_memories=4,
)

# Senri-Only（全6層がSenriLayer）
SENRI_ONLY_CONFIG = ModelConfig(
    num_layers=6,
    senri_layer_indices=(0, 1, 2, 3, 4, 5),  # 全層Senri
    num_memories=1,
)


# =============================================================================
# モデル作成関数
# =============================================================================

def create_model_from_config(config: ModelConfig):
    """
    ModelConfigからTransformerLMを作成

    Args:
        config: モデル構成

    Returns:
        TransformerLM インスタンス

    使用例:
        from src.config import SENRI_CONFIG
        from src.config.models import create_model_from_config

        model = create_model_from_config(SENRI_CONFIG)
    """
    # 循環importを避けるためここでimport
    from src.models import TransformerLM, SenriLayer, PythiaLayer

    layers = []
    for i in range(config.num_layers):
        if i in config.senri_layer_indices:
            layers.append(SenriLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                num_memories=config.num_memories,
                memory_head_dim=config.memory_head_dim,
                use_delta_rule=config.use_delta_rule,
            ))
        else:
            layers.append(PythiaLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                rotary_pct=config.rotary_pct,
                max_position_embeddings=config.max_position_embeddings,
            ))

    return TransformerLM(layers=layers, vocab_size=config.vocab_size)
