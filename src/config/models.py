"""
Model Presets

プリセットモデル構成を提供。
全てのパラメータを明示的に指定し、デフォルト値に依存しない。
PythiaLayerとSenriLayerは独立した設定を持つ。

使用例:
    from src.config.models import SENRI_MODEL, PYTHIA_MODEL

    model = SENRI_MODEL()  # Senri: 1 Senri + 5 Pythia
    model = PYTHIA_MODEL()  # Pythia: 6 Pythia layers
"""

from src.models.model import SenriModel
from src.models.layers import SenriLayer, PythiaLayer
from src.config.constants import (
    # Model共通
    MODEL_HIDDEN_SIZE,
    MODEL_VOCAB_SIZE,
    # PythiaLayer専用
    PYTHIA_HIDDEN_SIZE,
    PYTHIA_NUM_HEADS,
    PYTHIA_INTERMEDIATE_SIZE,
    PYTHIA_ROTARY_PCT,
    PYTHIA_MAX_POSITION_EMBEDDINGS,
    # SenriLayer専用
    SENRI_HIDDEN_SIZE,
    SENRI_NUM_HEADS,
    SENRI_INTERMEDIATE_SIZE,
    SENRI_NUM_MEMORIES,
    SENRI_MEMORY_HEAD_DIM,
    SENRI_USE_DELTA_RULE,
)


def _create_pythia_layer() -> PythiaLayer:
    """PythiaLayerを明示的なパラメータで作成"""
    return PythiaLayer(
        hidden_size=PYTHIA_HIDDEN_SIZE,
        num_heads=PYTHIA_NUM_HEADS,
        intermediate_size=PYTHIA_INTERMEDIATE_SIZE,
        rotary_pct=PYTHIA_ROTARY_PCT,
        max_position_embeddings=PYTHIA_MAX_POSITION_EMBEDDINGS,
    )


def _create_senri_layer(num_memories: int) -> SenriLayer:
    """SenriLayerを明示的なパラメータで作成"""
    return SenriLayer(
        hidden_size=SENRI_HIDDEN_SIZE,
        num_heads=SENRI_NUM_HEADS,
        intermediate_size=SENRI_INTERMEDIATE_SIZE,
        num_memories=num_memories,
        memory_head_dim=SENRI_MEMORY_HEAD_DIM,
        use_delta_rule=SENRI_USE_DELTA_RULE,
    )


def PYTHIA_MODEL() -> SenriModel:
    """
    Pythia ベースライン（全6層がPythiaLayer）

    構成:
        Layer 0-5: PythiaLayer (RoPE + Softmax Attention)

    パラメータ (constants.py から):
        PYTHIA_HIDDEN_SIZE: 512
        PYTHIA_NUM_HEADS: 8
        PYTHIA_INTERMEDIATE_SIZE: 2048
        PYTHIA_ROTARY_PCT: 0.25
    """
    return SenriModel(
        layers=[
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
        ],
        vocab_size=MODEL_VOCAB_SIZE,
        hidden_size=MODEL_HIDDEN_SIZE,
    )


def SENRI_MODEL() -> SenriModel:
    """
    Senri 標準構成（Layer 0がSenri、Layer 1-5がPythia）

    構成:
        Layer 0: SenriLayer (Compressive Memory + Linear Attention)
        Layer 1-5: PythiaLayer (RoPE + Softmax Attention)

    パラメータ (constants.py から):
        SENRI_HIDDEN_SIZE: 512
        SENRI_NUM_HEADS: 8
        SENRI_INTERMEDIATE_SIZE: 2048
        SENRI_NUM_MEMORIES: 1
        SENRI_MEMORY_HEAD_DIM: 512
    """
    return SenriModel(
        layers=[
            _create_senri_layer(num_memories=SENRI_NUM_MEMORIES),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
        ],
        vocab_size=MODEL_VOCAB_SIZE,
        hidden_size=MODEL_HIDDEN_SIZE,
    )


def SENRI_MULTI_MEMORY_MODEL() -> SenriModel:
    """
    Senri 複数メモリ構成（4メモリ）

    構成:
        Layer 0: SenriLayer (4 memories)
        Layer 1-5: PythiaLayer

    パラメータ:
        num_memories: 4 (他はSENRI_MODELと同じ)
    """
    return SenriModel(
        layers=[
            _create_senri_layer(num_memories=4),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
            _create_pythia_layer(),
        ],
        vocab_size=MODEL_VOCAB_SIZE,
        hidden_size=MODEL_HIDDEN_SIZE,
    )


def SENRI_ONLY_MODEL() -> SenriModel:
    """
    Senri-Only（全6層がSenriLayer）

    構成:
        Layer 0-5: SenriLayer (Compressive Memory + Linear Attention)
    """
    return SenriModel(
        layers=[
            _create_senri_layer(num_memories=SENRI_NUM_MEMORIES),
            _create_senri_layer(num_memories=SENRI_NUM_MEMORIES),
            _create_senri_layer(num_memories=SENRI_NUM_MEMORIES),
            _create_senri_layer(num_memories=SENRI_NUM_MEMORIES),
            _create_senri_layer(num_memories=SENRI_NUM_MEMORIES),
            _create_senri_layer(num_memories=SENRI_NUM_MEMORIES),
        ],
        vocab_size=MODEL_VOCAB_SIZE,
        hidden_size=MODEL_HIDDEN_SIZE,
    )


# プリセット名と関数のマッピング
MODEL_PRESETS = {
    "pythia": PYTHIA_MODEL,
    "senri": SENRI_MODEL,
    "senri-multi": SENRI_MULTI_MEMORY_MODEL,
    "senri-only": SENRI_ONLY_MODEL,
}


def create_model(preset: str) -> SenriModel:
    """
    プリセット名からモデルを作成

    Args:
        preset: プリセット名 ("pythia", "senri", "senri-multi", "senri-only")

    Returns:
        SenriModel インスタンス

    Example:
        model = create_model("senri")
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[preset]()
