"""
Model Presets

プリセットモデル構成を提供。
レイヤーリストを直接定義することで、モデル構造が一目でわかる。

使用例:
    from src.config.models import SENRI_MODEL, PYTHIA_MODEL

    model = SENRI_MODEL()  # Senri: 1 Senri + 5 Pythia
    model = PYTHIA_MODEL()  # Pythia: 6 Pythia layers
"""

from src.models.model import SenriModel
from src.models.layers import SenriLayer, PythiaLayer


def PYTHIA_MODEL() -> SenriModel:
    """
    Pythia ベースライン（全6層がPythiaLayer）

    構成:
        Layer 0-5: PythiaLayer (RoPE + Softmax Attention)
    """
    return SenriModel([
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
    ])


def SENRI_MODEL() -> SenriModel:
    """
    Senri 標準構成（Layer 0がSenri、Layer 1-5がPythia）

    構成:
        Layer 0: SenriLayer (Compressive Memory + Linear Attention)
        Layer 1-5: PythiaLayer (RoPE + Softmax Attention)
    """
    return SenriModel([
        SenriLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
    ])


def SENRI_MULTI_MEMORY_MODEL() -> SenriModel:
    """
    Senri 複数メモリ構成（4メモリ）

    構成:
        Layer 0: SenriLayer (4 memories)
        Layer 1-5: PythiaLayer
    """
    return SenriModel([
        SenriLayer(num_memories=4),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
    ])


def SENRI_ONLY_MODEL() -> SenriModel:
    """
    Senri-Only（全6層がSenriLayer）

    構成:
        Layer 0-5: SenriLayer (Compressive Memory + Linear Attention)
    """
    return SenriModel([
        SenriLayer(),
        SenriLayer(),
        SenriLayer(),
        SenriLayer(),
        SenriLayer(),
        SenriLayer(),
    ])


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
