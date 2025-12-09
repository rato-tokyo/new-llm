"""
Model Presets

プリセットモデル構成を提供。
レイヤーパラメータは直接数値で指定（レイヤーごとに変更可能）。

使用例:
    from src.config.models import SENRI_MODEL, PYTHIA_MODEL

    model = SENRI_MODEL()  # Senri: 1 Senri (2 memories) + 5 Pythia
    model = PYTHIA_MODEL()  # Pythia: 6 Pythia layers
"""

from src.models.model import TransformerLM
from src.models.layers import SenriLayer, PythiaLayer
from src.config.constants import MODEL_VOCAB_SIZE


def PYTHIA_MODEL() -> TransformerLM:
    """
    Pythia ベースライン（全6層がPythiaLayer）

    構成:
        Layer 0-5: PythiaLayer (RoPE + Softmax Attention)
    """
    return TransformerLM(
        layers=[
            # PythiaLayer引数:
            #   hidden_size: 隠れ層の次元数
            #   num_heads: Multi-Head Attentionのヘッド数
            #   intermediate_size: MLP中間層の次元数（通常hidden_size * 4）
            #   rotary_pct: RoPEを適用する次元の割合（0.25 = 25%）
            #   max_position_embeddings: 最大位置埋め込み長（RoPEの周波数計算に使用）
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
        ],
        vocab_size=MODEL_VOCAB_SIZE,
    )


def SENRI_MODEL() -> TransformerLM:
    """
    Senri 標準構成（Layer 0がSenri、Layer 1-5がPythia）

    構成:
        Layer 0: SenriLayer (num_memories=2)
          - memory[0]: Working Memory（会話コンテキスト、常に更新）
          - memory[1]: Detail Memory（知識格納、freeze可能）
        Layer 1-5: PythiaLayer (RoPE + Softmax Attention)
    """
    return TransformerLM(
        layers=[
            # SenriLayer引数:
            #   hidden_size: 隠れ層の次元数
            #   num_heads: Linear Attentionのヘッド数
            #   intermediate_size: MLP中間層の次元数
            #   num_memories: メモリスロット数（>=2推奨: Working + Detail）
            #   memory_head_dim: メモリ操作時のヘッド次元（=hidden_sizeでシングルヘッド推奨）
            #   use_delta_rule: Delta Rule使用（メモリ上書き防止、True推奨）
            SenriLayer(hidden_size=512, num_heads=8, intermediate_size=2048, num_memories=2, memory_head_dim=512, use_delta_rule=True),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
            PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048, rotary_pct=0.25, max_position_embeddings=2048),
        ],
        vocab_size=MODEL_VOCAB_SIZE,
    )


# プリセット名と関数のマッピング
MODEL_PRESETS = {
    "pythia": PYTHIA_MODEL,
    "senri": SENRI_MODEL,
}


def create_model(preset: str) -> TransformerLM:
    """
    プリセット名からモデルを作成

    Args:
        preset: プリセット名 ("pythia", "senri")

    Returns:
        TransformerLM インスタンス

    Example:
        model = create_model("senri")
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[preset]()
