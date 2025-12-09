"""
Senri Configuration

Senri: Japanese LLM with Compressive Memory
OpenCALMトークナイザーを使用した日本語LLM。

トークナイザーとvocab_sizeの設定のみを管理。
レイヤー設定はlayers.pyのLayerConfigを使用。
実験設定はexperiment.pyのExperimentConfigを使用。

Usage:
    from src.config import OPEN_CALM_TOKENIZER, OPEN_CALM_VOCAB_SIZE
    from src.config import default_senri_layers
    from src.models import create_model

    # デフォルト構成でモデル作成
    layers = default_senri_layers()
    model = create_model(layers)
"""

from .open_calm import OPEN_CALM_VOCAB_SIZE, OPEN_CALM_TOKENIZER

# Re-export for convenience
__all__ = [
    "OPEN_CALM_VOCAB_SIZE",
    "OPEN_CALM_TOKENIZER",
]
