"""
Senri Configuration

Senri: Japanese LLM with Compressive Memory
OpenCALMトークナイザーを使用した日本語LLM。

特徴:
- Infini-Attention による圧縮メモリ
- Working Memory / Index Memory / Detail Memory の3層構成
- OpenCALMトークナイザー（UNKなし、byte_fallback対応）

Usage:
    from src.config import SenriConfig
    from src.models import create_model

    config = SenriConfig()
    model = create_model("infini", base_config=config)
"""

from .open_calm import OPEN_CALM_VOCAB_SIZE, OPEN_CALM_TOKENIZER


class SenriConfig:
    """Senri 日本語LLM設定

    Compressive Memoryを持つ日本語LLM。
    OpenCALMトークナイザーを使用。

    Attributes:
        vocab_size: 語彙サイズ（OpenCALMに準拠: 52,000）
        hidden_size: 隠れ層の次元（512）
        num_layers: レイヤー数（6）
        num_attention_heads: アテンションヘッド数（8）
        intermediate_size: FFNの中間層サイズ（2048）
        max_position_embeddings: 最大シーケンス長（2048）
        rotary_pct: Rotary embeddingの割合（0.25）
        tokenizer_name: OpenCALMトークナイザー名
    """

    # ========== モデル構造 ==========
    vocab_size = OPEN_CALM_VOCAB_SIZE   # OpenCALMの語彙サイズ
    hidden_size = 512                   # 隠れ層の次元
    num_layers = 6                      # レイヤー数
    num_attention_heads = 8             # アテンションヘッド数
    intermediate_size = 2048            # FFNの中間層サイズ
    max_position_embeddings = 2048      # 最大シーケンス長
    rotary_pct = 0.25                   # Rotary embeddingの割合

    # ========== 学習設定 ==========
    learning_rate = 1e-4                # 学習率
    batch_size = 8                      # バッチサイズ
    num_epochs = 30                     # 最大エポック数

    # ========== トークナイザー ==========
    tokenizer_name = OPEN_CALM_TOKENIZER
