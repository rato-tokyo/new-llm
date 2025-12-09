"""
OpenCALM Japanese GPT Configuration

日本語LLM用の設定。CyberAgentのOpenCALMトークナイザーを使用。
https://huggingface.co/cyberagent/open-calm-small

特徴:
- UNKトークンなし（byte_fallback対応）
- 英語、絵文字も完全対応
- vocab_size=52000（Pythiaに近い）

Usage:
    from src.config import OpenCalmConfig
    from src.models import create_model

    config = OpenCalmConfig()
    model = create_model("pythia", base_config=config)
"""


class OpenCalmConfig:
    """OpenCALM 日本語LLM設定

    CyberAgentのOpenCALMトークナイザーを使用した日本語LLM。
    byte_fallback対応で、UNKトークンなし。

    Attributes:
        vocab_size: OpenCALMの語彙サイズ（52,000）
        hidden_size: 隠れ層の次元（512）
        num_layers: レイヤー数（6）
        num_attention_heads: アテンションヘッド数（8）
        intermediate_size: FFNの中間層サイズ（2048）
        max_position_embeddings: 最大シーケンス長（2048）
        rotary_pct: Rotary embeddingの割合（0.25）
        tokenizer_name: OpenCALMトークナイザー名
    """

    # ========== モデル構造 ==========
    vocab_size = 52000              # OpenCALMの語彙サイズ
    hidden_size = 512               # 隠れ層の次元
    num_layers = 6                  # レイヤー数
    num_attention_heads = 8         # アテンションヘッド数
    intermediate_size = 2048        # FFNの中間層サイズ
    max_position_embeddings = 2048  # 最大シーケンス長
    rotary_pct = 0.25               # Rotary embeddingの割合

    # ========== 学習設定 ==========
    learning_rate = 1e-4            # 学習率
    batch_size = 8                  # バッチサイズ
    num_epochs = 30                 # 最大エポック数

    # ========== トークナイザー ==========
    tokenizer_name = "cyberagent/open-calm-small"
