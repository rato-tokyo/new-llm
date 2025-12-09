"""
OpenCALM Tokenizer Configuration

CyberAgentのOpenCALMトークナイザー固有の情報。
https://huggingface.co/cyberagent/open-calm-small

特徴:
- UNKトークンなし（byte_fallback対応）
- 日本語に最適化された語彙
- 英語、絵文字も完全対応
- vocab_size=52,000

Usage:
    from src.config.open_calm import OPEN_CALM_TOKENIZER, OPEN_CALM_VOCAB_SIZE
    from src.utils.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(OPEN_CALM_TOKENIZER)
"""


# ========== OpenCALM トークナイザー定数 ==========
OPEN_CALM_TOKENIZER = "cyberagent/open-calm-small"
OPEN_CALM_VOCAB_SIZE = 52000

# ========== トークナイザー特性 ==========
# - byte_fallback: True（任意のバイト列を処理可能）
# - unk_token: None（UNKトークンなし）
# - eos_token: "</s>" (id=1)
# - pad_token: None → eos_tokenを使用


class OpenCalmConfig:
    """OpenCALM 互換設定（後方互換性のため残存）

    NOTE: 新規コードではSenriConfigを使用してください。
    このクラスは後方互換性のために残しています。

    Usage:
        # 推奨
        from src.config import SenriConfig
        config = SenriConfig()

        # 後方互換（非推奨）
        from src.config import OpenCalmConfig
        config = OpenCalmConfig()
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
