"""
Pythia-70M Configuration

英語ベースラインモデル用のレガシー設定。
日本語LLMにはSenriConfigを使用してください。

https://huggingface.co/EleutherAI/pythia-70m
"""


class PythiaConfig:
    """Pythia-70M モデル設定 (英語ベースライン用)

    Note:
        日本語LLMにはSenriConfigを使用してください。
        このクラスは英語ベースライン実験用に残されています。
    """

    # ========== モデル構造 ==========
    vocab_size = 50304              # Pythiaの語彙サイズ
    hidden_size = 512               # 隠れ層の次元
    num_layers = 6                  # レイヤー数
    num_attention_heads = 8         # アテンションヘッド数
    intermediate_size = 2048        # FFNの中間層サイズ
    max_position_embeddings = 2048  # 最大シーケンス長
    rotary_pct = 0.25               # Rotary embeddingの割合

    # ========== トークナイザー ==========
    tokenizer_name = "EleutherAI/pythia-70m"


# ========== 定数 ==========
# NOTE: Do not change this value without explicit user approval
EARLY_STOPPING_PATIENCE = 1

# Gradient clipping
GRADIENT_CLIP = 1.0
