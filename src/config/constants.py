"""
Configuration Constants

トークナイザーとモデル共通の定数のみを定義。
レイヤー固有のパラメータは models.py で直接数値指定する。
"""

# ========== OpenCALM トークナイザー定数 ==========
# CyberAgentのOpenCALMトークナイザー
# https://huggingface.co/cyberagent/open-calm-small
#
# 特徴:
# - UNKトークンなし（byte_fallback対応）
# - 日本語に最適化された語彙
# - 英語、絵文字も完全対応
# - vocab_size=52,000

OPEN_CALM_TOKENIZER = "cyberagent/open-calm-small"
OPEN_CALM_VOCAB_SIZE = 52000

# ========== Pythia トークナイザー定数 ==========
PYTHIA_TOKENIZER = "EleutherAI/pythia-70m"

# ========== SenriModel 共通定数 ==========
# 埋め込み層のサイズ（全レイヤーの入出力次元と一致させる必要がある）
MODEL_HIDDEN_SIZE = 512
MODEL_VOCAB_SIZE = OPEN_CALM_VOCAB_SIZE
