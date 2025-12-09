"""
Configuration Constants

設定用の定数をまとめて定義。
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
#
# トークナイザー特性:
# - byte_fallback: True（任意のバイト列を処理可能）
# - unk_token: None（UNKトークンなし）
# - eos_token: "</s>" (id=1)
# - pad_token: None → eos_tokenを使用

OPEN_CALM_TOKENIZER = "cyberagent/open-calm-small"
OPEN_CALM_VOCAB_SIZE = 52000

# ========== Pythia トークナイザー定数 ==========
PYTHIA_TOKENIZER = "EleutherAI/pythia-70m"
