"""
Configuration Constants

設定用の定数をまとめて定義。
全ての値はここで一元管理し、他の場所ではデフォルト値を使わない。
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

# ========== モデルアーキテクチャ定数 ==========
# これらの値は models.py で明示的に使用される
# デフォルト値として暗黙的に使用してはならない

MODEL_HIDDEN_SIZE = 512
MODEL_NUM_HEADS = 8
MODEL_INTERMEDIATE_SIZE = 2048
MODEL_NUM_LAYERS = 6

# ========== SenriLayer 定数 ==========
SENRI_NUM_MEMORIES = 1
SENRI_MEMORY_HEAD_DIM = 512  # = hidden_size (シングルヘッド)
SENRI_USE_DELTA_RULE = True

# ========== PythiaLayer 定数 ==========
PYTHIA_ROTARY_PCT = 0.25
PYTHIA_MAX_POSITION_EMBEDDINGS = 2048
