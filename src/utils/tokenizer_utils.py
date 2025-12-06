"""
Tokenizer Utilities

トークナイザーの取得と設定を統一。
"""

from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(model_name: str = "EleutherAI/pythia-70m") -> PreTrainedTokenizer:
    """
    トークナイザーを取得（pad_token設定済み）

    Args:
        model_name: HuggingFaceモデル名

    Returns:
        tokenizer: 設定済みトークナイザー
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
