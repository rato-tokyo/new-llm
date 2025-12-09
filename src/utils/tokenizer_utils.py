"""
Tokenizer Utilities

トークナイザーの取得と設定を統一。
"""

from transformers import AutoTokenizer, PreTrainedTokenizer

from src.config import OPEN_CALM_TOKENIZER

# 定義済みトークナイザー名（OpenCALM以外）
PYTHIA_TOKENIZER = "EleutherAI/pythia-70m"


def get_tokenizer(model_name: str = OPEN_CALM_TOKENIZER) -> PreTrainedTokenizer:
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


def get_open_calm_tokenizer() -> PreTrainedTokenizer:
    """
    OpenCALMトークナイザーを取得

    byte_fallback対応で、UNKトークンなし。
    英語、絵文字、特殊文字も完全対応。

    Returns:
        tokenizer: OpenCALM日本語トークナイザー
    """
    return get_tokenizer(OPEN_CALM_TOKENIZER)


def test_tokenizer_coverage(tokenizer: PreTrainedTokenizer, text: str) -> dict:
    """
    テキストのトークナイズ結果をテスト

    UNKトークンが含まれるかどうかを確認。

    Args:
        tokenizer: トークナイザー
        text: テストするテキスト

    Returns:
        result: {
            "tokens": トークンリスト,
            "token_ids": トークンIDリスト,
            "has_unk": UNKが含まれるか,
            "unk_count": UNKの数
        }
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    unk_id = tokenizer.unk_token_id
    unk_count = token_ids.count(unk_id) if unk_id is not None else 0

    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "has_unk": unk_count > 0,
        "unk_count": unk_count,
    }
