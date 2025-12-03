"""
Embedding utilities

GPT-2 pretrained embeddings のロードと初期化を共通化
"""

import torch.nn as nn

from src.utils.io import print_flush


def load_pretrained_gpt2_embeddings(
    vocab_size: int,
    embed_dim: int,
    freeze: bool = True
) -> nn.Embedding:
    """
    GPT-2 pretrained embeddings をロード

    Args:
        vocab_size: ボキャブラリサイズ
        embed_dim: 埋め込み次元
        freeze: 重みを凍結するか（デフォルト: True）

    Returns:
        nn.Embedding: GPT-2 embeddingsをロードしたEmbedding層
    """
    try:
        from transformers import GPT2Model
        print_flush("Loading GPT-2 pretrained embeddings...")

        gpt2 = GPT2Model.from_pretrained('gpt2')
        pretrained_embeddings = gpt2.wte.weight.data

        embedding = nn.Embedding(vocab_size, embed_dim)
        embedding.weight.data.copy_(pretrained_embeddings)
        embedding.weight.requires_grad = not freeze

        print_flush(f"✓ Loaded GPT-2 embeddings: {pretrained_embeddings.shape}")

        return embedding

    except Exception as e:
        print_flush(f"Warning: Failed to load GPT-2 embeddings: {e}")
        print_flush("Falling back to random initialization...")
        return create_random_embedding(vocab_size, embed_dim, freeze=False)


def create_random_embedding(
    vocab_size: int,
    embed_dim: int,
    freeze: bool = False,
    mean: float = 0.0,
    std: float = 0.02
) -> nn.Embedding:
    """
    ランダム初期化されたEmbedding層を作成

    Args:
        vocab_size: ボキャブラリサイズ
        embed_dim: 埋め込み次元
        freeze: 重みを凍結するか（デフォルト: False）
        mean: 初期化の平均値
        std: 初期化の標準偏差

    Returns:
        nn.Embedding: ランダム初期化されたEmbedding層
    """
    embedding = nn.Embedding(vocab_size, embed_dim)
    nn.init.normal_(embedding.weight, mean=mean, std=std)
    embedding.weight.requires_grad = not freeze

    return embedding
