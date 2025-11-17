"""GPT-2 baseline model for comparison with New-LLM

GPT-2を比較ベースラインとして使用。
New-LLMと同じタスクで評価し、性能を比較する。
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class GPT2Baseline(nn.Module):
    """GPT-2 Small wrapper for fair comparison

    HuggingFaceのGPT-2を使用しつつ、New-LLMと同じ
    インターフェースで訓練・評価できるようにラップ。

    Note: GPT-2 Smallは約117Mパラメータ
    New-LLM (4M)と比較するには不公平だが、
    「attention機構の効果」を測る基準として重要。
    """

    def __init__(self, vocab_size: int, max_seq_length: int = 64):
        super().__init__()

        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            raise ImportError(
                "Transformers library required. Install with:\n"
                "pip install transformers"
            )

        # GPT-2 Small相当の設定
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_length,  # 最大シーケンス長
            n_embd=768,                   # 埋め込み次元（GPT-2 Small標準）
            n_layer=12,                   # レイヤー数（GPT-2 Small標準）
            n_head=12,                    # アテンションヘッド数
            resid_pdrop=0.1,              # Dropout
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )

        # GPT-2モデル生成（事前学習なし、スクラッチから訓練）
        self.model = GPT2LMHeadModel(config)

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        outputs = self.model(input_ids)
        return outputs.logits

    def get_num_parameters(self) -> int:
        """モデルのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyGPT2Baseline(nn.Module):
    """Tiny GPT-2 for fairer comparison with New-LLM

    GPT-2 Smallは117Mパラメータで大きすぎるため、
    New-LLM (4M)と同規模のミニGPT-2を作成。

    パラメータ数を揃えて「アーキテクチャの差」のみを比較。
    """

    def __init__(self, vocab_size: int, max_seq_length: int = 64):
        super().__init__()

        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            raise ImportError("Transformers library required")

        # New-LLMと同規模（約4M）になるよう調整
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_length,
            n_embd=256,        # New-LLMと同じ
            n_layer=6,         # レイヤー数削減
            n_head=4,          # ヘッド数削減（n_embdの約数である必要）
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )

        self.model = GPT2LMHeadModel(config)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # パラメータ数確認
        num_params = self.get_num_parameters()
        print(f"TinyGPT2 parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        outputs = self.model(input_ids)
        return outputs.logits

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gpt2_baseline(config, tiny: bool = True):
    """Create GPT-2 baseline model

    Args:
        config: Configuration object with vocab_size, max_seq_length
        tiny: If True, create TinyGPT2 (~4M params) for fair comparison
              If False, create GPT-2 Small (117M params) for reference

    Returns:
        GPT-2 model instance
    """
    if tiny:
        print("Creating TinyGPT2 baseline (~4M params)...")
        model = TinyGPT2Baseline(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length
        )
    else:
        print("Creating GPT-2 Small baseline (117M params)...")
        model = GPT2Baseline(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length
        )

    return model
