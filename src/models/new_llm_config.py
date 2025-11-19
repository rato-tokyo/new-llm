"""
HuggingFace-compatible configuration for New-LLM

This allows New-LLM to use all HuggingFace Transformers features:
- Trainer
- GenerationMixin
- Pipeline API
- Model Hub integration
"""

from transformers import PretrainedConfig


class NewLLMConfig(PretrainedConfig):
    """Configuration for New-LLM (Context Vector Propagation Language Model)

    New-LLM replaces attention mechanisms with context vector propagation,
    achieving O(1) memory usage instead of O(n²).

    Args:
        vocab_size (int): Vocabulary size. Default: 10000
        embed_dim (int): Token embedding dimension. Default: 256
        hidden_dim (int): Hidden layer dimension. Default: 512
        context_vector_dim (int): Context vector dimension. Default: 256
        num_layers (int): Number of layers. Default: 1
        max_seq_length (int): Maximum sequence length. Default: 512
        dropout (float): Dropout rate. Default: 0.1
        pad_token_id (int): ID for padding token. Default: 0
        bos_token_id (int): ID for beginning-of-sequence token. Default: 2
        eos_token_id (int): ID for end-of-sequence token. Default: 3
    """

    model_type = "new_llm"

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=256,
        hidden_dim=512,
        context_vector_dim=256,
        num_layers=1,
        max_seq_length=512,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_vector_dim = context_vector_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.dropout = dropout
