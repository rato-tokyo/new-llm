"""
HuggingFace-compatible New-LLM implementation

Wraps the existing ContextVectorLLM in a PreTrainedModel interface.
This enables use of:
- HuggingFace Trainer
- GenerationMixin (with beam search, sampling, etc.)
- Model Hub
- Pipeline API
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .new_llm_config import NewLLMConfig
from .context_vector_llm import ContextVectorLLM


class NewLLMForCausalLM(PreTrainedModel, GenerationMixin):
    """HuggingFace-compatible New-LLM for Causal Language Modeling

    This wraps the original ContextVectorLLM to make it compatible
    with HuggingFace Transformers ecosystem.

    Example:
        >>> from transformers import AutoTokenizer
        >>> config = NewLLMConfig(vocab_size=10000, num_layers=1)
        >>> model = NewLLMForCausalLM(config)
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>>
        >>> # Use with Trainer
        >>> from transformers import Trainer, TrainingArguments
        >>> trainer = Trainer(model=model, args=TrainingArguments(...))
        >>> trainer.train()
        >>>
        >>> # Use for generation
        >>> input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
        >>> outputs = model.generate(input_ids, max_length=50)
    """

    config_class = NewLLMConfig

    def __init__(self, config: NewLLMConfig):
        super().__init__(config)

        # Create the underlying New-LLM model
        # We need to create a legacy config object for compatibility
        from src.utils.config import NewLLMConfig as LegacyConfig

        legacy_config = LegacyConfig()
        legacy_config.vocab_size = config.vocab_size
        legacy_config.embed_dim = config.embed_dim
        legacy_config.hidden_dim = config.hidden_dim
        legacy_config.context_vector_dim = config.context_vector_dim
        legacy_config.num_layers = config.num_layers
        legacy_config.max_seq_length = config.max_seq_length
        legacy_config.dropout = config.dropout

        self.model = ContextVectorLLM(legacy_config)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        """Required by PreTrainedModel"""
        return self.model.embedding

    def set_input_embeddings(self, value):
        """Required by PreTrainedModel"""
        self.model.embedding = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
        **kwargs
    ):
        """Forward pass compatible with HuggingFace

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask (ignored for New-LLM, O(1) memory)
            labels: Labels for language modeling loss [batch_size, seq_len]
            return_dict: Whether to return ModelOutput object

        Returns:
            CausalLMOutputWithCrossAttentions with loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through New-LLM
        outputs = self.model(input_ids)

        if isinstance(outputs, tuple):
            logits, context = outputs
        else:
            logits = outputs

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute loss
            # DataCollatorForLanguageModeling uses -100 as ignore_index
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,  # New-LLM doesn't use key-value cache
            hidden_states=None,
            attentions=None,  # New-LLM doesn't have attention
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation

        Required by GenerationMixin for text generation.
        """
        return {"input_ids": input_ids}

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Reorder cache for beam search

        New-LLM doesn't use cache, so this is a no-op.
        """
        return past
