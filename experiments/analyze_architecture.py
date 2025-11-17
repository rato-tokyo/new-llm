"""Detailed architecture analysis of Baseline and New-LLM"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

from src.models.baseline_llm import BaselineLLM
from src.models.context_vector_llm import ContextVectorLLM
from src.utils.config import BaseConfig, NewLLMConfig


def count_parameters(model):
    """Count parameters by layer"""
    total = 0
    param_dict = {}
    for name, param in model.named_parameters():
        num_params = param.numel()
        param_dict[name] = (num_params, tuple(param.shape))
        total += num_params
    return total, param_dict


def analyze_baseline():
    """Analyze Baseline LSTM architecture"""
    config = BaseConfig()
    model = BaselineLLM(config)

    print("="*80)
    print("BASELINE LSTM MODEL ARCHITECTURE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Embedding dim: {config.embed_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num LSTM layers: {config.num_layers}")
    print(f"  Max sequence length: {config.max_seq_length}")

    print(f"\n{'-'*80}")
    print("LAYER-BY-LAYER BREAKDOWN")
    print(f"{'-'*80}")

    # Token Embedding
    print(f"\n1. Token Embedding Layer")
    print(f"   Input: token indices [batch, seq_len]")
    print(f"   Embedding matrix: [{config.vocab_size} x {config.embed_dim}]")
    print(f"   Output: [{config.embed_dim}] per token")
    print(f"   Parameters: {config.vocab_size * config.embed_dim:,}")

    # LSTM
    print(f"\n2. LSTM Layers (stacked x{config.num_layers})")
    print(f"   Input: [{config.embed_dim}] (first layer)")
    print(f"   Hidden state: [{config.hidden_dim}]")
    print(f"   Cell state: [{config.hidden_dim}]")

    # LSTM parameter calculation
    # For each LSTM layer:
    # 4 gates (input, forget, output, cell) × (input_size * hidden_size + hidden_size * hidden_size + bias)

    # Layer 0: input_size = embed_dim
    layer0_params = 4 * (config.embed_dim * config.hidden_dim +
                         config.hidden_dim * config.hidden_dim +
                         2 * config.hidden_dim)  # 2 bias vectors

    # Layers 1, 2: input_size = hidden_dim
    layer_n_params = 4 * (config.hidden_dim * config.hidden_dim +
                          config.hidden_dim * config.hidden_dim +
                          2 * config.hidden_dim)

    total_lstm_params = layer0_params + (config.num_layers - 1) * layer_n_params

    print(f"\n   Layer 0 (input: embed_dim={config.embed_dim}):")
    print(f"     4 gates × (W_ih: [{config.embed_dim} x {config.hidden_dim}]")
    print(f"                 W_hh: [{config.hidden_dim} x {config.hidden_dim}]")
    print(f"                 b_ih: [{config.hidden_dim}]")
    print(f"                 b_hh: [{config.hidden_dim}])")
    print(f"     Parameters: {layer0_params:,}")

    print(f"\n   Layers 1-2 (input: hidden_dim={config.hidden_dim}):")
    print(f"     4 gates × (W_ih: [{config.hidden_dim} x {config.hidden_dim}]")
    print(f"                 W_hh: [{config.hidden_dim} x {config.hidden_dim}]")
    print(f"                 b_ih: [{config.hidden_dim}]")
    print(f"                 b_hh: [{config.hidden_dim}])")
    print(f"     Parameters per layer: {layer_n_params:,}")
    print(f"     Total LSTM parameters: {total_lstm_params:,}")

    # Output projection
    print(f"\n3. Output Projection Layer")
    print(f"   Input: [{config.hidden_dim}]")
    print(f"   Weight matrix: [{config.hidden_dim} x {config.vocab_size}]")
    print(f"   Bias: [{config.vocab_size}]")
    print(f"   Output: [{config.vocab_size}] (logits)")
    output_params = config.hidden_dim * config.vocab_size + config.vocab_size
    print(f"   Parameters: {output_params:,}")

    # Total
    total, param_dict = count_parameters(model)
    print(f"\n{'-'*80}")
    print(f"TOTAL PARAMETERS: {total:,}")
    print(f"{'-'*80}")

    print(f"\nDetailed parameter breakdown:")
    for name, (count, shape) in sorted(param_dict.items()):
        print(f"  {name:40s} {str(shape):30s} {count:>12,} params")

    # Data flow example
    print(f"\n{'-'*80}")
    print("DATA FLOW EXAMPLE (batch_size=16, seq_len=10)")
    print(f"{'-'*80}")
    print(f"Input tokens:           [16, 10]")
    print(f"After embedding:        [16, 10, {config.embed_dim}]")
    print(f"After LSTM:             [16, 10, {config.hidden_dim}]")
    print(f"After output proj:      [16, 10, {config.vocab_size}]")
    print(f"Output (logits):        [16, 10, {config.vocab_size}]")

    return total


def analyze_newllm():
    """Analyze New-LLM architecture"""
    config = NewLLMConfig()
    model = ContextVectorLLM(config)

    print("\n\n" + "="*80)
    print("NEW-LLM (CONTEXT VECTOR PROPAGATION) ARCHITECTURE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Embedding dim: {config.embed_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num FNN layers: {config.num_layers}")
    print(f"  Context vector dim: {config.context_vector_dim}")
    print(f"  Max sequence length: {config.max_seq_length}")

    print(f"\n{'-'*80}")
    print("LAYER-BY-LAYER BREAKDOWN")
    print(f"{'-'*80}")

    # Token Embedding
    print(f"\n1. Token Embedding Layer")
    print(f"   Input: token indices [batch, seq_len]")
    print(f"   Embedding matrix: [{config.vocab_size} x {config.embed_dim}]")
    print(f"   Output: [{config.embed_dim}] per token")
    embed_params = config.vocab_size * config.embed_dim
    print(f"   Parameters: {embed_params:,}")

    # Positional Embedding
    print(f"\n2. Positional Embedding Layer")
    print(f"   Input: position indices [seq_len]")
    print(f"   Embedding matrix: [{config.max_seq_length} x {config.embed_dim}]")
    print(f"   Output: [{config.embed_dim}] per position")
    pos_params = config.max_seq_length * config.embed_dim
    print(f"   Parameters: {pos_params:,}")

    # FNN Layers (sequential processing)
    print(f"\n3. FNN Layers (sequential, x{config.num_layers})")
    print(f"   *** IMPORTANT: Processes each token sequentially ***")
    print(f"   *** Not parallel like standard FNN ***")

    fnn_input_dim = config.embed_dim + config.context_vector_dim

    print(f"\n   For each time step t:")
    print(f"     Input: Token embedding [{config.embed_dim}] + Context [{config.context_vector_dim}]")
    print(f"     Concatenated input: [{fnn_input_dim}]")

    # Layer 0
    layer0_w1 = fnn_input_dim * config.hidden_dim
    layer0_b1 = config.hidden_dim
    layer0_w2 = config.hidden_dim * config.hidden_dim
    layer0_b2 = config.hidden_dim
    layer0_total = layer0_w1 + layer0_b1 + layer0_w2 + layer0_b2

    print(f"\n   Layer 0:")
    print(f"     Linear 1: [{fnn_input_dim} x {config.hidden_dim}] + bias[{config.hidden_dim}]")
    print(f"     ReLU + Dropout")
    print(f"     Linear 2: [{config.hidden_dim} x {config.hidden_dim}] + bias[{config.hidden_dim}]")
    print(f"     Dropout")
    print(f"     Parameters: {layer0_total:,}")

    # Layers 1, 2
    layer_n_w1 = config.hidden_dim * config.hidden_dim
    layer_n_b1 = config.hidden_dim
    layer_n_w2 = config.hidden_dim * config.hidden_dim
    layer_n_b2 = config.hidden_dim
    layer_n_total = layer_n_w1 + layer_n_b1 + layer_n_w2 + layer_n_b2

    print(f"\n   Layers 1-2 (each):")
    print(f"     Linear 1: [{config.hidden_dim} x {config.hidden_dim}] + bias[{config.hidden_dim}]")
    print(f"     ReLU + Dropout")
    print(f"     Linear 2: [{config.hidden_dim} x {config.hidden_dim}] + bias[{config.hidden_dim}]")
    print(f"     Dropout")
    print(f"     Parameters per layer: {layer_n_total:,}")

    total_fnn_params = layer0_total + (config.num_layers - 1) * layer_n_total
    print(f"\n   Total FNN parameters: {total_fnn_params:,}")

    # Output heads
    print(f"\n4. Output Heads (dual)")

    print(f"\n   A. Token Prediction Head:")
    print(f"      Input: [{config.hidden_dim}]")
    print(f"      Weight matrix: [{config.hidden_dim} x {config.vocab_size}]")
    print(f"      Bias: [{config.vocab_size}]")
    print(f"      Output: [{config.vocab_size}] (logits)")
    token_head_params = config.hidden_dim * config.vocab_size + config.vocab_size
    print(f"      Parameters: {token_head_params:,}")

    print(f"\n   B. Context Update Head:")
    print(f"      Input: [{config.hidden_dim}]")
    print(f"      Weight matrix: [{config.hidden_dim} x {config.context_vector_dim}]")
    print(f"      Bias: [{config.context_vector_dim}]")
    print(f"      Output: [{config.context_vector_dim}] (delta)")
    context_head_params = config.hidden_dim * config.context_vector_dim + config.context_vector_dim
    print(f"      Parameters: {context_head_params:,}")

    print(f"\n5. Context Vector Update (no parameters)")
    print(f"   context[t] = context[t-1] + delta[t]")
    print(f"   Additive update, no learnable parameters")

    # Total
    total, param_dict = count_parameters(model)
    print(f"\n{'-'*80}")
    print(f"TOTAL PARAMETERS: {total:,}")
    print(f"{'-'*80}")

    print(f"\nDetailed parameter breakdown:")
    for name, (count, shape) in sorted(param_dict.items()):
        print(f"  {name:40s} {str(shape):30s} {count:>12,} params")

    # Data flow example
    print(f"\n{'-'*80}")
    print("DATA FLOW EXAMPLE (batch_size=16, seq_len=10)")
    print(f"{'-'*80}")
    print(f"Input tokens:           [16, 10]")
    print(f"After token embed:      [16, 10, {config.embed_dim}]")
    print(f"After pos embed:        [16, 10, {config.embed_dim}] (added)")
    print(f"\nSequential processing (for each t=0 to 9):")
    print(f"  Context[t]:           [16, {config.context_vector_dim}]")
    print(f"  Token[t]:             [16, {config.embed_dim}]")
    print(f"  Concat:               [16, {fnn_input_dim}]")
    print(f"  After FNN:            [16, {config.hidden_dim}]")
    print(f"  Token logits[t]:      [16, {config.vocab_size}]")
    print(f"  Context delta[t]:     [16, {config.context_vector_dim}]")
    print(f"  Updated context:      [16, {config.context_vector_dim}]")
    print(f"\nFinal output:")
    print(f"  All token logits:     [16, 10, {config.vocab_size}]")
    print(f"  Context trajectory:   [16, 10, {config.context_vector_dim}]")

    return total


def compare_models():
    """Compare both models"""
    print("\n\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    # Get actual parameter counts from models
    baseline_config = BaseConfig()
    baseline_model = BaselineLLM(baseline_config)
    baseline_params, _ = count_parameters(baseline_model)

    newllm_config = NewLLMConfig()
    newllm_model = ContextVectorLLM(newllm_config)
    newllm_params, _ = count_parameters(newllm_model)

    print(f"\nParameter counts:")
    print(f"  Baseline LSTM:  {baseline_params:>10,} parameters")
    print(f"  New-LLM:        {newllm_params:>10,} parameters")
    print(f"  Difference:     {baseline_params - newllm_params:>10,} parameters")
    print(f"  Ratio:          New-LLM uses {newllm_params/baseline_params*100:.1f}% of Baseline parameters")

    print(f"\nKey architectural differences:")
    print(f"\n  Baseline LSTM:")
    print(f"    - Uses recurrent hidden states (h, c)")
    print(f"    - Hidden state dim: {baseline_config.hidden_dim}")
    print(f"    - {baseline_config.num_layers} stacked LSTM layers")
    print(f"    - Processes sequence in parallel (PyTorch optimization)")
    print(f"    - Context stored in hidden/cell states")

    print(f"\n  New-LLM:")
    print(f"    - Uses explicit context vector")
    print(f"    - Context vector dim: {newllm_config.context_vector_dim} ({baseline_config.hidden_dim//newllm_config.context_vector_dim}x smaller than LSTM hidden)")
    print(f"    - {newllm_config.num_layers} FNN layers (processed sequentially)")
    print(f"    - Sequential processing (for t in seq_len)")
    print(f"    - Context stored explicitly and additively updated")

    lstm_context_size = 2 * baseline_config.hidden_dim  # h and c
    newllm_context_size = newllm_config.context_vector_dim
    print(f"\nContext representation:")
    print(f"  LSTM: 2 vectors × {baseline_config.hidden_dim} dim = {lstm_context_size} numbers per time step")
    print(f"  New-LLM: 1 vector × {newllm_config.context_vector_dim} dim = {newllm_context_size} numbers per time step")
    print(f"  Ratio: New-LLM uses {newllm_context_size/lstm_context_size*100:.1f}% of LSTM context capacity")


def main():
    analyze_baseline()
    analyze_newllm()
    compare_models()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
