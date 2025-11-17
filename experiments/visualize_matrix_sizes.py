"""
Visualize matrix sizes and operations in detail

Shows exactly what size matrices are multiplied at each layer.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import BaseConfig, NewLLMConfig


def print_matrix_op(description, input_size, weight_size, bias_size, output_size):
    """Print a matrix operation with sizes"""
    print(f"  {description}")
    print(f"    Input vector:   [{input_size:>4}]")
    print(f"    Weight matrix:  [{weight_size[0]:>4} x {weight_size[1]:>4}]")
    if bias_size:
        print(f"    Bias vector:    [{bias_size:>4}]")
    print(f"    Output vector:  [{output_size:>4}]")
    print(f"    → Formula: output[{output_size}] = W[{weight_size[0]}x{weight_size[1]}] @ input[{input_size}] + bias[{bias_size if bias_size else output_size}]")
    params = weight_size[0] * weight_size[1] + (bias_size if bias_size else 0)
    print(f"    → Parameters: {params:,}")
    print()


def analyze_baseline_lstm():
    """Analyze Baseline LSTM with detailed matrix sizes"""
    config = BaseConfig()

    print("=" * 80)
    print("BASELINE LSTM - DETAILED MATRIX OPERATIONS")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  vocab_size = {config.vocab_size}")
    print(f"  embed_dim = {config.embed_dim}")
    print(f"  hidden_dim = {config.hidden_dim}")
    print(f"  num_layers = {config.num_layers}")

    print("\n" + "-" * 80)
    print("LAYER-BY-LAYER MATRIX OPERATIONS")
    print("-" * 80)

    # Token Embedding
    print("\n[1] TOKEN EMBEDDING LAYER")
    print("─" * 80)
    print(f"  Lookup table (no matrix multiplication)")
    print(f"    Embedding matrix: [{config.vocab_size:>4} x {config.embed_dim:>4}]")
    print(f"    Input: token_id (integer)")
    print(f"    Output: embedding vector [{config.embed_dim}]")
    print(f"    Parameters: {config.vocab_size * config.embed_dim:,}")
    print()

    # LSTM Layer 0
    print("\n[2] LSTM LAYER 0")
    print("─" * 80)
    print(f"  Input dim: {config.embed_dim} (from embedding)")
    print(f"  Hidden dim: {config.hidden_dim}")
    print()
    print("  LSTM has 4 gates (input, forget, output, cell)")
    print("  Each gate has 2 weight matrices + 2 bias vectors:")
    print()

    # W_ih (input to hidden)
    print("  W_ih (Input to Hidden) - for all 4 gates combined:")
    print(f"    Weight matrix:  [{config.hidden_dim * 4:>4} x {config.embed_dim:>4}]")
    print(f"                  = [{config.hidden_dim} x {config.embed_dim}] × 4 gates")
    print(f"    Bias vector:    [{config.hidden_dim * 4:>4}]")
    w_ih_params = config.hidden_dim * 4 * config.embed_dim + config.hidden_dim * 4
    print(f"    Parameters: {w_ih_params:,}")
    print()

    # W_hh (hidden to hidden)
    print("  W_hh (Hidden to Hidden) - for all 4 gates combined:")
    print(f"    Weight matrix:  [{config.hidden_dim * 4:>4} x {config.hidden_dim:>4}]")
    print(f"                  = [{config.hidden_dim} x {config.hidden_dim}] × 4 gates")
    print(f"    Bias vector:    [{config.hidden_dim * 4:>4}]")
    w_hh_params = config.hidden_dim * 4 * config.hidden_dim + config.hidden_dim * 4
    print(f"    Parameters: {w_hh_params:,}")
    print()

    layer0_total = w_ih_params + w_hh_params
    print(f"  Total Layer 0: {layer0_total:,} parameters")
    print()

    print("  At each time step t:")
    print(f"    x[t]:     [{config.embed_dim}]      (input)")
    print(f"    h[t-1]:   [{config.hidden_dim}]      (previous hidden state)")
    print(f"    c[t-1]:   [{config.hidden_dim}]      (previous cell state)")
    print(f"    →")
    print(f"    h[t]:     [{config.hidden_dim}]      (new hidden state)")
    print(f"    c[t]:     [{config.hidden_dim}]      (new cell state)")

    # LSTM Layers 1, 2
    print("\n[3] LSTM LAYERS 1-2 (each)")
    print("─" * 80)
    print(f"  Input dim: {config.hidden_dim} (from previous LSTM layer)")
    print(f"  Hidden dim: {config.hidden_dim}")
    print()

    # W_ih for layers 1-2
    print("  W_ih (Input to Hidden) - for all 4 gates combined:")
    print(f"    Weight matrix:  [{config.hidden_dim * 4:>4} x {config.hidden_dim:>4}]")
    print(f"    Bias vector:    [{config.hidden_dim * 4:>4}]")
    w_ih_params_n = config.hidden_dim * 4 * config.hidden_dim + config.hidden_dim * 4
    print(f"    Parameters: {w_ih_params_n:,}")
    print()

    # W_hh for layers 1-2
    print("  W_hh (Hidden to Hidden) - for all 4 gates combined:")
    print(f"    Weight matrix:  [{config.hidden_dim * 4:>4} x {config.hidden_dim:>4}]")
    print(f"    Bias vector:    [{config.hidden_dim * 4:>4}]")
    w_hh_params_n = config.hidden_dim * 4 * config.hidden_dim + config.hidden_dim * 4
    print(f"    Parameters: {w_hh_params_n:,}")
    print()

    layer_n_total = w_ih_params_n + w_hh_params_n
    print(f"  Total per layer: {layer_n_total:,} parameters")
    print(f"  Total for layers 1-2: {layer_n_total * 2:,} parameters")

    # Output Projection
    print("\n[4] OUTPUT PROJECTION LAYER")
    print("─" * 80)
    print_matrix_op(
        "Project hidden state to vocabulary",
        config.hidden_dim,
        (config.vocab_size, config.hidden_dim),
        config.vocab_size,
        config.vocab_size
    )

    # Total
    total_lstm = layer0_total + layer_n_total * 2
    total = config.vocab_size * config.embed_dim + total_lstm + config.vocab_size * config.hidden_dim + config.vocab_size
    print("-" * 80)
    print(f"TOTAL PARAMETERS: {total:,}")
    print("-" * 80)


def analyze_new_llm():
    """Analyze New-LLM with detailed matrix sizes"""
    config = NewLLMConfig()

    print("\n\n" + "=" * 80)
    print("NEW-LLM - DETAILED MATRIX OPERATIONS")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  vocab_size = {config.vocab_size}")
    print(f"  embed_dim = {config.embed_dim}")
    print(f"  hidden_dim = {config.hidden_dim}")
    print(f"  num_layers = {config.num_layers}")
    print(f"  context_vector_dim = {config.context_vector_dim}  ★ KEY PARAMETER")
    print(f"  max_seq_length = {config.max_seq_length}")

    print("\n" + "-" * 80)
    print("LAYER-BY-LAYER MATRIX OPERATIONS")
    print("-" * 80)

    # Token Embedding
    print("\n[1] TOKEN EMBEDDING LAYER")
    print("─" * 80)
    print(f"  Embedding matrix: [{config.vocab_size:>4} x {config.embed_dim:>4}]")
    print(f"  Input: token_id (integer)")
    print(f"  Output: embedding vector [{config.embed_dim}]")
    print(f"  Parameters: {config.vocab_size * config.embed_dim:,}")

    # Positional Embedding
    print("\n[2] POSITIONAL EMBEDDING LAYER")
    print("─" * 80)
    print(f"  Embedding matrix: [{config.max_seq_length:>4} x {config.embed_dim:>4}]")
    print(f"  Input: position (integer 0 to {config.max_seq_length-1})")
    print(f"  Output: positional embedding [{config.embed_dim}]")
    print(f"  Parameters: {config.max_seq_length * config.embed_dim:,}")
    print()
    print(f"  Final embedding = token_embed[{config.embed_dim}] + pos_embed[{config.embed_dim}]")

    # Context Vector
    print("\n[3] CONTEXT VECTOR (Initial)")
    print("─" * 80)
    print(f"  Dimension: [{config.context_vector_dim}]")
    print(f"  Initial value: all zeros [0, 0, 0, ..., 0]")
    print(f"  Parameters: 0 (initialized to zero, not learned)")
    print()
    print(f"  ★ This is the key innovation: a {config.context_vector_dim}-dimensional")
    print(f"    vector that carries contextual information across time steps")

    # FNN Layer 0
    print("\n[4] FNN LAYER 0 (processes each time step)")
    print("─" * 80)
    concat_dim = config.embed_dim + config.context_vector_dim
    print(f"  Input preparation:")
    print(f"    Token embedding[{config.embed_dim}] + Context vector[{config.context_vector_dim}]")
    print(f"    → Concatenated input: [{concat_dim}]")
    print()

    print_matrix_op(
        "Linear 1",
        concat_dim,
        (config.hidden_dim, concat_dim),
        config.hidden_dim,
        config.hidden_dim
    )

    print(f"  ReLU activation: [{config.hidden_dim}] → [{config.hidden_dim}]")
    print(f"  Dropout (no parameters)")
    print()

    print_matrix_op(
        "Linear 2",
        config.hidden_dim,
        (config.hidden_dim, config.hidden_dim),
        config.hidden_dim,
        config.hidden_dim
    )

    print(f"  Dropout (no parameters)")
    print()

    layer0_params = concat_dim * config.hidden_dim + config.hidden_dim + \
                    config.hidden_dim * config.hidden_dim + config.hidden_dim
    print(f"  Total Layer 0: {layer0_params:,} parameters")

    # FNN Layers 1-2
    print("\n[5] FNN LAYERS 1-2 (each)")
    print("─" * 80)
    print(f"  Input: [{config.hidden_dim}] (from previous layer)")
    print()

    print_matrix_op(
        "Linear 1",
        config.hidden_dim,
        (config.hidden_dim, config.hidden_dim),
        config.hidden_dim,
        config.hidden_dim
    )

    print(f"  ReLU activation: [{config.hidden_dim}] → [{config.hidden_dim}]")
    print(f"  Dropout (no parameters)")
    print()

    print_matrix_op(
        "Linear 2",
        config.hidden_dim,
        (config.hidden_dim, config.hidden_dim),
        config.hidden_dim,
        config.hidden_dim
    )

    print(f"  Dropout (no parameters)")
    print()

    layer_n_params = config.hidden_dim * config.hidden_dim + config.hidden_dim + \
                     config.hidden_dim * config.hidden_dim + config.hidden_dim
    print(f"  Total per layer: {layer_n_params:,} parameters")
    print(f"  Total for layers 1-2: {layer_n_params * 2:,} parameters")

    # Output heads
    print("\n[6] OUTPUT HEADS (Dual)")
    print("─" * 80)
    print(f"  Input: [{config.hidden_dim}] (from FNN)")
    print()

    print("  A. TOKEN PREDICTION HEAD:")
    print_matrix_op(
        "  Project to vocabulary",
        config.hidden_dim,
        (config.vocab_size, config.hidden_dim),
        config.vocab_size,
        config.vocab_size
    )

    print("  B. CONTEXT UPDATE HEAD:")
    print_matrix_op(
        "  Project to context delta",
        config.hidden_dim,
        (config.context_vector_dim, config.hidden_dim),
        config.context_vector_dim,
        config.context_vector_dim
    )

    # Context update
    print("\n[7] CONTEXT VECTOR UPDATE (no parameters)")
    print("─" * 80)
    print(f"  context[t] = context[t-1] + context_delta[t]")
    print(f"    [{config.context_vector_dim}]  =  [{config.context_vector_dim}]  +  [{config.context_vector_dim}]")
    print(f"  Simple vector addition, no learnable parameters")
    print()
    print(f"  ★ The context vector accumulates information additively")
    print(f"    as we process each token in the sequence")

    # Summary
    print("\n" + "=" * 80)
    print("CONTEXT VECTOR DETAILS")
    print("=" * 80)
    print(f"\nContext Vector Dimension: {config.context_vector_dim}")
    print(f"\nAt each time step t:")
    print(f"  1. Concatenate: token_embed[{config.embed_dim}] + context[{config.context_vector_dim}] → [{concat_dim}]")
    print(f"  2. FNN processes: [{concat_dim}] → [{config.hidden_dim}]")
    print(f"  3. Predict token: [{config.hidden_dim}] → [{config.vocab_size}] (logits)")
    print(f"  4. Update context: [{config.hidden_dim}] → [{config.context_vector_dim}] (delta)")
    print(f"  5. Add to context: context[t] = context[t-1] + delta")
    print()
    print(f"Context capacity comparison:")
    print(f"  LSTM: 2 vectors × {config.hidden_dim} dim = {2 * config.hidden_dim} numbers")
    print(f"  New-LLM: 1 vector × {config.context_vector_dim} dim = {config.context_vector_dim} numbers")
    print(f"  Ratio: {config.context_vector_dim / (2 * config.hidden_dim) * 100:.1f}%")

    # Total
    total_fnn = layer0_params + layer_n_params * 2
    total = config.vocab_size * config.embed_dim + \
            config.max_seq_length * config.embed_dim + \
            total_fnn + \
            config.vocab_size * config.hidden_dim + config.vocab_size + \
            config.context_vector_dim * config.hidden_dim + config.context_vector_dim

    print("\n" + "-" * 80)
    print(f"TOTAL PARAMETERS: {total:,}")
    print("-" * 80)


def main():
    analyze_baseline_lstm()
    analyze_new_llm()

    print("\n\n" + "=" * 80)
    print("SUMMARY: KEY MATRIX SIZES")
    print("=" * 80)

    config_base = BaseConfig()
    config_new = NewLLMConfig()

    print("\nBaseline LSTM - Largest Matrices:")
    print(f"  LSTM Layer 0 W_ih: [{config_base.hidden_dim * 4} x {config_base.embed_dim}] = {config_base.hidden_dim * 4 * config_base.embed_dim:,} params")
    print(f"  LSTM Layer 0 W_hh: [{config_base.hidden_dim * 4} x {config_base.hidden_dim}] = {config_base.hidden_dim * 4 * config_base.hidden_dim:,} params")
    print(f"  Output projection: [{config_base.vocab_size} x {config_base.hidden_dim}] = {config_base.vocab_size * config_base.hidden_dim:,} params")

    print("\nNew-LLM - Largest Matrices:")
    concat_dim = config_new.embed_dim + config_new.context_vector_dim
    print(f"  FNN Layer 0 W1:    [{config_new.hidden_dim} x {concat_dim}] = {config_new.hidden_dim * concat_dim:,} params")
    print(f"  FNN Layer 1-2 W1:  [{config_new.hidden_dim} x {config_new.hidden_dim}] = {config_new.hidden_dim * config_new.hidden_dim:,} params")
    print(f"  Output projection: [{config_new.vocab_size} x {config_new.hidden_dim}] = {config_new.vocab_size * config_new.hidden_dim:,} params")
    print(f"  Context update:    [{config_new.context_vector_dim} x {config_new.hidden_dim}] = {config_new.context_vector_dim * config_new.hidden_dim:,} params")

    print("\nContext Vector:")
    print(f"  Dimension: {config_new.context_vector_dim}")
    print(f"  Initial value: zero vector [0, 0, ..., 0] ({config_new.context_vector_dim} zeros)")
    print(f"  Update: additive (no parameters)")
    print(f"  Stores: contextual information from all previous tokens")


if __name__ == "__main__":
    main()
