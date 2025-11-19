# New-LLM Architecture

**Pure PyTorch implementation** - O(1) memory architecture with reconstruction learning.

---

## Core Design Principles

### 🎯 Fixed Memory Usage (O(1))

**Critical**: New-LLM maintains **constant memory** regardless of sequence length.

| Architecture | Memory Complexity | Sequence Constraint |
|--------------|------------------|---------------------|
| **Transformer** | O(n²) | Limited by attention matrix size |
| **New-LLM** | **O(1)** | **Unlimited** (fixed context vector) |

This is the fundamental advantage over Transformers:
- **Transformer**: Memory grows quadratically with sequence length
- **New-LLM**: Memory stays constant (fixed-size context vector)

---

## Architecture Overview

```
Token 1 → [Embed + Context(0)] → FNN → [Token Pred 1, Context Update 1, Reconstruct 1]
                                                ↓
Token 2 → [Embed + Context(1)] → FNN → [Token Pred 2, Context Update 2, Reconstruct 2]
                                                ↓
Token 3 → [Embed + Context(2)] → FNN → [Token Pred 3, Context Update 3, Reconstruct 3]
                                                ↓
                                              ...
```

**Key idea**: The context vector carries information across positions. It learns to compress `[previous_context + current_token]` through reconstruction learning.

---

## Reconstruction Learning

### What is it?

At each timestep `t`, the context vector learns to compress:
- **Input**: `[context[t-1], token_embed[t]]` (512 dimensions)
- **Output**: `context[t]` (256 dimensions)

Then a decoder reconstructs the original 512 dimensions:
- **Decoder**: `context[t]` (256 dims) → `reconstructed` (512 dims)
- **Loss**: MSE between `reconstructed` and `[context[t-1], token_embed[t]]`

### Why Reconstruction Learning?

**Problem with previous approach**: We didn't know what the "correct" context vector should represent.

**Solution**: Define it explicitly:
> The context vector should be a compressed representation of "previous context + current token"

This is similar to an **autoencoder**:
- **Encoder**: FNN layers compress 512 → 256 dimensions
- **Decoder**: Linear layers reconstruct 256 → 512 dimensions

See `RECONSTRUCTION_LEARNING.md` for detailed explanation with examples.

---

## No Positional Embeddings

### Why Not?

```python
# ❌ BAD: Learned positional embeddings
self.position_embedding = nn.Embedding(max_seq_length, embed_dim)

# Problems:
# 1. Can only handle sequences up to max_seq_length
# 2. Memory usage tied to sequence length
# 3. Violates O(1) memory principle
```

### The Right Way

```python
# ✅ GOOD: Position information from sequential processing
for t in range(seq_len):  # Can be ANY length
    context[t] = update(context[t-1], input[t])
    # Position information emerges naturally from order
```

Like RNN/LSTM, position information is **implicitly learned** through sequential processing order.

---

## Implementation Rules

### ✅ ALLOWED

- **Fixed-size context vector** (e.g., 256 dimensions)
- **Token embeddings** (reused for each token, not stored)
- **FNN parameters** (fixed regardless of sequence)
- **Gated updates** (forget gate, input gate)
- **LayerNorm** (applied per-step)
- **Context decoder** (for reconstruction learning)

### ❌ PROHIBITED

- **Positional embeddings** that limit max sequence length
- **Storing all hidden states** (would grow with sequence length)
- **Any operation that depends on max_seq_length parameter**
- **Attention mechanisms** (defeats the purpose)

---

## Key Components

### 1. Context Vector

```python
context = torch.zeros(batch_size, context_vector_dim)  # Fixed size (256 dims)
```

- Carries all contextual information
- Updated at each time step
- Normalized with LayerNorm

### 2. Context Decoder (NEW)

```python
self.context_decoder = nn.Sequential(
    nn.Linear(256, 512),  # context_dim → (context_dim + embed_dim)
    nn.ReLU(),
    nn.Linear(512, 512),
)
```

- Reconstructs original `[context, token_embed]` from compressed context
- Trained with MSE loss
- Enables self-supervised learning

### 3. Gated Update Mechanism

```python
forget_gate = sigmoid(W_f @ [token_embed, context])
input_gate = sigmoid(W_i @ [token_embed, context])

context_delta = tanh(W_c @ hidden)
context = forget_gate * context + input_gate * context_delta
context = LayerNorm(context)
```

- **Forget gate**: Controls how much to retain from previous context
- **Input gate**: Controls how much new information to add
- **LayerNorm**: Stabilizes training

### 4. Feedforward Network (FNN)

```python
hidden = FNN([token_embed, context])
logits = output_layer(hidden)
```

- Processes token + context to predict next token
- Multiple layers (optimized: 4-5 layers)
- No attention mechanism

---

## Dual Loss Training

New-LLM is trained with two losses:

### 1. Token Prediction Loss (Cross-Entropy)

```python
token_loss = F.cross_entropy(logits, labels)
```

Standard next-token prediction loss.

### 2. Reconstruction Loss (MSE)

```python
reconstruction_target = torch.cat([context[t-1], token_embed[t]], dim=-1)
reconstructed = context_decoder(context[t])
reconstruction_loss = F.mse_loss(reconstructed, reconstruction_target)
```

Ensures context vector learns meaningful compression.

### Combined Loss

```python
total_loss = token_loss + context_loss_weight * reconstruction_loss
```

Typically `context_loss_weight = 1.0` for equal importance.

---

## Comparison with Transformers

| Feature | Transformer | New-LLM |
|---------|-------------|---------|
| **Attention** | Multi-head self-attention | None |
| **Memory** | O(n²) | O(1) |
| **Positional Encoding** | Required | Not needed |
| **Context** | All previous tokens | Fixed-size vector |
| **Sequence Length** | Limited by memory | Unlimited |
| **Training** | Next-token prediction | Token prediction + Reconstruction |
| **Teacher Data** | Not needed | Not needed (self-supervised) |

---

## Advantages

1. **Unlimited sequence length**: Not constrained by memory
2. **Self-supervised learning**: No external teacher model needed
3. **Simpler architecture**: Fewer components than Transformer
4. **Memory efficient**: O(1) memory usage
5. **Explicit learning objective**: Context vector has clear purpose

---

## Trade-offs

1. **Information compression**: Context vector must compress all information
2. **No explicit attention**: Cannot directly attend to specific positions
3. **Sequential processing**: Cannot parallelize across time (but can across batch)
4. **Additional decoder overhead**: Reconstruction decoder adds parameters

---

## Model Sizes

### Baseline (Layer 1)

| Parameter | Value |
|-----------|-------|
| Layers | 1 |
| Context dim | 256 |
| Embed dim | 256 |
| Hidden dim | 512 |
| **Total params** | **~2.7M** (including decoder) |

### Optimized (Layer 4-5)

| Parameter | Value |
|-----------|-------|
| Layers | 4-5 |
| Context dim | 256 |
| Embed dim | 256 |
| Hidden dim | 512 |
| **Total params** | **~3.5-3.7M** |

### Advanced (Layer 12)

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Context dim | 512 |
| Embed dim | 256 |
| Hidden dim | 512 |
| **Total params** | **~4.8M** |

---

## See Also

- `README.md` - Project overview
- `RECONSTRUCTION_LEARNING.md` - Detailed explanation of reconstruction learning
- `src/models/context_vector_llm.py` - Implementation
- `train.py` - Training script
