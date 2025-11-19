# New-LLM Architecture

**Now powered by HuggingFace Transformers** - combines New-LLM's innovative O(1) memory architecture with HuggingFace's battle-tested ecosystem.

---

## 🔧 HuggingFace Integration

New-LLM is now fully integrated with HuggingFace Transformers:

```python
# Model definition (src/models/new_llm_hf.py)
from transformers import PreTrainedModel, PretrainedConfig

class NewLLMForCausalLM(PreTrainedModel):
    """HuggingFace-compatible New-LLM wrapper"""

    def __init__(self, config: NewLLMConfig):
        super().__init__(config)
        self.model = ContextVectorLLM(config)  # Core New-LLM architecture

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.model(input_ids)
        # Standard HF CausalLM loss computation
        # ...
```

**Benefits of HuggingFace Integration**:
- ✅ Automatic tokenizer saving/loading
- ✅ Built-in text generation (`model.generate()`)
- ✅ Trainer API for training
- ✅ Compatible with all HF utilities

The **core New-LLM architecture** (`ContextVectorLLM`) remains unchanged - we just wrap it in HuggingFace interfaces.

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
Token 1 → [Embed + Context(0)] → FNN → [Token Pred 1, Context Update 1]
                                                ↓
Token 2 → [Embed + Context(1)] → FNN → [Token Pred 2, Context Update 2]
                                                ↓
Token 3 → [Embed + Context(2)] → FNN → [Token Pred 3, Context Update 3]
                                                ↓
                                              ...
```

**Key idea**: The context vector carries information across positions, and its updates are learned indirectly by optimizing next-token prediction loss.

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

- **Fixed-size context vector** (e.g., 512 dimensions)
- **Token embeddings** (reused for each token, not stored)
- **FNN parameters** (fixed regardless of sequence)
- **Gated updates** (forget gate, input gate)
- **LayerNorm** (applied per-step)

### ❌ PROHIBITED

- **Positional embeddings** that limit max sequence length
- **Storing all hidden states** (would grow with sequence length)
- **Any operation that depends on max_seq_length parameter**
- **Attention mechanisms** (defeats the purpose)

---

## Key Components

### 1. Context Vector

```python
context = torch.zeros(batch_size, context_vector_dim)  # Fixed size
```

- Carries all contextual information
- Updated at each time step
- Normalized with LayerNorm

### 2. Gated Update Mechanism

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

### 3. Feedforward Network (FNN)

```python
hidden = FNN([token_embed, context])
logits = output_layer(hidden)
```

- Processes token + context to predict next token
- Multiple layers (optimized: 4-5 layers)
- No attention mechanism

---

## Comparison with Transformers

| Feature | Transformer | New-LLM |
|---------|-------------|---------|
| **Attention** | Multi-head self-attention | None |
| **Memory** | O(n²) | O(1) |
| **Positional Encoding** | Required | Not needed |
| **Context** | All previous tokens | Fixed-size vector |
| **Sequence Length** | Limited by memory | Unlimited |
| **Training Speed** | Slower (attention) | Faster (FNN only) |

---

## Advantages

1. **Unlimited sequence length**: Not constrained by memory
2. **Faster training**: No attention computation
3. **Simpler architecture**: Fewer components
4. **Memory efficient**: O(1) memory usage

---

## Trade-offs

1. **Information compression**: Context vector must compress all information
2. **No explicit attention**: Cannot directly attend to specific positions
3. **Sequential processing**: Cannot parallelize across time (but can parallelize across batch)

---

## Model Sizes

### Baseline (Layer 1)

| Parameter | Value |
|-----------|-------|
| Layers | 1 |
| Context dim | 256 |
| Embed dim | 256 |
| Hidden dim | 512 |
| **Total params** | **~1.4M** |

### Optimized (Layer 4-5)

| Parameter | Value |
|-----------|-------|
| Layers | 4-5 |
| Context dim | 256 |
| Embed dim | 256 |
| Hidden dim | 512 |
| **Total params** | **~2.5-2.6M** |

### Advanced

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
- `src/models/context_vector_llm.py` - Implementation
- `experiments/` - Experimental results
