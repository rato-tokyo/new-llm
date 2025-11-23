# New-LLM Project Guidelines

## Distribution Regularization - Critical Design Specification

### Philosophy: Token-wise Normalization

**IMPORTANT**: Distribution regularization MUST be applied **per-token** (online), not across all tokens (batch).

### Why Token-wise?

1. **Theoretical Correctness**: Each token processes sequentially in a language model
2. **Online Learning**: Can't wait for all 92,047 tokens to compute statistics
3. **Prevents Trivial Solutions**: Batch normalization allows identity mapping convergence
4. **Scalability**: Works with any sequence length

### Implementation Method: Exponential Moving Average (EMA)

Use running statistics updated per token:

```python
# For each token t:
running_mean = momentum * running_mean + (1 - momentum) * context[t].mean()
running_var = momentum * running_var + (1 - momentum) * context[t].var()

# Penalize deviation from N(0,1)
dist_loss = (running_mean ** 2) + ((running_var - 1.0) ** 2)
```

**Parameters**:
- Momentum: 0.99 (typical for EMA)
- Update: Every token during forward pass
- Scope: Per-dimension statistics across tokens

### Object-Oriented Design for Clean Implementation

#### Current Problem: Scattered Logic
- Distribution regularization mixed with training loop
- Forward pass doesn't handle normalization internally
- Statistics calculation exposed to caller

#### Proposed Solution: Layer-based Encapsulation

```python
class CVFPLayer(nn.Module):
    """
    Context update layer with built-in distribution tracking
    """
    def __init__(self, context_dim, embed_dim, hidden_dim, use_dist_reg=True):
        super().__init__()
        self.fnn = nn.Linear(...)

        # EMA statistics (if distribution regularization enabled)
        if use_dist_reg:
            self.register_buffer('running_mean', torch.zeros(context_dim))
            self.register_buffer('running_var', torch.ones(context_dim))
            self.momentum = 0.99

    def forward(self, context, token_embed):
        # Update context
        new_context = self._update_context(context, token_embed)

        # Update running statistics (hidden from caller)
        if self.training and self.use_dist_reg:
            self._update_statistics(new_context)

        return new_context

    def _update_statistics(self, context):
        """Hidden implementation detail"""
        with torch.no_grad():
            batch_mean = context.mean(dim=0)
            batch_var = context.var(dim=0, unbiased=False)

            self.running_mean = self.momentum * self.running_mean + \
                               (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + \
                              (1 - self.momentum) * batch_var

    def get_distribution_loss(self):
        """Calculate loss based on accumulated statistics"""
        mean_penalty = (self.running_mean ** 2).mean()
        var_penalty = ((self.running_var - 1.0) ** 2).mean()
        return mean_penalty + var_penalty
```

#### Benefits of This Design:

1. **Encapsulation**: Statistics tracking hidden inside layer
2. **Automatic Updates**: No manual `mean()` / `var()` in training loop
3. **Clean Interface**: Caller just does `context = layer(context, token)`
4. **Testable**: Easy to test distribution tracking separately
5. **Reusable**: Same pattern for other normalization schemes

### Migration Strategy

1. Create `CVFPLayer` class in `src/models/layers.py`
2. Refactor `new_llm_residual.py` to use `CVFPLayer`
3. Update `phase1.py` to use `model.get_distribution_loss()`
4. Remove manual statistics calculation from training loop

### Expected Improvements

- **Prevent Identity Mapping**: Token-wise normalization forces diversity
- **Better Convergence**: Running statistics provide stable gradients
- **Cleaner Code**: ~50 lines removed from training loop
- **Easier Debugging**: Layer-level loss inspection

## Code Quality Standards

### Principles

1. **Encapsulation**: Hide implementation details in layers/modules
2. **Single Responsibility**: Each class does one thing well
3. **Clean Interfaces**: Minimal parameters, clear return values
4. **Self-Documenting**: Method names explain purpose

### Anti-Patterns to Avoid

- ❌ Manual statistics calculation in training loops
- ❌ Exposing internal state (running_mean, running_var) to callers
- ❌ Mixing forward pass logic with loss calculation
- ❌ Copy-pasted code for train/eval modes

### Preferred Patterns

- ✅ Layer classes handle their own statistics
- ✅ Properties/methods for loss retrieval
- ✅ `nn.Module` buffers for persistent state
- ✅ Automatic train/eval mode handling via `self.training`

---

Last Updated: 2025-11-23
