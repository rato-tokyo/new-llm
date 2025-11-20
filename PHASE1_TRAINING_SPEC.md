# Phase 1 Training Specification

**Date**: 2025-11-21
**Status**: Core specification for New-LLM Phase 1 training

---

## 🎯 Phase 1 Objective

**Learn model weights that produce stable fixed-point context vectors for each token in a sequence.**

The model should learn to create a function `f(context, token)` such that:
```
context_fixed = f(context_fixed, token)  (fixed-point condition)
```

---

## 🔑 Critical Understanding: Phase 1 MUST Train Weights

### ❌ WRONG: Fixed-point search without training

```python
# THIS IS WRONG - No training happening!
model.eval()
with torch.no_grad():
    for iteration in range(max_iterations):
        context = model(token, context)
        if converged:
            break
# Weights remain random - will not converge for Sequential!
```

### ✅ CORRECT: Train weights to reach fixed points

```python
# THIS IS CORRECT - Training weights
model.train()
for epoch in range(epochs):
    for sample in dataset:
        optimizer.zero_grad()

        # Fixed-point iteration
        context = zeros()
        for iteration in range(max_iterations):
            context_new = model(token, context)

        # Loss: Self-consistency (fixed-point condition)
        loss = MSE(context_new, context.detach())

        loss.backward()  # Compute gradients
        optimizer.step()  # UPDATE WEIGHTS!
```

**Key difference**: `optimizer.step()` updates weights based on fixed-point loss.

---

## 📐 Training Architecture

### Two-Stage Process

```
Phase 1: Context Vector Fixed-Point Learning (THIS DOCUMENT)
    ↓
    Learn weights that produce stable context vectors
    ↓
Phase 2: Token Prediction Learning
    ↓
    Use Phase 1 weights to predict next tokens
```

### Phase 1 Training Loop

```python
def phase1_train_sample(model, optimizer, input_ids, config):
    """Train model on one sample to reach fixed points"""

    model.train()  # Training mode

    token_embeds = model.token_embedding(input_ids)
    total_loss = 0.0

    for t in range(seq_len):
        current_token = token_embeds[t]

        # Initialize context
        context = torch.zeros(context_dim)

        # Fixed-point iteration with training
        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Forward pass through model
            context_new = model.update_context(current_token, context)

            # Fixed-point loss (self-consistency)
            loss = F.mse_loss(context_new, context.detach())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update context for next iteration
            context = context_new.detach()

            # Check convergence
            if loss.item() < tolerance:
                break

        total_loss += loss.item()

    return total_loss / seq_len
```

---

## 🧮 Loss Function: Fixed-Point Self-Consistency

### Mathematical Definition

For each token at position `t`:

```
Loss = ||f(context, token_t) - context||²

where:
- f(·, ·) is the model's context update function
- context is the current context vector
- token_t is the token embedding at position t
```

**Goal**: Minimize the difference between input context and output context (fixed-point condition).

### Why This Loss Works

When the loss is minimized:
```
f(context*, token) ≈ context*
```

This means `context*` is a fixed point of the function `f(·, token)`.

By training on many tokens, the model learns to:
1. Produce stable fixed points for any token
2. Create meaningful representations (different tokens → different fixed points)

---

## 🔄 Training Iterations

### Inner Loop: Fixed-Point Iteration (per token)

```python
for iteration in range(max_iterations):
    context_new = model(token, context)
    loss = MSE(context_new, context.detach())

    loss.backward()
    optimizer.step()  # Update weights

    context = context_new.detach()

    if loss < tolerance:
        break  # Converged for this token
```

**Purpose**: Train weights so that context converges to a fixed point.

### Outer Loop: Sequential Tokens (per sample)

```python
for t in range(seq_len):
    token = token_embeds[t]

    context = zeros()  # Reset for each token

    # Train to reach fixed point for this token
    for iteration in range(max_iterations):
        ...  # Inner loop above
```

**Purpose**: Learn fixed points for all tokens in the sequence.

### Epoch Loop: Multiple Passes (per dataset)

```python
for epoch in range(num_epochs):
    for sample in dataset:
        loss = train_sample(model, sample)

    print(f"Epoch {epoch}: avg loss = {loss}")
```

**Purpose**: Refine weights across entire dataset.

---

## 📊 Convergence Metrics

### Per-Token Metrics

For each token, track:
1. **Convergence status**: Did context reach fixed point? (loss < tolerance)
2. **Iterations to converge**: How many iterations needed?
3. **Final loss**: `||context_new - context||²`

### Per-Sample Metrics

For each sample (dialogue), track:
1. **Convergence rate**: Percentage of tokens that converged
2. **Average iterations**: Mean iterations across converged tokens
3. **Average loss**: Mean fixed-point loss

### Success Criterion

**Phase 1 is successful when**:
```
Convergence rate ≥ 95% across all samples
```

If convergence rate < 95%, need to:
- Train for more epochs
- Adjust architecture (num_layers, context_dim, hidden_dim)
- Adjust hyperparameters (learning_rate, tolerance)

---

## 🏗️ Architecture Comparison

### Sequential Architecture

**Structure**:
```python
hidden1 = FNN1([token, context])
hidden2 = FNN2(hidden1)
context_new = update(hidden2, context)
```

**Characteristics**:
- Deep composition of transformations
- Context updated only at final layer
- **Requires training** to create contraction mapping

**Expected behavior**:
- ❌ Fails without training (random weights → expansion)
- ✅ Should succeed with training (optimized weights → contraction)

### Layer-wise Architecture

**Structure**:
```python
# Layer 1
hidden1 = FNN1([token, context])
context = update1(hidden1, context)

# Layer 2
hidden2 = FNN2([token, context'])
context = update2(hidden2, context)
```

**Characteristics**:
- Gradual updates at each layer
- Multiple small transformations
- **More stable** even without training (gated updates + LayerNorm)

**Expected behavior**:
- ✅ Works partially without training (~75% convergence)
- ✅ Should improve significantly with training (→ 95%+ convergence)

---

## 🧪 Experimental Validation

### Hypothesis

**With proper Phase 1 training**:

| Architecture | Without Training | With Training |
|--------------|-----------------|---------------|
| Sequential | 0% convergence | **Should improve significantly** |
| Layer-wise | 75% convergence | **Should reach 95%+** |

### Test Procedure

1. **Clear cache**: Remove old untrained results
2. **Train Sequential**: Phase 1 training with fixed-point loss
3. **Train Layer-wise**: Phase 1 training with fixed-point loss
4. **Compare**: Convergence rates, iterations, training time

---

## 💻 Implementation Details

### Hyperparameters

```python
# Phase 1 Training
phase1_epochs = 5                    # Number of training epochs
phase1_learning_rate = 0.0001        # Learning rate for weight updates
phase1_max_iterations = 200          # Max iterations per token
phase1_convergence_threshold = 1e-2  # Tolerance for fixed-point
phase1_warmup_iterations = 10        # Warmup before checking convergence
```

### Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=phase1_learning_rate
)
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

Prevents exploding gradients during fixed-point iteration.

---

## 🔍 Why Previous Experiments Failed

### Previous Misunderstanding

**What we did**:
```python
model.eval()
with torch.no_grad():
    context = model.get_fixed_point_context(token)
    # No weight updates!
```

**Result**:
- Sequential: 0% convergence (random weights)
- Layer-wise: 75% convergence (architectural stability)

### Why Sequential Failed

With random weights, the function:
```
f(x) = update(W3 @ ReLU(W2 @ ReLU(W1 @ [token, x])))
```

is **not guaranteed** to be a contraction mapping (Lipschitz constant k > 1).

Therefore, fixed-point iteration diverges.

### Why Layer-wise Partially Succeeded

Gated updates:
```
context_new = forget * context_old + input * delta
where forget, input ∈ [0, 1]
```

create **implicit damping** even with random weights, leading to partial convergence.

---

## 🎓 Theoretical Background

### Fixed-Point Iteration

For iteration `x_{n+1} = f(x_n)` to converge to fixed point `x*`:

**Banach Fixed-Point Theorem** requires:
```
||f(x) - f(y)|| ≤ k ||x - y||  where k < 1  (contraction mapping)
```

### How Training Creates Contraction

**Training objective**:
```
minimize ||f(x, token) - x||²
```

Forces the model to learn weights such that:
- `f(x*, token) = x*` for some `x*` (fixed point exists)
- Small changes in `x` lead to small changes in `f(x)` (contraction)

This is achieved through gradient descent on the fixed-point loss.

---

## 📝 Summary

### Key Points

1. **Phase 1 MUST train weights** - not just search for fixed points
2. **Loss function**: `MSE(context_new, context)` (self-consistency)
3. **Sequential requires training** to create contraction mapping
4. **Layer-wise is more stable** but still benefits from training
5. **Success criterion**: 95%+ convergence rate

### Training vs Inference

| Phase | Mode | Weights | Purpose |
|-------|------|---------|---------|
| **Phase 1 Training** | `model.train()` | Updated | Learn to reach fixed points |
| **Phase 1 Cache** | `model.eval()` | Fixed | Store fixed-point contexts |
| **Phase 2 Training** | `model.train()` | Updated | Learn token prediction |
| **Phase 2 Inference** | `model.eval()` | Fixed | Generate text |

### Next Steps

1. ✅ Implement Phase 1 training code
2. ✅ Train both architectures (Sequential, Layer-wise)
3. ✅ Compare convergence rates
4. ✅ Achieve 95%+ convergence before Phase 2

---

**This specification is the foundation of New-LLM's two-phase training approach.**

---

**End of Specification**
