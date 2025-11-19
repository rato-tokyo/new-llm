# GPU Memory Optimization Guide for New-LLM

## 📊 Memory Usage Analysis (2025-11-20)

### Critical Discovery: Vocabulary Size Impact

**GPT-2's vocabulary size (50,257 tokens) causes massive memory consumption.**

## Memory Calculation Formula

### Logits Tensor Memory

```
Memory = batch_size × seq_len × vocab_size × 4 bytes (float32)
```

**Examples**:

| batch_size | seq_len | vocab_size | Memory (GiB) | Fits T4 (22GB)? |
|------------|---------|------------|--------------|-----------------|
| 512        | 512     | 50,257     | 49.08        | ❌ No           |
| 256        | 256     | 50,257     | 12.30        | ⚠️ Tight        |
| 128        | 256     | 50,257     | 6.15         | ⚠️ Maybe        |
| 64         | 256     | 50,257     | 3.08         | ✅ Yes          |
| 32         | 256     | 50,257     | 1.54         | ✅ Yes          |

### Total GPU Memory Components

1. **Logits tensor**: batch×seq×vocab×4 bytes
2. **Context trajectory**: batch×seq×context_dim×4 bytes
3. **Reconstruction targets**: batch×seq×(context_dim+embed_dim)×4 bytes
4. **Model parameters**: ~2-3 GiB (40M params)
5. **Optimizer state (Adam)**: ~2× model params ≈ 4-6 GiB
6. **Intermediate tensors**: ~2-3 GiB
7. **Token embeddings**: batch×seq×embed_dim×4 bytes

**Total**: Logits memory + 8-12 GiB overhead

## 🐛 Memory Leaks Discovered

### Bug 1: Computation Graph Retention

**Problem** (Fixed on 2025-11-20):
```python
# ❌ Bad - retains computation graph
self.reconstruction_targets = reconstruction_targets_tensor

# Issue: Accumulates full computation history across all batches
# Memory grows: batch1 + batch2 + batch3 + ... → 20+ GiB
```

**Solution**:
```python
# ✅ Good - detaches from computation graph
self.reconstruction_targets = reconstruction_targets_tensor.detach()
self.context_history = context_trajectory.detach()
```

**Effect**: `.detach()` sets `requires_grad=False` and breaks gradient flow.

**Memory saved**: ~10-15 GiB (75-85% reduction)

### Bug 2: Fixed Sequence Length

**Problem**: Hardcoded `max_length=512` caused 49 GiB allocation attempt.

**Solution**: Added `--max-length` parameter with default 256.

## 🎯 Recommended Settings

### T4 GPU (16 GB VRAM)

```bash
# Conservative (most stable)
--batch-size 32 --max-length 256 --context-dim 512
Memory: ~3-5 GiB

# Balanced (recommended)
--batch-size 64 --max-length 256 --context-dim 512
Memory: ~5-8 GiB

# Aggressive (risky with GPT-2 vocab)
--batch-size 128 --max-length 256 --context-dim 512
Memory: ~8-12 GiB (may OOM)
```

### L4 GPU (24 GB VRAM)

```bash
# Recommended
--batch-size 128 --max-length 256 --context-dim 512
Memory: ~8-12 GiB

# Maximum
--batch-size 256 --max-length 256 --context-dim 512
Memory: ~15-18 GiB (tight but possible)
```

### A100 GPU (40 GB VRAM)

```bash
# Recommended
--batch-size 512 --max-length 256 --context-dim 512
Memory: ~25-30 GiB
```

## 🔧 Optimization Strategies

### Strategy 1: Reduce Vocabulary Size

**Current**: GPT-2 (50,257 tokens) → 12.3 GiB for batch=256

**Alternative**: Train custom tokenizer (10,000 tokens) → 2.4 GiB for batch=256

**Memory reduction**: 5× smaller

**Implementation**:
```python
# Custom BPE tokenizer with smaller vocab
tokenizer = Tokenizer(models.BPE())
trainer = BpeTrainer(vocab_size=10000, ...)
```

### Strategy 2: Mixed Precision (FP16)

**Memory reduction**: 50%

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits, context = model(input_ids)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Effect**: 12.3 GiB → 6.15 GiB (for batch=256)

### Strategy 3: Gradient Accumulation

**Problem**: Need large effective batch size but limited GPU memory

**Solution**:
```python
accumulation_steps = 4  # Effective batch = 64 × 4 = 256

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit**: Large effective batch with small GPU memory footprint

### Strategy 4: Pre-allocate Tensors (Implemented)

**Before**:
```python
logits_list = []
for t in range(seq_len):
    logits_list.append(logit_t)
logits = torch.stack(logits_list)  # Creates copy → 2× memory
```

**After**:
```python
logits = torch.zeros(batch, seq_len, vocab_size, device=device)
for t in range(seq_len):
    logits[:, t, :] = logit_t  # In-place write → 1× memory
```

**Memory saved**: ~30-50% during forward pass

## 📋 Debugging Checklist

When encountering CUDA OOM errors:

1. **Check tensor shapes**:
   ```python
   print(f"Logits: {logits.shape} → {logits.numel() * 4 / 1e9:.2f} GiB")
   ```

2. **Verify detach() usage**:
   ```python
   print(f"requires_grad: {tensor.requires_grad}")  # Should be False
   ```

3. **Monitor GPU memory**:
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GiB")
   print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")
   ```

4. **Calculate expected memory**:
   ```
   logits_memory = batch × seq × vocab × 4 / 1e9
   total_expected = logits_memory + 8-12 GiB overhead
   ```

5. **Check for memory leaks**:
   - Look for tensors stored as instance variables without `.detach()`
   - Check for lists accumulating tensors across batches
   - Verify `optimizer.zero_grad()` is called

## 🚨 Common Pitfalls

### Pitfall 1: Storing Tensors with Gradients

```python
# ❌ Bad - causes memory leak
self.cached_tensor = some_computation(x)

# ✅ Good - breaks gradient chain
self.cached_tensor = some_computation(x).detach()
```

### Pitfall 2: Large Vocabulary + Large Batch

```python
# ❌ Bad - 49 GiB
batch=512, seq=512, vocab=50257

# ✅ Good - 3 GiB
batch=64, seq=256, vocab=50257

# ✅ Better - 0.6 GiB
batch=64, seq=256, vocab=10000
```

### Pitfall 3: Accumulating Lists

```python
# ❌ Bad - memory grows linearly
all_outputs = []
for batch in dataloader:
    output = model(batch)
    all_outputs.append(output)  # Keeps all in memory!

# ✅ Good - process and discard
for batch in dataloader:
    output = model(batch)
    metrics = compute_metrics(output)
    # output is garbage collected here
```

## 📖 References

- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
- Mixed Precision Training: https://pytorch.org/docs/stable/amp.html
- Gradient Accumulation: https://huggingface.co/docs/transformers/perf_train_gpu_one

## 📅 Change Log

- **2025-11-20**: Initial documentation
  - Fixed memory leak with `.detach()`
  - Added `--max-length` parameter
  - Analyzed vocab_size impact on memory
  - Established safe batch_size guidelines
