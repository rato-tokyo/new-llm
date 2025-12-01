# Google Colab Setup for 100k Token Training

## Quick Start Guide

### Step 1: Setup Colab Environment

```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/your-username/new-llm.git
%cd new-llm

# Install dependencies
!pip install -q transformers datasets torch accelerate

# Verify GPU
import torch
print(f"✅ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 2: Configure Paths for Google Drive

```python
# Create directories on Google Drive
!mkdir -p /content/drive/MyDrive/new-llm/cache
!mkdir -p /content/drive/MyDrive/new-llm/checkpoints

# Link to project
!ln -sf /content/drive/MyDrive/new-llm/cache ./cache
!ln -sf /content/drive/MyDrive/new-llm/checkpoints ./checkpoints
```

### Step 3: Test with 10k Tokens First

Edit `config.py`:
```python
num_samples = 78  # ~10k tokens (78 × 128)
```

Run Phase 1 training:
```python
!python3 test.py
```

Expected output:
- Training time: 1-2 minutes (GPU)
- Memory usage: <2 GB
- Effective Rank: 88-90%

### Step 4: Scale to 100k Tokens

Edit `config.py`:
```python
num_samples = 781  # ~100k tokens (781 × 128)
```

Run Phase 1 training:
```python
!python3 train.py
```

Expected output:
- Training time: 5-15 minutes (GPU)
- Memory usage: ~5.3 GB (peak)
- Effective Rank: 88-90%

---

## Memory Requirements

### 100k Token Training

| Component | Size | Notes |
|-----------|------|-------|
| Model Weights | 348 MB | 91.4M parameters |
| Optimizer (Adam) | 698 MB | 2 states per param |
| Context Storage | 293 MB | 100k × 768 dimensions |
| Activation Memory | 4,102 MB | Per-token processing |
| **Total** | **5,441 MB (5.3 GB)** | Peak usage |

### Colab GPU Limits

- **Free Tier (T4)**: 15 GB → **9.7 GB safety margin** ✅
- **Colab Pro (T4/P100/V100)**: 25-40 GB → **19.7+ GB safety margin** ✅

---

## Performance Expectations

### GPU (Colab T4/P100)

**Phase 1 (100k tokens)**:
- Per iteration: 0.5-1.3 minutes
- Total training: 5-15 minutes (10 iterations max)
- Early stopping: Usually 3-5 iterations

**Phase 2 (100k tokens)**:
- Per epoch: 1-2 minutes
- Total training (10 epochs): 10-20 minutes

### CPU (Fallback - Not Recommended)

**Phase 1 (100k tokens)**:
- Per iteration: 5-7 minutes
- Total training: 50-70 minutes
- **Use GPU instead for practical training**

---

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

```python
# Monitor GPU memory
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache manually
torch.cuda.empty_cache()
```

**Solutions**:
1. Restart runtime and clear all outputs
2. Reduce `num_samples` (test with 50k first: 390 samples)
3. Switch to High-RAM runtime (Colab Pro)

### Session Timeout

**Symptoms**: "Runtime disconnected" error

**Prevention**:
- Save checkpoints every 2-3 iterations
- Use Google Drive for persistent storage
- Run during off-peak hours (weekends/evenings)

**Recovery**:
- Re-run from checkpoint (automatically loads)
- Checkpoints saved at `./checkpoints/model_latest.pt`

### Slow Data Loading

**First Run**: 2-5 minutes (downloads HuggingFace dataset)

**Solution for Subsequent Runs**:
```python
# Cache is automatically saved to:
# ./cache/ultrachat_781samples_128len.pt

# Subsequent runs load in 1-3 seconds
```

---

## Recommended Workflow

### Incremental Scaling

Test at each scale before proceeding:

1. **6.4k tokens** (50 samples): ~30 seconds ✅ Baseline
2. **10k tokens** (78 samples): ~1 minute → Verify no errors
3. **25k tokens** (195 samples): ~2 minutes → Check memory usage
4. **50k tokens** (390 samples): ~5 minutes → Stress test
5. **100k tokens** (781 samples): ~10 minutes → Production run

### Progress Monitoring

```python
# Real-time progress (every 100 tokens)
# Expected output:
# Progress: 10,000/100,000 (10.0%) | Speed: 1200 tok/s | ETA: 75s
```

### Checkpoint Management

```python
# Checkpoints auto-save at:
# - End of Phase 1
# - End of Phase 2

# Manual save (if needed):
import torch
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {...}
}
torch.save(checkpoint, './checkpoints/backup.pt')
```

---

## Configuration for 100k Tokens

### config.py Settings

```python
# Data configuration
num_samples = 100                    # Adjust based on token count needed

# Phase 1: Fixed-Point Learning
phase1_max_iterations = 10           # Usually converges in 3-5
phase1_convergence_threshold = 0.05  # Can increase to 0.1 for faster training
phase1_learning_rate = 0.002         # Recommended

# Distribution regularization
dist_reg_weight = 0.5                # Balanced (50/50 CVFP/diversity)

# Phase 2: Token Prediction
phase2_learning_rate = 0.002         # Same as Phase 1
phase2_epochs = 10                   # Adjust based on time budget
freeze_context = False               # Full fine-tuning recommended

# Checkpoint management
save_checkpoint = True               # Always enable
load_checkpoint = True               # Auto-resume if exists
```

---

## Expected Results

### Phase 1 (100k tokens)

**Training Data**:
- Effective Rank: 88-90% (676-691/768 dimensions)
- CVFP Loss: ~0.0015-0.002
- Convergence: 3-5 iterations

**Validation Data (20k from train)**:
- Effective Rank: 80-83% (614-637/768 dimensions)

**Success Criteria**:
- ✅ Training completes without OOM
- ✅ Effective Rank > 85% (training)
- ✅ Effective Rank > 75% (validation)
- ✅ CVFP convergence check passes (final_diff < 0.001)

### Phase 2 (100k tokens)

**10 Epochs Expected**:
- Initial Loss: ~11.0
- Final Loss: ~4.5-5.5 (depending on data)
- Final Perplexity: ~90-250
- Token Accuracy: ~10-15%

---

## Advanced: Mixed Precision Training

For faster training with less memory:

```python
# Enable automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

# In Phase1Trainer, add:
scaler = GradScaler()

with autocast():
    # Your training code here
    loss = ...

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 30-50% faster training
- 20-30% less memory usage
- Minimal accuracy loss

---

## Full Example Notebook

```python
# === Cell 1: Setup ===
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/your-username/new-llm.git
%cd new-llm
!pip install -q transformers datasets torch

# === Cell 2: Verify Environment ===
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# === Cell 3: Setup Persistent Storage ===
!mkdir -p /content/drive/MyDrive/new-llm/cache
!mkdir -p /content/drive/MyDrive/new-llm/checkpoints
!ln -sf /content/drive/MyDrive/new-llm/cache ./cache
!ln -sf /content/drive/MyDrive/new-llm/checkpoints ./checkpoints

# === Cell 4: Test with 10k Tokens ===
!python3 -c "
from config import ResidualConfig
config = ResidualConfig()
config.num_samples = 78  # 10k tokens
"
!python3 test.py

# === Cell 5: Run 100k Token Training ===
!python3 -c "
from config import ResidualConfig
config = ResidualConfig()
config.num_samples = 781  # 100k tokens
"
!python3 train.py

# === Cell 6: Check Results ===
!ls -lh checkpoints/
!python3 -c "
import torch
checkpoint = torch.load('./checkpoints/model_latest.pt', map_location='cpu')
print(f\"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}\")
if 'phase2_history' in checkpoint:
    history = checkpoint['phase2_history']
    print(f\"Final train loss: {history['train_loss'][-1]:.4f}\")
    print(f\"Final val loss: {history['val_loss'][-1]:.4f}\")
"
```

---

## Contact & Support

For issues or questions:
- GitHub Issues: https://github.com/your-username/new-llm/issues
- Documentation: See `CLAUDE.md` for architecture details

---

Last Updated: 2025-11-24
