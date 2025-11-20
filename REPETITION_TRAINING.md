# Repetition Training: Context Vector Convergence Experiment

## 🎯 Purpose

Train New-LLM's context update mechanism to reach stable fixed points when processing repeated patterns.

**Hypothesis**: When the same phrase is repeated many times, the context vector should converge:

```
context("red" × 100) ≈ context("red" × 101)
```

## 📐 Theory

### Context Vector Fixed Point

For a repeated pattern, the context update should satisfy:

```
context_{n+1} = f(context_n, token)
```

When `token` is part of a repeating cycle, after sufficient repetitions:

```
context_n ≈ context_{n+1}  (fixed point reached)
```

### Training Strategy

**Staged Training**: Gradually increase pattern complexity

- **Stage 1**: Single token repetition
  - Input: `context("red" × n)`, Token: `"red"`
  - Target: `context("red" × (n+1))` ≈ `context("red" × n)`

- **Stage 2**: Two-token repetition
  - Pattern: `"red apple"` repeated
  - Target: After processing full cycle, context should match previous cycle

- **Stage 3+**: Multi-token repetition
  - Gradually increase complexity

### Loss Function

**Context Convergence Loss**:

```
L_convergence = MSE(context[t], context[t - cycle_length])
```

Where `cycle_length` is the number of tokens in the repeated phrase.

**Token Prediction Loss** (optional, usually disabled):

```
L_token = CrossEntropy(logits, targets)
```

**Total Loss**:

```
L_total = α × L_convergence + β × L_token
```

Typically: `α = 1.0`, `β = 0.0` (focus entirely on context convergence)

## 🚀 Quick Start (Google Colab)

### One-Line Execution

```bash
# Basic training (3 stages, 10 epochs per stage)
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_repetition.sh | bash
```

### Custom Parameters

```bash
# Example: 5 stages, 20 epochs per stage, 200 repetitions
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_repetition.sh | bash -s -- \
  --max-stage 5 \
  --epochs-per-stage 20 \
  --repetitions 200 \
  --context-dim 512 \
  --lr 0.0005
```

### Monitor Progress

```bash
# Watch training log
!tail -20 /content/repetition_training.log

# Stop training if needed
!pkill -9 -f train_repetition.py
```

## 🔧 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-stage` | 3 | Maximum stage (1=single token, 2=two tokens, ...) |
| `--epochs-per-stage` | 10 | Number of epochs per stage |
| `--repetitions` | 100 | How many times to repeat each phrase |
| `--batch-size` | 8 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--context-dim` | 256 | Context vector dimension |
| `--embed-dim` | 256 | Token embedding dimension |
| `--hidden-dim` | 512 | Hidden layer dimension |
| `--layers` | 2 | Number of FNN layers |
| `--convergence-weight` | 1.0 | Weight for convergence loss |
| `--token-weight` | 0.0 | Weight for token prediction loss |
| `--max-length` | 512 | Maximum sequence length |
| `--device` | cuda | Device (cuda/cpu) |
| `--output-dir` | checkpoints | Output directory |

## 📊 Metrics

During training, the following metrics are monitored:

1. **Convergence Loss**: MSE between `context[t]` and `context[t - cycle_length]`
   - Target: **Minimize** (closer to 0 = better convergence)

2. **Context Change Rate**: Average L2 norm of consecutive context changes
   - `||context[t] - context[t-1]||`
   - Target: **Decrease over time** (stable fixed point)

3. **Convergence Metric**: Average distance between cyclic positions
   - Similar to convergence loss but measured as L2 distance
   - Target: **Minimize**

## 💡 Expected Results

### Good Convergence

```
Stage 1 Epoch 10/10: conv_loss=0.0012, change=0.0015, conv_metric=0.0013
```

- Convergence loss < 0.01
- Context change rate decreases over epochs
- Model reaches stable fixed points

### Poor Convergence

```
Stage 1 Epoch 10/10: conv_loss=0.5432, change=0.8234, conv_metric=0.6123
```

- High convergence loss
- Context change rate remains high
- Model fails to stabilize

## 🧪 Example: Single Token Repetition

### Dataset

```python
# Input sequence: "red red red red red ... (100 times)"
tokens = ["red"] * 100

# Context trajectory:
context_0 = zeros(context_dim)
context_1 = update(context_0, embed("red"))
context_2 = update(context_1, embed("red"))
...
context_100 = update(context_99, embed("red"))

# Goal: context_99 ≈ context_100
```

### Loss Computation

```python
# Compare context at position t with context at position t-1
# For single token, cycle_length = 1
loss = MSE(context[1:], context[:-1])

# After training:
# context[99] ≈ context[100] (fixed point reached)
```

## 📈 Staged Training Example

```
Stage 1: ["red"] × 100
  Epoch 1: conv_loss=0.4521, change=0.5234
  Epoch 5: conv_loss=0.0523, change=0.0612
  Epoch 10: conv_loss=0.0012, change=0.0015 ✓

Stage 2: ["red apple"] × 100
  Epoch 1: conv_loss=0.6234, change=0.7123
  Epoch 5: conv_loss=0.1234, change=0.1523
  Epoch 10: conv_loss=0.0234, change=0.0312 ✓

Stage 3: ["red apple tree"] × 100
  Epoch 1: conv_loss=0.7123, change=0.8234
  Epoch 5: conv_loss=0.2134, change=0.2523
  Epoch 10: conv_loss=0.0456, change=0.0523 ✓
```

## 🔬 Advanced Usage

### Local Training

```bash
python scripts/train_repetition.py \
  --max-stage 3 \
  --epochs-per-stage 10 \
  --repetitions 100 \
  --context-dim 512 \
  --device cuda
```

### Custom Vocabulary

Modify `src/data/repetition_dataset.py`:

```python
vocabulary = [
    "custom", "words", "here",
    "your", "own", "vocabulary"
]
```

### Load Checkpoint

```python
import torch
from src.models.new_llm import NewLLM

# Load trained model
checkpoint = torch.load("checkpoints/new_llm_repetition_stage3.pt")
model = NewLLM(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Check metrics
print(checkpoint['metrics'])
```

## 🎓 Research Questions

1. **Convergence Speed**: How many epochs are needed for different stages?
2. **Context Dimension**: Does larger context_dim improve convergence?
3. **Pattern Complexity**: At what stage does convergence become difficult?
4. **Transfer Learning**: Does training on simple patterns help with complex patterns?
5. **Stability**: Do fixed points remain stable during inference?

## 🔗 Related Files

- **Dataset Generator**: `src/data/repetition_dataset.py`
- **Loss Function**: `src/training/convergence_loss.py`
- **Training Script**: `scripts/train_repetition.py`
- **Colab Wrapper**: `scripts/colab_train_repetition.sh`

## 🔬 Performance Evaluation

To evaluate CVFPT performance across multiple tokens:

- **See**: [CVFPT_TOPK_EVALUATION.md](CVFPT_TOPK_EVALUATION.md)
- **Scripts**:
  - `scripts/train_repetition_topk_fair.py` - Fair round-robin evaluation (recommended)
  - `scripts/train_repetition_topk.py` - Sequential evaluation (for comparison)

The Top-K evaluation measures how well CVFPT works across different tokens and provides detailed performance metrics.

## 📚 References

- **Fixed Point Theory**: Study of convergent sequences in dynamical systems
- **RNN Stability**: Analysis of recurrent network fixed points
- **Context Vector Learning**: New-LLM architecture design principles

---

**Note**: This is an experimental training method designed to improve the accuracy and stability of New-LLM's context update mechanism.
