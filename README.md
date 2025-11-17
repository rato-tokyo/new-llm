# New-LLM: Context Vector Propagation Language Model

An experimental language model architecture that replaces attention mechanisms with **context vector propagation**.

## Overview

This project explores a novel approach to sequence modeling:

- **No attention mechanism**: Instead of self-attention, we use a fixed-size context vector
- **Additive propagation**: Context vectors are updated additively at each position
- **Indirect learning**: Context updates emerge from optimizing token prediction only
- **FNN-based**: Uses feedforward neural networks instead of transformers

### Architecture Concept

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

## Project Structure

```
new-llm/
├── src/
│   ├── models/
│   │   ├── baseline_llm.py          # Standard FNN-based LM (no attention)
│   │   └── context_vector_llm.py    # New-LLM with context propagation
│   ├── training/
│   │   ├── dataset.py                # Data loading and tokenization
│   │   └── trainer.py                # Training loop
│   ├── evaluation/
│   │   └── metrics.py                # Loss, perplexity, accuracy
│   └── utils/
│       └── config.py                 # Model configurations
├── data/
│   └── sample_texts.txt              # Training data
├── experiments/
│   ├── train_baseline.py             # Train baseline model
│   ├── train_new_llm.py              # Train new-llm model
│   └── compare_models.py             # Compare performance
└── checkpoints/                      # Saved models
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Baseline Model

Train a standard FNN-based language model (without attention):

```bash
cd experiments
python train_baseline.py
```

This serves as a baseline to compare against new-llm.

### 2. Train New-LLM

Train the context vector propagation model:

```bash
python train_new_llm.py
```

### 3. Compare Results

Analyze and compare both models:

```bash
python compare_models.py
```

This will generate:
- Performance metrics comparison
- Training curves visualization
- Analysis of context vector behavior

## Configuration

**All configurations are centralized in `src/utils/config.py`**.

This is the single source of truth for all model parameters. Do not hardcode values elsewhere.

**BaseConfig** (for baseline LSTM model):
- `vocab_size`: 1000 - Vocabulary size
- `embed_dim`: 128 - Token embedding dimension
- `hidden_dim`: 256 - LSTM hidden state dimension
- `num_layers`: 3 - Number of stacked LSTM layers
- `batch_size`: 16 - Batch size for training
- `learning_rate`: 0.001 - Adam optimizer learning rate
- `num_epochs`: 50 - Number of training epochs
- (See `src/utils/config.py` for full documentation)

**NewLLMConfig** (extends BaseConfig):
- Inherits all BaseConfig parameters
- Additional `context_vector_dim`: 64 - Context vector dimension

## Experiment Design

### Baseline Model
- LSTM-based language model
- Uses recurrent hidden states to capture sequential context
- No attention mechanism
- ~1.8M parameters

### New-LLM Model
- Concatenates context vector to each token embedding
- FNN outputs: (1) next token prediction, (2) context update
- Context is additively updated: `context_t = context_t-1 + delta_t`
- First token has zero context vector
- **Training objective**: Only optimize token prediction loss
  - Context updates are **not** directly supervised
  - They emerge from optimizing token predictions
- ~587K parameters (68% fewer than LSTM baseline)

### Research Questions

1. **Can context vectors learn meaningful representations** without direct supervision?
2. **Is additive propagation sufficient** for capturing sequential dependencies?
3. **How does performance compare** to a standard baseline?
4. **What information is encoded** in the context vectors?

## Expected Outcomes

**If successful**, new-llm should:
- Achieve comparable or better perplexity than baseline
- Show meaningful context vector patterns
- Demonstrate that indirect learning of context is viable

**If unsuccessful**, it may indicate:
- Direct supervision of context is necessary
- Additive updates have limitations
- Fixed-size context bottleneck is too restrictive

## Hardware Requirements

- **RAM**: 16GB (optimized for limited resources)
- **GPU**: Not required (CPU training is feasible)
- **Storage**: <1GB for models and data

## Dataset

The project uses a small dataset of simple English sentences for proof-of-concept:
- 30 short sentences
- ~150 unique tokens
- Train/val split: 80/20

This minimal dataset allows rapid experimentation on the core architecture ideas.

## Results

After training both models, you can find:
- Model checkpoints in `checkpoints/`
- Comparison plots in `checkpoints/model_comparison.png`
- Performance metrics printed during training

## Future Work

If the basic concept proves viable:
1. Scale to larger datasets (WikiText, etc.)
2. Experiment with different context vector dimensions
3. Try different update mechanisms (gating, multiplicative, etc.)
4. Analyze what linguistic features the context captures
5. Compare to transformer baselines with attention

## License

MIT

## Citation

If you use this code in your research, please cite:

```
@misc{newllm2024,
  title={New-LLM: Context Vector Propagation for Language Modeling},
  author={Your Name},
  year={2024},
}
```

## Contact

For questions or discussions, please open an issue on GitHub.
