# New-LLM: Context Vector Propagation Language Model

An experimental language model architecture that replaces attention mechanisms with **context vector propagation**.

## Primary Research Question

**Can an LLM function without attention mechanisms?**

This project verifies whether context vector propagation can compete with standard Transformer attention.

## Overview

This project explores a novel approach to sequence modeling:

- **No attention mechanism**: Instead of self-attention, we use a fixed-size context vector
- **Additive propagation**: Context vectors are updated additively at each position
- **Indirect learning**: Context updates emerge from optimizing token prediction only
- **FNN-based**: Uses feedforward neural networks instead of attention layers

## Core Design Principles

### 🎯 Fixed Memory Usage Regardless of Sequence Length

**Critical principle**: New-LLM must maintain **constant memory usage** independent of sequence length.

This is the fundamental advantage over Transformers:
- **Transformer**: Memory grows with O(n²) where n = sequence length (attention matrix)
- **New-LLM**: Memory stays constant O(1) (fixed-size context vector)

#### Implementation Rules

✅ **ALLOWED**:
- Fixed-size context vector (e.g., 512 dimensions)
- Token embeddings (reused for each token, not stored)
- FNN parameters (fixed regardless of sequence)

❌ **PROHIBITED**:
- **Positional embeddings** that limit max sequence length
- Storing all hidden states (would grow with sequence length)
- Any operation that depends on max_seq_length parameter

#### Why No Positional Embeddings?

```python
# ❌ BAD: Learned positional embeddings
self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
# Problem: Can only handle sequences up to max_seq_length
# Problem: Memory usage tied to sequence length

# ✅ GOOD: Position information from sequential processing
for t in range(seq_len):  # Can be any length
    context[t] = update(context[t-1], input[t])
    # Position information emerges naturally from order
```

Like RNN/LSTM, position information is **implicitly learned** through sequential processing order, not explicit positional encodings.

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
│   │   ├── transformer_baseline.py    # Transformer with multi-head attention
│   │   ├── context_vector_llm.py      # New-LLM with context propagation
│   │   └── baseline_llm.py            # Legacy LSTM baseline
│   ├── training/
│   │   ├── dataset.py                 # Data loading and tokenization
│   │   ├── wikitext_dataset.py        # WikiText-2 dataset loader
│   │   └── trainer.py                 # Training loop with resume support
│   ├── evaluation/
│   │   └── metrics.py                 # Loss, perplexity, accuracy
│   └── utils/
│       ├── config.py                  # Model configurations (single source of truth)
│       └── config_helper.py           # Configuration utilities
├── scripts/
│   ├── train_wikitext_fp16.py         # FP16 mixed precision training (Colab)
│   ├── train_wikitext_advanced.py     # Advanced experiments (Colab)
│   ├── train_wikitext_int8.py         # INT8 quantization (Colab)
│   └── run_colab_experiments.sh       # One-command Colab execution
├── data/
│   └── sample_texts.txt               # Training data
├── experiments/
│   ├── baseline_wikitext_experiment.md     # Baseline results (2.74M params)
│   ├── colab_advanced_experiment.md        # Advanced results (4.84M params)
│   ├── gating_improvements_summary.md      # Gating mechanism improvements
│   └── layernorm_experiments_summary.md    # LayerNorm experiments
└── checkpoints/                       # Saved models and training curves
```

## Training on Google Colab (Recommended)

**All experiments are now conducted on Google Colab** for GPU acceleration and convenience.

### 🚀 Quick Start: One-Command Execution

The easiest way to run experiments on Google Colab:

**Step 1**: Open [Google Colab](https://colab.research.google.com/)

**Step 2**: Change runtime to GPU:
- `Runtime` → `Change runtime type` → `GPU` (Tesla T4, 15GB VRAM)

**Step 3**: Execute this single command:

```bash
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/run_colab_experiments.sh | bash
```

This will automatically:
1. Clone the latest repository (`rm -rf` + `git clone`)
2. Install dependencies
3. Start **Experiment 1**: FP16 training (Baseline model with mixed precision)
4. Start **Experiment 2**: Context 1024 experiment (4x larger context)
5. Both experiments run in parallel

### 📊 Monitor Progress

Check experiment progress with:

```bash
# Experiment 1 (FP16) progress
!tail -30 /content/fp16_log.txt

# Experiment 2 (Context 1024) progress
!tail -30 /content/ctx1024_log.txt

# GPU usage
!nvidia-smi
```

### 🎯 Available Experiments

#### Experiment 1: FP16 Mixed Precision Training

- **Script**: `scripts/train_wikitext_fp16.py`
- **Purpose**: 2x training speedup with PyTorch AMP
- **Model**: Baseline (context=256, layers=6, 2.74M params)
- **Expected time**: ~30 minutes (50 epochs on Tesla T4)
- **Memory**: ~50% less than FP32

#### Experiment 2: Advanced Model (Larger Context)

- **Script**: `scripts/train_wikitext_advanced.py`
- **Purpose**: Test larger model capacity
- **Model**: Advanced (context=512/1024, layers=12, 4.84M+ params)
- **Configurable**: Edit `context_vector_dim` in script
- **Expected time**: ~35-40 minutes (50 epochs on Tesla T4)

#### Experiment 3: INT8 Quantization

- **Script**: `scripts/train_wikitext_int8.py`
- **Purpose**: Post-training quantization for model compression
- **Workflow**: FP32 training → INT8 conversion
- **Expected compression**: 31 MB → 8 MB (~75% reduction)
- **Accuracy impact**: <3% perplexity increase

### 🔧 Manual Colab Execution

If you prefer to run experiments manually:

```bash
# Step 1: Clone repository (use git clone, not git pull)
%cd /content
!rm -rf new-llm
!git clone https://github.com/rato-tokyo/new-llm
%cd new-llm

# Step 2: Install dependencies
!pip install -q datasets

# Step 3: Run experiment
!python scripts/train_wikitext_fp16.py
```

**Important**: Always use `rm -rf` + `git clone` (not `git pull`) in Colab to ensure clean state.

## Configuration

**All configurations are centralized in `src/utils/config.py`**.

This is the single source of truth for all model parameters. Do not hardcode values elsewhere.

### NewLLMConfig (Baseline)

```python
# Model architecture
vocab_size = 1000           # Vocabulary size
embed_dim = 256             # Token embedding dimension
hidden_dim = 512            # FNN hidden dimension
num_layers = 6              # Number of FNN layers
context_vector_dim = 256    # Context vector dimension
dropout = 0.1               # Dropout rate

# Training hyperparameters
num_epochs = 50             # Training epochs
batch_size = 32             # Batch size (512 for GPU)
learning_rate = 0.0001      # Learning rate
gradient_clip = 1.0         # Adaptive gradient clipping
patience = 15               # Early stopping patience

# Dataset
max_seq_length = 64         # Maximum sequence length
```

### AdvancedConfig (Larger Model)

```python
# Model architecture (scaled up)
context_vector_dim = 512    # 2x larger context (or 1024 for 4x)
num_layers = 12             # 2x more layers

# Training hyperparameters (GPU-optimized)
batch_size = 512            # Full GPU RAM utilization
num_epochs = 50             # May need 100 for larger models
```

See `src/utils/config.py` for full configuration options.

## Experiment Results

### Baseline WikiText-2 Experiment

**Model**: New-LLM Baseline (2.74M params, context=256, layers=6)

| Metric | Value |
|--------|-------|
| **Best Val Perplexity** | **23.34** (Epoch 27) |
| **Improvement** | 52% (from 48.6 → 23.34) |
| **Training** | CPU, ~1 hour/epoch, 27 epochs |
| **Overfitting** | None detected |

**Key Findings**:
- ✅ Stable training without overfitting
- ✅ Continuous improvement for 27 epochs
- ✅ Reasonable performance with small model size
- ⏭️ GPU version achieves 100x speedup

See `experiments/baseline_wikitext_experiment.md` for details.

### Google Colab Advanced Experiment

**Model**: New-LLM Advanced (4.84M params, context=512, layers=12)

| Metric | Value |
|--------|-------|
| **Best Val Perplexity** | **36.45** (Epoch 50) |
| **Training Speed** | 0.6-0.7 min/epoch (GPU) |
| **GPU RAM Usage** | 3.2 / 15.0 GB (21%) |
| **Improvement** | 48.9% (from 71.3 → 36.45) |

**Surprising Result**: Larger model (4.84M) performed **worse** than baseline (2.74M)!

**Analysis**:
- Baseline (2.74M): PPL **23.34** ✓
- Advanced (4.84M): PPL **36.45** ✗
- **Root cause**: Insufficient training (50 epochs not enough for larger model)
- **Evidence**: Still improving at epoch 50, no early stopping triggered

**Recommendations**:
1. Extend training to 100 epochs
2. Reduce batch_size from 512 to 256 (more frequent updates)
3. Increase learning_rate to 0.0003 (faster convergence)

See `experiments/colab_advanced_experiment.md` for full analysis.

### Key Insights

1. **GPU Acceleration**: 100x speedup on Colab (0.7 min/epoch vs 60 min/epoch)
2. **Model Size**: Larger models need more training time
3. **Batch Size**: Very large batch_size (512) may slow convergence
4. **FP16 Mixed Precision**: Expected 2x speedup with minimal accuracy loss

## Hardware Requirements

### Google Colab (Recommended)

- **GPU**: Tesla T4 (15GB VRAM) - Free tier
- **Session limit**: 90 minutes (enough for most experiments)
- **Training speed**: 0.6-0.7 min/epoch for baseline models
- **Cost**: Free (or Colab Pro for longer sessions)

### Local Training (Legacy)

- **RAM**: 16GB recommended
- **GPU**: Optional (CPU training is slow but feasible)
- **Storage**: <1GB for models and data

**Note**: All current experiments use Google Colab for GPU acceleration.

## Datasets

### WikiText-2 (Pre-training)

- **Size**: ~4 million tokens
- **Source**: High-quality Wikipedia articles
- **Purpose**: Learn natural language patterns
- **Train/Val**: 36,718 / 3,760 sequences
- **Vocabulary**: 1,000 most frequent tokens

### DailyDialog (Fine-tuning - Planned)

- **Size**: 13,118 dialogues
- **Source**: Daily conversations (~10 turns each)
- **Purpose**: Learn dialogue/conversation patterns
- **Status**: Dataset prepared, fine-tuning experiments pending

## Training Resume Support

All Colab experiments support training resume:

```python
# Resume from checkpoint
trainer.train(resume_from="checkpoints/new_llm_wikitext_epoch_25.pt")
```

**Use case**: Google Colab 90-minute limit
- First session: Train to epoch 25
- Second session: Resume from epoch 25

See `RESUME_TRAINING.md` for detailed instructions.

## Future Experiments

### Planned

1. **Training Extension**: 100 epochs for Advanced model
2. **Batch Size Optimization**: Test 128, 256, 512
3. **Learning Rate Scheduling**: Cosine annealing, warmup
4. **FP16 Comparison**: FP16 vs FP32 speed/accuracy trade-off
5. **DailyDialog Fine-tuning**: Conversation ability

### Research Directions

1. Larger context vector dimensions (1024, 2048)
2. Multiple context vectors (multi-channel)
3. Hybrid architectures (sparse attention + context)
4. Better regularization for context stability
5. Analysis of what linguistic features context captures
6. Comparison with TinyGPT2 and other small LLMs

## Legacy Experiments

The project previously compared against Transformer and LSTM baselines on synthetic data. Those experiments are preserved in:

- `experiments/train_transformer.py` - Transformer baseline
- `experiments/train_new_llm.py` - New-LLM baseline
- `src/models/baseline_llm.py` - LSTM model

**Results Summary**:
- Transformer (5.26M params): PPL 126.5
- New-LLM (3.01M params): PPL 280.3
- **Conclusion**: LLMs can function without attention (43% fewer params, 16.5% higher loss)

## Installation (Local Development)

```bash
# Clone repository
git clone https://github.com/rato-tokyo/new-llm
cd new-llm

# Install dependencies
pip install -r requirements.txt
```

**Note**: For experiments, use Google Colab (no local installation needed).

## Documentation

- `CLAUDE.md` - Development guidelines and best practices
- `CONFIGURATION.md` - Detailed configuration documentation
- `RESUME_TRAINING.md` - Training resume instructions
- `experiments/*.md` - Experiment results and analysis

## Contributing

See `CLAUDE.md` for:
- Code cleanup policy (no old files, fixed file names)
- Experiment management (process termination checklist)
- Google Colab best practices (git clone, one-command execution)
- Architecture principles (O(1) memory, no positional embeddings)

## License

MIT

## Citation

If you use this code in your research, please cite:

```
@misc{newllm2024,
  title={New-LLM: Context Vector Propagation for Language Modeling Without Attention},
  author={Your Name},
  year={2024},
}
```

## Contact

For questions or discussions, please open an issue on GitHub.
