# New-LLM: Context Vector Propagation Language Model

An experimental language model that replaces attention mechanisms with **context vector propagation** for **O(1) memory usage**.

**Powered by Reconstruction Learning** - a self-supervised approach where the context vector learns to compress previous context and current token information.

---

## 🚀 Quick Start

### Google Colab (Recommended)

**One-line command** to start training:

```python
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_wikitext.sh | bash
```

### Local Training

```bash
# Install dependencies
pip install torch tokenizers datasets tqdm

# Quick test (100 samples, 10 epochs)
python train.py --max-samples 100 --epochs 10 --batch-size 8 --layers 1 --device cpu

# Full training on GPU
python train.py --epochs 30 --batch-size 32 --layers 4 --device cuda

# Results are saved to experiments/ directory by default
# - training_curves.png: Training visualization (overwritten each run)
# - final_model.pt: Latest model checkpoint (overwritten each run)
# - experiment_log.txt: Log of all experiments (appended each run)
```

---

## 🎯 What Makes New-LLM Different?

### O(1) Memory Usage

| Architecture | Memory Complexity | Max Sequence Length |
|--------------|-------------------|---------------------|
| **Transformer** | O(n²) | Limited by memory |
| **New-LLM** | **O(1)** | **Unlimited** ✨ |

**No attention mechanism, no positional embeddings** - position information emerges naturally from sequential processing (like RNN/LSTM).

See `ARCHITECTURE.md` for technical details.

### Reconstruction Learning

New-LLM uses a unique **dual-loss training approach**:

1. **Token Prediction Loss**: Standard next-token prediction (cross-entropy)
2. **Reconstruction Loss**: Context vector learns to compress `[previous_context + current_token_embedding]`

The context vector acts as an **autoencoder** - compressing 512 dimensions (256 context + 256 token) into 256 dimensions, then reconstructing it back.

See `RECONSTRUCTION_LEARNING.md` for details.

---

## 🏗️ Architecture Overview

```python
# Simplified pseudocode
for token in sequence:
    # 1. Embed token
    embedding = embed(token)

    # 2. Store reconstruction target
    reconstruction_target = concat([context, embedding])  # 512 dims

    # 3. Process with context
    hidden = FNN([embedding, context])  # Concatenate and process

    # 4. Update context (gated mechanism)
    forget_gate = sigmoid(W_forget @ hidden)
    input_gate = sigmoid(W_input @ hidden)
    context_delta = tanh(W_context @ hidden)

    context = forget_gate * context + input_gate * context_delta
    context = LayerNorm(context)  # Normalize for stability

    # 5. Predict next token
    logits = W_output @ hidden

    # 6. Reconstruct (for training)
    reconstructed = context_decoder(context)  # 256 → 512 dims
    reconstruction_loss = MSE(reconstructed, reconstruction_target)
```

**Key Innovation**: Fixed-size context vector (256 dims) instead of O(n²) attention.

---

## 📊 Project Structure

```
new-llm/
├── train.py                           # Pure PyTorch training script
├── src/
│   ├── models/
│   │   ├── context_vector_llm.py      # Core New-LLM architecture
│   │   └── transformer_baseline.py    # Transformer comparison model
│   └── utils/
│       └── config.py                  # Model hyperparameters
├── scripts/
│   ├── colab_train_wikitext.sh        # One-line Colab training
│   └── colab_train_ultrachat.sh       # UltraChat dataset training
├── checkpoints/                       # Trained models (auto-saved)
├── ARCHITECTURE.md                    # Architecture documentation
├── RECONSTRUCTION_LEARNING.md         # Reconstruction learning details
└── CLAUDE.md                          # Development guidelines
```

---

## 📖 Training Output Example

```
Epoch 1/2
Train: Loss 7.67 | Token 7.34 | Recon 0.33: 100%|████████| 13/13 [00:42<00:00,  3.25s/it]
Val: Loss 7.09, PPL 1026.57, Acc 2.82%

Epoch 2/2
Train: Loss 6.65 | Token 6.57 | Recon 0.08: 100%|████████| 13/13 [00:42<00:00,  3.22s/it]
Val: Loss 6.38, PPL 567.61, Acc 2.82%

Training complete!
Best checkpoint saved to: test_run/best_model.pt
```

**Key Metrics**:
- **Token Loss**: Next-token prediction accuracy
- **Recon Loss**: Context reconstruction accuracy
- **PPL**: Perplexity (lower is better)
- **Acc**: Token prediction accuracy

---

## 🧪 Development Guidelines

See `CLAUDE.md` for:

- **Git management policies** (prevent merge conflicts)
- **Colab experiment best practices** (1-line commands)
- **Testing policies** (test locally before commit)
- **Code cleanup rules** (no old code left behind)

---

## 🎓 Key Research Findings

### 1. Fixed Memory Complexity

New-LLM maintains **O(1) memory** regardless of sequence length, unlike Transformers' O(n²).

### 2. Emergent Position Information

Position is learned implicitly through sequential processing - no explicit positional embeddings needed.

### 3. Self-Supervised Reconstruction Learning

No external teacher model needed - the model learns to compress its own context vectors.

---

## 🚀 Future Work

1. **Longer context experiments** - Test O(1) memory advantage with 10k+ token sequences
2. **Multi-layer scaling** - Experiment with deeper architectures (8-12 layers)
3. **Comparison with Mamba/RWKV** - Benchmark against other sub-quadratic architectures
4. **Hybrid models** - Combine context propagation with sparse attention

---

## 📝 Citation

```bibtex
@misc{newllm2025,
  title={New-LLM: Context Vector Propagation for Language Modeling Without Attention},
  author={New-LLM Project},
  year={2025},
  url={https://github.com/rato-tokyo/new-llm}
}
```

---

## 📄 License

MIT

---

**Status**: Active research project using pure PyTorch with reconstruction learning.
