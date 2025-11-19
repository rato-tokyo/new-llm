# New-LLM: Context Vector Propagation Language Model

An experimental language model that replaces attention mechanisms with **context vector propagation** for **O(1) memory usage**.

**Now powered by HuggingFace Transformers** for maximum reliability and ease of use.

---

## 🚀 Quick Start

### Google Colab (Recommended)

**One-line command** to start training:

```python
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash -s -- --max-samples 1000 --epochs 5 --batch-size 32
```

**Full training** (all data, 50 epochs):

```python
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash -s -- --epochs 50 --batch-size 32
```

### Local Training

```bash
# Install dependencies
pip install transformers tokenizers datasets tensorboard

# Quick test (1000 samples, 5 epochs)
python train.py --dataset ultrachat --max-samples 1000 --epochs 5

# Full training
python train.py --dataset ultrachat --epochs 50 --batch-size 32
```

### Chat with Trained Model

```bash
python chat.py --model-path checkpoints/ultrachat/final_model --temperature 0.9
```

**Example**:
```
You: hello
Assistant: Hello! How can I help you today?
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

---

## 🏗️ Architecture Overview

```python
# Simplified pseudocode
for token in sequence:
    # 1. Embed token
    embedding = embed(token)

    # 2. Process with context
    hidden = FNN([embedding, context])  # Concatenate and process

    # 3. Update context (gated mechanism)
    forget_gate = sigmoid(W_forget @ hidden)
    input_gate = sigmoid(W_input @ hidden)
    context_delta = tanh(W_context @ hidden)

    context = forget_gate * context + input_gate * context_delta
    context = LayerNorm(context)  # Normalize for stability

    # 4. Predict next token
    logits = W_output @ hidden
```

**Key Innovation**: Fixed-size context vector (256-512 dims) instead of O(n²) attention.

---

## 📊 Project Structure

```
new-llm/
├── train.py                           # 🆕 HuggingFace Trainer-based training
├── chat.py                            # 🆕 HuggingFace generation-based chat
├── scripts/
│   └── colab_train_ultrachat.sh       # One-line Colab training script
├── src/
│   ├── models/
│   │   ├── context_vector_llm.py      # Core New-LLM architecture
│   │   ├── new_llm_config.py          # 🆕 HuggingFace PretrainedConfig
│   │   └── new_llm_hf.py              # 🆕 HuggingFace model wrapper
│   ├── training/
│   │   ├── hf_tokenizer.py            # 🆕 BPE tokenizer creation
│   │   ├── dataset.py                 # Dataset loading
│   │   └── ultrachat_dataset.py       # UltraChat-specific loader
│   └── utils/
│       └── config.py                  # Model hyperparameters
├── checkpoints/                       # Trained models (auto-saved)
└── experiments/                       # Experiment results & analysis
```

---

## 🔧 HuggingFace Integration

### Why HuggingFace Transformers?

✅ **Industry-standard BPE tokenization** - eliminates tokenizer bugs
✅ **Automatic checkpoint management** - model + tokenizer + config saved together
✅ **Built-in Exposure Bias mitigations** - temperature, top-p, repetition penalty
✅ **Simple interfaces** - `train.py` and `chat.py` are under 400 lines total
✅ **FP16 mixed precision** - 2x faster training on GPU

### Features Inherited from HuggingFace

- 📊 **Automatic metrics** - Loss, Perplexity, Accuracy displayed per epoch
- 💾 **Smart checkpointing** - Keep only best 3 checkpoints, auto-save tokenizer
- 📈 **TensorBoard logging** - Real-time training visualization
- 🎛️ **Generation utilities** - Beam search, nucleus sampling, temperature
- 🔄 **Resume training** - Automatic state restoration

---

## 📖 Training Output Example

```
================================================================================
📊 Epoch 1 Results:
================================================================================
  Loss:       9.1468
  Perplexity: 9384.11
  Accuracy:   0.04%
================================================================================

================================================================================
📊 Epoch 2 Results:
================================================================================
  Loss:       9.1134
  Perplexity: 9075.98
  Accuracy:   3.08%
================================================================================

... (continues for all epochs)

================================================================================
✅ Training Complete!
================================================================================

Model saved to: checkpoints/ultrachat/final_model
```

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

### 3. Scalability

Successfully scales from 1-layer to 12-layer models with proper hyperparameter tuning.

---

## 🚀 Future Work

1. **Multi-language support** - Japanese, Chinese datasets
2. **Longer sequences** - Test O(1) memory advantage with 10k+ token sequences
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

**Status**: Active research project, now powered by HuggingFace Transformers ecosystem.
