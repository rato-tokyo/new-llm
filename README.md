# New-LLM: Context Vector Propagation Language Model

An experimental language model that replaces attention mechanisms with **context vector propagation** for **O(1) memory usage**.

---

## 🎯 Research Question

**Can an LLM function without attention mechanisms?**

**Answer**: Yes! New-LLM achieves competitive performance using only context vector propagation.

---

## 🚀 Quick Start (Google Colab)

### Dolly-15k Dialog Training (Recommended)

```bash
# Clone repository
%cd /content
!rm -rf new-llm
!git clone https://github.com/rato-tokyo/new-llm
%cd new-llm

# Install dependencies
!pip install -q datasets

# Start training
!nohup python3 scripts/train_dolly.py --num_layers 1 > /content/dolly.log 2>&1 &

# Monitor
!tail -20 /content/dolly.log
```

**Hardware**: Google Colab Pro (L4 GPU, 24GB VRAM) recommended

---

## 📊 Latest Results

### Dolly-15k Dialog Training (2025-11-19)

| Model | Val PPL | Val Acc | Status |
|-------|---------|---------|--------|
| **Layer 1** | **15.6** | **46.6%** | ✅ **Complete** |

**Result**: **大成功！** - Dolly-15kで優れた対話能力を獲得

### WikiText-2 Language Modeling (2025-11-18)

| Layers | Val PPL | Val Acc | Status |
|--------|---------|---------|--------|
| **Layer 4** | **20.1** | **38.3%** | **Best** (partial) |
| **Layer 5** | **20.5** | **38.3%** | Complete |
| Layer 1 | 20.4 | 38.0% | Good |

**Finding**: Layer 4-5 optimal for WikiText-2

### Performance Comparison

| Dataset | Difficulty | PPL | Reasoning |
|---------|-----------|-----|-----------|
| **Dolly-15k** | Easier | **15.6** | Structured Q&A format |
| **WikiText-2** | Harder | **20.4** | Natural, diverse text |

**Key Insight**: Structured data (Dolly) is easier to model than natural text (WikiText).

---

## 🧠 Core Concept

### O(1) Memory Usage

| Architecture | Memory | Max Sequence |
|--------------|--------|--------------|
| Transformer | O(n²) | Limited |
| **New-LLM** | **O(1)** | **Unlimited** |

**No attention, no positional embeddings** - position emerges from sequential processing (like RNN/LSTM).

See `ARCHITECTURE.md` for details.

---

## 📂 Project Structure

```
new-llm/
├── scripts/                        # Training scripts
│   ├── train_dolly.py              # Dolly-15k dialog
│   ├── train_wikitext_fp16.py      # WikiText-2 baseline
│   └── train_wikitext_fp16_layers.py  # Layer optimization
├── src/
│   ├── models/context_vector_llm.py   # New-LLM implementation
│   ├── training/                      # Trainers & datasets
│   ├── evaluation/metrics.py          # Metrics
│   └── utils/config.py                # Configurations
├── tests/                          # Test suite
├── experiments/                    # Results & analysis
└── checkpoints/                    # Saved models
```

---

## 📖 Documentation

- **ARCHITECTURE.md** - Architecture details & design principles
- **DOLLY_TRAINING.md** - Dolly-15k training guide
- **experiments/README.md** - Experiment index
- **CLAUDE.md** - Development guidelines

---

## 🔬 Key Findings

### Scaling Rules

1. **Batch Size (Square Root Rule)**:
   ```
   batch 32→2048 (64x) → lr 0.0001→0.0008 (√64 = 8x)
   ```

2. **Model Size**:
   ```
   Larger model → Lower learning rate (prevent instability)
   ```

3. **Layer Optimization**:
   - Optimal: 4-5 layers for WikiText-2
   - Layer 1: Good for simple tasks (Dolly-15k)
   - Layer 7: Overfits

### GPU Optimization

| GPU | VRAM | Batch Size | Performance |
|-----|------|------------|-------------|
| T4 | 16GB | 512 | Baseline |
| **L4** | 24GB | **2048** | **4x faster** |
| A100 | 40GB | 4096 | 8x faster (est.) |

---

## 🚀 Future Work

1. **Layer 4 for Dolly**: Expected PPL 12-14
2. **Context Expansion**: 256→512 dimensions
3. **Japanese Dialog**: Japanese Alpaca dataset
4. **Longer Sequences**: Test O(1) memory with very long sequences

---

## 🧪 Running Tests

```bash
# Run all tests
python tests/test_all.py

# Run specific test
python tests/test_dolly_training.py
```

---

## 📝 Citation

```bibtex
@misc{newllm2024,
  title={New-LLM: Context Vector Propagation for Language Modeling Without Attention},
  author={New-LLM Project},
  year={2024},
  url={https://github.com/rato-tokyo/new-llm}
}
```

---

## 📄 License

MIT

---

**Status**: Active research project. Latest experiment: Dolly-15k dialog training (PPL 15.6, Acc 46.6%)
