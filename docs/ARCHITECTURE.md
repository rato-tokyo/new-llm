# New-LLM Architecture

## Core Concept: CVFP (Context Vector Fixed-Point Property)

New-LLM is based on the hypothesis that context vectors converge to fixed points through iterative refinement.

### Two-Phase Training

**Phase 1: Fixed-Point Learning (CVFP)**
- Learn context generation parameters
- Contexts converge to fixed points through iterations
- Uses distribution regularization to prevent dimension collapse

**Phase 2: Token Prediction**
- Train token output layer using fixed contexts
- Standard next-token prediction with cross-entropy loss

### Key Innovation: Distribution Regularization

Forces each dimension (across all tokens) to follow N(0,1):
- Prevents dimension collapse
- Maintains diversity (Effective Rank > 40%)
- Uses population variance (unbiased=False)

### Model Structure

```
Token → Embedding → Context Generation → Fixed Point → Token Output
                    ↑___________________|
                    (Iterative refinement)
```

### Configuration

Edit `config.py` to adjust:
- Model dimensions (layers, context_dim, embed_dim)
- Training parameters (learning rates, iterations, epochs)
- Data sources (UltraChat, text files, manual)