# Configuration Guide

## Overview

All model configurations are centralized in `src/utils/config.py`. This is the **single source of truth** for all parameters.

**DO NOT hardcode configuration values elsewhere in the codebase.**

## Configuration Files

### Main Configuration
- **Location**: `src/utils/config.py`
- **Classes**:
  - `BaseConfig`: Configuration for Baseline LSTM model
  - `NewLLMConfig`: Configuration for New-LLM (inherits from BaseConfig)

### Helper Utilities
- **Location**: `src/utils/config_helper.py`
- **Functions**:
  - `print_config()`: Display configuration in formatted way
  - `validate_config()`: Validate configuration parameters
  - `get_model_info()`: Get model information

## How to Change Configuration

### 1. Edit `src/utils/config.py`

```python
class BaseConfig:
    # Model Architecture
    vocab_size = 1000        # Change vocabulary size
    embed_dim = 128          # Change embedding dimension
    hidden_dim = 256         # Change hidden layer size
    num_layers = 3           # Change number of layers

    # Training
    batch_size = 16          # Change batch size
    learning_rate = 0.001    # Change learning rate
    num_epochs = 50          # Change number of epochs
```

### 2. Validate Your Changes

Run the config helper to validate:

```bash
python3 -m src.utils.config_helper
```

### 3. Re-train Models

After changing configuration:

```bash
# Train baseline model
python3 experiments/train_baseline.py

# Train new-llm model
python3 experiments/train_new_llm.py
```

## Configuration Parameters

### Model Architecture Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `vocab_size` | int | Size of vocabulary | 1000 |
| `embed_dim` | int | Dimension of token embeddings | 128 |
| `hidden_dim` | int | Dimension of hidden states/layers | 256 |
| `num_layers` | int | Number of stacked layers | 3 |
| `max_seq_length` | int | Maximum sequence length | 32 |
| `dropout` | float | Dropout rate (0 to 1) | 0.1 |

### New-LLM Specific Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `context_vector_dim` | int | Dimension of context vector | 64 |

### Training Hyperparameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `batch_size` | int | Number of samples per batch | 16 |
| `learning_rate` | float | Learning rate for optimizer | 0.001 |
| `num_epochs` | int | Number of training epochs | 50 |
| `gradient_clip` | float | Gradient clipping threshold | 1.0 |

### Data Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `train_split` | float | Fraction for training (0 to 1) | 0.8 |
| `random_seed` | int | Random seed for reproducibility | 42 |

### Device Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `device` | str | Device for training (cpu/cuda) | "cpu" |

## Common Configuration Changes

### Increase Model Capacity

To increase model capacity (more parameters):

```python
class BaseConfig:
    embed_dim = 256          # Increase from 128
    hidden_dim = 512         # Increase from 256
    num_layers = 6           # Increase from 3
```

**Warning**: Larger models require more memory and training time.

### Longer Sequences

To handle longer sequences:

```python
class BaseConfig:
    max_seq_length = 64      # Increase from 32
```

### Faster Training

To speed up training (may reduce quality):

```python
class BaseConfig:
    batch_size = 32          # Increase from 16 (if memory allows)
    num_epochs = 25          # Reduce from 50
```

### Adjust Context Vector Size

For New-LLM, adjust context vector dimension:

```python
class NewLLMConfig(BaseConfig):
    context_vector_dim = 128  # Increase from 64
```

**Trade-off**: Larger context vectors can store more information but increase parameters.

## Configuration Validation

The `validate_config()` function checks:

- All required parameters are present
- Values are within valid ranges
- Types are correct

Example:

```python
from src.utils.config import BaseConfig
from src.utils.config_helper import validate_config

config = BaseConfig()
validate_config(config)  # Raises ValueError if invalid
```

## Best Practices

1. **Always edit `config.py`**: Never hardcode values in scripts
2. **Validate after changes**: Run `config_helper.py` to check
3. **Document changes**: Add comments explaining why you changed values
4. **Test small first**: Test with small values before scaling up
5. **Version control**: Commit configuration changes with code changes

## Examples

### Example 1: Small Model for Testing

```python
class BaseConfig:
    vocab_size = 500
    embed_dim = 64
    hidden_dim = 128
    num_layers = 2
    batch_size = 8
    num_epochs = 10
```

### Example 2: Larger Model for Better Performance

```python
class BaseConfig:
    vocab_size = 5000
    embed_dim = 256
    hidden_dim = 512
    num_layers = 6
    batch_size = 32
    num_epochs = 100
```

### Example 3: GPU Training

```python
class BaseConfig:
    device = "cuda"
    batch_size = 64  # Can use larger batches with GPU
```

## Troubleshooting

### Out of Memory

If you get OOM errors:
- Reduce `batch_size`
- Reduce `hidden_dim`
- Reduce `num_layers`
- Reduce `max_seq_length`

### Training Too Slow

If training is too slow:
- Increase `batch_size` (if memory allows)
- Reduce `num_epochs`
- Use GPU (`device = "cuda"`)

### Model Not Learning

If the model isn't learning:
- Increase `num_epochs`
- Adjust `learning_rate` (try 0.0001 or 0.01)
- Increase model capacity (`hidden_dim`, `num_layers`)

## References

- Main config: `src/utils/config.py`
- Helper utilities: `src/utils/config_helper.py`
- Model implementations: `src/models/`
- Training scripts: `experiments/`
