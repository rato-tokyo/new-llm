"""Configuration for new-llm experiments"""

class BaseConfig:
    """Base configuration for models"""
    # Model architecture
    vocab_size = 1000
    embed_dim = 128
    hidden_dim = 256
    num_layers = 3
    max_seq_length = 32
    dropout = 0.1

    # Training
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    gradient_clip = 1.0

    # Data
    train_split = 0.8
    random_seed = 42

    # Device
    device = "cpu"  # Use CPU for 16GB RAM systems


class NewLLMConfig(BaseConfig):
    """Configuration for new-llm with context vectors"""
    context_vector_dim = 64  # Dimension of the context vector
    # The context vector dimension is fixed and carries contextual information
