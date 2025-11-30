"""
Weight initialization utilities for new-llm.

Provides shared initialization functions for consistent layer initialization.
"""

import torch.nn as nn


def init_linear_weights(
    module: nn.Module,
    weight_std: float = 0.1,
    bias_std: float = 0.01
) -> None:
    """
    Initialize Linear layer weights with normal distribution.

    Args:
        module: Module to initialize (recursively checks for Linear layers)
        weight_std: Standard deviation for weight initialization
        bias_std: Standard deviation for bias initialization
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.normal_(submodule.weight, mean=0.0, std=weight_std)
            if submodule.bias is not None:
                nn.init.normal_(submodule.bias, mean=0.0, std=bias_std)
