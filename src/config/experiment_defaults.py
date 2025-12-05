"""
Default experiment configuration values.

実験スクリプト間で共有されるデフォルト値を定義する。
"""

# Early stopping
# NOTE: Do not change this value without explicit user approval
EARLY_STOPPING_PATIENCE = 1

# Gradient clipping
GRADIENT_CLIP = 1.0

# Default learning rate
DEFAULT_LR = 1e-4

# RoPE default rotary percentage
DEFAULT_ROTARY_PCT = 0.25
