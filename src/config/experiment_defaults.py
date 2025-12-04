"""
Default experiment configuration values.

実験スクリプト間で共有されるデフォルト値を定義する。
"""

# Early stopping
EARLY_STOPPING_PATIENCE = 1

# Gradient clipping
GRADIENT_CLIP = 1.0

# Default learning rates
DEFAULT_LR = 1e-4
DEFAULT_PHASE1_LR = 1e-3  # For V-DProj Phase 1
DEFAULT_PHASE2_LR = 1e-4  # For V-DProj Phase 2

# Reconstruction loss weight
DEFAULT_RECON_WEIGHT = 0.1

# Default V projection dimension
DEFAULT_V_PROJ_DIM = 320
