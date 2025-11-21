"""Test different convergence metrics

Compare MSE loss vs L2 norm for convergence checking.
"""

import torch
import torch.nn.functional as F

# Simulate two context vectors that are "close"
context_old = torch.randn(1, 256)
context_new = context_old + torch.randn(1, 256) * 0.01  # Add small noise

# Metrics
mse_loss = F.mse_loss(context_new, context_old)
l2_norm = torch.norm(context_new - context_old, dim=-1)

print("="*70)
print("Convergence Metrics Comparison")
print("="*70)
print(f"\nContext dimension: 256")
print(f"Noise scale: 0.01 per dimension")
print(f"\nMSE Loss: {mse_loss.item():.6f}")
print(f"L2 Norm:  {l2_norm.item():.6f}")
print(f"\nRatio (L2 / sqrt(MSE)): {l2_norm.item() / torch.sqrt(mse_loss).item():.2f}")
print(f"Expected ratio: sqrt(256) = {torch.sqrt(torch.tensor(256.0)).item():.2f}")
print("\n" + "="*70)
print("Conclusion:")
print("  L2 norm ≈ sqrt(dim) * sqrt(MSE)")
print(f"  For dim=256: L2 norm ≈ 16 * sqrt(MSE)")
print(f"  If MSE threshold = 0.01, then L2 threshold ≈ {16 * torch.sqrt(torch.tensor(0.01)).item():.2f}")
print("="*70)
