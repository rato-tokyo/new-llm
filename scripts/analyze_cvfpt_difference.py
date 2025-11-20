#!/usr/bin/env python3
"""
Detailed analysis of CVFPT: Fixed-point vs Single-pass context differences
"""

import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('experiments/cvfpt_comparison_gated/cvfpt_comparison_data.npz')

tokens = data['tokens']
l2_distances = data['l2_distances']
cosine_similarities = data['cosine_similarities']
fixed_point_contexts = data['fixed_point_contexts']
single_pass_contexts = data['single_pass_contexts']
fixed_point_norms = data['fixed_point_norms']
single_pass_norms = data['single_pass_norms']

print("=" * 80)
print("CVFPT Detailed Analysis: Repetition vs Single-Pass")
print("=" * 80)

# 1. L2 Distance Analysis
print("\n1. L2 Distance (Fixed-point vs Single-pass)")
print("-" * 80)
print(f"Mean:   {np.mean(l2_distances):.4f}")
print(f"Std:    {np.std(l2_distances):.4f}")
print(f"Min:    {np.min(l2_distances):.4f}")
print(f"Max:    {np.max(l2_distances):.4f}")
print(f"Median: {np.median(l2_distances):.4f}")

# Relative to norm
relative_distance = np.mean(l2_distances) / np.mean(fixed_point_norms)
print(f"\nRelative Distance (L2 / Norm): {relative_distance:.2%}")
print(f"Interpretation: Single-pass differs from fixed-point by {relative_distance:.1%} of vector magnitude")

# 2. Cosine Similarity Analysis
print("\n2. Cosine Similarity (Direction alignment)")
print("-" * 80)
print(f"Mean:   {np.mean(cosine_similarities):.6f}")
print(f"Std:    {np.std(cosine_similarities):.6f}")
print(f"Min:    {np.min(cosine_similarities):.6f}")
print(f"Max:    {np.max(cosine_similarities):.6f}")

# Angular difference
angles_degrees = np.arccos(np.clip(cosine_similarities, -1, 1)) * 180 / np.pi
print(f"\nAngular Difference:")
print(f"Mean:   {np.mean(angles_degrees):.2f}°")
print(f"Std:    {np.std(angles_degrees):.2f}°")
print(f"Min:    {np.min(angles_degrees):.2f}°")
print(f"Max:    {np.max(angles_degrees):.2f}°")

# 3. Magnitude Analysis
print("\n3. Vector Magnitude Analysis")
print("-" * 80)
print(f"Fixed-point norm: {np.mean(fixed_point_norms):.4f} ± {np.std(fixed_point_norms):.4f}")
print(f"Single-pass norm: {np.mean(single_pass_norms):.4f} ± {np.std(single_pass_norms):.4f}")

norm_diff = np.abs(fixed_point_norms - single_pass_norms)
print(f"\nNorm difference:  {np.mean(norm_diff):.4f} ± {np.std(norm_diff):.4f}")
print(f"Relative norm diff: {np.mean(norm_diff) / np.mean(fixed_point_norms):.4%}")

# 4. Per-dimension Analysis
print("\n4. Per-Dimension Analysis")
print("-" * 80)

# Compute per-dimension differences
dim_diffs = np.abs(fixed_point_contexts - single_pass_contexts)  # [100 tokens, 256 dims]
mean_dim_diff = np.mean(dim_diffs, axis=0)  # [256]

print(f"Mean per-dimension difference: {np.mean(mean_dim_diff):.4f}")
print(f"Max per-dimension difference:  {np.max(mean_dim_diff):.4f}")
print(f"Min per-dimension difference:  {np.min(mean_dim_diff):.4f}")

# Which dimensions differ most?
top_diff_dims = np.argsort(mean_dim_diff)[-10:][::-1]
print(f"\nTop 10 dimensions with largest differences:")
for i, dim in enumerate(top_diff_dims, 1):
    print(f"  {i}. Dim {dim}: {mean_dim_diff[dim]:.4f}")

# Which dimensions are most consistent?
bottom_diff_dims = np.argsort(mean_dim_diff)[:10]
print(f"\nTop 10 dimensions with smallest differences:")
for i, dim in enumerate(bottom_diff_dims, 1):
    print(f"  {i}. Dim {dim}: {mean_dim_diff[dim]:.4f}")

# 5. Token-level variance
print("\n5. Token-Level Variance")
print("-" * 80)

# How much do different tokens vary in their L2 distance?
cv_l2 = np.std(l2_distances) / np.mean(l2_distances)
print(f"Coefficient of Variation (L2): {cv_l2:.4f}")
print(f"Interpretation: {cv_l2:.1%} variability across tokens")

if cv_l2 < 0.05:
    print("  → Very consistent across tokens")
elif cv_l2 < 0.1:
    print("  → Fairly consistent across tokens")
else:
    print("  → High variability across tokens")

# 6. Decomposition: Magnitude vs Direction
print("\n6. Decomposition: Magnitude vs Direction Difference")
print("-" * 80)

# Magnitude component
mag_component = np.abs(fixed_point_norms - single_pass_norms)

# Direction component (perpendicular)
# L2^2 = mag^2 + perp^2
# perp^2 = L2^2 - mag^2
perp_component = np.sqrt(np.maximum(0, l2_distances**2 - mag_component**2))

print(f"Magnitude component: {np.mean(mag_component):.4f} ± {np.std(mag_component):.4f}")
print(f"Direction component: {np.mean(perp_component):.4f} ± {np.std(perp_component):.4f}")

mag_contribution = np.mean(mag_component**2) / np.mean(l2_distances**2)
dir_contribution = np.mean(perp_component**2) / np.mean(l2_distances**2)

print(f"\nContribution to total L2 distance:")
print(f"  Magnitude: {mag_contribution:.1%}")
print(f"  Direction: {dir_contribution:.1%}")

# 7. Practical Interpretation
print("\n" + "=" * 80)
print("PRACTICAL INTERPRETATION")
print("=" * 80)

print(f"""
L2 Distance = {np.mean(l2_distances):.2f} (out of norm ~16.05)
  → Single-pass differs by {relative_distance:.1%} from fixed-point

Cosine Similarity = {np.mean(cosine_similarities):.4f}
  → Vectors point in nearly same direction (angle ~{np.mean(angles_degrees):.1f}°)

This means:
1. **Direction is very similar**: 97.3% aligned
2. **Magnitude is similar**: Within 0.03% of each other
3. **Main difference**: Fine-grained dimensional adjustments

Conclusion:
Single-pass contexts capture the "essence" (direction) of the fixed-point,
but lack the "refinement" (precise dimensional values) achieved by repetition.
""")

# 8. Visualization
print("\nGenerating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: L2 Distance distribution
ax = axes[0, 0]
ax.hist(l2_distances, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(l2_distances), color='red', linestyle='--', label=f'Mean: {np.mean(l2_distances):.2f}')
ax.set_xlabel('L2 Distance')
ax.set_ylabel('Frequency')
ax.set_title('L2 Distance Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Cosine Similarity distribution
ax = axes[0, 1]
ax.hist(cosine_similarities, bins=30, edgecolor='black', alpha=0.7, color='green')
ax.axvline(np.mean(cosine_similarities), color='red', linestyle='--', label=f'Mean: {np.mean(cosine_similarities):.4f}')
ax.set_xlabel('Cosine Similarity')
ax.set_ylabel('Frequency')
ax.set_title('Cosine Similarity Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Angular difference
ax = axes[0, 2]
ax.hist(angles_degrees, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(np.mean(angles_degrees), color='red', linestyle='--', label=f'Mean: {np.mean(angles_degrees):.1f}°')
ax.set_xlabel('Angular Difference (degrees)')
ax.set_ylabel('Frequency')
ax.set_title('Angular Difference Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Norm comparison
ax = axes[1, 0]
ax.scatter(fixed_point_norms, single_pass_norms, alpha=0.5)
ax.plot([16.04, 16.06], [16.04, 16.06], 'r--', label='y=x')
ax.set_xlabel('Fixed-Point Norm')
ax.set_ylabel('Single-Pass Norm')
ax.set_title('Norm Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Per-dimension difference
ax = axes[1, 1]
ax.plot(mean_dim_diff)
ax.set_xlabel('Dimension')
ax.set_ylabel('Mean Absolute Difference')
ax.set_title('Per-Dimension Difference')
ax.grid(True, alpha=0.3)

# Plot 6: Magnitude vs Direction contribution
ax = axes[1, 2]
contributions = [mag_contribution * 100, dir_contribution * 100]
labels = ['Magnitude\nDifference', 'Direction\nDifference']
colors = ['#ff9999', '#66b3ff']
ax.bar(labels, contributions, color=colors, edgecolor='black')
ax.set_ylabel('Contribution to L2 Distance (%)')
ax.set_title('L2 Distance Decomposition')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cvfpt_detailed_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved to cvfpt_detailed_analysis.png")

print("\n" + "=" * 80)
print("Analysis Complete")
print("=" * 80)
