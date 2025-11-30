#!/usr/bin/env python3
# mypy: ignore-errors
"""
Context Dim Scaling Experiment Visualization

context_dim変化によるスケーリング特性の可視化
768d (1x params) vs 1200d (2x params) vs 1537d (3x params)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data from experiments
# 768d baseline from matrix experiment
baseline_768d = {
    "config_name": "1L_768d_1tok",
    "context_dim": 768,
    "params_ratio": 1.0,
    "alpha": -0.5460,
    "A": 267189,
    "r_squared": 0.984,
    "best_ppl": 198.2,
    "best_acc": 0.226,
    "best_train_ppl": 108.3,
    "best_val_er": 0.775,
    "samples_data": [
        {"samples": 50, "tokens": 62891, "val_ppl": 683.9, "train_ppl": 170.6, "val_er": 0.776, "train_er": 0.779, "p1_iter": 40, "final_conv": 0.76, "val_acc": 0.173},
        {"samples": 100, "tokens": 122795, "val_ppl": 415.8, "train_ppl": 150.8, "val_er": 0.775, "train_er": 0.783, "p1_iter": 40, "final_conv": 0.75, "val_acc": 0.191},
        {"samples": 200, "tokens": 240132, "val_ppl": 294.5, "train_ppl": 129.0, "val_er": 0.775, "train_er": 0.784, "p1_iter": 40, "final_conv": 0.74, "val_acc": 0.201},
        {"samples": 500, "tokens": 587970, "val_ppl": 198.2, "train_ppl": 108.3, "val_er": 0.775, "train_er": 0.786, "p1_iter": 40, "final_conv": 0.76, "val_acc": 0.226},
    ]
}

# 1200d from context_dim experiment (2x params)
data_1200d = {
    "config_name": "1L_1200d_1tok",
    "context_dim": 1200,
    "params_ratio": 2.0,
    "alpha": -0.5133,
    "A": 176398,
    "r_squared": 0.989,
    "best_ppl": 199.5,
    "best_acc": 0.231,
    "best_train_ppl": 100.6,
    "best_val_er": 0.724,
    "samples_data": [
        {"samples": 50, "tokens": 62891, "val_ppl": 639.1, "train_ppl": 154.9, "val_er": 0.726, "train_er": 0.732, "p1_iter": 40, "final_conv": 0.71, "val_acc": 0.172},
        {"samples": 100, "tokens": 122795, "val_ppl": 407.0, "train_ppl": 139.5, "val_er": 0.724, "train_er": 0.735, "p1_iter": 40, "final_conv": 0.71, "val_acc": 0.192},
        {"samples": 200, "tokens": 240132, "val_ppl": 296.6, "train_ppl": 117.7, "val_er": 0.725, "train_er": 0.738, "p1_iter": 40, "final_conv": 0.64, "val_acc": 0.207},
        {"samples": 500, "tokens": 587970, "val_ppl": 199.5, "train_ppl": 100.6, "val_er": 0.724, "train_er": 0.739, "p1_iter": 40, "final_conv": 0.69, "val_acc": 0.231},
    ]
}

# 1537d from context_dim experiment (3x params)
data_1537d = {
    "config_name": "1L_1537d_1tok",
    "context_dim": 1537,
    "params_ratio": 3.0,
    "alpha": -0.5103,
    "A": 168322,
    "r_squared": 0.988,
    "best_ppl": 198.3,
    "best_acc": 0.230,
    "best_train_ppl": 98.3,
    "best_val_er": 0.686,
    "samples_data": [
        {"samples": 50, "tokens": 62891, "val_ppl": 632.6, "train_ppl": 160.9, "val_er": 0.689, "train_er": 0.696, "p1_iter": 40, "final_conv": 0.75, "val_acc": 0.171},
        {"samples": 100, "tokens": 122795, "val_ppl": 400.2, "train_ppl": 139.9, "val_er": 0.687, "train_er": 0.702, "p1_iter": 40, "final_conv": 0.75, "val_acc": 0.190},
        {"samples": 200, "tokens": 240132, "val_ppl": 294.6, "train_ppl": 116.6, "val_er": 0.689, "train_er": 0.705, "p1_iter": 40, "final_conv": 0.70, "val_acc": 0.202},
        {"samples": 500, "tokens": 587970, "val_ppl": 198.3, "train_ppl": 98.3, "val_er": 0.686, "train_er": 0.705, "p1_iter": 40, "final_conv": 0.74, "val_acc": 0.230},
    ]
}

all_configs_v1 = [baseline_768d, data_1200d, data_1537d]

# ============================================================
# v2 Data (dist_reg_weight=0.9, noise=0.0, epochs=20)
# ============================================================
v2_768d = {
    "config_name": "1L_768d_1tok",
    "context_dim": 768,
    "params_ratio": 1.0,
    "alpha": -0.4761,
    "A": 106737,
    "r_squared": 0.993,
    "best_ppl": 196.3,
    "best_acc": 0.228,
    "best_train_ppl": 93.4,
    "best_val_er": 0.792,
    "samples_data": [
        {"samples": 50, "tokens": 62891, "val_ppl": 576.0, "train_ppl": 65.9, "val_er": 0.793, "train_er": 0.797, "p1_iter": 23, "val_acc": 0.183},
        {"samples": 100, "tokens": 122795, "val_ppl": 386.5, "train_ppl": 91.3, "val_er": 0.792, "train_er": 0.799, "p1_iter": 23, "val_acc": 0.197},
        {"samples": 200, "tokens": 240132, "val_ppl": 286.4, "train_ppl": 94.0, "val_er": 0.791, "train_er": 0.800, "p1_iter": 25, "val_acc": 0.204},
        {"samples": 500, "tokens": 587970, "val_ppl": 196.3, "train_ppl": 93.4, "val_er": 0.792, "train_er": 0.803, "p1_iter": 23, "val_acc": 0.228},
    ]
}

v2_1200d = {
    "config_name": "1L_1200d_1tok",
    "context_dim": 1200,
    "params_ratio": 2.0,
    "alpha": -0.4853,
    "A": 123874,
    "r_squared": 0.995,
    "best_ppl": 201.1,
    "best_acc": 0.230,
    "best_train_ppl": 90.1,
    "best_val_er": 0.756,
    "samples_data": [
        {"samples": 50, "tokens": 62891, "val_ppl": 601.4, "train_ppl": 81.4, "val_er": 0.757, "train_er": 0.762, "p1_iter": 29, "val_acc": 0.179},
        {"samples": 100, "tokens": 122795, "val_ppl": 404.9, "train_ppl": 88.0, "val_er": 0.757, "train_er": 0.764, "p1_iter": 29, "val_acc": 0.197},
        {"samples": 200, "tokens": 240132, "val_ppl": 297.5, "train_ppl": 100.9, "val_er": 0.754, "train_er": 0.764, "p1_iter": 31, "val_acc": 0.206},
        {"samples": 500, "tokens": 587970, "val_ppl": 201.1, "train_ppl": 90.1, "val_er": 0.756, "train_er": 0.769, "p1_iter": 29, "val_acc": 0.230},
    ]
}

v2_1537d = {
    "config_name": "1L_1537d_1tok",
    "context_dim": 1537,
    "params_ratio": 3.0,
    "alpha": -0.4717,
    "A": 104896,
    "r_squared": 0.996,
    "best_ppl": 202.3,
    "best_acc": 0.229,
    "best_train_ppl": 93.5,
    "best_val_er": 0.726,
    "samples_data": [
        {"samples": 50, "tokens": 62891, "val_ppl": 588.6, "train_ppl": 79.3, "val_er": 0.728, "train_er": 0.734, "p1_iter": 32, "val_acc": 0.176},
        {"samples": 100, "tokens": 122795, "val_ppl": 401.6, "train_ppl": 97.0, "val_er": 0.729, "train_er": 0.736, "p1_iter": 32, "val_acc": 0.191},
        {"samples": 200, "tokens": 240132, "val_ppl": 301.8, "train_ppl": 98.1, "val_er": 0.724, "train_er": 0.736, "p1_iter": 34, "val_acc": 0.204},
        {"samples": 500, "tokens": 587970, "val_ppl": 202.3, "train_ppl": 93.5, "val_er": 0.726, "train_er": 0.741, "p1_iter": 32, "val_acc": 0.229},
    ]
}

all_configs_v2 = [v2_768d, v2_1200d, v2_1537d]

# Default to v1 for backwards compatibility
all_configs = all_configs_v1

# Create output directory
output_dir = "importants"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Colors for each config
colors = ['#2ecc71', '#3498db', '#9b59b6']  # green, blue, purple
markers = ['o', 's', '^']

# Figure 1: Scaling Laws Comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Plot 1: Scaling Laws (log-log)
ax1 = axes[0]
for i, config in enumerate(all_configs):
    tokens = [d["tokens"] for d in config["samples_data"]]
    ppls = [d["val_ppl"] for d in config["samples_data"]]
    label = f'{config["context_dim"]}d ({config["params_ratio"]:.0f}x)'
    ax1.scatter(tokens, ppls, color=colors[i], marker=markers[i], s=80, label=label, zorder=5)

    # Fit line
    log_tokens = np.linspace(np.log(min(tokens)*0.8), np.log(max(tokens)*1.2), 100)
    log_ppl = np.log(config["A"]) + config["alpha"] * log_tokens
    ax1.plot(np.exp(log_tokens), np.exp(log_ppl), color=colors[i], linestyle='--', alpha=0.7, linewidth=2)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Training Tokens')
ax1.set_ylabel('Validation PPL')
ax1.set_title('Scaling Laws: context_dim Comparison')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Alpha and A values
ax2 = axes[1]
x_pos = np.arange(len(all_configs))
width = 0.35

# Alpha values (left y-axis)
alphas = [config["alpha"] for config in all_configs]
bars1 = ax2.bar(x_pos - width/2, [-a for a in alphas], width, color=colors, alpha=0.8, label='|α|')
ax2.set_ylabel('|α| (higher = better scaling)')
ax2.set_ylim(0.45, 0.60)

# A values (right y-axis)
ax2_twin = ax2.twinx()
A_values = [config["A"] / 1e5 for config in all_configs]
bars2 = ax2_twin.bar(x_pos + width/2, A_values, width, color=colors, alpha=0.4, hatch='//', label='A (×10⁵)')
ax2_twin.set_ylabel('A (×10⁵)')

ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{c["context_dim"]}d\n({c["params_ratio"]:.0f}x)' for c in all_configs])
ax2.set_title('Scaling Parameters: α and A')

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Plot 3: Performance at 500 samples
ax3 = axes[2]
metrics = ['Val PPL', 'Train PPL', 'Val Acc (%)']
x_pos = np.arange(len(metrics))
width = 0.25

for i, config in enumerate(all_configs):
    values = [
        config["best_ppl"],
        config["best_train_ppl"],
        config["best_acc"] * 100
    ]
    ax3.bar(x_pos + i*width - width, values, width, color=colors[i], alpha=0.8,
            label=f'{config["context_dim"]}d ({config["params_ratio"]:.0f}x)')

ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics)
ax3.set_title('Performance at 500 Samples')
ax3.legend(loc='upper right')
ax3.set_ylabel('Value')

plt.tight_layout()
plt.savefig(f'{output_dir}/context_dim_scaling.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/context_dim_scaling.png")

# Figure 2: Detailed Analysis
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))

# Plot 1: Effective Rank comparison
ax1 = axes2[0]
for i, config in enumerate(all_configs):
    ers = [d["val_er"] * 100 for d in config["samples_data"]]
    samples = [d["samples"] for d in config["samples_data"]]
    label = f'{config["context_dim"]}d'
    ax1.plot(samples, ers, color=colors[i], marker=markers[i], markersize=8, linewidth=2, label=label)

ax1.set_xlabel('Samples')
ax1.set_ylabel('Effective Rank (%)')
ax1.set_title('Effective Rank vs Samples')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(65, 85)

# Plot 2: Parameter efficiency (PPL per param)
ax2 = axes2[1]
# Calculate context_block params: (context_dim + embed_dim) * context_dim + 3*context_dim
embed_dim = 768
params = [(c["context_dim"] + embed_dim) * c["context_dim"] + 3*c["context_dim"] for c in all_configs]
ppls = [c["best_ppl"] for c in all_configs]
efficiency = [ppl / (p / 1e6) for ppl, p in zip(ppls, params)]  # PPL per million params

bars = ax2.bar(range(len(all_configs)), efficiency, color=colors, alpha=0.8)
ax2.set_xticks(range(len(all_configs)))
ax2.set_xticklabels([f'{c["context_dim"]}d' for c in all_configs])
ax2.set_ylabel('PPL / Million Params')
ax2.set_title('Parameter Efficiency\n(lower = better)')

# Add value labels
for bar, eff in zip(bars, efficiency):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{eff:.1f}', ha='center', va='bottom', fontsize=9)

# Plot 3: Params vs PPL tradeoff
ax3 = axes2[2]
params_millions = [p / 1e6 for p in params]
for i, config in enumerate(all_configs):
    ax3.scatter(params_millions[i], config["best_ppl"],
                color=colors[i], marker=markers[i], s=150,
                label=f'{config["context_dim"]}d', zorder=5)
    ax3.annotate(f'{config["context_dim"]}d',
                 (params_millions[i], config["best_ppl"]),
                 textcoords="offset points", xytext=(10, 5), fontsize=9)

ax3.set_xlabel('ContextBlock Params (Millions)')
ax3.set_ylabel('Best Val PPL')
ax3.set_title('Params vs PPL Tradeoff')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/context_dim_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/context_dim_analysis.png")

# Print summary
print("\n" + "="*60)
print("CONTEXT DIM SCALING EXPERIMENT SUMMARY")
print("="*60)

print("\n### Scaling Parameters ###")
print(f"{'Config':<15} {'context_dim':>10} {'Params':>10} {'α':>10} {'A':>12} {'R²':>8}")
print("-"*65)
for config in all_configs:
    p = (config["context_dim"] + embed_dim) * config["context_dim"] + 3*config["context_dim"]
    print(f"{config['config_name']:<15} {config['context_dim']:>10} {p/1e6:>9.2f}M {config['alpha']:>10.4f} {config['A']:>12.0f} {config['r_squared']:>8.3f}")

print("\n### Performance at 500 Samples ###")
print(f"{'Config':<15} {'Val PPL':>10} {'Train PPL':>10} {'Val Acc':>10} {'Val ER':>10}")
print("-"*55)
for config in all_configs:
    print(f"{config['config_name']:<15} {config['best_ppl']:>10.1f} {config['best_train_ppl']:>10.1f} {config['best_acc']*100:>9.1f}% {config['best_val_er']*100:>9.1f}%")

print("\n### Key Findings ###")
print("1. α値は768dが最高 (-0.546) → パラメータ増加でスケーリング効率低下")
print("2. A値は大きいcontext_dimで低い → 少データでの初期性能は良い")
print("3. 最終PPLはほぼ同等 (198-199) → パラメータ増加の効果は限定的")
print("4. Effective Rankは768dが最高 → 大きいcontext_dimは次元を活用しきれていない")
print("5. パラメータ効率は768dが圧倒的に良い")

# Detailed results table (all sample sizes)
print("\n" + "="*120)
print("DETAILED RESULTS (All Sample Sizes)")
print("="*120)
header = f"{'Config':<15} {'Samples':>8} {'Tokens':>10} {'P1 Iter':>8} {'Conv':>6} {'Train ER':>9} {'Val ER':>8} {'Train PPL':>10} {'Val PPL':>9} {'Val Acc':>8}"
print(header)
print("-"*120)

for config in all_configs:
    for data in config["samples_data"]:
        print(f"{config['config_name']:<15} "
              f"{data['samples']:>8} "
              f"{data['tokens']:>10,} "
              f"{data['p1_iter']:>8} "
              f"{data['final_conv']*100:>5.0f}% "
              f"{data['train_er']*100:>8.1f}% "
              f"{data['val_er']*100:>7.1f}% "
              f"{data['train_ppl']:>10.1f} "
              f"{data['val_ppl']:>9.1f} "
              f"{data['val_acc']*100:>7.1f}%")
    print()  # Empty line between configs

# Summary comparison at 500 samples
print("="*120)
print("SUMMARY COMPARISON (500 Samples)")
print("="*120)
header = f"{'Config':<15} {'context_dim':>10} {'Params':>10} {'P1 Iter':>8} {'Conv':>6} {'Val ER':>8} {'Train PPL':>10} {'Val PPL':>9} {'Val Acc':>8} {'α':>8}"
print(header)
print("-"*120)

for config in all_configs:
    p = (config["context_dim"] + embed_dim) * config["context_dim"] + 3*config["context_dim"]
    data_500 = [d for d in config["samples_data"] if d["samples"] == 500][0]
    print(f"{config['config_name']:<15} "
          f"{config['context_dim']:>10} "
          f"{p/1e6:>9.2f}M "
          f"{data_500['p1_iter']:>8} "
          f"{data_500['final_conv']*100:>5.0f}% "
          f"{data_500['val_er']*100:>7.1f}% "
          f"{data_500['train_ppl']:>10.1f} "
          f"{data_500['val_ppl']:>9.1f} "
          f"{config['best_acc']*100:>7.1f}% "
          f"{config['alpha']:>8.4f}")

print("="*120)

# ============================================================
# Figure 3: v1 vs v2 Comparison
# ============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Alpha comparison (v1 vs v2)
ax = axes3[0, 0]
x_pos = np.arange(3)
width = 0.35
v1_alphas = [-c["alpha"] for c in all_configs_v1]
v2_alphas = [-c["alpha"] for c in all_configs_v2]
ax.bar(x_pos - width/2, v1_alphas, width, label='v1', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, v2_alphas, width, label='v2', color='#e74c3c', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(['768d', '1200d', '1537d'])
ax.set_ylabel('|α| (higher = better scaling)')
ax.set_title('α Values: v1 vs v2')
ax.legend()
ax.set_ylim(0.4, 0.6)

# Plot 2: Effective Rank comparison
ax = axes3[0, 1]
v1_ers = [c["best_val_er"] * 100 for c in all_configs_v1]
v2_ers = [c["best_val_er"] * 100 for c in all_configs_v2]
ax.bar(x_pos - width/2, v1_ers, width, label='v1', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, v2_ers, width, label='v2', color='#e74c3c', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(['768d', '1200d', '1537d'])
ax.set_ylabel('Effective Rank (%)')
ax.set_title('Effective Rank: v1 vs v2 (↑ = better)')
ax.legend()
ax.set_ylim(65, 85)

# Add improvement annotations
for i, (v1, v2) in enumerate(zip(v1_ers, v2_ers)):
    diff = v2 - v1
    ax.annotate(f'+{diff:.1f}%', (i + width/2, v2 + 0.5), ha='center', fontsize=9, color='green')

# Plot 3: PPL comparison
ax = axes3[1, 0]
v1_ppls = [c["best_ppl"] for c in all_configs_v1]
v2_ppls = [c["best_ppl"] for c in all_configs_v2]
ax.bar(x_pos - width/2, v1_ppls, width, label='v1', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, v2_ppls, width, label='v2', color='#e74c3c', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(['768d', '1200d', '1537d'])
ax.set_ylabel('Val PPL (↓ = better)')
ax.set_title('Best Val PPL: v1 vs v2')
ax.legend()
ax.set_ylim(190, 210)

# Plot 4: P1 Iterations comparison
ax = axes3[1, 1]
v1_iters = [c["samples_data"][-1].get("p1_iter", 40) for c in all_configs_v1]
v2_iters = [c["samples_data"][-1]["p1_iter"] for c in all_configs_v2]
ax.bar(x_pos - width/2, v1_iters, width, label='v1', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, v2_iters, width, label='v2', color='#e74c3c', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(['768d', '1200d', '1537d'])
ax.set_ylabel('Phase 1 Iterations (↓ = faster)')
ax.set_title('Convergence Speed: v1 vs v2')
ax.legend()

# Add improvement annotations
for i, (v1, v2) in enumerate(zip(v1_iters, v2_iters)):
    diff_pct = (v2 - v1) / v1 * 100
    ax.annotate(f'{diff_pct:.0f}%', (i + width/2, v2 + 1), ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.savefig(f'{output_dir}/context_dim_v1_v2_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/context_dim_v1_v2_comparison.png")

# Print v1 vs v2 comparison summary
print("\n" + "="*80)
print("V1 vs V2 COMPARISON SUMMARY")
print("="*80)
print("\nSetting changes: dist_reg_weight 0.8→0.9, noise 0.1→0.0, epochs 10→20")
print("\n" + "-"*80)
print(f"{'Metric':<20} {'768d v1':>10} {'768d v2':>10} {'1200d v1':>10} {'1200d v2':>10} {'1537d v1':>10} {'1537d v2':>10}")
print("-"*80)
print(f"{'α':.<20} {all_configs_v1[0]['alpha']:>10.4f} {all_configs_v2[0]['alpha']:>10.4f} {all_configs_v1[1]['alpha']:>10.4f} {all_configs_v2[1]['alpha']:>10.4f} {all_configs_v1[2]['alpha']:>10.4f} {all_configs_v2[2]['alpha']:>10.4f}")
print(f"{'Val ER (%)':.<20} {all_configs_v1[0]['best_val_er']*100:>10.1f} {all_configs_v2[0]['best_val_er']*100:>10.1f} {all_configs_v1[1]['best_val_er']*100:>10.1f} {all_configs_v2[1]['best_val_er']*100:>10.1f} {all_configs_v1[2]['best_val_er']*100:>10.1f} {all_configs_v2[2]['best_val_er']*100:>10.1f}")
print(f"{'Val PPL':.<20} {all_configs_v1[0]['best_ppl']:>10.1f} {all_configs_v2[0]['best_ppl']:>10.1f} {all_configs_v1[1]['best_ppl']:>10.1f} {all_configs_v2[1]['best_ppl']:>10.1f} {all_configs_v1[2]['best_ppl']:>10.1f} {all_configs_v2[2]['best_ppl']:>10.1f}")
print(f"{'P1 Iter':.<20} {v1_iters[0]:>10} {v2_iters[0]:>10} {v1_iters[1]:>10} {v2_iters[1]:>10} {v1_iters[2]:>10} {v2_iters[2]:>10}")
print("="*80)

plt.show()
