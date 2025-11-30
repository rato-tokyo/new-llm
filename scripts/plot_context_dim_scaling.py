#!/usr/bin/env python3
"""
Context Dim Scaling Experiment Visualization

context_dim変化によるスケーリング特性の可視化
768d (1x params) vs 1200d (2x params) vs 1537d (3x params)
"""

import json
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
        {"samples": 50, "tokens": 62891, "val_ppl": 683.9, "train_ppl": 170.6, "val_er": 0.776},
        {"samples": 100, "tokens": 122795, "val_ppl": 415.8, "train_ppl": 150.8, "val_er": 0.775},
        {"samples": 200, "tokens": 240132, "val_ppl": 294.5, "train_ppl": 129.0, "val_er": 0.775},
        {"samples": 500, "tokens": 587970, "val_ppl": 198.2, "train_ppl": 108.3, "val_er": 0.775},
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
        {"samples": 50, "tokens": 62891, "val_ppl": 639.1, "train_ppl": 154.9, "val_er": 0.726},
        {"samples": 100, "tokens": 122795, "val_ppl": 407.0, "train_ppl": 139.5, "val_er": 0.724},
        {"samples": 200, "tokens": 240132, "val_ppl": 296.6, "train_ppl": 117.7, "val_er": 0.725},
        {"samples": 500, "tokens": 587970, "val_ppl": 199.5, "train_ppl": 100.6, "val_er": 0.724},
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
        {"samples": 50, "tokens": 62891, "val_ppl": 632.6, "train_ppl": 160.9, "val_er": 0.689},
        {"samples": 100, "tokens": 122795, "val_ppl": 400.2, "train_ppl": 139.9, "val_er": 0.687},
        {"samples": 200, "tokens": 240132, "val_ppl": 294.6, "train_ppl": 116.6, "val_er": 0.689},
        {"samples": 500, "tokens": 587970, "val_ppl": 198.3, "train_ppl": 98.3, "val_er": 0.686},
    ]
}

all_configs = [baseline_768d, data_1200d, data_1537d]

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

plt.show()
