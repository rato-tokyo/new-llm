"""
Sample Size Scaling Law Analysis

べき乗則: PPL = A * samples^(-a)
または: log(PPL) = log(A) - a * log(samples)

実験データからスケーリング係数 a と定数 A を求める。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def analyze_scaling_law():
    """サンプルサイズとPPLの関係を分析"""

    # 実験データ
    samples = np.array([100, 200, 400, 800, 1600])
    ppls = np.array([321.4, 253.1, 186.0, 155.8, 134.7])

    # 対数変換
    log_samples = np.log(samples)
    log_ppls = np.log(ppls)

    # 線形回帰（全データ）
    slope_all, intercept_all, r_all, p_all, se_all = stats.linregress(log_samples, log_ppls)
    a_all = -slope_all
    A_all = np.exp(intercept_all)

    print("=" * 70)
    print("SAMPLE SIZE SCALING LAW ANALYSIS")
    print("=" * 70)
    print("\nModel: PPL = A × samples^(-a)")
    print("\n全データ (100-1600):")
    print(f"  a = {a_all:.4f}")
    print(f"  A = {A_all:.2f}")
    print(f"  R² = {r_all**2:.4f}")

    # 区間ごとの分析
    print("\n" + "-" * 70)
    print("区間ごとの分析:")
    print("-" * 70)

    intervals = [
        (0, 3, "100-400"),
        (1, 4, "200-800"),
        (2, 5, "400-1600"),
    ]

    results = []
    for start, end, label in intervals:
        x = log_samples[start:end]
        y = log_ppls[start:end]
        slope, intercept, r, p, se = stats.linregress(x, y)
        a = -slope
        A = np.exp(intercept)
        results.append((label, a, A, r**2))
        print(f"\n  {label}:")
        print(f"    a = {a:.4f}")
        print(f"    A = {A:.2f}")
        print(f"    R² = {r**2:.4f}")

    # 予測値と実測値の比較
    print("\n" + "-" * 70)
    print("予測値 vs 実測値 (全データフィット):")
    print("-" * 70)
    print(f"{'Samples':>10} | {'Actual PPL':>12} | {'Predicted PPL':>14} | {'Error':>8}")
    print("-" * 50)

    for s, actual_ppl in zip(samples, ppls):
        predicted_ppl = A_all * (s ** (-a_all))
        error = (predicted_ppl - actual_ppl) / actual_ppl * 100
        print(f"{s:>10} | {actual_ppl:>12.1f} | {predicted_ppl:>14.1f} | {error:>+7.1f}%")

    # 外挿予測
    print("\n" + "-" * 70)
    print("外挿予測:")
    print("-" * 70)

    future_samples = [3200, 6400, 10000, 50000, 100000]
    for s in future_samples:
        predicted = A_all * (s ** (-a_all))
        print(f"  {s:>6} samples → PPL ≈ {predicted:.1f}")

    # グラフ作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 線形スケール
    ax1 = axes[0]
    ax1.scatter(samples, ppls, s=100, c='blue', zorder=5, label='Experimental')

    # フィット曲線
    x_fit = np.linspace(50, 2000, 100)
    y_fit = A_all * (x_fit ** (-a_all))
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: PPL = {A_all:.0f} × n^(-{a_all:.3f})')

    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('Validation PPL', fontsize=12)
    ax1.set_title('Sample Size vs PPL (Linear Scale)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右: 対数スケール
    ax2 = axes[1]
    ax2.scatter(samples, ppls, s=100, c='blue', zorder=5, label='Experimental')

    # フィット線
    x_fit_log = np.logspace(1.8, 3.5, 100)
    y_fit_log = A_all * (x_fit_log ** (-a_all))
    ax2.plot(x_fit_log, y_fit_log, 'r-', linewidth=2, label=f'Fit: a={a_all:.3f}, R²={r_all**2:.4f}')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size (log)', fontsize=12)
    ax2.set_ylabel('Validation PPL (log)', fontsize=12)
    ax2.set_title('Sample Size vs PPL (Log-Log Scale)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('importants/logs/sample_size_scaling_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Graph saved: importants/logs/sample_size_scaling_analysis.png")

    # 区間ごとのグラフ
    fig2, ax3 = plt.subplots(figsize=(10, 6))

    colors = ['red', 'green', 'purple']
    for (label, a, A, r2), color in zip(results, colors):
        x_range = np.logspace(1.8, 3.5, 100)
        y_range = A * (x_range ** (-a))
        ax3.plot(x_range, y_range, '-', color=color, linewidth=1.5,
                 label=f'{label}: a={a:.3f}, A={A:.0f}')

    ax3.scatter(samples, ppls, s=100, c='blue', zorder=5, marker='o', label='Experimental')

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Sample Size (log)', fontsize=12)
    ax3.set_ylabel('Validation PPL (log)', fontsize=12)
    ax3.set_title('Scaling Law by Interval', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('importants/logs/sample_size_scaling_by_interval.png', dpi=150, bbox_inches='tight')
    print("✓ Graph saved: importants/logs/sample_size_scaling_by_interval.png")

    plt.show()

    return {
        'a': a_all,
        'A': A_all,
        'r_squared': r_all**2,
        'intervals': results
    }


if __name__ == "__main__":
    results = analyze_scaling_law()
