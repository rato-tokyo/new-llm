"""
Context Dim Comparison Analysis

context_dim=256 と context_dim=320 の実験結果を比較分析する。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def main():
    # 実験データ
    samples = np.array([100, 200, 400, 800, 1600])

    # context_dim=256 の結果
    ppls_256 = np.array([321.4, 253.1, 186.0, 155.8, 134.7])
    er_256 = np.array([68.4, 68.1, 68.9, 68.1, 67.0])
    acc_256 = np.array([19.1, 19.9, 21.4, 22.6, 23.6])

    # context_dim=320 の結果
    ppls_320 = np.array([319.9, 250.1, 189.0, 156.0, 132.6])
    er_320 = np.array([63.6, 65.5, 65.8, 65.2, 65.8])
    acc_320 = np.array([18.9, 20.1, 21.5, 22.6, 23.4])

    print("=" * 70)
    print("CONTEXT DIM COMPARISON ANALYSIS")
    print("=" * 70)

    # ========================================
    # 基本比較
    # ========================================
    print("\n" + "-" * 70)
    print("PPL Comparison")
    print("-" * 70)
    print(f"{'Samples':>8} | {'dim=256':>10} | {'dim=320':>10} | {'Diff':>8} | {'%':>8}")
    print("-" * 55)

    for i, s in enumerate(samples):
        diff = ppls_320[i] - ppls_256[i]
        pct = diff / ppls_256[i] * 100
        print(f"{s:>8} | {ppls_256[i]:>10.1f} | {ppls_320[i]:>10.1f} | {diff:>+8.1f} | {pct:>+7.1f}%")

    print("-" * 55)
    avg_diff = np.mean(ppls_320 - ppls_256)
    avg_pct = np.mean((ppls_320 - ppls_256) / ppls_256 * 100)
    print(f"{'Average':>8} | {'-':>10} | {'-':>10} | {avg_diff:>+8.1f} | {avg_pct:>+7.1f}%")

    # ========================================
    # Effective Rank 比較
    # ========================================
    print("\n" + "-" * 70)
    print("Effective Rank Comparison")
    print("-" * 70)
    print(f"{'Samples':>8} | {'dim=256':>12} | {'dim=320':>12} | {'Abs Diff':>10}")
    print("-" * 55)

    for i, s in enumerate(samples):
        # 絶対値での比較
        er_abs_256 = er_256[i] * 256 / 100
        er_abs_320 = er_320[i] * 320 / 100
        diff = er_abs_320 - er_abs_256
        print(f"{s:>8} | {er_256[i]:>5.1f}% ({er_abs_256:>5.1f}) | {er_320[i]:>5.1f}% ({er_abs_320:>5.1f}) | {diff:>+9.1f}")

    print("-" * 55)
    print("Note: Abs = absolute effective rank (dimensions used)")

    # ========================================
    # 飽和モデルフィッティング
    # ========================================
    print("\n" + "-" * 70)
    print("Saturation Model Fitting: PPL = PPL_min + A × n^(-a)")
    print("-" * 70)

    def saturation_model(n, ppl_min, A, a):
        return ppl_min + A * (n ** (-a))

    # dim=256
    try:
        popt_256, _ = curve_fit(
            saturation_model, samples, ppls_256,
            p0=[80, 1000, 0.3],
            bounds=([0, 0, 0], [200, 10000, 2]),
            maxfev=10000
        )
        ppl_min_256, A_256, a_256 = popt_256
        pred_256 = saturation_model(samples, *popt_256)
        ss_res_256 = np.sum((ppls_256 - pred_256) ** 2)
        ss_tot_256 = np.sum((ppls_256 - np.mean(ppls_256)) ** 2)
        r2_256 = 1 - ss_res_256 / ss_tot_256

        print(f"\ndim=256:")
        print(f"  PPL_min = {ppl_min_256:.2f}")
        print(f"  A = {A_256:.2f}")
        print(f"  a = {a_256:.4f}")
        print(f"  R² = {r2_256:.6f}")
    except Exception as e:
        print(f"dim=256 fitting failed: {e}")
        ppl_min_256, A_256, a_256 = None, None, None

    # dim=320
    try:
        popt_320, _ = curve_fit(
            saturation_model, samples, ppls_320,
            p0=[80, 1000, 0.3],
            bounds=([0, 0, 0], [200, 10000, 2]),
            maxfev=10000
        )
        ppl_min_320, A_320, a_320 = popt_320
        pred_320 = saturation_model(samples, *popt_320)
        ss_res_320 = np.sum((ppls_320 - pred_320) ** 2)
        ss_tot_320 = np.sum((ppls_320 - np.mean(ppls_320)) ** 2)
        r2_320 = 1 - ss_res_320 / ss_tot_320

        print(f"\ndim=320:")
        print(f"  PPL_min = {ppl_min_320:.2f}")
        print(f"  A = {A_320:.2f}")
        print(f"  a = {a_320:.4f}")
        print(f"  R² = {r2_320:.6f}")
    except Exception as e:
        print(f"dim=320 fitting failed: {e}")
        ppl_min_320, A_320, a_320 = None, None, None

    # ========================================
    # 外挿予測比較
    # ========================================
    print("\n" + "-" * 70)
    print("Extrapolation Predictions (Saturation Model)")
    print("-" * 70)

    future_samples = [3200, 6400, 10000, 50000]
    print(f"{'Samples':>8} | {'dim=256':>10} | {'dim=320':>10} | {'Diff':>8}")
    print("-" * 45)

    for s in future_samples:
        if ppl_min_256 is not None and ppl_min_320 is not None:
            pred_256 = saturation_model(s, ppl_min_256, A_256, a_256)
            pred_320 = saturation_model(s, ppl_min_320, A_320, a_320)
            diff = pred_320 - pred_256
            print(f"{s:>8} | {pred_256:>10.1f} | {pred_320:>10.1f} | {diff:>+8.1f}")

    if ppl_min_256 is not None and ppl_min_320 is not None:
        print("-" * 45)
        print(f"{'PPL_min':>8} | {ppl_min_256:>10.1f} | {ppl_min_320:>10.1f} | {ppl_min_320 - ppl_min_256:>+8.1f}")

    # ========================================
    # 結論
    # ========================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print("\n1. PPL差はほぼなし:")
    print(f"   平均差: {avg_pct:+.1f}% (dim=320 vs dim=256)")
    print("   → context_dimの増加はPPLに大きな影響を与えない")

    print("\n2. Effective Rank の低下:")
    print(f"   dim=256: {np.mean(er_256):.1f}%")
    print(f"   dim=320: {np.mean(er_320):.1f}%")
    print("   → 次元を増やしても利用率は下がる")

    if ppl_min_256 is not None and ppl_min_320 is not None:
        print(f"\n3. 飽和点 (PPL_min):")
        print(f"   dim=256: {ppl_min_256:.1f}")
        print(f"   dim=320: {ppl_min_320:.1f}")
        print(f"   差: {ppl_min_320 - ppl_min_256:+.1f}")
        if abs(ppl_min_320 - ppl_min_256) < 10:
            print("   → 飽和点はcontext_dimにほぼ依存しない")

    print("\n4. 実用的含意:")
    print("   - context_dimを増やしてもPPL改善は限定的")
    print("   - ボトルネックはデータ量またはアーキテクチャ")
    print("   - context_dim=256で十分な可能性")

    # ========================================
    # グラフ作成
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上: PPL比較 (線形)
    ax1 = axes[0, 0]
    ax1.scatter(samples, ppls_256, s=80, c='blue', marker='o', label='dim=256', zorder=5)
    ax1.scatter(samples, ppls_320, s=80, c='red', marker='s', label='dim=320', zorder=5)

    if ppl_min_256 is not None:
        x_fit = np.linspace(50, 2000, 100)
        y_fit_256 = saturation_model(x_fit, ppl_min_256, A_256, a_256)
        ax1.plot(x_fit, y_fit_256, 'b-', alpha=0.7, linewidth=1.5)
        ax1.axhline(y=ppl_min_256, color='blue', linestyle=':', alpha=0.5)

    if ppl_min_320 is not None:
        y_fit_320 = saturation_model(x_fit, ppl_min_320, A_320, a_320)
        ax1.plot(x_fit, y_fit_320, 'r-', alpha=0.7, linewidth=1.5)
        ax1.axhline(y=ppl_min_320, color='red', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Sample Size', fontsize=11)
    ax1.set_ylabel('Validation PPL', fontsize=11)
    ax1.set_title('PPL vs Sample Size', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2000)

    # 右上: PPL比較 (対数)
    ax2 = axes[0, 1]
    ax2.scatter(samples, ppls_256, s=80, c='blue', marker='o', label='dim=256', zorder=5)
    ax2.scatter(samples, ppls_320, s=80, c='red', marker='s', label='dim=320', zorder=5)

    if ppl_min_256 is not None:
        x_log = np.logspace(1.8, 3.5, 100)
        y_log_256 = saturation_model(x_log, ppl_min_256, A_256, a_256)
        ax2.plot(x_log, y_log_256, 'b-', alpha=0.7, linewidth=1.5)

    if ppl_min_320 is not None:
        y_log_320 = saturation_model(x_log, ppl_min_320, A_320, a_320)
        ax2.plot(x_log, y_log_320, 'r-', alpha=0.7, linewidth=1.5)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size (log)', fontsize=11)
    ax2.set_ylabel('Validation PPL (log)', fontsize=11)
    ax2.set_title('PPL vs Sample Size (Log Scale)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # 左下: Effective Rank
    ax3 = axes[1, 0]
    x_pos = np.arange(len(samples))
    width = 0.35

    ax3.bar(x_pos - width/2, er_256, width, label='dim=256', color='blue', alpha=0.7)
    ax3.bar(x_pos + width/2, er_320, width, label='dim=320', color='red', alpha=0.7)

    ax3.set_xlabel('Sample Size', fontsize=11)
    ax3.set_ylabel('Effective Rank (%)', fontsize=11)
    ax3.set_title('Effective Rank Comparison', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(samples)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 100)

    # 右下: Accuracy
    ax4 = axes[1, 1]
    ax4.plot(samples, acc_256, 'bo-', markersize=8, label='dim=256', linewidth=2)
    ax4.plot(samples, acc_320, 'rs-', markersize=8, label='dim=320', linewidth=2)

    ax4.set_xlabel('Sample Size', fontsize=11)
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax4.set_title('Accuracy Comparison', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('importants/logs/context_dim_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Graph saved: importants/logs/context_dim_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
