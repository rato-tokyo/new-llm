"""
Sample Size Scaling Model Comparison

複数のモデルを比較して、どの関数がベストフィットかを分析する。

モデル候補:
1. 単純べき乗則: PPL = A × n^(-a)
2. 飽和モデル: PPL = PPL_min + A × n^(-a)
3. 対数補正べき乗則: PPL = A × n^(-a) × log(n)^b
4. 指数減衰: PPL = A × exp(-b × n^c)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


def main():
    # 実験データ
    samples = np.array([100, 200, 400, 800, 1600])
    ppls = np.array([321.4, 253.1, 186.0, 155.8, 134.7])

    print("=" * 70)
    print("SCALING MODEL COMPARISON")
    print("=" * 70)
    print("\n実験データ:")
    for s, p in zip(samples, ppls):
        print(f"  {s:>5} samples → PPL = {p:.1f}")

    # ========================================
    # Model 1: 単純べき乗則 PPL = A × n^(-a)
    # ========================================
    print("\n" + "=" * 70)
    print("Model 1: 単純べき乗則 PPL = A × n^(-a)")
    print("=" * 70)

    log_samples = np.log(samples)
    log_ppls = np.log(ppls)
    slope, intercept, r, _, _ = stats.linregress(log_samples, log_ppls)
    a1 = -slope
    A1 = np.exp(intercept)

    pred1 = A1 * (samples ** (-a1))
    residuals1 = ppls - pred1
    ss_res1 = np.sum(residuals1 ** 2)
    ss_tot = np.sum((ppls - np.mean(ppls)) ** 2)
    r2_1 = 1 - ss_res1 / ss_tot

    print(f"  a = {a1:.4f}")
    print(f"  A = {A1:.2f}")
    print(f"  R² = {r2_1:.6f}")
    print(f"  残差二乗和 = {ss_res1:.2f}")

    print("\n  予測 vs 実測:")
    for s, actual, pred in zip(samples, ppls, pred1):
        err = (pred - actual) / actual * 100
        print(f"    {s:>5}: {actual:.1f} → {pred:.1f} ({err:+.1f}%)")

    # ========================================
    # Model 2: 飽和モデル PPL = PPL_min + A × n^(-a)
    # ========================================
    print("\n" + "=" * 70)
    print("Model 2: 飽和モデル PPL = PPL_min + A × n^(-a)")
    print("=" * 70)

    def saturation_model(n, ppl_min, A, a):
        return ppl_min + A * (n ** (-a))

    # 初期値を設定（PPL_min ≈ 50-100, A ≈ 1000, a ≈ 0.3）
    try:
        popt2, pcov2 = curve_fit(
            saturation_model, samples, ppls,
            p0=[80, 1000, 0.3],
            bounds=([0, 0, 0], [150, 10000, 2]),
            maxfev=10000
        )
        ppl_min2, A2, a2 = popt2

        pred2 = saturation_model(samples, *popt2)
        residuals2 = ppls - pred2
        ss_res2 = np.sum(residuals2 ** 2)
        r2_2 = 1 - ss_res2 / ss_tot

        print(f"  PPL_min = {ppl_min2:.2f}")
        print(f"  A = {A2:.2f}")
        print(f"  a = {a2:.4f}")
        print(f"  R² = {r2_2:.6f}")
        print(f"  残差二乗和 = {ss_res2:.2f}")

        print("\n  予測 vs 実測:")
        for s, actual, pred in zip(samples, ppls, pred2):
            err = (pred - actual) / actual * 100
            print(f"    {s:>5}: {actual:.1f} → {pred:.1f} ({err:+.1f}%)")

        # 外挿予測
        print("\n  外挿予測:")
        for s in [3200, 6400, 10000, 50000]:
            pred = saturation_model(s, *popt2)
            print(f"    {s:>6} samples → PPL ≈ {pred:.1f}")
    except Exception as e:
        print(f"  フィッティング失敗: {e}")
        ppl_min2, A2, a2 = None, None, None
        r2_2 = 0
        ss_res2 = float('inf')

    # ========================================
    # Model 3: 対数補正べき乗則 PPL = A × n^(-a) × log(n)^b
    # ========================================
    print("\n" + "=" * 70)
    print("Model 3: 対数補正べき乗則 PPL = A × n^(-a) × log(n)^b")
    print("=" * 70)

    def log_corrected_power(n, A, a, b):
        return A * (n ** (-a)) * (np.log(n) ** b)

    try:
        popt3, pcov3 = curve_fit(
            log_corrected_power, samples, ppls,
            p0=[1000, 0.3, 0.5],
            bounds=([0, 0, -5], [10000, 2, 5]),
            maxfev=10000
        )
        A3, a3, b3 = popt3

        pred3 = log_corrected_power(samples, *popt3)
        residuals3 = ppls - pred3
        ss_res3 = np.sum(residuals3 ** 2)
        r2_3 = 1 - ss_res3 / ss_tot

        print(f"  A = {A3:.2f}")
        print(f"  a = {a3:.4f}")
        print(f"  b = {b3:.4f}")
        print(f"  R² = {r2_3:.6f}")
        print(f"  残差二乗和 = {ss_res3:.2f}")

        print("\n  予測 vs 実測:")
        for s, actual, pred in zip(samples, ppls, pred3):
            err = (pred - actual) / actual * 100
            print(f"    {s:>5}: {actual:.1f} → {pred:.1f} ({err:+.1f}%)")
    except Exception as e:
        print(f"  フィッティング失敗: {e}")
        A3, a3, b3 = None, None, None
        r2_3 = 0
        ss_res3 = float('inf')

    # ========================================
    # Model 4: 指数減衰 PPL = A × exp(-b × n^c) + PPL_min
    # ========================================
    print("\n" + "=" * 70)
    print("Model 4: 指数減衰 PPL = PPL_min + A × exp(-b × n^c)")
    print("=" * 70)

    def exp_decay(n, ppl_min, A, b, c):
        return ppl_min + A * np.exp(-b * (n ** c))

    try:
        popt4, pcov4 = curve_fit(
            exp_decay, samples, ppls,
            p0=[100, 300, 0.01, 0.5],
            bounds=([0, 0, 0, 0], [200, 1000, 1, 1]),
            maxfev=10000
        )
        ppl_min4, A4, b4, c4 = popt4

        pred4 = exp_decay(samples, *popt4)
        residuals4 = ppls - pred4
        ss_res4 = np.sum(residuals4 ** 2)
        r2_4 = 1 - ss_res4 / ss_tot

        print(f"  PPL_min = {ppl_min4:.2f}")
        print(f"  A = {A4:.2f}")
        print(f"  b = {b4:.6f}")
        print(f"  c = {c4:.4f}")
        print(f"  R² = {r2_4:.6f}")
        print(f"  残差二乗和 = {ss_res4:.2f}")

        print("\n  予測 vs 実測:")
        for s, actual, pred in zip(samples, ppls, pred4):
            err = (pred - actual) / actual * 100
            print(f"    {s:>5}: {actual:.1f} → {pred:.1f} ({err:+.1f}%)")
    except Exception as e:
        print(f"  フィッティング失敗: {e}")
        ppl_min4, A4, b4, c4 = None, None, None, None
        r2_4 = 0
        ss_res4 = float('inf')

    # ========================================
    # Model 5: 変動指数モデル a(n) = a0 - a1*log(n)
    # ========================================
    print("\n" + "=" * 70)
    print("Model 5: 変動指数 PPL = A × n^(-(a0 - a1×log(n)))")
    print("=" * 70)

    def varying_exponent(n, A, a0, a1):
        a_n = a0 - a1 * np.log(n)
        return A * (n ** (-a_n))

    try:
        popt5, pcov5 = curve_fit(
            varying_exponent, samples, ppls,
            p0=[1000, 0.5, 0.02],
            bounds=([0, 0, -0.5], [10000, 2, 0.5]),
            maxfev=10000
        )
        A5, a0_5, a1_5 = popt5

        pred5 = varying_exponent(samples, *popt5)
        residuals5 = ppls - pred5
        ss_res5 = np.sum(residuals5 ** 2)
        r2_5 = 1 - ss_res5 / ss_tot

        print(f"  A = {A5:.2f}")
        print(f"  a0 = {a0_5:.4f}")
        print(f"  a1 = {a1_5:.6f}")
        print(f"  R² = {r2_5:.6f}")
        print(f"  残差二乗和 = {ss_res5:.2f}")

        print("\n  各サンプル数での実効指数 a(n):")
        for s in samples:
            a_eff = a0_5 - a1_5 * np.log(s)
            print(f"    n={s:>5}: a(n) = {a_eff:.4f}")

        print("\n  予測 vs 実測:")
        for s, actual, pred in zip(samples, ppls, pred5):
            err = (pred - actual) / actual * 100
            print(f"    {s:>5}: {actual:.1f} → {pred:.1f} ({err:+.1f}%)")
    except Exception as e:
        print(f"  フィッティング失敗: {e}")
        A5, a0_5, a1_5 = None, None, None
        r2_5 = 0
        ss_res5 = float('inf')

    # ========================================
    # 比較サマリー
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    models = [
        ("1. 単純べき乗則", r2_1, ss_res1, 2),
        ("2. 飽和モデル", r2_2, ss_res2, 3),
        ("3. 対数補正べき乗則", r2_3, ss_res3, 3),
        ("4. 指数減衰", r2_4, ss_res4, 4),
        ("5. 変動指数", r2_5, ss_res5, 3),
    ]

    # AIC/BIC計算（簡易版）
    n_data = len(samples)
    print(f"\n{'Model':<25} | {'R²':>10} | {'RSS':>10} | {'Params':>6} | {'AIC':>10}")
    print("-" * 70)

    best_aic = float('inf')
    best_model = None

    for name, r2, rss, n_params in models:
        if rss > 0 and rss < float('inf'):
            # AIC = n * log(RSS/n) + 2k
            aic = n_data * np.log(rss / n_data) + 2 * n_params
            if aic < best_aic:
                best_aic = aic
                best_model = name
        else:
            aic = float('inf')
        print(f"{name:<25} | {r2:>10.6f} | {rss:>10.2f} | {n_params:>6} | {aic:>10.2f}")

    print("-" * 70)
    print(f"\n★ Best model (by AIC): {best_model}")

    # ========================================
    # グラフ作成
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_fit = np.linspace(50, 3000, 200)

    # 左: 線形スケール
    ax1 = axes[0]
    ax1.scatter(samples, ppls, s=100, c='black', zorder=10, label='Experimental')

    # Model 1
    y1 = A1 * (x_fit ** (-a1))
    ax1.plot(x_fit, y1, 'b-', linewidth=1.5, alpha=0.7, label=f'M1: Power (R²={r2_1:.4f})')

    # Model 2
    if ppl_min2 is not None:
        y2 = saturation_model(x_fit, ppl_min2, A2, a2)
        ax1.plot(x_fit, y2, 'r-', linewidth=1.5, alpha=0.7, label=f'M2: Saturation (R²={r2_2:.4f})')
        ax1.axhline(y=ppl_min2, color='r', linestyle=':', alpha=0.5, label=f'PPL_min={ppl_min2:.1f}')

    # Model 5
    if A5 is not None:
        y5 = varying_exponent(x_fit, A5, a0_5, a1_5)
        ax1.plot(x_fit, y5, 'g-', linewidth=1.5, alpha=0.7, label=f'M5: Varying exp (R²={r2_5:.4f})')

    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('Validation PPL', fontsize=12)
    ax1.set_title('Model Comparison (Linear Scale)', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3000)
    ax1.set_ylim(0, 400)

    # 右: 対数スケール
    ax2 = axes[1]
    ax2.scatter(samples, ppls, s=100, c='black', zorder=10, label='Experimental')

    x_log = np.logspace(1.8, 3.5, 200)

    # Model 1
    y1_log = A1 * (x_log ** (-a1))
    ax2.plot(x_log, y1_log, 'b-', linewidth=1.5, alpha=0.7, label='M1: Power')

    # Model 2
    if ppl_min2 is not None:
        y2_log = saturation_model(x_log, ppl_min2, A2, a2)
        ax2.plot(x_log, y2_log, 'r-', linewidth=1.5, alpha=0.7, label='M2: Saturation')

    # Model 5
    if A5 is not None:
        y5_log = varying_exponent(x_log, A5, a0_5, a1_5)
        ax2.plot(x_log, y5_log, 'g-', linewidth=1.5, alpha=0.7, label='M5: Varying exp')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size (log)', fontsize=12)
    ax2.set_ylabel('Validation PPL (log)', fontsize=12)
    ax2.set_title('Model Comparison (Log-Log Scale)', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('importants/logs/scaling_model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Graph saved: importants/logs/scaling_model_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
