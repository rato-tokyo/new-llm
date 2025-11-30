#!/usr/bin/env python3
"""
9つのモデル設定のスケーリング則をプロット

Usage:
  python3 scripts/plot_scaling_laws.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# データ（ログから抽出）
# 各設定について [50samples, 100samples, 200samples, 500samples] のトークン数とPPL
DATA = {
    "1L_768d_1tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [683.9, 415.8, 294.5, 198.2],
        "alpha": -0.546,
        "A": 2.67e5,
    },
    "2L_768d_1tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [647.2, 445.5, 331.5, 222.2],
        "alpha": -0.473,
        "A": 1.18e5,
    },
    "3L_768d_1tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [722.5, 480.4, 348.6, 236.7],
        "alpha": -0.495,
        "A": 1.65e5,
    },
    "1L_768d_2tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [591.3, 400.3, 295.1, 189.4],
        "alpha": -0.503,
        "A": 1.51e5,
    },
    "2L_768d_2tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [622.3, 391.9, 284.9, 197.0],
        "alpha": -0.507,
        "A": 1.59e5,
    },
    "3L_768d_2tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [651.1, 400.8, 293.8, 194.6],
        "alpha": -0.530,
        "A": 2.14e5,
    },
    "1L_768d_3tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [611.3, 414.1, 301.6, 195.9],
        "alpha": -0.505,
        "A": 1.58e5,
    },
    "2L_768d_3tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [608.7, 402.1, 285.7, 198.1],
        "alpha": -0.499,
        "A": 1.45e5,
    },
    "3L_768d_3tok": {
        "tokens": [62891, 122795, 240132, 587970],
        "ppl": [650.7, 412.0, 297.0, 193.7],
        "alpha": -0.535,
        "A": 2.28e5,
    },
}

# カラーマップ（層数とトークン数で分類）
COLORS = {
    # 1token: 青系
    "1L_768d_1tok": "#1f77b4",  # 濃い青
    "2L_768d_1tok": "#7fcdff",  # 薄い青
    "3L_768d_1tok": "#aec7e8",  # もっと薄い青
    # 2token: 緑系
    "1L_768d_2tok": "#2ca02c",  # 濃い緑
    "2L_768d_2tok": "#98df8a",  # 薄い緑
    "3L_768d_2tok": "#c5e1a5",  # もっと薄い緑
    # 3token: 赤系
    "1L_768d_3tok": "#d62728",  # 濃い赤
    "2L_768d_3tok": "#ff9896",  # 薄い赤
    "3L_768d_3tok": "#ffcccb",  # もっと薄い赤
}

# マーカー（層数で分類）
MARKERS = {
    1: "o",  # 1層: 丸
    2: "s",  # 2層: 四角
    3: "^",  # 3層: 三角
}


def plot_scaling_laws():
    """スケーリング則をプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === 左図: 全設定のスケーリング則 ===
    ax1 = axes[0]

    for config_name, data in DATA.items():
        tokens = np.array(data["tokens"])
        ppl = np.array(data["ppl"])
        alpha = data["alpha"]
        A = data["A"]

        # 層数を抽出
        parts = config_name.split("_")
        num_layers = int(parts[0][0])

        # データポイントをプロット
        ax1.scatter(tokens, ppl,
                   color=COLORS[config_name],
                   marker=MARKERS[num_layers],
                   s=80,
                   label=f"{config_name} (α={alpha:.3f})",
                   zorder=3)

        # フィットライン
        x_fit = np.logspace(np.log10(tokens.min()), np.log10(tokens.max() * 1.5), 100)
        y_fit = A * (x_fit ** alpha)
        ax1.plot(x_fit, y_fit, color=COLORS[config_name], alpha=0.5, linewidth=1.5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Tokens", fontsize=12)
    ax1.set_ylabel("Validation PPL", fontsize=12)
    ax1.set_title("Scaling Laws: PPL = A × tokens^α", fontsize=14)
    ax1.legend(fontsize=8, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(5e4, 1e6)
    ax1.set_ylim(150, 800)

    # === 右図: α値の比較（ヒートマップ風） ===
    ax2 = axes[1]

    # データ整理
    alpha_matrix = np.zeros((3, 3))

    for config_name, data in DATA.items():
        parts = config_name.split("_")
        layer_idx = int(parts[0][0]) - 1  # 0-indexed
        token_idx = int(parts[2][0]) - 1  # 0-indexed
        alpha_matrix[layer_idx, token_idx] = data["alpha"]

    # ヒートマップ
    im = ax2.imshow(alpha_matrix, cmap="RdYlGn_r", aspect="auto",
                    vmin=-0.55, vmax=-0.47)

    # ラベル
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["1 token", "2 tokens", "3 tokens"])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["1 layer", "2 layers", "3 layers"])
    ax2.set_xlabel("Input Tokens", fontsize=12)
    ax2.set_ylabel("Layers", fontsize=12)
    ax2.set_title("α Value (Data Efficiency)\nMore negative = better", fontsize=14)

    # 値を表示
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{alpha_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=12,
                    color="white" if alpha_matrix[i, j] < -0.52 else "black")

    # カラーバー
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("α value", fontsize=10)

    plt.tight_layout()

    # 保存
    output_path = Path("importants/scaling_law_9configs.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # 追加グラフ: 500サンプル時のPPLとAccuracy比較
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # PPL比較（500サンプル）
    ax3 = axes2[0]
    configs = list(DATA.keys())
    ppls = [DATA[c]["ppl"][-1] for c in configs]  # 500サンプル時
    colors = [COLORS[c] for c in configs]

    bars = ax3.bar(range(len(configs)), ppls, color=colors)
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=45, ha="right")
    ax3.set_ylabel("Validation PPL (500 samples)", fontsize=12)
    ax3.set_title("PPL Comparison at 500 samples\nLower is better", fontsize=14)
    ax3.axhline(y=189.4, color="green", linestyle="--", alpha=0.7, label="Best: 189.4")
    ax3.legend()

    # 値を表示
    for bar, ppl in zip(bars, ppls):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{ppl:.1f}", ha="center", va="bottom", fontsize=9)

    # α値比較
    ax4 = axes2[1]
    alphas = [DATA[c]["alpha"] for c in configs]

    bars = ax4.bar(range(len(configs)), alphas, color=colors)
    ax4.set_xticks(range(len(configs)))
    ax4.set_xticklabels(configs, rotation=45, ha="right")
    ax4.set_ylabel("α (Scaling Exponent)", fontsize=12)
    ax4.set_title("α Value Comparison\nMore negative = better data efficiency", fontsize=14)
    ax4.axhline(y=-0.5, color="gray", linestyle="--", alpha=0.7, label="α = -0.5")
    ax4.axhline(y=-0.546, color="blue", linestyle="--", alpha=0.7, label="Best: -0.546")
    ax4.legend()

    # 値を表示
    for bar, alpha in zip(bars, alphas):
        y_pos = bar.get_height() - 0.005 if alpha < -0.5 else bar.get_height() + 0.005
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{alpha:.3f}", ha="center", va="top" if alpha < -0.5 else "bottom", fontsize=9)

    plt.tight_layout()

    output_path2 = Path("importants/scaling_comparison_9configs.png")
    plt.savefig(output_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path2}")

    plt.show()


if __name__ == "__main__":
    plot_scaling_laws()
