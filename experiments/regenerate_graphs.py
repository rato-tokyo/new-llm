"""既存のチェックポイントから正しいグラフを再生成"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt


def load_and_plot(checkpoint_path, model_name, output_name):
    """チェックポイントから訓練履歴を読み込んでグラフを生成"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ {checkpoint_path} が見つかりません")
        return

    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])

    if not train_losses or not val_losses:
        print(f"⚠️ {checkpoint_path} に訓練履歴がありません")
        return

    # Perplexityを計算
    import math
    train_ppls = [math.exp(loss) if loss < 20 else 1e9 for loss in train_losses]
    val_ppls = [math.exp(loss) if loss < 20 else 1e9 for loss in val_losses]

    # グラフ作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    # 左側: Loss curves
    axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', markersize=3, color='#1f77b4')
    axes[0].plot(epochs, val_losses, label='Val Loss', linewidth=2, marker='s', markersize=3, color='#ff7f0e')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右側: Perplexity curves (線形スケール)
    axes[1].plot(epochs, train_ppls, label='Train PPL', linewidth=2, marker='o', markersize=3, color='#1f77b4')
    axes[1].plot(epochs, val_ppls, label='Val PPL', linewidth=2, marker='s', markersize=3, color='#ff7f0e')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title(f'{model_name} - Perplexity (Linear Scale)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # スパイクが見やすいように y軸の範囲を調整
    if len(val_ppls) > 0:
        max_ppl = max(val_ppls)
        if max_ppl > 1000:  # スパイクがある場合
            # 上位5%の外れ値を除外して範囲を設定
            sorted_ppls = sorted(val_ppls)
            percentile_95 = sorted_ppls[int(len(sorted_ppls) * 0.95)]
            axes[1].set_ylim(0, min(percentile_95 * 1.5, max_ppl))

            # スパイクの注釈を追加
            spike_epochs = [i+1 for i, ppl in enumerate(val_ppls) if ppl > percentile_95]
            if spike_epochs:
                axes[1].axhline(y=percentile_95, color='red', linestyle='--', alpha=0.5, label='95th percentile')
                for ep in spike_epochs[:3]:  # 最初の3つのスパイクに注釈
                    axes[1].annotate(f'Spike\nEpoch {ep}',
                                   xy=(ep, val_ppls[ep-1]),
                                   xytext=(ep+5, percentile_95 * 0.8),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                   fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()

    # 保存
    save_path = f"checkpoints/{output_name}"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ {save_path} を生成しました")

    # 統計情報を表示
    print(f"\n{model_name} 統計:")
    print(f"  エポック数: {len(train_losses)}")
    print(f"  Best Val Loss: {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses))+1})")
    print(f"  Final Val Loss: {val_losses[-1]:.4f}")
    print(f"  Max Val PPL: {max(val_ppls):.1f}")
    if max(val_ppls) > 1000:
        spike_count = sum(1 for ppl in val_ppls if ppl > 1000)
        print(f"  ⚠️ スパイク発生回数: {spike_count}回")
    print()

    plt.close()


def main():
    print("=" * 60)
    print("グラフ再生成スクリプト")
    print("=" * 60)
    print()

    # チェックポイントファイルを探す
    checkpoints_dir = "checkpoints"

    # Transformer baseline
    transformer_path = os.path.join(checkpoints_dir, "best_transformer_baseline.pt")
    load_and_plot(transformer_path, "Transformer Baseline", "transformer_baseline_curves_fixed.png")

    # New-LLM (最新の実験)
    newllm_path = os.path.join(checkpoints_dir, "best_new_llm.pt")
    load_and_plot(newllm_path, "New-LLM", "new_llm_curves_fixed.png")

    print("=" * 60)
    print("完了！")
    print("=" * 60)
    print()
    print("生成されたファイル:")
    print("  - checkpoints/transformer_baseline_curves_fixed.png")
    print("  - checkpoints/new_llm_curves_fixed.png")
    print()
    print("これらのグラフでは:")
    print("  ✓ 左側: Loss (損失値)")
    print("  ✓ 右側: Perplexity (線形スケール)")
    print("  ✓ スパイクが赤い注釈で明示")
    print()


if __name__ == "__main__":
    main()
