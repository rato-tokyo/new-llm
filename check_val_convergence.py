"""
Validation Data Convergence Check

学習済みモデルで検証データを複数回順伝播させ、CVFP損失の推移を観察。
収束性（減少傾向）を自動判定する。

Usage:
    python3 check_val_convergence.py --num_trials 10
    python3 check_val_convergence.py --num_trials 20 --checkpoint_path ./checkpoints/my_model.pt
"""

import torch
import torch.nn.functional as F
import argparse
import numpy as np
from config import ResidualConfig
from src.models.llm import LLM
from src.data.loader import load_data
from src.trainers.phase1 import forward_all_tokens_sequential


def load_checkpoint(checkpoint_path, config, device):
    """
    学習済みモデルをロード

    Args:
        checkpoint_path: チェックポイントパス
        config: 設定オブジェクト
        device: torch device

    Returns:
        model: ロードされたモデル
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")

    # モデル作成（E案アーキテクチャ）
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        context_layers=config.context_layers,
        token_layers=config.token_layers,
        use_pretrained_embeddings=True
    )

    # チェックポイントロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  ✅ Model loaded successfully\n")
    return model


def check_convergence(model, val_token_ids, device, num_trials=10):
    """
    検証データの収束性をチェック

    Args:
        model: 学習済みモデル
        val_token_ids: 検証データトークンID
        device: torch device
        num_trials: 試行回数

    Returns:
        dict: 収束性評価結果
    """
    print(f"{'='*70}")
    print(f"Validation Convergence Check")
    print(f"{'='*70}\n")
    print(f"Validation tokens: {len(val_token_ids)}")
    print(f"Number of trials: {num_trials}\n")

    # トークン埋め込みを計算（1回のみ）
    with torch.no_grad():
        val_token_embeds = model.token_embedding(val_token_ids.unsqueeze(0).to(device))
        val_token_embeds = model.embed_norm(val_token_embeds).squeeze(0)

    print(f"{'='*70}")
    print(f"CVFP Loss Progression")
    print(f"{'='*70}\n")

    previous_contexts = None
    loss_history = []

    # 複数回順伝播
    for trial in range(num_trials):
        with torch.no_grad():
            # 順伝播
            contexts = forward_all_tokens_sequential(
                model,
                val_token_embeds,
                previous_contexts,
                device
            )

            # CVFP損失計算（2回目以降）
            if trial > 0:
                cvfp_loss = F.mse_loss(contexts, previous_contexts)
                loss_history.append(cvfp_loss.item())
                print(f"Trial {trial+1:2d}/{num_trials}: CVFP Loss = {cvfp_loss.item():.6f}")
            else:
                print(f"Trial {trial+1:2d}/{num_trials}: CVFP Loss = N/A (baseline, no previous context)")

            # 次回用に保存
            previous_contexts = contexts.detach().clone()

    # 収束性分析
    print(f"\n{'='*70}")
    print(f"Convergence Analysis")
    print(f"{'='*70}\n")

    analysis = analyze_convergence_trend(loss_history)

    return analysis


def analyze_convergence_trend(losses):
    """
    損失の推移から収束性を判断

    Args:
        losses: 損失履歴リスト

    Returns:
        dict: 分析結果
    """
    if len(losses) < 2:
        return {
            'status': 'INSUFFICIENT_DATA',
            'message': 'Need at least 2 trials to analyze trend'
        }

    losses = np.array(losses)
    n = len(losses)

    # 統計量計算
    initial_loss = losses[0]
    final_loss = losses[-1]
    mean_loss = losses.mean()
    std_loss = losses.std()

    # 減少率
    reduction = (initial_loss - final_loss) / initial_loss * 100

    # 線形回帰で傾き計算
    x = np.arange(n)
    slope = np.polyfit(x, losses, 1)[0]

    # 変動係数（安定性指標）
    cv = (std_loss / mean_loss) * 100 if mean_loss > 0 else 0

    # 判定基準
    if slope < -0.001:  # 明確な減少傾向
        status = "CONVERGING"
        symbol = "✅"
        message = "Loss is decreasing - model is converging on validation data"
    elif abs(slope) < 0.001 and std_loss < 0.01:  # ほぼ横ばい
        status = "CONVERGED"
        symbol = "✅"
        message = "Loss is stable - model has converged on validation data"
    elif slope > 0.001:  # 増加傾向
        status = "DIVERGING"
        symbol = "❌"
        message = "Loss is increasing - model is NOT converging on validation data"
    else:  # 不安定
        status = "UNSTABLE"
        symbol = "⚠️"
        message = "Loss is fluctuating - convergence is unclear"

    # 結果表示
    print(f"Statistics:")
    print(f"  - Initial Loss (Trial 2): {initial_loss:.6f}")
    print(f"  - Final Loss (Trial {len(losses)+1}): {final_loss:.6f}")
    print(f"  - Mean Loss: {mean_loss:.6f}")
    print(f"  - Std Dev: {std_loss:.6f}")
    print(f"  - Coefficient of Variation: {cv:.2f}%")
    print(f"\nTrend Analysis:")
    print(f"  - Reduction: {reduction:+.2f}%")
    print(f"  - Slope (linear fit): {slope:.6f}")
    print(f"\nVerdict:")
    print(f"  {symbol} {status}: {message}\n")

    return {
        'status': status,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'reduction_percent': reduction,
        'slope': slope,
        'mean': mean_loss,
        'std': std_loss,
        'cv': cv,
        'message': message,
        'is_converging': status in ['CONVERGING', 'CONVERGED']
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check validation data convergence with trained model"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of forward pass trials (default: 10)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/model_latest.pt",
        help="Path to model checkpoint (default: ./checkpoints/model_latest.pt)"
    )

    args = parser.parse_args()

    # 設定ロード
    config = ResidualConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Validation Convergence Checker")
    print(f"{'='*70}\n")
    print(f"Configuration:")
    print(f"  - Checkpoint: {args.checkpoint_path}")
    print(f"  - Number of trials: {args.num_trials}")
    print(f"  - Device: {device}")
    print(f"  - Validation data: {config.val_text_file}")

    # データロード
    print(f"\nLoading validation data...")
    _, val_token_ids = load_data(config)
    print(f"  ✅ Loaded {len(val_token_ids)} validation tokens")

    # モデルロード
    model = load_checkpoint(args.checkpoint_path, config, device)

    # 収束性チェック
    analysis = check_convergence(model, val_token_ids, device, args.num_trials)

    # 最終サマリー
    print(f"{'='*70}")
    print(f"Final Summary")
    print(f"{'='*70}\n")

    if analysis['is_converging']:
        print(f"✅ VALIDATION DATA CONVERGES")
        print(f"   The trained model shows convergence on validation data.")
        print(f"   This indicates stable fixed-point behavior.")
    else:
        print(f"❌ VALIDATION DATA DOES NOT CONVERGE")
        print(f"   The model may need more training or different hyperparameters.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
