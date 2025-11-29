"""
Validation Data Convergence Check

学習済みモデルで検証データを複数回順伝播させ、CVFP損失の推移を観察。
収束性（減少傾向）を自動判定する。

Usage:
    python3 scripts/check_val_convergence.py --num_trials 10
    python3 scripts/check_val_convergence.py --num_trials 20 --checkpoint_path ./checkpoints/my_model.pt
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import argparse
import numpy as np
from config import ResidualConfig
from src.models.llm import LLM
from src.providers import create_data_provider


def forward_sequential(model, token_embeds, previous_contexts, device, num_input_tokens=1):
    """順次処理で全トークンのコンテキストを計算"""
    if previous_contexts is None:
        context = torch.zeros(1, model.context_dim, device=device)
    else:
        context = previous_contexts[-1].unsqueeze(0).detach()

    # トークン履歴を初期化（ゼロベクトルで埋める）
    token_history = [torch.zeros(model.embed_dim, device=device)
                     for _ in range(num_input_tokens - 1)]

    context_list = []
    for token_embed in token_embeds:
        # 履歴 + 現在のトークンを結合
        token_history.append(token_embed)
        combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)

        context = model.context_block(context, combined_tokens.unsqueeze(0))
        context_list.append(context.squeeze(0))

    return torch.stack(context_list)


def load_checkpoint(checkpoint_path, config, device):
    """学習済みモデルをロード"""
    print(f"\nLoading checkpoint: {checkpoint_path}")

    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        num_input_tokens=getattr(config, 'num_input_tokens', 1),
        use_pretrained_embeddings=True
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("  ✅ Model loaded successfully\n")
    return model


def check_convergence(model, val_token_ids, device, num_trials=10, num_input_tokens=1):
    """検証データの収束性をチェック"""
    print(f"{'='*70}")
    print("Validation Convergence Check")
    print(f"{'='*70}\n")
    print(f"Validation tokens: {len(val_token_ids)}")
    print(f"Number of trials: {num_trials}")
    print(f"num_input_tokens: {num_input_tokens}\n")

    with torch.no_grad():
        val_token_embeds = model.token_embedding(val_token_ids.unsqueeze(0).to(device))
        val_token_embeds = model.embed_norm(val_token_embeds).squeeze(0)

    print(f"{'='*70}")
    print("CVFP Loss Progression")
    print(f"{'='*70}\n")

    previous_contexts = None
    loss_history = []

    for trial in range(num_trials):
        with torch.no_grad():
            contexts = forward_sequential(model, val_token_embeds, previous_contexts, device, num_input_tokens)

            if trial > 0:
                cvfp_loss = F.mse_loss(contexts, previous_contexts)
                loss_history.append(cvfp_loss.item())
                print(f"Trial {trial+1:2d}/{num_trials}: CVFP Loss = {cvfp_loss.item():.6f}")
            else:
                print(f"Trial {trial+1:2d}/{num_trials}: CVFP Loss = N/A (baseline)")

            previous_contexts = contexts.detach().clone()

    print(f"\n{'='*70}")
    print("Convergence Analysis")
    print(f"{'='*70}\n")

    return analyze_convergence_trend(loss_history)


def analyze_convergence_trend(losses):
    """損失の推移から収束性を判断"""
    if len(losses) < 2:
        return {'status': 'INSUFFICIENT_DATA', 'message': 'Need at least 2 trials'}

    losses = np.array(losses)
    n = len(losses)

    initial_loss = losses[0]
    final_loss = losses[-1]
    mean_loss = losses.mean()
    std_loss = losses.std()

    reduction = (initial_loss - final_loss) / initial_loss * 100
    slope = np.polyfit(np.arange(n), losses, 1)[0]
    cv = (std_loss / mean_loss) * 100 if mean_loss > 0 else 0

    if slope < -0.001:
        status, symbol = "CONVERGING", "✅"
        message = "Loss is decreasing - model is converging"
    elif abs(slope) < 0.001 and std_loss < 0.01:
        status, symbol = "CONVERGED", "✅"
        message = "Loss is stable - model has converged"
    elif slope > 0.001:
        status, symbol = "DIVERGING", "❌"
        message = "Loss is increasing - model is NOT converging"
    else:
        status, symbol = "UNSTABLE", "⚠️"
        message = "Loss is fluctuating - convergence unclear"

    print("Statistics:")
    print(f"  - Initial Loss: {initial_loss:.6f}")
    print(f"  - Final Loss: {final_loss:.6f}")
    print(f"  - Reduction: {reduction:+.2f}%")
    print(f"  - Slope: {slope:.6f}")
    print(f"\nVerdict: {symbol} {status}: {message}\n")

    return {
        'status': status,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'reduction_percent': reduction,
        'slope': slope,
        'is_converging': status in ['CONVERGING', 'CONVERGED']
    }


def main():
    parser = argparse.ArgumentParser(description="Check validation convergence")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/model_latest.pt")
    args = parser.parse_args()

    config = ResidualConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("Validation Convergence Checker")
    print(f"{'='*70}\n")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Trials: {args.num_trials}")
    print(f"  Device: {device}")

    # データロード
    data_provider = create_data_provider("memory", config)
    _, val_token_ids = data_provider.load_data()
    val_token_ids = val_token_ids.to(device)
    print(f"  Validation tokens: {len(val_token_ids)}")

    # モデルロード・チェック
    num_input_tokens = getattr(config, 'num_input_tokens', 1)
    model = load_checkpoint(args.checkpoint_path, config, device)
    analysis = check_convergence(model, val_token_ids, device, args.num_trials, num_input_tokens)

    print(f"{'='*70}")
    print(f"Final: {'✅ CONVERGES' if analysis['is_converging'] else '❌ DOES NOT CONVERGE'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
