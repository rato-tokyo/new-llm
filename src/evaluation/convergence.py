"""
Convergence Evaluation - CVFP収束判定

検証データを複数回イテレーションして、CVFP損失が収束するかを判定。
Phase1トレーナー（Memory/Storage）から共通利用。
"""

import sys
from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn.functional as F
import numpy as np


def print_flush(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


@dataclass
class ConvergenceResult:
    """収束判定の結果"""
    status: str  # CONVERGING, CONVERGED, DIVERGING, UNSTABLE
    is_converging: bool
    initial_loss: float
    final_loss: float
    reduction_percent: float
    slope: float
    loss_history: List[float]
    contexts: torch.Tensor  # 最終イテレーションのコンテキスト


def forward_sequential(model, token_embeds: torch.Tensor, previous_contexts: Optional[torch.Tensor], device: torch.device, num_input_tokens: int = 1) -> torch.Tensor:
    """
    順次処理で全トークンのコンテキストを計算

    Args:
        model: LLMモデル
        token_embeds: トークン埋め込み [num_tokens, embed_dim]
        previous_contexts: 前回のコンテキスト（Noneの場合はゼロから開始）
        device: デバイス
        num_input_tokens: 入力トークン数

    Returns:
        contexts: 新しいコンテキスト [num_tokens, context_dim]
    """
    if previous_contexts is None:
        context = torch.zeros(1, model.context_dim, device=device)
    else:
        context = previous_contexts[-1].unsqueeze(0).detach()

    # トークン履歴を初期化（ゼロベクトルで埋める）
    # イテレーション開始時はトークン履歴をリセット
    token_history = [torch.zeros(model.embed_dim, device=device)
                     for _ in range(num_input_tokens - 1)]

    context_list = []
    for token_embed in token_embeds:
        # 履歴 + 現在のトークンを結合
        token_history.append(token_embed)
        combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)
        # combined_tokens: [embed_dim * num_input_tokens]

        context = model.context_block(context, combined_tokens.unsqueeze(0))
        context_list.append(context.squeeze(0))

    return torch.stack(context_list)


def check_convergence(
    model,
    token_ids: torch.Tensor,
    device: torch.device,
    num_trials: int = 10,
    verbose: bool = True,
    num_input_tokens: int = 1
) -> ConvergenceResult:
    """
    検証データの収束性をチェック

    複数回イテレーションして、CVFP損失（前回との差分MSE）が
    減少傾向にあるかを判定。

    Args:
        model: LLMモデル
        token_ids: トークンID
        device: デバイス
        num_trials: イテレーション回数
        verbose: 詳細出力
        num_input_tokens: 入力トークン数

    Returns:
        ConvergenceResult: 収束判定結果
    """
    model.eval()

    if verbose:
        print_flush(f"\n{'='*70}")
        print_flush("CVFP Convergence Check")
        print_flush(f"{'='*70}")
        print_flush(f"  Tokens: {len(token_ids):,}")
        print_flush(f"  Trials: {num_trials}")
        print_flush(f"  num_input_tokens: {num_input_tokens}")

    # トークン埋め込みを計算
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    previous_contexts = None
    loss_history = []

    if verbose:
        print_flush(f"\n{'='*70}")
        print_flush("CVFP Loss Progression")
        print_flush(f"{'='*70}\n")

    for trial in range(num_trials):
        with torch.no_grad():
            contexts = forward_sequential(model, token_embeds, previous_contexts, device, num_input_tokens)

            if trial > 0:
                cvfp_loss = F.mse_loss(contexts, previous_contexts)
                loss_history.append(cvfp_loss.item())
                if verbose:
                    print_flush(f"Trial {trial+1:2d}/{num_trials}: CVFP Loss = {cvfp_loss.item():.6f}")
            else:
                if verbose:
                    print_flush(f"Trial {trial+1:2d}/{num_trials}: CVFP Loss = N/A (baseline)")

            previous_contexts = contexts.detach().clone()

    # 収束分析
    result = _analyze_convergence_trend(loss_history, contexts, verbose)

    model.train()
    return result


def _analyze_convergence_trend(losses: List[float], contexts: torch.Tensor, verbose: bool) -> ConvergenceResult:
    """損失の推移から収束性を判断"""
    if len(losses) < 2:
        return ConvergenceResult(
            status='INSUFFICIENT_DATA',
            is_converging=False,
            initial_loss=0.0,
            final_loss=0.0,
            reduction_percent=0.0,
            slope=0.0,
            loss_history=losses,
            contexts=contexts
        )

    losses_np = np.array(losses)
    n = len(losses_np)

    initial_loss = losses_np[0]
    final_loss = losses_np[-1]
    mean_loss = losses_np.mean()
    std_loss = losses_np.std()

    reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
    slope = np.polyfit(np.arange(n), losses_np, 1)[0]

    # 判定
    if slope < -0.001:
        status, symbol = "CONVERGING", "✅"
        message = "Loss is decreasing - model is converging"
        is_converging = True
    elif abs(slope) < 0.001 and std_loss < 0.01:
        status, symbol = "CONVERGED", "✅"
        message = "Loss is stable - model has converged"
        is_converging = True
    elif slope > 0.001:
        status, symbol = "DIVERGING", "❌"
        message = "Loss is increasing - model is NOT converging"
        is_converging = False
    else:
        status, symbol = "UNSTABLE", "⚠️"
        message = "Loss is fluctuating - convergence unclear"
        is_converging = False

    if verbose:
        print_flush(f"\n{'='*70}")
        print_flush("Convergence Analysis")
        print_flush(f"{'='*70}\n")
        print_flush(f"Statistics:")
        print_flush(f"  - Initial Loss: {initial_loss:.6f}")
        print_flush(f"  - Final Loss: {final_loss:.6f}")
        print_flush(f"  - Reduction: {reduction:+.2f}%")
        print_flush(f"  - Slope: {slope:.6f}")
        print_flush(f"\nVerdict: {symbol} {status}: {message}\n")

    return ConvergenceResult(
        status=status,
        is_converging=is_converging,
        initial_loss=initial_loss,
        final_loss=final_loss,
        reduction_percent=reduction,
        slope=slope,
        loss_history=losses,
        contexts=contexts
    )
