"""
Phase 1 Trainer: CVFP固定点学習（分離アーキテクチャ版）

ContextBlockのみを学習（TokenBlockは未使用）

Iteration 0: シーケンシャル処理（学習なし、previous_contextsを初期化）
Iteration 1+: 並列処理（前回contextを使用、23x高速化）

CVFP理論: CVFP損失 = MSE(contexts, previous_contexts)
  - 前回のコンテキストと比較して固定点への収束を学習
  - previous_contextsは毎イテレーション更新

設定: dist_reg_weight=0.9（並列版最適化: 55.9% ER達成）
"""

import sys
import torch
import torch.nn.functional as F
import time


def phase1_train(
    model,
    token_ids,
    device,
    max_iterations=10,
    convergence_threshold=0.1,
    min_converged_ratio=0.95,
    learning_rate=0.002,
    dist_reg_weight=0.9,
    label="Train"
):
    """
    Phase 1訓練: CVFP固定点学習（ContextBlockのみ）

    分離アーキテクチャ:
    - ContextBlockのみを学習
    - TokenBlockは未使用（Phase 2で学習）

    Args:
        model: LLMモデル
        token_ids: 入力トークンID [num_tokens]
        device: torch device
        max_iterations: 最大イテレーション数
        convergence_threshold: 収束判定のMSE閾値
        min_converged_ratio: 早期停止の収束率閾値
        learning_rate: 学習率
        dist_reg_weight: 多様性正則化の重み（並列版最適値: 0.9）
        label: ログ用ラベル

    Returns:
        final_contexts: 収束したコンテキストベクトル [num_tokens, context_dim]
    """
    model.train()
    num_tokens = len(token_ids)

    print_flush(f"\n{'='*70}")
    print_flush(f"PHASE 1: 固定点コンテキスト学習 (ContextBlock) - {label}")
    print_flush(f"{'='*70}")

    # Optimizer設定: ContextBlockのパラメータのみ
    if model.use_separated_architecture:
        context_params = list(model.context_block.parameters())
        print_flush(f"Training ContextBlock only ({sum(p.numel() for p in context_params)} parameters)")
    else:
        # Legacy: token_outputとtoken_embedding以外
        context_params = [
            p for name, p in model.named_parameters()
            if 'token_output' not in name and 'token_embedding' not in name
        ]
    optimizer = torch.optim.Adam(context_params, lr=learning_rate)

    # トークン埋め込みを計算（1回のみ）
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)  # [num_tokens, embed_dim]

    # 前回のコンテキスト（CVFP損失計算用）
    previous_contexts = None

    # イテレーションループ
    for iteration in range(max_iterations):
        start_time = time.time()

        # Iteration 0: シーケンシャル（必須）- 学習なし
        if iteration == 0:
            contexts = forward_all_tokens_sequential(
                model, token_embeds, None, device
            )
            previous_contexts = contexts.detach()
            elapsed = time.time() - start_time
            print_flush(f"Iteration 1/{max_iterations}: 順伝播のみ（シーケンシャル） [{elapsed:.2f}s]")
            continue

        # Iteration 1+: 並列処理（23x高速化）
        contexts = forward_all_tokens_parallel(
            model, token_embeds, previous_contexts, device
        )

        # 結合損失による最適化（CVFP + Diversity）
        # CVFP理論: 前回のコンテキストと比較（固定点への収束）
        cvfp_loss = compute_cvfp_loss(contexts, previous_contexts)
        diversity_loss = compute_diversity_loss(contexts)

        # 重み付き結合損失（並列版最適: dwr=0.9）
        total_loss = (1 - dist_reg_weight) * cvfp_loss + dist_reg_weight * diversity_loss

        if not torch.isnan(total_loss) and not torch.isinf(total_loss):
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 収束状態を更新
        convergence_rate = update_convergence(contexts, previous_contexts, convergence_threshold, num_tokens)
        previous_contexts = contexts.detach()

        elapsed = time.time() - start_time

        # ログ出力
        log_message = (
            f"Iteration {iteration+1}/{max_iterations}: "
            f"収束={convergence_rate*100:.1f}% | "
            f"Total={total_loss.item():.6f} | "
            f"CVFP={cvfp_loss.item():.6f} | "
            f"Div={diversity_loss.item():.6f} | "
            f"Time={elapsed:.2f}s"
        )
        print_flush(log_message)

        # Early stopping判定
        if convergence_rate >= min_converged_ratio:
            print_flush(f"  → Early stopping: 収束率 = {convergence_rate*100:.1f}%")
            break

        # GPU メモリ最適化
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 最終サマリー
    num_converged = int(convergence_rate * num_tokens)
    print_flush(f"\nPhase 1 完了: {num_converged}/{num_tokens} トークンが収束\n")

    return contexts.detach()


def forward_all_tokens_sequential(model, token_embeds, previous_contexts, device):
    """
    全トークンを順次処理（シーケンシャル版）

    分離アーキテクチャ: ContextBlockのみ使用（TokenBlockは未使用）

    Args:
        model: LLMモデル
        token_embeds: トークン埋め込み [num_tokens, embed_dim]
        previous_contexts: 前回の最終コンテキスト（Noneの場合はゼロ初期化）
        device: torch device

    Returns:
        contexts: [num_tokens, context_dim]
    """
    # イテレーション間でコンテキストを引き継ぐ
    if previous_contexts is None:
        context = torch.zeros(1, model.context_dim, device=device)
    else:
        context = previous_contexts[-1].unsqueeze(0).detach()

    # 全トークンを順次処理
    context_list = []
    for t, token_embed in enumerate(token_embeds):
        token_embed_current = token_embed.unsqueeze(0)

        # ContextBlockを通過（TokenBlockは使用しない）
        if model.use_separated_architecture:
            context = model.context_block(context, token_embed_current)
        else:
            # Legacy: CVFPブロックを通過
            for block in model.blocks:
                context, token_embed_current = block(context, token_embed_current)

        context_list.append(context.squeeze(0))

    # contextsテンソルを構築
    contexts = torch.stack(context_list)

    return contexts


def forward_all_tokens_parallel(model, token_embeds, previous_contexts, device, batch_size=8192):
    """
    全トークンを並列処理（前回contextを使用）

    分離アーキテクチャ: ContextBlockのみ使用（TokenBlockは未使用）

    並列化により23x高速化達成（265秒 → 11秒）
    1トークン分のcontext遅延により情報の若干の遅れがあるが、
    dist_reg_weight=0.9により多様性を維持（55.9% ER達成）

    大規模データセット対応: バッチ分割処理でメモリオーバーフロー防止

    Args:
        model: LLMモデル
        token_embeds: トークン埋め込み [num_tokens, embed_dim]
        previous_contexts: 前回イテレーションの全context [num_tokens, context_dim]
        device: torch device
        batch_size: バッチサイズ（デフォルト: 8192）

    Returns:
        contexts: [num_tokens, context_dim]
    """
    num_tokens = len(token_embeds)

    # 各トークンに割り当てるcontextを準備
    # Token i には previous_contexts[i-1] を使用（Token 0 は前イテレーション最終）
    contexts_for_batch = torch.zeros(num_tokens, model.context_dim, device=device)

    if previous_contexts is not None:
        # Token 1~N: 前のトークンのcontextを使用（1トークン分のずれ）
        contexts_for_batch[1:] = previous_contexts[:-1].detach()
        # Token 0: 最後のトークンのcontextを使用（イテレーション引継ぎ）
        contexts_for_batch[0] = previous_contexts[-1].detach()

    # 大規模データセット: バッチ分割処理（メモリオーバーフロー防止）
    if num_tokens > batch_size:
        all_contexts = []
        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)
            batch_contexts = contexts_for_batch[start_idx:end_idx]
            batch_embeds = token_embeds[start_idx:end_idx]

            if model.use_separated_architecture:
                batch_output = model.context_block(batch_contexts, batch_embeds)
            else:
                batch_output = batch_contexts
                batch_tokens = batch_embeds
                for block in model.blocks:
                    batch_output, batch_tokens = block(batch_output, batch_tokens)

            all_contexts.append(batch_output)

        current_contexts = torch.cat(all_contexts, dim=0)
    else:
        # 小規模データセット: 一括処理
        if model.use_separated_architecture:
            current_contexts = model.context_block(contexts_for_batch, token_embeds)
        else:
            current_contexts = contexts_for_batch
            current_tokens = token_embeds
            for block in model.blocks:
                current_contexts, current_tokens = block(current_contexts, current_tokens)

    return current_contexts


def compute_cvfp_loss(contexts, previous_contexts):
    """
    CVFP損失: 前回のコンテキストとのMSE（固定点への収束）

    Args:
        contexts: 現在のコンテキスト [num_tokens, context_dim]
        previous_contexts: 前回イテレーションのコンテキスト [num_tokens, context_dim]

    Returns:
        cvfp_loss: MSE損失（スカラー）
    """
    return F.mse_loss(contexts, previous_contexts)


def compute_diversity_loss(contexts):
    """
    多様性損失: 全トークンの平均からの偏差（負の損失で最大化）

    Args:
        contexts: 現在のコンテキスト [num_tokens, context_dim]

    Returns:
        diversity_loss: 多様性損失（スカラー）
    """
    context_mean = contexts.mean(dim=0)  # [context_dim]
    deviation = contexts - context_mean  # [num_tokens, context_dim]
    diversity_loss = -torch.norm(deviation, p=2) / len(contexts)
    return diversity_loss


def update_convergence(current_contexts, previous_contexts, threshold, num_tokens):
    """
    収束状態を更新

    Args:
        current_contexts: [num_tokens, context_dim]
        previous_contexts: [num_tokens, context_dim]
        threshold: 収束判定のMSE閾値
        num_tokens: トークン数

    Returns:
        convergence_rate: 収束率（0.0~1.0）
    """
    with torch.no_grad():
        token_losses = ((current_contexts - previous_contexts) ** 2).mean(dim=1)
        converged_tokens = token_losses < threshold
        num_converged = converged_tokens.sum().item()
        convergence_rate = num_converged / num_tokens

    return convergence_rate


def print_flush(msg):
    """リアルタイム出力（バッファリング完全無効化）"""
    print(msg, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
