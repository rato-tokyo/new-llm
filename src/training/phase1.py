"""
Phase 1 訓練 (バージョン7): 超シンプル実装 + 固定学習率

主な改善点:
1. モデルが全ての最適化、状態管理を処理
2. Phase1はトークン入力と収束判定のみを担当
3. 学習率は固定（LRスケジュール削除）
4. 引数で実際の値を明示（一目瞭然）
5. デフォルト値なし（想定外の処理を防止）
"""

import torch


def train_phase1(
    model,
    token_ids,
    device,
    max_iterations,
    convergence_threshold,
    min_converged_ratio,
    learning_rate,
    dist_reg_weight,
    label
):
    """
    Phase 1: CVFP固定点学習（超シンプル版）

    モデルが全ての訓練ロジックを処理。このスクリプトはトークン入力と収束判定のみ。

    Args:
        model: 言語モデル (enable_cvfp_learning=True必須)
        token_ids: 入力トークンID [num_tokens]
        device: torch デバイス
        max_iterations: 最大イテレーション数
        convergence_threshold: 収束判定のMSE閾値
        min_converged_ratio: 早期停止の収束率閾値
        learning_rate: 学習率
        dist_reg_weight: 分布正則化の重み
        label: ログ用ラベル (例: "Train", "Val")

    Returns:
        final_contexts: 収束したコンテキストベクトル [num_tokens, context_dim]
    """
    print_flush(f"\n{'='*70}")
    print_flush(f"PHASE 1: 固定点コンテキスト学習 (CVFP){' - ' + label if label else ''}")
    print_flush(f"{'='*70}")

    model.to(device)

    # 訓練モードまたは評価モードの設定
    if model.training:
        # Phase 1訓練のセットアップ（モデルが全てを処理）
        context_params = [
            p for name, p in model.named_parameters()
            if 'token_output' not in name and 'token_embedding' not in name
        ]
        model.setup_phase1_training(context_params, learning_rate, dist_reg_weight)

    # トークン埋め込みを計算（1回のみ）
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # 収束判定用の前回コンテキストを初期化
    prev_contexts = None

    # 反復改善ループ
    for iteration in range(max_iterations):
        # イテレーション開始をモデルに通知（状態リセットを処理）
        if model.training:
            model.start_phase1_iteration(iteration)

        # トークンごとの処理
        context = torch.zeros(1, model.context_dim, device=device)
        current_contexts = []

        for t, token_embed in enumerate(token_embeds):
            # 順伝播（モデルが自動的に最適化を処理）
            if model.training:
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context.detach() if t > 0 else context
                )
            else:
                with torch.no_grad():
                    context = model._update_context_one_step(
                        token_embed.unsqueeze(0),
                        context
                    )

            current_contexts.append(context.detach())

        # 全コンテキストをスタック
        current_contexts_tensor = torch.cat(current_contexts, dim=0)

        # 収束判定
        if iteration > 0 and prev_contexts is not None:
            convergence_rate = check_convergence(
                current_contexts_tensor,
                prev_contexts,
                convergence_threshold,
                max_iterations,
                iteration
            )

            # Early stopping: 収束率が閾値を超えたら停止
            if convergence_rate >= min_converged_ratio:
                print_flush(f"  → Early stopping: 収束率 = {convergence_rate*100:.1f}%")
                break
        else:
            # Iteration 0: コンテキストを保存するのみ
            print_flush(f"Iteration 1/{max_iterations}: 順伝播のみ（コンテキスト保存）")

        # 次のイテレーションのために前回コンテキストを更新
        prev_contexts = current_contexts_tensor

    # 最終サマリー
    if prev_contexts is not None:
        with torch.no_grad():
            token_losses = ((current_contexts_tensor - prev_contexts) ** 2).mean(dim=1)
            converged_tokens = token_losses < convergence_threshold
            num_converged = converged_tokens.sum().item()
    else:
        num_converged = 0

    print_flush(f"\nPhase 1 完了: {num_converged}/{len(token_ids)} トークンが収束\n")

    return current_contexts_tensor


def check_convergence(current_contexts, prev_contexts, threshold, max_iterations, iteration):
    """
    収束判定とログ出力

    Args:
        current_contexts: 現在のコンテキスト [num_tokens, context_dim]
        prev_contexts: 前回のコンテキスト [num_tokens, context_dim]
        threshold: 収束判定のMSE閾値
        max_iterations: 最大イテレーション数
        iteration: イテレーション番号

    Returns:
        convergence_rate: 収束率 (0.0~1.0)
    """
    with torch.no_grad():
        token_losses = ((current_contexts - prev_contexts) ** 2).mean(dim=1)
        converged_tokens = token_losses < threshold
        convergence_rate = converged_tokens.float().mean().item()

    # ログ出力
    print_flush(f"Iteration {iteration+1}/{max_iterations}: 収束={convergence_rate*100:.1f}%")

    return convergence_rate


def print_flush(msg):
    """リアルタイム出力のための即座フラッシュ付き出力"""
    print(msg, flush=True)
