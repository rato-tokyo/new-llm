"""
Phase 1 訓練 (バージョン8): 究極にシンプル - モデルが収束判定も担当

主な改善点:
1. モデルが全ての最適化、状態管理、収束判定を処理
2. Phase1はトークン入力とモデルへの収束確認のみ
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
    Phase 1: CVFP固定点学習（究極シンプル版）

    モデルが全ての訓練ロジック（最適化・収束判定）を処理。
    このスクリプトはトークン入力とループ制御のみ。

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

    # 訓練モードの場合のみセットアップ
    if model.training:
        context_params = [
            p for name, p in model.named_parameters()
            if 'token_output' not in name and 'token_embedding' not in name
        ]
        model.setup_phase1_training(
            context_params,
            learning_rate,
            dist_reg_weight,
            convergence_threshold,
            min_converged_ratio
        )

    # トークン埋め込みを計算（1回のみ）
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    num_tokens = len(token_ids)

    # 反復改善ループ
    for iteration in range(max_iterations):
        # イテレーション開始をモデルに通知
        if model.training:
            model.start_phase1_iteration(iteration, num_tokens)

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

        # モデルに収束状態を更新させる
        if model.training:
            model.update_convergence_state(current_contexts_tensor)

        # ログ出力
        if iteration == 0:
            print_flush(f"Iteration 1/{max_iterations}: 順伝播のみ（コンテキスト保存）")
        else:
            convergence_rate = model.get_convergence_rate()
            print_flush(f"Iteration {iteration+1}/{max_iterations}: 収束={convergence_rate*100:.1f}%")

            # Early stopping: モデルに収束判定を委譲
            if model.is_converged():
                print_flush(f"  → Early stopping: 収束率 = {convergence_rate*100:.1f}%")
                break

    # 最終サマリー
    num_converged = model.num_converged_tokens if model.training else 0
    print_flush(f"\nPhase 1 完了: {num_converged}/{num_tokens} トークンが収束\n")

    return current_contexts_tensor


def print_flush(msg):
    """リアルタイム出力のための即座フラッシュ付き出力"""
    print(msg, flush=True)
