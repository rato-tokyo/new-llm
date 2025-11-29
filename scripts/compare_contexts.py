#!/usr/bin/env python3
"""
Phase 1の最終イテレーションcontextとキャッシュ収集時のcontextを比較

目的:
- Phase 1の並列処理で得られるcontextと、キャッシュ収集（シーケンシャル）で得られるcontextの差を分析
- キャッシュ収集が本当に必要かを検討するための材料を提供

比較条件:
- 両者とも同じ初期contextベクトルから開始（前回イテレーションの最初の入力context）
- Phase 1最終イテレーション: 並列処理（前トークンのcontextを使用）
- キャッシュ収集: シーケンシャル処理（因果的依存を正確に反映）

使用方法:
  python3 scripts/compare_contexts.py
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# 設定
NUM_SAMPLES = 50
NUM_LAYERS = 6
CONTEXT_DIM = 768
EMBED_DIM = 768
RANDOM_SEED = 42


def print_flush(msg):
    print(msg, flush=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def forward_sequential_with_initial_context(
    model, token_embeds, initial_context, config, device
):
    """
    シーケンシャル処理（指定された初期contextから開始）

    Args:
        model: LLMモデル
        token_embeds: トークン埋め込み [num_tokens, embed_dim]
        initial_context: 初期context [1, context_dim]
        config: 設定
        device: デバイス

    Returns:
        contexts: 各トークンのcontext [num_tokens, context_dim]
    """
    num_tokens = len(token_embeds)
    num_input_tokens = getattr(config, 'num_input_tokens', 1)

    contexts = torch.zeros(num_tokens, model.context_dim, device=device)
    context = initial_context.clone()

    # トークン履歴を初期化
    token_history = [torch.zeros(model.embed_dim, device=device)
                     for _ in range(num_input_tokens - 1)]

    model.eval()
    with torch.no_grad():
        for i, token_embed in enumerate(token_embeds):
            token_history.append(token_embed)
            combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)

            context = model.context_block(context, combined_tokens.unsqueeze(0))
            contexts[i] = context.squeeze(0)

            if len(token_history) > num_input_tokens:
                token_history = token_history[-num_input_tokens:]

    return contexts


def forward_parallel_single_iteration(
    model, token_embeds, previous_contexts, config, device
):
    """
    並列処理（1イテレーション分）

    Phase 1と同じ方式: token i は previous_contexts[i-1] を入力として使用

    Args:
        model: LLMモデル
        token_embeds: トークン埋め込み [num_tokens, embed_dim]
        previous_contexts: 前回イテレーションのcontext [num_tokens, context_dim]
        config: 設定
        device: デバイス

    Returns:
        contexts: 各トークンのcontext [num_tokens, context_dim]
    """
    num_tokens = len(token_embeds)
    num_input_tokens = getattr(config, 'num_input_tokens', 1)

    contexts = torch.zeros(num_tokens, model.context_dim, device=device)

    model.eval()
    with torch.no_grad():
        for i in range(num_tokens):
            # token履歴を構築
            start_idx = max(0, i - num_input_tokens + 1)
            token_window = token_embeds[start_idx:i+1]

            # パディング（必要な場合）
            if len(token_window) < num_input_tokens:
                padding = torch.zeros(
                    num_input_tokens - len(token_window),
                    model.embed_dim,
                    device=device
                )
                token_window = torch.cat([padding, token_window], dim=0)

            combined_tokens = token_window.flatten()

            # 並列処理: token i は previous_contexts[i-1] を使用
            if i == 0:
                # 最初のトークン: previous_contextsの最後（または初期値）
                prev_ctx = previous_contexts[-1:].clone()
            else:
                prev_ctx = previous_contexts[i-1:i].clone()

            context = model.context_block(prev_ctx, combined_tokens.unsqueeze(0))
            contexts[i] = context.squeeze(0)

    return contexts


def main():
    print_flush("=" * 60)
    print_flush("CONTEXT COMPARISON: Parallel vs Sequential")
    print_flush("=" * 60)

    set_seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_flush(f"Device: {device}")

    # インポート
    from config import ResidualConfig
    from src.models.llm import LLM
    from src.providers.data import MemoryDataProvider
    from src.trainers.phase1 import MemoryPhase1Trainer

    # 設定
    config = ResidualConfig()
    config.num_layers = NUM_LAYERS
    config.context_dim = CONTEXT_DIM
    config.embed_dim = EMBED_DIM
    config.num_samples = NUM_SAMPLES

    # データロード
    print_flush(f"\nLoading data ({NUM_SAMPLES} samples)...")
    data_provider = MemoryDataProvider(config, shuffle_samples=False)
    full_train_token_ids, val_token_ids = data_provider.load_data()

    # 検証部分を除外
    val_size = len(val_token_ids)
    train_token_ids = full_train_token_ids[:-val_size].to(device)

    print_flush(f"Train tokens: {len(train_token_ids):,}")

    # モデル作成
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        num_input_tokens=config.num_input_tokens,
        num_context_splits=getattr(config, 'num_context_splits', 1),
        use_pretrained_embeddings=config.use_pretrained_embeddings,
        use_weight_tying=config.use_weight_tying,
        token_input_all_layers=getattr(config, 'token_input_all_layers', False)
    )
    model.to(device)

    # Phase 1訓練（通常通り）
    print_flush("\n--- Phase 1 Training ---")
    phase1_trainer = MemoryPhase1Trainer(model, config, device)

    # trainを実行（return_all_layers=Falseで最終contextのみ取得）
    # ただし、最終イテレーションのprevious_contextsも欲しい
    # → trainの内部処理を模倣して比較用データを取得

    # トークン埋め込みを計算
    with torch.no_grad():
        token_embeds_gpu = model.token_embedding(train_token_ids.unsqueeze(0))
        token_embeds_gpu = model.embed_norm(token_embeds_gpu).squeeze(0)
        token_embeds = token_embeds_gpu.cpu()

    num_tokens = len(train_token_ids)

    # Phase 1訓練を手動で実行（previous_contextsを各イテレーションで保存）
    print_flush(f"\nRunning Phase 1 ({config.phase1_max_iterations} iterations)...")

    context_params = list(model.context_block.parameters())
    optimizer = torch.optim.Adam(context_params, lr=config.phase1_learning_rate)

    previous_contexts = None
    iteration_contexts_history = []  # 各イテレーション終了時のcontextsを保存

    model.train()

    for iteration in range(config.phase1_max_iterations):
        start_time = time.time()

        if iteration == 0:
            # Iteration 0: ランダム初期化
            previous_contexts = torch.randn(num_tokens, model.context_dim) * 0.01
            print_flush(f"  Iter {iteration+1}: random init")
            iteration_contexts_history.append(previous_contexts.clone())
            continue

        # 並列処理
        token_embeds_gpu = token_embeds.to(device)
        previous_contexts_gpu = previous_contexts.to(device)

        contexts, total_loss, _, _ = phase1_trainer._forward_parallel_with_grad_accum(
            token_embeds_gpu, previous_contexts_gpu, optimizer
        )

        # 収束率
        convergence_rate = phase1_trainer._compute_convergence_rate(
            contexts, previous_contexts_gpu, num_tokens
        )

        previous_contexts = contexts.detach().cpu()
        iteration_contexts_history.append(previous_contexts.clone())

        del token_embeds_gpu, previous_contexts_gpu, contexts
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        print_flush(
            f"  Iter {iteration+1}: conv={convergence_rate*100:.0f}% "
            f"loss={total_loss:.4f} [{elapsed:.1f}s]"
        )

        # 早期停止
        min_iterations = getattr(config, 'phase1_min_iterations', 3)
        if iteration + 1 >= min_iterations and convergence_rate >= config.phase1_min_converged_ratio:
            print_flush(f"  → Converged at iter {iteration+1}")
            break

    final_iteration = len(iteration_contexts_history) - 1
    print_flush(f"\nTotal iterations: {final_iteration + 1}")

    # 最終イテレーションのcontext（並列処理で得られたもの）
    parallel_contexts = iteration_contexts_history[-1].to(device)

    # 比較のための初期context: 前イテレーション開始時の状態
    # Phase 1では、最初のトークンは previous_contexts[-1] を使用
    # キャッシュ収集時も同じ初期contextを使用するため、
    # 前イテレーション終了時の最後のcontextを初期値とする
    if final_iteration >= 1:
        # 前イテレーションの最終context（= 今イテレーションの初期context）
        initial_context_for_comparison = iteration_contexts_history[-2][-1:].to(device)
    else:
        # イテレーション0しかない場合（ほぼありえない）
        initial_context_for_comparison = torch.zeros(1, model.context_dim, device=device)

    print_flush(f"\nInitial context norm: {initial_context_for_comparison.norm().item():.4f}")

    # シーケンシャル処理（同じ初期contextから開始）
    print_flush("\n--- Sequential Processing (for comparison) ---")
    sequential_start = time.time()

    # Phase 2用のキャッシュ収集と同じ条件：最後のトークンを除く
    input_token_embeds = token_embeds[:-1].to(device)

    sequential_contexts = forward_sequential_with_initial_context(
        model, input_token_embeds, initial_context_for_comparison, config, device
    )

    sequential_elapsed = time.time() - sequential_start
    print_flush(f"Sequential processing: {len(input_token_embeds):,} tokens [{sequential_elapsed:.1f}s]")

    # 並列contextも同じ範囲で比較（最後のトークンを除く）
    parallel_contexts_compare = parallel_contexts[:-1]

    # === 比較分析 ===
    print_flush("\n" + "=" * 60)
    print_flush("COMPARISON RESULTS")
    print_flush("=" * 60)

    # 1. 全体的な差
    diff = parallel_contexts_compare - sequential_contexts
    mse = (diff ** 2).mean().item()
    rmse = mse ** 0.5
    mae = diff.abs().mean().item()

    print_flush("\n1. Overall Difference:")
    print_flush(f"   MSE:  {mse:.6f}")
    print_flush(f"   RMSE: {rmse:.6f}")
    print_flush(f"   MAE:  {mae:.6f}")

    # 2. コサイン類似度
    cos_sim = torch.nn.functional.cosine_similarity(
        parallel_contexts_compare, sequential_contexts, dim=1
    )
    print_flush("\n2. Cosine Similarity:")
    print_flush(f"   Mean:   {cos_sim.mean().item():.6f}")
    print_flush(f"   Min:    {cos_sim.min().item():.6f}")
    print_flush(f"   Max:    {cos_sim.max().item():.6f}")
    print_flush(f"   Std:    {cos_sim.std().item():.6f}")

    # 3. ノルムの比較
    parallel_norms = parallel_contexts_compare.norm(dim=1)
    sequential_norms = sequential_contexts.norm(dim=1)
    norm_diff = (parallel_norms - sequential_norms).abs()

    print_flush("\n3. Norm Comparison:")
    print_flush(f"   Parallel norm (mean):   {parallel_norms.mean().item():.4f}")
    print_flush(f"   Sequential norm (mean): {sequential_norms.mean().item():.4f}")
    print_flush(f"   Norm diff (mean):       {norm_diff.mean().item():.4f}")
    print_flush(f"   Norm diff (max):        {norm_diff.max().item():.4f}")

    # 4. 位置による差の分析
    print_flush("\n4. Position-based Analysis:")

    # 最初の100トークン
    first_100 = diff[:100]
    first_100_mse = (first_100 ** 2).mean().item()
    first_100_cos = cos_sim[:100].mean().item()
    print_flush(f"   First 100 tokens:  MSE={first_100_mse:.6f}, CosSim={first_100_cos:.6f}")

    # 中間の100トークン
    mid_start = len(diff) // 2 - 50
    mid_100 = diff[mid_start:mid_start+100]
    mid_100_mse = (mid_100 ** 2).mean().item()
    mid_100_cos = cos_sim[mid_start:mid_start+100].mean().item()
    print_flush(f"   Middle 100 tokens: MSE={mid_100_mse:.6f}, CosSim={mid_100_cos:.6f}")

    # 最後の100トークン
    last_100 = diff[-100:]
    last_100_mse = (last_100 ** 2).mean().item()
    last_100_cos = cos_sim[-100:].mean().item()
    print_flush(f"   Last 100 tokens:   MSE={last_100_mse:.6f}, CosSim={last_100_cos:.6f}")

    # 5. 相対誤差
    relative_error = diff.norm(dim=1) / (sequential_contexts.norm(dim=1) + 1e-8)
    print_flush("\n5. Relative Error (||diff|| / ||sequential||):")
    print_flush(f"   Mean: {relative_error.mean().item():.6f}")
    print_flush(f"   Max:  {relative_error.max().item():.6f}")
    print_flush(f"   95th percentile: {torch.quantile(relative_error, 0.95).item():.6f}")

    # 6. 判定
    print_flush("\n" + "=" * 60)
    print_flush("CONCLUSION")
    print_flush("=" * 60)

    mean_cos_sim = cos_sim.mean().item()
    if mean_cos_sim > 0.99:
        print_flush("✅ Very High Similarity (>0.99)")
        print_flush("   → キャッシュ収集は省略可能かもしれません")
        print_flush("   → 並列処理の結果をそのままPhase 2で使用できる可能性")
    elif mean_cos_sim > 0.95:
        print_flush("⚠️ High Similarity (0.95-0.99)")
        print_flush("   → 近いが完全ではない")
        print_flush("   → Phase 2の精度に影響する可能性あり")
    elif mean_cos_sim > 0.90:
        print_flush("⚠️ Moderate Similarity (0.90-0.95)")
        print_flush("   → キャッシュ収集は必要と思われる")
    else:
        print_flush("❌ Low Similarity (<0.90)")
        print_flush("   → キャッシュ収集は必須")
        print_flush("   → 並列処理とシーケンシャル処理で大きく異なる")

    # クリーンアップ
    data_provider.close()

    print_flush("\nDone.")


if __name__ == '__main__':
    main()
