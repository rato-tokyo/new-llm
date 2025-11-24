"""
Evaluation metrics for New-LLM

Fixed-point analysis, effective rank calculation, and other metrics.
"""

import sys
import torch
import torch.nn.functional as F


def print_flush(msg):
    """Print with immediate flush"""
    print(msg)
    sys.stdout.flush()


def analyze_fixed_points(contexts, label="", verbose=True):
    """
    Analyze fixed-point contexts for quality metrics.

    Args:
        contexts: Fixed-point contexts [num_tokens, context_dim]
        label: Label for display (e.g., "Train", "Val")
        verbose: If True, print detailed analysis

    Returns:
        dict: Analysis metrics including effective rank
    """
    if verbose:
        print_flush(f"\n{'='*70}")
        print_flush(f"FIXED-POINT ANALYSIS{' - ' + label if label else ''}")
        print_flush(f"{'='*70}\n")

    device = contexts.device
    num_tokens = contexts.shape[0]
    context_dim = contexts.shape[1]

    # 1. Global Attractor Detection
    # Compute pairwise L2 distances
    distances = torch.cdist(contexts, contexts, p=2)
    # Exclude diagonal (self-distances)
    mask = ~torch.eye(num_tokens, dtype=bool, device=device)
    pairwise_distances = distances[mask]

    avg_distance = pairwise_distances.mean().item()
    min_distance = pairwise_distances.min().item()
    max_distance = pairwise_distances.max().item()

    # Cosine similarity
    normalized = F.normalize(contexts, p=2, dim=1)
    cosine_sim = torch.mm(normalized, normalized.t())
    pairwise_cosine = cosine_sim[mask]

    avg_cosine = pairwise_cosine.mean().item()
    min_cosine = pairwise_cosine.min().item()
    max_cosine = pairwise_cosine.max().item()

    if verbose:
        print_flush("1. Global Attractor Detection:")
        print_flush(f"  Avg L2 Distance: {avg_distance:.6f} (Range: [{min_distance:.6f}, {max_distance:.6f}])")
        print_flush(f"  Avg Cosine Sim:  {avg_cosine:.6f} (Range: [{min_cosine:.6f}, {max_cosine:.6f}])")

    # Global attractor detection
    is_global_attractor = avg_distance < 0.1 and avg_cosine > 0.99
    if is_global_attractor:
        if verbose:
            print_flush("  ⚠️ DEGENERATE: Global attractor detected (all contexts nearly identical)")
    else:
        if verbose:
            print_flush("  ✅ Token-specific fixed points")

    # 2. Zero Solution Detection
    norms = torch.norm(contexts, dim=1)
    avg_norm = norms.mean().item()
    min_norm = norms.min().item()
    max_norm = norms.max().item()

    if verbose:
        print_flush("\n2. Zero Solution Detection:")
        print_flush(f"  Avg Norm: {avg_norm:.6f} (Range: [{min_norm:.6f}, {max_norm:.6f}])")

    is_zero_solution = avg_norm < 0.1
    if is_zero_solution:
        if verbose:
            print_flush("  ⚠️ DEGENERATE: Near-zero contexts")
    else:
        if verbose:
            print_flush("  ✅ Non-zero contexts")

    # 3. Distribution Statistics
    norm_mean = norms.mean().item()
    norm_median = norms.median().item()
    norm_std = norms.std().item()

    pairwise_mean = pairwise_distances.mean().item()
    pairwise_median = pairwise_distances.median().item()

    # Sparsity
    sparsity = (contexts.abs() < 0.01).float().mean().item()

    if verbose:
        print_flush("\n3. Distribution Statistics:")
        print_flush(f"  Norm - Mean: {norm_mean:.4f}, Median: {norm_median:.4f}, Std: {norm_std:.4f}")
        print_flush(f"  Pairwise Dist - Mean: {pairwise_mean:.4f}, Median: {pairwise_median:.4f}")
        print_flush(f"  Sparsity: {sparsity*100:.2f}% of values < 0.01")

    # 4. Information Content (Effective Rank)
    # Compute SVD
    U, S, V = torch.svd(contexts)

    # Actual rank (number of non-zero singular values)
    # Consider values > 1e-6 as non-zero to account for numerical precision
    actual_rank = (S > 1e-6).sum().item()

    # Effective rank (entropy-based)
    S_normalized = S / S.sum()
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    if verbose:
        print_flush("\n4. Information Content:")
        print_flush(f"  Actual Rank: {actual_rank} / {context_dim} ({actual_rank/context_dim*100:.1f}%)")
        print_flush(f"  Effective Rank: {effective_rank:.2f} / {context_dim} ({effective_rank/context_dim*100:.1f}%)")
        print_flush(f"  Top 5 Singular Values: {S[:5].tolist()}")

    # Determine quality
    if effective_rank < context_dim * 0.3:
        if verbose:
            print_flush(f"  ⚠️ Low dimensional diversity (ER={effective_rank:.2f}/{context_dim})")
    else:
        if verbose:
            print_flush("  ✅ Good diversity")

    if verbose:
        print_flush(f"{'='*70}\n")

    return {
        "actual_rank": actual_rank,
        "effective_rank": effective_rank,
        "avg_distance": avg_distance,
        "avg_cosine": avg_cosine,
        "avg_norm": avg_norm,
        "is_global_attractor": is_global_attractor,
        "is_zero_solution": is_zero_solution,
        "singular_values": S.tolist()
    }


def check_identity_mapping(model, token_embeds, contexts, device):
    """
    恒等写像チェック: 学習が起きているかを確認

    ランダム初期化モデルと訓練済みモデルの出力を比較し、
    学習が実際に起きているかを確認。

    Args:
        model: 訓練済みモデル
        token_embeds: トークン埋め込み [num_tokens, embed_dim]
        contexts: 学習されたコンテキスト [num_tokens, context_dim]
        device: torch device

    Returns:
        dict: 恒等写像チェック結果
    """
    print_flush("\n" + "="*70)
    print_flush("IDENTITY MAPPING CHECK (恒等写像チェック)")
    print_flush("="*70 + "\n")

    model.eval()

    # 学習されたコンテキストのノルムを確認
    learned_norms = torch.norm(contexts, dim=1)
    avg_learned_norm = learned_norms.mean().item()
    std_learned_norm = learned_norms.std().item()

    print_flush("1. Learned Context Statistics:")
    print_flush(f"   Average Norm: {avg_learned_norm:.6f}")
    print_flush(f"   Std Dev: {std_learned_norm:.6f}")

    # ゼロコンテキストとの差分
    zero_distance = avg_learned_norm

    if zero_distance < 0.1:
        print_flush("   ❌ DEGENERATE: Contexts are near zero")
        print_flush("   → Model is NOT learning meaningful context updates")
        is_identity = True
    else:
        print_flush("   ✅ PASSED: Contexts are non-zero")

        # さらに詳細なチェック：コンテキストの多様性
        # 全コンテキストが同じ値か確認
        with torch.no_grad():
            context_mean = contexts.mean(dim=0)  # [context_dim]
            deviations = contexts - context_mean.unsqueeze(0)
            avg_deviation = torch.norm(deviations, dim=1).mean().item()

        print_flush(f"\n2. Context Diversity:")
        print_flush(f"   Average deviation from mean: {avg_deviation:.6f}")

        if avg_deviation < 0.1:
            print_flush("   ❌ DEGENERATE: All contexts are identical (global attractor)")
            is_identity = True
        else:
            print_flush("   ✅ PASSED: Contexts are diverse")
            is_identity = False

    # トークン埋め込みとの類似度（恒等写像確認）
    # コンテキストとトークン埋め込みの次元が異なる場合の対処
    if contexts.shape[1] == token_embeds.shape[1]:
        embed_similarity = F.cosine_similarity(contexts, token_embeds, dim=1).mean().item()

        print_flush(f"\n3. Context vs Token Embedding Similarity:")
        print_flush(f"   Cosine Similarity: {embed_similarity:.6f}")

        if embed_similarity > 0.95:
            print_flush("   ⚠️ WARNING: Contexts too similar to token embeddings")
            print_flush("   → Possible identity mapping (no transformation)")
        else:
            print_flush("   ✅ PASSED: Contexts are transformed from embeddings")
    else:
        embed_similarity = None
        print_flush(f"\n3. Context vs Token Embedding Similarity:")
        print_flush(f"   (Skipped: dimension mismatch {contexts.shape[1]} vs {token_embeds.shape[1]})")

    print_flush("="*70 + "\n")

    return {
        "context_diff_from_zero": avg_deviation if not is_identity else 0.0,
        "embed_similarity": embed_similarity,
        "is_identity": is_identity
    }


def check_cvfp_convergence(trainer, token_ids, device, max_test_iterations=5):
    """
    CVFP収束チェック: 固定点学習ができているかを確認

    追加イテレーションを実行し、コンテキストが固定点に収束しているかチェック。

    Args:
        trainer: Phase1Trainer インスタンス
        token_ids: 入力トークンID
        device: torch device
        max_test_iterations: 追加テストイテレーション数

    Returns:
        dict: CVFP収束チェック結果
    """
    print_flush("\n" + "="*70)
    print_flush("CVFP CONVERGENCE CHECK (固定点収束チェック)")
    print_flush("="*70 + "\n")

    model = trainer.model
    model.eval()

    with torch.no_grad():
        # トークン埋め込み取得
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

        # 複数イテレーション実行
        iteration_contexts = []

        for iteration in range(max_test_iterations):
            iteration_context_list = []

            # CRITICAL FIX: イテレーション間でコンテキストを引き継ぐ
            if iteration == 0:
                # 初回のみゼロ初期化
                context = torch.zeros(1, model.context_dim, device=device)
            else:
                # 前イテレーションの最終コンテキストを初期値として使用
                context = iteration_contexts[-1][-1].unsqueeze(0).detach()

            for token_embed in token_embeds:
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context
                )
                iteration_context_list.append(context)

            iteration_contexts.append(torch.cat(iteration_context_list, dim=0))

        # イテレーション間の変化を測定
        convergence_diffs = []
        for i in range(1, max_test_iterations):
            diff = torch.norm(
                iteration_contexts[i] - iteration_contexts[i-1],
                dim=1
            ).mean().item()
            convergence_diffs.append(diff)

    print_flush(f"Testing convergence over {max_test_iterations} iterations:\n")

    for i, diff in enumerate(convergence_diffs):
        print_flush(f"  Iteration {i+1} → {i+2}: Avg change = {diff:.6f}")

    # 最終イテレーション間の変化
    final_diff = convergence_diffs[-1] if convergence_diffs else float('inf')

    print_flush(f"\nFinal convergence (iteration {max_test_iterations-1} → {max_test_iterations}):")
    print_flush(f"  Average change: {final_diff:.6f}")

    if final_diff < 1e-4:
        print_flush("  ✅ EXCELLENT: Strong fixed-point convergence")
        convergence_quality = "excellent"
    elif final_diff < 1e-3:
        print_flush("  ✅ GOOD: Fixed-point convergence achieved")
        convergence_quality = "good"
    elif final_diff < 1e-2:
        print_flush("  ⚠️ MODERATE: Weak convergence (acceptable)")
        convergence_quality = "moderate"
    else:
        print_flush("  ❌ FAILED: No fixed-point convergence")
        print_flush("  → Model is NOT learning stable fixed points")
        convergence_quality = "failed"

    print_flush("="*70 + "\n")

    return {
        "convergence_diffs": convergence_diffs,
        "final_diff": final_diff,
        "quality": convergence_quality
    }


def analyze_singular_vectors(contexts, token_ids=None, tokenizer=None, top_k=2, top_tokens=5):
    """
    Analyze which tokens contribute most to top singular vectors.

    Args:
        contexts: Fixed-point contexts [num_tokens, context_dim]
        token_ids: Token IDs for decoding (optional)
        tokenizer: Tokenizer for decoding (optional)
        top_k: Number of top singular vectors to analyze
        top_tokens: Number of top contributing tokens to show

    Returns:
        dict: Analysis of token contributions to singular vectors
    """
    print_flush("\n5. Singular Vector Analysis:")
    print_flush("  Analyzing which tokens contribute to top singular vectors...\n")

    # Compute SVD
    U, S, V = torch.svd(contexts)

    results = {}
    for i in range(min(top_k, len(S))):
        # Get projections onto singular vector i
        projections = U[:, i] * S[i]

        # Find top contributing tokens
        top_indices = torch.argsort(projections.abs(), descending=True)[:top_tokens]

        print_flush(f"  --- Singular Vector {i+1} (Value: {S[i]:.2f}) ---")
        print_flush(f"  Top {top_tokens} tokens contributing to SV{i+1}:")

        sv_results = []
        for j, idx in enumerate(top_indices):
            idx_val = idx.item()
            proj_val = projections[idx].item()

            if token_ids is not None and tokenizer is not None:
                token = tokenizer.decode([token_ids[idx_val].item()])
                print_flush(f"    {j+1}. Token {token_ids[idx_val].item()} '{token}': proj={proj_val:.3f}")
            else:
                print_flush(f"    {j+1}. Token index {idx_val}: proj={proj_val:.3f}")

            sv_results.append({
                "index": idx_val,
                "projection": proj_val,
                "token_id": token_ids[idx_val].item() if token_ids is not None else None
            })

        results[f"sv_{i+1}"] = {
            "singular_value": S[i].item(),
            "top_tokens": sv_results
        }

    return results


