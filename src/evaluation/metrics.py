"""
Evaluation metrics for New-LLM

Fixed-point analysis, effective rank calculation, and other metrics.
"""

import torch
import torch.nn.functional as F


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


def check_identity_mapping(model, context_dim, device, num_samples=100, threshold=0.95):
    """
    恒等写像かどうかをチェック

    モデルがコンテキストをほぼそのまま返す（恒等写像）場合、
    トークン情報を学習していない可能性がある。

    Args:
        model: NewLLMResidualモデル
        context_dim: コンテキストの次元数
        device: デバイス（'cpu' or 'cuda'）
        num_samples: テストするサンプル数
        threshold: 恒等写像と判定するコサイン類似度の閾値（デフォルト: 0.95）

    Returns:
        dict: {
            'is_identity_mapping': bool,  # 恒等写像かどうか
            'avg_similarity': float,      # 平均コサイン類似度
            'max_similarity': float,      # 最大コサイン類似度
            'min_similarity': float,      # 最小コサイン類似度
            'samples_above_threshold': int  # 閾値を超えたサンプル数
        }
    """
    model.eval()

    similarities = []

    with torch.no_grad():
        for _ in range(num_samples):
            # ランダムなコンテキストとトークンを生成
            test_context = torch.randn(1, context_dim, device=device)
            test_token = torch.randn(1, model.embed_dim, device=device)

            # モデルでコンテキストを更新
            output_context = model._update_context_one_step(test_token, test_context)

            # コサイン類似度を計算
            similarity = F.cosine_similarity(test_context, output_context, dim=1).item()
            similarities.append(similarity)

    # 統計計算
    similarities_tensor = torch.tensor(similarities)
    avg_similarity = similarities_tensor.mean().item()
    max_similarity = similarities_tensor.max().item()
    min_similarity = similarities_tensor.min().item()
    samples_above = (similarities_tensor > threshold).sum().item()

    # 恒等写像判定
    is_identity = avg_similarity > threshold

    return {
        'is_identity_mapping': is_identity,
        'avg_similarity': avg_similarity,
        'max_similarity': max_similarity,
        'min_similarity': min_similarity,
        'samples_above_threshold': samples_above,
        'total_samples': num_samples
    }


def print_identity_mapping_warning(check_result):
    """
    恒等写像チェック結果を表示し、必要に応じて警告を出す

    Args:
        check_result: check_identity_mapping() の結果

    Returns:
        bool: 恒等写像が検出された場合True
    """
    print_flush("\n" + "="*70)
    print_flush("恒等写像チェック (Identity Mapping Check)")
    print_flush("="*70)

    avg_sim = check_result['avg_similarity']
    max_sim = check_result['max_similarity']
    min_sim = check_result['min_similarity']
    above = check_result['samples_above_threshold']
    total = check_result['total_samples']

    print_flush(f"コサイン類似度統計 ({total}サンプル):")
    print_flush(f"  平均: {avg_sim:.4f}")
    print_flush(f"  最大: {max_sim:.4f}")
    print_flush(f"  最小: {min_sim:.4f}")
    print_flush(f"  閾値(0.95)超過: {above}/{total} ({above/total*100:.1f}%)")

    if check_result['is_identity_mapping']:
        print_flush("\n⚠️  警告: 恒等写像が検出されました！")
        print_flush("    モデルが入力コンテキストをほぼそのまま返しています。")
        print_flush("    これは以下を意味する可能性があります:")
        print_flush("      - トークン情報が無視されている")
        print_flush("      - 固定点学習が自明な解に収束している")
        print_flush("      - モデルが有意義な表現を学習していない")
        print_flush("\n    推奨対応:")
        print_flush("      1. モデルの初期化を変更（std を大きく）")
        print_flush("      2. CVFP損失関数を再設計")
        print_flush("      3. より深いネットワーク構造を試す")
        print_flush("\n    Phase 2（トークン予測）はスキップすることを推奨します。")
        print_flush("="*70 + "\n")
        return True
    else:
        print_flush("\n✅ 正常: 恒等写像ではありません")
        print_flush("    モデルがトークン情報を使用してコンテキストを変換しています。")
        print_flush("="*70 + "\n")
        return False


def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)