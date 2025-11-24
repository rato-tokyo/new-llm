"""
訓練診断ツール (Training Diagnostics)

訓練中の問題を検出するための診断機能:
- 恒等写像検出（Identity Mapping Detection）
- 勾配フロー検証（Gradient Flow Verification）
- その他の訓練問題診断
"""

import torch
import torch.nn.functional as F


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


def check_gradient_flow(trainer, token_ids, device, num_tokens_to_check=100):
    """
    トークン間の勾配フロー（gradient flow）を検証

    【検証方法】:
    1. Phase1Trainerで訓練処理を1イテレーション実行
    2. 最初のトークンの文脈ベクトルの値を確認
    3. ほぼゼロ = 勾配が届いていない（detach()バグ）

    【このバグを検出】:
    - トークン間でcontextがdetach()されている
    - 系列全体での勾配伝播が途切れている
    - CVFP学習が機能していない

    Args:
        trainer: Phase1Trainerインスタンス
        token_ids: テスト用トークンID [num_tokens]
        device: デバイス
        num_tokens_to_check: 検証するトークン数（デフォルト: 100）

    Returns:
        dict: {
            'has_gradient_flow': bool,      # 勾配フローがあるか
            'first_context_norm': float,    # 最初のトークンの文脈ノルム
            'mean_context_norm': float,     # 全トークンの平均ノルム
            'norm_ratio': float,            # 比率（first/mean）
            'status': str                   # 診断結果
        }
    """
    # テストデータの準備
    test_tokens = token_ids[:num_tokens_to_check].to(device)

    # トークン埋め込み取得
    with torch.no_grad():
        token_embeds = trainer.model.token_embedding(test_tokens.unsqueeze(0).to(device))
        token_embeds = trainer.model.embed_norm(token_embeds).squeeze(0)

    # 1イテレーション訓練実行（内部でcontext伝播）
    trainer.model.train()
    trainer.previous_contexts = None  # 初期化
    contexts = trainer._process_tokens(token_embeds, device, is_training=True)

    # 最初と平均のノルムを計算
    first_context_norm = contexts[0].norm().item()
    mean_context_norm = contexts.norm(dim=1).mean().item()

    # ノルム比率（first/mean）
    norm_ratio = first_context_norm / mean_context_norm if mean_context_norm > 1e-6 else 0.0

    # 判定: 最初のコンテキストが異常に小さい = 勾配が届いていない
    # 正常なら first_context_norm ≈ mean_context_norm （比率 ≈ 1.0）
    # バグあり detach() → first_context_norm ≈ 0 （比率 ≈ 0）
    has_flow = norm_ratio > 0.5  # 平均の50%以上ならOK

    status = "PASS" if has_flow else "FAIL"

    return {
        'has_gradient_flow': has_flow,
        'first_context_norm': first_context_norm,
        'mean_context_norm': mean_context_norm,
        'norm_ratio': norm_ratio,
        'status': status
    }


def print_gradient_flow_result(check_result):
    """
    勾配フロー検証結果を表示

    Args:
        check_result: check_gradient_flow() の結果

    Returns:
        bool: 勾配フローに問題がある場合True
    """
    print_flush("\n" + "="*70)
    print_flush("勾配フロー検証 (Gradient Flow Check)")
    print_flush("="*70)

    has_flow = check_result['has_gradient_flow']
    first_norm = check_result['first_context_norm']
    mean_norm = check_result['mean_context_norm']
    ratio = check_result['norm_ratio']
    status = check_result['status']

    print_flush(f"ステータス: {status}")
    print_flush(f"最初のトークン文脈ノルム: {first_norm:.4f}")
    print_flush(f"平均文脈ノルム: {mean_norm:.4f}")
    print_flush(f"ノルム比率 (first/mean): {ratio:.4f}")

    if not has_flow:
        print_flush("\n❌ 重大な問題: トークン間の勾配フローが遮断されています！")
        print_flush("\n【症状】:")
        print_flush(f"  - 最初のトークンの文脈が異常に小さい (ノルム={first_norm:.4f})")
        print_flush(f"  - 他のトークンと比べて {ratio*100:.1f}% しかない")
        print_flush("  - 系列全体での最適化ができない")
        print_flush("  - CVFP学習が機能しない")
        print_flush("\n【原因】:")
        print_flush("  トークン処理ループ内でcontext.detach()を使用している")
        print_flush("\n【修正方法】:")
        print_flush("  → phase1_trainer.py:237 を確認")
        print_flush("  → context_with_noise.detach() → context_with_noise に修正")
        print_flush("  → 勾配がトークン間で伝播するようにする")
        print_flush("\n【期待される結果（修正後）】:")
        print_flush("  - ノルム比率: 0.9-1.1 （ほぼ同じ）")
        print_flush("  - CVFP Convergence Check: PASS")
        print_flush("  - final_diff < 0.001")
        print_flush("="*70 + "\n")
        return True
    else:
        print_flush("\n✅ 正常: トークン間で勾配が伝播しています")
        print_flush(f"  - 最初のトークン文脈: {first_norm:.4f}")
        print_flush(f"  - 平均比率: {ratio:.2f} （正常範囲）")
        print_flush("="*70 + "\n")
        return False


def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)
