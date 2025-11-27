"""
訓練診断ツール (Training Diagnostics)

訓練中の問題を検出するための診断機能:
- 恒等写像検出（Identity Mapping Detection）
"""

import torch
import torch.nn.functional as F


def check_identity_mapping(model, context_dim, device, num_samples=100, threshold=0.95, num_input_tokens=1):
    """
    恒等写像かどうかをチェック

    モデルがコンテキストをほぼそのまま返す（恒等写像）場合、
    トークン情報を学習していない可能性がある。

    Args:
        model: LLMモデル
        context_dim: コンテキストの次元数
        device: デバイス（'cpu' or 'cuda'）
        num_samples: テストするサンプル数
        threshold: 恒等写像と判定するコサイン類似度の閾値（デフォルト: 0.95）
        num_input_tokens: 入力トークン数

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
            # num_input_tokensに対応した結合トークンを生成
            test_token = torch.randn(1, model.embed_dim * num_input_tokens, device=device)

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
    print("\n" + "="*70)
    print("恒等写像チェック (Identity Mapping Check)")
    print("="*70)

    avg_sim = check_result['avg_similarity']
    max_sim = check_result['max_similarity']
    min_sim = check_result['min_similarity']
    above = check_result['samples_above_threshold']
    total = check_result['total_samples']

    print(f"コサイン類似度統計 ({total}サンプル):")
    print(f"  平均: {avg_sim:.4f}")
    print(f"  最大: {max_sim:.4f}")
    print(f"  最小: {min_sim:.4f}")
    print(f"  閾値(0.95)超過: {above}/{total} ({above/total*100:.1f}%)")

    if check_result['is_identity_mapping']:
        print("\n⚠️  警告: 恒等写像が検出されました！")
        print("    モデルが入力コンテキストをほぼそのまま返しています。")
        print("    これは以下を意味する可能性があります:")
        print("      - トークン情報が無視されている")
        print("      - 固定点学習が自明な解に収束している")
        print("      - モデルが有意義な表現を学習していない")
        print("\n    推奨対応:")
        print("      1. モデルの初期化を変更（std を大きく）")
        print("      2. CVFP損失関数を再設計")
        print("      3. より深いネットワーク構造を試す")
        print("\n    Phase 2（トークン予測）はスキップすることを推奨します。")
        print("="*70 + "\n")
        return True
    else:
        print("\n✅ 正常: 恒等写像ではありません")
        print("    モデルがトークン情報を使用してコンテキストを変換しています。")
        print("="*70 + "\n")
        return False
