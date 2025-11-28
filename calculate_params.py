#!/usr/bin/env python3
"""
CVFPモデル（E案アーキテクチャ - 等差減少設計）のパラメータ数計算スクリプト

使用方法:
    python calculate_params.py
"""

from config import ResidualConfig


def calculate_params(
    num_layers: int,
    context_dim: int,
    embed_dim: int,
    num_input_tokens: int,
    vocab_size: int
) -> dict:
    """
    CVFPモデルのパラメータ数を計算（Weight Tying有効、等差減少設計）

    Args:
        num_layers: レイヤー数
        context_dim: コンテキスト次元
        embed_dim: 埋め込み次元
        num_input_tokens: 入力トークン数
        vocab_size: 語彙サイズ

    Returns:
        パラメータ数の詳細辞書
    """
    token_input_dim = embed_dim * num_input_tokens

    # ========== ContextBlock（等差減少設計） ==========
    # 最初のレイヤー入力: context_dim + token_input_dim → 出力: 等差減少
    # 2番目以降: token入力なし、等差減少で context_dim まで縮小
    input_context_dim = context_dim + token_input_dim
    output_context_dim = context_dim
    total_reduction = input_context_dim - output_context_dim

    # 各レイヤーの入出力次元を計算
    context_dims = []
    for i in range(num_layers + 1):
        dim = input_context_dim - (total_reduction * i) // num_layers
        context_dims.append(dim)

    context_block_total = 0
    for i in range(num_layers):
        if i == 0:
            # 最初のレイヤー: context_dim + token_input_dim → context_dims[1]
            layer_input_dim = context_dim + token_input_dim
        else:
            # 2番目以降: context_dims[i] → context_dims[i+1] (token入力なし)
            layer_input_dim = context_dims[i]

        layer_output_dim = context_dims[i + 1]

        # FNN: Linear(layer_input_dim, layer_output_dim) + bias
        fnn_params = layer_input_dim * layer_output_dim + layer_output_dim
        # LayerNorm: gamma + beta
        layernorm_params = layer_output_dim * 2
        # 残差射影（次元が異なる場合のみ）
        if i == 0:
            residual_input = context_dim
        else:
            residual_input = context_dims[i]

        if residual_input != layer_output_dim:
            residual_proj_params = residual_input * layer_output_dim + layer_output_dim
        else:
            residual_proj_params = 0

        context_block_total += fnn_params + layernorm_params + residual_proj_params

    # ========== TokenBlock（等差減少設計） ==========
    # 入力: embed_dim * num_input_tokens → 出力: embed_dim
    input_token_dim = embed_dim * num_input_tokens
    output_token_dim = embed_dim
    token_reduction = input_token_dim - output_token_dim

    token_dims = []
    for i in range(num_layers + 1):
        dim = input_token_dim - (token_reduction * i) // num_layers
        token_dims.append(dim)

    token_block_total = 0
    for i in range(num_layers):
        token_in = token_dims[i]
        token_out = token_dims[i + 1]
        ctx_dim = context_dims[i + 1]  # E案: ContextBlockの各レイヤー出力を参照

        # FNN: Linear(ctx_dim + token_in, token_out) + bias
        fnn_params = (ctx_dim + token_in) * token_out + token_out
        # LayerNorm: gamma + beta
        layernorm_params = token_out * 2
        # 残差射影（次元が異なる場合のみ）
        if token_in != token_out:
            residual_proj_params = token_in * token_out + token_out
        else:
            residual_proj_params = 0

        token_block_total += fnn_params + layernorm_params + residual_proj_params

    # Output Head: Weight Tyingで重み共有（追加パラメータなし）
    output_head = 0

    # Embedding (Weight TyingによりPhase 2で学習)
    embedding = vocab_size * embed_dim

    # Embedding Norm (LayerNorm)
    embed_norm = embed_dim * 2

    # 合計
    total = embedding + embed_norm + context_block_total + token_block_total + output_head
    trainable_phase1 = context_block_total
    trainable_phase2 = token_block_total + embedding  # TokenBlock + Embedding
    trainable_phase2_frozen_embed = token_block_total  # TokenBlockのみ（Embedding凍結時）

    return {
        'num_layers': num_layers,
        'context_dim': context_dim,
        'embed_dim': embed_dim,
        'num_input_tokens': num_input_tokens,
        'vocab_size': vocab_size,
        'context_dims': context_dims,
        'token_dims': token_dims,
        'embedding': embedding,
        'embed_norm': embed_norm,
        'context_block': context_block_total,
        'token_block': token_block_total,
        'output_head': output_head,
        'total': total,
        'trainable_phase1': trainable_phase1,
        'trainable_phase2': trainable_phase2,
        'trainable_phase2_frozen_embed': trainable_phase2_frozen_embed,
    }


def format_number(n: int) -> str:
    """数値を読みやすい形式に変換"""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(n)


def main():
    # config.pyから設定を取得
    config = ResidualConfig()
    params = calculate_params(
        num_layers=config.num_layers,
        context_dim=config.context_dim,
        embed_dim=config.embed_dim,
        num_input_tokens=config.num_input_tokens,
        vocab_size=config.vocab_size
    )

    print("\n" + "=" * 70)
    print("CVFPモデル パラメータ数計算（Weight Tying有効、等差減少設計）")
    print("=" * 70)
    print(f"\n【設定】")
    print(f"  num_layers:       {params['num_layers']}")
    print(f"  context_dim:      {params['context_dim']}")
    print(f"  embed_dim:        {params['embed_dim']}")
    print(f"  num_input_tokens: {params['num_input_tokens']}")
    print(f"  vocab_size:       {params['vocab_size']:,}")

    print(f"\n【等差減少設計】")
    print(f"  ContextBlock次元: {' → '.join(map(str, params['context_dims']))}")
    print(f"  TokenBlock次元:   {' → '.join(map(str, params['token_dims']))}")

    print(f"\n【パラメータ内訳】")
    print(f"  Token Embedding:          {params['embedding']:>12,} ({format_number(params['embedding'])}) ← GPT-2事前学習（凍結）")
    print(f"  Embed Norm:               {params['embed_norm']:>12,} ({format_number(params['embed_norm'])})")
    print(f"  ContextBlock ({params['num_layers']}層):      {params['context_block']:>12,} ({format_number(params['context_block'])})")
    print(f"  TokenBlock ({params['num_layers']}層):        {params['token_block']:>12,} ({format_number(params['token_block'])})")
    print(f"  Output Head:              {params['output_head']:>12,} (Weight Tying: 共有)")

    print(f"\n【合計】")
    print(f"  全体:             {params['total']:>12,} ({format_number(params['total'])})")
    print(f"  Phase 1 学習対象: {params['trainable_phase1']:>12,} ({format_number(params['trainable_phase1'])})")
    print(f"  Phase 2 学習対象:")
    print(f"    Embedding凍結時:  {params['trainable_phase2_frozen_embed']:>10,} ({format_number(params['trainable_phase2_frozen_embed'])}) [推奨]")
    print(f"    Embedding学習時:  {params['trainable_phase2']:>10,} ({format_number(params['trainable_phase2'])}) [非推奨]")

    # Chinchilla則
    ultrachat_tokens = 200_000_000

    print(f"\n【Chinchilla則】")
    print(f"  UltraChatトークン数: {format_number(ultrachat_tokens)}")

    # Embedding凍結時（推奨）
    optimal_tokens_frozen = params['trainable_phase2_frozen_embed'] * 20
    ratio_frozen = optimal_tokens_frozen / ultrachat_tokens
    print(f"\n  ◆ Embedding凍結時 [推奨]:")
    print(f"    最適トークン数: {format_number(optimal_tokens_frozen)} (= {format_number(params['trainable_phase2_frozen_embed'])} × 20)")
    print(f"    UltraChat比:    {ratio_frozen:.1f}x → {'データ不足' if ratio_frozen > 1 else '適切'}")

    # Embedding学習時（非推奨）
    optimal_tokens = params['trainable_phase2'] * 20
    ratio = optimal_tokens / ultrachat_tokens
    print(f"\n  ◆ Embedding学習時 [非推奨]:")
    print(f"    最適トークン数: {format_number(optimal_tokens)} (= {format_number(params['trainable_phase2'])} × 20)")
    print(f"    UltraChat比:    {ratio:.1f}x → データ不足")
    print("=" * 70)


if __name__ == '__main__':
    main()
