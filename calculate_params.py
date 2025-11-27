#!/usr/bin/env python3
"""
CVFPモデル（E案アーキテクチャ）のパラメータ数計算スクリプト

使用方法:
    python calculate_params.py
"""


def calculate_params(
    num_layers: int = 6,
    context_dim: int = 768,
    embed_dim: int = 768,
    num_input_tokens: int = 1,
    vocab_size: int = 50257
) -> dict:
    """
    CVFPモデルのパラメータ数を計算（Weight Tying有効）

    Args:
        num_layers: レイヤー数
        context_dim: コンテキスト次元
        embed_dim: 埋め込み次元（GPT-2: 768）
        num_input_tokens: 入力トークン数
        vocab_size: 語彙サイズ（GPT-2: 50257）

    Returns:
        パラメータ数の詳細辞書
    """
    # FNN入力次元
    input_dim = context_dim + embed_dim * num_input_tokens

    # ContextBlock (1層あたり)
    context_linear = input_dim * context_dim + context_dim
    context_layernorm = context_dim * 2
    context_layer_params = context_linear + context_layernorm
    context_block_total = context_layer_params * num_layers

    # TokenBlock (1層あたり)
    token_linear = input_dim * embed_dim + embed_dim
    token_layernorm = embed_dim * 2
    token_layer_params = token_linear + token_layernorm
    token_block_total = token_layer_params * num_layers

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

    return {
        'num_layers': num_layers,
        'context_dim': context_dim,
        'embed_dim': embed_dim,
        'num_input_tokens': num_input_tokens,
        'vocab_size': vocab_size,
        'input_dim': input_dim,
        'embedding': embedding,
        'embed_norm': embed_norm,
        'context_block': context_block_total,
        'token_block': token_block_total,
        'output_head': output_head,
        'total': total,
        'trainable_phase1': trainable_phase1,
        'trainable_phase2': trainable_phase2,
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
    # デフォルト設定（6層, Weight Tying有効）
    params = calculate_params()

    print("\n" + "=" * 60)
    print("CVFPモデル パラメータ数計算（Weight Tying有効）")
    print("=" * 60)
    print(f"\n【設定】")
    print(f"  num_layers:       {params['num_layers']}")
    print(f"  context_dim:      {params['context_dim']}")
    print(f"  embed_dim:        {params['embed_dim']}")
    print(f"  num_input_tokens: {params['num_input_tokens']}")
    print(f"  vocab_size:       {params['vocab_size']:,}")
    print(f"  FNN input_dim:    {params['input_dim']}")

    print(f"\n【パラメータ内訳】")
    print(f"  Token Embedding:          {params['embedding']:>12,} ({format_number(params['embedding'])}) ← Phase 2で学習")
    print(f"  Embed Norm:               {params['embed_norm']:>12,} ({format_number(params['embed_norm'])})")
    print(f"  ContextBlock ({params['num_layers']}層):      {params['context_block']:>12,} ({format_number(params['context_block'])})")
    print(f"  TokenBlock ({params['num_layers']}層):        {params['token_block']:>12,} ({format_number(params['token_block'])})")
    print(f"  Output Head:              {params['output_head']:>12,} (Weight Tying: 共有)")

    print(f"\n【合計】")
    print(f"  全体:             {params['total']:>12,} ({format_number(params['total'])})")
    print(f"  Phase 1 学習対象: {params['trainable_phase1']:>12,} ({format_number(params['trainable_phase1'])})")
    print(f"  Phase 2 学習対象: {params['trainable_phase2']:>12,} ({format_number(params['trainable_phase2'])})")
    print(f"    (TokenBlock + Embedding)")

    # Chinchilla則
    optimal_tokens = params['trainable_phase2'] * 20
    ultrachat_tokens = 200_000_000
    ratio = optimal_tokens / ultrachat_tokens

    print(f"\n【Chinchilla則】")
    print(f"  最適トークン数:   {format_number(optimal_tokens)} (= Phase2パラメータ × 20)")
    print(f"  UltraChat比:      {ratio:.1f}x ({format_number(ultrachat_tokens)})")
    print("=" * 60)


if __name__ == '__main__':
    main()
