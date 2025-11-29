#!/usr/bin/env python3
"""
CVFPモデル（E案アーキテクチャ - token継ぎ足し方式）のパラメータ数計算スクリプト

token継ぎ足し方式: 全レイヤーで同じ次元、全レイヤーでtoken入力

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
    CVFPモデルのパラメータ数を計算（token継ぎ足し方式）

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

    # ========== ContextBlock（token継ぎ足し方式） ==========
    # 全レイヤーで同じ次元、全レイヤーでtoken入力
    context_block_total = 0
    for i in range(num_layers):
        # 入力: context_dim + token_input_dim → 出力: context_dim
        layer_input_dim = context_dim + token_input_dim
        layer_output_dim = context_dim

        # FNN: Linear(layer_input_dim, layer_output_dim) + bias
        fnn_params = layer_input_dim * layer_output_dim + layer_output_dim
        # LayerNorm: gamma + beta
        layernorm_params = layer_output_dim * 2
        # 残差射影: なし（次元が同じ）

        context_block_total += fnn_params + layernorm_params

    # ========== TokenBlock（token継ぎ足し方式） ==========
    # 全レイヤーで同じ次元
    token_block_total = 0
    for i in range(num_layers):
        token_in = embed_dim
        token_out = embed_dim
        ctx_dim = context_dim  # E案: ContextBlockの各レイヤー出力を参照

        # FNN: Linear(ctx_dim + token_in, token_out) + bias
        fnn_params = (ctx_dim + token_in) * token_out + token_out
        # LayerNorm: gamma + beta
        layernorm_params = token_out * 2
        # 残差射影: なし（次元が同じ）

        token_block_total += fnn_params + layernorm_params

    # Output Head: Weight Tyingで重み共有（追加パラメータなし）
    output_head = 0

    # Embedding (Weight TyingによりPhase 2で凍結)
    embedding = vocab_size * embed_dim

    # Embedding Norm (LayerNorm)
    embed_norm = embed_dim * 2

    # 合計
    total = embedding + embed_norm + context_block_total + token_block_total + output_head
    trainable_phase1 = context_block_total
    trainable_phase2 = token_block_total  # TokenBlockのみ（Embedding凍結）

    return {
        'num_layers': num_layers,
        'context_dim': context_dim,
        'embed_dim': embed_dim,
        'num_input_tokens': num_input_tokens,
        'vocab_size': vocab_size,
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
    print("CVFPモデル パラメータ数計算（token継ぎ足し方式 + Weight Tying）")
    print("=" * 70)
    print("\n【設定】")
    print(f"  num_layers:       {params['num_layers']}")
    print(f"  context_dim:      {params['context_dim']}")
    print(f"  embed_dim:        {params['embed_dim']}")
    print(f"  num_input_tokens: {params['num_input_tokens']}")
    print(f"  vocab_size:       {params['vocab_size']:,}")

    print("\n【token継ぎ足し方式】")
    print(f"  ContextBlock: 全レイヤー {params['context_dim']}次元（token入力あり）")
    print(f"  TokenBlock:   全レイヤー {params['embed_dim']}次元")

    print("\n【パラメータ内訳】")
    print(f"  Token Embedding:          {params['embedding']:>12,} ({format_number(params['embedding'])}) ← GPT-2事前学習（凍結）")
    print(f"  Embed Norm:               {params['embed_norm']:>12,} ({format_number(params['embed_norm'])})")
    print(f"  ContextBlock ({params['num_layers']}層):      {params['context_block']:>12,} ({format_number(params['context_block'])})")
    print(f"  TokenBlock ({params['num_layers']}層):        {params['token_block']:>12,} ({format_number(params['token_block'])})")
    print(f"  Output Head:              {params['output_head']:>12,} (Weight Tying: 共有)")

    print("\n【合計】")
    print(f"  全体:             {params['total']:>12,} ({format_number(params['total'])})")
    print(f"  Phase 1 学習対象: {params['trainable_phase1']:>12,} ({format_number(params['trainable_phase1'])}) [ContextBlock]")
    print(f"  Phase 2 学習対象: {params['trainable_phase2']:>12,} ({format_number(params['trainable_phase2'])}) [TokenBlockのみ, Embedding凍結]")

    # Chinchilla則
    ultrachat_tokens = 200_000_000

    print("\n【Chinchilla則】")
    print(f"  UltraChatトークン数: {format_number(ultrachat_tokens)}")
    optimal_tokens = params['trainable_phase2'] * 20
    ratio = optimal_tokens / ultrachat_tokens
    print(f"  Phase 2最適トークン数: {format_number(optimal_tokens)} (= {format_number(params['trainable_phase2'])} × 20)")
    print(f"  UltraChat比: {ratio:.2f}x → {'データ十分' if ratio <= 1 else 'データ不足'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
