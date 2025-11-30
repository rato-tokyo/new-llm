#!/usr/bin/env python3
"""
CVFPモデルのパラメータ数計算スクリプト

使用方法:
    python calculate_params.py
"""

import sys
sys.path.insert(0, '.')

from config import ResidualConfig
from src.models.llm import LLM


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
    config = ResidualConfig()

    # モデルを作成してパラメータ数を取得
    print("Creating model...")
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        num_input_tokens=config.num_input_tokens,
        use_pretrained_embeddings=False,  # 計算用なのでGPT-2ロード不要
        use_weight_tying=config.use_weight_tying,
        config=config
    )

    params = model.num_params()

    print("\n" + "=" * 70)
    print("CVFPモデル パラメータ数計算（token継ぎ足し方式 + Weight Tying）")
    print("=" * 70)
    print("\n【設定】")
    print(f"  num_layers:       {config.num_layers}")
    print(f"  context_dim:      {config.context_dim}")
    print(f"  embed_dim:        {config.embed_dim}")
    print(f"  num_input_tokens: {config.num_input_tokens}")
    print(f"  vocab_size:       {config.vocab_size:,}")

    print("\n【token継ぎ足し方式】")
    print(f"  ContextBlock: 全レイヤー {config.context_dim}次元（token入力あり）")
    print(f"  TokenBlock:   全レイヤー {config.embed_dim}次元")

    print("\n【パラメータ内訳】")
    print(f"  Token Embedding:          {params['embedding']:>12,} ({format_number(params['embedding'])}) ← GPT-2事前学習（凍結）")
    print(f"  Embed Norm:               {params['embed_norm']:>12,} ({format_number(params['embed_norm'])})")
    print(f"  ContextBlock ({config.num_layers}層):      {params['context_block']:>12,} ({format_number(params['context_block'])})")
    print(f"  TokenBlock ({config.num_layers}層):        {params['token_block']:>12,} ({format_number(params['token_block'])})")
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
