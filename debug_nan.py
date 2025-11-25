"""
NaN問題のデバッグスクリプト
各ステップで値を確認
"""

import torch
import torch.nn.functional as F
from src.models.llm import LLM
from src.data.loader import load_data_source
import config as cfg


def debug_nan_issue():
    """NaN問題をデバッグ"""
    print("=" * 80)
    print("NaN問題のデバッグ")
    print("=" * 80)

    device = torch.device("cpu")

    # データ読み込み（100トークンのみで高速検証）
    class TempConfig:
        train_data_source = "text_file"
        train_text_file = "./data/small_train.txt"
        num_samples = None
        max_seq_length = None
        cache_dir = "./cache"

    temp_config = TempConfig()
    train_token_ids = load_data_source(
        source_type="text_file",
        config=temp_config,
        is_training=True
    )

    # 最初の100トークンのみ使用
    train_token_ids = train_token_ids[:100]
    print(f"\nデバッグデータ: {len(train_token_ids)} トークン")

    # モデル初期化
    torch.manual_seed(42)
    model = LLM(
        vocab_size=cfg.ResidualConfig.vocab_size,
        embed_dim=cfg.ResidualConfig.embed_dim,
        context_dim=cfg.ResidualConfig.context_dim,
        hidden_dim=cfg.ResidualConfig.hidden_dim,
        layer_structure=[1] * cfg.ResidualConfig.num_layers,
        layernorm_mix=0.5,  # LayerNormを有効化（勾配爆発防止）
        use_pretrained_embeddings=cfg.ResidualConfig.use_pretrained_embeddings
    ).to(device)

    model.train()

    # Optimizer設定
    context_params = [
        p for name, p in model.named_parameters()
        if 'token_output' not in name and 'token_embedding' not in name
    ]
    optimizer = torch.optim.Adam(context_params, lr=0.002)

    # トークン埋め込み
    with torch.no_grad():
        token_embeds = model.token_embedding(train_token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)
        print(f"\nToken embeds shape: {token_embeds.shape}")
        print(f"Token embeds mean: {token_embeds.mean().item():.6f}")
        print(f"Token embeds std: {token_embeds.std().item():.6f}")
        print(f"Token embeds has NaN: {torch.isnan(token_embeds).any().item()}")
        print(f"Token embeds has Inf: {torch.isinf(token_embeds).any().item()}")

    # Iteration 0: 順伝搬のみ
    print("\n" + "=" * 80)
    print("Iteration 0: 順伝搬のみ（固定点目標確立）")
    print("=" * 80)

    context = torch.zeros(1, model.context_dim, device=device)
    context_list = []

    for t, token_embed in enumerate(token_embeds[:10]):  # 最初の10トークンのみチェック
        print(f"\n  Token {t}:")
        print(f"    Input context mean: {context.mean().item():.6f}, std: {context.std().item():.6f}")
        print(f"    Input token mean: {token_embed.mean().item():.6f}, std: {token_embed.std().item():.6f}")

        for block_idx, block in enumerate(model.blocks):
            context, token_embed_out = block(token_embed.unsqueeze(0), context)

            print(f"    After block {block_idx}: context mean={context.mean().item():.6f}, std={context.std().item():.6f}")

            if torch.isnan(context).any():
                print(f"    ⚠️ NaN detected in context after block {block_idx}!")
                return
            if torch.isinf(context).any():
                print(f"    ⚠️ Inf detected in context after block {block_idx}!")
                return

        context_list.append(context.squeeze(0))

    # 全トークン処理
    print("\n全100トークンを処理中...")
    context = torch.zeros(1, model.context_dim, device=device)
    context_list = []

    for t, token_embed in enumerate(token_embeds):
        for block in model.blocks:
            context, token_embed_out = block(token_embed.unsqueeze(0), context)
        context_list.append(context.squeeze(0))

    contexts_0 = torch.stack(context_list)
    print(f"\nIteration 0 完了:")
    print(f"  Contexts shape: {contexts_0.shape}")
    print(f"  Contexts mean: {contexts_0.mean().item():.6f}")
    print(f"  Contexts std: {contexts_0.std().item():.6f}")
    print(f"  Contexts has NaN: {torch.isnan(contexts_0).any().item()}")
    print(f"  Contexts has Inf: {torch.isinf(contexts_0).any().item()}")

    target_contexts = contexts_0.detach().clone()
    previous_contexts = contexts_0.detach()

    # Iteration 1: 最初の学習
    print("\n" + "=" * 80)
    print("Iteration 1: 最初の学習（NaN発生ポイント）")
    print("=" * 80)

    # 順伝搬
    context = previous_contexts[-1].unsqueeze(0).detach()
    context_list = []

    for t, token_embed in enumerate(token_embeds):
        for block in model.blocks:
            context, token_embed_out = block(token_embed.unsqueeze(0), context)
        context_list.append(context.squeeze(0))

    contexts_1 = torch.stack(context_list)
    print(f"\n順伝搬完了:")
    print(f"  Contexts shape: {contexts_1.shape}")
    print(f"  Contexts mean: {contexts_1.mean().item():.6f}")
    print(f"  Contexts std: {contexts_1.std().item():.6f}")
    print(f"  Contexts has NaN: {torch.isnan(contexts_1).any().item()}")
    print(f"  Contexts has Inf: {torch.isinf(contexts_1).any().item()}")

    # CVFP損失
    cvfp_loss = F.mse_loss(contexts_1, target_contexts)
    print(f"\nCVFP Loss: {cvfp_loss.item():.6f}")
    print(f"CVFP Loss is NaN: {torch.isnan(cvfp_loss).item()}")

    # 多様性損失
    context_mean = contexts_1.mean(dim=0)
    deviation = contexts_1 - context_mean
    diversity_loss = -torch.norm(deviation, p=2) / len(contexts_1)
    print(f"\nDiversity Loss: {diversity_loss.item():.6f}")
    print(f"Diversity Loss is NaN: {torch.isnan(diversity_loss).item()}")

    # 合計損失
    dist_reg_weight = 0.5
    total_loss = (1 - dist_reg_weight) * cvfp_loss + dist_reg_weight * diversity_loss
    print(f"\nTotal Loss: {total_loss.item():.6f}")
    print(f"Total Loss is NaN: {torch.isnan(total_loss).item()}")

    if torch.isnan(total_loss):
        print("\n⚠️ Total Loss is NaN!")
        print("原因を特定中...")

        if torch.isnan(cvfp_loss):
            print("  - CVFP Loss がNaN")
            print(f"    contexts_1 has NaN: {torch.isnan(contexts_1).any().item()}")
            print(f"    target_contexts has NaN: {torch.isnan(target_contexts).any().item()}")

        if torch.isnan(diversity_loss):
            print("  - Diversity Loss がNaN")
            print(f"    context_mean has NaN: {torch.isnan(context_mean).any().item()}")
            print(f"    deviation has NaN: {torch.isnan(deviation).any().item()}")
            print(f"    norm value: {torch.norm(deviation, p=2).item():.6f}")

        return

    # 逆伝搬
    print("\n逆伝搬実行中...")
    optimizer.zero_grad()
    total_loss.backward()

    # 勾配チェック
    print("\n勾配チェック:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()

            print(f"  {name}:")
            print(f"    Grad mean: {grad_mean:.6f}, std: {grad_std:.6f}, max: {grad_max:.6f}")
            print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")

            if has_nan or has_inf:
                print(f"    ⚠️ 問題発見: {name} の勾配に異常値")
                return

    # 勾配クリッピング
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # パラメータ更新
    optimizer.step()

    print("\n✅ Iteration 1 完了（NaN発生なし）")


if __name__ == "__main__":
    debug_nan_issue()
