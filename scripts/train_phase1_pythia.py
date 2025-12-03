#!/usr/bin/env python3
"""
Phase 1 Training for Context-Pythia

ContextBlockのOACD学習を行う。
Pythia-70Mのtoken embeddingsを使用。

Usage:
    python3 scripts/train_phase1_pythia.py --tokens 100000
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, ".")

from config import Phase1Config, ContextPythiaConfig
from src.models.context_pythia import ContextPythiaModel
from src.utils.data_pythia import prepare_pythia_phase1_data
from src.utils.io import print_flush
from src.losses.diversity import oacd_loss


def compute_convergence_rate(
    current: torch.Tensor,
    previous: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> float:
    """
    収束率を計算（バッチ処理でメモリ効率化）

    Args:
        current: 現在のコンテキスト [num_tokens, context_dim]
        previous: 前回のコンテキスト [num_tokens, context_dim]
        threshold: 収束判定の閾値
        device: デバイス

    Returns:
        収束率（0.0-1.0）
    """
    num_tokens = len(current)
    batch_size = 100000

    with torch.no_grad():
        converged_count = 0
        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)

            current_batch = current[start_idx:end_idx].to(device)
            previous_batch = previous[start_idx:end_idx].to(device)

            token_losses = ((current_batch - previous_batch) ** 2).mean(dim=1)
            converged_count += (token_losses < threshold).sum().item()

            del current_batch, previous_batch

        return converged_count / num_tokens


def train_iteration(
    model: ContextPythiaModel,
    token_embeds: torch.Tensor,
    previous_contexts: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    context_noise: float,
    gradient_clip: float,
) -> tuple[torch.Tensor, float]:
    """
    1イテレーションの学習

    Args:
        model: モデル
        token_embeds: [num_tokens, embed_dim] (CPU tensor)
        previous_contexts: [num_tokens, context_dim] (CPU tensor)
        optimizer: オプティマイザ
        device: デバイス
        batch_size: バッチサイズ
        context_noise: ガウシアンノイズの標準偏差
        gradient_clip: 勾配クリッピング値

    Returns:
        new_contexts: [num_tokens, context_dim] (CPU tensor)
        avg_loss: 平均損失
    """
    num_tokens = len(token_embeds)
    num_batches = (num_tokens + batch_size - 1) // batch_size

    optimizer.zero_grad()

    # shifted_prev_contextを作成
    init_ctx = torch.zeros(1, model.context_dim, device='cpu')
    shifted_prev_context = torch.cat([init_ctx, previous_contexts[:-1]], dim=0)

    all_contexts_cpu = []
    total_loss_sum = 0.0

    for start_idx in range(0, num_tokens, batch_size):
        end_idx = min(start_idx + batch_size, num_tokens)
        current_batch_size = end_idx - start_idx

        # バッチ分だけGPUに転送
        batch_contexts = shifted_prev_context[start_idx:end_idx].to(device)
        batch_embeds = token_embeds[start_idx:end_idx].to(device)

        # ノイズ追加（汎化性能向上）
        if context_noise > 0 and model.training:
            noise = torch.randn_like(batch_contexts) * context_noise
            batch_contexts = batch_contexts + noise

        # Forward pass
        batch_output = model.context_block(batch_contexts, batch_embeds)

        # OACD損失
        loss = oacd_loss(batch_output, centroid_weight=0.1)

        # 勾配累積
        scaled_loss = loss / num_batches
        if not torch.isnan(scaled_loss) and not torch.isinf(scaled_loss):
            scaled_loss.backward()

        total_loss_sum += loss.item() * current_batch_size

        # 結果をCPUに保存
        all_contexts_cpu.append(batch_output.detach().cpu())

        del batch_contexts, batch_embeds, batch_output

    # 勾配クリッピングとパラメータ更新
    torch.nn.utils.clip_grad_norm_(model.context_block.parameters(), gradient_clip)
    optimizer.step()

    # GPUキャッシュクリア
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # コンテキストを結合
    new_contexts = torch.cat(all_contexts_cpu, dim=0)

    return new_contexts, total_loss_sum / num_tokens


def train_phase1(
    model: ContextPythiaModel,
    train_token_embeds: torch.Tensor,
    val_token_embeds: torch.Tensor,
    device: torch.device,
    phase1_config: Phase1Config,
) -> dict:
    """
    Phase 1学習のメインループ

    Args:
        model: Context-Pythiaモデル
        train_token_embeds: 訓練用トークン埋め込み [num_train_tokens, embed_dim] (CPU)
        val_token_embeds: 検証用トークン埋め込み [num_val_tokens, embed_dim] (CPU)
        device: デバイス
        phase1_config: Phase 1設定

    Returns:
        stats: 学習統計
    """
    model.train()
    num_train_tokens = len(train_token_embeds)
    num_val_tokens = len(val_token_embeds)

    print_flush(f"\nPhase 1 Training:")
    print_flush(f"  Train tokens: {num_train_tokens:,}")
    print_flush(f"  Val tokens: {num_val_tokens:,}")
    print_flush(f"  Max iterations: {phase1_config.max_iterations}")
    print_flush(f"  Learning rate: {phase1_config.learning_rate}")
    print_flush(f"  Early stopping rate: {int(phase1_config.early_stopping_threshold * 100)}%")

    optimizer = torch.optim.Adam(
        model.context_block.parameters(),
        lr=phase1_config.learning_rate,
    )

    previous_contexts = None
    val_previous_contexts = None
    final_loss = 0.0
    final_conv_rate = 0.0

    stats = {
        'iterations': 0,
        'early_stopped': False,
        'stop_reason': 'max_iterations',
        'best_val_loss': float('inf'),
        'final_conv_rate': 0.0,
    }

    for iteration in range(phase1_config.max_iterations):
        iter_start = time.time()

        if iteration == 0:
            # Iteration 0: ランダム初期化
            previous_contexts = torch.randn(num_train_tokens, model.context_dim) * 0.01
            val_previous_contexts = torch.randn(num_val_tokens, model.context_dim) * 0.01
            print_flush(f"  Iter  1: random init")
            continue

        # 学習イテレーション
        assert previous_contexts is not None
        current_contexts, train_loss = train_iteration(
            model, train_token_embeds, previous_contexts, optimizer, device,
            batch_size=phase1_config.batch_size,
            context_noise=phase1_config.context_noise,
            gradient_clip=phase1_config.gradient_clip,
        )
        final_loss = train_loss

        # 収束率計算
        conv_rate = compute_convergence_rate(
            current_contexts, previous_contexts,
            phase1_config.convergence_threshold, device,
        )
        final_conv_rate = conv_rate

        iter_time = time.time() - iter_start
        print_flush(f"  Iter {iteration+1:2d}: loss={train_loss:.4f}, conv={int(conv_rate*100)}% [{iter_time:.1f}s]")

        previous_contexts = current_contexts

        # Early stopping
        if phase1_config.early_stopping and conv_rate >= phase1_config.early_stopping_threshold:
            stats['early_stopped'] = True
            stats['stop_reason'] = 'early_stopping'
            print_flush(f"  → Early stop: conv {int(conv_rate*100)}% >= {int(phase1_config.early_stopping_threshold*100)}%")
            break

        stats['iterations'] = iteration + 1
        stats['final_conv_rate'] = final_conv_rate

    print_flush(f"\nPhase 1 completed:")
    print_flush(f"  Iterations: {stats['iterations']}")
    print_flush(f"  Final loss: {final_loss:.4f}")
    print_flush(f"  Final conv rate: {int(final_conv_rate*100)}%")
    print_flush(f"  Stop reason: {stats['stop_reason']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Training for Context-Pythia")
    parser.add_argument("--tokens", type=int, required=True, help="Number of tokens (REQUIRED)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    # Configs
    pythia_config = ContextPythiaConfig()
    phase1_config = Phase1Config()

    print_flush("=" * 70)
    print_flush("PHASE 1: CONTEXTBLOCK OACD TRAINING (PYTHIA)")
    print_flush("=" * 70)
    print_flush(f"Tokens: {args.tokens:,}")
    print_flush(f"Context dim: {pythia_config.context_dim}")
    print_flush(f"Checkpoint: {pythia_config.phase1_checkpoint_path}")
    print_flush("=" * 70)

    # Data
    train_ids, val_ids = prepare_pythia_phase1_data(
        num_tokens=args.tokens,
        val_split=args.val_split,
        tokenizer_name=pythia_config.tokenizer_name,
        device=device,
    )

    # Model
    print_flush("\n[Model] Creating Context-Pythia...")
    model = ContextPythiaModel(
        vocab_size=pythia_config.vocab_size,
        hidden_size=pythia_config.hidden_size,
        context_dim=pythia_config.context_dim,
        num_layers=pythia_config.num_layers,
        num_heads=pythia_config.num_attention_heads,
        intermediate_size=pythia_config.intermediate_size,
        max_position_embeddings=pythia_config.max_position_embeddings,
        rotary_pct=pythia_config.rotary_pct,
    ).to(device)

    context_params = sum(p.numel() for p in model.context_block.parameters())
    print_flush(f"ContextBlock parameters: {context_params:,}")

    # Compute token embeddings (CPUに保存)
    # ⚠️ 重要: embed_norm による正規化が必須（Phase 1収束に必要）
    print_flush("\n[Embeddings] Computing token embeddings...")
    with torch.no_grad():
        # Train
        train_embeds_gpu = model.embed_in(train_ids)
        train_embeds_gpu = model.embed_norm(train_embeds_gpu)  # ⚠️ 正規化必須
        train_token_embeds = train_embeds_gpu.cpu()
        del train_embeds_gpu
        # Val
        val_embeds_gpu = model.embed_in(val_ids)
        val_embeds_gpu = model.embed_norm(val_embeds_gpu)  # ⚠️ 正規化必須
        val_token_embeds = val_embeds_gpu.cpu()
        del val_embeds_gpu

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print_flush(f"  Train embeddings: {train_token_embeds.shape}")
    print_flush(f"  Val embeddings: {val_token_embeds.shape}")

    # Train
    start_time = time.time()
    stats = train_phase1(
        model, train_token_embeds, val_token_embeds, device, phase1_config,
    )
    total_time = time.time() - start_time
    print_flush(f"  Total time: {total_time:.1f}s")

    # Save checkpoint
    checkpoint_path = Path(pythia_config.phase1_checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "context_block_state_dict": model.context_block.state_dict(),
        "config": {
            "context_dim": pythia_config.context_dim,
            "hidden_size": pythia_config.hidden_size,
            "num_tokens": args.tokens,
        },
        "stats": stats,
        "final_loss": stats.get('final_conv_rate', 0.0),
    }
    torch.save(checkpoint, checkpoint_path)
    print_flush(f"\nCheckpoint saved: {checkpoint_path}")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
