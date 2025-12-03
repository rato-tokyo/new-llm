#!/usr/bin/env python3
"""
Phase 1: ContextBlock OACD Training

ContextBlockの多様性学習を行い、チェックポイントを保存する。
このスクリプトで学習したパラメータは、experiment_pythia_comparison.pyで使用される。

機能（削除禁止）:
- 収束率表示: 各イテレーションでconv=XX%を表示
- Early Stopping: 収束率が閾値以上で停止
- Validation: 検証データでの評価
- max iteration: 最大イテレーション数

Usage:
    python3 scripts/train_phase1.py --tokens 100000
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

# Add project root to path
sys.path.insert(0, ".")

from config import Phase1Config, ContextPythiaConfig
from src.models.context_pythia import ContextPythiaModel
from src.losses.diversity import oacd_loss
from src.utils.io import print_flush
from src.utils.data import prepare_phase1_data


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_convergence_rate_batched(
    current_contexts: torch.Tensor,
    previous_contexts: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> float:
    """
    収束率を計算（バッチ処理でメモリ効率化）

    Args:
        current_contexts: 現在のcontext [N, context_dim] (CPU tensor)
        previous_contexts: 前回のcontext [N, context_dim] (CPU tensor)
        threshold: 収束判定の閾値
        device: 計算用デバイス

    Returns:
        収束率 (0.0-1.0)
    """
    with torch.no_grad():
        num_tokens = len(current_contexts)
        batch_size = 100000  # 10万トークンずつ処理
        converged_count = 0

        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)

            # バッチ分だけGPUに転送して計算
            current_batch = current_contexts[start_idx:end_idx].to(device)
            previous_batch = previous_contexts[start_idx:end_idx].to(device)

            # 各トークンの変化量
            token_losses = ((current_batch - previous_batch) ** 2).mean(dim=1)
            converged_count += (token_losses < threshold).sum().item()

            del current_batch, previous_batch, token_losses

        return converged_count / num_tokens


def forward_all_tokens(
    model: ContextPythiaModel,
    token_embeds: torch.Tensor,
    previous_contexts: torch.Tensor,
    device: torch.device,
    batch_size: int = 10000,
    compute_loss: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    全トークンに対してContextBlockのforward passを実行

    Args:
        model: ContextPythiaModel
        token_embeds: [num_tokens, embed_dim] (CPU tensor)
        previous_contexts: [num_tokens, context_dim] (CPU tensor)
        device: デバイス
        batch_size: バッチサイズ
        compute_loss: 損失を計算するか

    Returns:
        new_contexts: [num_tokens, context_dim] (CPU tensor)
        total_loss: 平均損失
    """
    num_tokens = len(token_embeds)

    # shifted_prev_contextを作成（ゼロベクトルから開始）
    # token i の処理には previous_contexts[i-1] を使用
    init_ctx = torch.zeros(1, model.context_block.context_dim, device='cpu')
    shifted_prev_context = torch.cat([init_ctx, previous_contexts[:-1]], dim=0)

    # 結果を格納（CPUに保存してGPUメモリ節約）
    all_contexts_cpu = []
    total_loss_sum = 0.0

    for start_idx in range(0, num_tokens, batch_size):
        end_idx = min(start_idx + batch_size, num_tokens)
        current_batch_size = end_idx - start_idx

        # バッチ分だけGPUに転送
        batch_contexts = shifted_prev_context[start_idx:end_idx].to(device)
        batch_embeds = token_embeds[start_idx:end_idx].to(device)

        # Forward pass
        with torch.set_grad_enabled(compute_loss):
            batch_output = model.context_block(batch_contexts, batch_embeds)

            if compute_loss:
                # OACD損失
                loss = oacd_loss(batch_output, centroid_weight=0.1)
                total_loss_sum += loss.item() * current_batch_size

        # 結果をCPUに保存
        all_contexts_cpu.append(batch_output.detach().cpu())

        del batch_contexts, batch_embeds, batch_output

    # コンテキストを結合
    new_contexts = torch.cat(all_contexts_cpu, dim=0)
    avg_loss = total_loss_sum / num_tokens if compute_loss else 0.0

    return new_contexts, avg_loss


def train_iteration(
    model: ContextPythiaModel,
    token_embeds: torch.Tensor,
    previous_contexts: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    num_batches_for_grad: int,
    context_noise: float = 0.0,
    gradient_clip: float = 2.0,
) -> Tuple[torch.Tensor, float]:
    """
    1イテレーションの学習

    Args:
        model: モデル
        token_embeds: [num_tokens, embed_dim] (CPU tensor)
        previous_contexts: [num_tokens, context_dim] (CPU tensor)
        optimizer: オプティマイザ
        device: デバイス
        batch_size: バッチサイズ
        num_batches_for_grad: 勾配累積するバッチ数
        context_noise: ガウシアンノイズの標準偏差
        gradient_clip: 勾配クリッピング値

    Returns:
        new_contexts: [num_tokens, context_dim] (CPU tensor)
        avg_loss: 平均損失
    """
    num_tokens = len(token_embeds)

    # shifted_prev_contextを作成
    init_ctx = torch.zeros(1, model.context_block.context_dim, device='cpu')
    shifted_prev_context = torch.cat([init_ctx, previous_contexts[:-1]], dim=0)

    # 勾配をゼロに
    optimizer.zero_grad()

    # 結果を格納
    all_contexts_cpu = []
    total_loss_sum = 0.0
    num_batches = (num_tokens + batch_size - 1) // batch_size

    for batch_idx, start_idx in enumerate(range(0, num_tokens, batch_size)):
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

        # 勾配累積（全バッチで累積）
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
    avg_loss = total_loss_sum / num_tokens

    return new_contexts, avg_loss


def evaluate_diversity(
    model: ContextPythiaModel,
    token_embeds: torch.Tensor,
    previous_contexts: torch.Tensor,
    device: torch.device,
) -> float:
    """
    検証データでの多様性損失を計算

    Returns:
        平均OACD損失
    """
    model.eval()
    _, val_loss = forward_all_tokens(
        model, token_embeds, previous_contexts, device,
        compute_loss=True,
    )
    model.train()
    return val_loss


def train_phase1(
    model: ContextPythiaModel,
    train_token_embeds: torch.Tensor,
    val_token_embeds: torch.Tensor,
    phase1_config: Phase1Config,
    device: torch.device,
) -> Tuple[float, dict]:
    """
    Phase 1: Train ContextBlock with OACD diversity loss.

    昔の実装方式：全トークンに対してcontextを計算し、前回と比較

    Returns:
        final_loss: 最終損失
        stats: 訓練統計
    """
    model.train()

    # Only optimize context_block
    optimizer = torch.optim.AdamW(
        model.context_block.parameters(),
        lr=phase1_config.learning_rate,
    )

    num_train_tokens = len(train_token_embeds)
    num_val_tokens = len(val_token_embeds)

    print_flush(f"\nPhase 1 Training:")
    print_flush(f"  Train tokens: {num_train_tokens:,}")
    print_flush(f"  Val tokens: {num_val_tokens:,}")
    print_flush(f"  Max iterations: {phase1_config.max_iterations}")
    print_flush(f"  Learning rate: {phase1_config.learning_rate}")
    print_flush(f"  Early stopping rate: {phase1_config.early_stopping_rate * 100:.0f}%")

    start_time = time.time()

    # 前回のcontextを保存（収束率計算用）
    previous_contexts: Optional[torch.Tensor] = None
    val_previous_contexts: Optional[torch.Tensor] = None
    best_val_loss = float('inf')
    final_loss = 0.0
    final_conv_rate = 0.0

    stats = {
        'iterations': 0,
        'early_stopped': False,
        'stop_reason': 'max_iterations',
        'best_val_loss': float('inf'),
        'final_conv_rate': 0.0,
    }

    batch_size = phase1_config.batch_size

    for iteration in range(phase1_config.max_iterations):
        iter_start = time.time()

        if iteration == 0:
            # Iteration 0: ランダム初期化
            previous_contexts = torch.randn(num_train_tokens, model.context_block.context_dim) * 0.01
            val_previous_contexts = torch.randn(num_val_tokens, model.context_block.context_dim) * 0.01
            print_flush("  Iter  1: random init")
            continue

        # 学習イテレーション（全トークンに対して）
        assert previous_contexts is not None
        current_contexts, train_loss = train_iteration(
            model, train_token_embeds, previous_contexts, optimizer, device,
            batch_size=batch_size,
            num_batches_for_grad=phase1_config.batches_per_iteration,
            context_noise=phase1_config.context_noise,
            gradient_clip=phase1_config.gradient_clip,
        )
        final_loss = train_loss

        # 収束率計算（全トークンに対して）
        conv_rate = compute_convergence_rate_batched(
            current_contexts, previous_contexts,
            phase1_config.convergence_threshold, device,
        )
        final_conv_rate = conv_rate

        # Validation評価
        assert val_previous_contexts is not None
        val_contexts, val_loss = forward_all_tokens(
            model, val_token_embeds, val_previous_contexts, device,
            compute_loss=True,
        )

        # ログ出力
        elapsed = time.time() - iter_start
        print_flush(
            f"  Iter {iteration + 1:2d}: loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, conv={conv_rate*100:.0f}% [{elapsed:.1f}s]"
        )

        # 改善チェック
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Early Stopping判定（収束率ベース）
        if conv_rate >= phase1_config.early_stopping_rate:
            stats['early_stopped'] = True
            stats['stop_reason'] = f'convergence_rate >= {phase1_config.early_stopping_rate*100:.0f}%'
            print_flush(f"  → Early stop: conv {conv_rate*100:.0f}% >= {phase1_config.early_stopping_rate*100:.0f}%")
            previous_contexts = current_contexts
            val_previous_contexts = val_contexts
            break

        previous_contexts = current_contexts
        val_previous_contexts = val_contexts

    total_time = time.time() - start_time
    stats['iterations'] = iteration + 1
    stats['best_val_loss'] = best_val_loss
    stats['final_conv_rate'] = final_conv_rate

    print_flush(f"\nPhase 1 completed:")
    print_flush(f"  Iterations: {stats['iterations']}")
    print_flush(f"  Final loss: {final_loss:.4f}")
    print_flush(f"  Best val loss: {best_val_loss:.4f}")
    print_flush(f"  Final conv rate: {final_conv_rate*100:.0f}%")
    print_flush(f"  Stop reason: {stats['stop_reason']}")
    print_flush(f"  Total time: {total_time:.1f}s")

    return final_loss, stats


def save_checkpoint(
    model: ContextPythiaModel,
    model_config: ContextPythiaConfig,
    phase1_config: Phase1Config,
    final_loss: float,
    stats: dict,
) -> None:
    """Save ContextBlock checkpoint."""
    checkpoint_path = Path(phase1_config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "context_block_state_dict": model.context_block.state_dict(),
        "config": {
            "context_dim": model_config.context_dim,
            "hidden_size": model_config.hidden_size,
        },
        "final_loss": final_loss,
        "stats": stats,
    }

    torch.save(checkpoint, checkpoint_path)
    print_flush(f"\nCheckpoint saved: {checkpoint_path}")


def main() -> None:
    model_config = ContextPythiaConfig()
    phase1_config = Phase1Config()

    parser = argparse.ArgumentParser(
        description='Phase 1: ContextBlock OACD Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--tokens', type=int, required=True,
        help='Number of tokens (REQUIRED)'
    )
    parser.add_argument(
        '--seed', type=int, default=model_config.random_seed,
        help=f'Random seed (default: {model_config.random_seed})'
    )

    args = parser.parse_args()

    # Device setup
    device = torch.device(model_config.device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    set_seed(args.seed)

    print_flush("=" * 70)
    print_flush("PHASE 1: CONTEXTBLOCK OACD TRAINING")
    print_flush("=" * 70)
    print_flush(f"Tokens: {args.tokens:,}")
    print_flush(f"Checkpoint: {phase1_config.checkpoint_path}")
    print_flush("=" * 70)

    # Load data (with train/val split) - キャッシュ付き
    train_ids, val_ids = prepare_phase1_data(
        num_tokens=args.tokens,
        seq_length=phase1_config.internal_seq_length,
        val_split=phase1_config.val_split,
        tokenizer_name=model_config.tokenizer_name,
        device=device,
    )

    # Create model
    print_flush("\n[Model] Creating Context-Pythia...")
    model = ContextPythiaModel(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        context_dim=model_config.context_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        rotary_pct=model_config.rotary_pct,
    ).to(device)

    context_params = sum(p.numel() for p in model.context_block.parameters())
    print_flush(f"ContextBlock parameters: {context_params:,}")

    # Compute token embeddings (CPUに保存)
    print_flush("\n[Embeddings] Computing token embeddings...")
    with torch.no_grad():
        # Train
        train_embeds_gpu = model.embed_in(train_ids.view(-1).to(device))
        train_token_embeds = train_embeds_gpu.cpu()
        del train_embeds_gpu
        # Val
        val_embeds_gpu = model.embed_in(val_ids.view(-1).to(device))
        val_token_embeds = val_embeds_gpu.cpu()
        del val_embeds_gpu

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print_flush(f"  Train embeddings: {train_token_embeds.shape}")
    print_flush(f"  Val embeddings: {val_token_embeds.shape}")

    # Train Phase 1
    final_loss, stats = train_phase1(
        model, train_token_embeds, val_token_embeds, phase1_config, device
    )

    # Save checkpoint
    save_checkpoint(model, model_config, phase1_config, final_loss, stats)

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
