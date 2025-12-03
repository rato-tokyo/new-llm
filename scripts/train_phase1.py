#!/usr/bin/env python3
"""
Phase 1: ContextBlock OACD Training

ContextBlockの多様性学習を行い、チェックポイントを保存する。
このスクリプトで学習したパラメータは、experiment_pythia_comparison.pyで使用される。

機能（削除禁止）:
- 収束率表示: 各イテレーションでconv=XX%を表示
- Early Stopping: 収束率が閾値以上で停止
- No Improvement Patience: N回改善なしで停止
- Validation: 検証データでの評価
- min/max iteration: 最小・最大イテレーション数

Usage:
    python3 scripts/train_phase1.py --tokens 100000
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub.utils import HfHubHTTPError

# Add project root to path
sys.path.insert(0, ".")

from config import Phase1Config, ContextPythiaConfig
from src.models.context_pythia import ContextPythiaModel
from src.losses.diversity import oacd_loss
from src.utils.io import print_flush


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pile_tokens(
    num_tokens: int,
    model_config: ContextPythiaConfig,
    phase1_config: Phase1Config,
    device: torch.device,
    max_retries: int = 5,
    retry_delay: float = 30.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load tokens from Pile dataset for Phase 1 training.

    Args:
        num_tokens: 必要なトークン数
        max_retries: 429エラー時の最大リトライ回数
        retry_delay: リトライ間の待機時間（秒）

    Returns:
        train_ids: [num_train_sequences, internal_seq_length]
        val_ids: [num_val_sequences, internal_seq_length]
    """
    seq_length = phase1_config.internal_seq_length

    # シーケンス長で割り切れるようにトークン数を調整
    num_sequences = (num_tokens + seq_length - 1) // seq_length
    actual_tokens = num_sequences * seq_length

    # Train/Val分割
    val_sequences = max(1, int(num_sequences * phase1_config.val_split))
    train_sequences = num_sequences - val_sequences

    print_flush(f"Loading Pile dataset: {num_tokens:,} tokens")
    print_flush(f"  Train: {train_sequences:,} sequences ({train_sequences * seq_length:,} tokens)")
    print_flush(f"  Val: {val_sequences:,} sequences ({val_sequences * seq_length:,} tokens)")

    # Load tokenizer
    print_flush(f"  Loading tokenizer: {model_config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Pile dataset (streaming)
    print_flush("  Loading dataset (streaming)...")
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    # Collect tokens with retry logic
    all_tokens = []
    retry_count = 0

    print_flush(f"  Tokenizing...")

    dataset_iter = iter(dataset)
    while len(all_tokens) < actual_tokens:
        try:
            example = next(dataset_iter)
            text = example["text"]
            if not text or len(text) < 100:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

            if len(all_tokens) % 100000 == 0 and len(all_tokens) > 0:
                print_flush(f"    Collected {len(all_tokens):,} tokens...")

            retry_count = 0  # Reset on success

        except StopIteration:
            break
        except HfHubHTTPError as e:
            if "429" in str(e) and retry_count < max_retries:
                retry_count += 1
                print_flush(f"    Rate limited (429). Retry {retry_count}/{max_retries} after {retry_delay}s...")
                time.sleep(retry_delay)
                dataset = load_dataset(
                    "monology/pile-uncopyrighted",
                    split="train",
                    streaming=True,
                )
                dataset_iter = iter(dataset)
                skip_count = len(all_tokens) // 500
                for _ in range(skip_count):
                    try:
                        next(dataset_iter)
                    except StopIteration:
                        break
            else:
                raise

    # Truncate and reshape
    all_tokens = all_tokens[:actual_tokens]
    input_ids = torch.tensor(all_tokens, dtype=torch.long, device=device)
    input_ids = input_ids.view(num_sequences, seq_length)

    # Split into train/val
    train_ids = input_ids[:train_sequences]
    val_ids = input_ids[train_sequences:]

    print_flush(f"  Loaded {input_ids.numel():,} tokens")

    return train_ids, val_ids


def compute_convergence_rate(
    current_contexts: torch.Tensor,
    previous_contexts: torch.Tensor,
    threshold: float,
) -> float:
    """
    収束率を計算（CPUで計算してメモリ節約）

    Args:
        current_contexts: 現在のcontext [N, context_dim] (CPU tensor)
        previous_contexts: 前回のcontext [N, context_dim] (CPU tensor)
        threshold: 収束判定の閾値

    Returns:
        収束率 (0.0-1.0)
    """
    with torch.no_grad():
        # CPUで計算
        current = current_contexts.cpu() if current_contexts.is_cuda else current_contexts
        previous = previous_contexts.cpu() if previous_contexts.is_cuda else previous_contexts

        # 各トークンの変化量
        diff = ((current - previous) ** 2).mean(dim=1)
        # 閾値以下の割合
        converged = (diff < threshold).float().mean().item()
    return converged


def evaluate_diversity(
    model: ContextPythiaModel,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    検証データでの多様性損失を計算

    Returns:
        平均OACD損失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for (inputs,) in val_loader:
            inputs = inputs.to(device)
            _, context = model.forward_with_context_output(inputs)
            context_flat = context.view(-1, context.size(-1))
            loss = oacd_loss(context_flat, centroid_weight=0.1)
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def train_phase1(
    model: ContextPythiaModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    phase1_config: Phase1Config,
    device: torch.device,
) -> Tuple[float, dict]:
    """
    Phase 1: Train ContextBlock with OACD diversity loss.

    機能（削除禁止）:
    - 収束率表示
    - Early Stopping（収束率ベース）
    - No Improvement Patience
    - Validation評価
    - min/max iteration

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

    print_flush(f"\nPhase 1 Training:")
    print_flush(f"  Min iterations: {phase1_config.min_iterations}")
    print_flush(f"  Max iterations: {phase1_config.max_iterations}")
    print_flush(f"  Learning rate: {phase1_config.learning_rate}")
    print_flush(f"  Early stopping rate: {phase1_config.early_stopping_rate * 100:.0f}%")
    print_flush(f"  No improvement patience: {phase1_config.no_improvement_patience}")

    start_time = time.time()

    # 前回のcontextを保存（収束率計算用）
    previous_contexts: Optional[torch.Tensor] = None
    best_val_loss = float('inf')
    no_improvement_count = 0
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
        epoch_loss = 0.0
        batch_count = 0
        all_contexts_cpu = []  # CPUに保存してメモリ節約

        # 勾配累積: 複数バッチの勾配を累積してから1回更新
        optimizer.zero_grad()

        for batch_idx, (inputs,) in enumerate(train_loader):
            if batch_idx >= phase1_config.batches_per_iteration:
                break

            inputs = inputs.to(device)

            # Forward to get context
            _, context = model.forward_with_context_output(inputs)

            # Flatten context: [batch*seq, context_dim]
            context_flat = context.view(-1, context.size(-1))

            # OACD diversity loss（バッチ数で割って勾配累積）
            loss = oacd_loss(context_flat, centroid_weight=0.1)
            scaled_loss = loss / phase1_config.batches_per_iteration

            scaled_loss.backward()

            epoch_loss += loss.item()
            batch_count += 1
            all_contexts_cpu.append(context_flat.detach().cpu())  # CPUに保存

            # メモリ解放
            del context, context_flat

        # 勾配クリッピングとパラメータ更新（イテレーションごとに1回）
        torch.nn.utils.clip_grad_norm_(model.context_block.parameters(), 1.0)
        optimizer.step()

        # GPUキャッシュクリア
        if device.type == "cuda":
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(batch_count, 1)
        final_loss = avg_loss

        # 収束率計算（CPUで計算してメモリ節約）
        current_contexts_cpu = torch.cat(all_contexts_cpu, dim=0)
        if previous_contexts is not None and len(previous_contexts) == len(current_contexts_cpu):
            conv_rate = compute_convergence_rate(
                current_contexts_cpu, previous_contexts, phase1_config.convergence_threshold
            )
        else:
            conv_rate = 0.0
        final_conv_rate = conv_rate

        # Validation評価
        val_loss = evaluate_diversity(model, val_loader, device)

        # ログ出力
        elapsed = time.time() - start_time
        print_flush(
            f"  Iter {iteration + 1:2d}: loss={avg_loss:.4f}, "
            f"val_loss={val_loss:.4f}, conv={conv_rate*100:.0f}% [{elapsed:.1f}s]"
        )

        # 改善チェック
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early Stopping判定（min_iterations後のみ）
        if iteration >= phase1_config.min_iterations - 1:
            # 収束率ベースのEarly Stopping
            if conv_rate >= phase1_config.early_stopping_rate:
                stats['early_stopped'] = True
                stats['stop_reason'] = f'convergence_rate >= {phase1_config.early_stopping_rate*100:.0f}%'
                print_flush(f"  → Early stop: conv {conv_rate*100:.0f}% >= {phase1_config.early_stopping_rate*100:.0f}%")
                previous_contexts = current_contexts_cpu.clone()
                break

            # No Improvement Patience
            if no_improvement_count >= phase1_config.no_improvement_patience:
                stats['early_stopped'] = True
                stats['stop_reason'] = f'no_improvement for {phase1_config.no_improvement_patience} iterations'
                print_flush(f"  → Early stop: no improvement for {phase1_config.no_improvement_patience} iterations")
                previous_contexts = current_contexts_cpu.clone()
                break

        previous_contexts = current_contexts_cpu.clone()

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
        '--batch-size', type=int, default=model_config.phase2_batch_size,
        help=f'Batch size (default: {model_config.phase2_batch_size})'
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
    print_flush(f"Batch size: {args.batch_size}")
    print_flush(f"Checkpoint: {phase1_config.checkpoint_path}")
    print_flush("=" * 70)

    # Load data (with train/val split)
    train_ids, val_ids = load_pile_tokens(
        num_tokens=args.tokens,
        model_config=model_config,
        phase1_config=phase1_config,
        device=device,
    )

    # Create dataloaders
    train_dataset = TensorDataset(train_ids)
    val_dataset = TensorDataset(val_ids)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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

    # Train Phase 1
    final_loss, stats = train_phase1(model, train_loader, val_loader, phase1_config, device)

    # Save checkpoint
    save_checkpoint(model, model_config, phase1_config, final_loss, stats)

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
