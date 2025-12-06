#!/usr/bin/env python3
"""
Multi-Memory Infini-Attention Experiment

複数の独立したメモリをAttention-based方式で動的選択するInfini-Attentionの実験。

特徴:
- 各メモリは独立して更新される（ラウンドロビン）
- クエリとメモリのz（正規化項）との内積で関連度を計算
- 関連度に基づいてSoftmax重み付けで混合
- 追加パラメータなし（学習が安定）

Usage:
    python3 scripts/experiment_multi_memory.py --num-memories 4
    python3 scripts/experiment_multi_memory.py --num-memories 8 --samples 10000
"""

import argparse
import sys
import time
from typing import Any, Union

import torch

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE  # noqa: E402
from src.models.pythia import PythiaModel  # noqa: E402
from src.models.infini_pythia import InfiniPythiaModel  # noqa: E402
from src.models.multi_memory_pythia import MultiMemoryInfiniPythiaModel  # noqa: E402
from src.data.reversal_pairs import get_reversal_pairs  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.training import (  # noqa: E402
    prepare_data_loaders,
    get_device,
    get_tokenizer,
    train_epoch,
    evaluate,
)
from src.utils.evaluation import (  # noqa: E402
    evaluate_position_wise_ppl,
    evaluate_reversal_curse,
)
from src.utils.device import clear_gpu_cache  # noqa: E402


def train_model(
    model: Union[InfiniPythiaModel, MultiMemoryInfiniPythiaModel],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    reset_memory_per_epoch: bool = True,
) -> tuple[float, int, dict]:
    """
    モデルを訓練

    Returns:
        best_val_ppl, best_epoch, best_state_dict
    """
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        if reset_memory_per_epoch:
            model.reset_memory()

        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(input_ids, update_memory=True)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            total_tokens += (labels != -100).sum().item()

        train_ppl = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()

        # Validation
        model.eval()
        model.reset_memory()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits = model(input_ids, update_memory=True)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

                val_loss += loss.item() * labels.numel()
                val_tokens += (labels != -100).sum().item()

        val_ppl = torch.exp(torch.tensor(val_loss / max(val_tokens, 1))).item()

        elapsed = time.time() - start_time

        is_best = val_ppl < best_val_ppl
        if is_best:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:7.1f}, "
            f"val_ppl={val_ppl:7.1f} ({elapsed:.1f}s){marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush(f"  Early stopping at epoch {epoch}")
            break

    return best_val_ppl, best_epoch, best_state_dict


def run_experiment(
    num_samples: int = 5000,
    seq_length: int = 256,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    use_delta_rule: bool = True,
    skip_baseline: bool = False,
    skip_single: bool = False,
    num_memories: int = 4,
) -> dict[str, Any]:
    """Run Multi-Memory experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("MULTI-MEMORY INFINI-ATTENTION EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"Delta rule: {use_delta_rule}")
    print_flush(f"Number of memories: {num_memories}")
    print_flush(f"Skip baseline: {skip_baseline}")
    print_flush(f"Skip single memory: {skip_single}")
    print_flush("=" * 70)

    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
    )

    results: dict[str, Any] = {}

    # ===== 1. Pythia Baseline (RoPE) =====
    if skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA (RoPE) - SKIPPED")
        print_flush("=" * 70)
        results["pythia"] = None
    else:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA (RoPE baseline)")
        print_flush("=" * 70)

        pythia_model = PythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
        ).to(device)

        param_info = pythia_model.num_parameters()
        print_flush(f"\n  Total parameters: {param_info['total']:,}")

        optimizer = torch.optim.AdamW(pythia_model.parameters(), lr=lr)

        print_flush("\n[Pythia] Training...")
        best_val_ppl, best_epoch = 0.0, 0
        patience_counter = 0
        best_state_dict = None

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_ppl = train_epoch(pythia_model, train_loader, optimizer, device)
            val_ppl = evaluate(pythia_model, val_loader, device)
            elapsed = time.time() - start_time

            is_best = val_ppl < best_val_ppl or epoch == 1
            if is_best:
                best_val_ppl = val_ppl
                best_epoch = epoch
                best_state_dict = {k: v.cpu().clone() for k, v in pythia_model.state_dict().items()}
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 1
                marker = ""

            print_flush(
                f"  Epoch {epoch:2d}: train_ppl={train_ppl:7.1f}, "
                f"val_ppl={val_ppl:7.1f} ({elapsed:.1f}s){marker}"
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print_flush(f"  Early stopping at epoch {epoch}")
                break

        print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

        # Position-wise PPL
        pythia_model.load_state_dict(best_state_dict)
        pythia_model.eval()
        pythia_pos_ppl = evaluate_position_wise_ppl(pythia_model, val_loader, device)
        print_flush("\n  Position-wise PPL:")
        for pos_range, ppl in pythia_pos_ppl.items():
            print_flush(f"    {pos_range}: {ppl:.1f}")

        # Reversal Curse
        tokenizer = get_tokenizer(config.tokenizer_name)
        reversal_pairs = get_reversal_pairs()
        pythia_reversal = evaluate_reversal_curse(
            pythia_model, tokenizer, reversal_pairs, device
        )
        print_flush(f"\n  Reversal Curse:")
        print_flush(f"    Forward PPL: {pythia_reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {pythia_reversal['backward_ppl']:.1f}")

        results["pythia"] = {
            "best_ppl": best_val_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": pythia_pos_ppl,
            "reversal_curse": pythia_reversal,
            "model_state_dict": best_state_dict,
        }

        del pythia_model
        clear_gpu_cache(device)

    # ===== 2. Infini-Pythia (Single Memory) =====
    if skip_single:
        print_flush("\n" + "=" * 70)
        print_flush("2. INFINI-PYTHIA (Single Memory) - SKIPPED")
        print_flush("=" * 70)
        results["single"] = None
    else:
        print_flush("\n" + "=" * 70)
        print_flush("2. INFINI-PYTHIA (Single Memory)")
        print_flush("=" * 70)
        print_flush("  Layer 0: Infini-Attention (NoPE, Memory Only)")
        print_flush("  Layer 1-5: Standard Pythia (RoPE)")

        single_model = InfiniPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
            use_delta_rule=use_delta_rule,
            num_memory_banks=1,
        ).to(device)

        param_info = single_model.num_parameters()
        print_flush(f"\n  Total parameters: {param_info['total']:,}")
        memory_info = single_model.memory_info()
        print_flush(f"  Memory: {memory_info['total_bytes']:,} bytes")

        optimizer = torch.optim.AdamW(single_model.parameters(), lr=lr)

        print_flush("\n[Single] Training...")
        best_val_ppl, best_epoch, best_state_dict = train_model(
            single_model, train_loader, val_loader, optimizer, device, num_epochs
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

        # Position-wise PPL
        single_model.load_state_dict(best_state_dict)
        single_model.eval()
        single_model.reset_memory()
        single_pos_ppl = evaluate_position_wise_ppl(single_model, val_loader, device)
        print_flush("\n  Position-wise PPL:")
        for pos_range, ppl in single_pos_ppl.items():
            print_flush(f"    {pos_range}: {ppl:.1f}")

        # Reversal Curse
        tokenizer = get_tokenizer(config.tokenizer_name)
        reversal_pairs = get_reversal_pairs()
        single_reversal = evaluate_reversal_curse(
            single_model, tokenizer, reversal_pairs, device
        )
        print_flush(f"\n  Reversal Curse:")
        print_flush(f"    Forward PPL: {single_reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {single_reversal['backward_ppl']:.1f}")

        results["single"] = {
            "best_ppl": best_val_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": single_pos_ppl,
            "reversal_curse": single_reversal,
            "model_state_dict": best_state_dict,
        }

        del single_model
        clear_gpu_cache(device)

    # ===== 3. Multi-Memory Infini-Pythia =====
    print_flush("\n" + "=" * 70)
    print_flush(f"3. MULTI-MEMORY INFINI-PYTHIA ({num_memories} memories)")
    print_flush("=" * 70)
    print_flush("  Layer 0: Multi-Memory Infini-Attention (Attention-based selection)")
    print_flush(f"    - {num_memories} independent memories")
    print_flush("    - phi(Q) @ z for relevance scoring")
    print_flush("    - Softmax weighted combination")
    print_flush("  Layer 1-5: Standard Pythia (RoPE)")

    multi_model = MultiMemoryInfiniPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
        use_delta_rule=use_delta_rule,
        num_memories=num_memories,
    ).to(device)

    param_info = multi_model.num_parameters()
    print_flush(f"\n  Total parameters: {param_info['total']:,}")
    memory_info = multi_model.memory_info()
    print_flush(f"  Total memory: {memory_info['total_bytes']:,} bytes ({num_memories} memories)")

    optimizer = torch.optim.AdamW(multi_model.parameters(), lr=lr)

    print_flush("\n[Multi] Training...")
    best_val_ppl, best_epoch, best_state_dict = train_model(
        multi_model, train_loader, val_loader, optimizer, device, num_epochs
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    # Position-wise PPL
    multi_model.load_state_dict(best_state_dict)
    multi_model.eval()
    multi_model.reset_memory()
    multi_pos_ppl = evaluate_position_wise_ppl(multi_model, val_loader, device)
    print_flush("\n  Position-wise PPL:")
    for pos_range, ppl in multi_pos_ppl.items():
        print_flush(f"    {pos_range}: {ppl:.1f}")

    # Reversal Curse
    tokenizer = get_tokenizer(config.tokenizer_name)
    reversal_pairs = get_reversal_pairs()
    multi_reversal = evaluate_reversal_curse(
        multi_model, tokenizer, reversal_pairs, device
    )
    print_flush(f"\n  Reversal Curse:")
    print_flush(f"    Forward PPL: {multi_reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {multi_reversal['backward_ppl']:.1f}")

    results["multi"] = {
        "best_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "position_wise_ppl": multi_pos_ppl,
        "reversal_curse": multi_reversal,
        "num_memories": num_memories,
        "model_state_dict": best_state_dict,
    }

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Best PPL | Best Epoch | Memory |")
    print_flush("|-------|----------|------------|--------|")

    if results.get("pythia"):
        print_flush(
            f"| Pythia | {results['pythia']['best_ppl']:.1f} | "
            f"{results['pythia']['best_epoch']} | - |"
        )

    if results.get("single"):
        print_flush(
            f"| Single Memory | {results['single']['best_ppl']:.1f} | "
            f"{results['single']['best_epoch']} | 1 |"
        )

    print_flush(
        f"| Multi Memory | {results['multi']['best_ppl']:.1f} | "
        f"{results['multi']['best_epoch']} | {num_memories} |"
    )

    # PPL comparison
    if results.get("single"):
        ppl_diff = results["multi"]["best_ppl"] - results["single"]["best_ppl"]
        print_flush(f"\nMulti vs Single: {ppl_diff:+.1f} PPL")

    if results.get("pythia"):
        ppl_diff = results["multi"]["best_ppl"] - results["pythia"]["best_ppl"]
        print_flush(f"Multi vs Pythia: {ppl_diff:+.1f} PPL")

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(
        description="Multi-Memory Infini-Attention Experiment"
    )
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument(
        "--epochs", type=int, default=config.num_epochs, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.batch_size, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=config.learning_rate, help="Learning rate"
    )
    parser.add_argument(
        "--no-delta-rule", action="store_true", help="Disable delta rule"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Pythia baseline"
    )
    parser.add_argument(
        "--skip-single", action="store_true", help="Skip single memory Infini"
    )
    parser.add_argument(
        "--num-memories", type=int, default=4, help="Number of memories"
    )

    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_delta_rule=not args.no_delta_rule,
        skip_baseline=args.skip_baseline,
        skip_single=args.skip_single,
        num_memories=args.num_memories,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
