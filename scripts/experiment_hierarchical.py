#!/usr/bin/env python3
"""
Hierarchical Memory Experiment

階層的メモリ（学習可能な展開判断）の実験。

比較対象:
1. Single Memory Infini-Pythia (ベースライン)
2. Multi-Memory Infini-Pythia (Attention-based selection)
3. Hierarchical Memory Pythia (Learned expansion)

Usage:
    python3 scripts/experiment_hierarchical.py --samples 5000 --epochs 30
    python3 scripts/experiment_hierarchical.py --num-fine-memories 8
    python3 scripts/experiment_hierarchical.py --skip-baseline --skip-multi
"""

import argparse
import sys
import time
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE  # noqa: E402
from src.models.infini_pythia import InfiniPythiaModel  # noqa: E402
from src.models.multi_memory_pythia import MultiMemoryInfiniPythiaModel  # noqa: E402
from src.models.hierarchical_pythia import HierarchicalMemoryPythiaModel  # noqa: E402
from src.data.reversal_pairs import get_reversal_pairs  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.training import (  # noqa: E402
    prepare_data_loaders,
    get_device,
    get_tokenizer,
)
from src.utils.evaluation import (  # noqa: E402
    evaluate_position_wise_ppl,
    evaluate_reversal_curse,
)
from src.utils.device import clear_gpu_cache  # noqa: E402


def train_memory_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    model_name: str = "Model",
) -> tuple[float, int, dict]:
    """
    メモリモデルを訓練

    Returns:
        best_val_ppl, best_epoch, best_state_dict
    """
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Reset memory at epoch start
        if hasattr(model, 'reset_memory'):
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
                reduction="sum",
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_tokens += labels.numel()

        train_loss = total_loss / total_tokens
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Validation
        model.eval()
        eval_loss = 0.0
        eval_tokens = 0

        if hasattr(model, 'reset_memory'):
            model.reset_memory()

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits = model(input_ids, update_memory=False)

                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )

                eval_loss += loss.item()
                eval_tokens += labels.numel()

        val_ppl = torch.exp(torch.tensor(eval_loss / eval_tokens)).item()
        elapsed = time.time() - start_time

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:7.1f}, val_ppl={val_ppl:7.1f} "
            f"({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


def run_experiment(
    num_samples: int = 5000,
    seq_length: int = 256,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    num_fine_memories: int = 4,
    skip_baseline: bool = False,
    skip_multi: bool = False,
    skip_hierarchical: bool = False,
) -> dict[str, Any]:
    """Run hierarchical memory experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("HIERARCHICAL MEMORY EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"Fine memories: {num_fine_memories}")
    print_flush(f"Skip baseline: {skip_baseline}")
    print_flush(f"Skip multi: {skip_multi}")
    print_flush(f"Skip hierarchical: {skip_hierarchical}")
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
    tokenizer = get_tokenizer(config.tokenizer_name)
    reversal_pairs = get_reversal_pairs()

    # ===== 1. Single Memory Baseline =====
    if skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. SINGLE MEMORY - SKIPPED")
        print_flush("=" * 70)
        results["single"] = None
    else:
        print_flush("\n" + "=" * 70)
        print_flush("1. SINGLE MEMORY INFINI-PYTHIA")
        print_flush("=" * 70)

        single_model = InfiniPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
            use_delta_rule=True,
        ).to(device)

        param_info = single_model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")

        optimizer = torch.optim.AdamW(single_model.parameters(), lr=lr)

        print_flush("\n[Single] Training...")
        best_ppl, best_epoch, best_state = train_memory_model(
            single_model, train_loader, val_loader, optimizer, device, num_epochs, "Single"
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        # Load best weights for evaluation
        single_model.load_state_dict(best_state)
        single_model.eval()
        single_model.reset_memory()

        print_flush("\n  Position-wise PPL:")
        pos_ppl = evaluate_position_wise_ppl(single_model, val_loader, device)
        for pos_range, ppl in pos_ppl.items():
            print_flush(f"    {pos_range}: {ppl:.1f}")

        print_flush("\n  Reversal Curse:")
        single_model.reset_memory()
        reversal = evaluate_reversal_curse(single_model, tokenizer, reversal_pairs, device)
        print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")

        results["single"] = {
            "best_val_ppl": best_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": pos_ppl,
            "reversal_curse": reversal,
        }

        del single_model
        clear_gpu_cache(device)

    # ===== 2. Multi-Memory =====
    if skip_multi:
        print_flush("\n" + "=" * 70)
        print_flush("2. MULTI-MEMORY - SKIPPED")
        print_flush("=" * 70)
        results["multi"] = None
    else:
        print_flush("\n" + "=" * 70)
        print_flush(f"2. MULTI-MEMORY INFINI-PYTHIA ({num_fine_memories} memories)")
        print_flush("=" * 70)

        multi_model = MultiMemoryInfiniPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
            use_delta_rule=True,
            num_memories=num_fine_memories,
        ).to(device)

        param_info = multi_model.num_parameters()
        memory_info = multi_model.memory_info()
        print_flush(f"  Parameters: {param_info['total']:,}")
        print_flush(f"  Memory: {memory_info['total_bytes']:,} bytes")

        optimizer = torch.optim.AdamW(multi_model.parameters(), lr=lr)

        print_flush("\n[Multi] Training...")
        best_ppl, best_epoch, best_state = train_memory_model(
            multi_model, train_loader, val_loader, optimizer, device, num_epochs, "Multi"
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        multi_model.load_state_dict(best_state)
        multi_model.eval()
        multi_model.reset_memory()

        print_flush("\n  Position-wise PPL:")
        pos_ppl = evaluate_position_wise_ppl(multi_model, val_loader, device)
        for pos_range, ppl in pos_ppl.items():
            print_flush(f"    {pos_range}: {ppl:.1f}")

        print_flush("\n  Reversal Curse:")
        multi_model.reset_memory()
        reversal = evaluate_reversal_curse(multi_model, tokenizer, reversal_pairs, device)
        print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")

        results["multi"] = {
            "best_val_ppl": best_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": pos_ppl,
            "reversal_curse": reversal,
        }

        del multi_model
        clear_gpu_cache(device)

    # ===== 3. Hierarchical Memory =====
    if skip_hierarchical:
        print_flush("\n" + "=" * 70)
        print_flush("3. HIERARCHICAL MEMORY - SKIPPED")
        print_flush("=" * 70)
        results["hierarchical"] = None
    else:
        print_flush("\n" + "=" * 70)
        print_flush(f"3. HIERARCHICAL MEMORY PYTHIA ({num_fine_memories} fine memories)")
        print_flush("=" * 70)
        print_flush("  Features:")
        print_flush("    - Fine memories: stored permanently")
        print_flush("    - Coarse memory: dynamically computed (sum of fine)")
        print_flush("    - Expansion gate: learned decision")

        hier_model = HierarchicalMemoryPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
            use_delta_rule=True,
            num_fine_memories=num_fine_memories,
        ).to(device)

        param_info = hier_model.num_parameters()
        memory_info = hier_model.memory_info()
        print_flush(f"\n  Parameters: {param_info['total']:,}")
        print_flush(f"  Expansion gate params: {param_info['expansion_gate']:,}")
        print_flush(f"  Memory: {memory_info['total_bytes']:,} bytes")

        optimizer = torch.optim.AdamW(hier_model.parameters(), lr=lr)

        print_flush("\n[Hierarchical] Training...")
        best_ppl, best_epoch, best_state = train_memory_model(
            hier_model, train_loader, val_loader, optimizer, device, num_epochs, "Hierarchical"
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        hier_model.load_state_dict(best_state)
        hier_model.eval()
        hier_model.reset_memory()

        print_flush("\n  Position-wise PPL:")
        pos_ppl = evaluate_position_wise_ppl(hier_model, val_loader, device)
        for pos_range, ppl in pos_ppl.items():
            print_flush(f"    {pos_range}: {ppl:.1f}")

        print_flush("\n  Reversal Curse:")
        hier_model.reset_memory()
        reversal = evaluate_reversal_curse(hier_model, tokenizer, reversal_pairs, device)
        print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")

        results["hierarchical"] = {
            "best_val_ppl": best_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": pos_ppl,
            "reversal_curse": reversal,
        }

        del hier_model
        clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Best PPL | Epoch | Memories |")
    print_flush("|-------|----------|-------|----------|")

    if results.get("single"):
        print_flush(
            f"| Single Memory | {results['single']['best_val_ppl']:.1f} | "
            f"{results['single']['best_epoch']} | 1 |"
        )
    if results.get("multi"):
        print_flush(
            f"| Multi Memory | {results['multi']['best_val_ppl']:.1f} | "
            f"{results['multi']['best_epoch']} | {num_fine_memories} |"
        )
    if results.get("hierarchical"):
        print_flush(
            f"| Hierarchical | {results['hierarchical']['best_val_ppl']:.1f} | "
            f"{results['hierarchical']['best_epoch']} | {num_fine_memories} |"
        )

    # Reversal Curse comparison
    print_flush("\n| Model | Forward PPL | Backward PPL | Gap |")
    print_flush("|-------|-------------|--------------|-----|")

    for name, key in [("Single", "single"), ("Multi", "multi"), ("Hierarchical", "hierarchical")]:
        if results.get(key):
            rev = results[key]["reversal_curse"]
            gap = rev["backward_ppl"] - rev["forward_ppl"]
            print_flush(
                f"| {name} | {rev['forward_ppl']:.1f} | "
                f"{rev['backward_ppl']:.1f} | {gap:+.1f} |"
            )

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="Hierarchical Memory Experiment")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument(
        "--num-fine-memories", type=int, default=4,
        help="Number of fine-grained memories"
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Skip single memory baseline")
    parser.add_argument("--skip-multi", action="store_true", help="Skip multi-memory")
    parser.add_argument("--skip-hierarchical", action="store_true", help="Skip hierarchical")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_fine_memories=args.num_fine_memories,
        skip_baseline=args.skip_baseline,
        skip_multi=args.skip_multi,
        skip_hierarchical=args.skip_hierarchical,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
