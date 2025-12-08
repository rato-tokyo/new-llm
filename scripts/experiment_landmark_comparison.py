#!/usr/bin/env python3
"""
Landmark方式比較実験

案1: memory_norm - memory_normをそのまま使用（追加パラメータなし）
案2: learned - 学習可能な射影でLandmarkを計算

両方式でMultiMemoryLayerを訓練し、PPLを比較する。
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.pythia import PythiaConfig
from src.models import create_model, LandmarkType
from src.utils.device import clear_gpu_cache
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import get_device, prepare_data_loaders


@dataclass
class ExpConfig:
    """実験設定"""
    num_samples: int = 5000
    seq_length: int = 256
    val_split: float = 0.1
    num_epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4
    num_memories: int = 4
    early_stopping_patience: int = 5


def train_and_evaluate(
    landmark_type: LandmarkType,
    train_loader,
    val_loader,
    config: PythiaConfig,
    exp_config: ExpConfig,
    device: torch.device,
) -> dict:
    """指定したlandmark_typeでモデルを訓練・評価"""

    print_flush(f"\n{'='*60}")
    print_flush(f"Landmark Type: {landmark_type}")
    print_flush(f"{'='*60}")

    # Create model
    model = create_model(
        "multi_memory",
        config=config,
        num_memories=exp_config.num_memories,
        landmark_type=landmark_type,
    )
    model = model.to(device)

    # Print model info
    param_info = model.num_parameters()
    print_flush(f"Parameters: {param_info['total']:,}")
    print_flush(f"Layer 0 params: {param_info['per_layer'][0]:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config.lr)

    # Training
    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, exp_config.num_epochs + 1):
        start_time = time.time()

        # Reset memory
        model.reset_memory()
        model.train()

        total_loss = 0.0
        total_tokens = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, update_memory=True)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_tokens += labels.numel()

        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

        # Validation
        model.eval()
        model.reset_memory()

        eval_loss = 0.0
        eval_tokens = 0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits = model(input_ids, update_memory=False)

                loss = nn.functional.cross_entropy(
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
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train={train_ppl:7.1f}, val={val_ppl:7.1f} "
            f"({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= exp_config.early_stopping_patience:
            print_flush("  Early stopping")
            break

    print_flush(f"  Best: epoch {best_epoch}, val_ppl={best_val_ppl:.1f}")

    # Cleanup
    del model
    clear_gpu_cache(device)

    return {
        "landmark_type": landmark_type,
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "total_params": param_info['total'],
    }


def main():
    print_flush("=" * 60)
    print_flush("Landmark方式比較実験")
    print_flush("=" * 60)

    set_seed(42)
    device = get_device()
    config = PythiaConfig()
    exp_config = ExpConfig()

    print_flush(f"\nDevice: {device}")
    print_flush(f"Samples: {exp_config.num_samples}")
    print_flush(f"Seq length: {exp_config.seq_length}")
    print_flush(f"Epochs: {exp_config.num_epochs}")
    print_flush(f"Memories: {exp_config.num_memories}")

    # Load data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=exp_config.num_samples,
        seq_length=exp_config.seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=exp_config.val_split,
        batch_size=exp_config.batch_size,
    )

    # Run experiments
    results = []

    for landmark_type in ["memory_norm", "learned"]:
        result = train_and_evaluate(
            landmark_type=landmark_type,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            exp_config=exp_config,
            device=device,
        )
        results.append(result)

    # Summary
    print_flush("\n" + "=" * 60)
    print_flush("SUMMARY")
    print_flush("=" * 60)
    print_flush("\n| Landmark Type | Best PPL | Epoch | Params |")
    print_flush("|---------------|----------|-------|--------|")

    for r in results:
        print_flush(
            f"| {r['landmark_type']:13} | {r['best_val_ppl']:8.1f} | "
            f"{r['best_epoch']:5} | {r['total_params']:,} |"
        )

    # Comparison
    ppl_diff = results[1]["best_val_ppl"] - results[0]["best_val_ppl"]
    param_diff = results[1]["total_params"] - results[0]["total_params"]

    print_flush(f"\nPPL差 (learned - memory_norm): {ppl_diff:+.1f}")
    print_flush(f"追加パラメータ: {param_diff:,}")

    if ppl_diff < 0:
        print_flush("\n→ learned方式が優れている")
    elif ppl_diff > 0:
        print_flush("\n→ memory_norm方式が優れている")
    else:
        print_flush("\n→ 同等")


if __name__ == "__main__":
    main()
