#!/usr/bin/env python3
"""
V-DProj 2-Phase Experiment: Value Compression with Pretraining

Phase 1: V Reconstruction事前学習
  - v_compress, v_restore のみを学習
  - Loss: ||V - V_restored||^2
  - 目標: 圧縮・復元ペアを事前に最適化

Phase 2: LM学習（v_compress, v_restore凍結）
  - v_compress, v_restore を凍結
  - 残りのパラメータでLM学習
  - Loss: Cross-entropy

Usage:
    python3 scripts/experiment_vdproj_2phase.py --samples 10000
    python3 scripts/experiment_vdproj_2phase.py --samples 10000 --phase1-epochs 10
"""

import argparse
import sys
import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import (  # noqa: E402
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
    DEFAULT_PHASE1_LR,
    DEFAULT_PHASE2_LR,
)
from src.models.vdproj_pythia import VDProjPythiaModel  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.training import prepare_data_loaders, get_device  # noqa: E402
from src.utils.evaluation import evaluate_ppl, evaluate_position_wise_ppl  # noqa: E402
from src.utils.device import clear_gpu_cache  # noqa: E402


def get_v_proj_params(model: VDProjPythiaModel) -> list:
    """Get V projection parameters (v_compress, v_restore)."""
    params = []
    for layer in model.layers:
        params.extend(layer.attention.v_compress.parameters())
        params.extend(layer.attention.v_restore.parameters())
    return params


def freeze_v_proj(model: VDProjPythiaModel) -> None:
    """Freeze V projection parameters."""
    for param in get_v_proj_params(model):
        param.requires_grad = False


def train_phase1_epoch(
    model: VDProjPythiaModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Phase 1: Train V reconstruction only. Returns average reconstruction loss."""
    model.train()
    total_recon_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids, _ = batch
        input_ids = input_ids.to(device)

        optimizer.zero_grad()
        _, recon_loss = model(input_ids, return_reconstruction_loss=True)

        if recon_loss is not None:
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            total_recon_loss += recon_loss.item()
            num_batches += 1

    return total_recon_loss / num_batches if num_batches > 0 else 0.0


def evaluate_reconstruction(
    model: VDProjPythiaModel,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate reconstruction loss."""
    model.eval()
    total_recon_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, _ = batch
            input_ids = input_ids.to(device)

            _, recon_loss = model(input_ids, return_reconstruction_loss=True)

            if recon_loss is not None:
                total_recon_loss += recon_loss.item()
                num_batches += 1

    return total_recon_loss / num_batches if num_batches > 0 else 0.0


def train_phase2_epoch(
    model: VDProjPythiaModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Phase 2: Train LM (with V projection frozen). Returns train PPL."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in train_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, return_reconstruction_loss=False)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def run_experiment(
    num_samples: int = 10000,
    seq_length: int = 128,
    phase1_epochs: int = 10,
    phase2_epochs: int = 30,
    batch_size: int = 8,
    phase1_lr: float = DEFAULT_PHASE1_LR,
    phase2_lr: float = DEFAULT_PHASE2_LR,
    v_proj_dim: int = 320,
) -> Dict[str, Any]:
    """Run V-DProj 2-Phase experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("V-DPROJ 2-PHASE EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Phase 1 epochs: {phase1_epochs} (V reconstruction)")
    print_flush(f"Phase 2 epochs: {phase2_epochs} (LM training)")
    print_flush(f"Phase 1 LR: {phase1_lr}")
    print_flush(f"Phase 2 LR: {phase2_lr}")
    print_flush(f"V proj dim: {v_proj_dim}")
    print_flush("=" * 70)

    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
    )

    results: Dict[str, Any] = {}

    print_flush("\n[Model] Creating V-DProj Pythia...")
    model = VDProjPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        v_proj_dim=v_proj_dim,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
    )
    model = model.to(device)

    param_info = model.num_parameters()
    print_flush(f"  Total parameters: {param_info['total']:,}")
    print_flush(f"  V projection: {param_info['v_projection']:,}")

    v_proj_params = get_v_proj_params(model)
    v_proj_count = sum(p.numel() for p in v_proj_params)
    print_flush(f"  V proj trainable: {v_proj_count:,}")

    # ===== Phase 1: V Reconstruction =====
    print_flush("\n" + "=" * 70)
    print_flush("PHASE 1: V RECONSTRUCTION PRETRAINING")
    print_flush("=" * 70)
    print_flush("Training only v_compress and v_restore")
    print_flush("Loss: ||V - V_restored||^2")

    optimizer_phase1 = torch.optim.AdamW(v_proj_params, lr=phase1_lr)

    best_recon_loss = float("inf")
    patience_counter = 0
    phase1_final_epoch = 0

    for epoch in range(1, phase1_epochs + 1):
        start_time = time.time()

        train_recon = train_phase1_epoch(model, train_loader, optimizer_phase1, device)
        val_recon = evaluate_reconstruction(model, val_loader, device)

        elapsed = time.time() - start_time
        phase1_final_epoch = epoch

        improved = val_recon < best_recon_loss
        if improved:
            best_recon_loss = val_recon
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_recon={train_recon:.6f} "
            f"val_recon={val_recon:.6f} [{elapsed:.1f}s] {marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("  -> Early stop (reconstruction converged)")
            break

    print_flush(f"  Best reconstruction loss: {best_recon_loss:.6f}")

    results["phase1"] = {
        "best_recon_loss": best_recon_loss,
        "epochs_trained": phase1_final_epoch,
    }

    # ===== Phase 2: LM Training (V projection frozen) =====
    print_flush("\n" + "=" * 70)
    print_flush("PHASE 2: LM TRAINING (V projection frozen)")
    print_flush("=" * 70)

    freeze_v_proj(model)

    trainable_after_freeze = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_after_freeze = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    print_flush(f"  Trainable: {trainable_after_freeze:,}")
    print_flush(f"  Frozen (V proj): {frozen_after_freeze:,}")

    non_v_proj_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_phase2 = torch.optim.AdamW(non_v_proj_params, lr=phase2_lr)

    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0

    print_flush("\n[Phase 2] Training...")

    for epoch in range(1, phase2_epochs + 1):
        start_time = time.time()

        train_ppl = train_phase2_epoch(model, train_loader, optimizer_phase2, device)
        val_metrics = evaluate_ppl(model, val_loader, device, return_recon_loss=True)

        elapsed = time.time() - start_time

        improved = val_metrics["ppl"] < best_val_ppl
        if improved:
            best_val_ppl = val_metrics["ppl"]
            best_epoch = epoch
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} "
            f"val_ppl={val_metrics['ppl']:.1f} "
            f"recon={val_metrics['recon_loss']:.6f} "
            f"[{elapsed:.1f}s] {marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("  -> Early stop")
            break

    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    print_flush("\n  Position-wise PPL:")
    pos_ppl = evaluate_position_wise_ppl(model, val_loader, device, return_recon_loss=True)
    for pos_range, ppl in pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    results["phase2"] = {
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "position_wise_ppl": pos_ppl,
    }

    cache_info = model.kv_cache_size(seq_length)
    results["cache_reduction"] = cache_info["reduction_percent"]

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\nPhase 1 (V Reconstruction):")
    print_flush(f"  Best reconstruction loss: {results['phase1']['best_recon_loss']:.6f}")
    print_flush(f"  Epochs: {results['phase1']['epochs_trained']}")

    print_flush("\nPhase 2 (LM Training):")
    print_flush(f"  Best PPL: {results['phase2']['best_val_ppl']:.1f}")
    print_flush(f"  Best epoch: {results['phase2']['best_epoch']}")

    print_flush(f"\nKV Cache reduction: {results['cache_reduction']:.1f}%")

    del model
    clear_gpu_cache(device)

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="V-DProj 2-Phase Experiment")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--phase1-epochs", type=int, default=10, help="Phase 1 epochs"
    )
    parser.add_argument(
        "--phase2-epochs", type=int, default=config.num_epochs, help="Phase 2 epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.batch_size, help="Batch size"
    )
    parser.add_argument(
        "--phase1-lr", type=float, default=DEFAULT_PHASE1_LR, help="Phase 1 learning rate"
    )
    parser.add_argument(
        "--phase2-lr", type=float, default=DEFAULT_PHASE2_LR, help="Phase 2 LR"
    )
    parser.add_argument(
        "--v-proj-dim", type=int, default=320, help="V projection dimension"
    )
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        v_proj_dim=args.v_proj_dim,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
