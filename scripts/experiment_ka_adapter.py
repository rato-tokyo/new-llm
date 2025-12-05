#!/usr/bin/env python3
"""
KA Cache Adapter Experiment

案1: Adapter方式のKAキャッシュ実験。
既存モデルの重みを凍結し、Adapterのみを学習してKAキャッシュの精度を改善。

Usage:
    # フルパイプライン: 事前学習 → Adapter学習 → 評価
    python3 scripts/experiment_ka_adapter.py --samples 5000 --epochs 10

    # Adapter学習をスキップ（Adapterなしでの推論のみ）
    python3 scripts/experiment_ka_adapter.py --samples 5000 --skip-adapter-training

    # 事前学習済みモデルを使用
    python3 scripts/experiment_ka_adapter.py --pretrained-path model.pt
"""

import argparse
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from config.pythia import PythiaConfig
from src.config.experiment_defaults import (
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
)
from src.models.ka_adapter import KAAdapterPythiaModel
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import prepare_data_loaders, get_device, get_tokenizer
from src.utils.evaluation import evaluate_reversal_curse
from src.data.reversal_pairs import get_reversal_pairs


def train_base_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
) -> Dict[str, Any]:
    """Train base model (all parameters)"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_ppl = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print_flush("  Training base model (all parameters)...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

        # Validate
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )
                val_loss += loss.item()
                val_tokens += labels.numel()

        val_ppl = torch.exp(torch.tensor(val_loss / val_tokens)).item()

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(f"    Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} {marker}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("    -> Early stop")
            break

    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")
    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
    }


def train_adapter_only(
    model: KAAdapterPythiaModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
) -> Dict[str, Any]:
    """Train only adapter parameters (base model frozen)"""
    model = model.to(device)

    # Freeze base model, keep only adapter trainable
    model.freeze_base_model()

    # Only optimize adapter parameters
    adapter_params = model.get_adapter_parameters()
    optimizer = torch.optim.AdamW(adapter_params, lr=lr)

    param_info = model.num_parameters()
    print_flush(f"  Adapter parameters: {param_info['adapter']:,}")
    print_flush(f"  Trainable parameters: {param_info['trainable']:,}")

    best_val_ppl = float("inf")
    best_epoch = 0
    best_adapter_state = {}
    patience_counter = 0

    print_flush("  Training adapter only (base frozen)...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            batch_size, seq_len = input_ids.shape

            optimizer.zero_grad()

            # Train with KA adapter cache (autoregressive)
            cache = None
            all_logits = []

            for t in range(seq_len):
                if t == 0:
                    token_input = input_ids[:, :1]
                else:
                    token_input = input_ids[:, t:t+1]

                logits, cache = model.forward_with_cache(
                    token_input, cache, use_ka_adapter=True
                )
                all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=1)
            loss = F.cross_entropy(
                all_logits.view(-1, all_logits.size(-1)),
                labels.view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params, GRADIENT_CLIP)
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

        # Validate with KA adapter
        val_ppl = evaluate_with_cache(
            model, val_loader, device, use_ka_adapter=True
        )["ppl"]

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            # Save only adapter state
            best_adapter_state = {
                k: v.clone() for k, v in model.state_dict().items()
                if "adapter" in k
            }
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(f"    Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} {marker}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("    -> Early stop")
            break

    print_flush(f"  Best adapter: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    # Restore best adapter state
    current_state = model.state_dict()
    current_state.update(best_adapter_state)
    model.load_state_dict(current_state)

    return {
        "model": model,
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
    }


@torch.no_grad()
def evaluate_with_cache(
    model: KAAdapterPythiaModel,
    val_loader: DataLoader,
    device: torch.device,
    use_ka_adapter: bool,
) -> Dict[str, float]:
    """Evaluate model with KV or KA adapter cache"""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_time = 0.0

    for batch in val_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        batch_size, seq_len = input_ids.shape

        start_time = time.time()

        cache = None
        all_logits = []

        for t in range(seq_len):
            if t == 0:
                token_input = input_ids[:, :1]
            else:
                token_input = input_ids[:, t:t+1]

            logits, cache = model.forward_with_cache(
                token_input, cache, use_ka_adapter=use_ka_adapter
            )
            all_logits.append(logits)

        elapsed = time.time() - start_time
        total_time += elapsed

        all_logits = torch.cat(all_logits, dim=1)
        loss = F.cross_entropy(
            all_logits.view(-1, all_logits.size(-1)),
            labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    tokens_per_sec = total_tokens / total_time

    return {
        "ppl": ppl,
        "total_time": total_time,
        "tokens_per_sec": tokens_per_sec,
    }


def run_experiment(
    num_samples: int = 5000,
    seq_length: int = 128,
    num_epochs: int = 10,
    adapter_epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    adapter_lr: float = 1e-3,
    adapter_bottleneck: int = 64,
    skip_adapter_training: bool = False,
    pretrained_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run KA adapter experiment"""
    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("KA CACHE ADAPTER EXPERIMENT (案1)")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Base epochs: {num_epochs}")
    print_flush(f"Adapter epochs: {adapter_epochs}")
    print_flush(f"Adapter bottleneck: {adapter_bottleneck}")
    print_flush(f"Skip adapter training: {skip_adapter_training}")
    print_flush("=" * 70)

    # Prepare data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
        include_reversal_pairs=False,
    )

    # Create model
    print_flush("\n[Model] Creating KAAdapterPythiaModel...")
    model = KAAdapterPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
        adapter_bottleneck=adapter_bottleneck,
    )
    param_info = model.num_parameters()
    print_flush(f"  Total parameters: {param_info['total']:,}")
    print_flush(f"  Adapter parameters: {param_info['adapter']:,}")

    results = {}

    # Phase 1: Train base model (or load pretrained)
    print_flush("\n" + "=" * 70)
    print_flush("PHASE 1: BASE MODEL TRAINING")
    print_flush("=" * 70)

    if pretrained_path:
        print_flush(f"  Loading pretrained model from {pretrained_path}...")
        model.load_state_dict(torch.load(pretrained_path))
        model = model.to(device)
        results["base_training"] = {"loaded_from": pretrained_path}
    else:
        base_result = train_base_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
        )
        model = base_result["model"]
        results["base_training"] = {
            "best_val_ppl": base_result["best_val_ppl"],
            "best_epoch": base_result["best_epoch"],
        }

    # Evaluate KV cache (baseline)
    print_flush("\n" + "=" * 70)
    print_flush("PHASE 2: BASELINE EVALUATION (KV Cache)")
    print_flush("=" * 70)

    print_flush("\n[KV Cache] Evaluating...")
    kv_result = evaluate_with_cache(model, val_loader, device, use_ka_adapter=False)
    print_flush(f"  PPL: {kv_result['ppl']:.1f}")
    print_flush(f"  Tokens/sec: {kv_result['tokens_per_sec']:.1f}")
    results["kv_cache"] = kv_result

    # Evaluate KA without adapter (案3 equivalent)
    print_flush("\n[KA Cache (no adapter)] Evaluating...")
    ka_no_adapter_result = evaluate_with_cache(model, val_loader, device, use_ka_adapter=True)
    print_flush(f"  PPL: {ka_no_adapter_result['ppl']:.1f}")
    print_flush(f"  Tokens/sec: {ka_no_adapter_result['tokens_per_sec']:.1f}")
    results["ka_no_adapter"] = ka_no_adapter_result

    # Phase 3: Train adapter (optional)
    if not skip_adapter_training:
        print_flush("\n" + "=" * 70)
        print_flush("PHASE 3: ADAPTER TRAINING")
        print_flush("=" * 70)

        adapter_result = train_adapter_only(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=adapter_epochs,
            lr=adapter_lr,
        )
        model = adapter_result["model"]
        results["adapter_training"] = {
            "best_val_ppl": adapter_result["best_val_ppl"],
            "best_epoch": adapter_result["best_epoch"],
        }

        # Evaluate KA with trained adapter
        print_flush("\n[KA Cache (with adapter)] Evaluating...")
        ka_with_adapter_result = evaluate_with_cache(model, val_loader, device, use_ka_adapter=True)
        print_flush(f"  PPL: {ka_with_adapter_result['ppl']:.1f}")
        print_flush(f"  Tokens/sec: {ka_with_adapter_result['tokens_per_sec']:.1f}")
        results["ka_with_adapter"] = ka_with_adapter_result

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush(f"\n| Method | PPL | Tokens/sec |")
    print_flush(f"|--------|-----|------------|")
    print_flush(f"| KV Cache (baseline) | {results['kv_cache']['ppl']:.1f} | {results['kv_cache']['tokens_per_sec']:.1f} |")
    print_flush(f"| KA Cache (no adapter) | {results['ka_no_adapter']['ppl']:.1f} | {results['ka_no_adapter']['tokens_per_sec']:.1f} |")

    if "ka_with_adapter" in results:
        print_flush(f"| KA Cache (with adapter) | {results['ka_with_adapter']['ppl']:.1f} | {results['ka_with_adapter']['tokens_per_sec']:.1f} |")

    # Analysis
    print_flush("\n" + "=" * 70)
    print_flush("ANALYSIS")
    print_flush("=" * 70)

    kv_ppl = results["kv_cache"]["ppl"]
    ka_no_ppl = results["ka_no_adapter"]["ppl"]
    diff_no_adapter = ((ka_no_ppl - kv_ppl) / kv_ppl) * 100

    print_flush(f"\nKA (no adapter) vs KV: {diff_no_adapter:+.1f}%")

    if "ka_with_adapter" in results:
        ka_with_ppl = results["ka_with_adapter"]["ppl"]
        diff_with_adapter = ((ka_with_ppl - kv_ppl) / kv_ppl) * 100
        improvement = ((ka_no_ppl - ka_with_ppl) / ka_no_ppl) * 100

        print_flush(f"KA (with adapter) vs KV: {diff_with_adapter:+.1f}%")
        print_flush(f"Adapter improvement: {improvement:+.1f}%")

    # Reversal Curse Evaluation
    print_flush("\n" + "=" * 70)
    print_flush("REVERSAL CURSE EVALUATION")
    print_flush("=" * 70)

    tokenizer = get_tokenizer(config.tokenizer_name)
    reversal_pairs = get_reversal_pairs()

    print_flush(f"\n  Evaluating {len(reversal_pairs)} pairs...")
    reversal_result = evaluate_reversal_curse(
        model, tokenizer, reversal_pairs, device
    )

    print_flush(f"\n  Forward PPL: {reversal_result['forward_ppl']:.1f}")
    print_flush(f"  Backward PPL: {reversal_result['backward_ppl']:.1f}")
    print_flush(f"  Reversal Ratio: {reversal_result['reversal_ratio']:.4f}")
    print_flush(f"  Reversal Gap: {reversal_result['reversal_gap']:.1f}")

    results["reversal_curse"] = reversal_result

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="KA Cache Adapter Experiment")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Base model epochs")
    parser.add_argument("--adapter-epochs", type=int, default=5, help="Adapter training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base model learning rate")
    parser.add_argument("--adapter-lr", type=float, default=1e-3, help="Adapter learning rate")
    parser.add_argument("--adapter-bottleneck", type=int, default=64, help="Adapter bottleneck dimension")
    parser.add_argument("--skip-adapter-training", action="store_true", help="Skip adapter training")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to pretrained model")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        adapter_epochs=args.adapter_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        adapter_lr=args.adapter_lr,
        adapter_bottleneck=args.adapter_bottleneck,
        skip_adapter_training=args.skip_adapter_training,
        pretrained_path=args.pretrained_path,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
