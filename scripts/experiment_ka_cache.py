#!/usr/bin/env python3
"""
KA Cache Inference Experiment

KVキャッシュとKAキャッシュの推論時性能を比較する実験。
学習は不要で、同じ訓練済みモデルを使用して推論のみを比較。

Usage:
    # 両方を比較
    python3 scripts/experiment_ka_cache.py --samples 1000 --epochs 10

    # KAキャッシュのみ（baselineスキップ）
    python3 scripts/experiment_ka_cache.py --samples 1000 --skip-baseline

    # KVキャッシュのみ
    python3 scripts/experiment_ka_cache.py --samples 1000 --skip-ka
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
from src.models.ka_cache import KACachePythiaModel
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import prepare_data_loaders, get_device, get_tokenizer
from src.utils.device import clear_gpu_cache
from src.utils.evaluation import evaluate_reversal_curse
from src.data.reversal_pairs import get_reversal_pairs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    model_name: str,
) -> Dict[str, Any]:
    """Train model and return best checkpoint"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_ppl = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print_flush(f"\n  Training {model_name}...")

    for epoch in range(1, num_epochs + 1):
        # Train
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

    # Restore best model
    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
    }


@torch.no_grad()
def evaluate_with_cache(
    model: KACachePythiaModel,
    val_loader: DataLoader,
    device: torch.device,
    use_ka_cache: bool,
    cache_name: str,
) -> Dict[str, float]:
    """Evaluate model using KV or KA cache for inference"""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_time = 0.0

    for batch in val_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        batch_size, seq_len = input_ids.shape

        # Simulate autoregressive inference with cache
        start_time = time.time()

        cache = None
        all_logits = []

        for t in range(seq_len):
            if t == 0:
                # First token: no cache
                token_input = input_ids[:, :1]
            else:
                # Subsequent tokens: use cache
                token_input = input_ids[:, t:t+1]

            logits, cache = model.forward_with_cache(
                token_input, cache, use_ka_cache=use_ka_cache
            )
            all_logits.append(logits)

        elapsed = time.time() - start_time
        total_time += elapsed

        # Compute loss
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
        "cache_type": cache_name,
    }


@torch.no_grad()
def evaluate_generation_quality(
    model: KACachePythiaModel,
    tokenizer: Any,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 30,
) -> Dict[str, List[str]]:
    """Compare generation quality between KV and KA cache"""
    model.eval()

    results = {"prompts": prompts, "kv_outputs": [], "ka_outputs": []}

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate with KV cache
        kv_output = model.generate(
            input_ids.clone(),
            max_new_tokens=max_new_tokens,
            use_ka_cache=False,
            temperature=0.7,
        )
        kv_text = tokenizer.decode(kv_output[0], skip_special_tokens=True)
        results["kv_outputs"].append(kv_text)

        # Generate with KA cache
        ka_output = model.generate(
            input_ids.clone(),
            max_new_tokens=max_new_tokens,
            use_ka_cache=True,
            temperature=0.7,
        )
        ka_text = tokenizer.decode(ka_output[0], skip_special_tokens=True)
        results["ka_outputs"].append(ka_text)

    return results


def run_experiment(
    num_samples: int = 5000,
    seq_length: int = 128,
    num_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    skip_baseline: bool = False,
    skip_ka: bool = False,
) -> Dict[str, Any]:
    """Run KA cache inference experiment"""
    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("KA CACHE INFERENCE EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"Skip baseline (KV): {skip_baseline}")
    print_flush(f"Skip KA cache: {skip_ka}")
    print_flush("=" * 70)

    # Prepare data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
        include_reversal_pairs=False,  # Not needed for this experiment
    )

    # Create model
    print_flush("\n[Model] Creating KACachePythiaModel...")
    model = KACachePythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
    )
    print_flush(f"  Parameters: {model.num_parameters()['total']:,}")

    # Train model (same model for both cache types)
    print_flush("\n" + "=" * 70)
    print_flush("TRAINING")
    print_flush("=" * 70)

    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        model_name="KACachePythia",
    )
    model = train_result["model"]

    results = {
        "train_ppl": train_result["best_val_ppl"],
        "train_epoch": train_result["best_epoch"],
    }

    # Evaluate with different cache types
    print_flush("\n" + "=" * 70)
    print_flush("INFERENCE EVALUATION")
    print_flush("=" * 70)

    # KV Cache (baseline)
    if not skip_baseline:
        print_flush("\n[KV Cache] Evaluating...")
        kv_result = evaluate_with_cache(
            model, val_loader, device, use_ka_cache=False, cache_name="KV"
        )
        print_flush(f"  PPL: {kv_result['ppl']:.1f}")
        print_flush(f"  Tokens/sec: {kv_result['tokens_per_sec']:.1f}")
        print_flush(f"  Total time: {kv_result['total_time']:.2f}s")
        results["kv_cache"] = kv_result

    # KA Cache
    if not skip_ka:
        print_flush("\n[KA Cache] Evaluating...")
        ka_result = evaluate_with_cache(
            model, val_loader, device, use_ka_cache=True, cache_name="KA"
        )
        print_flush(f"  PPL: {ka_result['ppl']:.1f}")
        print_flush(f"  Tokens/sec: {ka_result['tokens_per_sec']:.1f}")
        print_flush(f"  Total time: {ka_result['total_time']:.2f}s")
        results["ka_cache"] = ka_result

    # Generation comparison
    if not skip_baseline and not skip_ka:
        print_flush("\n" + "=" * 70)
        print_flush("GENERATION COMPARISON")
        print_flush("=" * 70)

        tokenizer = get_tokenizer(config.tokenizer_name)
        prompts = [
            "The capital of France is",
            "In the year 2024,",
            "The quick brown fox",
        ]

        gen_results = evaluate_generation_quality(
            model, tokenizer, prompts, device, max_new_tokens=20
        )

        for i, prompt in enumerate(prompts):
            print_flush(f"\n  Prompt: \"{prompt}\"")
            print_flush(f"  KV: \"{gen_results['kv_outputs'][i]}\"")
            print_flush(f"  KA: \"{gen_results['ka_outputs'][i]}\"")

        results["generation"] = gen_results

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

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush(f"\nTraining PPL: {results['train_ppl']:.1f} (epoch {results['train_epoch']})")

    if not skip_baseline and not skip_ka:
        kv_ppl = results["kv_cache"]["ppl"]
        ka_ppl = results["ka_cache"]["ppl"]
        ppl_diff = ka_ppl - kv_ppl
        ppl_diff_pct = (ppl_diff / kv_ppl) * 100

        print_flush(f"\n| Cache Type | PPL | Tokens/sec |")
        print_flush(f"|------------|-----|------------|")
        print_flush(f"| KV (baseline) | {kv_ppl:.1f} | {results['kv_cache']['tokens_per_sec']:.1f} |")
        print_flush(f"| KA | {ka_ppl:.1f} | {results['ka_cache']['tokens_per_sec']:.1f} |")
        print_flush(f"\nPPL Difference: {ppl_diff:+.1f} ({ppl_diff_pct:+.1f}%)")

        # Analysis
        print_flush("\n" + "=" * 70)
        print_flush("ANALYSIS")
        print_flush("=" * 70)

        if abs(ppl_diff_pct) < 5:
            print_flush("KA cache achieves comparable performance to KV cache.")
            print_flush("This suggests KA cache is a viable alternative for inference.")
        elif ppl_diff_pct > 0:
            print_flush(f"KA cache shows {ppl_diff_pct:.1f}% higher PPL than KV cache.")
            print_flush("This degradation may be acceptable depending on use case.")
        else:
            print_flush(f"KA cache shows {-ppl_diff_pct:.1f}% lower PPL than KV cache.")
            print_flush("This is unexpected - worth investigating further.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="KA Cache Inference Experiment")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip KV cache baseline")
    parser.add_argument("--skip-ka", action="store_true", help="Skip KA cache evaluation")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        skip_baseline=args.skip_baseline,
        skip_ka=args.skip_ka,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
