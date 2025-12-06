#!/usr/bin/env python3
"""
Layer 0 Distillation Script

既存Pythia-70MのLayer 0の出力をInfini Layerに蒸留する。

Usage:
    python3 scripts/distill_layer0.py --samples 5000 --epochs 10

    # ALiBi付き
    python3 scripts/distill_layer0.py --samples 5000 --epochs 10 --alibi
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer

from src.models.infini_adapter import create_pythia_with_infini
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import get_device


def load_pile_data(tokenizer, num_samples: int, seq_length: int):
    """Pileデータをロード"""
    from datasets import load_dataset

    print_flush(f"Loading {num_samples} samples from Pile...")

    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        text = example["text"]
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=seq_length,
            return_tensors="pt",
        )
        if tokens["input_ids"].size(1) >= seq_length // 2:
            samples.append(tokens["input_ids"].squeeze(0))

    print_flush(f"Loaded {len(samples)} samples")
    return samples


def train_distillation(
    model,
    samples: list,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    lr: float,
):
    """蒸留訓練"""
    optimizer = torch.optim.AdamW(model.infini_layer.parameters(), lr=lr)

    print_flush(f"\nDistillation Training:")
    print_flush(f"  Samples: {len(samples)}")
    print_flush(f"  Epochs: {num_epochs}")
    print_flush(f"  Batch size: {batch_size}")
    print_flush(f"  Learning rate: {lr}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.infini_layer.reset_memory(device)

        total_loss = 0.0
        num_batches = 0

        # Shuffle samples
        indices = torch.randperm(len(samples))

        for i in range(0, len(samples), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [samples[idx] for idx in batch_indices]

            # Pad batch
            max_len = max(s.size(0) for s in batch)
            padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            for j, s in enumerate(batch):
                padded[j, :s.size(0)] = s.to(device)

            optimizer.zero_grad()

            # Compute distillation loss
            loss = model.compute_distillation_loss(padded)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.infini_layer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time

        improved = avg_loss < best_loss
        if improved:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.infini_layer.state_dict().items()}
            marker = "*"
        else:
            marker = ""

        print_flush(f"  Epoch {epoch:2d}: loss={avg_loss:.6f} ({elapsed:.1f}s) {marker}")

    return best_loss, best_state


def evaluate_model(model, samples: list, device: torch.device, use_infini: bool):
    """モデルを評価（PPL）"""
    model.eval()
    model.use_infini_layer(use_infini)

    if use_infini:
        model.reset_memory()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sample in samples[:100]:  # Use subset for evaluation
            input_ids = sample.unsqueeze(0).to(device)
            labels = input_ids.clone()

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Layer 0 Distillation")

    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="Base model")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alibi", action="store_true", help="Use ALiBi")
    parser.add_argument("--output", default="distilled_layer0.pt", help="Output path")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("LAYER 0 DISTILLATION")
    print_flush("=" * 70)
    print_flush(f"Model: {args.model}")
    print_flush(f"Device: {device}")
    print_flush(f"ALiBi: {args.alibi}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    samples = load_pile_data(tokenizer, args.samples, args.seq_length)

    # Create model
    print_flush("\nCreating model...")
    model = create_pythia_with_infini(
        model_name=args.model,
        use_delta_rule=True,
        use_alibi=args.alibi,
        freeze_other_layers=True,
    )
    model = model.to(device)

    # Evaluate original model
    print_flush("\nEvaluating original model...")
    original_ppl = evaluate_model(model, samples, device, use_infini=False)
    print_flush(f"  Original Layer 0 PPL: {original_ppl:.1f}")

    # Train distillation
    print_flush("\n" + "=" * 70)
    best_loss, best_state = train_distillation(
        model=model,
        samples=samples,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Load best weights
    model.infini_layer.load_state_dict(best_state)

    # Evaluate with Infini Layer
    print_flush("\nEvaluating with Infini Layer...")
    infini_ppl = evaluate_model(model, samples, device, use_infini=True)
    print_flush(f"  Infini Layer PPL: {infini_ppl:.1f}")

    # Save
    print_flush(f"\nSaving to {args.output}...")
    torch.save({
        "infini_layer_state_dict": best_state,
        "config": {
            "hidden_size": model.config.hidden_size,
            "num_heads": model.config.num_attention_heads,
            "intermediate_size": model.config.intermediate_size,
            "use_alibi": args.alibi,
        },
        "distillation_loss": best_loss,
        "original_ppl": original_ppl,
        "infini_ppl": infini_ppl,
    }, args.output)

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush(f"| Model | PPL |")
    print_flush(f"|-------|-----|")
    print_flush(f"| Original Layer 0 | {original_ppl:.1f} |")
    print_flush(f"| Infini Layer (distilled) | {infini_ppl:.1f} |")
    print_flush(f"| Distillation Loss | {best_loss:.6f} |")

    ppl_diff = infini_ppl - original_ppl
    print_flush(f"\nPPL difference: {ppl_diff:+.1f}")

    if ppl_diff < 5:
        print_flush("Distillation successful! Ready for Phase 2 (long-context finetuning)")
    else:
        print_flush("Warning: Large PPL gap. Consider more epochs or hyperparameter tuning.")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
