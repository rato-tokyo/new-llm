#!/usr/bin/env python3
"""
Parallel Infini Adapter Training Script

既存Pythia-70MにInfini Adapterを並列挿入し、長文データで訓練。
Layer 0を置き換えるのではなく、並列に追加することで既存性能を維持。

Architecture:
    Input Embedding
          ↓
    ┌─────┴─────┐
    │  Original │  Infini Adapter
    │  Layer 0  │  (Memory)
    └─────┬─────┘      │
          ↓            ↓
        Output + α × Infini_Output
          ↓
    Layer 1-5 (unchanged)

Usage:
    # Basic training
    python3 scripts/train_parallel_adapter.py --num-docs 100 --epochs 50

    # With ALiBi
    python3 scripts/train_parallel_adapter.py --num-docs 100 --epochs 50 --alibi

    # Custom initial alpha
    python3 scripts/train_parallel_adapter.py --initial-alpha 0.1
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer

from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE
from src.models.infini_adapter import create_pythia_with_parallel_infini
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import get_device


def load_long_documents(tokenizer, num_docs: int, tokens_per_doc: int):
    """長文データをロード"""
    from datasets import load_dataset

    print_flush(f"Loading {num_docs} long documents ({tokens_per_doc} tokens each)...")

    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    documents = []
    current_tokens = []

    for example in dataset:
        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        current_tokens.extend(tokens)

        while len(current_tokens) >= tokens_per_doc:
            doc = current_tokens[:tokens_per_doc]
            documents.append(torch.tensor(doc, dtype=torch.long))
            current_tokens = current_tokens[tokens_per_doc:]

            if len(documents) >= num_docs:
                break

        if len(documents) >= num_docs:
            break

    print_flush(f"Loaded {len(documents)} documents")
    return documents


def evaluate_model(model, documents: list, device: torch.device, segment_length: int):
    """モデルを評価（PPL）"""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for doc in documents:
            doc = doc.to(device)
            model.reset_memory()

            for start in range(0, len(doc) - 1, segment_length):
                end = min(start + segment_length, len(doc))
                segment = doc[start:end]

                if len(segment) < 2:
                    continue

                input_ids = segment[:-1].unsqueeze(0)
                labels = segment[1:].unsqueeze(0)

                outputs = model(input_ids, labels=labels)
                loss = outputs.loss

                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def train_parallel_adapter(
    model,
    train_docs: list,
    val_docs: list,
    device: torch.device,
    num_epochs: int,
    segment_length: int,
    lr: float,
    patience: int = EARLY_STOPPING_PATIENCE,
):
    """Parallel Adapterを訓練"""
    # Only train Infini Adapter parameters
    trainable_params = list(model.infini_adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print_flush(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    print_flush(f"\nParallel Adapter Training:")
    print_flush(f"  Train documents: {len(train_docs)}")
    print_flush(f"  Val documents: {len(val_docs)}")
    print_flush(f"  Epochs: {num_epochs}")
    print_flush(f"  Segment length: {segment_length}")
    print_flush(f"  Learning rate: {lr}")
    print_flush(f"  Early stopping patience: {patience}")

    best_val_ppl = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Training
        model.train()
        total_loss = 0.0
        total_tokens = 0

        # Shuffle documents
        indices = torch.randperm(len(train_docs))

        for doc_idx in indices:
            doc = train_docs[doc_idx].to(device)

            # Reset memory for each document
            model.reset_memory()

            # Process document in segments
            for start in range(0, len(doc) - 1, segment_length):
                end = min(start + segment_length, len(doc))
                segment = doc[start:end]

                if len(segment) < 2:
                    continue

                input_ids = segment[:-1].unsqueeze(0)
                labels = segment[1:].unsqueeze(0)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()

        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

        # Validation
        val_ppl = evaluate_model(model, val_docs, device, segment_length)

        elapsed = time.time() - start_time
        alpha = model.get_alpha()

        # Check for improvement
        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_state = {
                k: v.cpu().clone() for k, v in model.infini_adapter.state_dict().items()
            }
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"alpha={alpha:.4f} ({elapsed:.1f}s) {marker}"
        )

        # Early stopping
        if patience_counter >= patience:
            print_flush(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    return best_val_ppl, best_state


def main():
    parser = argparse.ArgumentParser(description="Parallel Infini Adapter Training")

    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="Base model")
    parser.add_argument("--num-docs", type=int, default=100, help="Number of documents")
    parser.add_argument("--tokens-per-doc", type=int, default=4096, help="Tokens per document")
    parser.add_argument("--seq-length", type=int, default=256, help="Segment length")
    parser.add_argument("--epochs", type=int, default=50, help="Max number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience"
    )
    parser.add_argument("--initial-alpha", type=float, default=0.0, help="Initial alpha value")
    parser.add_argument("--alibi", action="store_true", help="Use ALiBi")
    parser.add_argument("--output", default="parallel_adapter.pt", help="Output path")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("PARALLEL INFINI ADAPTER TRAINING")
    print_flush("=" * 70)
    print_flush(f"Model: {args.model}")
    print_flush(f"Device: {device}")
    print_flush(f"ALiBi: {args.alibi}")
    print_flush(f"Initial alpha: {args.initial_alpha}")
    print_flush(f"Tokens per document: {args.tokens_per_doc}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load documents
    documents = load_long_documents(tokenizer, args.num_docs, args.tokens_per_doc)

    # Split into train/val
    val_size = max(10, len(documents) // 10)
    train_docs = documents[:-val_size]
    val_docs = documents[-val_size:]
    print_flush(f"Train: {len(train_docs)}, Val: {len(val_docs)}")

    # Create model
    print_flush("\nCreating model with Parallel Infini Adapter...")
    model = create_pythia_with_parallel_infini(
        model_name=args.model,
        use_delta_rule=True,
        use_alibi=args.alibi,
        initial_alpha=args.initial_alpha,
        freeze_base_model=True,
    )
    model = model.to(device)

    # Evaluate before training (should be close to original Pythia)
    print_flush("\nEvaluating before training...")
    pre_ppl = evaluate_model(model, val_docs, device, args.seq_length)
    print_flush(f"  PPL (before training): {pre_ppl:.1f}")
    print_flush(f"  Alpha: {model.get_alpha():.4f}")

    # Train
    print_flush("\n" + "=" * 70)
    best_ppl, best_state = train_parallel_adapter(
        model=model,
        train_docs=train_docs,
        val_docs=val_docs,
        device=device,
        num_epochs=args.epochs,
        segment_length=args.seq_length,
        lr=args.lr,
        patience=args.patience,
    )

    # Load best weights
    model.infini_adapter.load_state_dict(best_state)

    # Evaluate after training
    print_flush("\nEvaluating after training...")
    post_ppl = evaluate_model(model, val_docs, device, args.seq_length)
    print_flush(f"  PPL (after training): {post_ppl:.1f}")
    print_flush(f"  Final alpha: {model.get_alpha():.4f}")

    # Save
    print_flush(f"\nSaving to {args.output}...")
    save_dict = {
        "infini_adapter_state_dict": best_state,
        "config": {
            "hidden_size": model.config.hidden_size,
            "num_heads": model.config.num_attention_heads,
            "use_alibi": args.alibi,
            "initial_alpha": args.initial_alpha,
        },
        "pre_training_ppl": pre_ppl,
        "post_training_ppl": post_ppl,
        "final_alpha": model.get_alpha(),
    }

    # Also save memory state as example
    model.reset_memory()
    doc = train_docs[0].to(device)
    for start in range(0, len(doc) - 1, args.seq_length):
        end = min(start + args.seq_length, len(doc))
        segment = doc[start:end]
        if len(segment) >= 2:
            with torch.no_grad():
                _ = model(segment[:-1].unsqueeze(0))
    save_dict["example_memory_state"] = model.get_memory_state()

    torch.save(save_dict, args.output)

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush("| Metric | Value |")
    print_flush("|--------|-------|")
    print_flush(f"| PPL (before training) | {pre_ppl:.1f} |")
    print_flush(f"| PPL (after training) | {post_ppl:.1f} |")
    print_flush(f"| Improvement | {pre_ppl - post_ppl:+.1f} |")
    print_flush(f"| Final alpha | {model.get_alpha():.4f} |")

    if post_ppl < pre_ppl:
        print_flush("\nTraining successful! Parallel adapter improved performance.")
    else:
        print_flush("\nWarning: No improvement. Try more epochs or different learning rate.")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
