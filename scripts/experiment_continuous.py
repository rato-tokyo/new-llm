#!/usr/bin/env python3
"""
Continuous Representation LM Experiment

仮説: トークン離散化による情報損失を検証
- Pythia baseline: 5000サンプルで事前に実験済み（PPL ~68）
- ContinuousLM: 同条件で訓練し比較

2つのモードで訓練・評価:
1. 通常モード (use_continuous=False): 標準的なTeacher Forcing
2. 連続モード (use_continuous=True): 前ステップの隠れ表現を入力に使用
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.pythia import PythiaConfig
from src.models import create_model, ContinuousLM
from src.utils.data_pythia import load_pile_tokens_cached
from src.utils.seed import set_seed
from src.utils.training import get_device
from src.utils.io import print_flush
from src.data.reversal_pairs import get_reversal_pairs
from src.utils.evaluation import evaluate_reversal_curse


def prepare_data(
    num_samples: int,
    seq_length: int,
    val_split: float = 0.1,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """データを準備"""
    print_flush(f"Loading {num_samples:,} samples...")

    tokens = load_pile_tokens_cached(
        num_tokens=num_samples * (seq_length + 1),
        tokenizer_name="EleutherAI/pythia-70m",
    )

    # Create samples
    inputs, labels = [], []
    for i in range(num_samples):
        start = i * seq_length
        inputs.append(tokens[start:start + seq_length])
        labels.append(tokens[start + 1:start + seq_length + 1])

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)

    # Split
    val_size = int(num_samples * val_split)
    train_inputs, val_inputs = inputs[:-val_size], inputs[-val_size:]
    train_labels, val_labels = labels[:-val_size], labels[-val_size:]

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs, val_labels),
        batch_size=batch_size,
    )

    print_flush(f"Train: {len(train_inputs):,}, Val: {len(val_inputs):,}")
    return train_loader, val_loader


def train_epoch_standard(
    model: ContinuousLM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """通常モードで1エポック訓練"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 通常モード
        logits, _ = model(input_ids, use_continuous=False)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

    return total_loss / total_tokens


def train_epoch_continuous(
    model: ContinuousLM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    continuous_ratio: float = 0.5,
) -> float:
    """
    連続モードで1エポック訓練

    Args:
        continuous_ratio: 連続モードを使用する確率
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        batch_size, seq_len = input_ids.shape

        optimizer.zero_grad()

        # ランダムに連続モードを使用するか決定
        use_continuous = torch.rand(1).item() < continuous_ratio

        if use_continuous and seq_len > 1:
            # 最初のトークンで隠れ表現を取得
            with torch.no_grad():
                _, prev_hidden = model(input_ids[:, :1], use_continuous=False)

            # 残りを連続モードで処理
            logits, _ = model(
                input_ids,
                use_continuous=True,
                prev_hidden=prev_hidden,
            )
        else:
            logits, _ = model(input_ids, use_continuous=False)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(
    model: ContinuousLM,
    val_loader: DataLoader,
    device: torch.device,
    use_continuous: bool = False,
) -> float:
    """評価"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in val_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if use_continuous:
            # 最初のトークンで隠れ表現を取得
            _, prev_hidden = model(input_ids[:, :1], use_continuous=False)
            logits, _ = model(
                input_ids,
                use_continuous=True,
                prev_hidden=prev_hidden,
            )
        else:
            logits, _ = model(input_ids, use_continuous=False)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Continuous LM Experiment")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mode", choices=["standard", "continuous", "both"], default="both")
    parser.add_argument("--continuous-ratio", type=float, default=0.5,
                        help="Ratio of continuous mode usage in training")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("CONTINUOUS LM EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples:,}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Mode: {args.mode}")
    print_flush(f"Continuous ratio: {args.continuous_ratio}")
    print_flush("=" * 70)

    # Load data
    train_loader, val_loader = prepare_data(
        num_samples=args.samples,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
    )

    # Create model
    print_flush("\n[Model] Creating ContinuousLM...")
    model = create_model("continuous")
    model = model.to(device)

    params = model.num_parameters()
    print_flush(f"Total parameters: {params['total']:,}")
    print_flush(f"  - Embedding: {params['embedding']:,}")
    print_flush(f"  - Hidden proj: {params['hidden_proj']:,}")
    print_flush(f"  - Transformer: {params['transformer']:,}")
    print_flush(f"  - LM head: {params['lm_head']:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training
    print_flush("\n[Training]")
    best_ppl = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        if args.mode == "standard":
            train_loss = train_epoch_standard(model, train_loader, optimizer, device)
        elif args.mode == "continuous":
            train_loss = train_epoch_continuous(
                model, train_loader, optimizer, device,
                continuous_ratio=args.continuous_ratio,
            )
        else:  # both
            # 最初は通常モードで訓練し、徐々に連続モードを増やす
            ratio = min(args.continuous_ratio, epoch / args.epochs)
            train_loss = train_epoch_continuous(
                model, train_loader, optimizer, device,
                continuous_ratio=ratio,
            )

        # Evaluate in both modes
        ppl_standard = evaluate(model, val_loader, device, use_continuous=False)
        ppl_continuous = evaluate(model, val_loader, device, use_continuous=True)

        epoch_time = time.time() - epoch_start

        if ppl_standard < best_ppl:
            best_ppl = ppl_standard

        print_flush(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"PPL(std): {ppl_standard:.2f} | "
            f"PPL(cont): {ppl_continuous:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )

    total_time = time.time() - start_time

    # Final evaluation
    print_flush("\n" + "=" * 70)
    print_flush("FINAL RESULTS")
    print_flush("=" * 70)

    final_ppl_std = evaluate(model, val_loader, device, use_continuous=False)
    final_ppl_cont = evaluate(model, val_loader, device, use_continuous=True)

    print_flush(f"Final PPL (standard mode):   {final_ppl_std:.2f}")
    print_flush(f"Final PPL (continuous mode): {final_ppl_cont:.2f}")
    print_flush(f"Best PPL (standard):         {best_ppl:.2f}")

    # Compare with Pythia baseline
    print_flush("\n[Comparison with Pythia Baseline]")
    print_flush("Pythia baseline (5000 samples): PPL ~68")
    print_flush(f"ContinuousLM (standard):        PPL {final_ppl_std:.2f}")
    print_flush(f"ContinuousLM (continuous):      PPL {final_ppl_cont:.2f}")

    diff_std = final_ppl_std - 68
    diff_cont = final_ppl_cont - 68
    print_flush(f"Difference (standard):   {diff_std:+.2f}")
    print_flush(f"Difference (continuous): {diff_cont:+.2f}")

    # Reversal Curse evaluation
    print_flush("\n[Reversal Curse Evaluation]")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    pairs = get_reversal_pairs()

    # Need to wrap model for evaluation
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            logits, _ = self.model(input_ids, use_continuous=False)
            return logits

    wrapped = ModelWrapper(model)
    rev_results = evaluate_reversal_curse(wrapped, tokenizer, pairs, device)

    print_flush(f"Forward PPL:    {rev_results['forward_ppl']:.2f}")
    print_flush(f"Backward PPL:   {rev_results['backward_ppl']:.2f}")
    print_flush(f"Reversal Gap:   {rev_results['reversal_gap']:+.2f}")

    print_flush(f"\nTotal training time: {total_time/60:.1f} min")
    print_flush("\nDONE")


if __name__ == "__main__":
    main()
