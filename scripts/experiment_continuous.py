#!/usr/bin/env python3
"""
Continuous Representation LM Experiment

仮説: トークン離散化による情報損失を検証
- Pythia baseline: 同条件で訓練（比較用）
- ContinuousLM: 前ステップの隠れ表現を入力に使用

2つのモードで訓練・評価:
1. 通常モード (use_continuous=False): 標準的なTeacher Forcing
2. 連続モード (use_continuous=True): 前ステップの隠れ表現を入力に使用

Usage:
    # Pythia と ContinuousLM の両方を実行
    python3 scripts/experiment_continuous.py

    # ContinuousLM のみ
    python3 scripts/experiment_continuous.py --models continuous

    # Pythia のみ
    python3 scripts/experiment_continuous.py --models pythia

    # 設定カスタマイズ
    python3 scripts/experiment_continuous.py --samples 10000 --epochs 50 --patience 3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.pythia import PythiaConfig
from src.models import create_model, ContinuousLM, TransformerLM
from src.data.reversal_pairs import get_reversal_pairs
from src.utils.device import clear_gpu_cache
from src.utils.evaluation import evaluate_reversal_curse
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.training import get_device, prepare_data_loaders
from src.utils.io import print_flush


# ============================================================================
# Training functions
# ============================================================================

def train_epoch_pythia(
    model: TransformerLM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> float:
    """Pythia（標準モデル）を1エポック訓練"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="sum",
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        total_tokens += labels.numel()

    return total_loss / total_tokens


def train_epoch_continuous(
    model: ContinuousLM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    continuous_ratio: float = 0.5,
    gradient_clip: float = 1.0,
) -> float:
    """
    ContinuousLMを1エポック訓練

    Args:
        continuous_ratio: 連続モードを使用する確率
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in train_loader:
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
            reduction="sum",
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        total_tokens += labels.numel()

    return total_loss / total_tokens


# ============================================================================
# Evaluation functions
# ============================================================================

@torch.no_grad()
def evaluate_pythia(
    model: TransformerLM,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Pythiaを評価してPPLを返す"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in val_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)

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


@torch.no_grad()
def evaluate_continuous(
    model: ContinuousLM,
    val_loader: DataLoader,
    device: torch.device,
    use_continuous: bool = False,
) -> float:
    """ContinuousLMを評価してPPLを返す"""
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


# ============================================================================
# Model wrapper for reversal curse evaluation
# ============================================================================

class ContinuousModelWrapper(nn.Module):
    """ContinuousLMをTransformerLMと同じインターフェースでラップ"""
    def __init__(self, model: ContinuousLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits, _ = self.model(input_ids, use_continuous=False)
        return logits


# ============================================================================
# Training with early stopping
# ============================================================================

def train_pythia_with_early_stopping(
    model: TransformerLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    gradient_clip: float = 1.0,
) -> tuple[float, int, Optional[dict]]:
    """
    Pythiaを訓練（Early stopping付き）

    Returns:
        best_val_ppl, best_epoch, best_state_dict
    """
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch_pythia(
            model, train_loader, optimizer, device, gradient_clip
        )
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Validate
        val_ppl = evaluate_pythia(model, val_loader, device)

        elapsed = time.time() - start_time

        # Check improvement
        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:7.1f}, val_ppl={val_ppl:7.1f} "
            f"({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


def train_continuous_with_early_stopping(
    model: ContinuousLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    continuous_ratio: float = 0.5,
    gradient_clip: float = 1.0,
) -> tuple[float, int, Optional[dict]]:
    """
    ContinuousLMを訓練（Early stopping付き）

    Returns:
        best_val_ppl, best_epoch, best_state_dict
    """
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # 訓練時の連続モード比率を徐々に上げる
        current_ratio = min(continuous_ratio, epoch / num_epochs * continuous_ratio * 2)

        # Train
        train_loss = train_epoch_continuous(
            model, train_loader, optimizer, device, current_ratio, gradient_clip
        )
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Validate in both modes
        val_ppl_std = evaluate_continuous(model, val_loader, device, use_continuous=False)
        val_ppl_cont = evaluate_continuous(model, val_loader, device, use_continuous=True)

        elapsed = time.time() - start_time

        # Use standard mode PPL for early stopping (fair comparison with Pythia)
        improved = val_ppl_std < best_val_ppl
        if improved:
            best_val_ppl = val_ppl_std
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train={train_ppl:7.1f}, val(std)={val_ppl_std:7.1f}, "
            f"val(cont)={val_ppl_cont:7.1f} ({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


# ============================================================================
# Single model experiment
# ============================================================================

def run_pythia_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: PythiaConfig,
    num_epochs: int,
    patience: int,
    lr: float,
) -> dict[str, Any]:
    """Pythia実験を実行"""
    print_flush("\n" + "=" * 70)
    print_flush("PYTHIA BASELINE")
    print_flush("=" * 70)

    model = create_model("pythia", config)
    model = model.to(device)

    params = model.num_parameters()
    print_flush(f"  Parameters: {params['total']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print_flush("\n  Training...")
    best_ppl, best_epoch, best_state = train_pythia_with_early_stopping(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, patience
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Reversal Curse evaluation
    print_flush("\n  Reversal Curse:")
    tokenizer = get_tokenizer(config.tokenizer_name)
    pairs = get_reversal_pairs()
    reversal = evaluate_reversal_curse(model, tokenizer, pairs, device)
    print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
    print_flush(f"    Gap: {reversal['reversal_gap']:+.1f}")

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "reversal_curse": reversal,
    }

    del model
    clear_gpu_cache(device)

    return result


def run_continuous_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: PythiaConfig,
    num_epochs: int,
    patience: int,
    lr: float,
    continuous_ratio: float,
) -> dict[str, Any]:
    """ContinuousLM実験を実行"""
    print_flush("\n" + "=" * 70)
    print_flush("CONTINUOUS LM")
    print_flush("=" * 70)

    model = create_model("continuous", config)
    model = model.to(device)

    params = model.num_parameters()
    print_flush(f"  Parameters: {params['total']:,}")
    print_flush(f"    - hidden_proj: {params['hidden_proj']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print_flush("\n  Training...")
    best_ppl, best_epoch, best_state = train_continuous_with_early_stopping(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, patience, continuous_ratio
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation in both modes
    print_flush("\n  Final Evaluation:")
    final_ppl_std = evaluate_continuous(model, val_loader, device, use_continuous=False)
    final_ppl_cont = evaluate_continuous(model, val_loader, device, use_continuous=True)
    print_flush(f"    Standard mode PPL: {final_ppl_std:.1f}")
    print_flush(f"    Continuous mode PPL: {final_ppl_cont:.1f}")

    # Reversal Curse evaluation
    print_flush("\n  Reversal Curse:")
    tokenizer = get_tokenizer(config.tokenizer_name)
    pairs = get_reversal_pairs()
    wrapped = ContinuousModelWrapper(model)
    reversal = evaluate_reversal_curse(wrapped, tokenizer, pairs, device)
    print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
    print_flush(f"    Gap: {reversal['reversal_gap']:+.1f}")

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "final_ppl_standard": final_ppl_std,
        "final_ppl_continuous": final_ppl_cont,
        "reversal_curse": reversal,
    }

    del model, wrapped
    clear_gpu_cache(device)

    return result


# ============================================================================
# Main
# ============================================================================

def print_summary(results: dict[str, Any]) -> None:
    """結果サマリーを出力"""
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    # PPL comparison
    print_flush("\n| Model | Best PPL | Epoch |")
    print_flush("|-------|----------|-------|")

    if "pythia" in results:
        r = results["pythia"]
        print_flush(f"| Pythia | {r['best_val_ppl']:.1f} | {r['best_epoch']} |")

    if "continuous" in results:
        r = results["continuous"]
        print_flush(f"| ContinuousLM | {r['best_val_ppl']:.1f} | {r['best_epoch']} |")
        print_flush(f"|   (std mode) | {r['final_ppl_standard']:.1f} | - |")
        print_flush(f"|   (cont mode) | {r['final_ppl_continuous']:.1f} | - |")

    # Reversal Curse comparison
    print_flush("\n| Model | Forward PPL | Backward PPL | Gap |")
    print_flush("|-------|-------------|--------------|-----|")

    for name in ["pythia", "continuous"]:
        if name in results:
            r = results[name]
            rev = r["reversal_curse"]
            display_name = "Pythia" if name == "pythia" else "ContinuousLM"
            print_flush(
                f"| {display_name} | {rev['forward_ppl']:.1f} | "
                f"{rev['backward_ppl']:.1f} | {rev['reversal_gap']:+.1f} |"
            )

    # Comparison
    if "pythia" in results and "continuous" in results:
        pythia_ppl = results["pythia"]["best_val_ppl"]
        cont_ppl = results["continuous"]["best_val_ppl"]
        diff = cont_ppl - pythia_ppl
        print_flush(f"\nContinuousLM vs Pythia: {diff:+.1f} PPL")


def main():
    parser = argparse.ArgumentParser(description="Continuous LM Experiment")
    parser.add_argument(
        "--models", nargs="+", default=["pythia", "continuous"],
        choices=["pythia", "continuous"],
        help="Models to run (default: both)"
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument(
        "--continuous-ratio", type=float, default=0.5,
        help="Ratio of continuous mode usage in training"
    )
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("CONTINUOUS LM EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Models: {args.models}")
    print_flush(f"Samples: {args.samples:,}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Learning rate: {args.lr}")
    print_flush(f"Patience: {args.patience}")
    print_flush(f"Continuous ratio: {args.continuous_ratio}")
    print_flush("=" * 70)

    # Load data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=args.samples,
        seq_length=args.seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=args.batch_size,
    )

    # Run experiments
    start_time = time.time()
    results: dict[str, Any] = {}

    if "pythia" in args.models:
        results["pythia"] = run_pythia_experiment(
            train_loader, val_loader, device, config,
            args.epochs, args.patience, args.lr
        )

    if "continuous" in args.models:
        results["continuous"] = run_continuous_experiment(
            train_loader, val_loader, device, config,
            args.epochs, args.patience, args.lr, args.continuous_ratio
        )

    total_time = time.time() - start_time

    # Summary
    print_summary(results)

    print_flush(f"\nTotal time: {total_time/60:.1f} min")
    print_flush("\nDONE")


if __name__ == "__main__":
    main()
