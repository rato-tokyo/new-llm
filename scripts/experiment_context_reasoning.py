#!/usr/bin/env python3
"""
Reversal Curse 汎化性能実験

仮説:
  Reversal Curseの本質は「汎化性能の低さ」である。
  パターン学習ペアで学んだ逆方向推論を、Valペアに汎化できるかを検証。

設計:
  - パターン学習ペア: 順方向・逆方向の両方を学習
  - Valペア: 順方向のみ学習 → 逆方向で評価（汎化テスト）

比較:
  - Baseline: 全文を学習（丸暗記傾向）
  - Modified: コンテキスト分離学習（推論パターン抽出）
"""

import argparse
import sys
import time
from typing import Optional

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import PythiaConfig, GRADIENT_CLIP
from src.data.family_relations import (
    FamilyPair,
    create_baseline_pattern_samples,
    create_baseline_val_samples,
    create_modified_pattern_samples,
    create_modified_val_samples,
    generate_family_pairs,
    split_pairs_for_experiment,
)
from src.models import create_model
from src.utils.device import clear_gpu_cache
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.data_pythia import load_pile_tokens_cached
from src.utils.training import get_device


# =============================================================================
# 統一データセット
# =============================================================================


class UnifiedDataset(Dataset):
    """
    統一データセット: Pile + Family samples

    Baseline/Modified両方で同じPileサンプル数を使用するため、
    num_pile_samplesを明示的に指定する。
    """

    def __init__(
        self,
        pile_samples: list[dict],  # 事前に作成されたPileサンプル
        family_samples: list[dict],  # Familyサンプル
        tokenizer,
        seq_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.eos_token_id

        self.pile_samples = pile_samples
        self.family_samples = family_samples
        self.all_samples = self.pile_samples + self.family_samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]

        if sample["type"] == "pile":
            tokens = sample["tokens"]
            if len(tokens) > self.seq_length:
                tokens = tokens[: self.seq_length]
            input_ids = torch.tensor(tokens, dtype=torch.long)
            labels = input_ids.clone()
            return input_ids, labels

        # Family sample
        if "context_len" in sample:
            # Modified: コンテキスト分離
            tokens = sample["tokens"]
            context_len = sample["context_len"]

            if len(tokens) > self.seq_length:
                tokens = tokens[: self.seq_length]
                context_len = min(context_len, self.seq_length - 1)
            else:
                tokens = tokens + [self.pad_token_id] * (self.seq_length - len(tokens))

            input_ids = torch.tensor(tokens, dtype=torch.long)
            labels = input_ids.clone()
            labels[:context_len] = -100
            return input_ids, labels
        else:
            # Baseline: 全文学習
            tokens = sample["tokens"]
            if len(tokens) > self.seq_length:
                tokens = tokens[: self.seq_length]
            else:
                tokens = tokens + [self.pad_token_id] * (self.seq_length - len(tokens))

            input_ids = torch.tensor(tokens, dtype=torch.long)
            labels = input_ids.clone()
            return input_ids, labels


def create_pile_samples(
    pile_tokens: torch.Tensor,
    seq_length: int,
    num_samples: int,
) -> list[dict]:
    """Pileサンプルを作成（数を明示的に指定）"""
    samples: list[dict] = []
    for i in range(0, len(pile_tokens) - seq_length, seq_length):
        if len(samples) >= num_samples:
            break
        tokens = pile_tokens[i : i + seq_length].tolist()
        samples.append({"type": "pile", "tokens": tokens})
    return samples


def create_baseline_family_samples(
    pattern_samples: list[dict],
    val_samples: list[dict],
    tokenizer,
) -> list[dict]:
    """Baseline用Familyサンプルを作成"""
    family_samples = []

    for sample in pattern_samples + val_samples:
        tokens = tokenizer.encode(sample["text"])
        family_samples.append({
            "type": sample["type"],
            "tokens": tokens,
        })

    return family_samples


def create_modified_family_samples(
    pattern_samples: list[dict],
    val_samples: list[dict],
    tokenizer,
) -> list[dict]:
    """Modified用Familyサンプルを作成"""
    family_samples = []

    # コンテキスト分離サンプル（context部分のみマスク）
    for sample in pattern_samples:
        context = sample["context"]
        question = sample["question"]
        answer = sample["answer"]

        # context部分のみマスク（question + answerは学習対象）
        full_text = f"{context} {question}{answer}"
        # 注意: context + " " の長さを取得（スペースも含む）
        context_only = f"{context} "

        full_tokens = tokenizer.encode(full_text)
        context_len = len(tokenizer.encode(context_only))

        family_samples.append({
            "type": sample["type"],
            "tokens": full_tokens,
            "context_len": context_len,
        })

    # Valサンプル（全文学習、Baselineと同一）
    for sample in val_samples:
        tokens = tokenizer.encode(sample["text"])
        family_samples.append({
            "type": sample["type"],
            "tokens": tokens,
        })

    return family_samples


# =============================================================================
# 訓練・評価関数
# =============================================================================


def compute_ppl(model, data_loader, device) -> float:
    """PPL計算"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            mask = shift_labels != -100
            if mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += mask.sum().item()

    if total_tokens == 0:
        return float("inf")
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def compute_target_loss(
    model,
    prompt: str,
    target: str,
    tokenizer,
    device,
) -> float:
    """ターゲット部分のlossを計算"""
    prompt_tokens = tokenizer.encode(prompt)
    target_tokens = tokenizer.encode(target)
    full_tokens = prompt_tokens + target_tokens

    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids)

    # ターゲット部分のlossのみ計算
    prompt_len = len(prompt_tokens)
    target_logits = logits[0, prompt_len - 1 : -1, :]
    target_labels = torch.tensor(target_tokens, dtype=torch.long, device=device)

    if len(target_labels) == 0:
        return 0.0

    loss = F.cross_entropy(target_logits, target_labels, reduction="mean")
    return loss.item()


def evaluate_reversal_curse(
    model,
    val_pairs: list[FamilyPair],
    tokenizer,
    device,
) -> dict:
    """
    Reversal Curse評価（汎化テスト）

    Valペアは順方向のみ学習。逆方向で評価して汎化できるか確認。
    """
    model.eval()

    backward_losses = []

    with torch.no_grad():
        for pair in val_pairs:
            # 逆方向: "Who is {parent}'s child?" → child
            # Valペアではこの方向を学習していない
            prompt = f"Who is {pair.parent_name}'s child?"
            target = f" {pair.child_name}"
            loss = compute_target_loss(model, prompt, target, tokenizer, device)
            backward_losses.append(loss)

    backward_ppl = torch.exp(
        torch.tensor(sum(backward_losses) / len(backward_losses))
    ).item()

    return {"reversal_ppl": backward_ppl}


def train_model(
    model,
    train_loader,
    pile_val_loader,
    optimizer,
    device,
    num_epochs: int,
    patience: int,
    gradient_clip: float,
) -> tuple[float, int, Optional[dict]]:
    """モデル訓練（Pile Val PPLでearly stopping）"""
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()

        epoch_loss = 0.0
        epoch_tokens = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            mask = shift_labels != -100
            epoch_loss += loss.item() * mask.sum().item()
            epoch_tokens += mask.sum().item()

        train_ppl = torch.exp(torch.tensor(epoch_loss / max(epoch_tokens, 1))).item()
        val_ppl = compute_ppl(model, pile_val_loader, device)
        elapsed = time.time() - start_time

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
            f"  Epoch {epoch:2d}: train={train_ppl:7.1f}, pile_val={val_ppl:7.1f} "
            f"({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


# =============================================================================
# 実験実行
# =============================================================================


def run_baseline(
    config: PythiaConfig,
    pattern_pairs: list[FamilyPair],
    val_pairs: list[FamilyPair],
    pile_samples: list[dict],
    pile_val_loader,
    tokenizer,
    device,
    args,
) -> dict:
    """Baseline: 全文学習"""
    print_flush("\n" + "=" * 70)
    print_flush("BASELINE (Traditional LM Training)")
    print_flush("=" * 70)

    # サンプル生成
    pattern_samples = create_baseline_pattern_samples(pattern_pairs)
    val_samples = create_baseline_val_samples(val_pairs)

    # Familyサンプル作成
    family_samples = create_baseline_family_samples(
        pattern_samples, val_samples, tokenizer
    )

    # データセット作成（Pileサンプルは共通）
    train_dataset = UnifiedDataset(
        pile_samples, family_samples, tokenizer, seq_length=args.seq_length
    )

    print_flush(f"  Train samples: {len(train_dataset)}")
    print_flush(f"    - Pile: {len(pile_samples)}")
    print_flush(f"    - Pattern (forward+backward): {len(pattern_samples)}")
    print_flush(f"    - Val (forward only): {len(val_samples)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # モデル作成
    model = create_model("pythia", config)
    model = model.to(device)

    print_flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print_flush("\n[Baseline] Training...")
    best_ppl, best_epoch, best_state = train_model(
        model, train_loader, pile_val_loader, optimizer, device,
        args.epochs, args.patience, GRADIENT_CLIP
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    if best_state:
        model.load_state_dict(best_state)

    # 評価
    print_flush("\n  Reversal Curse (Val pairs, backward direction):")
    reversal = evaluate_reversal_curse(model, val_pairs, tokenizer, device)
    print_flush(f"    'Who is [parent]'s child?' PPL: {reversal['reversal_ppl']:.1f}")

    print_flush("\n  Pile PPL:")
    pile_ppl = compute_ppl(model, pile_val_loader, device)
    print_flush(f"    Val PPL: {pile_ppl:.1f}")

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "reversal_ppl": reversal["reversal_ppl"],
        "pile_ppl": pile_ppl,
    }

    del model
    clear_gpu_cache(device)

    return result


def run_modified(
    config: PythiaConfig,
    pattern_pairs: list[FamilyPair],
    val_pairs: list[FamilyPair],
    pile_samples: list[dict],
    pile_val_loader,
    tokenizer,
    device,
    args,
) -> dict:
    """Modified: コンテキスト分離学習"""
    print_flush("\n" + "=" * 70)
    print_flush("MODIFIED (Context-Separated Training)")
    print_flush("=" * 70)

    # サンプル生成
    pattern_samples = create_modified_pattern_samples(pattern_pairs)
    val_samples = create_modified_val_samples(val_pairs)

    # Familyサンプル作成
    family_samples = create_modified_family_samples(
        pattern_samples, val_samples, tokenizer
    )

    # データセット作成（Pileサンプルは共通）
    train_dataset = UnifiedDataset(
        pile_samples, family_samples, tokenizer, seq_length=args.seq_length
    )

    print_flush(f"  Train samples: {len(train_dataset)}")
    print_flush(f"    - Pile: {len(pile_samples)}")
    print_flush(f"    - Pattern (context-separated): {len(pattern_samples)}")
    print_flush(f"    - Val (forward only): {len(val_samples)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # モデル作成
    model = create_model("pythia", config)
    model = model.to(device)

    print_flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print_flush("\n[Modified] Training...")
    best_ppl, best_epoch, best_state = train_model(
        model, train_loader, pile_val_loader, optimizer, device,
        args.epochs, args.patience, GRADIENT_CLIP
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    if best_state:
        model.load_state_dict(best_state)

    # 評価
    print_flush("\n  Reversal Curse (Val pairs, backward direction):")
    reversal = evaluate_reversal_curse(model, val_pairs, tokenizer, device)
    print_flush(f"    'Who is [parent]'s child?' PPL: {reversal['reversal_ppl']:.1f}")

    print_flush("\n  Pile PPL:")
    pile_ppl = compute_ppl(model, pile_val_loader, device)
    print_flush(f"    Val PPL: {pile_ppl:.1f}")

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "reversal_ppl": reversal["reversal_ppl"],
        "pile_ppl": pile_ppl,
    }

    del model
    clear_gpu_cache(device)

    return result


# =============================================================================
# メイン
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Reversal Curse Generalization Experiment"
    )
    parser.add_argument(
        "--num-pairs", type=int, default=500,
        help="Total number of family pairs"
    )
    parser.add_argument(
        "--num-val-pairs", type=int, default=100,
        help="Number of val pairs (for reversal curse test)"
    )
    parser.add_argument(
        "--num-pile-samples", type=int, default=2000,
        help="Number of Pile samples (fixed for fair comparison)"
    )
    parser.add_argument("--pile-tokens", type=int, default=500000)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--nope", action="store_true", help="Use NoPE")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    if args.nope:
        config.rotary_pct = 0.0

    tokenizer = get_tokenizer(config.tokenizer_name)

    # 実験情報
    print_flush("=" * 70)
    print_flush("REVERSAL CURSE GENERALIZATION EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Total pairs: {args.num_pairs}")
    print_flush(f"  - Pattern pairs: {args.num_pairs - args.num_val_pairs}")
    print_flush(f"  - Val pairs: {args.num_val_pairs}")
    print_flush(f"Pile samples: {args.num_pile_samples} (fixed)")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Position Encoding: {'NoPE' if args.nope else 'RoPE'}")
    print_flush("=" * 70)

    # データ準備
    print_flush("\n[Data] Generating family pairs...")
    all_pairs = generate_family_pairs(args.num_pairs)
    pattern_pairs, val_pairs = split_pairs_for_experiment(
        all_pairs, num_val_pairs=args.num_val_pairs
    )

    print_flush(f"  Pattern pairs: {len(pattern_pairs)}")
    print_flush(f"  Val pairs: {len(val_pairs)}")

    # 最初のいくつかのペアを表示
    print_flush("\n  Sample pattern pairs:")
    for pair in pattern_pairs[:3]:
        print_flush(f"    {pair.parent_name} is {pair.child_name}'s {pair.relation}")

    print_flush("\n  Sample val pairs:")
    for pair in val_pairs[:3]:
        print_flush(f"    {pair.parent_name} is {pair.child_name}'s {pair.relation}")

    print_flush("\n[Data] Loading Pile data...")
    pile_tokens = load_pile_tokens_cached(args.pile_tokens, config.tokenizer_name)

    # Pileサンプル作成（Baseline/Modified共通）
    pile_samples = create_pile_samples(
        pile_tokens, args.seq_length, args.num_pile_samples
    )
    print_flush(f"  Pile samples: {len(pile_samples)} (shared between Baseline/Modified)")

    # Pile validation用のDataLoader
    val_size = args.pile_tokens // 10
    pile_val_tokens = pile_tokens[-val_size:]
    pile_val_samples = []
    for i in range(0, len(pile_val_tokens) - args.seq_length, args.seq_length):
        tokens = pile_val_tokens[i : i + args.seq_length]
        pile_val_samples.append((tokens, tokens.clone()))

    pile_val_dataset = torch.utils.data.TensorDataset(
        torch.stack([s[0] for s in pile_val_samples]),
        torch.stack([s[1] for s in pile_val_samples]),
    )
    pile_val_loader = DataLoader(pile_val_dataset, batch_size=args.batch_size)

    results = {}

    # Baseline
    results["baseline"] = run_baseline(
        config, pattern_pairs, val_pairs,
        pile_samples, pile_val_loader, tokenizer, device, args
    )

    # Modified
    results["modified"] = run_modified(
        config, pattern_pairs, val_pairs,
        pile_samples, pile_val_loader, tokenizer, device, args
    )

    # サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Reversal PPL | Pile PPL |")
    print_flush("|-------|--------------|----------|")

    baseline = results["baseline"]
    print_flush(
        f"| Baseline | {baseline['reversal_ppl']:.1f} | {baseline['pile_ppl']:.1f} |"
    )

    modified = results["modified"]
    print_flush(
        f"| Modified | {modified['reversal_ppl']:.1f} | {modified['pile_ppl']:.1f} |"
    )

    print_flush("\n* Reversal PPL: PPL for 'Who is [parent]'s child?' (lower is better)")
    print_flush("* This tests generalization: can the model apply learned patterns")
    print_flush("  to val pairs that were only trained on forward direction?")
    print_flush("\nDONE")


if __name__ == "__main__":
    main()
