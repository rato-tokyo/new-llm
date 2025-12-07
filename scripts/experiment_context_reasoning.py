#!/usr/bin/env python3
"""
Context-Dependent Reasoning (CDR) Training Experiment

目的: 推論パターンをFFNに学習させ、知識は外部コンテキストに預ける

比較:
- Baseline: 従来のLM学習（コンテキスト+質問+回答を全体として学習）
  → 知識と推論がFFNに混在 → Reversal Curse発生
- CDR: コンテキストを初期状態として与え、質問→回答のみ学習
  → 推論パターンをFFNに、知識はAttentionで処理 → Reversal Curse軽減

訓練データ:
- Baseline: "{parent} is {child}'s {relation}. Who is {child}'s parent? {parent}"
- CDR:
  - コンテキストあり: context="{parent} is {child}'s {relation}." → "Who is X?" → answer
  - コンテキストなし: context="" → "Who is X?" → "I don't know"
"""

import argparse
import sys
import time
from typing import Optional

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config.pythia import PythiaConfig
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE, GRADIENT_CLIP
from src.data.family_relations import (
    FamilyPair,
    create_baseline_samples,
    create_cdr_samples,
    generate_family_pairs,
    split_pairs,
)
from src.models import create_model
from src.utils.device import clear_gpu_cache
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.data_pythia import load_pile_tokens_cached
from src.utils.training import get_device


class BaselineDataset(Dataset):
    """Baseline用データセット: Pile + 全文学習"""

    def __init__(
        self,
        pile_tokens: torch.Tensor,
        family_samples: list[dict],
        tokenizer,
        seq_length: int = 128,
        pile_ratio: float = 0.9,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.eos_token_id

        # Pileデータからサンプル作成
        self.pile_samples = []
        for i in range(0, len(pile_tokens) - seq_length, seq_length):
            tokens = pile_tokens[i : i + seq_length].tolist()
            self.pile_samples.append({"type": "pile", "tokens": tokens})

        # Baselineサンプル: 全文を学習
        self.family_samples = []
        for sample in family_samples:
            full_text = sample["input"]  # 全文が入っている
            tokens = tokenizer.encode(full_text)
            self.family_samples.append({
                "type": "baseline",
                "tokens": tokens,
            })

        # 比率調整
        total_family = len(self.family_samples)
        if 0 < pile_ratio < 1:
            target_pile = int(total_family * pile_ratio / (1 - pile_ratio))
            target_pile = min(target_pile, len(self.pile_samples))
        else:
            target_pile = len(self.pile_samples) if pile_ratio == 1 else 0

        self.pile_samples = self.pile_samples[:target_pile]
        self.all_samples = self.pile_samples + self.family_samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        tokens = sample["tokens"]

        if len(tokens) > self.seq_length:
            tokens = tokens[: self.seq_length]
        else:
            tokens = tokens + [self.pad_token_id] * (self.seq_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        return input_ids, labels


class CDRDataset(Dataset):
    """CDR訓練用データセット: Pile + コンテキスト分離学習"""

    def __init__(
        self,
        pile_tokens: torch.Tensor,
        family_samples: list[dict],
        tokenizer,
        seq_length: int = 128,
        pile_ratio: float = 0.9,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pad_token_id = tokenizer.eos_token_id

        # Pileデータからサンプル作成
        self.pile_samples = []
        for i in range(0, len(pile_tokens) - seq_length, seq_length):
            tokens = pile_tokens[i : i + seq_length].tolist()
            self.pile_samples.append({"type": "pile", "tokens": tokens})

        # CDRサンプル: コンテキスト部分はloss計算から除外
        self.family_samples = []
        for sample in family_samples:
            context = sample["context"]
            question = sample["input"]
            target = sample["target"]

            # コンテキスト + 質問 + 回答
            if context:
                full_text = f"{context} {question}{target}"
                context_tokens = tokenizer.encode(f"{context} {question}")
            else:
                full_text = f"{question}{target}"
                context_tokens = tokenizer.encode(question)

            full_tokens = tokenizer.encode(full_text)
            context_len = len(context_tokens)

            self.family_samples.append({
                "type": "cdr",
                "tokens": full_tokens,
                "context_len": context_len,  # この位置までloss計算から除外
            })

        # 比率調整
        total_family = len(self.family_samples)
        if 0 < pile_ratio < 1:
            target_pile = int(total_family * pile_ratio / (1 - pile_ratio))
            target_pile = min(target_pile, len(self.pile_samples))
        else:
            target_pile = len(self.pile_samples) if pile_ratio == 1 else 0

        self.pile_samples = self.pile_samples[:target_pile]
        self.all_samples = self.pile_samples + self.family_samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]

        if sample["type"] == "pile":
            tokens = sample["tokens"]
            input_ids = torch.tensor(tokens, dtype=torch.long)
            labels = input_ids.clone()
            return input_ids, labels

        # CDRサンプル
        tokens = sample["tokens"]
        context_len = sample["context_len"]

        if len(tokens) > self.seq_length:
            tokens = tokens[: self.seq_length]
            context_len = min(context_len, self.seq_length - 1)
        else:
            tokens = tokens + [self.pad_token_id] * (self.seq_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        # コンテキスト+質問部分はloss計算から除外（回答部分のみ学習）
        labels[:context_len] = -100

        return input_ids, labels


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


def evaluate_with_context(
    model,
    pairs: list[FamilyPair],
    tokenizer,
    device,
) -> dict:
    """
    コンテキストありで評価（CDR訓練の成功指標）

    コンテキストを与えた状態で順方向・逆方向の質問に回答できるか評価。
    """
    model.eval()

    forward_losses = []
    backward_losses = []

    with torch.no_grad():
        for pair in pairs:
            context = f"{pair.parent_name} is {pair.child_name}'s {pair.relation}."

            # 順方向: コンテキスト + "Who is child's parent?" → parent
            forward_prompt = f"{context} Who is {pair.child_name}'s parent?"
            forward_target = f" {pair.parent_name}"
            forward_loss = compute_target_loss(
                model, forward_prompt, forward_target, tokenizer, device
            )
            forward_losses.append(forward_loss)

            # 逆方向: コンテキスト + "Who is parent's child?" → child
            backward_prompt = f"{context} Who is {pair.parent_name}'s child?"
            backward_target = f" {pair.child_name}"
            backward_loss = compute_target_loss(
                model, backward_prompt, backward_target, tokenizer, device
            )
            backward_losses.append(backward_loss)

    forward_ppl = torch.exp(torch.tensor(sum(forward_losses) / len(forward_losses))).item()
    backward_ppl = torch.exp(torch.tensor(sum(backward_losses) / len(backward_losses))).item()

    return {
        "forward_ppl": forward_ppl,
        "backward_ppl": backward_ppl,
        "gap": backward_ppl - forward_ppl,
    }


def evaluate_without_context(
    model,
    pairs: list[FamilyPair],
    tokenizer,
    device,
) -> dict:
    """
    コンテキストなしで評価

    コンテキストなしで質問したとき "I don't know" と回答できるか評価。
    """
    model.eval()

    losses = []

    with torch.no_grad():
        for pair in pairs:
            # コンテキストなしで質問
            prompt = f"Who is {pair.child_name}'s parent?"
            target = " I don't know"
            loss = compute_target_loss(model, prompt, target, tokenizer, device)
            losses.append(loss)

    ppl = torch.exp(torch.tensor(sum(losses) / len(losses))).item()
    return {"ppl": ppl}


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


def train_model(
    model,
    train_loader,
    pile_val_loader,
    optimizer,
    device,
    num_epochs: int,
    patience: int,
    gradient_clip: float,
    model_name: str,
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


def run_baseline(
    config: PythiaConfig,
    train_pairs: list[FamilyPair],
    val_pairs: list[FamilyPair],
    pile_tokens: torch.Tensor,
    pile_val_loader,
    tokenizer,
    device,
    args,
) -> dict:
    """Baseline: 従来のLM学習"""
    print_flush("\n" + "=" * 70)
    print_flush("BASELINE (Traditional LM Training)")
    print_flush("=" * 70)

    # Baselineサンプル生成（全文学習）
    baseline_samples = create_baseline_samples(train_pairs)

    # データセット作成
    train_dataset = BaselineDataset(
        pile_tokens, baseline_samples, tokenizer,
        seq_length=args.seq_length, pile_ratio=args.pile_ratio
    )

    print_flush(f"  Train samples: {len(train_dataset)}")
    print_flush(f"    - Pile: {len(train_dataset.pile_samples)}")
    print_flush(f"    - Family: {len(train_dataset.family_samples)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # モデル作成
    model = create_model("pythia", config)
    model = model.to(device)

    print_flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print_flush("\n[Baseline] Training...")
    best_ppl, best_epoch, best_state = train_model(
        model, train_loader, pile_val_loader, optimizer, device,
        args.epochs, args.patience, GRADIENT_CLIP, "baseline"
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    if best_state:
        model.load_state_dict(best_state)

    # 評価
    print_flush("\n  With Context (Train pairs):")
    train_ctx = evaluate_with_context(model, train_pairs, tokenizer, device)
    print_flush(f"    Forward PPL: {train_ctx['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {train_ctx['backward_ppl']:.1f}")
    print_flush(f"    Gap: {train_ctx['gap']:+.1f}")

    print_flush("\n  With Context (Val pairs - generalization):")
    val_ctx = evaluate_with_context(model, val_pairs, tokenizer, device)
    print_flush(f"    Forward PPL: {val_ctx['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {val_ctx['backward_ppl']:.1f}")
    print_flush(f"    Gap: {val_ctx['gap']:+.1f}")

    print_flush("\n  Without Context (should answer 'I don't know'):")
    no_ctx = evaluate_without_context(model, val_pairs, tokenizer, device)
    print_flush(f"    'I don't know' PPL: {no_ctx['ppl']:.1f}")

    print_flush("\n  Pile PPL:")
    pile_ppl = compute_ppl(model, pile_val_loader, device)
    print_flush(f"    Val PPL: {pile_ppl:.1f}")

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "train_forward_ppl": train_ctx["forward_ppl"],
        "train_backward_ppl": train_ctx["backward_ppl"],
        "train_gap": train_ctx["gap"],
        "val_forward_ppl": val_ctx["forward_ppl"],
        "val_backward_ppl": val_ctx["backward_ppl"],
        "val_gap": val_ctx["gap"],
        "no_context_ppl": no_ctx["ppl"],
        "pile_ppl": pile_ppl,
    }

    del model
    clear_gpu_cache(device)

    return result


def run_cdr(
    config: PythiaConfig,
    train_pairs: list[FamilyPair],
    val_pairs: list[FamilyPair],
    pile_tokens: torch.Tensor,
    pile_val_loader,
    tokenizer,
    device,
    args,
) -> dict:
    """CDR訓練: Context-Dependent Reasoning"""
    print_flush("\n" + "=" * 70)
    print_flush("CDR (Context-Dependent Reasoning Training)")
    print_flush("=" * 70)

    # CDRサンプル生成
    cdr_samples = create_cdr_samples(train_pairs)

    # データセット作成
    train_dataset = CDRDataset(
        pile_tokens, cdr_samples, tokenizer,
        seq_length=args.seq_length, pile_ratio=args.pile_ratio
    )

    ctx_samples = [s for s in cdr_samples if s["has_context"]]
    no_ctx_samples = [s for s in cdr_samples if not s["has_context"]]

    print_flush(f"  Train samples: {len(train_dataset)}")
    print_flush(f"    - Pile: {len(train_dataset.pile_samples)}")
    print_flush(f"    - With context: {len(ctx_samples)}")
    print_flush(f"    - Without context: {len(no_ctx_samples)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # モデル作成
    model = create_model("pythia", config)
    model = model.to(device)

    print_flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print_flush("\n[CDR] Training...")
    best_ppl, best_epoch, best_state = train_model(
        model, train_loader, pile_val_loader, optimizer, device,
        args.epochs, args.patience, GRADIENT_CLIP, "cdr"
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    if best_state:
        model.load_state_dict(best_state)

    # 評価
    print_flush("\n  With Context (Train pairs):")
    train_ctx = evaluate_with_context(model, train_pairs, tokenizer, device)
    print_flush(f"    Forward PPL: {train_ctx['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {train_ctx['backward_ppl']:.1f}")
    print_flush(f"    Gap: {train_ctx['gap']:+.1f}")

    print_flush("\n  With Context (Val pairs - generalization):")
    val_ctx = evaluate_with_context(model, val_pairs, tokenizer, device)
    print_flush(f"    Forward PPL: {val_ctx['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {val_ctx['backward_ppl']:.1f}")
    print_flush(f"    Gap: {val_ctx['gap']:+.1f}")

    print_flush("\n  Without Context (should answer 'I don't know'):")
    no_ctx = evaluate_without_context(model, val_pairs, tokenizer, device)
    print_flush(f"    'I don't know' PPL: {no_ctx['ppl']:.1f}")

    print_flush("\n  Pile PPL:")
    pile_ppl = compute_ppl(model, pile_val_loader, device)
    print_flush(f"    Val PPL: {pile_ppl:.1f}")

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "train_forward_ppl": train_ctx["forward_ppl"],
        "train_backward_ppl": train_ctx["backward_ppl"],
        "train_gap": train_ctx["gap"],
        "val_forward_ppl": val_ctx["forward_ppl"],
        "val_backward_ppl": val_ctx["backward_ppl"],
        "val_gap": val_ctx["gap"],
        "no_context_ppl": no_ctx["ppl"],
        "pile_ppl": pile_ppl,
    }

    del model
    clear_gpu_cache(device)

    return result


def main():
    parser = argparse.ArgumentParser(description="CDR Training Experiment")
    parser.add_argument("--num-pairs", type=int, default=200, help="Number of family pairs")
    parser.add_argument("--pile-tokens", type=int, default=500000, help="Number of Pile tokens")
    parser.add_argument("--pile-ratio", type=float, default=0.9, help="Ratio of Pile data")
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
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
    print_flush("CONTEXT-DEPENDENT REASONING (CDR) EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Family pairs: {args.num_pairs}")
    print_flush(f"Pile tokens: {args.pile_tokens:,}")
    print_flush(f"Pile ratio: {args.pile_ratio:.0%}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Position Encoding: {'NoPE' if args.nope else 'RoPE'}")
    print_flush("=" * 70)

    # データ準備
    print_flush("\n[Data] Generating family pairs...")
    all_pairs = generate_family_pairs(args.num_pairs)
    train_pairs, val_pairs = split_pairs(all_pairs, train_ratio=0.8)

    print_flush(f"  Train pairs: {len(train_pairs)}")
    print_flush(f"  Val pairs: {len(val_pairs)}")

    print_flush("\n[Data] Loading Pile data...")
    pile_tokens = load_pile_tokens_cached(args.pile_tokens, config.tokenizer_name)

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
        config, train_pairs, val_pairs,
        pile_tokens, pile_val_loader, tokenizer, device, args
    )

    # CDR
    results["cdr"] = run_cdr(
        config, train_pairs, val_pairs,
        pile_tokens, pile_val_loader, tokenizer, device, args
    )

    # サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Train Gap | Val Gap | No-Ctx PPL | Pile PPL |")
    print_flush("|-------|-----------|---------|------------|----------|")

    baseline = results["baseline"]
    print_flush(
        f"| Baseline | {baseline['train_gap']:+.1f} | "
        f"{baseline['val_gap']:+.1f} | {baseline['no_context_ppl']:.1f} | "
        f"{baseline['pile_ppl']:.1f} |"
    )

    cdr = results["cdr"]
    print_flush(
        f"| CDR | {cdr['train_gap']:+.1f} | "
        f"{cdr['val_gap']:+.1f} | {cdr['no_context_ppl']:.1f} | "
        f"{cdr['pile_ppl']:.1f} |"
    )

    print_flush("\n* Gap = Backward PPL - Forward PPL (closer to 0 is better)")
    print_flush("* No-Ctx PPL: PPL for 'I don't know' response (lower is better for CDR)")
    print_flush("\nDONE")


if __name__ == "__main__":
    main()
