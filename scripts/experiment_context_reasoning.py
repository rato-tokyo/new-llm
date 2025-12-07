#!/usr/bin/env python3
"""
Context-Based Reasoning Experiment

仮説: 知識と推論を分離することで、FFNに関係性パターンを学習させ、
     Reversal Curseを軽減できる。

比較:
- Baseline: 通常の事実文を直接学習（"Tom's mother is Alice."）
- Context: コンテキスト付き推論訓練
  - コンテキストあり: 与えられた情報から推論
  - コンテキストなし: "I don't know."

評価:
- 新しい人物での推論能力（Forward/Backward PPL）
- 有名人テストペア
- 通常のPPL
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
    create_context_qa_samples,
    create_test_pairs,
    generate_family_pairs,
    get_celebrity_test_pairs,
    split_pairs,
)
from src.models import create_model
from src.utils.device import clear_gpu_cache
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.training import get_device, prepare_data_loaders


class ContextReasoningDataset(Dataset):
    """コンテキスト推論用データセット"""

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_length: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["input"]
        target_text = sample["target"]

        # 入力+出力を結合してトークン化
        full_text = input_text + target_text
        tokens = self.tokenizer.encode(full_text)

        # 入力部分の長さを記録（loss計算時に除外するため）
        input_tokens = self.tokenizer.encode(input_text)
        input_len = len(input_tokens)

        # パディング/切り詰め
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
            input_len = min(input_len, self.max_length - 1)
        else:
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        # 入力部分はloss計算から除外
        labels[:input_len] = -100

        return input_ids, labels


class BaselineDataset(Dataset):
    """ベースライン用データセット（通常のLM訓練）"""

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_length: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["input"]

        tokens = self.tokenizer.encode(text)

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

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

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # -100を除外してloss計算
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


def evaluate_relation_reasoning(
    model,
    pairs: list[FamilyPair],
    tokenizer,
    device,
    use_context: bool = True,
) -> dict:
    """
    関係性推論能力を評価

    Args:
        model: モデル
        pairs: テスト用ペア
        tokenizer: トークナイザー
        device: デバイス
        use_context: コンテキストを与えるか

    Returns:
        forward_ppl: 順方向（親→子の質問に答える）
        backward_ppl: 逆方向（子→親の質問に答える）
    """
    model.eval()

    forward_losses = []
    backward_losses = []

    with torch.no_grad():
        for pair in pairs:
            context = f"{pair.parent_name} is {pair.child_name}'s {pair.relation}."

            if use_context:
                # 順方向: Who is X's parent? → 親の名前
                forward_input = f"Context: {context} Question: Who is {pair.child_name}'s parent? Answer:"
                forward_target = f" {pair.parent_name}"

                # 逆方向: Who is X's child? → 子の名前
                backward_input = f"Context: {context} Question: Who is {pair.parent_name}'s child? Answer:"
                backward_target = f" {pair.child_name}"
            else:
                # コンテキストなしでの直接質問
                forward_input = f"Question: Who is {pair.child_name}'s parent? Answer:"
                forward_target = f" {pair.parent_name}"

                backward_input = f"Question: Who is {pair.parent_name}'s child? Answer:"
                backward_target = f" {pair.child_name}"

            # 順方向のPPL
            forward_loss = compute_target_loss(
                model, forward_input, forward_target, tokenizer, device
            )
            forward_losses.append(forward_loss)

            # 逆方向のPPL
            backward_loss = compute_target_loss(
                model, backward_input, backward_target, tokenizer, device
            )
            backward_losses.append(backward_loss)

    forward_ppl = torch.exp(torch.tensor(sum(forward_losses) / len(forward_losses))).item()
    backward_ppl = torch.exp(torch.tensor(sum(backward_losses) / len(backward_losses))).item()

    return {
        "forward_ppl": forward_ppl,
        "backward_ppl": backward_ppl,
        "gap": backward_ppl - forward_ppl,
    }


def compute_target_loss(
    model,
    input_text: str,
    target_text: str,
    tokenizer,
    device,
) -> float:
    """ターゲット部分のlossを計算"""
    input_tokens = tokenizer.encode(input_text)
    target_tokens = tokenizer.encode(target_text)
    full_tokens = input_tokens + target_tokens

    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids)

    # ターゲット部分のlossのみ計算
    input_len = len(input_tokens)
    target_logits = logits[0, input_len - 1 : -1, :]  # 予測位置
    target_labels = torch.tensor(target_tokens, dtype=torch.long, device=device)

    if len(target_labels) == 0:
        return 0.0

    loss = F.cross_entropy(target_logits, target_labels, reduction="mean")
    return loss.item()


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs: int,
    patience: int,
    gradient_clip: float,
    model_name: str,
) -> tuple[float, int, Optional[dict]]:
    """モデル訓練"""
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
        val_ppl = compute_ppl(model, val_loader, device)
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
            f"  Epoch {epoch:2d}: train={train_ppl:7.1f}, val={val_ppl:7.1f} "
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
    test_pairs: list[FamilyPair],
    celebrity_pairs: list[FamilyPair],
    pile_train_loader,
    pile_val_loader,
    tokenizer,
    device,
    args,
) -> dict:
    """ベースラインモデルの訓練と評価"""
    print_flush("\n" + "=" * 70)
    print_flush("BASELINE (Direct Learning)")
    print_flush("=" * 70)

    # ベースラインサンプル生成（順方向の事実のみ）
    baseline_samples = create_baseline_samples(train_pairs)
    val_samples = create_baseline_samples(val_pairs)

    print_flush(f"  Train samples: {len(baseline_samples)}")
    print_flush(f"  Val samples: {len(val_samples)}")

    train_dataset = BaselineDataset(baseline_samples, tokenizer, args.seq_length)
    val_dataset = BaselineDataset(val_samples, tokenizer, args.seq_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # モデル作成
    model = create_model("pythia", config)
    model = model.to(device)

    print_flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print_flush("\n[Baseline] Training on family relations...")
    best_ppl, best_epoch, best_state = train_model(
        model, train_loader, val_loader, optimizer, device,
        args.epochs, args.patience, GRADIENT_CLIP, "baseline"
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    if best_state:
        model.load_state_dict(best_state)

    # 評価
    print_flush("\n  Relation Reasoning (Test pairs, no context):")
    test_result = evaluate_relation_reasoning(
        model, test_pairs, tokenizer, device, use_context=False
    )
    print_flush(f"    Forward PPL: {test_result['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {test_result['backward_ppl']:.1f}")
    print_flush(f"    Gap: {test_result['gap']:+.1f}")

    print_flush("\n  Celebrity Test (no context):")
    celeb_result = evaluate_relation_reasoning(
        model, celebrity_pairs, tokenizer, device, use_context=False
    )
    print_flush(f"    Forward PPL: {celeb_result['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {celeb_result['backward_ppl']:.1f}")
    print_flush(f"    Gap: {celeb_result['gap']:+.1f}")

    print_flush("\n  Pile PPL:")
    pile_ppl = compute_ppl(model, pile_val_loader, device)
    print_flush(f"    Val PPL: {pile_ppl:.1f}")

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "test_forward_ppl": test_result["forward_ppl"],
        "test_backward_ppl": test_result["backward_ppl"],
        "test_gap": test_result["gap"],
        "celeb_forward_ppl": celeb_result["forward_ppl"],
        "celeb_backward_ppl": celeb_result["backward_ppl"],
        "celeb_gap": celeb_result["gap"],
        "pile_ppl": pile_ppl,
    }

    del model
    clear_gpu_cache(device)

    return result


def run_context_model(
    config: PythiaConfig,
    train_pairs: list[FamilyPair],
    val_pairs: list[FamilyPair],
    test_pairs: list[FamilyPair],
    celebrity_pairs: list[FamilyPair],
    pile_train_loader,
    pile_val_loader,
    tokenizer,
    device,
    args,
) -> dict:
    """コンテキスト推論モデルの訓練と評価"""
    print_flush("\n" + "=" * 70)
    print_flush("CONTEXT REASONING")
    print_flush("=" * 70)

    # コンテキスト付きサンプル生成
    context_samples = create_context_qa_samples(train_pairs, include_negative=True)
    val_samples = create_context_qa_samples(val_pairs, include_negative=True)

    print_flush(f"  Train samples: {len(context_samples)}")
    print_flush(f"    - With context: {sum(1 for s in context_samples if s['has_context'])}")
    print_flush(f"    - Without context: {sum(1 for s in context_samples if not s['has_context'])}")
    print_flush(f"  Val samples: {len(val_samples)}")

    train_dataset = ContextReasoningDataset(context_samples, tokenizer, args.seq_length)
    val_dataset = ContextReasoningDataset(val_samples, tokenizer, args.seq_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # モデル作成
    model = create_model("pythia", config)
    model = model.to(device)

    print_flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print_flush("\n[Context] Training on context-based QA...")
    best_ppl, best_epoch, best_state = train_model(
        model, train_loader, val_loader, optimizer, device,
        args.epochs, args.patience, GRADIENT_CLIP, "context"
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    if best_state:
        model.load_state_dict(best_state)

    # 評価（コンテキスト付き）
    print_flush("\n  Relation Reasoning (Test pairs, with context):")
    test_with_ctx = evaluate_relation_reasoning(
        model, test_pairs, tokenizer, device, use_context=True
    )
    print_flush(f"    Forward PPL: {test_with_ctx['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {test_with_ctx['backward_ppl']:.1f}")
    print_flush(f"    Gap: {test_with_ctx['gap']:+.1f}")

    # 評価（コンテキストなし - 転移能力）
    print_flush("\n  Relation Reasoning (Test pairs, no context - transfer):")
    test_no_ctx = evaluate_relation_reasoning(
        model, test_pairs, tokenizer, device, use_context=False
    )
    print_flush(f"    Forward PPL: {test_no_ctx['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {test_no_ctx['backward_ppl']:.1f}")
    print_flush(f"    Gap: {test_no_ctx['gap']:+.1f}")

    print_flush("\n  Celebrity Test (with context):")
    celeb_with_ctx = evaluate_relation_reasoning(
        model, celebrity_pairs, tokenizer, device, use_context=True
    )
    print_flush(f"    Forward PPL: {celeb_with_ctx['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {celeb_with_ctx['backward_ppl']:.1f}")
    print_flush(f"    Gap: {celeb_with_ctx['gap']:+.1f}")

    print_flush("\n  Pile PPL:")
    pile_ppl = compute_ppl(model, pile_val_loader, device)
    print_flush(f"    Val PPL: {pile_ppl:.1f}")

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "test_with_ctx_forward_ppl": test_with_ctx["forward_ppl"],
        "test_with_ctx_backward_ppl": test_with_ctx["backward_ppl"],
        "test_with_ctx_gap": test_with_ctx["gap"],
        "test_no_ctx_forward_ppl": test_no_ctx["forward_ppl"],
        "test_no_ctx_backward_ppl": test_no_ctx["backward_ppl"],
        "test_no_ctx_gap": test_no_ctx["gap"],
        "celeb_with_ctx_forward_ppl": celeb_with_ctx["forward_ppl"],
        "celeb_with_ctx_backward_ppl": celeb_with_ctx["backward_ppl"],
        "celeb_with_ctx_gap": celeb_with_ctx["gap"],
        "pile_ppl": pile_ppl,
    }

    del model
    clear_gpu_cache(device)

    return result


def main():
    parser = argparse.ArgumentParser(description="Context-Based Reasoning Experiment")
    parser.add_argument("--num-pairs", type=int, default=200, help="Number of family pairs")
    parser.add_argument("--samples", type=int, default=2500, help="Number of Pile samples")
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
    print_flush("CONTEXT-BASED REASONING EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Family pairs: {args.num_pairs}")
    print_flush(f"Pile samples: {args.samples}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Position Encoding: {'NoPE' if args.nope else 'RoPE'}")
    print_flush("=" * 70)

    # データ準備
    print_flush("\n[Data] Generating family pairs...")
    all_pairs = generate_family_pairs(args.num_pairs)
    train_pairs, val_pairs = split_pairs(all_pairs, train_ratio=0.8)
    test_pairs = create_test_pairs(50)  # 別の名前でテスト
    celebrity_pairs = get_celebrity_test_pairs()

    print_flush(f"  Train pairs: {len(train_pairs)}")
    print_flush(f"  Val pairs: {len(val_pairs)}")
    print_flush(f"  Test pairs: {len(test_pairs)}")
    print_flush(f"  Celebrity pairs: {len(celebrity_pairs)}")

    print_flush("\n[Data] Loading Pile data...")
    pile_train_loader, pile_val_loader = prepare_data_loaders(
        num_samples=args.samples,
        seq_length=args.seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=args.batch_size,
    )

    results = {}

    # ベースライン
    results["baseline"] = run_baseline(
        config, train_pairs, val_pairs, test_pairs, celebrity_pairs,
        pile_train_loader, pile_val_loader, tokenizer, device, args
    )

    # コンテキスト推論
    results["context"] = run_context_model(
        config, train_pairs, val_pairs, test_pairs, celebrity_pairs,
        pile_train_loader, pile_val_loader, tokenizer, device, args
    )

    # サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Relation Gap | Celebrity Gap | Pile PPL |")
    print_flush("|-------|--------------|---------------|----------|")

    baseline = results["baseline"]
    print_flush(
        f"| Baseline | {baseline['test_gap']:+.1f} | "
        f"{baseline['celeb_gap']:+.1f} | {baseline['pile_ppl']:.1f} |"
    )

    context = results["context"]
    print_flush(
        f"| Context (w/ ctx) | {context['test_with_ctx_gap']:+.1f} | "
        f"{context['celeb_with_ctx_gap']:+.1f} | {context['pile_ppl']:.1f} |"
    )
    print_flush(
        f"| Context (no ctx) | {context['test_no_ctx_gap']:+.1f} | "
        f"- | - |"
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
