#!/usr/bin/env python3
"""
Quick Model Training & Evaluation Script

モデルの訓練とPPL測定を簡単に行うスクリプト。
日本語Wikipedia（wikipedia 20231101.ja）を使用。

Usage:
    # 評価のみ（訓練なし）
    python3 scripts/quick_model.py --model pythia

    # 簡単な訓練 + 評価
    python3 scripts/quick_model.py --model pythia --train --epochs 3

    # 訓練トークン数を指定
    python3 scripts/quick_model.py --model senri --train --train-tokens 100000 --epochs 5

    # Senri with multiple memories
    python3 scripts/quick_model.py --model senri-multi --train
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch

from src.config import (
    OPEN_CALM_VOCAB_SIZE,
    OPEN_CALM_TOKENIZER,
    MODEL_PRESETS,
    create_model,
)
from src.models import TransformerLM
from src.utils.data_utils import load_wiki_ja_tokens_cached
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_open_calm_tokenizer, test_tokenizer_coverage
from src.utils.training import get_device


def get_model(model_type: str) -> TransformerLM:
    """モデルを作成（src/config/models.pyのプリセットを使用）"""
    model = create_model(model_type)
    return model, model.describe()


def train_model(
    model: TransformerLM,
    tokens: torch.Tensor,
    device: torch.device,
    seq_length: int = 128,
    num_tokens: int = 100000,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
) -> dict:
    """モデルを訓練"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # バッチ作成
    num_sequences = min(num_tokens, len(tokens) - seq_length - 1) // seq_length
    batches = []
    for i in range(0, num_sequences * seq_length, seq_length):
        input_ids = tokens[i:i + seq_length]
        labels = tokens[i + 1:i + seq_length + 1]
        batches.append((input_ids, labels))

    # バッチをまとめる
    batch_groups = []
    for i in range(0, len(batches), batch_size):
        group = batches[i:i + batch_size]
        input_batch = torch.stack([b[0] for b in group]).to(device)
        label_batch = torch.stack([b[1] for b in group]).to(device)
        batch_groups.append((input_batch, label_batch))

    history = {"train_ppl": [], "epoch_time": []}

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_tokens = 0

        # メモリリセット
        if hasattr(model, 'reset_memory'):
            model.reset_memory()

        for input_ids, labels in batch_groups:
            optimizer.zero_grad()

            logits = model(input_ids, update_memory=True)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum'
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += labels.numel()

        epoch_time = time.time() - epoch_start
        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

        history["train_ppl"].append(train_ppl)
        history["epoch_time"].append(epoch_time)

        print_flush(f"    Epoch {epoch + 1}/{epochs}: PPL={train_ppl:.1f}, Time={epoch_time:.1f}s")

    return history


def evaluate_ppl(
    model: TransformerLM,
    tokens: torch.Tensor,
    device: torch.device,
    seq_length: int = 128,
    num_tokens: int = 50000,
    update_memory: bool = True,
) -> float:
    """PPLを計算"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # メモリリセット（Senriモデルの場合）
    if hasattr(model, 'reset_memory'):
        model.reset_memory()

    with torch.no_grad():
        for i in range(0, min(num_tokens, len(tokens) - seq_length), seq_length):
            input_ids = tokens[i:i + seq_length].unsqueeze(0).to(device)
            labels = tokens[i + 1:i + seq_length + 1].unsqueeze(0).to(device)

            logits = model(input_ids, update_memory=update_memory)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def test_tokenizer() -> bool:
    """トークナイザーの動作確認"""
    tokenizer = get_open_calm_tokenizer()

    test_cases = [
        ("日本語", "今日は良い天気ですね。"),
        ("英語混在", "AIの発展は目覚ましい。GPUで学習を高速化。"),
        ("技術用語", "APIを呼び出してHTTPリクエストを送信。"),
        ("絵文字", "完了しました！🎉"),
    ]

    all_passed = True
    for name, text in test_cases:
        result = test_tokenizer_coverage(tokenizer, text)
        status = "OK" if not result["has_unk"] else "NG"
        if result["has_unk"]:
            all_passed = False
        print_flush(f"    [{status}] {name}: {len(result['tokens'])} tokens")

    # vocab_size確認
    if tokenizer.vocab_size != OPEN_CALM_VOCAB_SIZE:
        print_flush(f"    [NG] Vocab size mismatch: {tokenizer.vocab_size} != {OPEN_CALM_VOCAB_SIZE}")
        all_passed = False
    else:
        print_flush(f"    [OK] Vocab size: {tokenizer.vocab_size:,}")

    return all_passed


def test_generation(
    model: TransformerLM,
    device: torch.device,
    prompt: str = "今日は",
    max_tokens: int = 20,
) -> str:
    """テキスト生成テスト"""
    tokenizer = get_open_calm_tokenizer()
    model.eval()

    # メモリリセット
    if hasattr(model, 'reset_memory'):
        model.reset_memory()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_token_id = logits[0, -1, :].argmax().item()
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=device)],
                dim=1
            )
            if next_token_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Quick Model Training & Evaluation")
    parser.add_argument(
        "--model", type=str, default="pythia",
        choices=list(MODEL_PRESETS.keys()),
        help="Model type (default: pythia). See src/config/models.py for details."
    )
    parser.add_argument(
        "--num-tokens", type=int, default=50000,
        help="Number of tokens for PPL evaluation (default: 50000)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=128,
        help="Sequence length (default: 128)"
    )
    # 訓練オプション
    parser.add_argument(
        "--train", action="store_true",
        help="Enable training mode"
    )
    parser.add_argument(
        "--train-tokens", type=int, default=100000,
        help="Number of tokens for training (default: 100000)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for training (default: 4)"
    )
    # スキップオプション
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip text generation test"
    )
    parser.add_argument(
        "--skip-ppl", action="store_true",
        help="Skip PPL evaluation"
    )
    parser.add_argument(
        "--skip-tokenizer", action="store_true",
        help="Skip tokenizer test"
    )

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    mode = "TRAINING & EVALUATION" if args.train else "EVALUATION ONLY"
    print_flush("=" * 70)
    print_flush(f"QUICK MODEL {mode}")
    print_flush("=" * 70)

    # トークナイザーテスト
    if not args.skip_tokenizer:
        print_flush("\n[0] Tokenizer Test")
        tokenizer_ok = test_tokenizer()
        if not tokenizer_ok:
            print_flush("    WARNING: Tokenizer test failed!")

    # モデル作成
    print_flush(f"\n[1] Creating model: {args.model}")
    start_time = time.time()
    model, model_name = get_model(args.model)
    model = model.to(device)
    elapsed = time.time() - start_time

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"    Model: {model_name}")
    print_flush(f"    Parameters: {total_params:,}")
    print_flush(f"    Created in {elapsed:.2f}s")

    # データロード（日本語Wikipedia）
    max_tokens = max(args.num_tokens, args.train_tokens if args.train else 0)
    print_flush(f"\n[2] Loading Japanese Wikipedia ({max_tokens:,} tokens)...")
    tokens = load_wiki_ja_tokens_cached(max_tokens + args.seq_length + 1, OPEN_CALM_TOKENIZER)
    print_flush(f"    Loaded {len(tokens):,} tokens")

    # 訓練
    train_history = None
    if args.train:
        print_flush(f"\n[3] Training ({args.train_tokens:,} tokens, {args.epochs} epochs)")
        train_history = train_model(
            model, tokens, device,
            seq_length=args.seq_length,
            num_tokens=args.train_tokens,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )

    # テキスト生成テスト
    step = 4 if args.train else 3
    if not args.skip_generation:
        print_flush(f"\n[{step}] Text Generation Test")
        prompts = ["今日は", "人工知能は", "日本の首都は"]
        for prompt in prompts:
            generated = test_generation(model, device, prompt, max_tokens=15)
            print_flush(f"    \"{prompt}\" → \"{generated}\"")
        step += 1

    # PPL評価
    val_ppl = None
    if not args.skip_ppl:
        print_flush(f"\n[{step}] PPL Evaluation ({args.num_tokens:,} tokens)")
        print_flush("    Calculating PPL...")
        start_time = time.time()
        val_ppl = evaluate_ppl(
            model, tokens, device,
            seq_length=args.seq_length,
            num_tokens=args.num_tokens,
        )
        elapsed = time.time() - start_time
        print_flush(f"    Val PPL: {val_ppl:.1f}")
        print_flush(f"    Evaluated in {elapsed:.1f}s")

    # サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush(f"Model: {model_name}")
    print_flush(f"Parameters: {total_params:,}")
    if args.train and train_history:
        print_flush(f"Training: {args.epochs} epochs, final train PPL={train_history['train_ppl'][-1]:.1f}")
    if val_ppl is not None:
        print_flush(f"Val PPL: {val_ppl:.1f}")
    print_flush("=" * 70)
    print_flush("DONE")


if __name__ == "__main__":
    main()
