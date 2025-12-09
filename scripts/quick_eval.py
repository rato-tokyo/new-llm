#!/usr/bin/env python3
"""
Quick Model Evaluation Script

モデルの動作確認とPPL測定を簡単に行うスクリプト。

Usage:
    # Pythia Baseline
    python3 scripts/quick_eval.py --model pythia

    # Senri (1 Senri + 5 Pythia)
    python3 scripts/quick_eval.py --model senri

    # Senri with multiple memories
    python3 scripts/quick_eval.py --model senri --num-memories 4

    # カスタム設定
    python3 scripts/quick_eval.py --model senri --num-tokens 50000 --seq-length 256
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch

from src.config import OPEN_CALM_VOCAB_SIZE, OPEN_CALM_TOKENIZER
from src.models import TransformerLM, pythia_layers, senri_layers
from src.utils.data_pythia import load_pile_tokens_cached
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_open_calm_tokenizer, test_tokenizer_coverage
from src.utils.training import get_device


def create_model(model_type: str, num_memories: int = 1) -> TransformerLM:
    """モデルを作成"""
    if model_type == "pythia":
        layers = pythia_layers(6)
        name = "Pythia (6 layers)"
    elif model_type == "senri":
        layers = senri_layers(1, num_memories=num_memories) + pythia_layers(5)
        name = f"Senri (1 Senri + 5 Pythia, {num_memories} memories)"
    elif model_type == "senri-only":
        layers = senri_layers(6, num_memories=num_memories)
        name = f"Senri-Only (6 Senri layers, {num_memories} memories)"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = TransformerLM(layers=layers, vocab_size=OPEN_CALM_VOCAB_SIZE)
    return model, name


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
    parser = argparse.ArgumentParser(description="Quick Model Evaluation")
    parser.add_argument(
        "--model", type=str, default="pythia",
        choices=["pythia", "senri", "senri-only"],
        help="Model type (default: pythia)"
    )
    parser.add_argument(
        "--num-memories", type=int, default=1,
        help="Number of memories for Senri (default: 1)"
    )
    parser.add_argument(
        "--num-tokens", type=int, default=50000,
        help="Number of tokens for PPL evaluation (default: 50000)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=128,
        help="Sequence length (default: 128)"
    )
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

    print_flush("=" * 70)
    print_flush("QUICK MODEL EVALUATION")
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
    model, model_name = create_model(args.model, args.num_memories)
    model = model.to(device)
    elapsed = time.time() - start_time

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"    Model: {model_name}")
    print_flush(f"    Parameters: {total_params:,}")
    print_flush(f"    Created in {elapsed:.2f}s")

    # テキスト生成テスト
    if not args.skip_generation:
        print_flush("\n[2] Text Generation Test")
        prompts = ["今日は", "人工知能は", "日本の首都は"]
        for prompt in prompts:
            generated = test_generation(model, device, prompt, max_tokens=15)
            print_flush(f"    \"{prompt}\" → \"{generated}\"")

    # PPL評価
    if not args.skip_ppl:
        print_flush(f"\n[3] PPL Evaluation ({args.num_tokens:,} tokens)")
        print_flush("    Loading Pile data...")

        tokens = load_pile_tokens_cached(args.num_tokens + args.seq_length + 1, OPEN_CALM_TOKENIZER)

        print_flush("    Calculating PPL...")
        start_time = time.time()
        ppl = evaluate_ppl(
            model, tokens, device,
            seq_length=args.seq_length,
            num_tokens=args.num_tokens,
        )
        elapsed = time.time() - start_time

        print_flush(f"    PPL: {ppl:.1f}")
        print_flush(f"    Evaluated in {elapsed:.1f}s")

    # サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush(f"Model: {model_name}")
    print_flush(f"Parameters: {total_params:,}")
    if not args.skip_ppl:
        print_flush(f"PPL: {ppl:.1f}")
    print_flush("=" * 70)
    print_flush("DONE")


if __name__ == "__main__":
    main()
