#!/usr/bin/env python3
"""
Senri Integration Verification Script

Senri日本語LLMの動作確認スクリプト。
トークナイザーとモデルが正しく動作することを確認する。

Usage:
    python3 scripts/verify_senri.py
"""

import sys

sys.path.insert(0, ".")

import torch

from src.config import SenriConfig
from src.models import create_model
from src.utils.tokenizer_utils import get_open_calm_tokenizer, test_tokenizer_coverage
from src.utils.training import get_device
from src.utils.io import print_flush


def test_tokenizer():
    """トークナイザーテスト"""
    print_flush("\n" + "=" * 70)
    print_flush("1. TOKENIZER TEST")
    print_flush("=" * 70)

    tokenizer = get_open_calm_tokenizer()
    print_flush(f"  Tokenizer: {type(tokenizer).__name__}")
    print_flush(f"  Vocab size: {tokenizer.vocab_size:,}")
    print_flush(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print_flush(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # テストケース
    test_cases = [
        # 日本語
        ("基本日本語", "今日は良い天気ですね。"),
        ("長文日本語", "人工知能の発展は目覚ましく、様々な分野で活用されています。"),
        # 英語（よく使う単語）
        ("英語混在", "AIの発展は目覚ましい。GPUで学習を高速化。"),
        ("プログラミング", "Pythonでdef main():を書く。"),
        ("技術用語", "APIを呼び出してHTTPリクエストを送信。"),
        # 特殊文字
        ("絵文字", "完了しました！"),
        ("記号", "①②③ → ←"),
        ("URL", "https://example.com/path?q=test"),
    ]

    print_flush("\n  Coverage Test:")
    all_passed = True
    for name, text in test_cases:
        result = test_tokenizer_coverage(tokenizer, text)
        status = "OK" if not result["has_unk"] else f"NG (UNK: {result['unk_count']})"
        if result["has_unk"]:
            all_passed = False
        print_flush(f"    [{status:>6}] {name}: {text[:30]}...")
        print_flush(f"           → {len(result['tokens'])} tokens")

    print_flush(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def test_config():
    """設定テスト"""
    print_flush("\n" + "=" * 70)
    print_flush("2. CONFIG TEST")
    print_flush("=" * 70)

    config = SenriConfig()
    print_flush(f"  vocab_size: {config.vocab_size:,}")
    print_flush(f"  hidden_size: {config.hidden_size}")
    print_flush(f"  num_layers: {config.num_layers}")
    print_flush(f"  num_attention_heads: {config.num_attention_heads}")
    print_flush(f"  intermediate_size: {config.intermediate_size}")
    print_flush(f"  max_position_embeddings: {config.max_position_embeddings}")
    print_flush(f"  tokenizer_name: {config.tokenizer_name}")

    # vocab_sizeがトークナイザーと一致するか確認
    tokenizer = get_open_calm_tokenizer()
    match = config.vocab_size == tokenizer.vocab_size
    print_flush(f"\n  Vocab size match: {'OK' if match else 'MISMATCH'}")
    print_flush(f"    Config: {config.vocab_size}, Tokenizer: {tokenizer.vocab_size}")

    return match


def test_model_creation():
    """モデル作成テスト"""
    print_flush("\n" + "=" * 70)
    print_flush("3. MODEL CREATION TEST")
    print_flush("=" * 70)

    config = SenriConfig()

    # Pythiaモデル
    print_flush("\n  [pythia] Standard Transformer:")
    model = create_model("pythia", base_config=config)
    params = sum(p.numel() for p in model.parameters())
    print_flush(f"    Parameters: {params:,}")

    # Infiniモデル
    print_flush("\n  [infini] Infini-Attention (Layer 0):")
    model = create_model("infini", base_config=config)
    params = sum(p.numel() for p in model.parameters())
    print_flush(f"    Parameters: {params:,}")

    # Multi-Memoryモデル
    print_flush("\n  [multi_memory] Multi-Memory Attention:")
    model = create_model("multi_memory", base_config=config)
    params = sum(p.numel() for p in model.parameters())
    print_flush(f"    Parameters: {params:,}")

    return True


def test_forward_pass():
    """フォワードパステスト"""
    print_flush("\n" + "=" * 70)
    print_flush("4. FORWARD PASS TEST")
    print_flush("=" * 70)

    device = get_device()
    print_flush(f"  Device: {device}")

    config = SenriConfig()
    tokenizer = get_open_calm_tokenizer()

    # テスト入力
    test_text = "今日は良い天気ですね。AIの発展は目覚ましいです。"
    input_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
    print_flush(f"\n  Input: \"{test_text}\"")
    print_flush(f"  Input shape: {input_ids.shape}")

    # 各モデルでテスト
    model_types = ["pythia", "infini", "multi_memory"]

    for model_type in model_types:
        print_flush(f"\n  [{model_type}]")
        model = create_model(model_type, base_config=config)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(input_ids)

        print_flush(f"    Output shape: {output.shape}")
        print_flush(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")

        # 次トークン予測
        next_token_logits = output[0, -1, :]
        next_token_id = next_token_logits.argmax().item()
        next_token = tokenizer.decode([next_token_id])
        print_flush(f"    Next token prediction: \"{next_token}\"")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return True


def test_generation():
    """テキスト生成テスト"""
    print_flush("\n" + "=" * 70)
    print_flush("5. TEXT GENERATION TEST")
    print_flush("=" * 70)

    device = get_device()
    config = SenriConfig()
    tokenizer = get_open_calm_tokenizer()

    # Infiniモデルで生成テスト
    model = create_model("infini", base_config=config)
    model = model.to(device)
    model.eval()

    prompt = "今日は"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print_flush(f"  Prompt: \"{prompt}\"")
    print_flush("  Generating 20 tokens...")

    generated_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(20):
            output = model(generated_ids)
            next_token_logits = output[0, -1, :]
            next_token_id = next_token_logits.argmax().item()
            generated_ids = torch.cat(
                [generated_ids, torch.tensor([[next_token_id]], device=device)],
                dim=1
            )

            # EOS check
            if next_token_id == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print_flush(f"  Generated: \"{generated_text}\"")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return True


def main():
    print_flush("=" * 70)
    print_flush("SENRI INTEGRATION VERIFICATION")
    print_flush("=" * 70)
    print_flush("This script verifies that Senri (Japanese LLM) is correctly")
    print_flush("configured with OpenCALM tokenizer.")

    results = {}

    # 1. トークナイザーテスト
    results["tokenizer"] = test_tokenizer()

    # 2. 設定テスト
    results["config"] = test_config()

    # 3. モデル作成テスト
    results["model_creation"] = test_model_creation()

    # 4. フォワードパステスト
    results["forward_pass"] = test_forward_pass()

    # 5. テキスト生成テスト
    results["generation"] = test_generation()

    # サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print_flush(f"  {name:20} : {status}")

    print_flush("\n" + "=" * 70)
    if all_passed:
        print_flush("ALL TESTS PASSED - Senri is ready!")
    else:
        print_flush("SOME TESTS FAILED - Please check the errors above.")
    print_flush("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
