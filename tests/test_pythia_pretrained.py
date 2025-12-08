#!/usr/bin/env python3
"""
Pythia-70mが事前学習済みかどうかを確認するテスト
"""

import sys
sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from src.utils.io import print_flush
from src.utils.training import get_device


def main():
    device = get_device()
    print_flush(f"Device: {device}")

    # Load model
    print_flush("\nLoading EleutherAI/pythia-70m...")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    # Test 1: Simple text generation
    print_flush("\n" + "=" * 70)
    print_flush("TEST 1: Text Generation")
    print_flush("=" * 70)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print_flush(f"Prompt: {prompt}")
    print_flush(f"Generated: {generated}")

    # Check if output makes sense
    if "Paris" in generated:
        print_flush("✓ Model knows Paris is the capital of France!")
    else:
        print_flush("⚠️ Model did NOT generate 'Paris' - may not be pretrained correctly")

    # Test 2: PPL on simple sentence
    print_flush("\n" + "=" * 70)
    print_flush("TEST 2: PPL on Simple Sentence")
    print_flush("=" * 70)

    test_sentences = [
        "The cat sat on the mat.",
        "Hello, how are you today?",
        "Machine learning is a field of artificial intelligence.",
    ]

    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            ppl = torch.exp(torch.tensor(loss)).item()
        print_flush(f"  '{sentence[:40]}...' -> PPL: {ppl:.1f}")

    # Test 3: Random text PPL (should be high)
    print_flush("\n" + "=" * 70)
    print_flush("TEST 3: Random Token PPL (should be high)")
    print_flush("=" * 70)

    random_tokens = torch.randint(0, tokenizer.vocab_size, (1, 100)).to(device)
    with torch.no_grad():
        outputs = model(random_tokens, labels=random_tokens)
        loss = outputs.loss.item()
        ppl = torch.exp(torch.tensor(loss)).item()
    print_flush(f"  Random tokens -> PPL: {ppl:.1f}")

    # Test 4: Check model weights
    print_flush("\n" + "=" * 70)
    print_flush("TEST 4: Model Weight Statistics")
    print_flush("=" * 70)

    # Check embedding weights
    embed_weights = model.gpt_neox.embed_in.weight
    print_flush("  Embedding weight stats:")
    print_flush(f"    Mean: {embed_weights.mean().item():.6f}")
    print_flush(f"    Std:  {embed_weights.std().item():.6f}")
    print_flush(f"    Min:  {embed_weights.min().item():.6f}")
    print_flush(f"    Max:  {embed_weights.max().item():.6f}")

    # If weights are all similar or zero, it's not pretrained
    if embed_weights.std().item() < 0.01:
        print_flush("  ⚠️ WARNING: Embedding weights have very low variance - may not be pretrained!")
    else:
        print_flush("  ✓ Embedding weights look normal")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
