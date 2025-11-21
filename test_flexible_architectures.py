"""Test Flexible Architecture: Compare Sequential, Layer-wise, and Mixed

Test configurations:
1. Sequential [4]: 4 layers, single context update at end
2. Layer-wise [1, 1, 1, 1]: 4 layers, context update after each
3. Mixed [2, 2]: 2 blocks of 2-layer sequential (your proposed architecture)

No config classes needed - everything specified as simple lists.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from src.models.new_llm_flexible import NewLLMFlexible


def load_single_sample(tokenizer_model='gpt2', max_length=512):
    """Load a single UltraChat sample"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    sample = dataset[0]
    text = ""
    for msg in sample['messages']:
        text += msg['content'] + " "

    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)
    return torch.tensor(tokens), text


def phase1_train(model, token_ids, learning_rate=0.0001, num_epochs=3, threshold=0.01):
    """Phase 1: Train fixed-point contexts"""
    print(f"\nPhase 1: Training Fixed-Point Contexts ({len(token_ids)} tokens)")
    print("="*70)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    input_ids = token_ids.unsqueeze(0)
    token_embeds = model.token_embedding(input_ids)
    token_embeds = model.embed_norm(token_embeds)
    token_embeds = token_embeds.squeeze(0)

    for epoch in range(num_epochs):
        total_loss = 0
        for t, token_embed in enumerate(token_embeds):
            context = torch.zeros(1, model.context_dim)

            for iteration in range(50):
                optimizer.zero_grad()
                context_new = model._update_context_one_step(
                    token_embed.unsqueeze(0).detach(),
                    context.detach()
                )

                loss = torch.nn.functional.mse_loss(context_new, context.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                context = context_new.detach()
                total_loss += loss.item()

                if loss.item() < threshold:
                    break

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {total_loss/len(token_ids):.6f}")

    print("Phase 1 Complete\n")


def compute_fixed_contexts(model, token_ids, threshold=0.01):
    """Compute fixed-point contexts"""
    print("Computing Fixed-Point Contexts...")

    model.eval()
    with torch.no_grad():
        contexts, converged, iters = model.get_fixed_point_context(
            token_ids.unsqueeze(0),
            max_iterations=50,
            tolerance=threshold,
            warmup_iterations=0
        )

    conv_count = converged.sum().item()
    print(f"  Converged: {conv_count}/{len(token_ids)} ({conv_count/len(token_ids):.1%})")
    print(f"  Avg Iterations: {iters.float().mean():.1f}\n")

    return contexts.squeeze(0), converged.squeeze(0)


def phase2_train(model, token_ids, fixed_contexts, learning_rate=0.0001, num_epochs=10):
    """Phase 2: Train token prediction"""
    print(f"Phase 2: Training Token Prediction ({len(token_ids)} tokens)")
    print("="*70)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0

        for t in range(len(token_ids) - 1):
            optimizer.zero_grad()

            context = fixed_contexts[t].unsqueeze(0)
            token_id = token_ids[t].unsqueeze(0).unsqueeze(0)
            token_embed = model.token_embedding(token_id)
            token_embed = model.embed_norm(token_embed).squeeze(1)

            _, hidden = model._update_context_one_step(token_embed, context, return_hidden=True)
            logits = model.token_output(hidden)
            target = token_ids[t + 1].unsqueeze(0)

            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == target).sum().item()

        avg_loss = total_loss / (len(token_ids) - 1)
        acc = correct / (len(token_ids) - 1) * 100
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, PPL = {ppl:.2f}, Acc = {acc:.1f}%")

    print("Phase 2 Complete\n")


def evaluate(model, token_ids, fixed_contexts):
    """Evaluate model"""
    print("Evaluation")
    print("="*70)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0

    with torch.no_grad():
        for t in range(len(token_ids) - 1):
            context = fixed_contexts[t].unsqueeze(0)
            token_id = token_ids[t].unsqueeze(0).unsqueeze(0)
            token_embed = model.token_embedding(token_id)
            token_embed = model.embed_norm(token_embed).squeeze(1)

            _, hidden = model._update_context_one_step(token_embed, context, return_hidden=True)
            logits = model.token_output(hidden)
            target = token_ids[t + 1].unsqueeze(0)

            loss = criterion(logits, target)
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == target).sum().item()

    avg_loss = total_loss / (len(token_ids) - 1)
    acc = correct / (len(token_ids) - 1) * 100
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Accuracy: {acc:.1f}%\n")

    return avg_loss, ppl, acc


def test_architecture(name, layer_structure, token_ids, vocab_size=50259,
                     embed_dim=256, context_dim=256, hidden_dim=256):
    """Test a single architecture configuration"""
    print("\n" + "="*70)
    print(f"Testing: {name}")
    print(f"Structure: {layer_structure}")
    print("="*70)

    # Create model
    model = NewLLMFlexible(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        layer_structure=layer_structure,
        dropout=0.1
    )

    print(f"Architecture: {model.get_architecture_description()}")
    print(f"Parameters: {model.count_parameters():,}\n")

    # Phase 1
    phase1_train(model, token_ids, learning_rate=0.0001, num_epochs=3, threshold=0.01)
    fixed_contexts, converged = compute_fixed_contexts(model, token_ids, threshold=0.01)

    if converged.sum() < len(token_ids) * 0.95:
        print(f"❌ {name} failed (< 95% converged)")
        return None

    print(f"✅ {name} passed Phase 1\n")

    # Phase 2
    phase2_train(model, token_ids, fixed_contexts, learning_rate=0.0001, num_epochs=10)

    # Evaluate
    loss, ppl, acc = evaluate(model, token_ids, fixed_contexts)

    return {
        'name': name,
        'structure': layer_structure,
        'loss': loss,
        'perplexity': ppl,
        'accuracy': acc,
        'params': model.count_parameters()
    }


def main():
    """Compare Sequential, Layer-wise, and Mixed architectures"""
    print("="*70)
    print("Flexible Architecture Comparison")
    print("No config classes - just simple lists!")
    print("="*70)

    # Load data
    print("\nLoading UltraChat sample...")
    token_ids, text = load_single_sample(max_length=512)
    print(f"Loaded {len(token_ids)} tokens")
    print(f"Preview: {text[:150]}...\n")

    # Test configurations
    configs = [
        ("Sequential [4]", [4]),
        ("Layer-wise [1,1,1,1]", [1, 1, 1, 1]),
        ("Mixed [2,2]", [2, 2]),  # Your proposed architecture!
    ]

    results = []

    for name, structure in configs:
        result = test_architecture(name, structure, token_ids)
        if result:
            results.append(result)

    # Final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n{'Architecture':<25} {'PPL':<10} {'Acc':<10} {'Params':<12}")
    print("-" * 70)

    for r in results:
        struct_str = str(r['structure'])
        print(f"{r['name']:<25} {r['perplexity']:<10.2f} {r['accuracy']:<10.1f}% {r['params']:<12,}")

    print("\n" + "="*70)

    # Determine winner
    if results:
        best = min(results, key=lambda x: x['perplexity'])
        print(f"\n🏆 Winner: {best['name']} (PPL = {best['perplexity']:.2f})")

    print("="*70)


if __name__ == "__main__":
    main()
