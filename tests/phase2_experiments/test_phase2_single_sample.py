"""Phase 2 Training Test on Single Sample

Test both Sequential and Layer-wise architectures:
1. Phase 1: Train to find fixed-point contexts
2. Phase 2: Train token prediction using fixed contexts
3. Evaluate perplexity and accuracy

Uses a single UltraChat sample for initial testing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def load_single_sample(tokenizer_model='gpt2', max_length=512):
    """Load a single UltraChat sample and tokenize it"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    # Get first sample
    sample = dataset[0]

    # Concatenate all messages
    text = ""
    for msg in sample['messages']:
        text += msg['content'] + " "

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)

    return torch.tensor(tokens), text


def phase1_train_fixed_points(model, config, token_ids, num_epochs=3):
    """Phase 1: Train model to learn fixed-point contexts"""
    num_tokens = len(token_ids)
    print(f"\n{'='*70}")
    print(f"Phase 1: Training Fixed-Point Contexts ({num_tokens} tokens)")
    print(f"{'='*70}")

    optimizer = optim.Adam(model.parameters(), lr=config.phase1_learning_rate)

    model.train()
    input_ids = token_ids.unsqueeze(0)
    token_embeds = model.token_embedding(input_ids)
    if hasattr(model, 'embed_norm'):
        token_embeds = model.embed_norm(token_embeds)
    token_embeds = token_embeds.squeeze(0)

    # Train each token to find its fixed point
    for epoch in range(num_epochs):
        total_loss = 0
        for t, token_embed in enumerate(token_embeds):
            context = torch.zeros(1, config.context_dim)

            # Fixed-point iteration
            for iteration in range(50):
                optimizer.zero_grad()
                token_detached = token_embed.unsqueeze(0).detach()
                context_detached = context.detach()

                context_new = model._update_context_one_step(token_detached, context_detached)

                loss = torch.nn.functional.mse_loss(context_new, context.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                context = context_new.detach()
                total_loss += loss.item()

                if loss.item() < config.phase1_convergence_threshold:
                    break

        avg_loss = total_loss / num_tokens
        print(f"  Epoch {epoch+1}/{num_epochs}: Avg Loss = {avg_loss:.6f}")

    print(f"  Phase 1 Training Complete\n")


def phase1_compute_fixed_contexts(model, config, token_ids):
    """Compute and cache fixed-point contexts for all tokens"""
    num_tokens = len(token_ids)
    print(f"Computing Fixed-Point Contexts...")

    input_ids = token_ids.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        fixed_contexts, converged, num_iters = model.get_fixed_point_context(
            input_ids,
            max_iterations=50,
            tolerance=config.phase1_convergence_threshold,
            warmup_iterations=0
        )

    converged_count = converged.sum().item()
    avg_iters = num_iters.float().mean().item()

    print(f"  Converged: {converged_count}/{num_tokens} ({converged_count/num_tokens:.1%})")
    print(f"  Avg Iterations: {avg_iters:.1f}\n")

    if converged_count < num_tokens * 0.95:
        print(f"  ⚠️  Warning: Less than 95% converged!")

    return fixed_contexts.squeeze(0), converged.squeeze(0)


def phase2_train_token_prediction(model, config, token_ids, fixed_contexts, num_epochs=10):
    """Phase 2: Train token prediction using fixed contexts"""
    num_tokens = len(token_ids)
    print(f"\n{'='*70}")
    print(f"Phase 2: Training Token Prediction ({num_tokens} tokens)")
    print(f"{'='*70}")

    optimizer = optim.Adam(model.parameters(), lr=config.phase2_learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()

    # Prepare data: input tokens and target tokens (shifted by 1)
    # For token prediction: given tokens[0:t], predict tokens[t]
    # We use fixed context for each position

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for t in range(num_tokens - 1):  # -1 because we need next token as target
            optimizer.zero_grad()

            # Get fixed context and current token embedding
            context = fixed_contexts[t].unsqueeze(0)  # [1, context_dim]
            current_token_id = token_ids[t].unsqueeze(0).unsqueeze(0)  # [1, 1]
            token_embed = model.token_embedding(current_token_id)
            if hasattr(model, 'embed_norm'):
                token_embed = model.embed_norm(token_embed)
            token_embed = token_embed.squeeze(1)  # [1, embed_dim]

            # Get hidden state using shared method
            _, hidden = model._update_context_one_step(token_embed, context, return_hidden=True)

            # Predict next token
            logits = model.token_output(hidden)  # [1, vocab_size]

            # Target: next token
            target = token_ids[t + 1].unsqueeze(0)  # [1]

            # Compute loss
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.phase2_gradient_clip)
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            pred = logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += 1

        avg_loss = total_loss / (num_tokens - 1)
        accuracy = correct / total * 100
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, PPL = {perplexity:.2f}, Acc = {accuracy:.1f}%")

    print(f"  Phase 2 Training Complete\n")


def phase2_evaluate(model, config, token_ids, fixed_contexts):
    """Evaluate token prediction performance"""
    num_tokens = len(token_ids)
    print(f"\n{'='*70}")
    print(f"Phase 2: Evaluation")
    print(f"{'='*70}")

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for t in range(num_tokens - 1):
            # Get fixed context and current token embedding
            context = fixed_contexts[t].unsqueeze(0)
            current_token_id = token_ids[t].unsqueeze(0).unsqueeze(0)
            token_embed = model.token_embedding(current_token_id)
            if hasattr(model, 'embed_norm'):
                token_embed = model.embed_norm(token_embed)
            token_embed = token_embed.squeeze(1)

            # Get hidden state
            _, hidden = model._update_context_one_step(token_embed, context, return_hidden=True)

            # Predict next token
            logits = model.token_output(hidden)

            # Target: next token
            target = token_ids[t + 1].unsqueeze(0)

            # Compute loss
            loss = criterion(logits, target)
            total_loss += loss.item()

            # Compute accuracy
            pred = logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += 1

    avg_loss = total_loss / (num_tokens - 1)
    accuracy = correct / total * 100
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"  Final Results:")
    print(f"    Loss: {avg_loss:.4f}")
    print(f"    Perplexity: {perplexity:.2f}")
    print(f"    Accuracy: {accuracy:.1f}%")
    print()

    return avg_loss, perplexity, accuracy


def test_architecture(architecture_name, config, token_ids):
    """Test a single architecture (Sequential or Layer-wise)"""
    print(f"\n{'='*70}")
    print(f"Testing {architecture_name} Architecture")
    print(f"{'='*70}")

    # Create model
    if architecture_name == "Sequential":
        model = NewLLMSequential(config)
    else:
        model = NewLLMLayerwise(config)

    print(f"Model Parameters: {model.count_parameters():,}")

    # Phase 1: Train fixed-point contexts
    phase1_train_fixed_points(model, config, token_ids, num_epochs=3)

    # Compute and cache fixed contexts
    fixed_contexts, converged = phase1_compute_fixed_contexts(model, config, token_ids)

    if converged.sum() < len(token_ids) * 0.95:
        print(f"❌ {architecture_name} failed Phase 1: Less than 95% converged")
        return None

    print(f"✅ {architecture_name} passed Phase 1")

    # Phase 2: Train token prediction
    phase2_train_token_prediction(model, config, token_ids, fixed_contexts, num_epochs=10)

    # Evaluate
    loss, ppl, acc = phase2_evaluate(model, config, token_ids, fixed_contexts)

    return {
        'architecture': architecture_name,
        'loss': loss,
        'perplexity': ppl,
        'accuracy': acc
    }


def main():
    """Test Phase 2 on single sample"""
    print("="*70)
    print("Phase 2 Test: Single Sample")
    print("Testing Sequential vs Layer-wise")
    print("="*70)

    # Load single sample
    print("\nLoading UltraChat sample...")
    token_ids, text = load_single_sample(max_length=512)
    num_tokens = len(token_ids)
    print(f"Loaded {num_tokens} tokens")
    print(f"Text preview: {text[:200]}...\n")

    results = []

    # Test Sequential
    config_seq = Small2LayerSequentialConfig()
    config_seq.vocab_size = 50259
    result_seq = test_architecture("Sequential", config_seq, token_ids)
    if result_seq:
        results.append(result_seq)

    # Test Layer-wise
    config_layer = Small2LayerLayerwiseConfig()
    config_layer.vocab_size = 50259
    result_layer = test_architecture("Layer-wise", config_layer, token_ids)
    if result_layer:
        results.append(result_layer)

    # Final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n{'Architecture':<15} {'Loss':<10} {'Perplexity':<12} {'Accuracy':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['architecture']:<15} {r['loss']:<10.4f} {r['perplexity']:<12.2f} {r['accuracy']:<10.1f}%")

    print("\n" + "="*70)

    # Determine winner
    if len(results) == 2:
        if results[0]['perplexity'] < results[1]['perplexity']:
            print(f"\n🏆 Winner: {results[0]['architecture']} (Lower Perplexity)")
        elif results[1]['perplexity'] < results[0]['perplexity']:
            print(f"\n🏆 Winner: {results[1]['architecture']} (Lower Perplexity)")
        else:
            print(f"\n🤝 Tie: Both architectures have similar performance")

    print("="*70)


if __name__ == "__main__":
    main()
