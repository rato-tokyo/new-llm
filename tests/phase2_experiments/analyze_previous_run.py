"""Analyze previous run by re-running forward pass only"""
import sys
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

import torch
from transformers import AutoTokenizer
from src.models.new_llm_residual import NewLLMResidual
import config

# Load configuration
cfg = config.Residual4Layer16Ctx()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load manual validation data
val_ids = torch.load('/Users/sakajiritomoyoshi/Desktop/git/new-llm/cache/manual_val_tokens.pt')

print("="*70)
print("ANALYZING VALIDATION DATA (23 tokens)")
print("="*70)
print(f"Val tokens: {val_ids.tolist()}")
print(f"\nDecoded tokens:")
for i, tid in enumerate(val_ids):
    print(f"  {i}: Token {tid.item()} = '{tokenizer.decode([tid])}'")

# Create model (we'll use random weights, but structure is same)
print(f"\n{'='*70}")
print("NOTE: Using random weights (previous weights not saved)")
print("To get exact analysis, we need to save model weights during training")
print(f"{'='*70}\n")

hidden_dim = cfg.embed_dim + cfg.context_dim
model = NewLLMResidual(
    vocab_size=tokenizer.vocab_size,
    embed_dim=cfg.embed_dim,
    context_dim=cfg.context_dim,
    hidden_dim=hidden_dim,
    layer_structure=cfg.layer_structure
)
model.eval()

# Forward pass to get contexts
with torch.no_grad():
    token_embeds = model.token_embedding(val_ids.unsqueeze(0))
    token_embeds = model.embed_norm(token_embeds).squeeze(0)

    contexts = []
    context = torch.zeros(1, model.context_dim)

    for token_embed in token_embeds:
        context = model._update_context_one_step(token_embed.unsqueeze(0), context)
        contexts.append(context.squeeze(0))

    contexts = torch.stack(contexts)

# Singular Value Decomposition
print("\n" + "="*70)
print("SINGULAR VALUE DECOMPOSITION")
print("="*70)

U, S, V = torch.svd(contexts)

print(f"\nAll Singular Values:")
for i, s in enumerate(S):
    percentage = (s / S.sum() * 100).item()
    print(f"  SV{i+1}: {s.item():.4f} ({percentage:.1f}%)")

# Analyze top 2 singular vectors
for vec_idx in range(min(2, len(S))):
    print(f"\n{'='*70}")
    print(f"Singular Vector {vec_idx+1} (Value: {S[vec_idx].item():.4f})")
    print(f"{'='*70}")

    # Project contexts onto this singular vector
    projections = contexts @ V[:, vec_idx]

    print(f"\nAll token projections onto SV{vec_idx+1}:")
    for i, (tid, proj) in enumerate(zip(val_ids, projections)):
        token_text = tokenizer.decode([tid])
        print(f"  {i}: Token {tid.item():5d} '{token_text:20s}' projection={proj.item():+.4f}")

    # Show which dimensions of the singular vector are important
    print(f"\nSingular Vector {vec_idx+1} dimension weights (all 16 dims):")
    vec = V[:, vec_idx]
    for dim in range(len(vec)):
        print(f"  Dim {dim:2d}: {vec[dim].item():+.4f}")

print("\n" + "="*70)
print("To get exact results from previous training:")
print("1. Add model checkpoint saving in test_residual.py")
print("2. Re-run experiment with checkpoint saving enabled")
print("3. Load checkpoint and analyze")
print("="*70)
