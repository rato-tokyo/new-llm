"""Analyze what the top singular vectors represent"""
import sys
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

import torch
import numpy as np
from transformers import AutoTokenizer

def analyze_singular_vectors(contexts, token_ids, label=""):
    """Analyze which tokens contribute to top singular vectors"""
    print(f"\n{'='*70}")
    print(f"SINGULAR VECTOR ANALYSIS - {label}")
    print(f"{'='*70}")

    # SVD
    U, S, Vt = torch.svd(contexts)

    print(f"\nTop 10 Singular Values:")
    for i, s in enumerate(S[:10]):
        print(f"  {i+1}: {s.item():.2f}")

    # Analyze top 3 singular vectors (Vt rows)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    for vec_idx in range(min(3, len(S))):
        print(f"\n{'='*70}")
        print(f"Singular Vector {vec_idx+1} (Singular Value: {S[vec_idx].item():.2f})")
        print(f"{'='*70}")

        # Project contexts onto this singular vector
        projections = contexts @ Vt[vec_idx, :]  # Shape: [num_tokens]

        # Find tokens with highest/lowest projections
        top_k = 10
        top_indices = torch.topk(projections, k=min(top_k, len(projections))).indices
        bottom_indices = torch.topk(projections, k=min(top_k, len(projections)), largest=False).indices

        print(f"\nTokens with HIGHEST projection onto SV{vec_idx+1}:")
        for rank, idx in enumerate(top_indices[:10]):
            token_id = token_ids[idx].item()
            token_text = tokenizer.decode([token_id])
            projection = projections[idx].item()
            print(f"  {rank+1}. Token {token_id} ('{token_text}'): projection = {projection:.3f}")

        print(f"\nTokens with LOWEST projection onto SV{vec_idx+1}:")
        for rank, idx in enumerate(bottom_indices[:10]):
            token_id = token_ids[idx].item()
            token_text = tokenizer.decode([token_id])
            projection = projections[idx].item()
            print(f"  {rank+1}. Token {token_id} ('{token_text}'): projection = {projection:.3f}")

        # Dimension analysis of the singular vector itself
        print(f"\nSingular Vector {vec_idx+1} Components (top 5 dimensions):")
        vec = Vt[vec_idx, :]
        top_dims = torch.topk(torch.abs(vec), k=5).indices
        for rank, dim in enumerate(top_dims):
            print(f"  Dim {dim.item()}: {vec[dim].item():.4f}")

    return U, S, Vt


if __name__ == "__main__":
    # This will be called from test_residual.py
    pass
