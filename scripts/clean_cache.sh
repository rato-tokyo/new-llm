#!/bin/bash
# Clean experiment caches while preserving tokenizer and datasets

echo "Cleaning experiment cache..."

# Remove ONLY experiment results (not tokenizer or datasets)
rm -f cache/context_cache.pt
rm -rf checkpoints/*.pt

echo "✓ Experiment cache cleaned"
echo ""
echo "Preserved:"
echo "  ✓ cache/tokenizer.json (if exists)"
echo "  ✓ HuggingFace dataset cache"
echo ""
echo "Deleted:"
echo "  ✗ cache/context_cache.pt (experiment results)"
echo "  ✗ checkpoints/*.pt (model checkpoints)"
