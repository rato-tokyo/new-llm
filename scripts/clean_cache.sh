#!/bin/bash
# Clean experiment caches while preserving tokenizer and datasets

echo "Cleaning experiment cache..."

# Remove ONLY experiment results (not tokenizer or datasets)
rm -f cache/contexts/context_cache.pt
rm -f checkpoints/*.pt 2>/dev/null || true

echo "✓ Experiment cache cleaned"
echo ""
echo "Preserved:"
echo "  ✓ cache/tokenizer/ (tokenizer cache)"
echo "  ✓ HuggingFace dataset cache"
echo ""
echo "Deleted:"
echo "  ✗ cache/contexts/context_cache.pt (experiment results)"
echo "  ✗ checkpoints/*.pt (model checkpoints)"
