#!/usr/bin/env python3
"""Inspect checkpoint file contents"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch

# Import config classes so they can be unpickled
try:
    from train_ultrachat import UltraChatTrainConfig
except ImportError:
    pass

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best_new_llm_ultrachat_layers1.pt'

print(f'📂 Inspecting: {checkpoint_path}')
print('='*60)

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f'\n✓ Checkpoint loaded successfully')
    print(f'\n📋 Keys in checkpoint:')
    for key in checkpoint.keys():
        val = checkpoint[key]
        if isinstance(val, dict):
            print(f'   {key}: dict with {len(val)} items')
        elif isinstance(val, torch.Tensor):
            print(f'   {key}: tensor {val.shape}')
        else:
            print(f'   {key}: {type(val).__name__}')

    print(f'\n📊 Details:')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f'   Model parameters: {total_params:,}')

    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f'   Config type: {type(config).__name__}')
        if hasattr(config, 'num_layers'):
            print(f'   Num layers: {config.num_layers}')
        if hasattr(config, 'vocab_size'):
            print(f'   Vocab size: {config.vocab_size}')

    if 'tokenizer' in checkpoint:
        tokenizer = checkpoint['tokenizer']
        print(f'   Tokenizer type: {type(tokenizer).__name__}')
        if hasattr(tokenizer, 'word2idx'):
            print(f'   Tokenizer vocab: {len(tokenizer.word2idx)} words')
    else:
        print(f'   ⚠️  No tokenizer in checkpoint')

    print('\n' + '='*60)
    print('✅ Inspection complete')

except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()
