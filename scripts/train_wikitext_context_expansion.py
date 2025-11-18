#!/usr/bin/env python3
"""
WikiText-2 Context Vector Expansion Training

Progressive growing approach inspired by biological neural development.
Expands context_vector_dim while preserving learned parameters.
New dimensions are zero-initialized to maintain training stability.

Usage:
    # Pattern A: Freeze base dimensions (Transfer Learning)
    python scripts/train_wikitext_context_expansion.py \
        --base_checkpoint best_new_llm_wikitext_fp16_layers1.pt \
        --base_context_dim 256 \
        --expanded_context_dim 512 \
        --num_layers 1 \
        --freeze_base_dims

    # Pattern B: Fine-tune all dimensions (Standard)
    python scripts/train_wikitext_context_expansion.py \
        --base_checkpoint best_new_llm_wikitext_fp16_layers1.pt \
        --base_context_dim 256 \
        --expanded_context_dim 512 \
        --num_layers 1
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.utils.train_utils import print_git_info, print_gpu_info
from src.utils.config import NewLLML4Config
from src.models.context_vector_llm import ContextVectorLLM
from src.training.wikitext_dataset import load_wikitext_data
from src.training.context_expansion_trainer import ContextExpansionTrainer
from torch.utils.data import DataLoader
import time
import argparse


# For checkpoint compatibility - these classes must be defined even if not used
class LayerExperimentConfig(NewLLML4Config):
    """Layer experiment configuration (for pickle compatibility)

    This class is needed to load checkpoints saved from train_wikitext_fp16_layers.py
    """
    max_seq_length = 64
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512
    num_layers = 6  # Default, will be overridden
    context_vector_dim = 256
    dropout = 0.1
    num_epochs = 150
    weight_decay = 0.0
    gradient_clip = 1.0
    patience = 30
    use_amp = True

    def __init__(self, num_layers=6):
        super().__init__()
        self.num_layers = num_layers


class FP16Config(NewLLML4Config):
    """FP16 config (for pickle compatibility)

    This class is needed to load checkpoints saved from train_wikitext_fp16.py
    """
    max_seq_length = 64
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512
    num_layers = 6
    context_vector_dim = 256
    dropout = 0.1
    num_epochs = 50
    weight_decay = 0.0
    gradient_clip = 1.0
    patience = 15
    use_amp = True


class ContextExpansionConfig(NewLLML4Config):
    """Configuration for context vector expansion experiment

    Inherits L4 GPU optimization:
    - batch_size = 2048
    - learning_rate = 0.0008 (will be adjusted based on model size)
    - device = "cuda"
    """
    # データ関連（WikiText-2用）
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ
    embed_dim = 256
    hidden_dim = 512
    num_layers = 1  # Will be set via __init__
    context_vector_dim = 512  # Will be set via __init__ (expanded size)
    dropout = 0.1

    # 訓練ハイパーパラメータ
    num_epochs = 150
    weight_decay = 0.0
    gradient_clip = 1.0
    patience = 30

    # FP16設定
    use_amp = True

    def __init__(self, num_layers=1, expanded_context_dim=512):
        """Initialize config with specified parameters

        Args:
            num_layers: Number of layers in the model
            expanded_context_dim: Expanded context vector dimension
        """
        super().__init__()
        self.num_layers = num_layers
        self.context_vector_dim = expanded_context_dim


def expand_context_vector_weights(old_state_dict, old_context_dim, new_context_dim, num_layers):
    """Expand context vector dimensions while preserving learned weights

    Strategy (inspired by biological neural development):
    1. Copy existing weights to the first old_context_dim dimensions
    2. Initialize new dimensions with zeros (like new neurons)
    3. Maintains stability during training

    Args:
        old_state_dict: State dict from base model checkpoint
        old_context_dim: Original context vector dimension (e.g., 256)
        new_context_dim: Expanded context vector dimension (e.g., 512)
        num_layers: Number of layers in the model

    Returns:
        New state dict with expanded context dimensions
    """
    new_state_dict = {}

    print(f"\n🧠 Expanding context vector: {old_context_dim} → {new_context_dim}")
    print(f"   Strategy: Zero-padding new {new_context_dim - old_context_dim} dimensions\n")

    for key, value in old_state_dict.items():
        if 'context_proj' in key:
            # context_proj: (embed_dim + old_context, old_context) -> (embed_dim + new_context, new_context)
            if 'weight' in key:
                old_weight = value  # [old_context_dim, embed_dim + old_context_dim]
                embed_dim = old_weight.size(1) - old_context_dim

                # Create new weight: [new_context_dim, embed_dim + new_context_dim]
                new_weight = torch.zeros(new_context_dim, embed_dim + new_context_dim)

                # Copy old weights to top-left block
                new_weight[:old_context_dim, :embed_dim] = old_weight[:, :embed_dim]  # embed part
                new_weight[:old_context_dim, embed_dim:embed_dim+old_context_dim] = old_weight[:, embed_dim:]  # context part

                new_state_dict[key] = new_weight
                print(f"   ✓ {key}: {old_weight.shape} → {new_weight.shape}")

            elif 'bias' in key:
                old_bias = value  # [old_context_dim]
                new_bias = torch.zeros(new_context_dim)
                new_bias[:old_context_dim] = old_bias
                new_state_dict[key] = new_bias
                print(f"   ✓ {key}: {old_bias.shape} → {new_bias.shape}")

        elif 'layers' in key and 'fnn' in key:
            # FNN input layer: (embed_dim + old_context, hidden) -> (embed_dim + new_context, hidden)
            if 'weight' in key and '.0.weight' in key:  # First layer of FNN
                old_weight = value  # [hidden_dim, embed_dim + old_context_dim]
                embed_dim = old_weight.size(1) - old_context_dim
                hidden_dim = old_weight.size(0)

                # Create new weight: [hidden_dim, embed_dim + new_context_dim]
                new_weight = torch.zeros(hidden_dim, embed_dim + new_context_dim)

                # Copy old weights
                new_weight[:, :embed_dim] = old_weight[:, :embed_dim]  # embed part
                new_weight[:, embed_dim:embed_dim+old_context_dim] = old_weight[:, embed_dim:]  # context part

                new_state_dict[key] = new_weight
                print(f"   ✓ {key}: {old_weight.shape} → {new_weight.shape}")
            else:
                # Other FNN layers remain unchanged
                new_state_dict[key] = value

        elif 'context_update' in key:
            # Context update: (hidden, old_context) -> (hidden, new_context)
            if 'weight' in key:
                old_weight = value  # [old_context_dim, hidden_dim]
                hidden_dim = old_weight.size(1)

                # Create new weight: [new_context_dim, hidden_dim]
                new_weight = torch.zeros(new_context_dim, hidden_dim)
                new_weight[:old_context_dim, :] = old_weight

                new_state_dict[key] = new_weight
                print(f"   ✓ {key}: {old_weight.shape} → {new_weight.shape}")

            elif 'bias' in key:
                old_bias = value  # [old_context_dim]
                new_bias = torch.zeros(new_context_dim)
                new_bias[:old_context_dim] = old_bias
                new_state_dict[key] = new_bias
                print(f"   ✓ {key}: {old_bias.shape} → {new_bias.shape}")

        elif 'context_norm' in key:
            # LayerNorm for context vector
            if 'weight' in key or 'bias' in key:
                old_param = value  # [old_context_dim]
                new_param = torch.zeros(new_context_dim) if 'bias' in key else torch.ones(new_context_dim)
                new_param[:old_context_dim] = old_param
                new_state_dict[key] = new_param
                print(f"   ✓ {key}: {old_param.shape} → {new_param.shape}")

        elif 'forget_gate' in key or 'input_gate' in key:
            # Gates: (hidden, old_context) -> (hidden, new_context)
            if 'weight' in key:
                old_weight = value  # [old_context_dim, hidden_dim]
                hidden_dim = old_weight.size(1)

                # Create new weight: [new_context_dim, hidden_dim]
                new_weight = torch.zeros(new_context_dim, hidden_dim)
                new_weight[:old_context_dim, :] = old_weight

                new_state_dict[key] = new_weight
                print(f"   ✓ {key}: {old_weight.shape} → {new_weight.shape}")

            elif 'bias' in key:
                old_bias = value  # [old_context_dim]
                new_bias = torch.zeros(new_context_dim)
                new_bias[:old_context_dim] = old_bias
                new_state_dict[key] = new_bias
                print(f"   ✓ {key}: {old_bias.shape} → {new_bias.shape}")

        else:
            # All other parameters remain unchanged
            new_state_dict[key] = value

    print(f"\n✓ Context vector expansion complete!")
    print(f"   Old dimensions preserved: {old_context_dim}")
    print(f"   New dimensions (zero-init): {new_context_dim - old_context_dim}")

    return new_state_dict


def main():
    """Context vector expansion training"""
    parser = argparse.ArgumentParser(description='WikiText-2 Context Vector Expansion Training')
    parser.add_argument('--base_checkpoint', type=str, required=True,
                        help='Base checkpoint file (e.g., best_new_llm_wikitext_fp16_layers1.pt)')
    parser.add_argument('--base_context_dim', type=int, required=True,
                        help='Base context vector dimension (e.g., 256)')
    parser.add_argument('--expanded_context_dim', type=int, required=True,
                        help='Expanded context vector dimension (e.g., 512)')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='Number of layers (e.g., 1)')
    parser.add_argument('--freeze_base_dims', action='store_true',
                        help='Freeze base dimensions during training (only train new dimensions)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"WikiText-2 Context Vector Expansion Training")
    print(f"Biological Inspired: Progressive Neural Growth")
    print("="*80)

    # Git version information
    try:
        import subprocess
        git_commit_short = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__) + '/..'
        ).decode().strip()
        git_date = subprocess.check_output(
            ['git', 'log', '-1', '--format=%cd', '--date=short'],
            cwd=os.path.dirname(__file__) + '/..'
        ).decode().strip()
        print(f"\n📌 Git Version: {git_commit_short} ({git_date})")
    except Exception:
        print(f"\n📌 Git Version: Unknown")

    print("="*80)

    # GPU check
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU not available! This training requires GPU.")

    # Configuration
    config = ContextExpansionConfig(
        num_layers=args.num_layers,
        expanded_context_dim=args.expanded_context_dim
    )

    print(f"\n🧠 Expansion Configuration:")
    print(f"  Base checkpoint: {args.base_checkpoint}")
    print(f"  Base context dim: {args.base_context_dim}")
    print(f"  Expanded context dim: {args.expanded_context_dim}")
    print(f"  Expansion ratio: {args.expanded_context_dim / args.base_context_dim:.1f}x")
    print(f"  New dimensions: {args.expanded_context_dim - args.base_context_dim} (zero-initialized)")
    print(f"  Num layers: {args.num_layers}")
    print(f"\n🎯 Training Strategy:")
    if args.freeze_base_dims:
        print(f"  Mode: Freeze Base (Transfer Learning)")
        print(f"  ├─ Base dims (0:{args.base_context_dim}): FROZEN 🔒")
        print(f"  └─ New dims ({args.base_context_dim}:{args.expanded_context_dim}): TRAINABLE ✓")
    else:
        print(f"  Mode: Fine-tune All (Standard)")
        print(f"  └─ All dims (0:{args.expanded_context_dim}): TRAINABLE ✓")

    # Device info
    print(f"\n🖥️  Device Information:")
    print(f"  Device: CUDA (GPU)")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {gpu_memory:.1f} GB")
    print(f"  FP16 Mixed Precision: ENABLED ✓")

    print(f"\n⚙️  Training Configuration:")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"\n{'='*80}\n")

    # Load base checkpoint
    checkpoint_path = os.path.join("checkpoints", args.base_checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Base checkpoint not found: {checkpoint_path}")

    print(f"📂 Loading base checkpoint: {checkpoint_path}")
    base_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    base_state_dict = base_checkpoint['model_state_dict']

    # Expand context vector weights
    expanded_state_dict = expand_context_vector_weights(
        base_state_dict,
        old_context_dim=args.base_context_dim,
        new_context_dim=args.expanded_context_dim,
        num_layers=args.num_layers
    )

    # Load data
    print("\nLoading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Create expanded model
    print(f"\nCreating expanded model...")
    print(f"  Context Vector Dim: {args.base_context_dim} → {args.expanded_context_dim}")
    print(f"  Num Layers: {args.num_layers}")

    model = ContextVectorLLM(config)

    # Load expanded weights
    model.load_state_dict(expanded_state_dict)
    print(f"✓ Expanded weights loaded successfully")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Create trainer with freeze mode
    freeze_suffix = "_freeze" if args.freeze_base_dims else "_finetune"
    model_name = f"new_llm_wikitext_ctx{args.expanded_context_dim}_layers{args.num_layers}_expanded{freeze_suffix}"

    trainer = ContextExpansionTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name=model_name,
        use_amp=config.use_amp,
        base_context_dim=args.base_context_dim,
        freeze_base_dims=args.freeze_base_dims
    )

    # Print parameter analysis
    trainer.print_trainable_params()

    # Train
    print(f"\n🚀 Starting context expansion training...")
    print(f"   Biologically inspired progressive growth")
    if args.freeze_base_dims:
        print(f"   🔒 Freeze Base Mode: Only new dimensions will learn")
    else:
        print(f"   🔓 Fine-tune Mode: All dimensions will be optimized")
    print()

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("Context Vector Expansion Training Completed!")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Checkpoint saved: checkpoints/best_{model_name}.pt")

    # Performance summary
    if trainer.val_ppls:
        best_val_ppl = min(trainer.val_ppls)
        best_epoch = trainer.val_ppls.index(best_val_ppl) + 1

        print(f"\n📊 Final Results:")
        print(f"  Configuration: {args.num_layers} layers, ctx {args.base_context_dim}→{args.expanded_context_dim}")
        print(f"  Parameters: {num_params/1e6:.2f}M")
        print(f"  Best Val PPL: {best_val_ppl:.2f} (Epoch {best_epoch})")

        # Note about base performance
        print(f"\n💡 Compare with base model (ctx={args.base_context_dim}) to see")
        print(f"   if progressive expansion improved performance!")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
