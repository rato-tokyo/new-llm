#!/usr/bin/env python3
"""
Interactive chat interface for New-LLM

Usage:
    python scripts/chat.py --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt
    python scripts/chat.py --checkpoint checkpoints/best_new_llm_dolly_layers1.pt --temperature 0.7
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from src.models.context_vector_llm import ContextVectorLLM
from src.training.dataset import SimpleTokenizer
from src.inference.generator import TextGenerator

# Import config classes for checkpoint loading
# These are needed because configs are pickled in checkpoints
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from train_ultrachat import UltraChatTrainConfig
except ImportError:
    pass  # Config may not be needed if not in checkpoint


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        (model, tokenizer, config) tuple
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    config = checkpoint['config']

    print(f"✓ Config loaded:")
    print(f"   Layers: {config.num_layers}")
    print(f"   Context dim: {config.context_vector_dim}")
    print(f"   Vocab size: {config.vocab_size}")

    # Extract or load tokenizer
    if 'tokenizer' in checkpoint:
        tokenizer = checkpoint['tokenizer']
        print(f"✓ Tokenizer loaded from checkpoint")
    else:
        # Try to load pre-built tokenizer
        tokenizer_path = "checkpoints/ultrachat_tokenizer.pkl"
        if os.path.exists(tokenizer_path):
            print(f"⚠️  Tokenizer not in checkpoint, loading from {tokenizer_path}")
            import pickle
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"✓ Tokenizer loaded: {len(tokenizer.word2idx)} words")
        else:
            print(f"❌ Tokenizer not found!")
            print(f"   Please run: python scripts/save_ultrachat_tokenizer.py")
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    # Create model
    model = ContextVectorLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params:,} parameters")

    return model, tokenizer, config


def chat_loop(generator, max_length=100, temperature=0.8, top_p=0.9):
    """Interactive chat loop

    Args:
        generator: TextGenerator instance
        max_length: Maximum tokens to generate per response
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    """
    print("\n" + "="*80)
    print("🤖 New-LLM Chat Interface")
    print("="*80)
    print(f"⚙️  Settings:")
    print(f"   Max length: {max_length}")
    print(f"   Temperature: {temperature}")
    print(f"   Top-p: {top_p}")
    print(f"\n💡 Type 'exit' or 'quit' to end the conversation")
    print(f"💡 Type 'reset' to clear conversation context")
    print(f"💡 Type 'settings' to change generation settings")
    print("="*80 + "\n")

    context = ""
    turn = 0

    while True:
        # Get user input
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        # Handle commands
        if user_input.lower() in ['exit', 'quit']:
            print("\n👋 Goodbye!")
            break

        if user_input.lower() == 'reset':
            context = ""
            turn = 0
            print("🔄 Conversation context reset\n")
            continue

        if user_input.lower() == 'settings':
            print(f"\nCurrent settings:")
            print(f"  Max length: {max_length}")
            print(f"  Temperature: {temperature}")
            print(f"  Top-p: {top_p}\n")
            continue

        if not user_input.strip():
            continue

        # Generate response
        turn += 1
        print(f"Assistant: ", end='', flush=True)

        try:
            response, context = generator.chat(
                user_input,
                context=context,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            print(response)
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            print(f"   Context length: {len(context)}")
            # Reset context if too long
            if len(context) > 1000:
                print("   Context too long, resetting...")
                context = ""

        print()  # Empty line for readability


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Chat with New-LLM')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum tokens to generate (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling top-p (default: 0.9)')

    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    try:
        model, tokenizer, config = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Create generator
    generator = TextGenerator(model, tokenizer, args.device)

    # Start chat loop
    try:
        chat_loop(
            generator,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
    except Exception as e:
        print(f"\n❌ Chat loop error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
