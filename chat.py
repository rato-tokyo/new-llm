#!/usr/bin/env python3
"""
New-LLM Chat Interface

Simple chat interface that loads everything from checkpoint.

Usage:
    python chat.py --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt
    python chat.py --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt --temperature 0.9
"""

import argparse
import torch

from src.models.context_vector_llm import ContextVectorLLM
from src.inference.generator import TextGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with New-LLM')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    # Generation parameters
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum tokens to generate (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Sampling temperature (default: 0.9, higher = more random)')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Nucleus sampling top-p (default: 0.95)')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                       help='Repetition penalty (default: 1.2, prevents loops)')

    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """Load model, tokenizer, and config from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        (model, tokenizer, config) tuple
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract components
    config = checkpoint['config']
    print(f"✓ Config loaded:")
    print(f"   Layers: {config.num_layers}")
    print(f"   Context dim: {config.context_vector_dim}")
    print(f"   Vocab size: {config.vocab_size}")

    # Check for tokenizer
    if 'tokenizer' not in checkpoint:
        raise ValueError(
            f"❌ Checkpoint does not contain tokenizer!\n"
            f"   This checkpoint was created with an old version.\n"
            f"   Please retrain with: python train.py --dataset ultrachat --epochs 50"
        )

    tokenizer = checkpoint['tokenizer']
    print(f"✓ Tokenizer loaded: {len(tokenizer.word2idx)} words")

    # Create and load model
    model = ContextVectorLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params:,} parameters")

    return model, tokenizer, config


def chat_loop(generator, args):
    """Interactive chat loop

    Args:
        generator: TextGenerator instance
        args: Command-line arguments
    """
    print("\n" + "="*80)
    print("🤖 New-LLM Chat Interface")
    print("="*80)
    print(f"⚙️  Settings:")
    print(f"   Max length: {args.max_length}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Top-p: {args.top_p}")
    print(f"   Repetition penalty: {args.repetition_penalty}")
    print(f"\n💡 Commands:")
    print(f"   'exit' or 'quit' - End conversation")
    print(f"   'reset' - Clear conversation context")
    print(f"   'help' - Show this help")
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

        if user_input.lower() == 'help':
            print("\nCommands:")
            print("  exit/quit - End conversation")
            print("  reset - Clear conversation context")
            print("  help - Show this help\n")
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
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(response)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"   Try 'reset' to clear context")

        print()  # Empty line for readability


def main():
    args = parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    # Load checkpoint
    try:
        model, tokenizer, config = load_checkpoint(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return 1

    # Create generator with anti-repetition measures
    generator = TextGenerator(model, tokenizer, args.device)

    # Start chat
    try:
        chat_loop(generator, args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
