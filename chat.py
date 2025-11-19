#!/usr/bin/env python3
"""
New-LLM Chat with HuggingFace Generation

Uses HuggingFace's GenerationMixin for state-of-the-art text generation:
- Beam search
- Nucleus sampling
- Repetition penalty (built-in, no custom code needed!)
- Temperature sampling
- All Exposure Bias mitigations included

Usage:
    python chat.py --model-path checkpoints/ultrachat_50epochs/final_model
    python chat.py --model-path test_run/final_model --temperature 0.9
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with New-LLM')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory')

    # Generation parameters
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum tokens to generate (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Sampling temperature (default: 0.9)')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Nucleus sampling top-p (default: 0.95)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling (default: 50)')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                       help='Repetition penalty (default: 1.2)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Use CPU instead of CUDA')

    return parser.parse_args()


def chat_loop(model, tokenizer, args):
    """Interactive chat loop

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        args: Command-line arguments
    """
    print("\n" + "="*80)
    print("🤖 New-LLM Chat (HuggingFace Generation)")
    print("="*80)
    print(f"⚙️  Generation Settings:")
    print(f"   Max length: {args.max_length}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Top-p: {args.top_p}")
    print(f"   Top-k: {args.top_k}")
    print(f"   Repetition penalty: {args.repetition_penalty}")
    print(f"\n💡 Commands:")
    print(f"   'exit' or 'quit' - End conversation")
    print(f"   'reset' - Clear conversation context")
    print("="*80 + "\n")

    conversation_history = ""

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
            conversation_history = ""
            print("🔄 Conversation reset\n")
            continue

        if not user_input.strip():
            continue

        # Build prompt
        if conversation_history:
            prompt = f"{conversation_history}\nHuman: {user_input}\n\nAssistant:"
        else:
            prompt = f"Human: {user_input}\n\nAssistant:"

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if not args.no_cuda and model.device.type == 'cuda':
            input_ids = input_ids.to('cuda')

        # Generate with HuggingFace's generate() method
        # This includes ALL best practices:
        # - Repetition penalty
        # - Temperature sampling
        # - Top-p/top-k sampling
        # - Proper handling of special tokens
        print("Assistant: ", end='', flush=True)

        try:
            with model.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode output
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract assistant's response (everything after "Assistant:")
            if "Assistant:" in full_text:
                parts = full_text.split("Assistant:")
                response = parts[-1].strip()

                # Stop at next "Human:" if present
                if "Human:" in response:
                    response = response.split("Human:")[0].strip()
            else:
                # Fallback: use everything after the prompt
                response = full_text[len(prompt):].strip()

            print(response)

            # Update conversation history
            conversation_history = f"{prompt} {response}"

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Try 'reset' to clear context")

        print()  # Empty line


def main():
    args = parse_args()

    print("=" * 80)
    print("Loading New-LLM Model")
    print("=" * 80)

    # Load model and tokenizer
    print(f"\n📂 Loading from: {args.model_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True  # Allow custom models
        )

        # Move to device
        if not args.no_cuda:
            import torch
            if torch.cuda.is_available():
                model = model.to('cuda')
                print(f"✓ Model loaded on CUDA")
            else:
                print(f"⚠️  CUDA not available, using CPU")
        else:
            print(f"✓ Model loaded on CPU")

        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Start chat
    try:
        chat_loop(model, tokenizer, args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
