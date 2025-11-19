#!/usr/bin/env python3
"""
New-LLM Training with HuggingFace Transformers

This version uses HuggingFace Trainer for maximum reliability:
- Automatic checkpoint management
- Built-in logging and metrics
- Exposure bias mitigation
- Industry-standard best practices

Usage:
    # Quick test (local)
    python train.py --dataset ultrachat --max-samples 1000 --epochs 5 --output-dir test_run

    # Full training (Colab/GPU)
    python train.py --dataset ultrachat --epochs 50 --batch-size 32 --output-dir checkpoints/ultrachat_50epochs
"""

import argparse
import os
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.models.new_llm_config import NewLLMConfig
from src.models.new_llm_hf import NewLLMForCausalLM
from src.training.hf_tokenizer import create_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train New-LLM with HuggingFace')

    # Dataset
    parser.add_argument('--dataset', type=str, default='ultrachat',
                       choices=['ultrachat'],
                       help='Dataset to train on')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit dataset (for testing). None = full dataset')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size per device (default: 8 for local)')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps (effective batch = batch-size * this)')

    # Model
    parser.add_argument('--layers', type=int, default=1,
                       help='Number of layers')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size')

    # Output
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save-steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval-steps', type=int, default=100,
                       help='Evaluate every N steps')

    # Device
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA (use CPU)')

    return parser.parse_args()


def load_dataset_texts(dataset_name, max_samples=None):
    """Load dataset and extract texts

    Args:
        dataset_name: Name of dataset
        max_samples: Maximum samples to use (None = all)

    Returns:
        (train_texts, val_texts): Lists of text strings
    """
    print(f"\n📥 Loading {dataset_name} dataset...")

    if dataset_name == 'ultrachat':
        from datasets import load_dataset

        try:
            dataset = load_dataset("stingning/ultrachat")
        except Exception as e:
            print(f"⚠️  Trying alternative dataset...")
            dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

        train_data = dataset['train']

        # Create validation split
        if 'test' in dataset:
            val_data = dataset['test']
        else:
            print("   Creating 90/10 train/val split...")
            split = train_data.train_test_split(test_size=0.1, seed=42)
            train_data = split['train']
            val_data = split['test']

        # Limit samples
        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))
            val_data = val_data.select(range(min(max_samples // 10, len(val_data))))

        print(f"✓ Loaded {len(train_data):,} train, {len(val_data):,} val samples")

        # Extract texts
        train_texts = []
        val_texts = []

        print("   Extracting texts...")
        for example in train_data:
            if 'data' in example:
                conversation = example['data']
                parts = []
                for turn in conversation:
                    if isinstance(turn, dict) and 'content' in turn:
                        parts.append(turn['content'])
                    elif isinstance(turn, str):
                        parts.append(turn)
                if parts:
                    train_texts.append(' '.join(parts))

        for example in val_data:
            if 'data' in example:
                conversation = example['data']
                parts = []
                for turn in conversation:
                    if isinstance(turn, dict) and 'content' in turn:
                        parts.append(turn['content'])
                    elif isinstance(turn, str):
                        parts.append(turn)
                if parts:
                    val_texts.append(' '.join(parts))

        print(f"✓ Extracted {len(train_texts):,} train texts, {len(val_texts):,} val texts")

        return train_texts, val_texts

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def main():
    args = parse_args()

    print("=" * 80)
    print("New-LLM Training with HuggingFace Transformers")
    print("=" * 80)

    # Load dataset
    train_texts, val_texts = load_dataset_texts(args.dataset, args.max_samples)

    # Create tokenizer
    print(f"\n🔤 Creating BPE tokenizer...")
    tokenizer = create_tokenizer(
        train_texts,
        vocab_size=args.vocab_size,
        min_frequency=2
    )

    # Save tokenizer
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"✓ Tokenizer saved to {tokenizer_dir}")

    # Tokenize datasets
    print(f"\n⚙️  Tokenizing datasets...")

    def tokenize_function(texts):
        """Tokenize a list of texts"""
        return tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=False,  # Dynamic padding in collator
            return_special_tokens_mask=True
        )

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    # Create HF datasets
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)

    print(f"✓ Tokenized datasets created")
    print(f"   Train: {len(train_dataset):,} examples")
    print(f"   Val: {len(val_dataset):,} examples")

    # Create model
    print(f"\n🧠 Creating New-LLM model...")

    config = NewLLMConfig(
        vocab_size=len(tokenizer),
        embed_dim=256,
        hidden_dim=512,
        context_vector_dim=256,
        num_layers=args.layers,
        max_seq_length=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = NewLLMForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only 3 checkpoints
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        no_cuda=args.no_cuda,
        fp16=not args.no_cuda,  # Use FP16 if CUDA available
        dataloader_num_workers=0,
    )

    # Create Trainer
    print(f"\n🏋️  Creating HuggingFace Trainer...")
    print(f"   Batch size: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps} (effective)")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   FP16: {training_args.fp16}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")

    trainer.train()

    # Save final model
    print(f"\n💾 Saving final model...")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"{'='*80}")
    print(f"\nModel saved to: {args.output_dir}/final_model")
    print(f"\nTo chat with your model:")
    print(f"  python chat.py --model-path {args.output_dir}/final_model")
    print()


if __name__ == "__main__":
    main()
