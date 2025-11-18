#!/usr/bin/env python3
"""
WikiText-2 FP16 Training with Variable Number of Layers

Usage:
    python scripts/train_wikitext_fp16_layers.py --num_layers 5
    python scripts/train_wikitext_fp16_layers.py --num_layers 7
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.amp
from src.utils.config import NewLLML4Config
from src.models.context_vector_llm import ContextVectorLLM
from src.training.wikitext_dataset import load_wikitext_data
from src.training.trainer import Trainer
from torch.utils.data import DataLoader
import time
import argparse


def create_config(num_layers_value):
    """Create config class dynamically based on num_layers"""

    class LayerExperimentConfig(NewLLML4Config):
        """Layer experiment configuration

        Fixed parameters:
        - batch_size = 2048 (L4 GPU optimized)
        - learning_rate = 0.0008 (Square Root Scaling)
        - context_vector_dim = 256

        Variable parameter:
        - num_layers (set dynamically)
        """
        # データ関連（WikiText-2用）
        max_seq_length = 64
        vocab_size = 1000

        # モデルアーキテクチャ
        embed_dim = 256
        hidden_dim = 512
        # num_layers will be set after class definition
        context_vector_dim = 256  # Fixed
        dropout = 0.1

        # 訓練ハイパーパラメータ（NewLLML4Configから継承）
        # batch_size = 2048     ← 固定（L4用）
        # learning_rate = 0.0008 ← 固定（Square Root Scaling）
        # device = "cuda"       ← 固定
        num_epochs = 150
        weight_decay = 0.0
        gradient_clip = 1.0

        # Early Stopping
        patience = 30

        # FP16設定
        use_amp = True

    # Set num_layers after class definition
    LayerExperimentConfig.num_layers = num_layers_value

    return LayerExperimentConfig


class FP16Trainer(Trainer):
    """FP16混合精度対応のTrainer

    PyTorch AMPを使ったFP16訓練
    """

    def __init__(self, *args, use_amp=True, **kwargs):
        super().__init__(*args, **kwargs)

        # GPU必須チェック
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available! FP16 training requires CUDA GPU.")

        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda')
        print(f"\n✓ FP16 Mixed Precision enabled (AMP)")

    def train_epoch(self) -> tuple[float, float]:
        """FP16対応の訓練エポック"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = len(self.train_dataloader)

        print(f"  Training... ", end='', flush=True)
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # FP16 mixed precision forward/backward
            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                from src.evaluation.metrics import compute_loss
                loss = compute_loss(logits, targets, pad_idx=0)

            # Scaled backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            self.scaler.unscale_(self.optimizer)

            # Adaptive gradient clipping
            if self.current_epoch < 10:
                clip_value = 0.5
            elif self.current_epoch < 30:
                clip_value = 1.0
            else:
                clip_value = 2.0

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_value
            )

            # Optimizer step with scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            total_loss += loss.item() * inputs.size(0)
            total_tokens += inputs.size(0)

            # Compact progress display
            if (batch_idx + 1) % max(1, num_batches // 5) == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"{progress:.0f}%", end=' ', flush=True)

        elapsed = time.time() - start_time

        from src.evaluation.metrics import compute_perplexity
        avg_loss = total_loss / total_tokens
        avg_ppl = compute_perplexity(avg_loss)
        print(f"| {elapsed/60:.1f}min | Loss: {avg_loss:.4f}")

        return avg_loss, avg_ppl


def main():
    """FP16混合精度訓練のメイン処理（レイヤー数実験）"""

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='WikiText-2 FP16 Training with Variable Layers')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='Number of layers (e.g., 5, 6, 7)')
    args = parser.parse_args()

    num_layers = args.num_layers

    print("\n" + "="*80)
    print(f"WikiText-2 FP16 Training - {num_layers} Layers Experiment")
    print("="*80)

    # Git version information
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__) + '/..').decode().strip()
        git_commit_short = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(__file__) + '/..').decode().strip()
        git_date = subprocess.check_output(['git', 'log', '-1', '--format=%cd', '--date=short'], cwd=os.path.dirname(__file__) + '/..').decode().strip()
        print(f"\n📌 Git Version: {git_commit_short} ({git_date})")
        print(f"   Full commit: {git_commit}")
    except Exception:
        print(f"\n📌 Git Version: Unknown (not a git repository)")

    print("="*80)

    # 設定作成
    ConfigClass = create_config(num_layers)
    config = ConfigClass()

    # GPU必須チェック
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU not available! FP16 training requires CUDA GPU.")

    # デバイス情報表示
    print(f"\n🖥️  Device Information:")
    print(f"  Device: CUDA (GPU)")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {gpu_memory:.1f} GB")
    print(f"  FP16 Mixed Precision: ENABLED ✓")

    print(f"\n🔬 実験設定:")
    print(f"  Model: New-LLM Baseline")
    print(f"  Num Layers: {num_layers} ← Variable")
    print(f"  Context Vector Dim: {config.context_vector_dim} (Fixed)")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size} (Fixed)")
    print(f"  Learning Rate: {config.learning_rate} (Fixed)")
    print(f"  Precision: FP16 (Mixed)")
    print(f"\n{'='*80}\n")

    # データロード
    print("Loading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # モデル作成
    print(f"\nCreating New-LLM model with {num_layers} layers...")
    model = ContextVectorLLM(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # FP16 Trainer作成
    model_name = f"new_llm_wikitext_fp16_layers{num_layers}"
    trainer = FP16Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name=model_name,
        use_amp=config.use_amp
    )

    # 訓練実行
    print(f"\nStarting FP16 training with {num_layers} layers...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print(f"{num_layers} Layers Training Completed!")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Checkpoint saved: checkpoints/best_{model_name}.pt")

    # 性能サマリー
    if trainer.val_losses:
        best_val_loss = min(trainer.val_losses)
        best_val_ppl = min(trainer.val_ppls)
        best_epoch = trainer.val_ppls.index(best_val_ppl) + 1

        print(f"\n📊 最終結果:")
        print(f"  Num Layers: {num_layers}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val Perplexity: {best_val_ppl:.2f} (Epoch {best_epoch})")

        # Baseline (6 layers) との比較
        baseline_ppl = 20.60  # 6 layers, 150 epochs
        diff = best_val_ppl - baseline_ppl
        diff_pct = (diff / baseline_ppl) * 100

        if diff < 0:
            print(f"  vs Baseline (6 layers): {diff:.2f} ({diff_pct:+.1f}%) ✓ BETTER")
        elif diff > 0:
            print(f"  vs Baseline (6 layers): +{diff:.2f} ({diff_pct:+.1f}%)")
        else:
            print(f"  vs Baseline (6 layers): Same")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
