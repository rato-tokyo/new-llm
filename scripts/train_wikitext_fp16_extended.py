#!/usr/bin/env python3
"""
WikiText-2 FP16 Extended Training - Resume from 50 epochs, train to 150 epochs total

This script continues training from the best checkpoint of the 50-epoch run,
extending training for an additional 100 epochs (total 150 epochs).

Usage:
    python scripts/train_wikitext_fp16_extended.py
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


class FP16Config(NewLLML4Config):
    """FP16混合精度訓練用の設定（L4 GPU最適化）

    This class is needed for checkpoint compatibility.
    The checkpoint was saved with FP16Config, so we need to define it here
    even though we use FP16ExtendedConfig for training.
    """
    # データ関連（WikiText-2用）
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ（Baseline）
    embed_dim = 256
    hidden_dim = 512
    num_layers = 6
    context_vector_dim = 256
    dropout = 0.1

    # 訓練ハイパーパラメータ（NewLLML4Configから継承）
    num_epochs = 50
    weight_decay = 0.0
    gradient_clip = 1.0

    # Early Stopping
    patience = 15

    # FP16設定
    use_amp = True


class FP16ExtendedConfig(NewLLML4Config):
    """FP16混合精度訓練用の設定（100エポック延長版）

    L4 GPU最適化設定（batch_size=2048, learning_rate=0.0004）を継承
    50エポックから再開し、合計150エポック訓練
    """
    # データ関連（WikiText-2用）
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ（Baseline）
    embed_dim = 256
    hidden_dim = 512
    num_layers = 6
    context_vector_dim = 256
    dropout = 0.1

    # 訓練ハイパーパラメータ（NewLLML4Configから継承）
    # batch_size = 2048     ← NewLLML4Configから自動継承（L4用）
    # learning_rate = 0.0008 ← NewLLML4Configから自動継承（Square Root Scaling適用済み）
    # device = "cuda"       ← NewLLML4Configから自動継承
    num_epochs = 150  # 合計150エポック（50から再開して+100）
    weight_decay = 0.0
    gradient_clip = 1.0

    # Early Stopping
    patience = 30  # 延長訓練のため長めに設定

    # FP16設定
    use_amp = True  # Automatic Mixed Precision (GPU必須)

    # チェックポイント設定
    checkpoint_to_resume = "best_new_llm_wikitext_fp16.pt"


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
    """FP16混合精度訓練のメイン処理（100エポック延長版）"""
    print("\n" + "="*80)
    print("WikiText-2 Extended Training with FP16 Mixed Precision")
    print("Resume from 50 epochs → Train to 150 epochs total (+100 epochs)")
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

    config = FP16ExtendedConfig()

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

    print(f"\n実験設定:")
    print(f"  Model: New-LLM Baseline")
    print(f"  Context Vector Dim: {config.context_vector_dim}")
    print(f"  Num Layers: {config.num_layers}")
    print(f"  Total Epochs: {config.num_epochs} (50 done + 100 extended)")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Precision: FP16 (Mixed)")
    print(f"  Resume from: {config.checkpoint_to_resume}")
    print(f"\n{'='*80}\n")

    # データロード
    print("Loading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # モデル作成
    print("\nCreating New-LLM model...")
    model = ContextVectorLLM(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # FP16 Trainer作成
    trainer = FP16Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name="new_llm_wikitext_fp16_extended",
        use_amp=config.use_amp
    )

    # チェックポイントから再開
    # Note: resume_from_checkpoint() will automatically add "checkpoints/" prefix
    checkpoint_name = config.checkpoint_to_resume
    checkpoint_path = os.path.join("checkpoints", checkpoint_name)
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Resuming from checkpoint: {checkpoint_path}")
        start_epoch = trainer.resume_from_checkpoint(checkpoint_name)
        print(f"✓ Resumed from epoch {start_epoch}")
        print(f"📊 Previous best validation PPL: {min(trainer.val_ppls) if trainer.val_ppls else 'N/A'}")
    else:
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}\n"
                                f"Please run train_wikitext_fp16.py first to generate the checkpoint.")

    # 訓練実行
    print("\nStarting extended training (100 more epochs)...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("Extended FP16 Mixed Precision Training Completed!")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Checkpoint saved: checkpoints/best_new_llm_wikitext_fp16_extended.pt")

    # 性能サマリー
    if trainer.val_ppls:
        best_ppl = min(trainer.val_ppls)
        best_epoch = trainer.val_ppls.index(best_ppl) + 1
        final_ppl = trainer.val_ppls[-1]

        print(f"\n📊 Performance Summary:")
        print(f"  Best Validation PPL: {best_ppl:.2f} (Epoch {best_epoch})")
        print(f"  Final Validation PPL: {final_ppl:.2f}")
        print(f"  Improvement from epoch 50: {trainer.val_ppls[49] if len(trainer.val_ppls) > 49 else 'N/A'} → {final_ppl:.2f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
