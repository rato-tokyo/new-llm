#!/usr/bin/env python3
"""
Infini-Attention 2-Stage Training: Distillation + Fine-tuning

Stage 1: Knowledge Distillation
  - Infini Layerが元のLayer 0の出力を模倣するように訓練
  - 元のモデルとの整合性を確保

Stage 2: Full Fine-tuning
  - 全レイヤーを訓練（Layer 0優先的に高学習率）
  - LM lossで最適化

シングルヘッドメモリ（memory_head_dim=hidden_size）で最大の表現力を確保。

Usage:
    # 基本的な使い方
    python3 scripts/train_infini_distill_finetune.py --distill-epochs 10 --finetune-epochs 20

    # Layer 0の学習率を2倍に
    python3 scripts/train_infini_distill_finetune.py --layer0-lr-scale 2.0

    # 蒸留のみ
    python3 scripts/train_infini_distill_finetune.py --distill-epochs 10 --finetune-epochs 0
"""

import argparse
import sys
import time
from typing import Optional

sys.path.insert(0, ".")

import torch
import torch.nn as nn
from transformers import GPTNeoXForCausalLM

from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE
from src.models.infini_adapter import InfiniAdapterLayer
from src.utils.data_loading import load_wikitext2
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.training import get_device


class PythiaWithDistillableInfini(nn.Module):
    """
    蒸留可能なInfini Layer付きPythia

    - 蒸留フェーズ: Infini LayerがオリジナルLayer 0の出力を模倣
    - Fine-tuneフェーズ: 全レイヤーを訓練（layer-wise学習率サポート）
    """

    def __init__(
        self,
        base_model,
        use_delta_rule: bool = True,
    ):
        super().__init__()

        self.base_model = base_model
        self.config = base_model.config

        # オリジナルLayer 0を保持（蒸留のターゲット）
        self.original_layer0 = base_model.gpt_neox.layers[0]

        # Infini Layer作成（シングルヘッドメモリ: memory_head_dim=hidden_size）
        self.infini_layer = InfiniAdapterLayer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            use_delta_rule=use_delta_rule,
            layer_norm_eps=self.config.layer_norm_eps,
        )

        # 現在Layer 0として使用しているのはどちらか
        self._using_infini = False

    def use_infini_layer(self, use: bool = True) -> None:
        """Infini LayerをLayer 0として使用するかどうか切り替え"""
        if use:
            self.base_model.gpt_neox.layers[0] = self.infini_layer
        else:
            self.base_model.gpt_neox.layers[0] = self.original_layer0
        self._using_infini = use

    def get_distillation_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        蒸留損失を計算

        オリジナルLayer 0の出力とInfini Layerの出力のMSE
        """
        # Embedding取得
        hidden_states = self.base_model.gpt_neox.embed_in(input_ids)

        # Position embeddings取得
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.base_model.gpt_neox.rotary_emb(
            hidden_states, position_ids
        )

        # オリジナルLayer 0の出力（ターゲット）
        with torch.no_grad():
            original_outputs = self.original_layer0(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            target = original_outputs[0]

        # Infini Layerの出力（学習対象）
        infini_outputs = self.infini_layer(
            hidden_states,
            attention_mask=attention_mask,
            update_memory=True,
        )
        prediction = infini_outputs[0]

        # MSE Loss
        loss = nn.functional.mse_loss(prediction, target)

        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """通常のforward（LM loss計算）"""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def reset_memory(self) -> None:
        """メモリリセット"""
        self.infini_layer.reset_memory()

    def get_memory_state(self) -> dict:
        """メモリ状態を取得"""
        return self.infini_layer.get_memory_state()

    def set_memory_state(self, state: dict) -> None:
        """メモリ状態を設定"""
        device = next(self.parameters()).device
        self.infini_layer.set_memory_state(state, device)

    def get_parameter_groups(
        self,
        base_lr: float,
        layer0_lr_scale: float = 1.0,
        other_layers_lr_scale: float = 1.0,
    ) -> list:
        """
        Layer-wise学習率を持つパラメータグループを取得

        Args:
            base_lr: 基本学習率
            layer0_lr_scale: Layer 0（Infini）の学習率倍率
            other_layers_lr_scale: その他レイヤーの学習率倍率

        Returns:
            list of parameter groups for optimizer
        """
        # Layer 0 (Infini Layer) パラメータ
        layer0_params = list(self.infini_layer.parameters())

        # Embedding パラメータ
        embed_params = list(self.base_model.gpt_neox.embed_in.parameters())

        # Layer 1-N パラメータ
        other_layer_params = []
        for i, layer in enumerate(self.base_model.gpt_neox.layers):
            if i > 0:  # Skip layer 0 (it's replaced)
                other_layer_params.extend(list(layer.parameters()))

        # Final layer norm + LM head
        final_params = (
            list(self.base_model.gpt_neox.final_layer_norm.parameters()) +
            list(self.base_model.embed_out.parameters())
        )

        param_groups = [
            {
                "params": layer0_params,
                "lr": base_lr * layer0_lr_scale,
                "name": "layer0_infini",
            },
            {
                "params": embed_params,
                "lr": base_lr * other_layers_lr_scale,
                "name": "embeddings",
            },
            {
                "params": other_layer_params,
                "lr": base_lr * other_layers_lr_scale,
                "name": "layers_1_to_n",
            },
            {
                "params": final_params,
                "lr": base_lr * other_layers_lr_scale,
                "name": "final",
            },
        ]

        return param_groups


def evaluate_ppl_sliding_window(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    context_length: int = 2048,
    stride: int = 512,
) -> float:
    """Sliding window方式でPPL評価（HuggingFace標準）"""
    model.eval()
    if hasattr(model, "reset_memory"):
        model.reset_memory()

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    with torch.no_grad():
        for start in range(0, seq_len - 1, stride):
            end = min(start + context_length, seq_len)
            input_ids = tokens[start:end].unsqueeze(0)

            # 最初のstrideトークンはコンテキスト（loss計算しない）
            target_start = min(stride, end - start - 1)
            if target_start <= 0:
                continue

            labels = input_ids.clone()
            labels[0, :target_start] = -100

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            num_target_tokens = int((labels != -100).sum().item())
            if num_target_tokens > 0:
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def evaluate_ppl_segment(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    segment_length: int = 256,
) -> float:
    """セグメント分割でPPL評価（訓練中の比較用）"""
    model.eval()
    if hasattr(model, "reset_memory"):
        model.reset_memory()

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    with torch.no_grad():
        for start in range(0, seq_len - 1, segment_length):
            end = min(start + segment_length, seq_len)
            segment = tokens[start:end]

            if len(segment) < 2:
                continue

            input_ids = segment[:-1].unsqueeze(0)
            labels = segment[1:].unsqueeze(0)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def train_distillation_epoch(
    model: PythiaWithDistillableInfini,
    tokens: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    segment_length: int = 256,
) -> float:
    """蒸留エポック（Infini LayerがオリジナルLayer 0を模倣）"""
    model.train()
    model.reset_memory()

    # 蒸留中はオリジナルLayer 0を使用（Infiniは別途訓練）
    model.use_infini_layer(False)

    total_loss = 0.0
    num_segments = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    for start in range(0, seq_len - 1, segment_length):
        end = min(start + segment_length, seq_len)
        segment = tokens[start:end]

        if len(segment) < 2:
            continue

        input_ids = segment[:-1].unsqueeze(0)

        optimizer.zero_grad()

        loss = model.get_distillation_loss(input_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.infini_layer.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_segments += 1

    avg_loss = total_loss / num_segments if num_segments > 0 else float("inf")
    return avg_loss


def train_finetune_epoch(
    model: PythiaWithDistillableInfini,
    tokens: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    segment_length: int = 256,
) -> float:
    """Fine-tuneエポック（LM loss）"""
    model.train()
    model.reset_memory()

    # Fine-tune中はInfini Layerを使用
    model.use_infini_layer(True)

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    for start in range(0, seq_len - 1, segment_length):
        end = min(start + segment_length, seq_len)
        segment = tokens[start:end]

        if len(segment) < 2:
            continue

        input_ids = segment[:-1].unsqueeze(0)
        labels = segment[1:].unsqueeze(0)

        optimizer.zero_grad()

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(
        description="Infini-Attention 2-Stage Training: Distillation + Fine-tuning"
    )

    # Model settings
    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="Base model")

    # Training settings
    parser.add_argument("--distill-epochs", type=int, default=10, help="Distillation epochs")
    parser.add_argument("--finetune-epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--segment-length", type=int, default=256, help="Segment length")

    # Learning rate settings
    parser.add_argument("--distill-lr", type=float, default=1e-4, help="Distillation learning rate")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Fine-tuning base learning rate")
    parser.add_argument("--layer0-lr-scale", type=float, default=1.0, help="Layer 0 learning rate scale")
    parser.add_argument("--other-lr-scale", type=float, default=1.0, help="Other layers learning rate scale")

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience"
    )

    # Output
    parser.add_argument("--output", default="infini_distill_finetune.pt", help="Output path")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("INFINI-ATTENTION 2-STAGE TRAINING")
    print_flush("  Stage 1: Knowledge Distillation")
    print_flush("  Stage 2: Full Fine-tuning with Layer-wise LR")
    print_flush("=" * 70)
    print_flush(f"Model: {args.model}")
    print_flush(f"Device: {device}")
    print_flush(f"Segment length: {args.segment_length}")
    print_flush()
    print_flush("Stage 1 (Distillation):")
    print_flush(f"  Epochs: {args.distill_epochs}")
    print_flush(f"  Learning rate: {args.distill_lr}")
    print_flush()
    print_flush("Stage 2 (Fine-tuning):")
    print_flush(f"  Epochs: {args.finetune_epochs}")
    print_flush(f"  Base LR: {args.finetune_lr}")
    print_flush(f"  Layer 0 LR scale: {args.layer0_lr_scale}x")
    print_flush(f"  Other layers LR scale: {args.other_lr_scale}x")

    # Load tokenizer and data
    tokenizer = get_tokenizer(args.model)

    print_flush("\nLoading WikiText-2...")
    train_tokens = load_wikitext2(tokenizer, split="train")
    val_tokens = load_wikitext2(tokenizer, split="validation")

    print_flush(f"Train tokens: {len(train_tokens):,}")
    print_flush(f"Val tokens: {len(val_tokens):,}")

    # Create model
    print_flush("\nCreating model...")
    base_model = GPTNeoXForCausalLM.from_pretrained(args.model)
    model = PythiaWithDistillableInfini(
        base_model=base_model,
        use_delta_rule=True,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    infini_params = sum(p.numel() for p in model.infini_layer.parameters())
    print_flush(f"Total parameters: {total_params:,}")
    print_flush(f"Infini Layer parameters: {infini_params:,}")

    # =========================================================================
    # Evaluate baseline (original Pythia)
    # =========================================================================
    print_flush("\n" + "=" * 70)
    print_flush("BASELINE EVALUATION (Original Pythia)")
    print_flush("=" * 70)

    model.use_infini_layer(False)  # Use original
    baseline_ppl = evaluate_ppl_sliding_window(model, val_tokens, device)
    print_flush(f"  Baseline Val PPL (sliding window): {baseline_ppl:.2f}")

    # =========================================================================
    # Stage 1: Knowledge Distillation
    # =========================================================================
    if args.distill_epochs > 0:
        print_flush("\n" + "=" * 70)
        print_flush("STAGE 1: KNOWLEDGE DISTILLATION")
        print_flush("=" * 70)

        # Only train Infini Layer
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.infini_layer.parameters():
            param.requires_grad = True

        distill_optimizer = torch.optim.AdamW(
            model.infini_layer.parameters(),
            lr=args.distill_lr,
        )

        best_distill_loss = float("inf")
        best_distill_state = None
        patience_counter = 0

        for epoch in range(1, args.distill_epochs + 1):
            start_time = time.time()

            # Train distillation
            distill_loss = train_distillation_epoch(
                model, train_tokens, distill_optimizer, device, args.segment_length
            )

            # Evaluate with Infini Layer (segment-based for training monitoring)
            model.use_infini_layer(True)
            val_ppl = evaluate_ppl_segment(model, val_tokens, device, args.segment_length)

            elapsed = time.time() - start_time

            # Check improvement
            improved = distill_loss < best_distill_loss
            if improved:
                best_distill_loss = distill_loss
                best_distill_state = {
                    k: v.cpu().clone() for k, v in model.infini_layer.state_dict().items()
                }
                patience_counter = 0
                marker = "*"
            else:
                patience_counter += 1
                marker = ""

            print_flush(
                f"  Epoch {epoch:2d}: distill_loss={distill_loss:.6f} "
                f"val_ppl={val_ppl:.1f} ({elapsed:.1f}s) {marker}"
            )

            if patience_counter >= args.patience:
                print_flush(f"  Early stopping at epoch {epoch}")
                break

        # Load best distillation weights
        if best_distill_state is not None:
            model.infini_layer.load_state_dict(best_distill_state)

        # Evaluate after distillation (sliding window for fair comparison)
        model.use_infini_layer(True)
        post_distill_ppl = evaluate_ppl_sliding_window(model, val_tokens, device)
        print_flush(f"\n  Post-Distillation Val PPL (sliding window): {post_distill_ppl:.2f}")
    else:
        post_distill_ppl = None
        print_flush("\nSkipping Stage 1 (distillation)")

    # =========================================================================
    # Stage 2: Full Fine-tuning with Layer-wise LR
    # =========================================================================
    if args.finetune_epochs > 0:
        print_flush("\n" + "=" * 70)
        print_flush("STAGE 2: FULL FINE-TUNING")
        print_flush("=" * 70)

        # Switch to Infini Layer
        model.use_infini_layer(True)

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        # Layer-wise learning rate
        param_groups = model.get_parameter_groups(
            base_lr=args.finetune_lr,
            layer0_lr_scale=args.layer0_lr_scale,
            other_layers_lr_scale=args.other_lr_scale,
        )

        print_flush("\nParameter groups:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group["params"])
            print_flush(f"  {group['name']}: {num_params:,} params, lr={group['lr']:.2e}")

        finetune_optimizer = torch.optim.AdamW(param_groups)

        best_val_ppl = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, args.finetune_epochs + 1):
            start_time = time.time()

            # Train
            train_ppl = train_finetune_epoch(
                model, train_tokens, finetune_optimizer, device, args.segment_length
            )

            # Evaluate (segment-based for training monitoring)
            val_ppl = evaluate_ppl_segment(model, val_tokens, device, args.segment_length)

            elapsed = time.time() - start_time

            # Check improvement
            improved = val_ppl < best_val_ppl
            if improved:
                best_val_ppl = val_ppl
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
                marker = "*"
            else:
                patience_counter += 1
                marker = ""

            print_flush(
                f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                f"({elapsed:.1f}s) {marker}"
            )

            if patience_counter >= args.patience:
                print_flush(f"  Early stopping at epoch {epoch}")
                break

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation with sliding window
        post_finetune_ppl = evaluate_ppl_sliding_window(model, val_tokens, device)
        print_flush(f"\n  Post-Fine-tuning Val PPL (sliding window): {post_finetune_ppl:.2f}")
    else:
        post_finetune_ppl = None
        best_val_ppl = post_distill_ppl if post_distill_ppl else float("inf")
        print_flush("\nSkipping Stage 2 (fine-tuning)")

    # =========================================================================
    # Save model
    # =========================================================================
    print_flush(f"\nSaving to {args.output}...")

    save_dict = {
        "model_state_dict": model.state_dict(),
        "infini_layer_state_dict": model.infini_layer.state_dict(),
        "config": {
            "model_name": args.model,
            "distill_epochs": args.distill_epochs,
            "finetune_epochs": args.finetune_epochs,
            "layer0_lr_scale": args.layer0_lr_scale,
            "other_lr_scale": args.other_lr_scale,
        },
        "baseline_ppl": baseline_ppl,
        "post_distill_ppl": post_distill_ppl,
        "post_finetune_ppl": post_finetune_ppl,
        "best_val_ppl": best_val_ppl,
    }
    torch.save(save_dict, args.output)

    # =========================================================================
    # Summary
    # =========================================================================
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush("| Stage | PPL |")
    print_flush("|-------|-----|")
    print_flush(f"| Baseline (Original Pythia) | {baseline_ppl:.2f} |")
    if post_distill_ppl is not None:
        print_flush(f"| After Distillation | {post_distill_ppl:.2f} |")
    if post_finetune_ppl is not None:
        print_flush(f"| After Fine-tuning | {post_finetune_ppl:.2f} |")

    print_flush()
    if post_finetune_ppl is not None:
        if post_finetune_ppl < baseline_ppl:
            improvement = baseline_ppl - post_finetune_ppl
            print_flush(f"SUCCESS: Improved by {improvement:.2f} PPL")
        else:
            degradation = post_finetune_ppl - baseline_ppl
            print_flush(f"WARNING: Degraded by {degradation:.2f} PPL")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
