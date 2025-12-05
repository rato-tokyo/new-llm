#!/usr/bin/env python3
"""
Custom RoPE Experiment

カスタムRoPE設定の実験。異なる周波数設定でPPLを比較。

Usage:
    # 標準RoPEとカスタムRoPEの比較
    python3 scripts/experiment_custom_rope.py --samples 5000 --epochs 30

    # 特定の設定のみ
    python3 scripts/experiment_custom_rope.py --config-name linear

    # カスタムconfig_listを指定
    python3 scripts/experiment_custom_rope.py --custom-config "[[2,0.1],[2,0.2],[4,0.05],[8,0.01]]"
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from config.pythia import PythiaConfig
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import (
    prepare_data_loaders,
    get_device,
    get_tokenizer,
    train_epoch,
    evaluate,
)
from src.utils.evaluation import evaluate_position_wise_ppl, evaluate_reversal_curse
from src.data.reversal_pairs import get_reversal_pairs
from src.utils.device import clear_gpu_cache
from src.utils.rope import (
    RoPEConfig,
    CustomRotaryEmbedding,
    create_rope_from_config,
    standard_rope_config,
    custom_list_config,
    linear_frequency_config,
    exponential_frequency_config,
)


class CustomRoPEAttention(nn.Module):
    """カスタムRoPEを使用するAttention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_config: RoPEConfig,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        # カスタムRoPE
        self.rope = CustomRotaryEmbedding(
            head_dim=self.head_dim,
            config=rope_config,
            max_position_embeddings=max_position_embeddings,
        )
        self.rotary_dim = self.rope.rotary_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # カスタムRoPEを適用
        query, key = self.rope(query, key, position_ids)

        # Attention計算
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.dense(attn_output)

        return output


class CustomRoPEPythiaModel(nn.Module):
    """カスタムRoPEを使用するPythiaモデル"""

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        rope_config: Optional[RoPEConfig] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if rope_config is None:
            rope_config = standard_rope_config(rotary_pct=0.25, base=10000)
        self.rope_config = rope_config

        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        self.attentions = nn.ModuleList([
            CustomRoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rope_config=rope_config,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_layers)
        ])

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size),
            )
            for _ in range(num_layers)
        ])

        self.input_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_in(input_ids)

        for i in range(self.num_layers):
            residual = hidden_states
            hidden_states = self.input_layernorms[i](hidden_states)
            attn_output = self.attentions[i](hidden_states, position_ids)
            mlp_output = self.mlps[i](hidden_states)
            hidden_states = residual + attn_output + mlp_output

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}


def get_predefined_configs(head_dim: int = 64) -> Dict[str, RoPEConfig]:
    """事前定義されたRoPE設定を取得"""
    rotary_dim = int(head_dim * 0.25)  # 16 dims for head_dim=64

    configs = {
        # 標準RoPE (Pythia style)
        "standard": standard_rope_config(rotary_pct=0.25, base=10000),

        # 標準RoPE (全次元)
        "standard_full": standard_rope_config(rotary_pct=1.0, base=10000),

        # 線形周波数
        "linear": linear_frequency_config(rotary_dim, min_freq=0.001, max_freq=1.0),

        # 線形周波数（低周波帯）
        "linear_low": linear_frequency_config(rotary_dim, min_freq=0.0001, max_freq=0.1),

        # 線形周波数（高周波帯）
        "linear_high": linear_frequency_config(rotary_dim, min_freq=0.1, max_freq=2.0),

        # 指数周波数（標準と同等）
        "exponential": exponential_frequency_config(rotary_dim, base=10000),

        # カスタム: 2次元ずつ異なる周波数
        "custom_2d": custom_list_config([
            [2, 0.001],
            [2, 0.01],
            [2, 0.1],
            [2, 0.5],
            [2, 1.0],
            [2, 2.0],
            [2, 4.0],
            [2, 8.0],
        ]),

        # カスタム: 低周波重視
        "low_freq_bias": custom_list_config([
            [4, 0.001],
            [4, 0.005],
            [4, 0.01],
            [4, 0.05],
        ]),

        # カスタム: 高周波重視
        "high_freq_bias": custom_list_config([
            [4, 0.1],
            [4, 0.5],
            [4, 1.0],
            [4, 2.0],
        ]),

        # カスタム: 均等周波数
        "uniform": custom_list_config([
            [16, 0.1],  # 全16次元で同じ周波数
        ]),
    }

    return configs


def run_experiment(
    num_samples: int = 5000,
    seq_length: int = 128,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    config_names: Optional[List[str]] = None,
    custom_config: Optional[str] = None,
) -> Dict[str, Any]:
    """Run custom RoPE experiment"""
    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("CUSTOM ROPE EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush("=" * 70)

    # Prepare data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
    )

    # Get configs to test
    head_dim = config.hidden_size // config.num_attention_heads
    predefined_configs = get_predefined_configs(head_dim)

    if custom_config:
        # Parse custom config from JSON
        config_list = json.loads(custom_config)
        predefined_configs["custom"] = custom_list_config(config_list)
        if config_names is None:
            config_names = ["standard", "custom"]

    if config_names is None:
        config_names = ["standard", "linear", "exponential", "custom_2d"]

    results: Dict[str, Any] = {}

    for config_name in config_names:
        if config_name not in predefined_configs:
            print_flush(f"\nWARNING: Unknown config '{config_name}', skipping...")
            continue

        rope_config = predefined_configs[config_name]

        print_flush("\n" + "=" * 70)
        print_flush(f"CONFIG: {config_name}")
        print_flush("=" * 70)

        # Show config details
        if rope_config.mode == "standard":
            print_flush(f"  Mode: standard")
            print_flush(f"  rotary_pct: {rope_config.rotary_pct}")
            print_flush(f"  base: {rope_config.base}")
        elif rope_config.mode == "custom":
            print_flush(f"  Mode: custom")
            print_flush(f"  frequencies: {rope_config.frequencies[:5]}..." if len(rope_config.frequencies or []) > 5 else f"  frequencies: {rope_config.frequencies}")
        elif rope_config.mode == "custom_list":
            print_flush(f"  Mode: custom_list")
            print_flush(f"  config_list: {rope_config.config_list}")

        # Create model
        model = CustomRoPEPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            rope_config=rope_config,
        )
        model = model.to(device)

        rotary_dim = model.attentions[0].rotary_dim
        param_info = model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")
        print_flush(f"  Rotary dim: {rotary_dim}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        print_flush(f"\n  Training...")
        best_val_ppl = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_loss = train_epoch(model, train_loader, optimizer, device)
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            _, val_ppl = evaluate(model, val_loader, device)
            elapsed = time.time() - start_time

            improved = val_ppl < best_val_ppl
            if improved:
                best_val_ppl = val_ppl
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                marker = "*"
            else:
                patience_counter += 1
                marker = ""

            print_flush(
                f"    Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                f"[{elapsed:.1f}s] {marker}"
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print_flush("    -> Early stop")
                break

        print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)

        # Position-wise PPL
        print_flush("\n  Position-wise PPL:")
        pos_ppl = evaluate_position_wise_ppl(model, val_loader, device)
        for pos_range, ppl in pos_ppl.items():
            print_flush(f"    Position {pos_range}: {ppl:.1f}")

        # Reversal Curse evaluation
        print_flush("\n  Reversal Curse evaluation:")
        tokenizer = get_tokenizer(config.tokenizer_name)
        reversal_pairs = get_reversal_pairs()
        reversal = evaluate_reversal_curse(model, tokenizer, reversal_pairs, device)
        print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
        print_flush(f"    Reversal Ratio: {reversal['reversal_ratio']:.4f}")
        print_flush(f"    Reversal Gap: {reversal['reversal_gap']:.1f}")

        results[config_name] = {
            "best_val_ppl": best_val_ppl,
            "best_epoch": best_epoch,
            "params": param_info["total"],
            "rotary_dim": rotary_dim,
            "position_wise_ppl": pos_ppl,
            "reversal_curse": reversal,
            "config": {
                "mode": rope_config.mode,
                "rotary_pct": rope_config.rotary_pct if rope_config.mode == "standard" else None,
                "base": rope_config.base if rope_config.mode == "standard" else None,
                "frequencies": rope_config.frequencies,
                "config_list": rope_config.config_list,
            }
        }

        del model
        clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Config | PPL | Epoch | Rotary Dim |")
    print_flush("|--------|-----|-------|------------|")

    for config_name, r in sorted(results.items(), key=lambda x: x[1]["best_val_ppl"]):
        print_flush(f"| {config_name} | {r['best_val_ppl']:.1f} | {r['best_epoch']} | {r['rotary_dim']} |")

    # Best config
    if results:
        best_config = min(results.items(), key=lambda x: x[1]["best_val_ppl"])
        print_flush(f"\nBest config: {best_config[0]} (PPL={best_config[1]['best_val_ppl']:.1f})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom RoPE Experiment")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--config-names",
        type=str,
        nargs="+",
        default=None,
        help="Config names to test (e.g., standard linear exponential)"
    )
    parser.add_argument(
        "--custom-config",
        type=str,
        default=None,
        help='Custom config_list as JSON (e.g., "[[2,0.1],[2,0.2],[4,0.05]]")'
    )
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        config_names=args.config_names,
        custom_config=args.custom_config,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
