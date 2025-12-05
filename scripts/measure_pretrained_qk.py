#!/usr/bin/env python3
"""
Pretrained Pythia Q, K Statistics Measurement

訓練済みPythia-70mのQ, K統計を測定する。
学習は行わず、既存パラメータのまま測定のみ実施。

Usage:
    python3 scripts/measure_pretrained_qk.py --samples 5000 --seq-length 128
    python3 scripts/measure_pretrained_qk.py --model pythia-160m --samples 5000
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, ".")

from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.device import clear_gpu_cache


@dataclass
class QKStats:
    """Q, Kの統計情報を格納するデータクラス"""
    layer_idx: int
    q_max: float = 0.0
    q_mean: float = 0.0
    q_std: float = 0.0
    k_max: float = 0.0
    k_mean: float = 0.0
    k_std: float = 0.0
    q_dim_max: List[float] = field(default_factory=list)
    k_dim_max: List[float] = field(default_factory=list)


class PretrainedQKCollector:
    """
    HuggingFace Pythiaモデル用のQ, K統計コレクター
    """

    def __init__(
        self,
        model: GPTNeoXForCausalLM,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hooks: List[Any] = []
        self.stats: Dict[int, List[QKStats]] = {i: [] for i in range(num_layers)}
        self._rotary_dim: Optional[int] = None

    def _create_hook(self, layer_idx: int):
        """レイヤーごとのフックを作成"""

        def hook_fn(module, input_args, output):
            # GPTNeoXAttentionの query_key_value 出力を取得
            # output は (attn_output, present, attn_weights) のタプル
            # Q, K, V は内部で計算されるので、query_key_value の入力を使って再計算

            # GPTNeoXAttentionでは hidden_states を入力として受け取る
            hidden_states = input_args[0]
            attention_mask = input_args[1] if len(input_args) > 1 else None

            # query_key_value から Q, K, V を計算
            qkv = module.query_key_value(hidden_states)

            # GPTNeoX の qkv 分割方式
            # qkv: [batch, seq, 3 * num_heads * head_dim]
            # 分割: [batch, seq, num_heads, 3 * head_dim] -> split
            batch_size, seq_len, _ = qkv.shape

            # Reshape: [batch, seq, num_heads, 3, head_dim]
            qkv = qkv.view(batch_size, seq_len, self.num_heads, 3, self.head_dim)

            # Split into Q, K, V: [batch, seq, num_heads, head_dim]
            q = qkv[:, :, :, 0, :]  # Query
            k = qkv[:, :, :, 1, :]  # Key
            # v = qkv[:, :, :, 2, :]  # Value (不要)

            # Transpose to [batch, heads, seq, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            # RoPE適用後の値を取得するため、RoPEを適用
            # GPTNeoXRotaryEmbedding を使用
            rotary_emb = self.model.gpt_neox.rotary_emb
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            cos, sin = rotary_emb(k, position_ids)

            # rotary_dim: inv_freq の長さ × 2（cos/sin は rotary_dim 次元）
            # inv_freq: [rotary_dim / 2]
            rotary_dim = rotary_emb.inv_freq.shape[0] * 2

            # RoPE適用
            q_rotary, k_rotary = apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)

            # RoPE適用次元数を保存
            if self._rotary_dim is None:
                self._rotary_dim = rotary_dim

            with torch.no_grad():
                q_abs = q_rotary.abs()
                k_abs = k_rotary.abs()

                stats = QKStats(
                    layer_idx=layer_idx,
                    q_max=q_abs.max().item(),
                    q_mean=q_abs.mean().item(),
                    q_std=q_rotary.std().item(),
                    k_max=k_abs.max().item(),
                    k_mean=k_abs.mean().item(),
                    k_std=k_rotary.std().item(),
                )

                # 次元ごとの最大値: [head_dim]
                # q: [batch, heads, seq, head_dim] -> max over (batch, heads, seq)
                q_dim_max = q_abs.max(dim=0).values.max(dim=0).values.max(dim=0).values
                k_dim_max = k_abs.max(dim=0).values.max(dim=0).values.max(dim=0).values

                stats.q_dim_max = q_dim_max.tolist()
                stats.k_dim_max = k_dim_max.tolist()

                self.stats[layer_idx].append(stats)

        return hook_fn

    def register_hooks(self) -> None:
        """全レイヤーにフックを登録"""
        for layer_idx, layer in enumerate(self.model.gpt_neox.layers):
            hook = layer.attention.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """フックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self) -> None:
        """統計をクリア"""
        self.stats = {i: [] for i in range(self.num_layers)}

    def get_stats(self) -> Dict[int, QKStats]:
        """収集した統計の平均を返す"""
        result = {}
        for layer_idx, stats_list in self.stats.items():
            if not stats_list:
                continue

            avg_stats = QKStats(
                layer_idx=layer_idx,
                q_max=max(s.q_max for s in stats_list),
                q_mean=sum(s.q_mean for s in stats_list) / len(stats_list),
                q_std=sum(s.q_std for s in stats_list) / len(stats_list),
                k_max=max(s.k_max for s in stats_list),
                k_mean=sum(s.k_mean for s in stats_list) / len(stats_list),
                k_std=sum(s.k_std for s in stats_list) / len(stats_list),
            )

            # 次元ごとの最大値は全バッチでの最大を取る
            if stats_list[0].q_dim_max:
                head_dim = len(stats_list[0].q_dim_max)
                avg_stats.q_dim_max = [
                    max(s.q_dim_max[d] for s in stats_list)
                    for d in range(head_dim)
                ]
                avg_stats.k_dim_max = [
                    max(s.k_dim_max[d] for s in stats_list)
                    for d in range(head_dim)
                ]

            result[layer_idx] = avg_stats

        return result

    def get_summary(self) -> Dict[str, Any]:
        """サマリーを返す"""
        stats = self.get_stats()
        if not stats:
            return {}

        all_q_max = max(s.q_max for s in stats.values())
        all_k_max = max(s.k_max for s in stats.values())
        all_q_mean = sum(s.q_mean for s in stats.values()) / len(stats)
        all_k_mean = sum(s.k_mean for s in stats.values()) / len(stats)
        all_q_std = sum(s.q_std for s in stats.values()) / len(stats)
        all_k_std = sum(s.k_std for s in stats.values()) / len(stats)

        layer_q_max = {idx: s.q_max for idx, s in stats.items()}
        layer_k_max = {idx: s.k_max for idx, s in stats.items()}
        layer_q_mean = {idx: s.q_mean for idx, s in stats.items()}
        layer_k_mean = {idx: s.k_mean for idx, s in stats.items()}
        layer_q_std = {idx: s.q_std for idx, s in stats.items()}
        layer_k_std = {idx: s.k_std for idx, s in stats.items()}

        # 周波数帯域分析
        dim_analysis = {}
        first_layer_stats = stats.get(0)
        if first_layer_stats and first_layer_stats.q_dim_max:
            head_dim = len(first_layer_stats.q_dim_max)

            all_q_dim_max = [0.0] * head_dim
            all_k_dim_max = [0.0] * head_dim
            for s in stats.values():
                for d in range(head_dim):
                    all_q_dim_max[d] = max(all_q_dim_max[d], s.q_dim_max[d])
                    all_k_dim_max[d] = max(all_k_dim_max[d], s.k_dim_max[d])

            # RoPE適用次元
            rotary_dim = self._rotary_dim or head_dim
            rotary_half = rotary_dim // 2

            dim_analysis = {
                "q_high_freq_max": max(all_q_dim_max[:rotary_half]) if rotary_half > 0 else 0.0,
                "q_low_freq_max": max(all_q_dim_max[rotary_half:rotary_dim]) if rotary_dim > rotary_half else 0.0,
                "k_high_freq_max": max(all_k_dim_max[:rotary_half]) if rotary_half > 0 else 0.0,
                "k_low_freq_max": max(all_k_dim_max[rotary_half:rotary_dim]) if rotary_dim > rotary_half else 0.0,
                "q_passthrough_max": max(all_q_dim_max[rotary_dim:]) if rotary_dim < head_dim else 0.0,
                "k_passthrough_max": max(all_k_dim_max[rotary_dim:]) if rotary_dim < head_dim else 0.0,
                "rotary_dim": rotary_dim,
                "head_dim": head_dim,
                "q_dim_max": all_q_dim_max,
                "k_dim_max": all_k_dim_max,
            }

        return {
            "all_q_max": all_q_max,
            "all_k_max": all_k_max,
            "all_q_mean": all_q_mean,
            "all_k_mean": all_k_mean,
            "all_q_std": all_q_std,
            "all_k_std": all_k_std,
            "layer_q_max": layer_q_max,
            "layer_k_max": layer_k_max,
            "layer_q_mean": layer_q_mean,
            "layer_k_mean": layer_k_mean,
            "layer_q_std": layer_q_std,
            "layer_k_std": layer_k_std,
            "dim_analysis": dim_analysis,
        }


def apply_rotary_pos_emb(q, k, cos, sin, rotary_ndims):
    """RoPE適用"""
    # rotary_ndims 分だけRoPEを適用
    q_rot = q[..., :rotary_ndims]
    q_pass = q[..., rotary_ndims:]
    k_rot = k[..., :rotary_ndims]
    k_pass = k[..., rotary_ndims:]

    # cos, sin: [1, seq, rotary_ndims]
    # Reshape for broadcasting: [1, 1, seq, rotary_ndims]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply rotary embedding
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate rotary and passthrough parts
    q_out = torch.cat([q_embed, q_pass], dim=-1)
    k_out = torch.cat([k_embed, k_pass], dim=-1)

    return q_out, k_out


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def prepare_data(
    tokenizer_name: str,
    num_samples: int,
    seq_length: int,
    batch_size: int = 32,
) -> DataLoader:
    """Pileデータを準備"""
    from src.utils.data_pythia import load_pile_tokens_cached

    print_flush(f"Preparing data: {num_samples:,} samples, seq_len={seq_length}")

    total_tokens_needed = num_samples * seq_length
    tokens = load_pile_tokens_cached(total_tokens_needed + seq_length, tokenizer_name)

    all_input_ids_list = []
    all_labels_list = []

    for i in range(num_samples):
        start = i * seq_length
        input_ids = tokens[start:start + seq_length]
        labels = tokens[start + 1:start + seq_length + 1]
        all_input_ids_list.append(input_ids)
        all_labels_list.append(labels)

    input_tensor = torch.stack(all_input_ids_list)
    labels_tensor = torch.stack(all_labels_list)

    dataset = TensorDataset(input_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def measure_qk_stats(
    model: GPTNeoXForCausalLM,
    data_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, Any]:
    """Q, K統計を測定"""
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    collector = PretrainedQKCollector(model, num_layers, num_heads, head_dim)
    collector.register_hooks()

    try:
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break

                input_ids, _ = batch
                input_ids = input_ids.to(device)
                model(input_ids)

        summary = collector.get_summary()
    finally:
        collector.remove_hooks()

    return summary


@torch.no_grad()
def measure_ppl(
    model: GPTNeoXForCausalLM,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Perplexityを測定"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    for input_ids, labels in data_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids)
        logits = outputs.logits

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
        num_batches += 1

        clear_gpu_cache(device)

    avg_loss = total_loss / num_batches
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return ppl


def main():
    parser = argparse.ArgumentParser(description="Measure Q, K statistics on pretrained Pythia")
    parser.add_argument(
        "--model",
        type=str,
        default="pythia-70m",
        choices=["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"],
        help="Model size",
    )
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches for Q, K measurement")
    args = parser.parse_args()

    set_seed(42)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_flush(f"Device: cuda ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        device = torch.device("cpu")
        print_flush("Device: cpu")

    model_name = f"EleutherAI/{args.model}"

    print_flush("=" * 70)
    print_flush("PRETRAINED PYTHIA Q, K STATISTICS")
    print_flush("=" * 70)
    print_flush(f"Model: {model_name}")
    print_flush(f"Samples: {args.samples:,}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Batch size: {args.batch_size}")
    print_flush(f"Q, K measurement batches: {args.num_batches}")
    print_flush("=" * 70)

    # Load model
    print_flush(f"\n[Model] Loading {model_name}...")
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    # rotary_dim is computed from rotary_pct (default 0.25 for Pythia)
    rotary_pct = getattr(config, 'rotary_pct', 0.25)
    rotary_dim = int(head_dim * rotary_pct)
    # RoPE base from config
    rope_theta = getattr(config, 'rotary_emb_base', 10000)

    print_flush(f"  Hidden size: {config.hidden_size}")
    print_flush(f"  Layers: {config.num_hidden_layers}")
    print_flush(f"  Heads: {config.num_attention_heads}")
    print_flush(f"  Head dim: {head_dim}")
    print_flush(f"  Rotary pct: {rotary_pct}")
    print_flush(f"  Rotary dim: {rotary_dim}")
    print_flush(f"  RoPE base: {rope_theta}")
    print_flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare data
    print_flush("\n[Data] Loading Pile data...")
    data_loader = prepare_data(
        tokenizer_name=model_name,
        num_samples=args.samples,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
    )

    # Measure PPL
    print_flush("\n[Evaluation] Measuring perplexity...")
    ppl = measure_ppl(model, data_loader, device)
    print_flush(f"  Perplexity: {ppl:.1f}")

    # Measure Q, K statistics
    print_flush(f"\n[Q, K Statistics] Measuring ({args.num_batches} batches)...")
    qk_stats = measure_qk_stats(model, data_loader, device, num_batches=args.num_batches)

    if qk_stats:
        print_flush(f"\n  Overall Q: max={qk_stats['all_q_max']:.2f}, mean={qk_stats['all_q_mean']:.2f}, std={qk_stats['all_q_std']:.2f}")
        print_flush(f"  Overall K: max={qk_stats['all_k_max']:.2f}, mean={qk_stats['all_k_mean']:.2f}, std={qk_stats['all_k_std']:.2f}")

        print_flush("\n  Layer-wise statistics:")
        print_flush("  | Layer | Q max | Q mean | Q std | K max | K mean | K std |")
        print_flush("  |-------|-------|--------|-------|-------|--------|-------|")
        for layer_idx in sorted(qk_stats['layer_q_max'].keys()):
            q_max = qk_stats['layer_q_max'][layer_idx]
            q_mean = qk_stats['layer_q_mean'][layer_idx]
            q_std = qk_stats['layer_q_std'][layer_idx]
            k_max = qk_stats['layer_k_max'][layer_idx]
            k_mean = qk_stats['layer_k_mean'][layer_idx]
            k_std = qk_stats['layer_k_std'][layer_idx]
            print_flush(f"  | {layer_idx} | {q_max:.2f} | {q_mean:.2f} | {q_std:.2f} | {k_max:.2f} | {k_mean:.2f} | {k_std:.2f} |")

        if qk_stats.get('dim_analysis'):
            dim = qk_stats['dim_analysis']
            rotary_dim = dim.get('rotary_dim', 16)
            head_dim = dim.get('head_dim', 64)
            rotary_half = rotary_dim // 2

            print_flush(f"\n  Frequency band analysis (rotary_dim={rotary_dim}, head_dim={head_dim}):")
            print_flush(f"    RoPE applied: dims 0-{rotary_dim-1}")
            print_flush(f"      High-freq (dims 0-{rotary_half-1}):  Q={dim['q_high_freq_max']:.2f}, K={dim['k_high_freq_max']:.2f}")
            print_flush(f"      Low-freq (dims {rotary_half}-{rotary_dim-1}): Q={dim['q_low_freq_max']:.2f}, K={dim['k_low_freq_max']:.2f}")

            if rotary_dim < head_dim:
                print_flush(f"    Passthrough (dims {rotary_dim}-{head_dim-1}): Q={dim['q_passthrough_max']:.2f}, K={dim['k_passthrough_max']:.2f}")

            if dim['q_high_freq_max'] > 0:
                q_ratio = dim['q_low_freq_max'] / dim['q_high_freq_max']
                print_flush(f"    Q low/high ratio: {q_ratio:.2f}x")
            if dim['k_high_freq_max'] > 0:
                k_ratio = dim['k_low_freq_max'] / dim['k_high_freq_max']
                print_flush(f"    K low/high ratio: {k_ratio:.2f}x")

    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush(f"| Model | PPL | Q max | Q mean | Q std | K max | K mean | K std |")
    print_flush(f"|-------|-----|-------|--------|-------|-------|--------|-------|")
    print_flush(f"| {args.model} | {ppl:.1f} | {qk_stats['all_q_max']:.2f} | {qk_stats['all_q_mean']:.2f} | {qk_stats['all_q_std']:.2f} | {qk_stats['all_k_max']:.2f} | {qk_stats['all_k_mean']:.2f} | {qk_stats['all_k_std']:.2f} |")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
