"""
Evaluation utilities for experiments.

共通の評価関数を提供する。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
    # 次元ごとの最大値（高周波 vs 低周波の比較用）
    q_dim_max: List[float] = field(default_factory=list)
    k_dim_max: List[float] = field(default_factory=list)


class QKStatsCollector:
    """
    Q, Kの統計情報を収集するフッククラス

    Usage:
        collector = QKStatsCollector(model, num_layers=6, head_dim=64)
        collector.register_hooks()

        # Forward pass
        model(input_ids)

        # Get stats
        stats = collector.get_stats()
        collector.clear()

        # Remove hooks when done
        collector.remove_hooks()
    """

    def __init__(self, model: nn.Module, num_layers: int, head_dim: int):
        self.model = model
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.hooks: List[Any] = []
        self.stats: Dict[int, List[QKStats]] = {i: [] for i in range(num_layers)}

    def _create_hook(self, layer_idx: int):
        """レイヤーごとのフックを作成"""

        def hook_fn(module, input_args, output):
            # UnifiedAttentionのforwardから直接Q, Kを取得するのは難しいので、
            # モジュールの属性として一時保存する方式を使う
            if hasattr(module, '_last_q') and hasattr(module, '_last_k'):
                q = module._last_q
                k = module._last_k

                with torch.no_grad():
                    # 全体統計
                    q_abs = q.abs()
                    k_abs = k.abs()

                    stats = QKStats(
                        layer_idx=layer_idx,
                        q_max=q_abs.max().item(),
                        q_mean=q_abs.mean().item(),
                        q_std=q.std().item(),
                        k_max=k_abs.max().item(),
                        k_mean=k_abs.mean().item(),
                        k_std=k.std().item(),
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
        for layer_idx, layer in enumerate(self.model.layers):
            hook = layer.attention.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """フックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear(self) -> None:
        """収集した統計をクリア"""
        for layer_idx in self.stats:
            self.stats[layer_idx].clear()

    def get_stats(self) -> Dict[int, QKStats]:
        """
        各レイヤーの統計を取得（複数バッチの平均）

        Returns:
            Dict[layer_idx, QKStats]: レイヤーごとの統計
        """
        result = {}
        for layer_idx, stats_list in self.stats.items():
            if not stats_list:
                continue

            # 複数バッチの平均を計算
            avg_stats = QKStats(layer_idx=layer_idx)

            n = len(stats_list)
            avg_stats.q_max = max(s.q_max for s in stats_list)
            avg_stats.q_mean = sum(s.q_mean for s in stats_list) / n
            avg_stats.q_std = sum(s.q_std for s in stats_list) / n
            avg_stats.k_max = max(s.k_max for s in stats_list)
            avg_stats.k_mean = sum(s.k_mean for s in stats_list) / n
            avg_stats.k_std = sum(s.k_std for s in stats_list) / n

            # 次元ごとの最大値（全バッチでの最大）
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

    def get_summary(self, rotary_dim: Optional[int] = None) -> Dict[str, Any]:
        """
        全レイヤーのサマリーを取得

        Args:
            rotary_dim: RoPEが適用される次元数。指定しない場合はhead_dim全体を使用。

        Returns:
            Dict with summary statistics
        """
        stats = self.get_stats()
        if not stats:
            return {}

        # 全レイヤーでの最大値
        all_q_max = max(s.q_max for s in stats.values())
        all_k_max = max(s.k_max for s in stats.values())

        # 全レイヤーでの平均（レイヤー平均の平均）
        all_q_mean = sum(s.q_mean for s in stats.values()) / len(stats)
        all_k_mean = sum(s.k_mean for s in stats.values()) / len(stats)

        # 全レイヤーでの標準偏差（レイヤー平均の平均）
        all_q_std = sum(s.q_std for s in stats.values()) / len(stats)
        all_k_std = sum(s.k_std for s in stats.values()) / len(stats)

        # レイヤーごとの最大値
        layer_q_max = {idx: s.q_max for idx, s in stats.items()}
        layer_k_max = {idx: s.k_max for idx, s in stats.items()}

        # レイヤーごとの平均
        layer_q_mean = {idx: s.q_mean for idx, s in stats.items()}
        layer_k_mean = {idx: s.k_mean for idx, s in stats.items()}

        # レイヤーごとの標準偏差
        layer_q_std = {idx: s.q_std for idx, s in stats.items()}
        layer_k_std = {idx: s.k_std for idx, s in stats.items()}

        # 次元ごとの分析（高周波 vs 低周波）
        first_layer_stats = stats.get(0)
        dim_analysis = {}
        if first_layer_stats and first_layer_stats.q_dim_max:
            head_dim = len(first_layer_stats.q_dim_max)

            # 全レイヤーで次元ごとの最大値を集計
            all_q_dim_max = [0.0] * head_dim
            all_k_dim_max = [0.0] * head_dim
            for s in stats.values():
                for d in range(head_dim):
                    all_q_dim_max[d] = max(all_q_dim_max[d], s.q_dim_max[d])
                    all_k_dim_max[d] = max(all_k_dim_max[d], s.k_dim_max[d])

            # RoPEが適用される次元内で高周波/低周波を分析
            # rotary_dim内: 前半が高周波、後半が低周波
            # rotary_dim外: パススルー（位置情報なし）
            if rotary_dim is None:
                rotary_dim = head_dim

            rotary_half = rotary_dim // 2

            dim_analysis = {
                # RoPE適用範囲内での高周波/低周波
                "q_high_freq_max": max(all_q_dim_max[:rotary_half]) if rotary_half > 0 else 0.0,
                "q_low_freq_max": max(all_q_dim_max[rotary_half:rotary_dim]) if rotary_dim > rotary_half else 0.0,
                "k_high_freq_max": max(all_k_dim_max[:rotary_half]) if rotary_half > 0 else 0.0,
                "k_low_freq_max": max(all_k_dim_max[rotary_half:rotary_dim]) if rotary_dim > rotary_half else 0.0,
                # パススルー部分（RoPE未適用）
                "q_passthrough_max": max(all_q_dim_max[rotary_dim:]) if rotary_dim < head_dim else 0.0,
                "k_passthrough_max": max(all_k_dim_max[rotary_dim:]) if rotary_dim < head_dim else 0.0,
                # メタ情報
                "rotary_dim": rotary_dim,
                "head_dim": head_dim,
                # 全次元のmax値
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
            "per_layer": stats,
        }


def analyze_qk_stats(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, Any]:
    """
    モデルのQ, K統計を分析する

    Args:
        model: UnifiedPythiaModel
        val_loader: Validation data loader
        device: Device
        num_batches: 分析に使用するバッチ数

    Returns:
        統計情報のDict
    """
    model.eval()

    # モデル情報を取得
    num_layers = len(model.layers)
    head_dim = model.head_dim

    # rotary_dimを位置エンコーディングから取得
    rotary_dim = None
    first_layer = model.layers[0]
    pos_encoding = first_layer.attention.pos_encoding
    if hasattr(pos_encoding, 'rotary_dim'):
        rotary_dim = pos_encoding.rotary_dim

    # コレクターを作成
    collector = QKStatsCollector(model, num_layers, head_dim)

    # Attentionモジュールを修正してQ, Kを保存するようにする
    original_forwards = []
    for layer in model.layers:
        original_forward = layer.attention.forward
        original_forwards.append(original_forward)

        def make_hook_forward(orig_forward, attn_module):
            def hooked_forward(hidden_states, attention_mask=None):
                batch_size, seq_len, _ = hidden_states.shape

                # QKV projection
                qkv = attn_module.query_key_value(hidden_states)
                qkv = qkv.view(batch_size, seq_len, 3, attn_module.num_heads, attn_module.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                query, key, value = qkv[0], qkv[1], qkv[2]

                # Apply position encoding to Q/K
                query, key = attn_module.pos_encoding.apply_to_qk(query, key, seq_len)

                # Q, Kを一時保存
                attn_module._last_q = query
                attn_module._last_k = key

                # 残りの計算
                attn_weights = torch.matmul(query, key.transpose(-1, -2)) * attn_module.scale
                attn_weights = attn_module.pos_encoding.apply_to_scores(attn_weights, seq_len)
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, value)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(batch_size, seq_len, attn_module.hidden_size)
                output = attn_module.dense(attn_output)
                return output

            return hooked_forward

        layer.attention.forward = make_hook_forward(original_forward, layer.attention)

    # フックを登録
    collector.register_hooks()

    try:
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break

                input_ids, _ = batch
                input_ids = input_ids.to(device)
                model(input_ids)

        # 統計を取得（rotary_dimを渡して正しい周波数帯域分析を行う）
        summary = collector.get_summary(rotary_dim=rotary_dim)

    finally:
        # フックを削除
        collector.remove_hooks()

        # 元のforwardを復元
        for layer, orig_forward in zip(model.layers, original_forwards):
            layer.attention.forward = orig_forward

        # 一時属性を削除
        for layer in model.layers:
            if hasattr(layer.attention, '_last_q'):
                delattr(layer.attention, '_last_q')
            if hasattr(layer.attention, '_last_k'):
                delattr(layer.attention, '_last_k')

    return summary


def evaluate_ppl(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    return_recon_loss: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Evaluate model perplexity.

    Args:
        model: Model to evaluate (must return (logits, recon_loss) if return_recon_loss=True)
        val_loader: Validation data loader
        device: Device
        return_recon_loss: Whether to return reconstruction loss

    Returns:
        If return_recon_loss=False: PPL as float
        If return_recon_loss=True: Dict with 'ppl' and 'recon_loss'
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if return_recon_loss:
                logits, recon_loss = model(input_ids, return_reconstruction_loss=True)
            else:
                output = model(input_ids)
                # Handle both (logits,) tuple and logits tensor
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                recon_loss = None

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += labels.numel()

            if recon_loss is not None:
                total_recon_loss += recon_loss.item()
                num_batches += 1

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    if return_recon_loss:
        avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
        return {"ppl": ppl, "recon_loss": avg_recon}
    else:
        return ppl


def evaluate_position_wise_ppl(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    position_ranges: Optional[list] = None,
    return_recon_loss: bool = False,
) -> Dict[str, float]:
    """
    Evaluate position-wise perplexity.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device
        position_ranges: List of (start, end) tuples for position ranges.
                        If None, uses default ranges for seq_len.
        return_recon_loss: Whether model returns reconstruction loss

    Returns:
        Dictionary with position range keys and PPL values
    """
    model.eval()

    # Get sequence length from first batch
    first_batch = next(iter(val_loader))
    seq_len = first_batch[0].shape[1]

    # Default position ranges
    if position_ranges is None:
        position_ranges = get_default_position_ranges(seq_len)

    # Initialize accumulators for each range
    range_losses: Dict[str, float] = {}
    range_tokens: Dict[str, int] = {}
    for start, end in position_ranges:
        key = f"{start}-{end}"
        range_losses[key] = 0.0
        range_tokens[key] = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if return_recon_loss:
                logits, _ = model(input_ids, return_reconstruction_loss=True)
            else:
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

            # Compute per-position loss
            for start, end in position_ranges:
                key = f"{start}-{end}"

                range_logits = logits[:, start:end, :]
                range_labels = labels[:, start:end]

                loss = nn.functional.cross_entropy(
                    range_logits.reshape(-1, range_logits.size(-1)),
                    range_labels.reshape(-1),
                    reduction="sum",
                )

                range_losses[key] += loss.item()
                range_tokens[key] += range_labels.numel()

    # Compute PPL for each range
    results: Dict[str, float] = {}
    for key in range_losses:
        if range_tokens[key] > 0:
            avg_loss = range_losses[key] / range_tokens[key]
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            results[key] = ppl
        else:
            results[key] = float("inf")

    return results


def get_default_position_ranges(seq_len: int) -> list:
    """
    Get default position ranges for position-wise PPL evaluation.

    Args:
        seq_len: Sequence length

    Returns:
        List of (start, end) tuples
    """
    return [
        (0, 16),
        (16, 32),
        (32, 64),
        (64, 96),
        (96, seq_len),
    ]


def evaluate_reversal_curse(
    model: nn.Module,
    tokenizer: Any,
    pairs: List[Dict[str, str]],
    device: torch.device,
    max_length: int = 64,
) -> Dict[str, float]:
    """
    Evaluate Reversal Curse.

    順方向（"A is B"）と逆方向（"B is A"）のPPLを比較し、
    モデルが双方向の知識を持っているかを評価する。

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        pairs: List of dicts with 'forward' and 'backward' keys
        device: Device
        max_length: Maximum sequence length

    Returns:
        Dict with forward_ppl, backward_ppl, reversal_ratio, etc.
    """
    model.eval()

    forward_losses: List[float] = []
    backward_losses: List[float] = []

    with torch.no_grad():
        for pair in pairs:
            # Forward direction
            forward_loss = _compute_sentence_loss(
                model, tokenizer, pair["forward"], device, max_length
            )
            forward_losses.append(forward_loss)

            # Backward direction
            backward_loss = _compute_sentence_loss(
                model, tokenizer, pair["backward"], device, max_length
            )
            backward_losses.append(backward_loss)

    # Compute average PPL
    avg_forward_loss = sum(forward_losses) / len(forward_losses)
    avg_backward_loss = sum(backward_losses) / len(backward_losses)

    forward_ppl = torch.exp(torch.tensor(avg_forward_loss)).item()
    backward_ppl = torch.exp(torch.tensor(avg_backward_loss)).item()

    # Reversal Ratio: closer to 1.0 = less reversal curse
    # < 1.0 means backward is harder (typical reversal curse)
    reversal_ratio = forward_ppl / backward_ppl if backward_ppl > 0 else float("inf")

    # Reversal Gap: difference in PPL
    reversal_gap = backward_ppl - forward_ppl

    return {
        "forward_ppl": forward_ppl,
        "backward_ppl": backward_ppl,
        "reversal_ratio": reversal_ratio,
        "reversal_gap": reversal_gap,
        "num_pairs": len(pairs),
    }


def _compute_sentence_loss(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: torch.device,
    max_length: int,
) -> float:
    """
    Compute average cross-entropy loss for a single sentence.

    Args:
        model: Model
        tokenizer: Tokenizer
        sentence: Input sentence
        device: Device
        max_length: Max length

    Returns:
        Average loss per token
    """
    # Tokenize
    tokens = tokenizer.encode(sentence, add_special_tokens=False)

    # Truncate if needed
    if len(tokens) > max_length - 1:
        tokens = tokens[: max_length - 1]

    # Create input and label
    input_ids = torch.tensor([tokens[:-1]], device=device)
    labels = torch.tensor([tokens[1:]], device=device)

    if input_ids.size(1) == 0:
        return 0.0

    # Forward pass
    output = model(input_ids)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    # Compute loss
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="mean",
    )

    return loss.item()
