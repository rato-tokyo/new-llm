#!/usr/bin/env python3
"""
HSA vs Memory Norm 比較実験

2つのLandmark計算方式を正確に比較:
- HSA方式: ChunkEncoder([CLS] + Keys)[CLS] - 双方向エンコーダ（論文準拠）
- memory_norm方式: Σσ(k) - 書き込み操作の副産物

使用例:
    python3 scripts/experiment_hsa_vs_memory_norm.py
    python3 scripts/experiment_hsa_vs_memory_norm.py --epochs 30 --num-memories 8
"""

import argparse
import sys
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.pythia import PythiaConfig
from src.models import TransformerLM
from src.models.layers import PythiaLayer, BaseLayer, ChunkEncoder
from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
from torch.utils.data import DataLoader, TensorDataset
from src.utils import (
    set_seed,
    get_device,
    get_tokenizer,
    print_flush,
    train_model,
)
from datasets import load_dataset


# =============================================================================
# Landmark方式を切り替え可能なMultiMemoryAttention
# =============================================================================

LandmarkType = Literal["hsa", "memory_norm"]


class MultiMemoryAttentionComparable(nn.Module):
    """Landmark方式を切り替え可能なMultiMemoryAttention

    比較実験用。landmark_typeで方式を切り替え。

    - hsa: ChunkEncoder([CLS] + Keys)[CLS] - 双方向エンコーダ（論文準拠）
    - memory_norm: Σσ(k) - 書き込み操作の副産物
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memories: int = 4,
        use_delta_rule: bool = False,  # 数値安定性のためFalse
        landmark_type: LandmarkType = "hsa",
        max_keys_per_memory: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_memories = num_memories
        self.use_delta_rule = use_delta_rule
        self.landmark_type = landmark_type
        self.max_keys_per_memory = max_keys_per_memory

        # Q/K/V射影
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # HSA方式: 検索専用 Q_slc 射影（Attention用Qとは別）
        if landmark_type == "hsa":
            self.w_q_slc = nn.Linear(hidden_size, hidden_size, bias=False)
            # 各ヘッド用のChunkEncoder
            self.chunk_encoders = nn.ModuleList([
                ChunkEncoder(
                    head_dim=self.head_dim,
                    num_encoder_heads=max(1, self.head_dim // 16),
                    num_encoder_layers=2,
                )
                for _ in range(num_heads)
            ])

        self.gate = nn.Parameter(torch.zeros(num_heads))
        self.scale = self.head_dim ** -0.5

        # メモリ状態
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None
        # HSA方式用: キー列を保持
        self.key_sequences: Optional[list[list[torch.Tensor]]] = None
        self.register_buffer('current_memory_idx', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        self.memories = [
            torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memories)
        ]
        self.memory_norms = [
            torch.zeros(self.num_heads, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memories)
        ]
        # HSA方式用: キー列
        if self.landmark_type == "hsa":
            self.key_sequences = [
                [torch.zeros(0, self.head_dim, device=device, dtype=dtype)
                 for _ in range(self.num_heads)]
                for _ in range(self.num_memories)
            ]
        self.current_memory_idx = torch.tensor(0, device=device)

    def _compute_landmark_hsa(self, memory_idx: int, head_idx: int) -> torch.Tensor:
        """HSA方式: ChunkEncoderでLandmarkを計算"""
        assert self.key_sequences is not None
        key_seq = self.key_sequences[memory_idx][head_idx]

        if key_seq.size(0) == 0:
            return torch.zeros(self.head_dim, device=key_seq.device, dtype=key_seq.dtype)

        return self.chunk_encoders[head_idx](key_seq)

    def _compute_relevance(
        self, q: torch.Tensor, q_slc: Optional[torch.Tensor], memory_idx: int
    ) -> torch.Tensor:
        """メモリとの関連度を計算

        Args:
            q: (batch, num_heads, seq_len, head_dim) - Attention用Query
            q_slc: (batch, num_heads, seq_len, head_dim) - 検索専用Query（HSA方式のみ）
            memory_idx: メモリインデックス

        Returns:
            relevance: (batch, num_heads) - シーケンス平均のスコア
        """
        assert self.memory_norms is not None

        if self.landmark_type == "hsa":
            # HSA方式: ChunkEncoderでLandmark計算、Q_slcで検索
            assert q_slc is not None
            landmarks = []
            for h in range(self.num_heads):
                landmark = self._compute_landmark_hsa(memory_idx, h)
                landmarks.append(landmark)
            landmarks_tensor = torch.stack(landmarks, dim=0)  # (num_heads, head_dim)
            rel = torch.einsum('bhsd,hd->bhs', q_slc, landmarks_tensor) * self.scale
            return rel.mean(dim=-1)
        else:
            # memory_norm方式: Σσ(k)をLandmarkとして使用
            landmark = self.memory_norms[memory_idx]  # (num_heads, head_dim)
            rel = torch.einsum('bhsd,hd->bhs', q, landmark) * self.scale
            return rel.mean(dim=-1)

    def _is_memory_empty(self, memory_idx: int) -> bool:
        """メモリが空かどうかをチェック"""
        if self.landmark_type == "hsa":
            if self.key_sequences is None:
                return True
            total_keys = sum(
                self.key_sequences[memory_idx][h].size(0)
                for h in range(self.num_heads)
            )
            return total_keys == 0
        else:
            if self.memory_norms is None:
                return True
            return bool(self.memory_norms[memory_idx].sum() < 1e-6)

    def _retrieve_from_memory(
        self, q: torch.Tensor, q_slc: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """全メモリから検索し、Landmark関連度で加重統合"""
        if self.memories is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)
        outputs, relevances = [], []
        has_non_empty = False

        for memory_idx, (memory, memory_norm) in enumerate(
            zip(self.memories, self.memory_norms)  # type: ignore
        ):
            if self._is_memory_empty(memory_idx):
                outputs.append(torch.zeros_like(q))
                relevances.append(
                    torch.full((q.size(0), self.num_heads), float('-inf'), device=q.device)
                )
            else:
                has_non_empty = True
                # Linear Attentionでメモリから検索
                a_mem = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
                norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
                outputs.append(a_mem / norm.clamp(min=1e-6).unsqueeze(-1))

                # 関連度計算
                relevances.append(self._compute_relevance(q, q_slc, memory_idx))

        if not has_non_empty:
            return torch.zeros_like(q)

        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0)
        return (stacked * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """メモリを更新"""
        sigma_k = elu_plus_one(k)

        if self.memories is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None

        idx = int(self.current_memory_idx.item())
        memory = self.memories[idx]
        memory_norm = self.memory_norms[idx]

        batch_size, num_heads, seq_len, head_dim = k.shape

        # メモリ更新（シンプルな加算方式 - 数値安定性のため）
        update = torch.einsum('bhsd,bhse->hde', sigma_k, v) / (batch_size * seq_len)

        # NaN/Inf対策
        if not torch.isfinite(update).all():
            return

        self.memories[idx] = (memory + update).detach()
        norm_update = sigma_k.sum(dim=(0, 2)) / batch_size
        if torch.isfinite(norm_update).all():
            self.memory_norms[idx] = (memory_norm + norm_update).detach()

        # HSA方式: キー列を更新
        if self.landmark_type == "hsa" and self.key_sequences is not None:
            for h in range(num_heads):
                new_keys = k[:, h, :, :].reshape(-1, head_dim).detach()
                current_keys = self.key_sequences[idx][h]
                combined = torch.cat([current_keys, new_keys], dim=0)

                if combined.size(0) > self.max_keys_per_memory:
                    combined = combined[-self.max_keys_per_memory:]

                self.key_sequences[idx][h] = combined

        self.current_memory_idx = torch.tensor(
            (idx + 1) % self.num_memories, device=k.device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Q/K/V
        q = self.w_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # HSA方式: 検索専用Q_slc
        q_slc = None
        if self.landmark_type == "hsa":
            q_slc = self.w_q_slc(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        # メモリ検索
        memory_output = self._retrieve_from_memory(q, q_slc)
        local_output = causal_linear_attention(q, k, v)

        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        output = gate * memory_output + (1 - gate) * local_output

        if update_memory:
            self._update_memory(k, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.w_o(output)


class MultiMemoryLayerComparable(BaseLayer):
    """比較実験用MultiMemoryLayer"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_memories: int = 4,
        use_delta_rule: bool = False,
        landmark_type: LandmarkType = "hsa",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # LayerNorm
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

        self.attention = MultiMemoryAttentionComparable(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_memories=num_memories,
            use_delta_rule=use_delta_rule,
            landmark_type=landmark_type,
        )
        self.mlp = PythiaMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states, attention_mask, update_memory=update_memory
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        self.attention.reset_memory(device)


# =============================================================================
# 実験関数
# =============================================================================

def create_model(
    config: PythiaConfig,
    landmark_type: LandmarkType,
    num_memories: int,
) -> TransformerLM:
    """比較用モデルを作成"""
    layers: list[BaseLayer] = [
        MultiMemoryLayerComparable(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_memories=num_memories,
            use_delta_rule=False,
            landmark_type=landmark_type,
        )
    ]
    layers.extend([
        PythiaLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
        )
        for _ in range(config.num_layers - 1)
    ])
    return TransformerLM(
        layers=layers,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
    )


def prepare_data(
    tokenizer,
    num_samples: int,
    seq_length: int,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """データを準備"""
    print_flush(f"Loading {num_samples} samples from Pile...")

    # Pileデータセットから直接ロード
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    all_tokens: list[int] = []
    target_tokens = num_samples * seq_length * 2  # 十分な量を確保

    for example in dataset:
        text = example["text"]  # type: ignore
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        if len(all_tokens) >= target_tokens:
            break

    # シーケンスに分割
    all_tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    total_seqs = len(all_tokens_tensor) // seq_length
    all_tokens_tensor = all_tokens_tensor[:total_seqs * seq_length].view(-1, seq_length)

    # 必要なサンプル数に制限
    if len(all_tokens_tensor) > num_samples:
        all_tokens_tensor = all_tokens_tensor[:num_samples]

    # Train/Val分割
    num_val = max(1, int(len(all_tokens_tensor) * val_ratio))
    num_train = len(all_tokens_tensor) - num_val

    train_data = all_tokens_tensor[:num_train]
    val_data = all_tokens_tensor[num_train:]

    print_flush(f"  Train: {len(train_data)} samples")
    print_flush(f"  Val: {len(val_data)} samples")

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=8,
        shuffle=False,
    )

    return train_loader, val_loader


def run_experiment(
    landmark_type: LandmarkType,
    config: PythiaConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_memories: int,
    epochs: int,
    device: torch.device,
    patience: int = 5,
) -> dict:
    """単一の実験を実行"""
    print_flush(f"\n{'='*60}")
    print_flush(f"Running experiment: {landmark_type}")
    print_flush(f"{'='*60}")

    model = create_model(config, landmark_type, num_memories)
    model = model.to(device)

    # パラメータ数
    num_params = sum(p.numel() for p in model.parameters())
    print_flush(f"Parameters: {num_params:,}")

    # メモリリセット
    for layer in model.layers:
        if hasattr(layer, 'reset_memory'):
            layer.reset_memory(device)

    # 訓練
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=epochs,
        learning_rate=1e-4,
        patience=patience,
        model_name=f"MultiMemory-{landmark_type}",
    )

    return {
        "landmark_type": landmark_type,
        "best_ppl": result["best_val_ppl"],
        "best_epoch": result["best_epoch"],
        "num_params": num_params,
        "history": result["history"],
    }


def main():
    parser = argparse.ArgumentParser(description="HSA vs Memory Norm Comparison")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs")
    parser.add_argument("--num-memories", type=int, default=4, help="Number of memories")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = get_tokenizer()

    print_flush("=" * 60)
    print_flush("HSA vs Memory Norm Comparison Experiment")
    print_flush("=" * 60)
    print_flush(f"Device: {device}")
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Seq length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Memories: {args.num_memories}")
    print_flush(f"Patience: {args.patience}")
    print_flush()

    # Config（PythiaConfigはクラス属性で定義されているのでインスタンス化のみ）
    config = PythiaConfig()

    # データ準備
    train_loader, val_loader = prepare_data(
        tokenizer, args.samples, args.seq_length
    )

    # 実験実行
    results = {}

    # memory_norm方式
    results["memory_norm"] = run_experiment(
        landmark_type="memory_norm",
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        num_memories=args.num_memories,
        epochs=args.epochs,
        device=device,
        patience=args.patience,
    )

    # HSA方式（ChunkEncoder）
    results["hsa"] = run_experiment(
        landmark_type="hsa",
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        num_memories=args.num_memories,
        epochs=args.epochs,
        device=device,
        patience=args.patience,
    )

    # 結果まとめ
    print_flush("\n" + "=" * 60)
    print_flush("Results Summary")
    print_flush("=" * 60)
    print_flush(f"{'Method':<15} {'Best PPL':>12} {'Epoch':>8} {'Params':>15}")
    print_flush("-" * 60)

    for name, result in results.items():
        print_flush(
            f"{name:<15} {result['best_ppl']:>12.2f} {result['best_epoch']:>8} "
            f"{result['num_params']:>15,}"
        )

    # 比較
    print_flush("\n" + "=" * 60)
    print_flush("Comparison")
    print_flush("=" * 60)

    hsa_ppl = results["hsa"]["best_ppl"]
    norm_ppl = results["memory_norm"]["best_ppl"]
    diff = hsa_ppl - norm_ppl

    print_flush(f"HSA PPL: {hsa_ppl:.2f}")
    print_flush(f"Memory Norm PPL: {norm_ppl:.2f}")
    print_flush(f"Difference (HSA - Memory Norm): {diff:+.2f}")

    if diff < -1.0:
        print_flush("→ HSA方式（ChunkEncoder）が優れている")
    elif diff > 1.0:
        print_flush("→ Memory Norm方式が優れている")
    else:
        print_flush("→ ほぼ同等")

    print_flush("\n" + "=" * 60)
    print_flush("Notes")
    print_flush("=" * 60)
    print_flush("- HSA方式: 双方向エンコーダ（ChunkEncoder）でLandmarkを計算")
    print_flush("  Landmark = ChunkEncoder([CLS] + Keys)[CLS]")
    print_flush("  検索にはQ_slc（Attention用Qとは別の射影）を使用")
    print_flush()
    print_flush("- Memory Norm方式: 書き込み操作の副産物をLandmarkとして使用")
    print_flush("  Landmark = Σσ(k)")
    print_flush("  追加パラメータなし")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
