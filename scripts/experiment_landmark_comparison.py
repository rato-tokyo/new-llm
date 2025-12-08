#!/usr/bin/env python3
"""
Landmark方式比較実験: HSA vs memory_norm

2つのLandmark計算方式を比較:
- HSA方式: Landmark = mean(K) - キーの平均方向
- memory_norm方式: Landmark = Σσ(k) - 書き込み操作の副産物

使用例:
    python3 scripts/experiment_landmark_comparison.py
    python3 scripts/experiment_landmark_comparison.py --epochs 50 --num-memories 8
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
from src.models.layers import PythiaLayer, BaseLayer
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
from src.utils.data_loading import load_long_documents_from_pile


# =============================================================================
# Landmark方式を切り替え可能なMultiMemoryAttention
# =============================================================================

LandmarkType = Literal["hsa", "memory_norm"]


class MultiMemoryAttentionComparable(nn.Module):
    """Landmark方式を切り替え可能なMultiMemoryAttention

    比較実験用。landmark_typeで方式を切り替え。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memories: int = 4,
        use_delta_rule: bool = True,
        landmark_type: LandmarkType = "hsa",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_memories = num_memories
        self.use_delta_rule = use_delta_rule
        self.landmark_type = landmark_type

        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.gate = nn.Parameter(torch.zeros(num_heads))

        # メモリ状態
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None
        self.landmarks: Optional[list[torch.Tensor]] = None  # HSA方式用
        self.key_counts: Optional[list[torch.Tensor]] = None  # HSA方式用
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
        # HSA方式用
        self.landmarks = [
            torch.zeros(self.num_heads, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memories)
        ]
        self.key_counts = [
            torch.zeros(self.num_heads, device=device, dtype=dtype)
            for _ in range(self.num_memories)
        ]
        self.current_memory_idx = torch.tensor(0, device=device)

    def _compute_relevance(self, q: torch.Tensor, idx: int) -> torch.Tensor:
        """クエリとLandmarkの関連度を計算

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            idx: メモリインデックス

        Returns:
            relevance: (batch, num_heads)
        """
        assert self.memory_norms is not None
        assert self.landmarks is not None

        if self.landmark_type == "hsa":
            # HSA方式: Q @ Landmark（σ変換なしの内積）
            landmark = self.landmarks[idx]
            rel = torch.einsum('bhsd,hd->bhs', q, landmark)
        else:
            # memory_norm方式: σ(Q) @ memory_norm
            sigma_q = elu_plus_one(q)
            memory_norm = self.memory_norms[idx]
            rel = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)

        return rel.mean(dim=-1)

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        """全メモリから検索し、Landmark関連度で加重統合"""
        if self.memories is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)
        outputs, relevances = [], []

        for idx, (memory, memory_norm) in enumerate(zip(self.memories, self.memory_norms)):  # type: ignore
            # メモリが空かチェック
            is_empty = memory_norm.sum() < 1e-6  # type: ignore

            if is_empty:
                outputs.append(torch.zeros_like(q))
                relevances.append(
                    torch.full((q.size(0), self.num_heads), float('-inf'), device=q.device)
                )
            else:
                # Linear Attentionでメモリから検索
                a_mem = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
                norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
                outputs.append(a_mem / norm.clamp(min=1e-6).unsqueeze(-1))

                # 関連度計算（方式による）
                relevances.append(self._compute_relevance(q, idx))

        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0)
        return (stacked * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """メモリを更新"""
        sigma_k = elu_plus_one(k)

        if self.memories is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None
        assert self.landmarks is not None and self.key_counts is not None

        idx = int(self.current_memory_idx.item())
        memory = self.memories[idx]
        memory_norm = self.memory_norms[idx]

        # メモリ更新（共通）
        if self.use_delta_rule:
            retrieved = torch.einsum('bhsd,hde->bhse', sigma_k, memory)
            norm = torch.einsum('bhsd,hd->bhs', sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = v - retrieved / norm
            update = torch.einsum('bhsd,bhse->hde', sigma_k, delta_v) / (k.size(0) * k.size(2))
        else:
            update = torch.einsum('bhsd,bhse->hde', sigma_k, v) / (k.size(0) * k.size(2))

        self.memories[idx] = (memory + update).detach()
        self.memory_norms[idx] = (memory_norm + sigma_k.sum(dim=(0, 2)) / k.size(0)).detach()

        # HSA方式の場合のみLandmark更新
        if self.landmark_type == "hsa":
            landmark = self.landmarks[idx]
            key_count = self.key_counts[idx]
            batch_size, num_heads, seq_len, head_dim = k.shape
            k_sum = k.sum(dim=(0, 2))
            new_count = key_count + batch_size * seq_len

            self.landmarks[idx] = (
                (landmark * key_count.unsqueeze(-1) + k_sum) / new_count.clamp(min=1).unsqueeze(-1)
            ).detach()
            self.key_counts[idx] = new_count.detach()

        self.current_memory_idx = torch.tensor((idx + 1) % self.num_memories, device=k.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.w_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        memory_output = self._retrieve_from_memory(q)
        local_output = causal_linear_attention(q, k, v)

        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        output = gate * memory_output + (1 - gate) * local_output

        if update_memory:
            self._update_memory(k, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.w_o(output)

    def get_memory_state(self) -> dict:
        return {
            "memories": [m.cpu().clone() for m in self.memories] if self.memories else None,
            "memory_norms": [n.cpu().clone() for n in self.memory_norms] if self.memory_norms else None,
            "landmarks": [lm.cpu().clone() for lm in self.landmarks] if self.landmarks else None,
            "key_counts": [c.cpu().clone() for c in self.key_counts] if self.key_counts else None,
            "current_memory_idx": self.current_memory_idx.cpu().clone(),
        }

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        if state["memories"]:
            self.memories = [m.to(device) for m in state["memories"]]
        if state["memory_norms"]:
            self.memory_norms = [n.to(device) for n in state["memory_norms"]]
        if state.get("landmarks"):
            self.landmarks = [lm.to(device) for lm in state["landmarks"]]
        if state.get("key_counts"):
            self.key_counts = [c.to(device) for c in state["key_counts"]]
        self.current_memory_idx = state["current_memory_idx"].to(device)


class MultiMemoryLayerComparable(BaseLayer):
    """比較実験用MultiMemoryLayer"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_memories: int = 4,
        use_delta_rule: bool = True,
        landmark_type: LandmarkType = "hsa",
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
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
        attn_output = self.attention(hidden_states, attention_mask, update_memory)
        mlp_output = self.mlp(hidden_states)
        return residual + attn_output + mlp_output

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        self.attention.reset_memory(device)

    def get_memory_state(self) -> dict:
        return self.attention.get_memory_state()

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        self.attention.set_memory_state(state, device)


# =============================================================================
# モデル作成
# =============================================================================

def create_model_with_landmark_type(
    config: PythiaConfig,
    landmark_type: LandmarkType,
    num_memories: int = 4,
) -> TransformerLM:
    """指定したLandmark方式でモデルを作成"""
    h = config.hidden_size
    n = config.num_attention_heads
    i = config.intermediate_size
    r = config.rotary_pct
    m = config.max_position_embeddings

    layers: list[BaseLayer] = [
        MultiMemoryLayerComparable(h, n, i, num_memories, True, landmark_type),
        *[PythiaLayer(h, n, i, r, m) for _ in range(config.num_layers - 1)]
    ]

    return TransformerLM(
        layers=layers,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
    )


# =============================================================================
# 実験実行
# =============================================================================

def run_experiment(
    landmark_type: LandmarkType,
    config: PythiaConfig,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    num_memories: int,
) -> dict:
    """1つのLandmark方式で実験を実行"""
    print_flush(f"\n{'=' * 60}")
    print_flush(f"Landmark Type: {landmark_type}")
    print_flush("=" * 60)

    # モデル作成
    model = create_model_with_landmark_type(config, landmark_type, num_memories)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    layer0_params = sum(p.numel() for p in model.layers[0].parameters())
    print_flush(f"Parameters: {params:,}")
    print_flush(f"Layer 0 params: {layer0_params:,}")

    # 訓練
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=epochs,
        learning_rate=lr,
        patience=patience,
        model_name=f"MultiMemory ({landmark_type})",
    )

    return {
        "landmark_type": landmark_type,
        "best_val_ppl": result["best_val_ppl"],
        "best_epoch": result["best_epoch"],
        "params": params,
        "layer0_params": layer0_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Landmark方式比較実験")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-memories", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    config = PythiaConfig()
    tokenizer = get_tokenizer()

    print_flush("=" * 60)
    print_flush("Landmark方式比較実験")
    print_flush("=" * 60)
    print_flush(f"Device: {device}")
    print_flush()
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Seq length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Memories: {args.num_memories}")
    print_flush()

    # データ準備
    print_flush("[Data] Loading Pile data...")
    documents = load_long_documents_from_pile(
        tokenizer=tokenizer,
        num_docs=args.samples,
        tokens_per_doc=args.seq_length,
    )

    # Train/Val split
    train_size = int(len(documents) * 0.9)
    train_docs = documents[:train_size]
    val_docs = documents[train_size:]

    # DataLoader作成
    train_data = torch.stack(train_docs)
    val_data = torch.stack(val_docs)

    train_dataset = TensorDataset(train_data, train_data)
    val_dataset = TensorDataset(val_data, val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print_flush(f"  Train: {len(train_docs)} samples")
    print_flush(f"  Val: {len(val_docs)} samples")

    # 両方式で実験
    results = []
    for landmark_type in ["hsa", "memory_norm"]:
        result = run_experiment(
            landmark_type=landmark_type,  # type: ignore
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            num_memories=args.num_memories,
        )
        results.append(result)

    # 結果表示
    print_flush("\n" + "=" * 60)
    print_flush("SUMMARY")
    print_flush("=" * 60)
    print_flush()
    print_flush("| Landmark Type | Best PPL | Epoch | Params |")
    print_flush("|---------------|----------|-------|--------|")
    for r in results:
        print_flush(
            f"| {r['landmark_type']:<13} | {r['best_val_ppl']:>8.1f} | "
            f"{r['best_epoch']:>5} | {r['params']:,} |"
        )

    # 比較
    hsa_ppl = results[0]["best_val_ppl"]
    norm_ppl = results[1]["best_val_ppl"]
    diff = norm_ppl - hsa_ppl
    extra_params = results[0]["layer0_params"] - results[1]["layer0_params"]

    print_flush()
    print_flush(f"PPL差 (memory_norm - hsa): {diff:+.1f}")
    if extra_params != 0:
        print_flush(f"パラメータ差: {extra_params:,}")
    print_flush()

    if diff > 0:
        print_flush("→ HSA方式が優れている")
    elif diff < 0:
        print_flush("→ memory_norm方式が優れている")
    else:
        print_flush("→ ほぼ同等")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
