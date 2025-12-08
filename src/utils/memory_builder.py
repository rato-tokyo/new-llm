"""
Memory Builder - テキストからメモリを事前構築するユーティリティ

使用例:
    from src.utils.memory_builder import MemoryBuilder
    from src.models import create_model

    # モデル作成
    model = create_model("multi_memory", num_memories=4)

    # メモリビルダー作成
    builder = MemoryBuilder(model, tokenizer)

    # 各メモリに異なるドメインのテキストを書き込み
    builder.build_memory(0, "科学論文のテキスト...")
    builder.build_memory(1, "ニュース記事のテキスト...")
    builder.build_memory(2, "小説のテキスト...")
    builder.build_memory(3, "コードドキュメントのテキスト...")

    # メモリ状態を保存
    builder.save("memories/domain_specific.pt")

    # 後で読み込み
    builder.load("memories/domain_specific.pt")
"""

from typing import Optional, Union
from pathlib import Path

import torch
import torch.nn as nn


class MemoryBuilder:
    """テキストからメモリを事前構築するユーティリティ

    MultiMemoryLayerを持つモデルに対して、
    各メモリに異なるテキストを書き込むことができる。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        seq_length: int = 256,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: MultiMemoryLayerを持つTransformerLM
            tokenizer: トークナイザー
            seq_length: チャンクサイズ
            device: デバイス（Noneならモデルのデバイスを使用）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.device = device or next(model.parameters()).device

        # MultiMemoryLayerを探す
        self.memory_layer = self._find_memory_layer()
        if self.memory_layer is None:
            raise ValueError("Model does not have MultiMemoryLayer")

        self.num_memories = self.memory_layer.attention.num_memories

    def _find_memory_layer(self):
        """モデルからMultiMemoryLayerを探す"""
        from src.models.layers import MultiMemoryLayer

        for layer in self.model.layers:
            if isinstance(layer, MultiMemoryLayer):
                return layer
        return None

    def reset_all_memories(self) -> None:
        """全メモリをリセット"""
        self.memory_layer.reset_memory(self.device)

    def build_memory(
        self,
        memory_idx: int,
        text: str,
        batch_size: int = 8,
    ) -> dict:
        """特定のメモリにテキストを書き込む

        Args:
            memory_idx: 書き込むメモリのインデックス (0 ~ num_memories-1)
            text: 書き込むテキスト
            batch_size: バッチサイズ

        Returns:
            統計情報（トークン数、チャンク数など）
        """
        if memory_idx < 0 or memory_idx >= self.num_memories:
            raise ValueError(
                f"memory_idx must be 0 ~ {self.num_memories - 1}, got {memory_idx}"
            )

        # テキストをトークン化
        tokens = self.tokenizer.encode(text)
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)

        # チャンクに分割
        num_chunks = len(tokens) // self.seq_length
        if num_chunks == 0:
            # 短いテキストの場合、パディング
            padded = torch.full((self.seq_length,), self.tokenizer.eos_token_id)
            padded[:len(tokens)] = tokens
            chunks = [padded]
            num_chunks = 1
        else:
            chunks = [
                tokens[i * self.seq_length : (i + 1) * self.seq_length]
                for i in range(num_chunks)
            ]

        # メモリ状態を初期化（未初期化の場合）
        if self.memory_layer.attention.memories is None:
            self.memory_layer.reset_memory(self.device)

        # 書き込み先メモリインデックスを設定
        attn = self.memory_layer.attention
        attn.current_memory_idx = torch.tensor(memory_idx, device=self.device)

        # バッチ処理でメモリに書き込み
        self.model.eval()
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch = torch.stack(batch_chunks).to(self.device)

                # Embeddingを通して隠れ状態を取得
                hidden_states = self.model.embed_in(batch)

                # MultiMemoryLayerのforwardを呼び出し（メモリ更新のみ）
                # 注: update_memory=Trueで書き込み
                _ = self.memory_layer(hidden_states, update_memory=True)

                total_tokens += batch.numel()

        # 次のメモリに切り替え（ラウンドロビン）
        attn.current_memory_idx = torch.tensor(
            (memory_idx + 1) % self.num_memories, device=self.device
        )

        return {
            "memory_idx": memory_idx,
            "num_tokens": total_tokens,
            "num_chunks": len(chunks),
            "landmark": attn.landmarks[memory_idx].cpu().clone() if attn.landmarks else None,
            "key_count": attn.key_counts[memory_idx].cpu().clone() if attn.key_counts else None,
        }

    def build_memories_from_texts(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[dict]:
        """複数のテキストから複数のメモリを構築

        Args:
            texts: テキストのリスト（長さはnum_memories以下）
            batch_size: バッチサイズ

        Returns:
            各メモリの統計情報のリスト
        """
        if len(texts) > self.num_memories:
            raise ValueError(
                f"Too many texts ({len(texts)}) for {self.num_memories} memories"
            )

        self.reset_all_memories()

        stats = []
        for i, text in enumerate(texts):
            stat = self.build_memory(i, text, batch_size)
            stats.append(stat)

        return stats

    def get_memory_state(self) -> dict:
        """現在のメモリ状態を取得"""
        return self.memory_layer.get_memory_state()

    def set_memory_state(self, state: dict) -> None:
        """メモリ状態を設定"""
        self.memory_layer.set_memory_state(state, self.device)

    def save(self, path: Union[str, Path]) -> None:
        """メモリ状態をファイルに保存"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_memory_state(), path)

    def load(self, path: Union[str, Path]) -> None:
        """メモリ状態をファイルから読み込み"""
        state = torch.load(path, map_location=self.device)
        self.set_memory_state(state)

    def get_memory_info(self) -> dict:
        """各メモリの情報を取得"""
        attn = self.memory_layer.attention
        if attn.memories is None:
            return {"initialized": False}

        info = {
            "initialized": True,
            "num_memories": self.num_memories,
            "memories": [],
        }

        for i in range(self.num_memories):
            mem_info = {
                "idx": i,
                "key_count": float(attn.key_counts[i].sum().item()) if attn.key_counts else 0,
                "landmark_norm": float(attn.landmarks[i].norm().item()) if attn.landmarks else 0,
                "memory_norm": float(attn.memory_norms[i].norm().item()) if attn.memory_norms else 0,
            }
            info["memories"].append(mem_info)

        return info

    def print_memory_info(self) -> None:
        """メモリ情報を表示"""
        info = self.get_memory_info()

        if not info["initialized"]:
            print("Memory not initialized")
            return

        print(f"Memories: {info['num_memories']}")
        print("-" * 50)
        print(f"{'Idx':<5} {'Keys':<15} {'Landmark':<15} {'Memory':<15}")
        print("-" * 50)

        for mem in info["memories"]:
            print(
                f"{mem['idx']:<5} "
                f"{mem['key_count']:<15.1f} "
                f"{mem['landmark_norm']:<15.4f} "
                f"{mem['memory_norm']:<15.4f}"
            )


def create_domain_memories(
    model: nn.Module,
    tokenizer,
    domain_texts: dict[str, str],
    seq_length: int = 256,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> MemoryBuilder:
    """異なるドメインのテキストからメモリを構築するヘルパー関数

    Args:
        model: MultiMemoryLayerを持つモデル
        tokenizer: トークナイザー
        domain_texts: ドメイン名 → テキストの辞書
        seq_length: チャンクサイズ
        batch_size: バッチサイズ
        device: デバイス

    Returns:
        構築済みのMemoryBuilder

    Example:
        domain_texts = {
            "science": "量子力学は...",
            "news": "本日の株価は...",
            "fiction": "昔々あるところに...",
            "code": "def hello_world():...",
        }
        builder = create_domain_memories(model, tokenizer, domain_texts)
    """
    builder = MemoryBuilder(model, tokenizer, seq_length, device)
    texts = list(domain_texts.values())
    builder.build_memories_from_texts(texts, batch_size)
    return builder
