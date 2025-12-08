"""
Memory Builder - テキストからメモリを事前構築するユーティリティ

2つの方式を提供:

1. MemoryBuilder: モデルを通してメモリを構築（従来方式）
2. DirectMemoryBuilder: テキストから直接メモリを構築（モデル非依存）

使用例（DirectMemoryBuilder - 推奨）:
    from src.utils.memory_builder import DirectMemoryBuilder

    # 直接メモリ構築（モデル不要）
    builder = DirectMemoryBuilder(
        num_memories=4,
        num_heads=8,
        head_dim=64,
    )

    # 各メモリに異なるテキストを書き込み
    builder.build_from_texts([
        "Physics quantum mechanics relativity thermodynamics",
        "History war revolution renaissance industrial",
        "Technology machine learning programming python",
        "Geography mountain river ocean desert climate",
    ], tokenizer)

    # メモリ状態をモデルに設定
    builder.apply_to_model(model)

使用例（MemoryBuilder - 従来方式）:
    from src.utils.memory_builder import MemoryBuilder
    from src.models import create_model

    model = create_model("multi_memory", num_memories=4)
    builder = MemoryBuilder(model, tokenizer)
    builder.build_memory(0, "科学論文のテキスト...")
"""

from typing import Optional, Union
from pathlib import Path

import torch
import torch.nn as nn

from src.models.memory_utils import elu_plus_one


class DirectMemoryBuilder:
    """テキストから直接圧縮メモリを構築（モデル非依存）

    トークンの埋め込みベクトルを使って、モデルの重みに依存せず
    メモリ（K-V行列）とkey_sequences（ChunkEncoderの入力）を構築する。

    HSA方式では Landmark = ChunkEncoder([CLS] + Keys)[CLS] として計算されるため、
    ここではキー列 (key_sequences) を保持する。

    これにより：
    - ランダム初期化モデルでも意味的に異なるメモリを作成可能
    - 検証に必要な最低限の内容で構築可能
    - 完全に制御されたメモリ内容
    """

    def __init__(
        self,
        num_memories: int = 4,
        num_heads: int = 8,
        head_dim: int = 64,
        max_keys_per_memory: int = 256,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            num_memories: メモリ数
            num_heads: ヘッド数
            head_dim: ヘッド次元（hidden_size // num_heads）
            max_keys_per_memory: メモリあたりの最大キー数
            device: デバイス
            dtype: データ型
        """
        self.num_memories = num_memories
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_keys_per_memory = max_keys_per_memory
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # メモリ状態を初期化
        self._init_memory()

    def _init_memory(self) -> None:
        """メモリを初期化"""
        self.memories = [
            torch.zeros(self.num_heads, self.head_dim, self.head_dim,
                        device=self.device, dtype=self.dtype)
            for _ in range(self.num_memories)
        ]
        self.memory_norms = [
            torch.zeros(self.num_heads, self.head_dim,
                        device=self.device, dtype=self.dtype)
            for _ in range(self.num_memories)
        ]
        # HSA: キー列を保持（ChunkEncoderの入力）
        # key_sequences[memory_idx][head_idx] = (seq_len, head_dim)
        self.key_sequences: list[list[torch.Tensor]] = [
            [torch.zeros(0, self.head_dim, device=self.device, dtype=self.dtype)
             for _ in range(self.num_heads)]
            for _ in range(self.num_memories)
        ]

    def build_memory_from_embeddings(
        self,
        memory_idx: int,
        embeddings: torch.Tensor,
    ) -> dict:
        """埋め込みベクトルからメモリを構築

        Args:
            memory_idx: メモリインデックス
            embeddings: (seq_len, hidden_size) の埋め込みベクトル

        Returns:
            統計情報
        """
        if memory_idx < 0 or memory_idx >= self.num_memories:
            raise ValueError(f"memory_idx must be 0 ~ {self.num_memories - 1}")

        seq_len, hidden_size = embeddings.shape
        expected_hidden = self.num_heads * self.head_dim
        if hidden_size != expected_hidden:
            raise ValueError(
                f"embeddings hidden_size ({hidden_size}) != "
                f"num_heads * head_dim ({expected_hidden})"
            )

        # (seq_len, hidden) -> (seq_len, num_heads, head_dim)
        embeddings = embeddings.to(self.device, self.dtype)
        kv = embeddings.view(seq_len, self.num_heads, self.head_dim)

        # K = V = embeddings（簡略化）
        k = kv  # (seq_len, num_heads, head_dim)
        v = kv

        # σ(K) for Linear Attention
        sigma_k = elu_plus_one(k)  # (seq_len, num_heads, head_dim)

        # メモリ更新: M += σ(K)^T @ V
        # (num_heads, head_dim, head_dim)
        memory_update = torch.einsum('shd,she->hde', sigma_k, v)
        self.memories[memory_idx] = self.memories[memory_idx] + memory_update

        # memory_norm更新: z += Σ σ(k)
        norm_update = sigma_k.sum(dim=0)  # (num_heads, head_dim)
        self.memory_norms[memory_idx] = self.memory_norms[memory_idx] + norm_update

        # HSA: キー列を更新（ChunkEncoderの入力）
        for h in range(self.num_heads):
            new_keys = k[:, h, :]  # (seq_len, head_dim)
            current_keys = self.key_sequences[memory_idx][h]
            combined = torch.cat([current_keys, new_keys], dim=0)

            # max_keys_per_memory を超えたら古いキーを削除
            if combined.size(0) > self.max_keys_per_memory:
                combined = combined[-self.max_keys_per_memory:]

            self.key_sequences[memory_idx][h] = combined

        total_keys = sum(
            self.key_sequences[memory_idx][h].size(0)
            for h in range(self.num_heads)
        )

        return {
            "memory_idx": memory_idx,
            "seq_len": seq_len,
            "total_keys": total_keys,
        }

    def build_from_texts(
        self,
        texts: list[str],
        tokenizer,
        embedding_layer: Optional[nn.Embedding] = None,
    ) -> list[dict]:
        """テキストリストからメモリを構築

        Args:
            texts: テキストリスト（長さ <= num_memories）
            tokenizer: トークナイザー
            embedding_layer: 埋め込み層（Noneならランダム埋め込み使用）

        Returns:
            各メモリの統計情報
        """
        if len(texts) > self.num_memories:
            raise ValueError(f"Too many texts ({len(texts)}) for {self.num_memories} memories")

        # 埋め込み層がない場合、ランダム埋め込みを作成
        if embedding_layer is None:
            vocab_size = tokenizer.vocab_size
            hidden_size = self.num_heads * self.head_dim
            embedding_layer = nn.Embedding(vocab_size, hidden_size)
            embedding_layer = embedding_layer.to(self.device, self.dtype)

        stats = []
        for i, text in enumerate(texts):
            # トークン化
            tokens = tokenizer.encode(text)
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.to(self.device)

            # 埋め込み取得
            with torch.no_grad():
                embeddings = embedding_layer(tokens)  # (seq_len, hidden_size)

            # メモリ構築
            stat = self.build_memory_from_embeddings(i, embeddings)
            stat["text_preview"] = text[:50] + "..." if len(text) > 50 else text
            stats.append(stat)

        return stats

    def get_memory_state(self) -> dict:
        """メモリ状態を取得（モデルに設定可能な形式）"""
        return {
            "memories": [m.cpu().clone() for m in self.memories],
            "memory_norms": [n.cpu().clone() for n in self.memory_norms],
            "key_sequences": [
                [ks.cpu().clone() for ks in mem_keys]
                for mem_keys in self.key_sequences
            ],
            "current_memory_idx": torch.tensor(0),
        }

    def apply_to_model(self, model: nn.Module) -> None:
        """構築したメモリをモデルに適用

        Args:
            model: MultiMemoryLayerを持つモデル
        """
        from src.models.layers import MultiMemoryLayer

        for layer in model.layers:
            if isinstance(layer, MultiMemoryLayer):
                device = next(model.parameters()).device
                layer.set_memory_state(self.get_memory_state(), device)
                return

        raise ValueError("Model does not have MultiMemoryLayer")

    def build_orthogonal_memories(self, num_keys_per_memory: int = 100) -> None:
        """直交するキー列を持つメモリを構築（検証用）

        各メモリが異なる方向を向くキー列を持つように構築。
        ChunkEncoderが異なるLandmarkを生成することを確認できる。

        Args:
            num_keys_per_memory: 各メモリに格納するキー数
        """
        hidden_size = self.num_heads * self.head_dim

        # 直交基底を生成（QR分解）
        random_matrix = torch.randn(hidden_size, self.num_memories, device=self.device, dtype=self.dtype)
        q, _ = torch.linalg.qr(random_matrix)
        orthogonal_directions = q.T  # (num_memories, hidden_size)

        for i in range(self.num_memories):
            # 各メモリの方向ベクトル
            direction = orthogonal_directions[i]  # (hidden_size,)

            # その方向に沿ったキーを生成
            # keys: (num_keys, hidden_size)
            noise = torch.randn(num_keys_per_memory, hidden_size, device=self.device, dtype=self.dtype) * 0.1
            keys = direction.unsqueeze(0) + noise  # 方向+ノイズ

            # (num_keys, num_heads, head_dim)
            k = keys.view(num_keys_per_memory, self.num_heads, self.head_dim)
            v = k  # K = V

            sigma_k = elu_plus_one(k)

            # メモリ更新
            self.memories[i] = torch.einsum('shd,she->hde', sigma_k, v)
            self.memory_norms[i] = sigma_k.sum(dim=0)

            # HSA: キー列を保存（ChunkEncoderの入力）
            for h in range(self.num_heads):
                self.key_sequences[i][h] = k[:, h, :]  # (num_keys, head_dim)

    def get_landmark_similarity_matrix(self) -> torch.Tensor:
        """Landmark間のコサイン類似度行列を計算

        Returns:
            (num_memories, num_memories) の類似度行列
        """
        # (num_memories, num_heads * head_dim)
        landmarks_flat = torch.stack([
            lm.flatten() for lm in self.landmarks
        ])

        # 正規化
        landmarks_norm = landmarks_flat / landmarks_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # コサイン類似度
        return landmarks_norm @ landmarks_norm.T

    def print_info(self) -> None:
        """メモリ情報を表示"""
        print(f"DirectMemoryBuilder: {self.num_memories} memories")
        print(f"  num_heads={self.num_heads}, head_dim={self.head_dim}")
        print("-" * 50)
        print(f"{'Idx':<5} {'Keys':<15} {'Landmark':<15} {'Memory':<15}")
        print("-" * 50)

        for i in range(self.num_memories):
            key_count = float(self.key_counts[i].sum().item())
            landmark_norm = float(self.landmarks[i].norm().item())
            memory_norm = float(self.memory_norms[i].norm().item())
            print(f"{i:<5} {key_count:<15.1f} {landmark_norm:<15.4f} {memory_norm:<15.4f}")

        # Landmark類似度を表示
        sim_matrix = self.get_landmark_similarity_matrix()
        print("\nLandmark Similarity Matrix:")
        print("     " + "".join(f"{i:>8}" for i in range(self.num_memories)))
        for i in range(self.num_memories):
            row = "".join(f"{sim_matrix[i, j].item():>8.3f}" for j in range(self.num_memories))
            print(f"{i:>4} {row}")


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
