"""
ストリーミングデータローダー

UltraChatデータセットをストリーミングでダウンロードし、
メモリ効率的にトークン化してディスクに保存。
"""

import os
import json
import torch
import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Model

from src.utils.disk_offload import TokenIDCache, EmbeddingCache


def print_flush(msg: str):
    """即時フラッシュ付きprint。"""
    print(msg, flush=True)


class StreamingDataLoader:
    """
    UltraChatデータセットをストリーミングで処理。

    メモリに全データを載せずに、チャンク単位で処理してディスクに保存。
    """

    def __init__(
        self,
        output_dir: str,
        num_samples: int = 200_000,
        max_seq_length: int = 128,
        use_bf16: bool = True,
        chunk_size: int = 10_000  # ストリーミング処理のチャンクサイズ
    ):
        """
        Args:
            output_dir: 出力ディレクトリ（NVMeマウントポイント）
            num_samples: 処理するサンプル数
            max_seq_length: 最大シーケンス長
            use_bf16: bf16精度を使用するか
            chunk_size: 一度に処理するサンプル数
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        self.use_bf16 = use_bf16
        self.chunk_size = chunk_size

        self.metadata_path = os.path.join(output_dir, "metadata.json")
        self._metadata: Optional[Dict[str, Any]] = None

    def is_prepared(self) -> bool:
        """データが準備済みかどうか。"""
        return os.path.exists(self.metadata_path)

    def load_metadata(self) -> Dict[str, Any]:
        """メタデータを読み込み。"""
        if self._metadata is None:
            with open(self.metadata_path, 'r') as f:
                self._metadata = json.load(f)
        return self._metadata

    def get_num_tokens(self) -> int:
        """総トークン数を取得。"""
        return self.load_metadata()['num_tokens']

    def prepare(self, device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
        """
        UltraChatデータセットを準備。

        1. ストリーミングでダウンロード
        2. トークン化
        3. GPT-2埋め込みを計算
        4. ディスクに保存

        Args:
            device: 埋め込み計算に使用するデバイス

        Returns:
            メタデータ
        """
        print_flush(f"\n{'='*70}")
        print_flush("データ準備: UltraChat 200k")
        print_flush(f"{'='*70}")
        print_flush(f"  出力先: {self.output_dir}")
        print_flush(f"  サンプル数: {self.num_samples:,}")
        print_flush(f"  最大シーケンス長: {self.max_seq_length}")
        print_flush(f"  精度: {'bf16' if self.use_bf16 else 'float32'}")
        print_flush("")

        os.makedirs(self.output_dir, exist_ok=True)

        # トークナイザーをロード
        print_flush("トークナイザーをロード中...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # GPT-2埋め込み層をロード
        print_flush("GPT-2埋め込み層をロード中...")
        gpt2 = GPT2Model.from_pretrained("gpt2")
        embedding_layer = gpt2.wte
        embed_norm = gpt2.ln_f if hasattr(gpt2, 'ln_f') else None

        # bf16に変換
        if self.use_bf16:
            embedding_layer = embedding_layer.to(torch.bfloat16)
            if embed_norm is not None:
                embed_norm = embed_norm.to(torch.bfloat16)

        embedding_layer = embedding_layer.to(device)
        if embed_norm is not None:
            embed_norm = embed_norm.to(device)

        embedding_layer.eval()

        # Phase 1: トークン数をカウント
        print_flush("\nPhase 1: トークン数をカウント中...")
        total_tokens = self._count_tokens(tokenizer)
        print_flush(f"  総トークン数: {total_tokens:,}")

        # キャッシュを作成
        token_cache = TokenIDCache(self.output_dir, total_tokens)
        embed_cache = EmbeddingCache(
            self.output_dir, total_tokens, 768, self.use_bf16
        )

        token_cache.create()
        embed_cache.create()

        # Phase 2: トークン化と埋め込み計算
        print_flush("\nPhase 2: トークン化と埋め込み計算中...")
        token_cache.open('r+')
        embed_cache.open('r+')

        try:
            self._process_and_save(
                tokenizer, embedding_layer, embed_norm,
                token_cache, embed_cache, device
            )
        finally:
            token_cache.close()
            embed_cache.close()

        # メタデータを保存
        self._metadata = {
            'num_samples': self.num_samples,
            'num_tokens': total_tokens,
            'max_seq_length': self.max_seq_length,
            'use_bf16': self.use_bf16,
            'context_dim': 768,
            'embed_dim': 768,
            'vocab_size': tokenizer.vocab_size
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(self._metadata, f, indent=2)

        print_flush(f"\n{'='*70}")
        print_flush("データ準備完了")
        print_flush(f"{'='*70}")
        print_flush(f"  メタデータ: {self.metadata_path}")
        print_flush(f"  トークンID: {token_cache.file_path}")
        print_flush(f"  埋め込み: {embed_cache.file_path}")

        return self._metadata

    def _count_tokens(self, tokenizer) -> int:
        """トークン数をカウント（ストリーミング）。"""
        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            streaming=True
        )

        total_tokens = 0
        sample_count = 0

        for sample in tqdm(dataset, total=self.num_samples, desc="カウント中"):
            if sample_count >= self.num_samples:
                break

            messages = sample.get("messages", [])
            text = "\n".join([msg.get("content", "") for msg in messages])

            tokens = tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )

            # パディングを除いた実際のトークン数
            token_ids = tokens["input_ids"].squeeze(0)
            actual_length = (token_ids != tokenizer.pad_token_id).sum().item()
            # truncationの場合は全部有効
            actual_length = min(actual_length, self.max_seq_length)
            if actual_length == 0:
                actual_length = len(token_ids)

            total_tokens += actual_length
            sample_count += 1

        return total_tokens

    def _process_and_save(
        self,
        tokenizer,
        embedding_layer,
        embed_norm,
        token_cache: TokenIDCache,
        embed_cache: EmbeddingCache,
        device: torch.device
    ):
        """トークン化と埋め込み計算を実行し、ディスクに保存。"""
        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            streaming=True
        )

        write_idx = 0
        sample_count = 0

        # チャンク単位でバッファリング
        token_buffer = []
        embed_buffer = []

        with torch.no_grad():
            for sample in tqdm(dataset, total=self.num_samples, desc="処理中"):
                if sample_count >= self.num_samples:
                    break

                messages = sample.get("messages", [])
                text = "\n".join([msg.get("content", "") for msg in messages])

                tokens = tokenizer(
                    text,
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt"
                )

                token_ids = tokens["input_ids"].squeeze(0)

                # パディングを除去
                mask = token_ids != tokenizer.pad_token_id
                if mask.sum() == 0:
                    # 全部パディングの場合はスキップしない（少なくとも1トークン）
                    token_ids = token_ids[:1]
                else:
                    token_ids = token_ids[mask]

                # 埋め込み計算
                token_ids_device = token_ids.to(device)
                embeds = embedding_layer(token_ids_device)

                # LayerNorm適用（GPT-2の最終層の正規化）
                if embed_norm is not None:
                    embeds = embed_norm(embeds)

                # バッファに追加
                token_buffer.append(token_ids.cpu())
                embed_buffer.append(embeds.cpu())

                sample_count += 1

                # チャンクごとにディスクに書き込み
                if sample_count % self.chunk_size == 0:
                    self._flush_buffers(
                        token_buffer, embed_buffer,
                        token_cache, embed_cache,
                        write_idx
                    )
                    write_idx += sum(len(t) for t in token_buffer)
                    token_buffer = []
                    embed_buffer = []

            # 残りのバッファを書き込み
            if token_buffer:
                self._flush_buffers(
                    token_buffer, embed_buffer,
                    token_cache, embed_cache,
                    write_idx
                )

        token_cache.flush()
        embed_cache.flush()

    def _flush_buffers(
        self,
        token_buffer,
        embed_buffer,
        token_cache: TokenIDCache,
        embed_cache: EmbeddingCache,
        start_idx: int
    ):
        """バッファをディスクに書き込み。"""
        if not token_buffer:
            return

        # 連結
        all_tokens = torch.cat(token_buffer, dim=0)
        all_embeds = torch.cat(embed_buffer, dim=0)

        # 書き込み
        token_cache.set_chunk(start_idx, all_tokens)
        embed_cache.set_chunk(start_idx, all_embeds)
