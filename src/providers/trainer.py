"""
Phase1Trainer - CVFP固定点学習トレーナーの統一インターフェース

memory: 全データをメモリに展開して処理
storage: mmapでディスクから直接処理（大規模データ用）
"""

import os
import sys
import gc
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np

from src.utils.disk_offload import ContextSwapper, EmbeddingCache


def print_flush(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


class Phase1Trainer(ABC):
    """Phase 1トレーナーの抽象基底クラス"""

    def __init__(self, model: torch.nn.Module, config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self._training_stats: Dict[str, Any] = {}

    @abstractmethod
    def train(self, token_ids: torch.Tensor, label: str = "Train") -> torch.Tensor:
        pass

    @abstractmethod
    def evaluate(self, token_ids: torch.Tensor, label: str = "Val") -> torch.Tensor:
        pass

    def get_training_stats(self) -> Dict[str, Any]:
        return self._training_stats

    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        pass

    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': 'phase1_complete',
            'config': {
                'num_layers': self.config.num_layers,
                'embed_dim': self.config.embed_dim,
                'context_dim': self.config.context_dim,
                'vocab_size': self.config.vocab_size,
            },
            'training_stats': self._training_stats,
        }
        torch.save(checkpoint, path)
        print_flush(f"✓ Checkpoint saved: {path}")


class MemoryPhase1Trainer(Phase1Trainer):
    """メモリ展開型Phase 1トレーナー"""

    def train(self, token_ids: torch.Tensor, label: str = "Train") -> torch.Tensor:
        self.model.train()
        num_tokens = len(token_ids)

        print_flush(f"\n{'='*70}")
        print_flush(f"PHASE 1: 固定点コンテキスト学習 - {label}")
        print_flush(f"{'='*70}")
        print_flush(f"  Mode: Memory")
        print_flush(f"  Tokens: {num_tokens:,}")
        print_flush(f"  Max iterations: {self.config.phase1_max_iterations}")
        print_flush(f"  Learning rate: {self.config.phase1_learning_rate}")
        print_flush(f"  Diversity weight: {self.config.dist_reg_weight}")

        # ContextBlockのパラメータのみ学習
        context_params = list(self.model.context_block.parameters())
        print_flush(f"  ContextBlock params: {sum(p.numel() for p in context_params):,}")
        optimizer = torch.optim.Adam(context_params, lr=self.config.phase1_learning_rate)

        # トークン埋め込み（1回のみ計算）
        with torch.no_grad():
            token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)

        previous_contexts = None
        final_convergence_rate = 0.0

        for iteration in range(self.config.phase1_max_iterations):
            start_time = time.time()

            if iteration == 0:
                # Iteration 0: シーケンシャル（学習なし）
                contexts = self._forward_sequential(token_embeds, None)
                previous_contexts = contexts.detach()
                elapsed = time.time() - start_time
                print_flush(f"Iteration 1/{self.config.phase1_max_iterations}: シーケンシャル [{elapsed:.2f}s]")
                continue

            # Iteration 1+: 並列処理
            contexts = self._forward_parallel(token_embeds, previous_contexts)

            # 損失計算
            cvfp_loss = F.mse_loss(contexts, previous_contexts)
            diversity_loss = self._compute_diversity_loss(contexts)
            total_loss = (1 - self.config.dist_reg_weight) * cvfp_loss + self.config.dist_reg_weight * diversity_loss

            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            # 収束率
            convergence_rate = self._compute_convergence_rate(contexts, previous_contexts, num_tokens)
            previous_contexts = contexts.detach()
            final_convergence_rate = convergence_rate

            elapsed = time.time() - start_time
            print_flush(
                f"Iteration {iteration+1}/{self.config.phase1_max_iterations}: "
                f"収束={convergence_rate*100:.1f}% | "
                f"Loss={total_loss.item():.6f} | "
                f"CVFP={cvfp_loss.item():.6f} | "
                f"Div={diversity_loss.item():.6f} [{elapsed:.2f}s]"
            )

            if convergence_rate >= self.config.phase1_min_converged_ratio:
                print_flush(f"  → Early stopping")
                break

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        self._training_stats = {
            'iterations': iteration + 1,
            'convergence_rate': final_convergence_rate,
            'num_tokens': num_tokens,
        }

        print_flush(f"\nPhase 1 完了: {int(final_convergence_rate * num_tokens)}/{num_tokens} トークン収束\n")
        return contexts.detach()

    def evaluate(self, token_ids: torch.Tensor, label: str = "Val") -> torch.Tensor:
        self.model.eval()
        print_flush(f"\nEvaluating {label} data...")

        with torch.no_grad():
            token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)
            contexts = self._forward_sequential(token_embeds, None)

        self.model.train()
        return contexts

    def _forward_sequential(self, token_embeds: torch.Tensor, previous_contexts: Optional[torch.Tensor]) -> torch.Tensor:
        if previous_contexts is None:
            context = torch.zeros(1, self.model.context_dim, device=self.device)
        else:
            context = previous_contexts[-1].unsqueeze(0).detach()

        context_list = []
        for token_embed in token_embeds:
            context = self.model.context_block(context, token_embed.unsqueeze(0))
            context_list.append(context.squeeze(0))

        return torch.stack(context_list)

    def _forward_parallel(self, token_embeds: torch.Tensor, previous_contexts: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        num_tokens = len(token_embeds)

        contexts_for_batch = torch.zeros(num_tokens, self.model.context_dim, device=self.device)
        contexts_for_batch[1:] = previous_contexts[:-1].detach()
        contexts_for_batch[0] = previous_contexts[-1].detach()

        if self.config.phase1_context_noise > 0 and self.model.training:
            noise = torch.randn_like(contexts_for_batch) * self.config.phase1_context_noise
            contexts_for_batch = contexts_for_batch + noise

        if num_tokens > batch_size:
            all_contexts = []
            for start_idx in range(0, num_tokens, batch_size):
                end_idx = min(start_idx + batch_size, num_tokens)
                batch_output = self.model.context_block(
                    contexts_for_batch[start_idx:end_idx],
                    token_embeds[start_idx:end_idx]
                )
                all_contexts.append(batch_output)
            return torch.cat(all_contexts, dim=0)
        else:
            return self.model.context_block(contexts_for_batch, token_embeds)

    def _compute_diversity_loss(self, contexts: torch.Tensor) -> torch.Tensor:
        context_mean = contexts.mean(dim=0)
        deviation = contexts - context_mean
        return -torch.norm(deviation, p=2) / len(contexts)

    def _compute_convergence_rate(self, current: torch.Tensor, previous: torch.Tensor, num_tokens: int) -> float:
        with torch.no_grad():
            token_losses = ((current - previous) ** 2).mean(dim=1)
            converged = token_losses < self.config.phase1_convergence_threshold
            return converged.sum().item() / num_tokens

    @property
    def is_streaming(self) -> bool:
        return False


class StoragePhase1Trainer(Phase1Trainer):
    """ストレージベース型Phase 1トレーナー"""

    def __init__(self, model: torch.nn.Module, config, device: torch.device):
        super().__init__(model, config, device)
        self.storage_dir = config.disk_offload_dir
        self.chunk_size = getattr(config, 'disk_offload_chunk_size', 1_000_000)
        self.use_bf16 = getattr(config, 'use_bf16', True)
        self.torch_dtype = torch.bfloat16 if self.use_bf16 else torch.float32

    def train(self, token_ids: torch.Tensor, label: str = "Train") -> torch.Tensor:
        num_tokens = len(token_ids)

        print_flush(f"\n{'='*70}")
        print_flush(f"PHASE 1: 固定点コンテキスト学習 (Storage) - {label}")
        print_flush(f"{'='*70}")
        print_flush(f"  Mode: Storage (Disk Offload)")
        print_flush(f"  Tokens: {num_tokens:,}")
        print_flush(f"  Chunk size: {self.chunk_size:,}")
        print_flush(f"  Precision: {'bf16' if self.use_bf16 else 'float32'}")

        self.model.to(self.device)
        if self.use_bf16:
            self.model.to(self.torch_dtype)
        self.model.train()

        # Optimizer
        context_params = list(self.model.context_block.parameters())
        optimizer = torch.optim.Adam(context_params, lr=self.config.phase1_learning_rate)

        # ストレージ初期化
        context_swapper = ContextSwapper(
            self.storage_dir, num_tokens, self.config.context_dim, self.use_bf16
        )
        embed_cache = EmbeddingCache(
            self.storage_dir, num_tokens, self.config.embed_dim, self.use_bf16
        )

        # 埋め込みをキャッシュ
        self._prepare_embeddings(token_ids, embed_cache)

        context_swapper.create_storage()
        context_swapper.open('r+')
        embed_cache.open('r')

        final_convergence_rate = 0.0

        try:
            for iteration in range(self.config.phase1_max_iterations):
                start_time = time.time()

                if iteration == 0:
                    self._process_sequential(num_tokens, context_swapper, embed_cache)
                else:
                    convergence_rate = self._process_parallel(
                        num_tokens, context_swapper, embed_cache, optimizer
                    )
                    final_convergence_rate = convergence_rate

                elapsed = time.time() - start_time
                if iteration == 0:
                    print_flush(f"Iteration 1: シーケンシャル [{elapsed:.1f}s]")
                else:
                    print_flush(f"Iteration {iteration+1}: 収束={final_convergence_rate*100:.1f}% [{elapsed:.1f}s]")

                context_swapper.swap()
                context_swapper.flush()
                gc.collect()

        finally:
            context_swapper.close()
            embed_cache.close()

        self._training_stats = {
            'iterations': iteration + 1,
            'convergence_rate': final_convergence_rate,
            'num_tokens': num_tokens,
        }

        # 最終コンテキストを返す
        context_swapper.open('r')
        contexts = context_swapper.get_chunk(0, num_tokens, from_previous=False, device=self.device)
        context_swapper.close()

        print_flush(f"\nPhase 1 完了\n")
        return contexts

    def evaluate(self, token_ids: torch.Tensor, label: str = "Val") -> torch.Tensor:
        self.model.eval()
        print_flush(f"\nEvaluating {label} data...")

        with torch.no_grad():
            token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)
            if self.use_bf16:
                token_embeds = token_embeds.to(self.torch_dtype)

            context = torch.zeros(1, self.model.context_dim, device=self.device, dtype=self.torch_dtype)
            context_list = []
            for token_embed in token_embeds:
                context = self.model.context_block(context, token_embed.unsqueeze(0))
                context_list.append(context.squeeze(0))

        self.model.train()
        return torch.stack(context_list)

    def _prepare_embeddings(self, token_ids: torch.Tensor, embed_cache: EmbeddingCache):
        if embed_cache.exists():
            print_flush("  Embedding cache exists")
            return

        print_flush("  Preparing embeddings...")
        embed_cache.create()
        embed_cache.open('r+')

        num_tokens = len(token_ids)
        with torch.no_grad():
            for start in range(0, num_tokens, self.chunk_size):
                end = min(start + self.chunk_size, num_tokens)
                chunk_ids = token_ids[start:end].to(self.device)

                embeds = self.model.token_embedding(chunk_ids.unsqueeze(0))
                embeds = self.model.embed_norm(embeds).squeeze(0)
                if self.use_bf16:
                    embeds = embeds.to(torch.float16)

                embed_cache.set_chunk(start, embeds.cpu())

        embed_cache.flush()
        embed_cache.close()

    def _process_sequential(self, num_tokens: int, context_swapper: ContextSwapper, embed_cache: EmbeddingCache):
        self.model.eval()
        context = torch.zeros(1, self.config.context_dim, dtype=self.torch_dtype, device=self.device)

        with torch.no_grad():
            for start in range(0, num_tokens, self.chunk_size):
                end = min(start + self.chunk_size, num_tokens)
                embeds = embed_cache.get_chunk(start, end, self.device)

                chunk_contexts = []
                for t in range(end - start):
                    context = self.model.context_block(context, embeds[t:t+1])
                    chunk_contexts.append(context.squeeze(0))

                context_swapper.set_chunk(start, torch.stack(chunk_contexts))
                gc.collect()

        self.model.train()

    def _process_parallel(self, num_tokens: int, context_swapper: ContextSwapper, embed_cache: EmbeddingCache, optimizer) -> float:
        self.model.train()
        num_converged = 0

        last_context = context_swapper.get_chunk(num_tokens - 1, num_tokens, from_previous=True, device=self.device)

        for start in range(0, num_tokens, self.chunk_size):
            end = min(start + self.chunk_size, num_tokens)
            chunk_len = end - start

            embeds = embed_cache.get_chunk(start, end, self.device)
            prev_contexts = context_swapper.get_chunk(start, end, from_previous=True, device=self.device)

            shifted = torch.zeros(chunk_len, self.config.context_dim, dtype=self.torch_dtype, device=self.device)
            if start == 0:
                shifted[0] = last_context.squeeze(0)
            else:
                shifted[0] = context_swapper.get_chunk(start - 1, start, from_previous=False, device=self.device).squeeze(0)
            shifted[1:] = prev_contexts[:-1].detach()

            if self.config.phase1_context_noise > 0:
                shifted = shifted + torch.randn_like(shifted) * self.config.phase1_context_noise

            new_contexts = self.model.context_block(shifted, embeds)

            cvfp_loss = F.mse_loss(new_contexts, prev_contexts)
            div_loss = -torch.norm(new_contexts - new_contexts.mean(dim=0), p=2) / chunk_len
            loss = (1 - self.config.dist_reg_weight) * cvfp_loss + self.config.dist_reg_weight * div_loss

            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                token_mse = ((new_contexts - prev_contexts) ** 2).mean(dim=1)
                num_converged += (token_mse < self.config.phase1_convergence_threshold).sum().item()

            context_swapper.set_chunk(start, new_contexts.detach())
            gc.collect()

        return num_converged / num_tokens

    @property
    def is_streaming(self) -> bool:
        return True


def create_phase1_trainer(mode: str, model: torch.nn.Module, config, device: torch.device) -> Phase1Trainer:
    """
    Phase 1トレーナーを作成

    Args:
        mode: "memory" or "storage"
        model: LLMモデル
        config: ResidualConfig
        device: 計算デバイス

    Returns:
        Phase1Trainer
    """
    if mode == "memory":
        return MemoryPhase1Trainer(model, config, device)
    elif mode == "storage":
        return StoragePhase1Trainer(model, config, device)
    else:
        raise ValueError(f"Unknown trainer mode: {mode}")
