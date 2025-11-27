"""
StoragePhase1Trainer - ストレージベース型Phase 1トレーナー

mmapでディスクから直接処理（大規模データ用）
"""

import gc
import time
from typing import Union
import torch
import torch.nn.functional as F

from src.utils.disk_offload import ContextSwapper, EmbeddingCache
from src.evaluation import check_convergence, ConvergenceResult
from .base import Phase1Trainer, print_flush


class StoragePhase1Trainer(Phase1Trainer):
    """ストレージベース型Phase 1トレーナー"""

    def __init__(self, model: torch.nn.Module, config, device: torch.device):
        super().__init__(model, config, device)
        self.storage_dir = config.disk_offload_dir
        self.chunk_size = config.disk_offload_chunk_size
        self.use_bf16 = config.use_bf16
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

    def evaluate(
        self,
        token_ids: torch.Tensor,
        label: str = "Val",
        num_trials: int = None,
        return_contexts_only: bool = False
    ) -> Union[ConvergenceResult, torch.Tensor]:
        """
        検証データを評価（収束判定）

        複数回イテレーションして、CVFP損失が収束するかを判定。

        Args:
            token_ids: トークンID
            label: ラベル
            num_trials: イテレーション回数（Noneの場合はconfig.val_convergence_trialsを使用）
            return_contexts_only: Trueの場合はコンテキストのみ返す（後方互換性）

        Returns:
            ConvergenceResult または contexts
        """
        if num_trials is None:
            num_trials = getattr(self.config, 'val_convergence_trials', 10)

        print_flush(f"\nEvaluating {label} data...")

        result = check_convergence(
            model=self.model,
            token_ids=token_ids,
            device=self.device,
            num_trials=num_trials,
            verbose=True
        )

        if return_contexts_only:
            return result.contexts

        return result

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.phase1_gradient_clip)
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
