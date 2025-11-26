"""
Phase 1 ディスクオフロード版トレーナー

大規模データセット（25M+ トークン）のCVFP学習。
メモリマップファイルを使用してRAM制限を回避。
"""

import os
import sys
import json
import time
import gc
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

from src.utils.disk_offload import ContextSwapper, EmbeddingCache, TokenIDCache


def print_flush(msg: str):
    """即時フラッシュ付きprint。"""
    print(msg, flush=True)
    sys.stdout.flush()


def compute_cvfp_loss(
    contexts: torch.Tensor,
    previous_contexts: torch.Tensor
) -> torch.Tensor:
    """
    CVFP損失: 前回のコンテキストとのMSE。

    Args:
        contexts: 現在のコンテキスト [batch, context_dim]
        previous_contexts: 前回のコンテキスト [batch, context_dim]

    Returns:
        CVFP損失（スカラー）
    """
    return F.mse_loss(contexts, previous_contexts)


def compute_diversity_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    多様性損失: 平均からの偏差を最大化。

    Args:
        contexts: コンテキスト [batch, context_dim]

    Returns:
        多様性損失（負の値、最小化で多様性増加）
    """
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean
    diversity_loss = -torch.norm(deviation, p=2) / len(contexts)
    return diversity_loss


class Phase1DiskOffloadTrainer:
    """
    ディスクオフロード版Phase 1トレーナー。

    大規模データセットをチャンク単位で処理し、
    コンテキストをメモリマップファイルに保存。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        storage_dir: str,
        num_tokens: int,
        context_dim: int,
        use_bf16: bool = True,
        chunk_size: int = 1_000_000,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            model: LLMモデル
            storage_dir: ストレージディレクトリ
            num_tokens: 総トークン数
            context_dim: コンテキスト次元数
            use_bf16: bf16精度を使用するか
            chunk_size: チャンクサイズ（トークン数）
            device: 計算デバイス
        """
        self.model = model
        self.storage_dir = storage_dir
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.use_bf16 = use_bf16
        self.chunk_size = chunk_size
        self.device = device

        self.torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

        # ストレージ初期化
        self.context_swapper = ContextSwapper(
            storage_dir, num_tokens, context_dim, use_bf16
        )
        self.embed_cache = EmbeddingCache(
            storage_dir, num_tokens, context_dim, use_bf16
        )
        self.token_cache = TokenIDCache(storage_dir, num_tokens)

        # チェックポイントディレクトリ
        self.checkpoint_dir = os.path.join(storage_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 訓練状態
        self.current_iteration = 0
        self.training_stats = []

    def train(
        self,
        max_iterations: int = 10,
        learning_rate: float = 0.002,
        dist_reg_weight: float = 0.9,
        context_noise: float = 0.1,
        convergence_threshold: float = 0.03,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Phase 1訓練を実行。

        Args:
            max_iterations: 最大イテレーション数
            learning_rate: 学習率
            dist_reg_weight: 多様性損失の重み
            context_noise: コンテキストノイズの標準偏差
            convergence_threshold: 収束判定の閾値
            resume_from: 再開するチェックポイントのパス

        Returns:
            訓練統計
        """
        print_flush(f"\n{'='*70}")
        print_flush("Phase 1: CVFP Fixed-Point Learning (Disk Offload)")
        print_flush(f"{'='*70}")
        print_flush(f"  総トークン数: {self.num_tokens:,}")
        print_flush(f"  チャンクサイズ: {self.chunk_size:,}")
        print_flush(f"  最大イテレーション: {max_iterations}")
        print_flush(f"  学習率: {learning_rate}")
        print_flush(f"  多様性重み: {dist_reg_weight}")
        print_flush(f"  コンテキストノイズ: {context_noise}")
        print_flush(f"  精度: {'bf16' if self.use_bf16 else 'float32'}")
        print_flush(f"  デバイス: {self.device}")
        print_flush("")

        # モデルをデバイスに移動
        self.model.to(self.device)
        if self.use_bf16:
            self.model.to(self.torch_dtype)
        self.model.train()

        # オプティマイザ（ContextBlockのみ）
        context_params = list(self.model.context_block.parameters())
        optimizer = torch.optim.Adam(context_params, lr=learning_rate)

        print_flush(f"ContextBlockパラメータ数: {sum(p.numel() for p in context_params):,}")

        # 再開処理
        start_iteration = 0
        if resume_from and os.path.exists(resume_from):
            start_iteration = self._load_checkpoint(resume_from, optimizer)
            print_flush(f"チェックポイントから再開: イテレーション {start_iteration}")

        # ストレージを開く
        self.context_swapper.create_storage()
        self.context_swapper.open('r+')
        self.embed_cache.open('r')
        self.token_cache.open('r')

        try:
            for iteration in range(start_iteration, max_iterations):
                self.current_iteration = iteration
                iter_start = time.time()

                print_flush(f"\n--- イテレーション {iteration + 1}/{max_iterations} ---")

                if iteration == 0:
                    # Iteration 0: シーケンシャル処理（学習なし）
                    stats = self._process_sequential_iteration()
                else:
                    # Iteration 1+: 並列処理（学習あり）
                    stats = self._process_parallel_iteration(
                        optimizer, dist_reg_weight, context_noise, convergence_threshold
                    )

                iter_time = time.time() - iter_start
                stats['iteration'] = iteration + 1
                stats['time'] = iter_time
                self.training_stats.append(stats)

                # 統計表示
                self._print_iteration_stats(stats)

                # バッファを交換
                self.context_swapper.swap()
                self.context_swapper.flush()

                # チェックポイント保存
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"iteration_{iteration + 1:03d}.pt"
                )
                self._save_checkpoint(checkpoint_path, optimizer, stats)

                # メモリクリーンアップ
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        finally:
            self.context_swapper.close()
            self.embed_cache.close()
            self.token_cache.close()

        # 最終モデル保存
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        self._save_final_model(final_path)

        print_flush(f"\n{'='*70}")
        print_flush("Phase 1 訓練完了")
        print_flush(f"{'='*70}")
        print_flush(f"  最終モデル: {final_path}")

        return {
            'iterations': len(self.training_stats),
            'stats': self.training_stats,
            'final_model_path': final_path
        }

    def _process_sequential_iteration(self) -> Dict[str, Any]:
        """
        Iteration 0: シーケンシャル処理。

        各トークンを順番に処理し、コンテキストチェーンを構築。
        学習は行わない（ベースライン確立）。
        """
        print_flush("  シーケンシャル処理中（学習なし）...")

        self.model.eval()
        num_chunks = (self.num_tokens + self.chunk_size - 1) // self.chunk_size

        # 初期コンテキスト（ゼロ）
        context = torch.zeros(
            1, self.context_dim,
            dtype=self.torch_dtype, device=self.device
        )

        tokens_processed = 0

        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * self.chunk_size
                chunk_end = min(chunk_start + self.chunk_size, self.num_tokens)
                chunk_len = chunk_end - chunk_start

                # 埋め込みをロード
                embeds = self.embed_cache.get_chunk(chunk_start, chunk_end, self.device)

                # チャンク内のコンテキストを保存するバッファ
                chunk_contexts = torch.zeros(
                    chunk_len, self.context_dim,
                    dtype=self.torch_dtype, device=self.device
                )

                # シーケンシャル処理
                for t in range(chunk_len):
                    token_embed = embeds[t:t+1]
                    context = self.model.context_block(context, token_embed)
                    chunk_contexts[t] = context.squeeze(0)

                # ディスクに保存
                self.context_swapper.set_chunk(chunk_start, chunk_contexts)

                tokens_processed += chunk_len
                progress = tokens_processed / self.num_tokens * 100
                print_flush(f"    進捗: {tokens_processed:,}/{self.num_tokens:,} ({progress:.1f}%)")

                # メモリクリーンアップ
                del embeds, chunk_contexts
                gc.collect()

        self.model.train()

        return {
            'type': 'sequential',
            'tokens_processed': tokens_processed
        }

    def _process_parallel_iteration(
        self,
        optimizer: torch.optim.Optimizer,
        dist_reg_weight: float,
        context_noise: float,
        convergence_threshold: float
    ) -> Dict[str, Any]:
        """
        Iteration 1+: 並列処理（学習あり）。
        """
        print_flush("  並列処理中（学習あり）...")

        self.model.train()
        num_chunks = (self.num_tokens + self.chunk_size - 1) // self.chunk_size

        total_cvfp_loss = 0.0
        total_diversity_loss = 0.0
        total_loss = 0.0
        num_converged = 0
        tokens_processed = 0

        # 前イテレーションの最終コンテキスト（チャンク境界用）
        last_context = self.context_swapper.get_chunk(
            self.num_tokens - 1, self.num_tokens,
            from_previous=True, device=self.device
        )

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, self.num_tokens)
            chunk_len = chunk_end - chunk_start

            # データをロード
            embeds = self.embed_cache.get_chunk(chunk_start, chunk_end, self.device)
            prev_contexts = self.context_swapper.get_chunk(
                chunk_start, chunk_end,
                from_previous=True, device=self.device
            )

            # シフトしたコンテキストを準備（token i は context i-1 を使用）
            shifted_contexts = torch.zeros(
                chunk_len, self.context_dim,
                dtype=self.torch_dtype, device=self.device
            )

            if chunk_idx == 0:
                # 最初のチャンク: token 0 は前イテレーションの最終コンテキストを使用
                shifted_contexts[0] = last_context.squeeze(0)
            else:
                # チャンク境界: 前チャンクの最終コンテキスト
                shifted_contexts[0] = self.context_swapper.get_chunk(
                    chunk_start - 1, chunk_start,
                    from_previous=False, device=self.device
                ).squeeze(0)

            # token 1〜N は prev_contexts[0:N-1] を使用
            shifted_contexts[1:] = prev_contexts[:-1].detach()

            # コンテキストノイズ追加
            if context_noise > 0:
                noise = torch.randn_like(shifted_contexts) * context_noise
                shifted_contexts = shifted_contexts + noise

            # 順伝播（全レイヤーを通して最終コンテキストを取得）
            new_contexts = self.model.context_block(shifted_contexts, embeds)

            # CVFP損失
            cvfp_loss = compute_cvfp_loss(new_contexts, prev_contexts)

            # 多様性損失
            diversity_loss = compute_diversity_loss(new_contexts)

            # 総合損失
            loss = (1 - dist_reg_weight) * cvfp_loss + dist_reg_weight * diversity_loss

            # 逆伝播・更新
            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            # 統計更新
            with torch.no_grad():
                total_cvfp_loss += cvfp_loss.item() * chunk_len
                total_diversity_loss += diversity_loss.item() * chunk_len
                total_loss += loss.item() * chunk_len

                # 収束判定
                token_mse = ((new_contexts - prev_contexts) ** 2).mean(dim=1)
                num_converged += (token_mse < convergence_threshold).sum().item()

            # ディスクに保存
            self.context_swapper.set_chunk(chunk_start, new_contexts.detach())

            tokens_processed += chunk_len
            progress = tokens_processed / self.num_tokens * 100
            print_flush(
                f"    チャンク {chunk_idx + 1}/{num_chunks}: "
                f"CVFP={cvfp_loss.item():.6f} Div={diversity_loss.item():.6f} "
                f"({progress:.1f}%)"
            )

            # メモリクリーンアップ
            del embeds, prev_contexts, shifted_contexts, new_contexts
            gc.collect()

        return {
            'type': 'parallel',
            'tokens_processed': tokens_processed,
            'avg_cvfp_loss': total_cvfp_loss / self.num_tokens,
            'avg_diversity_loss': total_diversity_loss / self.num_tokens,
            'avg_total_loss': total_loss / self.num_tokens,
            'convergence_rate': num_converged / self.num_tokens,
            'num_converged': num_converged
        }

    def _print_iteration_stats(self, stats: Dict[str, Any]):
        """イテレーション統計を表示。"""
        print_flush(f"\n  イテレーション {stats['iteration']} 完了:")
        print_flush(f"    処理トークン: {stats['tokens_processed']:,}")
        print_flush(f"    処理時間: {stats['time']:.1f}秒")

        if stats['type'] == 'parallel':
            print_flush(f"    平均CVFP損失: {stats['avg_cvfp_loss']:.6f}")
            print_flush(f"    平均多様性損失: {stats['avg_diversity_loss']:.6f}")
            print_flush(f"    収束率: {stats['convergence_rate']*100:.1f}% ({stats['num_converged']:,}/{self.num_tokens:,})")

    def _save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        stats: Dict[str, Any]
    ):
        """チェックポイントを保存。"""
        checkpoint = {
            'iteration': self.current_iteration + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats,
            'training_stats': self.training_stats,
            'config': {
                'num_tokens': self.num_tokens,
                'context_dim': self.context_dim,
                'use_bf16': self.use_bf16,
                'chunk_size': self.chunk_size
            }
        }
        torch.save(checkpoint, path)
        print_flush(f"  チェックポイント保存: {path}")

    def _load_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer
    ) -> int:
        """チェックポイントをロード。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', [])
        return checkpoint['iteration']

    def _save_final_model(self, path: str):
        """最終モデルを保存（Phase 2用）。"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_tokens': self.num_tokens,
                'context_dim': self.context_dim,
                'use_bf16': self.use_bf16,
                'num_layers': self.model.context_block.num_layers
            },
            'training_stats': self.training_stats,
            'created_at': datetime.now().isoformat()
        }
        torch.save(checkpoint, path)
