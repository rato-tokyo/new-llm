#!/usr/bin/env python3
"""
Cascade Context 実験スクリプト（Initial Context Inheritance方式）

Initial Context Inheritance方式:
N個のContextBlockを順次学習し、各ブロックは独立してRNN学習を行う。
ブロック1以降は、最初のトークンの入力として前のブロックの最終出力を使用。

これにより、前のブロックの「文脈の継続性」を引き継ぎつつ、
全データでRNN学習を行う。Dual方式（前半/後半分割）と同様の効果を
全データで実現できる。

アーキテクチャ（1層固定、可変ブロック数）:
  Phase 1[0]:
    - ContextBlock[0] を全データで学習
    - 初期入力: ゼロベクトル
    - 出力: context[0] (キャッシュ)

  Phase 1[1..N-1]:
    - ContextBlock[i] を全データで学習
    - 初期入力: context[i-1]_final（前のブロックの最終出力）
    - それ以降の入力: 自身の前の出力（標準RNN）
    - 出力: context[i] (キャッシュ)

  Phase 2:
    - 全context[0..N-1]を連結 → cd=context_dim*N
    - TokenBlock で学習

使用方法:
  python3 scripts/experiment_cascade_context.py
  python3 scripts/experiment_cascade_context.py -s 2000 -n 2  # デフォルト: 2ブロック
  python3 scripts/experiment_cascade_context.py -s 2000 -n 3  # 3ブロック
  python3 scripts/experiment_cascade_context.py -s 2000 -n 1  # 1ブロック（カスケードなし）
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from config import Config
from src.models.blocks import ContextBlock, TokenBlock
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed
from src.utils.initialization import count_parameters
from config.experiment import DataConfig


class CascadeContextLLM(nn.Module):
    """
    Cascade Context LLM - N個のContextBlockをカスケード接続（1層固定）

    Phase 1[0]: ContextBlock[0] を学習（入力: ゼロ初期化context）
    Phase 1[i]: ContextBlock[i] を学習（入力: [i-1] の出力、固定）
    Phase 2: concat(context[0..N-1]) で TokenBlock を学習

    Args:
        vocab_size: 語彙サイズ
        embed_dim: トークン埋め込み次元
        context_dim: 各ContextBlockの出力次元
        num_input_tokens: 入力トークン数
        num_context_blocks: ContextBlockの数（デフォルト: 2）
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_dim: int,
        num_input_tokens: int = 1,
        num_context_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_context_blocks = num_context_blocks
        self.combined_context_dim = context_dim * num_context_blocks
        self.num_input_tokens = num_input_tokens

        # Token Embeddings (GPT-2 pretrained)
        self._load_pretrained_embeddings()
        self.embed_norm = nn.LayerNorm(embed_dim)

        # N個のContextBlock（各1層）
        self.context_blocks = nn.ModuleList([
            ContextBlock(
                context_dim=context_dim,
                embed_dim=embed_dim,
                num_input_tokens=num_input_tokens,
            )
            for _ in range(num_context_blocks)
        ])

        # TokenBlock（連結されたcontext用、cd=context_dim*N、1層）
        self.token_block = TokenBlock(
            context_dim=self.combined_context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        # Output Head (Weight Tying)
        self.token_output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_output.weight = self.token_embedding.weight
        print_flush("✓ Weight Tying: token_output shares weights with token_embedding")

    def _load_pretrained_embeddings(self) -> None:
        """GPT-2 embeddings をロード"""
        try:
            from transformers import GPT2Model
            print_flush("Loading GPT-2 pretrained embeddings...")
            gpt2 = GPT2Model.from_pretrained('gpt2')
            pretrained_embeddings = gpt2.wte.weight.data

            self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.token_embedding.weight.data.copy_(pretrained_embeddings)
            self.token_embedding.weight.requires_grad = False
            print_flush(f"✓ Loaded GPT-2 embeddings: {pretrained_embeddings.shape}")
        except Exception as e:
            print_flush(f"Warning: Failed to load GPT-2 embeddings: {e}")
            self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def forward_context(self, block_idx: int, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """指定されたContextBlockの順伝搬"""
        return self.context_blocks[block_idx](context, token_embeds)

    def forward_token(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """TokenBlock の順伝搬（1層固定、contextは連結済み）"""
        return self.token_block(context, token_embeds)

    def freeze_context_block(self, block_idx: int) -> None:
        """指定されたContextBlockをfreeze"""
        for param in self.context_blocks[block_idx].parameters():
            param.requires_grad = False

    def freeze_all_context_blocks(self) -> None:
        """全ContextBlockをfreeze"""
        for i in range(self.num_context_blocks):
            self.freeze_context_block(i)
        print_flush(f"✓ All {self.num_context_blocks} ContextBlocks frozen")

    def num_params(self) -> Dict[str, int]:
        """パラメータ数を返す"""
        embedding_params = self.token_embedding.weight.numel()
        embed_norm_params = count_parameters(self.embed_norm)

        context_block_params = {}
        total_context_params = 0
        for i, block in enumerate(self.context_blocks):
            params = count_parameters(block)
            context_block_params[f'context_block_{i}'] = params
            total_context_params += params

        token_block_params = count_parameters(self.token_block)

        total = embedding_params + embed_norm_params + total_context_params + token_block_params

        result = {
            'embedding': embedding_params,
            'embed_norm': embed_norm_params,
            'token_block': token_block_params,
            'total': total,
            'total_context_blocks': total_context_params,
        }
        result.update(context_block_params)
        return result


class SingleContextWrapper(nn.Module):
    """
    Phase 1 用: ContextBlock のラッパー
    MemoryPhase1Trainer と互換性を持たせる。
    """

    def __init__(self, cascade_model: CascadeContextLLM, block_idx: int = 0):
        super().__init__()
        self.cascade_model = cascade_model
        self.block_idx = block_idx

        # Phase1Trainerが期待するプロパティ
        self.token_embedding = cascade_model.token_embedding
        self.embed_norm = cascade_model.embed_norm
        self.context_dim = cascade_model.context_dim
        self.embed_dim = cascade_model.embed_dim
        self.num_input_tokens = cascade_model.num_input_tokens
        self.vocab_size = cascade_model.vocab_size

        self.context_block = cascade_model.context_blocks[block_idx]

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """ContextBlock の順伝搬"""
        return self.context_block(context, token_embeds)


class CascadePhase2Trainer:
    """Cascade Context 用の Phase 2 トレーナー"""

    def __init__(self, model: CascadeContextLLM, config: Any, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    def train_full(
        self,
        train_token_ids: torch.Tensor,
        val_token_ids: torch.Tensor,
        train_context_cache: torch.Tensor,
        train_token_embeds: torch.Tensor,
        val_context_cache: torch.Tensor,
        val_token_embeds: torch.Tensor,
    ) -> Dict[str, Any]:
        """Phase 2 学習を実行"""
        from torch.optim import AdamW

        self.model.to(self.device)
        self.model.freeze_all_context_blocks()
        self.model.token_embedding.weight.requires_grad = False
        print_flush("✓ Embedding frozen")

        # 学習対象のパラメータ
        trainable_params = [p for p in self.model.token_block.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = self.model.num_params()['total']
        print_flush(f"✓ Training TokenBlock only: {total_trainable:,}/{total_params:,} parameters")

        optimizer = AdamW(trainable_params, lr=self.config.phase2_learning_rate)
        criterion = nn.CrossEntropyLoss()

        # ターゲット
        train_targets = train_token_ids[1:].to(self.device)
        val_targets = val_token_ids[1:].to(self.device)

        history: Dict[str, Any] = {
            'train_ppl': [],
            'val_ppl': [],
            'val_acc': [],
            'best_epoch': 1,
        }

        best_val_ppl = float('inf')
        patience_counter = 0
        prev_val_ppl = float('inf')

        num_train = len(train_targets)
        batch_size = self.config.phase2_batch_size or 1000

        print_flush(f"\n[Phase 2] {num_train:,} train / {len(val_targets):,} val tokens, "
                    f"{self.config.phase2_epochs} epochs")

        for epoch in range(1, self.config.phase2_epochs + 1):
            epoch_start = time.time()

            # === Training ===
            self.model.train()
            total_loss = 0.0

            for start_idx in range(0, num_train, batch_size):
                end_idx = min(start_idx + batch_size, num_train)

                batch_token_embeds = train_token_embeds[start_idx:end_idx].to(self.device)
                batch_targets = train_targets[start_idx:end_idx]
                batch_context = train_context_cache[start_idx:end_idx].to(self.device)

                optimizer.zero_grad()
                token_out = self.model.forward_token(batch_context, batch_token_embeds)
                logits = self.model.token_output(token_out)

                loss = criterion(logits, batch_targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, self.config.phase2_gradient_clip)
                optimizer.step()

                total_loss += loss.item() * (end_idx - start_idx)

            train_ppl = torch.exp(torch.tensor(total_loss / num_train)).item()

            # === Validation ===
            self.model.eval()
            val_loss = 0.0
            correct = 0
            num_val = len(val_targets)

            with torch.no_grad():
                for start_idx in range(0, num_val, batch_size):
                    end_idx = min(start_idx + batch_size, num_val)

                    batch_token_embeds = val_token_embeds[start_idx:end_idx].to(self.device)
                    batch_targets = val_targets[start_idx:end_idx]
                    batch_context = val_context_cache[start_idx:end_idx].to(self.device)

                    token_out = self.model.forward_token(batch_context, batch_token_embeds)
                    logits = self.model.token_output(token_out)

                    val_loss += criterion(logits, batch_targets).item() * (end_idx - start_idx)
                    correct += (logits.argmax(dim=-1) == batch_targets).sum().item()

            val_ppl = torch.exp(torch.tensor(val_loss / num_val)).item()
            val_acc = correct / num_val

            history['train_ppl'].append(train_ppl)
            history['val_ppl'].append(val_ppl)
            history['val_acc'].append(val_acc)

            is_best = val_ppl < best_val_ppl
            marker = " ★" if is_best else ""

            if is_best:
                best_val_ppl = val_ppl
                history['best_epoch'] = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            elapsed = time.time() - epoch_start
            print_flush(f"  Epoch {epoch}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                        f"acc={val_acc*100:.1f}% [{elapsed:.1f}s]{marker}")

            # Early stopping
            ppl_improvement = prev_val_ppl - val_ppl
            min_improvement = getattr(self.config, 'phase2_min_ppl_improvement', 0.4)

            if epoch > 1 and ppl_improvement < min_improvement and ppl_improvement >= 0:
                print_flush(f"  → Early stop at epoch {epoch} (PPL improvement {ppl_improvement:.2f} < {min_improvement})")
                break

            if patience_counter >= self.config.phase2_patience:
                print_flush(f"  → Early stop at epoch {epoch}")
                break

            prev_val_ppl = val_ppl

        print_flush(f"  Best: epoch {history['best_epoch']}, ppl={best_val_ppl:.1f}, "
                    f"acc={history['val_acc'][history['best_epoch']-1]*100:.1f}%")

        return history


def collect_context_cache_for_val(
    model: CascadeContextLLM,
    token_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Validation用: 全ContextBlockのキャッシュを収集（Initial Context Inheritance方式）

    Initial Context Inheritance:
    - 各ブロックは独立してRNN処理を行う
    - ブロック0: 初期入力はゼロベクトル
    - ブロック1以降: 初期入力は前のブロックの最終出力

    Returns:
        (context_caches, token_embeds)
        - context_caches: List of [num_tokens, context_dim] for each block
        - token_embeds: [num_tokens, embed_dim]
    """
    model.eval()
    num_tokens = len(token_ids) - 1
    num_blocks = model.num_context_blocks

    with torch.no_grad():
        all_embeds = model.token_embedding(token_ids.to(device))
        all_embeds = model.embed_norm(all_embeds)
        all_embeds_cpu = all_embeds.cpu()
        del all_embeds
        clear_gpu_cache(device)

    input_embeds = all_embeds_cpu[:-1]
    del all_embeds_cpu

    # 各ブロック用のキャッシュを準備
    context_caches = [
        torch.zeros(num_tokens, model.context_dim, device='cpu', dtype=torch.float32)
        for _ in range(num_blocks)
    ]

    print_flush(f"    Collecting val cache ({num_tokens:,} tokens, {num_blocks} blocks)...")
    collect_start = time.time()

    with torch.no_grad():
        # 各ブロックを順に処理（Initial Context Inheritance方式）
        for block_idx in range(num_blocks):
            if block_idx == 0:
                # ブロック0: 初期入力はゼロベクトル
                prev_context = torch.zeros(1, model.context_dim, device=device)
            else:
                # ブロック1以降: 初期入力は前のブロックの最終出力
                prev_context = context_caches[block_idx - 1][-1:].to(device)

            # RNN的に順次処理
            for i in range(num_tokens):
                token_embed = input_embeds[i:i+1].to(device)
                new_context = model.forward_context(block_idx, prev_context, token_embed)
                context_caches[block_idx][i] = new_context.cpu()
                prev_context = new_context

            if block_idx < num_blocks - 1:
                print_flush(f"      Block {block_idx} done")

    print_flush(f"    Val cache collected [{time.time() - collect_start:.1f}s]")
    clear_gpu_cache(device)

    return context_caches, input_embeds


class CascadePhase1Trainer:
    """
    Phase 1用トレーナー: Initial Context Inheritance方式

    Initial Context Inheritance:
    - block_idx=0: 通常のRNN学習（初期入力: ゼロベクトル）
    - block_idx>0: 通常のRNN学習だが、最初のトークンの入力のみ
                   前のブロックの最終出力（context_prev_final）を使用

    これにより、前のブロックの「文脈の継続性」を引き継ぎつつ、
    全データでRNN学習を行う。Dual方式（前半/後半分割）と同様の効果を
    全データで実現できる。

    block_idx=0の場合: ゼロベクトルを初期入力として使用
    block_idx>0の場合: context_prev_final を初期入力として使用
    """

    def __init__(
        self,
        model: CascadeContextLLM,
        config: Any,
        device: torch.device,
        block_idx: int,
        prev_context_final: Optional[torch.Tensor],  # block_idx>0の場合: 前のブロックの最終出力 [1, context_dim]
        token_embeds: torch.Tensor
    ):
        self.model = model
        self.config = config
        self.device = device
        self.block_idx = block_idx
        self.prev_context_final = prev_context_final  # [1, context_dim] or None
        self.token_embeds = token_embeds  # [num_tokens, embed_dim]
        self._training_stats: Dict[str, Any] = {}

    def train(self, label: str = "Context") -> torch.Tensor:
        """
        ContextBlockを学習（Initial Context Inheritance方式）

        Initial Context Inheritance:
        - 最初のトークン: prev_context_final（前のブロックの最終出力）を入力
        - それ以降: previous_contexts[i-1]（自身の前の出力）を入力

        Returns:
            context_cache: [num_tokens, context_dim]
        """
        self.model.train()
        num_tokens = len(self.token_embeds)

        # このブロックのパラメータのみ学習
        block_params = list(self.model.context_blocks[self.block_idx].parameters())
        phase_label = f"Phase 1[{self.block_idx}]"
        print_flush(f"\n[{phase_label}] {label}: {num_tokens:,} tokens, {self.config.phase1_max_iterations} iterations")

        if self.block_idx == 0 or self.prev_context_final is None:
            print_flush("  Initial input: zero vector (standard RNN)")
        else:
            print_flush(f"  Initial input: context[{self.block_idx-1}]_final (Initial Context Inheritance)")

        optimizer = torch.optim.Adam(block_params, lr=self.config.phase1_learning_rate)

        # 収束率Early Stoppingの設定
        early_stopping = getattr(self.config, 'phase1_early_stopping', True)
        early_stopping_threshold = getattr(self.config, 'phase1_early_stopping_threshold', 0.30)
        min_convergence_improvement = getattr(self.config, 'phase1_min_convergence_improvement', 0.01)

        # 初期入力コンテキストの準備
        # Initial Context Inheritance: 最初のトークンの入力のみ prev_context_final を使用
        if self.block_idx == 0 or self.prev_context_final is None:
            initial_context = torch.zeros(1, self.model.context_dim, device='cpu')
        else:
            initial_context = self.prev_context_final.cpu()  # [1, context_dim]

        previous_contexts: Optional[torch.Tensor] = None
        final_convergence_rate = 0.0
        prev_convergence_rate = 0.0
        no_improvement_count = 0
        final_iter = 0

        batch_size = self.config.phase1_batch_size

        for iteration in range(self.config.phase1_max_iterations):
            final_iter = iteration
            iter_start = time.time()

            if iteration == 0:
                # Iteration 0: 小さなランダム値で初期化
                previous_contexts = torch.randn(num_tokens, self.model.context_dim) * 0.01
                print_flush("  Iter 1: random init")
                continue

            assert previous_contexts is not None
            optimizer.zero_grad()

            # shifted_prev_context を作成（RNN的に previous_contexts[i-1] を使用）
            # 最初のトークンの入力は initial_context（ゼロ or prev_context_final）
            shifted_prev_context = torch.cat([initial_context, previous_contexts[:-1]], dim=0)

            all_contexts = []
            total_loss = 0.0
            num_batches = (num_tokens + batch_size - 1) // batch_size

            for start_idx in range(0, num_tokens, batch_size):
                end_idx = min(start_idx + batch_size, num_tokens)
                current_batch_size = end_idx - start_idx

                batch_prev_context = shifted_prev_context[start_idx:end_idx].to(self.device)
                batch_token_embeds = self.token_embeds[start_idx:end_idx].to(self.device)

                # ノイズ追加
                if self.config.phase1_context_noise > 0:
                    noise = torch.randn_like(batch_prev_context) * self.config.phase1_context_noise
                    batch_prev_context = batch_prev_context + noise

                # Forward pass
                batch_output = self.model.forward_context(self.block_idx, batch_prev_context, batch_token_embeds)

                # 多様性損失
                from src.losses.diversity import oacd_loss
                diversity_loss = oacd_loss(batch_output)
                scaled_loss = diversity_loss / num_batches

                if not torch.isnan(scaled_loss) and not torch.isinf(scaled_loss):
                    scaled_loss.backward()

                all_contexts.append(batch_output.detach().cpu())
                total_loss += diversity_loss.item() * current_batch_size

                del batch_prev_context, batch_token_embeds, batch_output
                clear_gpu_cache(self.device)

            # パラメータ更新
            torch.nn.utils.clip_grad_norm_(
                self.model.context_blocks[self.block_idx].parameters(),
                max_norm=self.config.phase1_gradient_clip
            )
            optimizer.step()

            # 収束率計算
            contexts = torch.cat(all_contexts, dim=0)
            converged = ((contexts - previous_contexts) ** 2).mean(dim=1) < self.config.phase1_convergence_threshold
            convergence_rate = converged.float().mean().item()

            previous_contexts = contexts
            final_convergence_rate = convergence_rate

            # 改善幅計算
            improvement = convergence_rate - prev_convergence_rate
            improvement_marker = ""
            min_conv_for_check = 0.5

            if convergence_rate >= min_conv_for_check:
                if improvement < min_convergence_improvement:
                    no_improvement_count += 1
                    improvement_marker = f" (↑{improvement*100:.1f}%)"
                else:
                    no_improvement_count = 0

            elapsed = time.time() - iter_start
            avg_loss = total_loss / num_tokens
            print_flush(
                f"  Iter {iteration+1}: conv={convergence_rate*100:.0f}% "
                f"loss={avg_loss:.4f} [{elapsed:.1f}s]{improvement_marker}"
            )

            prev_convergence_rate = convergence_rate

            # Early Stopping
            if early_stopping and convergence_rate >= early_stopping_threshold:
                print_flush(f"  → Early stop: conv {convergence_rate*100:.0f}% >= {early_stopping_threshold*100:.0f}%")
                break

            if early_stopping and no_improvement_count >= 1 and convergence_rate >= min_conv_for_check:
                print_flush(f"  → Early stop: improvement {improvement*100:.1f}% < {min_convergence_improvement*100:.0f}%")
                break

        self._training_stats = {
            'iterations': final_iter + 1,
            'convergence_rate': final_convergence_rate,
            'num_tokens': num_tokens,
        }

        print_flush(f"  Done: {final_convergence_rate*100:.0f}% converged")

        # キャッシュを収集（学習後の最終状態で再計算）
        assert previous_contexts is not None
        context_cache = self._collect_cache(initial_context, previous_contexts)

        return context_cache

    def _collect_cache(self, initial_context: torch.Tensor, previous_contexts: torch.Tensor) -> torch.Tensor:
        """
        学習済みContextBlockでキャッシュを収集

        Args:
            initial_context: 最初のトークンの入力 [1, context_dim]
            previous_contexts: 学習後のコンテキスト [num_tokens, context_dim]
        """
        self.model.eval()
        num_tokens = len(self.token_embeds)
        batch_size = 50000

        # shifted_prev_context を作成
        shifted_prev_context = torch.cat([initial_context, previous_contexts[:-1]], dim=0)

        context_cache = torch.zeros(num_tokens, self.model.context_dim, device='cpu', dtype=torch.float32)

        print_flush(f"  Collecting context[{self.block_idx}] cache...")
        collect_start = time.time()

        with torch.no_grad():
            for start_idx in range(0, num_tokens, batch_size):
                end_idx = min(start_idx + batch_size, num_tokens)

                batch_prev_context = shifted_prev_context[start_idx:end_idx].to(self.device)
                batch_token_embeds = self.token_embeds[start_idx:end_idx].to(self.device)

                batch_output = self.model.forward_context(self.block_idx, batch_prev_context, batch_token_embeds)
                context_cache[start_idx:end_idx] = batch_output.cpu()

                del batch_prev_context, batch_token_embeds, batch_output
                clear_gpu_cache(self.device)

        print_flush(f"  Cache collected [{time.time() - collect_start:.1f}s]")
        self.model.train()

        return context_cache


class Phase1ConfigWrapper:
    """Phase1Trainer用のConfig wrapper"""

    def __init__(self, base: Config, context_dim: int):
        self.phase1_max_iterations = base.phase1_max_iterations
        self.phase1_convergence_threshold = base.phase1_convergence_threshold
        self.phase1_learning_rate = base.phase1_learning_rate
        self.phase1_batch_size = base.phase1_batch_size
        self.phase1_gradient_clip = base.phase1_gradient_clip
        self.phase1_context_noise = base.phase1_context_noise
        self.phase1_early_stopping = base.phase1_early_stopping
        self.phase1_early_stopping_threshold = base.phase1_early_stopping_threshold
        self.phase1_min_convergence_improvement = base.phase1_min_convergence_improvement
        self.context_dim = context_dim
        self.embed_dim = base.embed_dim
        self.vocab_size = base.vocab_size
        self.num_input_tokens = base.num_input_tokens


class Phase2ConfigWrapper:
    """Phase2Trainer用のConfig wrapper"""

    def __init__(self, base: Config):
        self.phase2_learning_rate = base.phase2_learning_rate
        self.phase2_epochs = base.phase2_epochs
        self.phase2_patience = base.phase2_patience
        self.phase2_batch_size = base.phase2_batch_size
        self.phase2_gradient_clip = base.phase2_gradient_clip
        self.phase2_min_ppl_improvement = base.phase2_min_ppl_improvement


def run_cascade_context_experiment(
    num_samples: int = 2000,
    context_dim: int = 500,
    num_context_blocks: int = 2,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Cascade Context 実験を実行

    Args:
        num_samples: サンプル数
        context_dim: 各ContextBlockの出力次元
        num_context_blocks: ContextBlockの数（1以上）
        seed: 乱数シード
        output_dir: 出力ディレクトリ
    """

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    base_config = Config()

    # データ読み込み
    print_flush("Loading data...")
    data_config = DataConfig.from_base(base_config, num_samples=num_samples)
    data_provider = MemoryDataProvider(data_config)
    train_token_ids, val_token_ids = data_provider.load_data()
    train_token_ids = train_token_ids.to(device)
    val_token_ids = val_token_ids.to(device)

    num_train_tokens = len(train_token_ids)
    num_val_tokens = len(val_token_ids)
    print_flush(f"Data: {num_train_tokens:,} train, {num_val_tokens:,} val tokens")

    # モデル作成
    combined_dim = context_dim * num_context_blocks
    print_flush(f"\nCreating CascadeContextLLM (cd={context_dim}x{num_context_blocks}={combined_dim})...")
    model = CascadeContextLLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=context_dim,
        num_input_tokens=base_config.num_input_tokens,
        num_context_blocks=num_context_blocks,
    )
    model.to(device)

    params = model.num_params()
    print_flush(f"Parameters: {params['total']:,} total")
    print_flush(f"  ContextBlocks ({num_context_blocks}): {params['total_context_blocks']:,}")
    for i in range(num_context_blocks):
        print_flush(f"    Block {i}: {params[f'context_block_{i}']:,}")
    print_flush(f"  TokenBlock: {params['token_block']:,}")

    config_wrapper = Phase1ConfigWrapper(base_config, context_dim)

    # ========== Phase 1: 各ContextBlockを順次学習 ==========
    train_context_caches: List[torch.Tensor] = []
    phase1_times: List[float] = []
    phase1_stats: List[Dict[str, Any]] = []
    train_token_embeds: Optional[torch.Tensor] = None

    for block_idx in range(num_context_blocks):
        if block_idx == 0:
            # 最初のブロック: MemoryPhase1Trainerを使用（token_embedsも取得）
            print_flush(f"\n[Phase 1[{block_idx}]] Training ContextBlock {block_idx} on full data...")
            wrapper = SingleContextWrapper(model, block_idx=block_idx)
            trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

            phase_start = time.time()
            result = trainer.train(train_token_ids, label=f"Context{block_idx}", return_all_layers=True)
            phase_time = time.time() - phase_start

            assert result.cache is not None
            assert result.token_embeds is not None
            train_context_caches.append(result.cache)
            train_token_embeds = result.token_embeds

            stats = trainer._training_stats
        else:
            # 後続ブロック: CascadePhase1Trainerを使用（Initial Context Inheritance）
            assert train_token_embeds is not None
            # 前のブロックの最終出力を取得 [1, context_dim]
            prev_context_final = train_context_caches[block_idx - 1][-1:].clone()
            trainer_cascade = CascadePhase1Trainer(
                model, config_wrapper, device,
                block_idx=block_idx,
                prev_context_final=prev_context_final,
                token_embeds=train_token_embeds
            )

            phase_start = time.time()
            cache = trainer_cascade.train(label=f"Context{block_idx}")
            phase_time = time.time() - phase_start

            train_context_caches.append(cache)
            stats = trainer_cascade._training_stats

        phase1_times.append(phase_time)
        phase1_stats.append(stats)

        print_flush(f"Phase 1[{block_idx}]: {phase_time:.1f}s, {stats.get('iterations', 0)} iter, "
                    f"conv={stats.get('convergence_rate', 0)*100:.0f}%")

        # このブロックをfreeze
        model.freeze_context_block(block_idx)
        print_flush(f"✓ ContextBlock {block_idx} frozen")

    phase1_total_time = sum(phase1_times)
    assert train_token_embeds is not None

    # ========== Validation キャッシュ収集 ==========
    print_flush("\n[Val Cache] Collecting validation cache...")
    cache_start = time.time()

    val_context_caches, val_token_embeds = collect_context_cache_for_val(
        model, val_token_ids, device
    )

    cache_time = time.time() - cache_start
    print_flush(f"Val cache collection: {cache_time:.1f}s")

    # 連結
    train_context_cache = torch.cat(train_context_caches, dim=-1)
    val_context_cache = torch.cat(val_context_caches, dim=-1)

    print_flush(f"  Train cache: {train_context_cache.shape}")
    print_flush(f"  Val cache: {val_context_cache.shape}")

    # メモリ解放
    del train_context_caches, val_context_caches

    # Effective Rank計算
    train_metrics = analyze_fixed_points(train_context_cache, label="Train", verbose=False)
    val_metrics = analyze_fixed_points(val_context_cache, label="Val", verbose=False)
    train_er = train_metrics['effective_rank']
    val_er = val_metrics['effective_rank']
    train_er_pct = train_er / combined_dim * 100
    val_er_pct = val_er / combined_dim * 100
    print_flush(f"Effective Rank: Train={train_er_pct:.1f}%, Val={val_er_pct:.1f}%")

    # ========== Phase 2: TokenBlock の学習 ==========
    print_flush(f"\n[Phase 2] Training TokenBlock with concatenated context (cd={combined_dim})...")

    phase2_config = Phase2ConfigWrapper(base_config)
    phase2_trainer = CascadePhase2Trainer(model, phase2_config, device)

    phase2_start = time.time()
    history = phase2_trainer.train_full(
        train_token_ids=train_token_ids,
        val_token_ids=val_token_ids,
        train_context_cache=train_context_cache,
        train_token_embeds=train_token_embeds,
        val_context_cache=val_context_cache,
        val_token_embeds=val_token_embeds,
    )
    phase2_time = time.time() - phase2_start

    best_epoch = history['best_epoch']
    best_ppl = history['val_ppl'][best_epoch - 1]
    best_acc = history['val_acc'][best_epoch - 1]

    total_time = phase1_total_time + cache_time + phase2_time

    print_flush(f"\nPhase 2: {phase2_time:.1f}s, Best epoch {best_epoch}")
    print_flush(f"Result: PPL={best_ppl:.1f}, Acc={best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")

    # ========== 結果サマリー ==========
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY - Cascade Context Experiment (Initial Context Inheritance)")
    print_flush("=" * 70)
    print_flush(f"Architecture: CascadeContextLLM ({num_context_blocks} blocks, 1L each)")
    for i in range(num_context_blocks):
        if i == 0:
            print_flush(f"  ContextBlock {i}: cd={context_dim}, initial_input=zero")
        else:
            print_flush(f"  ContextBlock {i}: cd={context_dim}, initial_input=context[{i-1}]_final")
    print_flush(f"  TokenBlock: cd={combined_dim} (concatenated)")
    print_flush(f"Parameters: {params['total']:,}")
    for i in range(num_context_blocks):
        print_flush(f"Phase 1[{i}]: {phase1_times[i]:.1f}s, conv={phase1_stats[i].get('convergence_rate', 0)*100:.0f}%")
    print_flush(f"Cache collection: {cache_time:.1f}s")
    print_flush(f"Phase 2: {phase2_time:.1f}s, epoch {best_epoch}")
    print_flush(f"Effective Rank: {val_er_pct:.1f}% (of {combined_dim})")
    print_flush(f"Val PPL: {best_ppl:.1f}")
    print_flush(f"Val Acc: {best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")
    print_flush("=" * 70)

    # 結果を保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, "results.txt")
        with open(result_file, 'w') as f:
            f.write("Cascade Context Experiment Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Samples: {num_samples}\n")
            f.write(f"Context dim per block: {context_dim}\n")
            f.write(f"Num context blocks: {num_context_blocks}\n")
            f.write(f"Combined context dim: {combined_dim}\n\n")
            f.write(f"Train tokens: {num_train_tokens:,}\n")
            f.write(f"Val tokens: {num_val_tokens:,}\n\n")
            for i in range(num_context_blocks):
                f.write(f"Phase 1[{i}] time: {phase1_times[i]:.1f}s\n")
                f.write(f"Phase 1[{i}] conv: {phase1_stats[i].get('convergence_rate', 0)*100:.0f}%\n")
            f.write(f"Cache collection time: {cache_time:.1f}s\n")
            f.write(f"Phase 2 time: {phase2_time:.1f}s\n")
            f.write(f"Best epoch: {best_epoch}\n\n")
            f.write(f"Val PPL: {best_ppl:.2f}\n")
            f.write(f"Val Acc: {best_acc*100:.2f}%\n")
            f.write(f"Val ER: {val_er_pct:.1f}%\n")
            f.write(f"Total time: {total_time:.1f}s\n")
        print_flush(f"\nResults saved to: {result_file}")

    # メモリ解放
    del model, phase2_trainer
    del train_context_cache, val_context_cache
    data_provider.close()
    clear_gpu_cache(device)

    return {
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'val_er_pct': val_er_pct,
        'total_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Cascade Context Experiment')
    parser.add_argument('--samples', '-s', type=int, default=2000, help='Number of samples')
    parser.add_argument('--context-dim', '-c', type=int, default=500, help='Context dim per block')
    parser.add_argument('--num-blocks', '-n', type=int, default=2, help='Number of context blocks (1, 2, 3, ...)')
    parser.add_argument('--output-dir', '-o', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or f"importants/logs/{timestamp}_cascade_context"

    combined_dim = args.context_dim * args.num_blocks

    print_flush("=" * 70)
    print_flush("CASCADE CONTEXT EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Context dim per block: {args.context_dim}")
    print_flush(f"Num context blocks: {args.num_blocks}")
    print_flush(f"Combined context dim: {combined_dim}")
    print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    run_cascade_context_experiment(
        num_samples=args.samples,
        context_dim=args.context_dim,
        num_context_blocks=args.num_blocks,
        seed=args.seed,
        output_dir=output_dir,
    )

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == '__main__':
    main()
