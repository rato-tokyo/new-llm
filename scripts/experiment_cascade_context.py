#!/usr/bin/env python3
"""
Cascade Context 実験スクリプト

カスケード方式: ContextBlock A → B の順で学習し、
B の学習時は A の出力を固定入力として使用。

アーキテクチャ（1層固定）:
  Phase 1A:
    - ContextBlock A を全データで学習
    - 出力: context_a (キャッシュ)

  Phase 1B:
    - ContextBlock B を全データで学習
    - 入力: context_a (Phase 1Aの出力、固定)
    - 出力: context_b (キャッシュ)

  Phase 2:
    - context_a と context_b を連結 → cd=1000
    - TokenBlock (cd=1000入力) で学習

使用方法:
  python3 scripts/experiment_cascade_context.py
  python3 scripts/experiment_cascade_context.py -s 2000
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

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
from config.experiment import DataConfig


class CascadeContextLLM(nn.Module):
    """
    Cascade Context LLM - 2つのContextBlockをカスケード接続（1層固定）

    Phase 1A: ContextBlock A を学習（入力: ゼロ初期化context）
    Phase 1B: ContextBlock B を学習（入力: A の出力、固定）
    Phase 2: concat(context_a, context_b) で TokenBlock を学習

    Args:
        vocab_size: 語彙サイズ
        embed_dim: トークン埋め込み次元
        context_dim: 各ContextBlockの出力次元（連結後は2倍）
        num_input_tokens: 入力トークン数
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_dim: int,
        num_input_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.combined_context_dim = context_dim * 2
        self.num_input_tokens = num_input_tokens

        # Token Embeddings (GPT-2 pretrained)
        self._load_pretrained_embeddings()
        self.embed_norm = nn.LayerNorm(embed_dim)

        # 2つのContextBlock（各1層）
        self.context_block_a = ContextBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )
        self.context_block_b = ContextBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        # TokenBlock（連結されたcontext用、cd=1000、1層）
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

    def forward_context_a(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """ContextBlock A の順伝搬"""
        return self.context_block_a(context, token_embeds)

    def forward_context_b(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """ContextBlock B の順伝搬"""
        return self.context_block_b(context, token_embeds)

    def forward_token(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """TokenBlock の順伝搬（1層固定、contextは連結済み）"""
        return self.token_block(context, token_embeds)

    def freeze_context_blocks(self) -> None:
        """両方のContextBlockをfreeze"""
        for param in self.context_block_a.parameters():
            param.requires_grad = False
        for param in self.context_block_b.parameters():
            param.requires_grad = False
        print_flush("✓ Both ContextBlocks frozen")

    def num_params(self) -> dict:
        """パラメータ数を返す"""
        embedding_params = self.token_embedding.weight.numel()
        embed_norm_params = sum(p.numel() for p in self.embed_norm.parameters())
        context_a_params = sum(p.numel() for p in self.context_block_a.parameters())
        context_b_params = sum(p.numel() for p in self.context_block_b.parameters())
        token_block_params = sum(p.numel() for p in self.token_block.parameters())

        total = embedding_params + embed_norm_params + context_a_params + context_b_params + token_block_params

        return {
            'embedding': embedding_params,
            'embed_norm': embed_norm_params,
            'context_block_a': context_a_params,
            'context_block_b': context_b_params,
            'token_block': token_block_params,
            'total': total,
        }


class SingleContextWrapper(nn.Module):
    """
    Phase 1 用: ContextBlock のラッパー
    MemoryPhase1Trainer と互換性を持たせる。
    """

    def __init__(self, cascade_model: CascadeContextLLM, block: str = 'a'):
        super().__init__()
        self.cascade_model = cascade_model

        # Phase1Trainerが期待するプロパティ
        self.token_embedding = cascade_model.token_embedding
        self.embed_norm = cascade_model.embed_norm
        self.context_dim = cascade_model.context_dim
        self.embed_dim = cascade_model.embed_dim
        self.num_input_tokens = cascade_model.num_input_tokens
        self.vocab_size = cascade_model.vocab_size

        if block == 'a':
            self.context_block = cascade_model.context_block_a
        else:
            self.context_block = cascade_model.context_block_b

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
        self.model.freeze_context_blocks()
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


def collect_context_a_cache(
    model: CascadeContextLLM,
    token_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ContextBlock A のキャッシュを収集"""
    model.eval()
    num_tokens = len(token_ids) - 1

    with torch.no_grad():
        all_embeds = model.token_embedding(token_ids.to(device))
        all_embeds = model.embed_norm(all_embeds)
        all_embeds_cpu = all_embeds.cpu()
        del all_embeds
        clear_gpu_cache(device)

    input_embeds = all_embeds_cpu[:-1]
    del all_embeds_cpu

    context_a_cache = torch.zeros(num_tokens, model.context_dim, device='cpu', dtype=torch.float32)

    print_flush(f"    Collecting context_a cache ({num_tokens:,} tokens)...")
    collect_start = time.time()

    context_a = torch.zeros(1, model.context_dim, device=device)

    with torch.no_grad():
        for i in range(num_tokens):
            token_embed = input_embeds[i:i+1].to(device)
            new_context_a = model.forward_context_a(context_a, token_embed)
            context_a_cache[i] = new_context_a.cpu()
            context_a = new_context_a

            if (i + 1) % 100000 == 0:
                print_flush(f"      {i+1:,}/{num_tokens:,} tokens processed...")

    print_flush(f"    Context A cache collected [{time.time() - collect_start:.1f}s]")
    clear_gpu_cache(device)

    return context_a_cache, input_embeds


def collect_context_b_cache_with_fixed_a(
    model: CascadeContextLLM,
    token_embeds: torch.Tensor,
    context_a_cache: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """ContextBlock B のキャッシュを収集（context_a を固定入力として使用）"""
    model.eval()
    num_tokens = len(token_embeds)

    context_b_cache = torch.zeros(num_tokens, model.context_dim, device='cpu', dtype=torch.float32)

    print_flush(f"    Collecting context_b cache with fixed context_a ({num_tokens:,} tokens)...")
    collect_start = time.time()

    with torch.no_grad():
        for i in range(num_tokens):
            token_embed = token_embeds[i:i+1].to(device)
            input_context = context_a_cache[i:i+1].to(device)
            new_context_b = model.forward_context_b(input_context, token_embed)
            context_b_cache[i] = new_context_b.cpu()

            if (i + 1) % 100000 == 0:
                print_flush(f"      {i+1:,}/{num_tokens:,} tokens processed...")

    print_flush(f"    Context B cache collected [{time.time() - collect_start:.1f}s]")
    clear_gpu_cache(device)

    return context_b_cache


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
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Cascade Context 実験を実行"""

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
    combined_dim = context_dim * 2
    print_flush(f"\nCreating CascadeContextLLM (cd={context_dim}x2={combined_dim})...")
    model = CascadeContextLLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=context_dim,
        num_input_tokens=base_config.num_input_tokens,
    )
    model.to(device)

    params = model.num_params()
    print_flush(f"Parameters: {params['total']:,} total")
    print_flush(f"  ContextBlock A: {params['context_block_a']:,}")
    print_flush(f"  ContextBlock B: {params['context_block_b']:,}")
    print_flush(f"  TokenBlock: {params['token_block']:,}")

    # ========== Phase 1A: ContextBlock A の学習 ==========
    print_flush("\n[Phase 1A] Training ContextBlock A on full data...")
    wrapper_a = SingleContextWrapper(model, block='a')

    config_wrapper = Phase1ConfigWrapper(base_config, context_dim)
    trainer_a = MemoryPhase1Trainer(wrapper_a, config_wrapper, device)

    phase1a_start = time.time()
    result_a = trainer_a.train(train_token_ids, label="ContextA", return_all_layers=True)
    phase1a_time = time.time() - phase1a_start

    stats_a = trainer_a._training_stats
    print_flush(f"Phase 1A: {phase1a_time:.1f}s, {stats_a.get('iterations', 0)} iter, "
                f"conv={stats_a.get('convergence_rate', 0)*100:.0f}%")

    # Context A のキャッシュを取得
    assert result_a.cache is not None
    assert result_a.token_embeds is not None
    train_context_a_cache = result_a.cache
    train_token_embeds = result_a.token_embeds

    # ContextBlock A を freeze
    for param in model.context_block_a.parameters():
        param.requires_grad = False
    print_flush("✓ ContextBlock A frozen")

    # ========== Phase 1B: ContextBlock B の学習 ==========
    print_flush("\n[Phase 1B] Training ContextBlock B on full data...")
    print_flush("  Note: ContextBlock B will use context_a as input during cache collection")

    wrapper_b = SingleContextWrapper(model, block='b')
    trainer_b = MemoryPhase1Trainer(wrapper_b, config_wrapper, device)

    phase1b_start = time.time()
    _ = trainer_b.train(train_token_ids, label="ContextB", return_all_layers=True)
    phase1b_time = time.time() - phase1b_start

    stats_b = trainer_b._training_stats
    print_flush(f"Phase 1B: {phase1b_time:.1f}s, {stats_b.get('iterations', 0)} iter, "
                f"conv={stats_b.get('convergence_rate', 0)*100:.0f}%")

    # ContextBlock B を freeze
    for param in model.context_block_b.parameters():
        param.requires_grad = False
    print_flush("✓ ContextBlock B frozen")

    phase1_total_time = phase1a_time + phase1b_time

    # ========== キャッシュ収集: context_a を入力として context_b を計算 ==========
    print_flush("\n[Cache Collection] Collecting context_b with fixed context_a...")
    cache_start = time.time()

    # Train data
    train_context_b_cache = collect_context_b_cache_with_fixed_a(
        model, train_token_embeds, train_context_a_cache, device
    )

    # Val data
    val_context_a_cache, val_token_embeds = collect_context_a_cache(
        model, val_token_ids, device
    )
    val_context_b_cache = collect_context_b_cache_with_fixed_a(
        model, val_token_embeds, val_context_a_cache, device
    )

    cache_time = time.time() - cache_start
    print_flush(f"Cache collection: {cache_time:.1f}s")

    # 連結
    train_context_cache = torch.cat([train_context_a_cache, train_context_b_cache], dim=-1)
    val_context_cache = torch.cat([val_context_a_cache, val_context_b_cache], dim=-1)

    print_flush(f"  Train cache: {train_context_cache.shape}")
    print_flush(f"  Val cache: {val_context_cache.shape}")

    # メモリ解放
    del train_context_a_cache, train_context_b_cache
    del val_context_a_cache, val_context_b_cache

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
    print_flush("SUMMARY - Cascade Context Experiment")
    print_flush("=" * 70)
    print_flush("Architecture: CascadeContextLLM (A→B cascade, 1L each)")
    print_flush(f"  ContextBlock A: cd={context_dim}, trained on full data")
    print_flush(f"  ContextBlock B: cd={context_dim}, input=context_a (fixed)")
    print_flush(f"  TokenBlock: cd={combined_dim} (concatenated)")
    print_flush(f"Parameters: {params['total']:,}")
    print_flush(f"Phase 1A: {phase1a_time:.1f}s, conv={stats_a.get('convergence_rate', 0)*100:.0f}%")
    print_flush(f"Phase 1B: {phase1b_time:.1f}s, conv={stats_b.get('convergence_rate', 0)*100:.0f}%")
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
            f.write(f"Combined context dim: {combined_dim}\n\n")
            f.write(f"Train tokens: {num_train_tokens:,}\n")
            f.write(f"Val tokens: {num_val_tokens:,}\n\n")
            f.write(f"Phase 1A time: {phase1a_time:.1f}s\n")
            f.write(f"Phase 1A conv: {stats_a.get('convergence_rate', 0)*100:.0f}%\n")
            f.write(f"Phase 1B time: {phase1b_time:.1f}s\n")
            f.write(f"Phase 1B conv: {stats_b.get('convergence_rate', 0)*100:.0f}%\n")
            f.write(f"Cache collection time: {cache_time:.1f}s\n")
            f.write(f"Phase 2 time: {phase2_time:.1f}s\n")
            f.write(f"Best epoch: {best_epoch}\n\n")
            f.write(f"Val PPL: {best_ppl:.2f}\n")
            f.write(f"Val Acc: {best_acc*100:.2f}%\n")
            f.write(f"Val ER: {val_er_pct:.1f}%\n")
            f.write(f"Total time: {total_time:.1f}s\n")
        print_flush(f"\nResults saved to: {result_file}")

    # メモリ解放
    del model, trainer_a, trainer_b, phase2_trainer
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
    parser.add_argument('--output-dir', '-o', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or f"importants/logs/{timestamp}_cascade_context"

    print_flush("=" * 70)
    print_flush("CASCADE CONTEXT EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Context dim per block: {args.context_dim}")
    print_flush(f"Combined context dim: {args.context_dim * 2}")
    print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    run_cascade_context_experiment(
        num_samples=args.samples,
        context_dim=args.context_dim,
        seed=args.seed,
        output_dir=output_dir,
    )

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == '__main__':
    main()
