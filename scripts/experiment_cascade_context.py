#!/usr/bin/env python3
"""
Multi Context 実験スクリプト（N分割方式）

N個のContextBlockを異なるデータで学習し、異なる表現を獲得する。

アーキテクチャ（1層固定、Nブロック）:
  Phase 1[i]: ContextBlock[i] を i 番目のデータ区間で学習
    - 初期入力: ゼロベクトル
    - データ: tokens[i*split:(i+1)*split]

  Phase 2:
    - 順次処理で全データのコンテキストキャッシュを収集
    - concat(context[0], ..., context[N-1]) で TokenBlock を学習

使用方法:
  python3 scripts/experiment_cascade_context.py -s 2000 -n 2  # 2ブロック（デフォルト）
  python3 scripts/experiment_cascade_context.py -s 2000 -n 4  # 4ブロック
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Any, List

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
from src.utils.cache import collect_multiblock_cache_to_files, ChunkedCacheDataset
from src.utils.embedding import load_pretrained_gpt2_embeddings
from src.config.wrappers import Phase1ConfigWrapper, Phase2ConfigWrapper
from config.experiment import DataConfig


class CascadeContextLLM(nn.Module):
    """
    Cascade Context LLM - N個のContextBlockをカスケード連結（1層固定）

    N分割方式:
    - Phase 1[i]: ContextBlock[i] を i 番目のデータ区間で学習（入力: ゼロベクトル）
    - Phase 2: concat(context[0], ..., context[N-1]) で TokenBlock を学習

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
        """GPT-2 embeddings をロード（共通ユーティリティ使用）"""
        self.token_embedding = load_pretrained_gpt2_embeddings(
            self.vocab_size, self.embed_dim, freeze=True
        )

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


def run_cascade_context_experiment(
    num_samples: int = 2000,
    context_dim: int = 500,
    num_context_blocks: int = 2,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Cascade Context 実験を実行（Dual方式: 前半/後半分割）

    Args:
        num_samples: サンプル数
        context_dim: 各ContextBlockの出力次元
        num_context_blocks: ContextBlockの数（2固定、Dual方式）
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

    # N分割方式: データをN等分
    train_data_splits: List[torch.Tensor] = []
    split_size = num_train_tokens // num_context_blocks
    split_info = []
    for i in range(num_context_blocks):
        start_idx = i * split_size
        if i == num_context_blocks - 1:
            # 最後のブロックは残り全部
            end_idx = num_train_tokens
        else:
            end_idx = (i + 1) * split_size + 1  # +1 for overlap
        train_data_splits.append(train_token_ids[start_idx:end_idx])
        split_info.append(f"Block{i}={len(train_data_splits[-1])-1:,}")
    print_flush(f"Split: {', '.join(split_info)} tokens")

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

    # ========== Phase 1: N分割方式で各ブロックを学習 ==========
    train_context_caches: List[torch.Tensor] = []
    phase1_times: List[float] = []
    phase1_stats: List[Dict[str, Any]] = []

    for block_idx in range(num_context_blocks):
        print_flush(f"\n[Phase 1-{block_idx}] Training ContextBlock {block_idx} on split {block_idx}...")
        wrapper = SingleContextWrapper(model, block_idx=block_idx)
        trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

        phase_start = time.time()
        result = trainer.train(
            train_data_splits[block_idx],
            label=f"Context{block_idx}",
            return_all_layers=True,
        )
        phase_time = time.time() - phase_start

        assert result.cache is not None
        train_context_caches.append(result.cache)

        stats = trainer._training_stats
        phase1_times.append(phase_time)
        phase1_stats.append(stats)

        print_flush(f"Phase 1-{block_idx}: {phase_time:.1f}s, {stats.get('iterations', 0)} iter, "
                    f"conv={stats.get('convergence_rate', 0)*100:.0f}%")

        model.freeze_context_block(block_idx)
        print_flush(f"✓ ContextBlock {block_idx} frozen")

    phase1_total_time = sum(phase1_times)

    # Phase 1で使ったキャッシュを解放
    del train_context_caches
    clear_gpu_cache(device)

    # ========== Phase 2 Prep: チャンク単位でキャッシュ収集 ==========
    # output_dirがNoneの場合はデフォルトを設定
    if output_dir is None:
        output_dir = f"importants/logs/temp_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cache_dir = os.path.join(output_dir, "cache_chunks")
    print_flush(f"\n[Phase 2 Prep] Collecting multi context cache (chunked, {num_context_blocks} blocks)...")
    cache_start = time.time()

    # チャンク単位でキャッシュ収集（メモリ効率が良い、GPU最適化済み）
    train_num_tokens, _, train_chunk_files = collect_multiblock_cache_to_files(
        model, train_token_ids, device, cache_dir, prefix="train"
    )
    val_num_tokens, _, val_chunk_files = collect_multiblock_cache_to_files(
        model, val_token_ids, device, cache_dir, prefix="val"
    )

    cache_time = time.time() - cache_start
    print_flush(f"Cache collection: {cache_time:.1f}s")
    print_flush(f"  Train: {train_num_tokens:,} tokens, {len(train_chunk_files)} chunks")
    print_flush(f"  Val: {val_num_tokens:,} tokens, {len(val_chunk_files)} chunks")

    # Effective Rank計算（Valデータをロードして計算）
    val_dataset = ChunkedCacheDataset(val_chunk_files)
    val_context_cache, val_token_embeds = val_dataset.get_all_data()
    val_metrics = analyze_fixed_points(val_context_cache, label="Val", verbose=False)
    val_er = val_metrics['effective_rank']
    val_er_pct = val_er / combined_dim * 100
    print_flush(f"Effective Rank: Val={val_er_pct:.1f}%")

    # ========== Phase 2: TokenBlock の学習 ==========
    print_flush(f"\n[Phase 2] Training TokenBlock with concatenated context (cd={combined_dim})...")

    # Trainデータもロード（Phase 2で使用）
    train_dataset = ChunkedCacheDataset(train_chunk_files)
    train_context_cache, train_token_embeds = train_dataset.get_all_data()

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

    # Phase 2後にキャッシュを解放
    del train_context_cache, train_token_embeds
    del val_context_cache, val_token_embeds
    del train_dataset, val_dataset
    clear_gpu_cache(device)

    best_epoch = history['best_epoch']
    best_ppl = history['val_ppl'][best_epoch - 1]
    best_acc = history['val_acc'][best_epoch - 1]

    total_time = phase1_total_time + cache_time + phase2_time

    print_flush(f"\nPhase 2: {phase2_time:.1f}s, Best epoch {best_epoch}")
    print_flush(f"Result: PPL={best_ppl:.1f}, Acc={best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")

    # ========== 結果サマリー ==========
    print_flush("\n" + "=" * 70)
    print_flush(f"SUMMARY - Multi Context Experiment ({num_context_blocks} blocks)")
    print_flush("=" * 70)
    print_flush(f"Architecture: CascadeContextLLM ({num_context_blocks} blocks, 1L each)")
    for i in range(num_context_blocks):
        block_tokens = len(train_data_splits[i]) - 1
        print_flush(f"  ContextBlock {i}: cd={context_dim}, trained on {block_tokens:,} tokens")
    print_flush(f"  TokenBlock: cd={combined_dim} (concatenated)")
    print_flush(f"Parameters: {params['total']:,}")
    for i in range(num_context_blocks):
        print_flush(f"Phase 1-{i}: {phase1_times[i]:.1f}s, conv={phase1_stats[i].get('convergence_rate', 0)*100:.0f}%")
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
    data_provider.close()
    clear_gpu_cache(device)

    # チャンクファイルを削除（オプション: コメントアウトで残す）
    # import shutil
    # shutil.rmtree(cache_dir, ignore_errors=True)

    return {
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'val_er_pct': val_er_pct,
        'total_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Multi Context Experiment (N分割方式)')
    parser.add_argument('--samples', '-s', type=int, default=2000, help='Number of samples')
    parser.add_argument('--context-dim', '-c', type=int, default=500, help='Context dim per block')
    parser.add_argument('--num-blocks', '-n', type=int, default=2, help='Number of ContextBlocks')
    parser.add_argument('--output-dir', '-o', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    num_context_blocks = args.num_blocks

    # 出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or f"importants/logs/{timestamp}_multi_context_{num_context_blocks}blocks"

    combined_dim = args.context_dim * num_context_blocks

    print_flush("=" * 70)
    print_flush("MULTI CONTEXT EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Num ContextBlocks: {num_context_blocks}")
    print_flush(f"Context dim per block: {args.context_dim}")
    print_flush(f"Combined context dim: {combined_dim}")
    print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    run_cascade_context_experiment(
        num_samples=args.samples,
        context_dim=args.context_dim,
        num_context_blocks=num_context_blocks,
        seed=args.seed,
        output_dir=output_dir,
    )

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == '__main__':
    main()
