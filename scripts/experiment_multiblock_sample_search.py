#!/usr/bin/env python3
"""
Multi-Block Sample Size Search Experiment

2ブロック（カスケード連結）でのサンプル数探索実験。
最後に指数減衰モデルでのフィッティングも自動実行。

使用方法:
  python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 1600 -c 256
  python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 3200 -c 256
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Any, List

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit

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


# ============================================================
# Model Classes (from experiment_cascade_context.py)
# ============================================================

class CascadeContextLLM(nn.Module):
    """Cascade Context LLM - 2ブロック版"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_dim: int,
        num_context_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_context_blocks = num_context_blocks
        self.combined_context_dim = context_dim * num_context_blocks

        self._load_pretrained_embeddings()
        self.embed_norm = nn.LayerNorm(embed_dim)

        self.context_blocks = nn.ModuleList([
            ContextBlock(context_dim=context_dim, embed_dim=embed_dim)
            for _ in range(num_context_blocks)
        ])

        self.token_block = TokenBlock(
            context_dim=self.combined_context_dim,
            embed_dim=embed_dim,
        )

        self.token_output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_output.weight = self.token_embedding.weight
        print_flush("✓ Weight Tying: token_output shares weights with token_embedding")

    def _load_pretrained_embeddings(self) -> None:
        self.token_embedding = load_pretrained_gpt2_embeddings(
            self.vocab_size, self.embed_dim, freeze=True
        )

    def forward_context(self, block_idx: int, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        return self.context_blocks[block_idx](context, token_embeds)

    def forward_token(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        return self.token_block(context, token_embeds)

    def freeze_context_block(self, block_idx: int) -> None:
        for param in self.context_blocks[block_idx].parameters():
            param.requires_grad = False

    def freeze_all_context_blocks(self) -> None:
        for i in range(self.num_context_blocks):
            self.freeze_context_block(i)
        print_flush(f"✓ All {self.num_context_blocks} ContextBlocks frozen")

    def num_params(self) -> Dict[str, int]:
        embedding_params = self.token_embedding.weight.numel()
        embed_norm_params = count_parameters(self.embed_norm)
        total_context_params = sum(count_parameters(block) for block in self.context_blocks)
        token_block_params = count_parameters(self.token_block)
        total = embedding_params + embed_norm_params + total_context_params + token_block_params
        return {
            'embedding': embedding_params,
            'embed_norm': embed_norm_params,
            'token_block': token_block_params,
            'total': total,
            'total_context_blocks': total_context_params,
        }


class SingleContextWrapper(nn.Module):
    """Phase 1 用ラッパー"""

    def __init__(self, cascade_model: CascadeContextLLM, block_idx: int = 0):
        super().__init__()
        self.cascade_model = cascade_model
        self.block_idx = block_idx
        self.token_embedding = cascade_model.token_embedding
        self.embed_norm = cascade_model.embed_norm
        self.context_dim = cascade_model.context_dim
        self.embed_dim = cascade_model.embed_dim
        self.vocab_size = cascade_model.vocab_size
        self.context_block = cascade_model.context_blocks[block_idx]

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        return self.context_block(context, token_embeds)


# ============================================================
# Phase 2 Training
# ============================================================

def train_phase2(
    model: CascadeContextLLM,
    train_token_ids: torch.Tensor,
    val_token_ids: torch.Tensor,
    train_context_cache: torch.Tensor,
    train_token_embeds: torch.Tensor,
    val_context_cache: torch.Tensor,
    val_token_embeds: torch.Tensor,
    config: Phase2ConfigWrapper,
    device: torch.device,
) -> Dict[str, Any]:
    """Phase 2: TokenBlock 学習"""
    model.freeze_all_context_blocks()
    model.token_embedding.weight.requires_grad = False
    print_flush("✓ Embedding frozen")

    trainable_params = [p for p in model.token_block.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print_flush(f"✓ Training TokenBlock only: {trainable_count:,}/{total_params:,} parameters")

    optimizer = torch.optim.Adam(trainable_params, lr=config.phase2_learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_train = len(train_context_cache)
    num_val = len(val_context_cache)
    batch_size = config.phase2_batch_size

    train_labels = train_token_ids[1:].to(device)
    val_labels = val_token_ids[1:].to(device)

    print_flush(f"\n[Phase 2] {num_train:,} train / {num_val:,} val tokens, {config.phase2_epochs} epochs")

    history: Dict[str, Any] = {
        'train_loss': [], 'train_ppl': [],
        'val_loss': [], 'val_ppl': [], 'val_acc': [],
    }
    best_val_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.phase2_epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss_sum = 0.0

        for start_idx in range(0, num_train, batch_size):
            end_idx = min(start_idx + batch_size, num_train)

            batch_context = train_context_cache[start_idx:end_idx].to(device)
            batch_token = train_token_embeds[start_idx:end_idx].to(device)
            batch_labels = train_labels[start_idx:end_idx]

            optimizer.zero_grad()
            token_out = model.forward_token(batch_context, batch_token)
            logits = model.token_output(token_out)
            loss = criterion(logits, batch_labels)
            loss.backward()

            if config.phase2_gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.phase2_gradient_clip)
            optimizer.step()

            train_loss_sum += loss.item() * (end_idx - start_idx)

        train_loss = train_loss_sum / num_train
        train_ppl = min(torch.exp(torch.tensor(train_loss)).item(), 1e7)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0

        with torch.no_grad():
            for start_idx in range(0, num_val, batch_size):
                end_idx = min(start_idx + batch_size, num_val)

                batch_context = val_context_cache[start_idx:end_idx].to(device)
                batch_token = val_token_embeds[start_idx:end_idx].to(device)
                batch_labels = val_labels[start_idx:end_idx]

                token_out = model.forward_token(batch_context, batch_token)
                logits = model.token_output(token_out)
                loss = criterion(logits, batch_labels)

                val_loss_sum += loss.item() * (end_idx - start_idx)
                val_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()

        val_loss = val_loss_sum / num_val
        val_ppl = min(torch.exp(torch.tensor(val_loss)).item(), 1e7)
        val_acc = val_correct / num_val

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        history['val_acc'].append(val_acc)

        improved = ""
        if val_ppl < best_val_ppl - config.phase2_min_ppl_improvement:
            best_val_ppl = val_ppl
            best_epoch = epoch
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        print_flush(f"    Epoch {epoch}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                    f"acc={val_acc*100:.1f}% [{epoch_time:.1f}s]{improved}")

        if patience_counter >= config.phase2_patience:
            print_flush(f"    → Early stop at epoch {epoch}")
            break

    print_flush(f"    Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}, acc={history['val_acc'][best_epoch-1]*100:.1f}%")

    history['best_epoch'] = best_epoch
    history['best_val_ppl'] = best_val_ppl

    return history


# ============================================================
# Exponential Decay Fitting
# ============================================================

def fit_exp_decay(samples: np.ndarray, ppls: np.ndarray) -> Dict[str, Any]:
    """指数減衰モデルでフィッティング"""

    def exp_decay(n, ppl_min, A, b, c):
        return ppl_min + A * np.exp(-b * (n ** c))

    try:
        popt, pcov = curve_fit(
            exp_decay, samples, ppls,
            p0=[100, 300, 0.01, 0.5],
            bounds=([0, 0, 0, 0], [200, 2000, 1, 1]),
            maxfev=10000
        )
        ppl_min, A, b, c = popt

        pred = exp_decay(samples, *popt)
        ss_res = np.sum((ppls - pred) ** 2)
        ss_tot = np.sum((ppls - np.mean(ppls)) ** 2)
        r2 = 1 - ss_res / ss_tot

        return {
            'success': True,
            'ppl_min': ppl_min,
            'A': A,
            'b': b,
            'c': c,
            'r2': r2,
            'predictions': pred,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================
# Main Search Function
# ============================================================

def run_multiblock_sample_search(
    context_dim: int = 256,
    num_blocks: int = 2,
    start_samples: int = 200,
    end_samples: int = 1600,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Multi-block sample size search"""

    set_seed(seed)

    # サンプル数リスト（倍増）
    sample_sizes: List[int] = []
    current = start_samples
    while current <= end_samples:
        sample_sizes.append(current)
        current *= 2

    combined_dim = context_dim * num_blocks

    print_flush("=" * 70)
    print_flush("MULTI-BLOCK SAMPLE SIZE SEARCH")
    print_flush("=" * 70)
    print_flush(f"Context dim per block: {context_dim}")
    print_flush(f"Num blocks: {num_blocks}")
    print_flush(f"Combined context dim: {combined_dim}")
    print_flush(f"Sample sizes: {sample_sizes}")
    if output_dir:
        print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    base_config = Config()

    results: List[Dict[str, Any]] = []
    best_ppl = float('inf')
    best_samples = start_samples

    for num_samples in sample_sizes:
        print_flush(f"\n{'='*70}")
        print_flush(f"[SAMPLES={num_samples}] Starting experiment ({num_blocks} blocks)...")
        print_flush("=" * 70)

        experiment_start = time.time()

        # データ読み込み
        print_flush(f"\nLoading data ({num_samples} samples)...")
        data_config = DataConfig.from_base(base_config, num_samples=num_samples)
        data_provider = MemoryDataProvider(data_config)
        train_token_ids, val_token_ids = data_provider.load_data()
        train_token_ids = train_token_ids.to(device)
        val_token_ids = val_token_ids.to(device)

        num_train_tokens = len(train_token_ids)
        num_val_tokens = len(val_token_ids)
        print_flush(f"Data: {num_train_tokens:,} train, {num_val_tokens:,} val tokens")

        # N分割
        train_data_splits: List[torch.Tensor] = []
        split_size = num_train_tokens // num_blocks
        for i in range(num_blocks):
            start_idx = i * split_size
            end_idx = num_train_tokens if i == num_blocks - 1 else (i + 1) * split_size + 1
            train_data_splits.append(train_token_ids[start_idx:end_idx])

        # モデル作成
        print_flush(f"\nCreating CascadeContextLLM (cd={context_dim}x{num_blocks}={combined_dim})...")
        model = CascadeContextLLM(
            vocab_size=base_config.vocab_size,
            embed_dim=base_config.embed_dim,
            context_dim=context_dim,
            num_context_blocks=num_blocks,
        )
        model.to(device)

        config_wrapper = Phase1ConfigWrapper(base_config, context_dim, patience=2)

        # Phase 1: 各ブロック学習
        phase1_times = []
        phase1_convs = []

        for block_idx in range(num_blocks):
            print_flush(f"\n[Phase 1-{block_idx}] Training ContextBlock {block_idx}...")
            wrapper = SingleContextWrapper(model, block_idx=block_idx)
            trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

            phase_start = time.time()
            _ = trainer.train(
                train_data_splits[block_idx],
                label=f"Context{block_idx}",
                return_all_layers=True,
            )
            phase_time = time.time() - phase_start

            stats = trainer._training_stats
            phase1_times.append(phase_time)
            phase1_convs.append(stats.get('convergence_rate', 0))

            iters = stats.get('iterations', 0)
            conv = stats.get('convergence_rate', 0)
            print_flush(f"Phase 1-{block_idx}: {phase_time:.1f}s, {iters} iter, conv={conv*100:.0f}%")

            model.freeze_context_block(block_idx)

        phase1_total_time = sum(phase1_times)

        # Phase 2 Prep
        print_flush("\n[Phase 2 Prep] Collecting context cache...")
        cache_start = time.time()

        cache_dir = f"/tmp/multiblock_cache_{num_samples}_{datetime.now().strftime('%H%M%S')}"
        train_num, _, train_chunks = collect_multiblock_cache_to_files(
            model, train_token_ids, device, cache_dir, prefix="train"
        )
        val_num, _, val_chunks = collect_multiblock_cache_to_files(
            model, val_token_ids, device, cache_dir, prefix="val"
        )

        train_dataset = ChunkedCacheDataset(train_chunks)
        val_dataset = ChunkedCacheDataset(val_chunks)
        train_context_cache, train_token_embeds = train_dataset.get_all_data()
        val_context_cache, val_token_embeds = val_dataset.get_all_data()

        cache_time = time.time() - cache_start
        print_flush(f"Cache collection: {cache_time:.1f}s")

        # Effective Rank
        er_analysis = analyze_fixed_points(val_context_cache, label="Val", verbose=False)
        val_er = er_analysis.get('effective_rank', 0)
        val_er_pct = val_er / combined_dim * 100
        print_flush(f"Effective Rank: Val={val_er_pct:.1f}% ({val_er:.1f}/{combined_dim})")

        # Phase 2
        print_flush("\n[Phase 2] Training TokenBlock...")
        phase2_config = Phase2ConfigWrapper(base_config)

        phase2_start = time.time()
        history = train_phase2(
            model,
            train_token_ids, val_token_ids,
            train_context_cache, train_token_embeds,
            val_context_cache, val_token_embeds,
            phase2_config, device,
        )
        phase2_time = time.time() - phase2_start

        total_time = time.time() - experiment_start

        val_ppl = history['best_val_ppl']
        val_acc = history['val_acc'][history['best_epoch'] - 1]

        result = {
            'num_samples': num_samples,
            'num_train_tokens': num_train_tokens,
            'val_ppl': val_ppl,
            'val_acc': val_acc,
            'val_er': val_er,
            'val_er_pct': val_er_pct,
            'phase1_time': phase1_total_time,
            'phase2_time': phase2_time,
            'total_time': total_time,
        }
        results.append(result)

        print_flush(f"\n[Result] samples={num_samples}: PPL={val_ppl:.1f}, Acc={val_acc*100:.1f}%, "
                    f"ER={val_er_pct:.1f}%, Time={total_time:.1f}s")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_samples = num_samples
            print_flush(f"  ★ New best! samples={num_samples}, PPL={val_ppl:.1f}")

        # メモリ解放
        del model, train_context_cache, val_context_cache
        del train_token_embeds, val_token_embeds
        del train_dataset, val_dataset
        del train_token_ids, val_token_ids
        data_provider.close()
        clear_gpu_cache(device)

        # キャッシュディレクトリ削除
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    # ========== 指数減衰フィッティング ==========
    print_flush("\n" + "=" * 70)
    print_flush("EXPONENTIAL DECAY MODEL FITTING")
    print_flush("=" * 70)

    samples_arr = np.array([r['num_samples'] for r in results])
    ppls_arr = np.array([r['val_ppl'] for r in results])

    fit_result = fit_exp_decay(samples_arr, ppls_arr)

    if fit_result['success']:
        print_flush(f"\nModel: PPL = PPL_min + A × exp(-b × n^c)")
        print_flush(f"  PPL_min = {fit_result['ppl_min']:.2f}")
        print_flush(f"  A = {fit_result['A']:.2f}")
        print_flush(f"  b = {fit_result['b']:.6f}")
        print_flush(f"  c = {fit_result['c']:.4f}")
        print_flush(f"  R² = {fit_result['r2']:.6f}")

        print_flush("\nPrediction vs Actual:")
        for s, actual, pred in zip(samples_arr, ppls_arr, fit_result['predictions']):
            err = (pred - actual) / actual * 100
            print_flush(f"  {int(s):>6} samples: actual={actual:.1f}, pred={pred:.1f} ({err:+.1f}%)")

        print_flush(f"\n★ Theoretical limit (PPL_min): {fit_result['ppl_min']:.1f}")
    else:
        print_flush(f"Fitting failed: {fit_result['error']}")

    # 最終サマリー
    print_flush("\n" + "=" * 70)
    print_flush(f"SUMMARY - Multi-Block Sample Size Search ({num_blocks} blocks)")
    print_flush("=" * 70)
    print_flush(f"Context dim per block: {context_dim}")
    print_flush(f"Combined context dim: {combined_dim}")
    print_flush(f"Sample sizes: {sample_sizes}")
    print_flush("\nResults:")
    print_flush(f"{'Samples':>8} | {'Tokens':>12} | {'PPL':>8} | {'Acc':>6} | {'ER%':>6} | {'Time':>8}")
    print_flush("-" * 60)
    for r in results:
        marker = " ★" if r['num_samples'] == best_samples else ""
        print_flush(f"{r['num_samples']:>8} | {r['num_train_tokens']:>12,} | {r['val_ppl']:>8.1f} | "
                    f"{r['val_acc']*100:>5.1f}% | {r['val_er_pct']:>5.1f}% | {r['total_time']:>7.1f}s{marker}")
    print_flush("-" * 60)
    print_flush(f"\n★ Best: samples={best_samples}, PPL={best_ppl:.1f}")

    if fit_result['success']:
        print_flush(f"\n★ Exp Decay PPL_min: {fit_result['ppl_min']:.1f} (R²={fit_result['r2']:.4f})")

    print_flush("=" * 70)

    # 結果を保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, "results.txt")
        with open(result_file, 'w') as f:
            f.write(f"Multi-Block Sample Size Search Results ({num_blocks} blocks)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Context dim per block: {context_dim}\n")
            f.write(f"Combined context dim: {combined_dim}\n")
            f.write(f"Sample sizes: {sample_sizes}\n\n")
            f.write(f"{'Samples':>8} | {'Tokens':>12} | {'PPL':>8} | {'Acc':>6} | {'ER%':>6} | {'Time':>8}\n")
            f.write("-" * 60 + "\n")
            for r in results:
                marker = " *" if r['num_samples'] == best_samples else ""
                f.write(f"{r['num_samples']:>8} | {r['num_train_tokens']:>12,} | {r['val_ppl']:>8.1f} | "
                        f"{r['val_acc']*100:>5.1f}% | {r['val_er_pct']:>5.1f}% | {r['total_time']:>7.1f}s{marker}\n")
            f.write("-" * 60 + "\n")
            f.write(f"\nBest: samples={best_samples}, PPL={best_ppl:.1f}\n")

            if fit_result['success']:
                f.write(f"\n\nExponential Decay Fitting:\n")
                f.write(f"  PPL = PPL_min + A × exp(-b × n^c)\n")
                f.write(f"  PPL_min = {fit_result['ppl_min']:.2f}\n")
                f.write(f"  A = {fit_result['A']:.2f}\n")
                f.write(f"  b = {fit_result['b']:.6f}\n")
                f.write(f"  c = {fit_result['c']:.4f}\n")
                f.write(f"  R² = {fit_result['r2']:.6f}\n")

        print_flush(f"\nResults saved to: {result_file}")

    return {
        'results': results,
        'best_samples': best_samples,
        'best_ppl': best_ppl,
        'fit_result': fit_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Multi-Block Sample Size Search')
    parser.add_argument('--context-dim', '-c', type=int, default=256,
                        help='Context dimension per block (default: 256)')
    parser.add_argument('--num-blocks', '-n', type=int, default=2,
                        help='Number of context blocks (default: 2)')
    parser.add_argument('--start', type=int, default=200,
                        help='Starting sample size (default: 200)')
    parser.add_argument('--end', type=int, default=1600,
                        help='Ending sample size (default: 1600)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    # 出力ディレクトリ
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"importants/logs/{timestamp}_multiblock_{args.num_blocks}b_sample_search"

    run_multiblock_sample_search(
        context_dim=args.context_dim,
        num_blocks=args.num_blocks,
        start_samples=args.start,
        end_samples=args.end,
        seed=args.seed,
        output_dir=output_dir,
    )

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == "__main__":
    main()
