#!/usr/bin/env python3
"""
Sample Size 探索実験スクリプト

context_dim を固定し、サンプル数を変化させてPPLの変化を観察する。

特徴:
- context_dim 固定（デフォルト: 256）
- サンプル数を倍増（100, 200, 400, 800, 1600, ...）
- 毎回新規モデル作成

使用方法:
  # 基本: 100から1600まで倍増
  python3 scripts/experiment_sample_size_search.py

  # カスタム: 開始・終了・context_dim指定
  python3 scripts/experiment_sample_size_search.py --start 100 --end 3200 --context-dim 300
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
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed
from src.utils.initialization import count_parameters
from src.utils.cache import collect_context_cache_sequential, collect_token_embeds_chunked
from src.utils.embedding import load_pretrained_gpt2_embeddings
from src.config.wrappers import Phase1ConfigWrapper, Phase2ConfigWrapper
from config.experiment import DataConfig


# ============================================================
# Model Classes (from experiment_context_dim_search.py)
# ============================================================

class SimpleContextBlock(nn.Module):
    """シンプルな ContextBlock"""

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
        num_input_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_input_tokens = num_input_tokens

        token_input_dim = embed_dim * num_input_tokens
        self.token_input_dim = token_input_dim

        input_dim = context_dim + token_input_dim
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, context_dim),
            nn.GELU()
        )
        self.context_norm = nn.LayerNorm(context_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        fnn_input = torch.cat([context, token_embeds], dim=-1)
        delta_context = self.fnn(fnn_input)
        new_context = self.context_norm(context + delta_context)
        return new_context

    def forward_batch(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """Batch forward pass（キャッシュ収集用）"""
        return self.forward(context, token_embeds)


class SimpleTokenBlock(nn.Module):
    """シンプルな TokenBlock"""

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
        num_input_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_input_tokens = num_input_tokens

        token_input_dim = embed_dim * num_input_tokens
        self.token_input_dim = token_input_dim

        input_dim = context_dim + token_input_dim
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU()
        )
        self.token_norm = nn.LayerNorm(embed_dim)

        self.residual_proj: Optional[nn.Linear] = None
        if token_input_dim != embed_dim:
            self.residual_proj = nn.Linear(token_input_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        fnn_input = torch.cat([context, token_embeds], dim=-1)
        delta_token = self.fnn(fnn_input)

        if self.residual_proj is not None:
            residual = self.residual_proj(token_embeds)
        else:
            residual = token_embeds

        new_token = self.token_norm(residual + delta_token)
        return new_token


class SimpleLLM(nn.Module):
    """Sample Size 探索用LLM"""

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
        self.num_input_tokens = num_input_tokens

        self._load_pretrained_embeddings()
        self.embed_norm = nn.LayerNorm(embed_dim)

        self.context_block = SimpleContextBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        self.token_block = SimpleTokenBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        self.token_output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_output.weight = self.token_embedding.weight
        print_flush("✓ Weight Tying: token_output shares weights with token_embedding")

    def _load_pretrained_embeddings(self) -> None:
        self.token_embedding = load_pretrained_gpt2_embeddings(
            self.vocab_size, self.embed_dim, freeze=True
        )

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        return self.context_block(context, token_embeds)

    def forward_token(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        return self.token_block(context, token_embeds)

    def freeze_context_block(self) -> None:
        for param in self.context_block.parameters():
            param.requires_grad = False
        print_flush("✓ ContextBlock frozen")

    def num_params(self) -> Dict[str, int]:
        embedding_params = self.token_embedding.weight.numel()
        embed_norm_params = count_parameters(self.embed_norm)
        context_block_params = count_parameters(self.context_block)
        token_block_params = count_parameters(self.token_block)
        total = embedding_params + embed_norm_params + context_block_params + token_block_params
        return {
            'embedding': embedding_params,
            'embed_norm': embed_norm_params,
            'context_block': context_block_params,
            'token_block': token_block_params,
            'total': total,
        }


class SingleContextWrapper(nn.Module):
    """Phase 1 用ラッパー"""

    def __init__(self, model: SimpleLLM):
        super().__init__()
        self.cascade_model = model
        self.token_embedding = model.token_embedding
        self.embed_norm = model.embed_norm
        self.context_dim = model.context_dim
        self.embed_dim = model.embed_dim
        self.num_input_tokens = model.num_input_tokens
        self.vocab_size = model.vocab_size
        self.context_block = model.context_block

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        return self.context_block(context, token_embeds)


# ============================================================
# Phase 2 Training
# ============================================================

def train_phase2(
    model: SimpleLLM,
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
    model.freeze_context_block()

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
# Model Creation
# ============================================================

def create_model(base_config: Config, context_dim: int, device: torch.device) -> SimpleLLM:
    """モデル作成"""
    model = SimpleLLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=context_dim,
        num_input_tokens=base_config.num_input_tokens,
    )
    model.to(device)
    return model


# ============================================================
# Main Search Function
# ============================================================

def run_sample_size_search(
    context_dim: int = 256,
    start_samples: int = 100,
    end_samples: int = 1600,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sample Size 探索を実行

    Args:
        context_dim: 固定 context_dim
        start_samples: 開始サンプル数
        end_samples: 終了サンプル数
        seed: ランダムシード
        output_dir: 出力ディレクトリ

    Returns:
        結果の辞書
    """
    set_seed(seed)

    # サンプル数リスト（倍増）
    sample_sizes: List[int] = []
    current = start_samples
    while current <= end_samples:
        sample_sizes.append(current)
        current *= 2

    print_flush("=" * 70)
    print_flush("SAMPLE SIZE SEARCH EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Context dim: {context_dim} (fixed)")
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
        print_flush(f"[SAMPLES={num_samples}] Starting experiment...")
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

        # モデル作成
        print_flush(f"\nCreating SimpleLLM (context_dim={context_dim})...")
        model = create_model(base_config, context_dim, device)

        # Phase 1 Config
        config_wrapper = Phase1ConfigWrapper(base_config, context_dim, patience=2)

        # Phase 1: ContextBlock 学習
        print_flush("\n[Phase 1] Training ContextBlock...")
        wrapper = SingleContextWrapper(model)
        trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

        phase1_start = time.time()
        _ = trainer.train(
            train_token_ids,
            label="Context",
            return_all_layers=True,
        )
        phase1_time = time.time() - phase1_start

        stats = trainer._training_stats
        phase1_iter = stats.get('iterations', 0)
        phase1_conv = stats.get('convergence_rate', 0)
        print_flush(f"Phase 1: {phase1_time:.1f}s, {phase1_iter} iter, conv={phase1_conv*100:.0f}%")

        # Phase 2 Prep
        print_flush("\n[Phase 2 Prep] Collecting context cache...")
        cache_start = time.time()

        train_context_cache = collect_context_cache_sequential(model, train_token_ids, device)
        val_context_cache = collect_context_cache_sequential(model, val_token_ids, device)

        print_flush("    Collecting token embeddings (chunked)...")
        train_token_embeds = collect_token_embeds_chunked(model, train_token_ids, device)
        val_token_embeds = collect_token_embeds_chunked(model, val_token_ids, device)

        cache_time = time.time() - cache_start
        print_flush(f"Cache collection: {cache_time:.1f}s")

        # Effective Rank
        er_analysis = analyze_fixed_points(val_context_cache, prefix="Val")
        val_er = er_analysis.get('effective_rank', 0)
        val_er_pct = val_er / context_dim * 100

        print_flush(f"Effective Rank: Val={val_er_pct:.1f}% ({val_er:.1f}/{context_dim})")

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
        best_epoch = history['best_epoch']

        result = {
            'num_samples': num_samples,
            'num_train_tokens': num_train_tokens,
            'val_ppl': val_ppl,
            'val_acc': val_acc,
            'val_er': val_er,
            'val_er_pct': val_er_pct,
            'phase1_iter': phase1_iter,
            'phase1_conv': phase1_conv,
            'phase1_time': phase1_time,
            'phase2_time': phase2_time,
            'best_epoch': best_epoch,
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
        del train_token_ids, val_token_ids
        del data_provider
        clear_gpu_cache(device)

    # 最終サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY - Sample Size Search")
    print_flush("=" * 70)
    print_flush(f"Context dim: {context_dim}")
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
    print_flush("=" * 70)

    # 結果を保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, "results.txt")
        with open(result_file, 'w') as f:
            f.write("Sample Size Search Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Context dim: {context_dim}\n")
            f.write(f"Sample sizes: {sample_sizes}\n\n")
            f.write(f"{'Samples':>8} | {'Tokens':>12} | {'PPL':>8} | {'Acc':>6} | {'ER%':>6} | {'Time':>8}\n")
            f.write("-" * 60 + "\n")
            for r in results:
                marker = " *" if r['num_samples'] == best_samples else ""
                f.write(f"{r['num_samples']:>8} | {r['num_train_tokens']:>12,} | {r['val_ppl']:>8.1f} | "
                        f"{r['val_acc']*100:>5.1f}% | {r['val_er_pct']:>5.1f}% | {r['total_time']:>7.1f}s{marker}\n")
            f.write("-" * 60 + "\n")
            f.write(f"\nBest: samples={best_samples}, PPL={best_ppl:.1f}\n")

        print_flush(f"\nResults saved to: {result_file}")

    return {
        'results': results,
        'best_samples': best_samples,
        'best_ppl': best_ppl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Sample Size Search Experiment')
    parser.add_argument('--context-dim', '-d', type=int, default=256,
                        help='Fixed context dimension (default: 256)')
    parser.add_argument('--start', type=int, default=100,
                        help='Starting sample size (default: 100)')
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
        output_dir = f"importants/logs/{timestamp}_sample_size_search"

    run_sample_size_search(
        context_dim=args.context_dim,
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
