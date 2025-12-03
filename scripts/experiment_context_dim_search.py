#!/usr/bin/env python3
"""
Context Dim 探索実験スクリプト

サンプル数に対する適正 context_dim を探索する。
context_dim を start_dim から n 刻みで増加/減少させ、
val PPL が2回連続で悪化した時点で停止する。

特徴:
- 毎回新規モデル作成（シンプルで安定）
- ContextBlock: 1つのみで実験
- 早期停止: val PPL が2回連続悪化 or 上限/下限到達
- 双方向探索対応: 増加（→max_dim）または減少（→min_dim）

使用方法:
  # 増加方向: 100から10刻みで500まで
  python3 scripts/experiment_context_dim_search.py -s 2000 -n 10

  # 増加方向: 200から20刻みで300まで
  python3 scripts/experiment_context_dim_search.py -s 1000 --start-dim 200 -n 20 -m 300

  # 減少方向: 200から20刻みで100まで
  python3 scripts/experiment_context_dim_search.py -s 1000 --start-dim 200 -n 20 --min-dim 100
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
# Search Experiment Configuration
# ============================================================
SEARCH_PPL_WORSE_PATIENCE = 2  # Val PPL が何回連続で悪化したら停止するか


class SimpleContextBlock(nn.Module):
    """
    シンプルな ContextBlock（探索実験用）
    """

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

        # FFN: [context + token_embeds] -> context_dim
        input_dim = context_dim + token_input_dim
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, context_dim),
            nn.GELU()
        )

        # LayerNorm
        self.context_norm = nn.LayerNorm(context_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """重みを初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        context: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, token_input_dim]

        Returns:
            new_context: [batch, context_dim]
        """
        fnn_input = torch.cat([context, token_embeds], dim=-1)
        delta_context = self.fnn(fnn_input)
        new_context = self.context_norm(context + delta_context)
        return new_context

    def forward_batch(
        self,
        context: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch forward pass（キャッシュ収集用）
        """
        return self.forward(context, token_embeds)


class SimpleTokenBlock(nn.Module):
    """
    シンプルな TokenBlock（探索実験用）
    """

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

        # FFN: [context + token_embeds] -> embed_dim
        input_dim = context_dim + token_input_dim
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU()
        )

        # LayerNorm
        self.token_norm = nn.LayerNorm(embed_dim)

        # 残差射影（token_input_dim != embed_dim の場合）
        self.residual_proj: Optional[nn.Linear] = None
        if token_input_dim != embed_dim:
            self.residual_proj = nn.Linear(token_input_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """重みを初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        context: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, token_input_dim]

        Returns:
            new_token: [batch, embed_dim]
        """
        fnn_input = torch.cat([context, token_embeds], dim=-1)
        delta_token = self.fnn(fnn_input)

        if self.residual_proj is not None:
            residual = self.residual_proj(token_embeds)
        else:
            residual = token_embeds

        new_token = self.token_norm(residual + delta_token)
        return new_token


class SimpleLLM(nn.Module):
    """
    context_dim 探索用LLM

    単一ContextBlock + TokenBlock
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
        self.num_input_tokens = num_input_tokens

        # Token Embeddings (GPT-2 pretrained)
        self._load_pretrained_embeddings()
        self.embed_norm = nn.LayerNorm(embed_dim)

        # ContextBlock
        self.context_block = SimpleContextBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        # TokenBlock
        self.token_block = SimpleTokenBlock(
            context_dim=context_dim,
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

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """ContextBlock の順伝搬"""
        return self.context_block(context, token_embeds)

    def forward_token(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """TokenBlock の順伝搬"""
        return self.token_block(context, token_embeds)

    def freeze_context_block(self) -> None:
        """ContextBlockをfreeze"""
        for param in self.context_block.parameters():
            param.requires_grad = False
        print_flush("✓ ContextBlock frozen")

    def unfreeze_context_block(self) -> None:
        """ContextBlockをunfreeze"""
        for param in self.context_block.parameters():
            param.requires_grad = True

    def num_params(self) -> Dict[str, int]:
        """パラメータ数を返す"""
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
    """Phase 1 用: SimpleLLM のラッパー"""

    def __init__(self, model: SimpleLLM):
        super().__init__()
        self.cascade_model = model

        # Phase1Trainerが期待するプロパティ
        self.token_embedding = model.token_embedding
        self.embed_norm = model.embed_norm
        self.context_dim = model.context_dim
        self.embed_dim = model.embed_dim
        self.num_input_tokens = model.num_input_tokens
        self.vocab_size = model.vocab_size

        self.context_block = model.context_block

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """ContextBlock の順伝搬"""
        return self.context_block(context, token_embeds)


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
    """
    Phase 2: TokenBlock 学習

    Args:
        model: SimpleLLM
        train/val_token_ids: トークンID
        train/val_context_cache: コンテキストキャッシュ
        train/val_token_embeds: トークン埋め込み
        config: 設定
        device: デバイス

    Returns:
        history: 学習履歴
    """
    model.freeze_context_block()

    # Embedding freeze
    model.token_embedding.weight.requires_grad = False
    print_flush("✓ Embedding frozen")

    # TokenBlockのみ学習
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
        train_batches = 0

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
            train_batches += 1

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


def create_model(
    base_config: Config,
    context_dim: int,
    device: torch.device,
) -> SimpleLLM:
    """
    新規モデルを作成

    Args:
        base_config: 基本設定
        context_dim: コンテキスト次元
        device: デバイス

    Returns:
        SimpleLLM
    """
    model = SimpleLLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=context_dim,
        num_input_tokens=base_config.num_input_tokens,
    )
    model.to(device)
    return model


def run_context_dim_search(
    num_samples: int = 2000,
    start_dim: int = 100,
    dim_step: int = 10,
    max_dim: int = 500,
    min_dim: int = 10,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Context Dim 探索を実行

    Args:
        num_samples: サンプル数
        start_dim: 開始 context_dim
        dim_step: dim の刻み幅（例: 10 なら 10刻み）
        max_dim: 最大 context_dim（増加方向探索時の上限）
        min_dim: 最小 context_dim（減少方向探索時の下限）
        seed: ランダムシード
        output_dir: 出力ディレクトリ

    探索方向:
        - start_dim < max_dim: 増加方向（start_dim → max_dim）
        - start_dim > min_dim and max_dim <= start_dim: 減少方向（start_dim → min_dim）

    Returns:
        結果の辞書
    """
    set_seed(seed)

    # 探索方向を決定
    if start_dim < max_dim:
        direction = "increase"
        end_dim = max_dim
    else:
        direction = "decrease"
        end_dim = min_dim

    print_flush("=" * 70)
    print_flush("CONTEXT DIM SEARCH EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples}")
    print_flush(f"Start dim: {start_dim}")
    print_flush(f"Dim step: {dim_step}")
    print_flush(f"Direction: {direction} ({start_dim} → {end_dim})")
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

    # 結果記録
    results: List[Dict[str, Any]] = []
    worse_count = 0
    best_ppl = float('inf')
    best_dim = start_dim

    current_dim = start_dim

    def should_continue(dim: int) -> bool:
        if direction == "increase":
            return dim <= max_dim
        else:
            return dim >= min_dim

    while should_continue(current_dim):
        print_flush(f"\n{'='*70}")
        print_flush(f"[DIM={current_dim}] Starting experiment...")
        print_flush("=" * 70)

        # 新規モデル作成
        print_flush(f"\nCreating SimpleLLM (context_dim={current_dim})...")
        model = create_model(base_config, current_dim, device)

        # Phase 1 Config
        config_wrapper = Phase1ConfigWrapper(base_config, current_dim, patience=2)

        # Phase 1: ContextBlock 学習
        print_flush(f"\n[Phase 1] Training ContextBlock (context_dim={current_dim})...")
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
        print_flush(f"Phase 1: {phase1_time:.1f}s, {stats.get('iterations', 0)} iter, "
                    f"conv={stats.get('convergence_rate', 0)*100:.0f}%")

        # Phase 2 Prep: キャッシュ収集
        print_flush("\n[Phase 2 Prep] Collecting context cache...")
        cache_start = time.time()

        train_context_cache = collect_context_cache_sequential(model, train_token_ids, device)
        val_context_cache = collect_context_cache_sequential(model, val_token_ids, device)

        # Token embeddings（チャンク処理でGPUメモリを節約）
        print_flush("    Collecting token embeddings (chunked)...")
        train_token_embeds = collect_token_embeds_chunked(model, train_token_ids, device)
        val_token_embeds = collect_token_embeds_chunked(model, val_token_ids, device)

        cache_time = time.time() - cache_start
        print_flush(f"Cache collection: {cache_time:.1f}s")

        # Effective Rank
        val_metrics = analyze_fixed_points(val_context_cache, label="Val", verbose=False)
        val_er = val_metrics['effective_rank']
        val_er_pct = val_er / current_dim * 100
        print_flush(f"Effective Rank: Val={val_er_pct:.1f}% ({val_er:.1f}/{current_dim})")

        # Phase 2: TokenBlock 学習
        print_flush("\n[Phase 2] Training TokenBlock...")
        phase2_config = Phase2ConfigWrapper(base_config)

        phase2_start = time.time()
        history = train_phase2(
            model=model,
            train_token_ids=train_token_ids,
            val_token_ids=val_token_ids,
            train_context_cache=train_context_cache,
            train_token_embeds=train_token_embeds,
            val_context_cache=val_context_cache,
            val_token_embeds=val_token_embeds,
            config=phase2_config,
            device=device,
        )
        phase2_time = time.time() - phase2_start

        best_epoch = history['best_epoch']
        val_ppl = history['val_ppl'][best_epoch - 1]
        val_acc = history['val_acc'][best_epoch - 1]

        total_time = phase1_time + cache_time + phase2_time

        # 結果記録
        result_entry = {
            'context_dim': current_dim,
            'val_ppl': val_ppl,
            'val_acc': val_acc,
            'val_er_pct': val_er_pct,
            'phase1_time': phase1_time,
            'cache_time': cache_time,
            'phase2_time': phase2_time,
            'total_time': total_time,
            'convergence_rate': stats.get('convergence_rate', 0),
        }
        results.append(result_entry)

        print_flush(f"\n[Result] dim={current_dim}: PPL={val_ppl:.1f}, Acc={val_acc*100:.1f}%, "
                    f"ER={val_er_pct:.1f}%, Time={total_time:.1f}s")

        # 悪化判定
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_dim = current_dim
            worse_count = 0
            print_flush(f"  ★ New best! dim={best_dim}, PPL={best_ppl:.1f}")
        else:
            worse_count += 1
            print_flush(f"  ↓ PPL increased ({worse_count}/{SEARCH_PPL_WORSE_PATIENCE} consecutive)")

        # N回連続悪化で停止
        if worse_count >= SEARCH_PPL_WORSE_PATIENCE:
            print_flush(f"\n⛔ Stopping: PPL increased {SEARCH_PPL_WORSE_PATIENCE} times consecutively")
            break

        # 次の dim（増加または減少）
        if direction == "increase":
            current_dim += dim_step
        else:
            current_dim -= dim_step

        # メモリ解放
        del model, train_context_cache, val_context_cache
        del train_token_embeds, val_token_embeds
        clear_gpu_cache(device)

    # 最終サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY - Context Dim Search")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples}")
    print_flush(f"Start dim: {start_dim}")
    print_flush(f"Dim step: {dim_step}")
    print_flush(f"Direction: {direction} ({start_dim} → {end_dim})")
    print_flush("\nResults:")
    print_flush(f"{'dim':>6} | {'PPL':>8} | {'Acc':>6} | {'ER%':>6} | {'Time':>8}")
    print_flush("-" * 45)
    for r in results:
        marker = " ★" if r['context_dim'] == best_dim else ""
        print_flush(f"{r['context_dim']:>6} | {r['val_ppl']:>8.1f} | {r['val_acc']*100:>5.1f}% | "
                    f"{r['val_er_pct']:>5.1f}% | {r['total_time']:>7.1f}s{marker}")
    print_flush("-" * 45)
    print_flush(f"\n★ Best: dim={best_dim}, PPL={best_ppl:.1f}")
    print_flush("=" * 70)

    # 結果を保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, "results.txt")
        with open(result_file, 'w') as f:
            f.write("Context Dim Search Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Samples: {num_samples}\n")
            f.write(f"Start dim: {start_dim}\n")
            f.write(f"Dim step: {dim_step}\n")
            f.write(f"Direction: {direction} ({start_dim} -> {end_dim})\n\n")
            f.write(f"{'dim':>6} | {'PPL':>8} | {'Acc':>6} | {'ER%':>6} | {'Time':>8}\n")
            f.write("-" * 50 + "\n")
            for r in results:
                marker = " *" if r['context_dim'] == best_dim else ""
                f.write(f"{r['context_dim']:>6} | {r['val_ppl']:>8.1f} | {r['val_acc']*100:>5.1f}% | "
                        f"{r['val_er_pct']:>5.1f}% | {r['total_time']:>7.1f}s{marker}\n")
            f.write("-" * 50 + "\n")
            f.write(f"\nBest: dim={best_dim}, PPL={best_ppl:.1f}\n")

        print_flush(f"\nResults saved to: {result_file}")

    return {
        'results': results,
        'best_dim': best_dim,
        'best_ppl': best_ppl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Context Dim Search Experiment')
    parser.add_argument('--samples', '-s', type=int, default=2000,
                        help='Number of training samples')
    parser.add_argument('--start-dim', type=int, default=100,
                        help='Starting context dim (default: 100)')
    parser.add_argument('--dim-step', '-n', type=int, default=10,
                        help='Context dim step size (e.g., 10 means +10 each iteration)')
    parser.add_argument('--max-dim', '-m', type=int, default=500,
                        help='Maximum context dim (for increasing search)')
    parser.add_argument('--min-dim', type=int, default=10,
                        help='Minimum context dim (for decreasing search)')
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
        output_dir = f"importants/logs/{timestamp}_context_dim_search"

    run_context_dim_search(
        num_samples=args.samples,
        start_dim=args.start_dim,
        dim_step=args.dim_step,
        max_dim=args.max_dim,
        min_dim=args.min_dim,
        seed=args.seed,
        output_dir=output_dir,
    )

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == "__main__":
    main()
