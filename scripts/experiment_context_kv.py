#!/usr/bin/env python3
"""
Context-KV Attention 実験スクリプト

チャンク単位のcontextをKVキャッシュとして使用するモデルの実験。

Usage:
    python3 scripts/experiment_context_kv.py -s 200 --chunk-size 100
"""

import argparse
import gc
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from config.experiment import DataConfig
from src.config.wrappers import Phase1ConfigWrapper
from src.providers.data.memory import MemoryDataProvider
from src.evaluation.metrics import compute_effective_rank
from src.models.context_kv import ContextKVAttentionLLM, ContextKVWrapper
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.device import clear_gpu_cache
from src.utils.memory import get_gpu_memory_info
import psutil


def get_memory_usage() -> str:
    """CPUとGPUのメモリ使用量を取得"""
    process = psutil.Process()
    cpu_gb = process.memory_info().rss / (1024**3)

    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        gpu_gb = gpu_info['allocated_gb']  # already in GB
        return f"CPU: {cpu_gb:.1f}GB, GPU: {gpu_gb:.1f}GB"
    return f"CPU: {cpu_gb:.1f}GB"


def collect_val_cache_parallel(
    model: ContextKVWrapper,
    token_ids: torch.Tensor,
    device: torch.device,
    batch_size: int = 50000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Val dataのキャッシュを並列処理で収集

    Phase 1のtrainと同じshifted_prev_context方式を使用。

    Returns:
        context_cache: [num_tokens-1, context_dim]
        token_embeds: [num_tokens-1, embed_dim]
    """
    model.eval()
    num_tokens = len(token_ids) - 1  # 最後のトークンを除く
    context_dim = model.context_dim

    # Token embeddings計算
    with torch.no_grad():
        token_embeds_gpu = model.token_embedding(token_ids[:-1].unsqueeze(0).to(device))
        token_embeds_gpu = model.embed_norm(token_embeds_gpu).squeeze(0)
        token_embeds = token_embeds_gpu.cpu()
        del token_embeds_gpu
        clear_gpu_cache(device)

    # 結果テンソルを事前確保
    contexts = torch.zeros(num_tokens, context_dim)

    # 初期context（ゼロベクトル）
    init_ctx = torch.zeros(1, context_dim)

    # 最初のパス: ランダム初期化 (小さいスケールで初期化)
    previous_contexts = torch.randn(num_tokens, context_dim) * 0.01

    # 数回iterationして収束させる（簡易版）
    for iteration in range(3):
        # shifted_prev: [zero, ctx[0], ctx[1], ..., ctx[n-2]]
        shifted_prev = torch.cat([init_ctx, previous_contexts[:-1]], dim=0)

        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)
            batch_contexts = shifted_prev[start_idx:end_idx].to(device)
            batch_embeds = token_embeds[start_idx:end_idx].to(device)

            with torch.no_grad():
                batch_output = model.context_block(batch_contexts, batch_embeds)
            contexts[start_idx:end_idx] = batch_output.cpu()

            del batch_contexts, batch_embeds, batch_output
            clear_gpu_cache(device)

        # 次のiteration用に更新 (最後のiterationではclone不要)
        if iteration < 2:
            previous_contexts = contexts.clone()
        del shifted_prev

    # 不要な中間テンソルを解放
    del previous_contexts, init_ctx
    gc.collect()
    clear_gpu_cache(device)

    return contexts, token_embeds


def build_batch_context_intervals(
    batch_indices: torch.Tensor,
    context_cache: torch.Tensor,
    interval: int,
    device: torch.device,
    max_contexts: int = 32,
) -> torch.Tensor:
    """
    バッチのcontext intervalsを動的に構築（新方式）

    Position i の予測には:
      [context[i], context[i-interval], context[i-2*interval], ...]
    を使用。常に「現在位置からの等間隔」でcontextを取得。

    例: interval=100, position=350
      → [context[350], context[250], context[150], context[50]]

    Args:
        batch_indices: バッチ内の位置インデックス [batch_size]
        context_cache: 全context [num_tokens, context_dim]
        interval: contextを取得する間隔
        device: 出力デバイス
        max_contexts: 使用するcontext数の上限（メモリ効率のため）

    Returns:
        batch_contexts: [batch_size, max_num_contexts, context_dim] on device
    """
    batch_size = len(batch_indices)
    context_dim = context_cache.shape[1]

    # 各サンプルが必要とするcontext数を計算
    # position i → (i // interval) + 1 個のcontext
    num_contexts_per_sample = (batch_indices // interval) + 1  # [batch_size]
    # max_contextsで上限を設定（OOM防止）
    num_contexts_per_sample = torch.clamp(num_contexts_per_sample, max=max_contexts)
    max_num_contexts = int(num_contexts_per_sample.max().item())

    # 出力テンソルを直接GPU上に作成（ゼロパディング）
    batch_contexts = torch.zeros(batch_size, max_num_contexts, context_dim, device=device)

    # 各サンプルのcontext位置を計算
    # position i の k番目のcontext → context[i - k*interval]
    for k in range(max_num_contexts):
        # このkが有効なサンプルのマスク
        valid_mask = num_contexts_per_sample > k  # [batch_size]

        if not valid_mask.any():
            break

        # context位置を計算: i - k*interval
        context_positions = batch_indices - k * interval  # [batch_size]

        # 有効なサンプルのみ処理
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        valid_positions = context_positions[valid_indices]

        # contextを取得してGPUに転送
        contexts = context_cache[valid_positions].to(device)
        batch_contexts[valid_indices, k] = contexts

    return batch_contexts


def train_phase2(
    model: ContextKVAttentionLLM,
    train_context_cache: torch.Tensor,
    train_token_embeds: torch.Tensor,
    train_targets: torch.Tensor,
    val_context_cache: torch.Tensor,
    val_token_embeds: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    context_interval: int,
    max_contexts: int = 32,
    num_epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 3,
) -> tuple[float, float, int]:
    """
    Phase 2: Context-KV Attention + FFN の学習（メモリ効率版）

    Args:
        context_interval: contextを取得する間隔（position i から i, i-interval, i-2*interval, ...）
        max_contexts: 使用するcontext数の上限（通常LLMのcontext windowに相当）

    Returns:
        best_ppl, best_acc, best_epoch
    """
    model.freeze_all_context_blocks()

    # Embedding も freeze
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.embed_norm.parameters():
        param.requires_grad = False
    print_flush("✓ Embedding frozen")

    # 学習対象パラメータ
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"✓ Training {trainable_params:,}/{total_params:,} parameters")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    num_train = len(train_targets)
    num_val = len(val_targets)

    best_val_ppl = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    print_flush(f"\n[Phase 2] {num_train:,} train / {num_val:,} val tokens, {num_epochs} epochs")

    num_batches = (num_train + batch_size - 1) // batch_size

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        indices = torch.randperm(num_train)

        for batch_num, start in enumerate(range(0, num_train, batch_size)):
            end = min(start + batch_size, num_train)
            batch_idx = indices[start:end]

            # 進捗表示（全エポックで詳細表示）
            if batch_num % 50 == 0 or batch_num == num_batches - 1:
                processed = batch_num * batch_size
                elapsed = time.time() - epoch_start
                if processed > 0:
                    interim_ppl = torch.exp(torch.tensor(train_loss / processed)).item()
                    interim_acc = train_correct / processed * 100
                    print_flush(f"    Epoch {epoch}: batch {batch_num+1}/{num_batches} "
                                f"(ppl={interim_ppl:.1f}, acc={interim_acc:.1f}%, {elapsed:.1f}s)")
                else:
                    print_flush(f"    Epoch {epoch}: batch {batch_num+1}/{num_batches} ({elapsed:.1f}s)...")

            # 動的にcontext intervalsを構築（GPU上で直接作成）
            batch_context_chunks = build_batch_context_intervals(
                batch_idx, train_context_cache, context_interval, device, max_contexts
            )
            batch_token_embeds = train_token_embeds[batch_idx].to(device)
            batch_targets = train_targets[batch_idx].to(device)

            optimizer.zero_grad()

            # Forward
            hidden = model.forward_attention(batch_token_embeds, batch_context_chunks)
            logits = model.forward_output(hidden)

            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_idx)
            train_correct += int((logits.argmax(dim=-1) == batch_targets).sum().item())

            del batch_context_chunks, batch_token_embeds, batch_targets
            clear_gpu_cache(device)

        train_loss /= num_train
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for start in range(0, num_val, batch_size):
                end = min(start + batch_size, num_val)
                batch_idx = torch.arange(start, end)

                batch_context_chunks = build_batch_context_intervals(
                    batch_idx, val_context_cache, context_interval, device, max_contexts
                )
                batch_token_embeds = val_token_embeds[start:end].to(device)
                batch_targets = val_targets[start:end].to(device)

                hidden = model.forward_attention(batch_token_embeds, batch_context_chunks)
                logits = model.forward_output(hidden)

                loss = criterion(logits, batch_targets)

                val_loss += loss.item() * len(batch_targets)
                val_correct += int((logits.argmax(dim=-1) == batch_targets).sum().item())

                del batch_context_chunks, batch_token_embeds, batch_targets
                clear_gpu_cache(device)

        val_loss /= num_val
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        val_acc = val_correct / num_val

        epoch_time = time.time() - epoch_start

        # Early stopping check
        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        print_flush(
            f"    Epoch {epoch}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"acc={val_acc*100:.1f}% [{epoch_time:.1f}s]{marker}"
        )

        if no_improve >= patience:
            print_flush(f"    → Early stop at epoch {epoch}")
            break

    print_flush(f"    Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}, acc={best_val_acc*100:.1f}%")

    return best_val_ppl, best_val_acc, best_epoch


def run_context_kv_experiment(
    num_samples: int = 200,
    context_dim: int = 256,
    chunk_size: int = 100,
    num_heads: int = 8,
    max_contexts: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """Context-KV Attention 実験を実行（1ブロック版）"""

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    # 設定
    base_config = Config()

    # データロード
    print_flush("Loading data...")
    data_config = DataConfig.from_base(base_config, num_samples=num_samples)
    data_provider = MemoryDataProvider(data_config)
    train_token_ids, val_token_ids = data_provider.load_data()
    num_train_tokens = len(train_token_ids)
    print_flush(f"Data: {num_train_tokens:,} train, {len(val_token_ids):,} val tokens")
    print_flush(f"Memory after data load: {get_memory_usage()}")

    # モデル作成（1ブロック）
    context_dims = [context_dim]
    print_flush(f"\nCreating ContextKVAttentionLLM (cd={context_dim}, heads={num_heads}, chunk={chunk_size})...")

    model = ContextKVAttentionLLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dims=context_dims,
        num_heads=num_heads,
        chunk_size=chunk_size,
    )
    model.to(device)

    params = model.num_params()
    print_flush(f"Parameters: {params['total']:,} total")
    print_flush(f"Memory after model init: {get_memory_usage()}")

    # ========== Phase 1: ContextBlock学習 ==========
    print_flush(f"\n[Phase 1] Training ContextBlock (cd={context_dim})...")
    wrapper = ContextKVWrapper(model, block_idx=0)
    config_wrapper = Phase1ConfigWrapper(base_config, context_dim)
    trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

    phase1_start = time.time()
    train_result = trainer.train(
        train_token_ids,
        label="Context",
        return_all_layers=True,
    )
    phase1_time = time.time() - phase1_start

    # Phase 1完了後、trainerを解放してメモリ節約
    del trainer
    gc.collect()
    clear_gpu_cache(device)

    model.freeze_context_block(0)
    print_flush("✓ ContextBlock frozen")
    print_flush(f"Memory after Phase 1: {get_memory_usage()}")

    # ========== Phase 2 Prep: コンテキストキャッシュを準備 ==========
    print_flush("\n[Phase 2 Prep] Preparing context cache...")
    cache_start = time.time()

    # Phase 1のキャッシュを使用（Trainデータ）
    assert train_result.cache is not None
    assert train_result.token_embeds is not None

    # メモリ効率化: 必要なデータのみ取り出し、train_resultを早期解放
    # Note: train_result.contextsは使用しない（cacheと重複するため）
    train_context_cache = train_result.cache
    train_token_embeds = train_result.token_embeds

    # train_result全体を解放（contexts含む）
    del train_result
    gc.collect()
    clear_gpu_cache(device)
    print_flush(f"Memory after train_result release: {get_memory_usage()}")

    train_targets = train_token_ids[1:].clone()

    # Val dataはPhase 1と同じ並列方式でキャッシュ収集
    print_flush(f"Memory before val cache: {get_memory_usage()}")
    print_flush("  Collecting val cache (parallel)...")
    val_context_cache, val_token_embeds = collect_val_cache_parallel(
        wrapper, val_token_ids, device, config_wrapper.phase1_batch_size
    )
    gc.collect()
    clear_gpu_cache(device)
    print_flush(f"Memory after val cache: {get_memory_usage()}")

    # wrapperとconfig_wrapperを解放（Phase 2では使わない）
    del wrapper, config_wrapper
    gc.collect()

    val_targets = val_token_ids[1:].clone()

    # 元のtoken_idsはtargetsに変換済みなので解放
    del train_token_ids, val_token_ids
    gc.collect()

    cache_time = time.time() - cache_start
    print_flush(f"Cache preparation: {cache_time:.1f}s")
    print_flush(f"  Train: {train_context_cache.shape}")
    print_flush(f"  Val: {val_context_cache.shape}")
    theoretical_max = len(train_context_cache) // chunk_size + 1
    print_flush(f"  Context interval: {chunk_size} (theoretical max: {theoretical_max}, using max: {max_contexts})")

    # キャッシュサイズを表示
    train_ctx_mb = train_context_cache.numel() * 4 / (1024**2)
    train_emb_mb = train_token_embeds.numel() * 4 / (1024**2)
    val_ctx_mb = val_context_cache.numel() * 4 / (1024**2)
    val_emb_mb = val_token_embeds.numel() * 4 / (1024**2)
    total_cache_mb = train_ctx_mb + train_emb_mb + val_ctx_mb + val_emb_mb
    print_flush(f"  Cache sizes: train_ctx={train_ctx_mb:.0f}MB, train_emb={train_emb_mb:.0f}MB, "
                f"val_ctx={val_ctx_mb:.0f}MB, val_emb={val_emb_mb:.0f}MB (total={total_cache_mb:.0f}MB)")
    print_flush(f"Memory after Phase 2 Prep: {get_memory_usage()}")

    # Effective Rank（最終位置のcontextで計算）
    val_er = compute_effective_rank(val_context_cache.cpu())
    val_er_pct = val_er / context_dim * 100
    print_flush(f"Effective Rank: Val={val_er_pct:.1f}%")

    clear_gpu_cache(device)

    # ========== Phase 2: Attention + FFN学習 ==========
    print_flush("\n[Phase 2] Training Context-KV Attention...")

    phase2_start = time.time()
    best_ppl, best_acc, best_epoch = train_phase2(
        model=model,
        train_context_cache=train_context_cache,
        train_token_embeds=train_token_embeds,
        train_targets=train_targets,
        val_context_cache=val_context_cache,
        val_token_embeds=val_token_embeds,
        val_targets=val_targets,
        device=device,
        context_interval=chunk_size,
        max_contexts=max_contexts,
    )
    phase2_time = time.time() - phase2_start

    total_time = phase1_time + cache_time + phase2_time

    # ========== サマリー ==========
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY - Context-KV Attention Experiment")
    print_flush("=" * 70)
    print_flush(f"  Context dim: {context_dim}")
    print_flush(f"  Num heads: {num_heads}")
    print_flush(f"  Context interval: {chunk_size}")
    print_flush(f"  Max contexts: {max_contexts}")
    print_flush(f"Parameters: {params['total']:,}")
    print_flush(f"Phase 1: {phase1_time:.1f}s")
    print_flush(f"Cache collection: {cache_time:.1f}s")
    print_flush(f"Phase 2: {phase2_time:.1f}s, epoch {best_epoch}")
    print_flush(f"Effective Rank: {val_er_pct:.1f}%")
    print_flush(f"Val PPL: {best_ppl:.1f}")
    print_flush(f"Val Acc: {best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")
    print_flush("=" * 70)

    return {
        'num_samples': num_samples,
        'context_dim': context_dim,
        'chunk_size': chunk_size,
        'num_heads': num_heads,
        'max_contexts': max_contexts,
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'val_er_pct': val_er_pct,
        'phase1_time': phase1_time,
        'cache_time': cache_time,
        'phase2_time': phase2_time,
        'total_time': total_time,
        'best_epoch': best_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Context-KV Attention Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # 200サンプル、チャンクサイズ100
  python3 scripts/experiment_context_kv.py -s 200 --chunk-size 100

  # 800サンプル、チャンクサイズ50
  python3 scripts/experiment_context_kv.py -s 800 --chunk-size 50
'''
    )
    parser.add_argument('-s', '--samples', type=int, default=200,
                        help='Number of samples (default: 200)')
    parser.add_argument('-c', '--context-dim', type=int, default=256,
                        help='Context dimension (default: 256)')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Chunk size for context KV (default: 100)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--max-contexts', type=int, default=32,
                        help='Max number of contexts (context window, default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    print_flush("=" * 70)
    print_flush("CONTEXT-KV ATTENTION EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Context dim: {args.context_dim}")
    print_flush(f"Chunk size (interval): {args.chunk_size}")
    print_flush(f"Num heads: {args.num_heads}")
    print_flush(f"Max contexts: {args.max_contexts}")
    print_flush("=" * 70)

    run_context_kv_experiment(
        num_samples=args.samples,
        context_dim=args.context_dim,
        chunk_size=args.chunk_size,
        num_heads=args.num_heads,
        max_contexts=args.max_contexts,
        seed=args.seed,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
