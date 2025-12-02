#!/usr/bin/env python3
"""
Context Dim 探索実験スクリプト

サンプル数に対する適正 context_dim を探索する。
context_dim を n 刻みで増加させ（10, 20, 30, ...）、
val PPL が2回連続で悪化した時点で停止する。

特徴:
- 重み再利用: dim=10 → 20 への拡張時、既存の重みを再利用
- 拡張部分: 小さいランダム値で初期化（0だと収束しない可能性）
- ContextBlock: 1つのみで実験
- 早期停止: val PPL が2回連続悪化 or 上限到達

使用方法:
  python3 scripts/experiment_context_dim_search.py -s 2000 -n 10  # 10刻み
  python3 scripts/experiment_context_dim_search.py -s 2000 -n 20 --max-dim 500
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
from config.experiment import DataConfig


class ExpandableContextBlock(nn.Module):
    """
    拡張可能な ContextBlock

    context_dim を拡張する際、既存の重みを再利用できる。
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

        # 残差射影なし（context_input_dim == context_output_dim）

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

    def expand_context_dim(self, new_context_dim: int, noise_scale: float = 0.01) -> None:
        """
        context_dim を拡張

        既存の重みを保持し、拡張部分は小さいランダム値で初期化。

        Args:
            new_context_dim: 新しい context_dim
            noise_scale: 拡張部分のノイズスケール
        """
        if new_context_dim <= self.context_dim:
            raise ValueError(f"new_context_dim ({new_context_dim}) must be > current ({self.context_dim})")

        old_context_dim = self.context_dim
        expansion = new_context_dim - old_context_dim

        # 旧FFN Linear層の重み・バイアス取得
        old_linear = self.fnn[0]
        old_weight = old_linear.weight.data  # [old_out, old_in]
        old_bias = old_linear.bias.data if old_linear.bias is not None else None

        # 新FFN作成
        new_input_dim = new_context_dim + self.token_input_dim
        new_fnn = nn.Sequential(
            nn.Linear(new_input_dim, new_context_dim),
            nn.GELU()
        )

        # 重みコピー（拡張部分はランダム）
        new_linear = new_fnn[0]
        with torch.no_grad():
            # 出力次元（行）: 既存部分コピー、拡張部分ランダム
            # 入力次元（列）: 既存context部分コピー、拡張context部分ランダム、token部分コピー

            # まずゼロで初期化
            new_linear.weight.zero_()
            if new_linear.bias is not None:
                new_linear.bias.zero_()

            # 既存の出力次元×既存のcontext入力次元をコピー
            new_linear.weight[:old_context_dim, :old_context_dim] = old_weight[:, :old_context_dim]

            # 既存の出力次元×拡張context入力次元は小さいランダム
            new_linear.weight[:old_context_dim, old_context_dim:new_context_dim] = (
                torch.randn(old_context_dim, expansion) * noise_scale
            )

            # 既存の出力次元×token入力次元をコピー
            new_linear.weight[:old_context_dim, new_context_dim:] = old_weight[:, old_context_dim:]

            # 拡張出力次元×全入力次元は小さいランダム
            new_linear.weight[old_context_dim:, :] = (
                torch.randn(expansion, new_input_dim) * noise_scale
            )

            # バイアス: 既存部分コピー、拡張部分ゼロ
            if old_bias is not None and new_linear.bias is not None:
                new_linear.bias[:old_context_dim] = old_bias
                new_linear.bias[old_context_dim:] = 0

        self.fnn = new_fnn

        # LayerNorm拡張
        old_norm = self.context_norm
        new_norm = nn.LayerNorm(new_context_dim)
        with torch.no_grad():
            new_norm.weight[:old_context_dim] = old_norm.weight
            new_norm.weight[old_context_dim:] = 1.0
            new_norm.bias[:old_context_dim] = old_norm.bias
            new_norm.bias[old_context_dim:] = 0

        self.context_norm = new_norm
        self.context_dim = new_context_dim


class ExpandableTokenBlock(nn.Module):
    """
    拡張可能な TokenBlock

    context_dim が拡張される際、入力次元を調整する。
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

    def expand_context_dim(self, new_context_dim: int, noise_scale: float = 0.01) -> None:
        """
        context_dim を拡張（入力次元のみ変更）

        Args:
            new_context_dim: 新しい context_dim
            noise_scale: 拡張部分のノイズスケール
        """
        if new_context_dim <= self.context_dim:
            raise ValueError(f"new_context_dim ({new_context_dim}) must be > current ({self.context_dim})")

        old_context_dim = self.context_dim
        expansion = new_context_dim - old_context_dim

        # 旧FFN Linear層の重み・バイアス取得
        old_linear = self.fnn[0]
        old_weight = old_linear.weight.data  # [embed_dim, old_context + token_input]
        old_bias = old_linear.bias.data if old_linear.bias is not None else None

        # 新FFN作成
        new_input_dim = new_context_dim + self.token_input_dim
        new_fnn = nn.Sequential(
            nn.Linear(new_input_dim, self.embed_dim),
            nn.GELU()
        )

        # 重みコピー
        new_linear = new_fnn[0]
        with torch.no_grad():
            new_linear.weight.zero_()
            if new_linear.bias is not None:
                new_linear.bias.zero_()

            # 既存context入力部分をコピー
            new_linear.weight[:, :old_context_dim] = old_weight[:, :old_context_dim]

            # 拡張context入力部分は小さいランダム
            new_linear.weight[:, old_context_dim:new_context_dim] = (
                torch.randn(self.embed_dim, expansion) * noise_scale
            )

            # token入力部分をコピー
            new_linear.weight[:, new_context_dim:] = old_weight[:, old_context_dim:]

            # バイアスコピー
            if old_bias is not None and new_linear.bias is not None:
                new_linear.bias.copy_(old_bias)

        self.fnn = new_fnn
        self.context_dim = new_context_dim


class SearchableLLM(nn.Module):
    """
    context_dim 探索用LLM

    単一ContextBlock + TokenBlock
    context_dim を拡張可能
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

        # ContextBlock（拡張可能）
        self.context_block = ExpandableContextBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        # TokenBlock（拡張可能）
        self.token_block = ExpandableTokenBlock(
            context_dim=context_dim,
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

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """ContextBlock の順伝搬"""
        return self.context_block(context, token_embeds)

    def forward_token(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """TokenBlock の順伝搬"""
        return self.token_block(context, token_embeds)

    def expand_context_dim(self, new_context_dim: int, noise_scale: float = 0.01) -> None:
        """
        context_dim を拡張

        ContextBlock と TokenBlock の両方を拡張する。

        Args:
            new_context_dim: 新しい context_dim
            noise_scale: 拡張部分のノイズスケール
        """
        print_flush(f"  Expanding context_dim: {self.context_dim} -> {new_context_dim}")

        self.context_block.expand_context_dim(new_context_dim, noise_scale)
        self.token_block.expand_context_dim(new_context_dim, noise_scale)
        self.context_dim = new_context_dim

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
    """Phase 1 用: SearchableLLM のラッパー"""

    def __init__(self, model: SearchableLLM):
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


def collect_context_cache_sequential(
    model: SearchableLLM,
    token_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    順次処理でコンテキストキャッシュを収集

    Args:
        model: SearchableLLM
        token_ids: トークンID
        device: デバイス

    Returns:
        context_cache: [num_tokens, context_dim]
    """
    model.eval()
    num_tokens = len(token_ids) - 1
    context_dim = model.context_dim

    context_cache = torch.zeros(num_tokens, context_dim, device='cpu')

    with torch.no_grad():
        # Token embeddings
        all_embeds = model.token_embedding(token_ids.to(device))
        all_embeds = model.embed_norm(all_embeds)
        token_embeds_all = all_embeds[:-1].cpu()
        del all_embeds
        clear_gpu_cache(device)

        # 初期context
        prev_context = torch.zeros(1, context_dim, device=device)

        # 順次処理
        for i in range(num_tokens):
            token_embed = token_embeds_all[i:i+1].to(device)
            new_context = model.forward_context(prev_context, token_embed)
            context_cache[i] = new_context.cpu().squeeze(0)
            prev_context = new_context

            if (i + 1) % 100000 == 0:
                print_flush(f"      {i+1:,}/{num_tokens:,} tokens processed...")

    return context_cache


def train_phase2(
    model: SearchableLLM,
    train_token_ids: torch.Tensor,
    val_token_ids: torch.Tensor,
    train_context_cache: torch.Tensor,
    train_token_embeds: torch.Tensor,
    val_context_cache: torch.Tensor,
    val_token_embeds: torch.Tensor,
    config: Phase2ConfigWrapper,
    device: torch.device,
) -> Dict[str, Any]:
    """Phase 2 学習を実行"""
    from torch.optim import AdamW

    model.to(device)
    model.freeze_context_block()
    model.token_embedding.weight.requires_grad = False
    print_flush("✓ Embedding frozen")

    # 学習対象のパラメータ
    trainable_params = [p for p in model.token_block.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = model.num_params()['total']
    print_flush(f"✓ Training TokenBlock only: {total_trainable:,}/{total_params:,} parameters")

    optimizer = AdamW(trainable_params, lr=config.phase2_learning_rate)
    criterion = nn.CrossEntropyLoss()

    # ターゲット
    train_targets = train_token_ids[1:].to(device)
    val_targets = val_token_ids[1:].to(device)

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
    batch_size = config.phase2_batch_size or 1000

    print_flush(f"\n[Phase 2] {num_train:,} train / {len(val_targets):,} val tokens, "
                f"{config.phase2_epochs} epochs")

    for epoch in range(1, config.phase2_epochs + 1):
        epoch_start = time.time()

        # === Training ===
        model.train()
        total_loss = 0.0

        for start_idx in range(0, num_train, batch_size):
            end_idx = min(start_idx + batch_size, num_train)

            batch_token_embeds = train_token_embeds[start_idx:end_idx].to(device)
            batch_targets = train_targets[start_idx:end_idx]
            batch_context = train_context_cache[start_idx:end_idx].to(device)

            optimizer.zero_grad()
            token_out = model.forward_token(batch_context, batch_token_embeds)
            logits = model.token_output(token_out)

            loss = criterion(logits, batch_targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, config.phase2_gradient_clip)
            optimizer.step()

            total_loss += loss.item() * (end_idx - start_idx)

        train_ppl = torch.exp(torch.tensor(total_loss / num_train)).item()

        # === Validation ===
        model.eval()
        val_loss = 0.0
        correct = 0
        num_val = len(val_targets)

        with torch.no_grad():
            for start_idx in range(0, num_val, batch_size):
                end_idx = min(start_idx + batch_size, num_val)

                batch_token_embeds = val_token_embeds[start_idx:end_idx].to(device)
                batch_targets = val_targets[start_idx:end_idx]
                batch_context = val_context_cache[start_idx:end_idx].to(device)

                token_out = model.forward_token(batch_context, batch_token_embeds)
                logits = model.token_output(token_out)

                val_loss += criterion(logits, batch_targets).item() * (end_idx - start_idx)
                correct += (logits.argmax(dim=-1) == batch_targets).sum().item()

        val_ppl = torch.exp(torch.tensor(val_loss / num_val)).item()
        val_acc = correct / num_val

        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)
        history['val_acc'].append(val_acc)

        is_best = val_ppl < best_val_ppl
        marker = " *" if is_best else ""

        if is_best:
            best_val_ppl = val_ppl
            history['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.time() - epoch_start
        print_flush(f"    Epoch {epoch}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                    f"acc={val_acc*100:.1f}% [{elapsed:.1f}s]{marker}")

        # Early stopping
        ppl_improvement = prev_val_ppl - val_ppl
        min_improvement = getattr(config, 'phase2_min_ppl_improvement', 0.4)

        if epoch > 1 and ppl_improvement < min_improvement and ppl_improvement >= 0:
            print_flush(f"    → Early stop at epoch {epoch}")
            break

        if patience_counter >= config.phase2_patience:
            print_flush(f"    → Early stop at epoch {epoch}")
            break

        prev_val_ppl = val_ppl

    print_flush(f"    Best: epoch {history['best_epoch']}, ppl={best_val_ppl:.1f}, "
                f"acc={history['val_acc'][history['best_epoch']-1]*100:.1f}%")

    return history


def run_context_dim_search(
    num_samples: int = 2000,
    dim_step: int = 10,
    max_dim: int = 500,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    context_dim 探索実験を実行

    Args:
        num_samples: サンプル数
        dim_step: context_dim の増加幅
        max_dim: 最大 context_dim
        seed: 乱数シード
        output_dir: 出力ディレクトリ

    Returns:
        実験結果
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

    # 初期モデル作成（dim_step から開始）
    initial_dim = dim_step
    print_flush(f"\nCreating SearchableLLM (initial context_dim={initial_dim})...")
    model = SearchableLLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=initial_dim,
        num_input_tokens=base_config.num_input_tokens,
    )
    model.to(device)

    # 結果記録
    results: List[Dict[str, Any]] = []
    worse_count = 0
    best_ppl = float('inf')
    best_dim = initial_dim

    current_dim = initial_dim

    while current_dim <= max_dim:
        print_flush(f"\n{'='*70}")
        print_flush(f"[DIM={current_dim}] Starting experiment...")
        print_flush(f"{'='*70}")

        # 拡張が必要な場合
        if current_dim > model.context_dim:
            model.expand_context_dim(current_dim, noise_scale=0.01)
            model.to(device)

        # ContextBlock をunfreeze
        model.unfreeze_context_block()

        # Phase 1 Config を更新
        config_wrapper = Phase1ConfigWrapper(base_config, current_dim)

        # Phase 1: ContextBlock 学習
        print_flush(f"\n[Phase 1] Training ContextBlock (context_dim={current_dim})...")
        wrapper = SingleContextWrapper(model)
        # wrapperのcontext_dimも更新
        wrapper.context_dim = current_dim
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

        # Token embeddings
        with torch.no_grad():
            train_embeds = model.token_embedding(train_token_ids.to(device))
            train_embeds = model.embed_norm(train_embeds)
            train_token_embeds = train_embeds[:-1].cpu()

            val_embeds = model.token_embedding(val_token_ids.to(device))
            val_embeds = model.embed_norm(val_embeds)
            val_token_embeds = val_embeds[:-1].cpu()

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
            print_flush(f"  ↓ PPL increased ({worse_count}/2 consecutive)")

        # 2回連続悪化で停止
        if worse_count >= 2:
            print_flush("\n⛔ Stopping: PPL increased 2 times consecutively")
            break

        # 次の dim
        current_dim += dim_step

        # メモリ解放
        del train_context_cache, val_context_cache
        del train_token_embeds, val_token_embeds
        clear_gpu_cache(device)

    # 最終サマリー
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY - Context Dim Search")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples}")
    print_flush(f"Dim step: {dim_step}")
    print_flush(f"Max dim: {max_dim}")
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
            f.write(f"Dim step: {dim_step}\n")
            f.write(f"Max dim: {max_dim}\n\n")
            f.write(f"{'dim':>6} | {'PPL':>8} | {'Acc':>6} | {'ER%':>6} | {'Time':>8}\n")
            f.write("-" * 50 + "\n")
            for r in results:
                marker = " *" if r['context_dim'] == best_dim else ""
                f.write(f"{r['context_dim']:>6} | {r['val_ppl']:>8.1f} | {r['val_acc']*100:>5.1f}% | "
                        f"{r['val_er_pct']:>5.1f}% | {r['total_time']:>7.1f}s{marker}\n")
            f.write("-" * 50 + "\n")
            f.write(f"\nBest: dim={best_dim}, PPL={best_ppl:.1f}\n")
        print_flush(f"\nResults saved to: {result_file}")

    # メモリ解放
    del model
    data_provider.close()
    clear_gpu_cache(device)

    return {
        'results': results,
        'best_dim': best_dim,
        'best_ppl': best_ppl,
    }


def main():
    parser = argparse.ArgumentParser(description='Context Dim Search Experiment')
    parser.add_argument('--samples', '-s', type=int, default=2000, help='Number of samples')
    parser.add_argument('--dim-step', '-n', type=int, default=10, help='Context dim step')
    parser.add_argument('--max-dim', '-m', type=int, default=500, help='Max context dim')
    parser.add_argument('--output-dir', '-o', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or f"importants/logs/{timestamp}_context_dim_search"

    print_flush("=" * 70)
    print_flush("CONTEXT DIM SEARCH EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Dim step: {args.dim_step}")
    print_flush(f"Max dim: {args.max_dim}")
    print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    run_context_dim_search(
        num_samples=args.samples,
        dim_step=args.dim_step,
        max_dim=args.max_dim,
        seed=args.seed,
        output_dir=output_dir,
    )

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == '__main__':
    main()
