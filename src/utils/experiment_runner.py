"""
Unified Experiment Runner

共通の実験実行ロジックを提供。
全モデルタイプ（Pythia, Infini, Multi-Memory, Hierarchical）に対応。
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn

from config.pythia import PythiaConfig
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE
from src.data.reversal_pairs import get_reversal_pairs
from src.models.pythia import PythiaModel
from src.models.infini_pythia import InfiniPythiaModel
from src.models.multi_memory_pythia import MultiMemoryInfiniPythiaModel
from src.models.hierarchical_pythia import HierarchicalMemoryPythiaModel
from src.utils.device import clear_gpu_cache
from src.utils.evaluation import evaluate_position_wise_ppl, evaluate_reversal_curse
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import (
    get_device,
    get_tokenizer,
    prepare_data_loaders,
)


class ModelType(Enum):
    """モデルタイプ"""
    PYTHIA = "pythia"
    INFINI = "infini"
    MULTI_MEMORY = "multi_memory"
    HIERARCHICAL = "hierarchical"


@dataclass
class ExperimentConfig:
    """実験設定"""
    # Data
    num_samples: int = 5000
    seq_length: int = 256
    val_split: float = 0.1

    # Training
    num_epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4

    # Model
    use_delta_rule: bool = True
    num_memories: int = 4  # Multi-Memory / Hierarchical用

    # Long context
    long_context_train: bool = False
    long_context_eval: bool = False
    num_long_documents: int = 50
    tokens_per_document: int = 4096

    # ALiBi (Infini用)
    use_alibi: bool = False
    alibi_scale: float = 1.0

    # Multi-Memory Bank (Infini用、廃止予定)
    num_memory_banks: int = 1
    segments_per_bank: int = 4


def create_model(
    model_type: ModelType,
    config: PythiaConfig,
    exp_config: ExperimentConfig,
) -> nn.Module:
    """モデルを作成"""
    if model_type == ModelType.PYTHIA:
        return PythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
        )

    elif model_type == ModelType.INFINI:
        return InfiniPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            use_delta_rule=exp_config.use_delta_rule,
            num_memory_banks=exp_config.num_memory_banks,
            segments_per_bank=exp_config.segments_per_bank,
            use_alibi=exp_config.use_alibi,
            alibi_scale=exp_config.alibi_scale,
        )

    elif model_type == ModelType.MULTI_MEMORY:
        return MultiMemoryInfiniPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            use_delta_rule=exp_config.use_delta_rule,
            num_memories=exp_config.num_memories,
        )

    elif model_type == ModelType.HIERARCHICAL:
        return HierarchicalMemoryPythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            use_delta_rule=exp_config.use_delta_rule,
            num_fine_memories=exp_config.num_memories,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    has_memory: bool = False,
) -> tuple[float, int, Optional[dict[str, Any]]]:
    """
    モデルを訓練

    Args:
        model: 訓練するモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        optimizer: オプティマイザ
        device: デバイス
        num_epochs: エポック数
        has_memory: メモリを持つモデルか

    Returns:
        best_val_ppl, best_epoch, best_state_dict
    """
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Reset memory at epoch start
        if has_memory and hasattr(model, 'reset_memory'):
            model.reset_memory()

        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if has_memory:
                logits = model(input_ids, update_memory=True)
            else:
                logits = model(input_ids)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_tokens += labels.numel()

        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

        # Validation
        model.eval()
        if has_memory and hasattr(model, 'reset_memory'):
            model.reset_memory()

        eval_loss = 0.0
        eval_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                if has_memory:
                    logits = model(input_ids, update_memory=False)
                else:
                    logits = model(input_ids)

                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )

                eval_loss += loss.item()
                eval_tokens += labels.numel()

        val_ppl = torch.exp(torch.tensor(eval_loss / eval_tokens)).item()
        elapsed = time.time() - start_time

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:7.1f}, val_ppl={val_ppl:7.1f} "
            f"({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    tokenizer_name: str,
    has_memory: bool = False,
) -> dict[str, Any]:
    """
    モデルを評価

    Returns:
        position_wise_ppl, reversal_curse
    """
    model.eval()

    if has_memory and hasattr(model, 'reset_memory'):
        model.reset_memory()

    # Position-wise PPL
    print_flush("\n  Position-wise PPL:")
    pos_ppl = evaluate_position_wise_ppl(model, val_loader, device)
    for pos_range, ppl in pos_ppl.items():
        print_flush(f"    {pos_range}: {ppl:.1f}")

    # Reversal Curse
    if has_memory and hasattr(model, 'reset_memory'):
        model.reset_memory()

    print_flush("\n  Reversal Curse:")
    tokenizer = get_tokenizer(tokenizer_name)
    reversal_pairs = get_reversal_pairs()
    reversal = evaluate_reversal_curse(model, tokenizer, reversal_pairs, device)
    print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
    print_flush(f"    Gap: {reversal['reversal_gap']:+.1f}")

    return {
        "position_wise_ppl": pos_ppl,
        "reversal_curse": reversal,
    }


def run_single_model_experiment(
    model_type: ModelType,
    exp_config: ExperimentConfig,
    device: torch.device,
    config: PythiaConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
) -> Optional[dict[str, Any]]:
    """
    単一モデルの実験を実行

    Returns:
        実験結果（best_val_ppl, best_epoch, evaluations, model_state_dict）
    """
    has_memory = model_type != ModelType.PYTHIA

    # Model name for display
    name_map = {
        ModelType.PYTHIA: "PYTHIA (RoPE)",
        ModelType.INFINI: "INFINI-PYTHIA",
        ModelType.MULTI_MEMORY: f"MULTI-MEMORY ({exp_config.num_memories} memories)",
        ModelType.HIERARCHICAL: f"HIERARCHICAL ({exp_config.num_memories} fine memories)",
    }
    model_name = name_map[model_type]

    print_flush("\n" + "=" * 70)
    print_flush(model_name)
    print_flush("=" * 70)

    # Create model
    model = create_model(model_type, config, exp_config)
    model = model.to(device)

    # Print model info
    param_info = model.num_parameters()
    print_flush(f"  Parameters: {param_info['total']:,}")

    if has_memory and hasattr(model, 'memory_info'):
        memory_info = model.memory_info()
        print_flush(f"  Memory: {memory_info['total_bytes']:,} bytes")

    if model_type == ModelType.HIERARCHICAL:
        print_flush(f"  Expansion gate: {param_info.get('expansion_gate', 0):,} params")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config.lr)

    print_flush(f"\n[{model_type.value}] Training...")
    best_ppl, best_epoch, best_state = train_model(
        model, train_loader, val_loader, optimizer, device,
        exp_config.num_epochs, has_memory=has_memory
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    # Load best weights and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
    evaluations = evaluate_model(
        model, val_loader, device, config.tokenizer_name, has_memory=has_memory
    )

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "model_state_dict": best_state,
        **evaluations,
    }

    # Cleanup
    del model
    clear_gpu_cache(device)

    return result


def print_summary(results: dict[str, Any], exp_config: ExperimentConfig) -> None:
    """結果サマリーを出力"""
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    # PPL comparison
    print_flush("\n| Model | Best PPL | Epoch |")
    print_flush("|-------|----------|-------|")

    for model_type in ModelType:
        key = model_type.value
        if results.get(key):
            r = results[key]
            print_flush(f"| {model_type.name} | {r['best_val_ppl']:.1f} | {r['best_epoch']} |")

    # Reversal Curse comparison
    print_flush("\n| Model | Forward PPL | Backward PPL | Gap |")
    print_flush("|-------|-------------|--------------|-----|")

    for model_type in ModelType:
        key = model_type.value
        if results.get(key):
            rev = results[key]["reversal_curse"]
            gap = rev["backward_ppl"] - rev["forward_ppl"]
            print_flush(
                f"| {model_type.name} | {rev['forward_ppl']:.1f} | "
                f"{rev['backward_ppl']:.1f} | {gap:+.1f} |"
            )


def run_experiment(
    model_types: list[ModelType],
    exp_config: Optional[ExperimentConfig] = None,
) -> dict[str, Any]:
    """
    実験を実行

    Args:
        model_types: 実行するモデルタイプのリスト
        exp_config: 実験設定（Noneならデフォルト）

    Returns:
        全モデルの結果を含む辞書
    """
    if exp_config is None:
        exp_config = ExperimentConfig()

    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    # Print experiment info
    print_flush("=" * 70)
    print_flush("EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Models: {[m.value for m in model_types]}")
    print_flush(f"Samples: {exp_config.num_samples:,}")
    print_flush(f"Sequence length: {exp_config.seq_length}")
    print_flush(f"Epochs: {exp_config.num_epochs}")
    print_flush(f"Learning rate: {exp_config.lr}")
    print_flush(f"Delta rule: {exp_config.use_delta_rule}")
    if exp_config.num_memories > 1:
        print_flush(f"Memories: {exp_config.num_memories}")
    print_flush("=" * 70)

    # Load data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=exp_config.num_samples,
        seq_length=exp_config.seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=exp_config.val_split,
        batch_size=exp_config.batch_size,
    )

    # Run experiments
    results: dict[str, Any] = {}

    for i, model_type in enumerate(model_types, 1):
        print_flush(f"\n{'='*70}")
        print_flush(f"[{i}/{len(model_types)}] {model_type.value.upper()}")

        result = run_single_model_experiment(
            model_type, exp_config, device, config, train_loader, val_loader
        )
        results[model_type.value] = result

    # Print summary
    print_summary(results, exp_config)

    return results
