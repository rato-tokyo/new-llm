#!/usr/bin/env python3
"""
Unified Experiment Script

全モデルタイプ（Pythia, Infini, Multi-Memory, Hierarchical）に対応した統一実験スクリプト。

Usage:
    # 全モデル比較
    python3 scripts/experiment.py --models pythia infini multi_memory hierarchical

    # Infiniのみ
    python3 scripts/experiment.py --models infini

    # Multi-MemoryとHierarchicalの比較
    python3 scripts/experiment.py --models multi_memory hierarchical --num-memories 4

    # 設定カスタマイズ
    python3 scripts/experiment.py --models infini --samples 10000 --epochs 50 --lr 5e-5

    # ALiBi付きInfini
    python3 scripts/experiment.py --models infini --alibi
"""

import argparse
import sys

# Add project root to path
sys.path.insert(0, ".")

from src.utils.experiment_runner import (  # noqa: E402
    ExperimentConfig,
    ModelType,
    run_experiment,
)
from src.utils.io import print_flush  # noqa: E402


def parse_model_types(model_names: list[str]) -> list[ModelType]:
    """モデル名をModelTypeに変換"""
    type_map = {
        "pythia": ModelType.PYTHIA,
        "infini": ModelType.INFINI,
        "multi_memory": ModelType.MULTI_MEMORY,
        "multi-memory": ModelType.MULTI_MEMORY,
        "hierarchical": ModelType.HIERARCHICAL,
    }

    model_types = []
    for name in model_names:
        name_lower = name.lower()
        if name_lower not in type_map:
            raise ValueError(
                f"Unknown model type: {name}. "
                f"Available: {list(type_map.keys())}"
            )
        model_types.append(type_map[name_lower])

    return model_types


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Experiment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all models
  python3 scripts/experiment.py --models pythia infini multi_memory hierarchical

  # Just Infini with ALiBi
  python3 scripts/experiment.py --models infini --alibi

  # Multi-Memory vs Hierarchical
  python3 scripts/experiment.py --models multi_memory hierarchical --num-memories 8
        """,
    )

    # Model selection
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model types to run: pythia, infini, multi_memory, hierarchical"
    )

    # Data settings
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")

    # Training settings
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Model settings
    parser.add_argument(
        "--no-delta-rule", action="store_true",
        help="Disable delta rule for memory models"
    )
    parser.add_argument(
        "--num-memories", type=int, default=4,
        help="Number of memories (for multi_memory and hierarchical)"
    )

    # ALiBi settings (for infini)
    parser.add_argument(
        "--alibi", action="store_true",
        help="Enable ALiBi position encoding (infini only)"
    )
    parser.add_argument(
        "--alibi-scale", type=float, default=1.0,
        help="ALiBi slope scale factor"
    )

    # Long context settings
    parser.add_argument(
        "--long-context-train", action="store_true",
        help="Train on long documents"
    )
    parser.add_argument(
        "--long-context-eval", action="store_true",
        help="Evaluate on long context"
    )
    parser.add_argument(
        "--num-long-docs", type=int, default=50,
        help="Number of long documents"
    )
    parser.add_argument(
        "--tokens-per-doc", type=int, default=4096,
        help="Tokens per document for long context"
    )

    args = parser.parse_args()

    # Parse model types
    try:
        model_types = parse_model_types(args.models)
    except ValueError as e:
        print_flush(f"Error: {e}")
        sys.exit(1)

    # Create experiment config
    exp_config = ExperimentConfig(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_delta_rule=not args.no_delta_rule,
        num_memories=args.num_memories,
        use_alibi=args.alibi,
        alibi_scale=args.alibi_scale,
        long_context_train=args.long_context_train,
        long_context_eval=args.long_context_eval,
        num_long_documents=args.num_long_docs,
        tokens_per_document=args.tokens_per_doc,
    )

    # Run experiment
    run_experiment(model_types, exp_config)

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
