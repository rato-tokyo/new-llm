#!/usr/bin/env python3
"""
Family Relation Data Generator for CDR Training

Reversal Curse 汎化性能仮説の検証用データを生成。
JSONファイルを出力し、quick_model.py --cdr-data で使用可能。

Usage:
    cd senri-fine-tuner
    python3 scripts/generate_family_data.py

    # カスタム設定
    python3 scripts/generate_family_data.py --num-pairs 100 --output data/family_100.json
"""

import argparse
import json
import sys
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from family_relations import (
    generate_family_pairs,
    split_pairs_for_experiment,
    FamilyPair,
)


def create_cdr_samples(pairs: list[FamilyPair], include_backward: bool = True) -> list[dict]:
    """
    CDR訓練用サンプルを生成

    Args:
        pairs: 親子ペアのリスト
        include_backward: 逆方向サンプルを含めるか

    Returns:
        CDRサンプルのリスト（context + target形式）
    """
    samples = []

    for pair in pairs:
        # 順方向: "X is Y's parent." → "Who is Y's parent? X"
        samples.append({
            "context": f"{pair.parent_name} is {pair.child_name}'s {pair.relation}.",
            "target": f"Who is {pair.child_name}'s parent? {pair.parent_name}",
        })

        if include_backward:
            # 逆方向: "Y is X's child." → "Who is X's child? Y"
            samples.append({
                "context": f"{pair.child_name} is {pair.parent_name}'s child.",
                "target": f"Who is {pair.parent_name}'s child? {pair.child_name}",
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate Family Relation Data for CDR Training")
    parser.add_argument(
        "--num-pairs", type=int, default=50,
        help="Number of family pairs to generate (default: 50)"
    )
    parser.add_argument(
        "--num-val-pairs", type=int, default=10,
        help="Number of validation pairs (forward only, for generalization test) (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default="data/family_cdr.json",
        help="Output JSON file path (default: data/family_cdr.json)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    print(f"Generating {args.num_pairs} family pairs (seed={args.seed})...")

    # ペア生成
    all_pairs = generate_family_pairs(args.num_pairs, seed=args.seed)
    pattern_pairs, val_pairs = split_pairs_for_experiment(all_pairs, args.num_val_pairs)

    print(f"  Pattern pairs: {len(pattern_pairs)} (forward + backward)")
    print(f"  Val pairs: {len(val_pairs)} (forward only, for generalization test)")

    # CDRサンプル生成
    pattern_samples = create_cdr_samples(pattern_pairs, include_backward=True)
    val_samples = create_cdr_samples(val_pairs, include_backward=False)

    # 統合
    all_samples = pattern_samples + val_samples

    # メタデータ追加
    output_data = {
        "metadata": {
            "description": "Family relation data for CDR training (Reversal Curse experiment)",
            "num_pattern_pairs": len(pattern_pairs),
            "num_val_pairs": len(val_pairs),
            "num_pattern_samples": len(pattern_samples),
            "num_val_samples": len(val_samples),
            "seed": args.seed,
        },
        "samples": all_samples,
    }

    # 出力ディレクトリ作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON出力
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(all_samples)} CDR samples:")
    print(f"  Pattern samples: {len(pattern_samples)} (forward + backward)")
    print(f"  Val samples: {len(val_samples)} (forward only)")
    print(f"\nSaved to: {output_path}")

    # サンプル表示
    print("\n--- Sample entries ---")
    for i, sample in enumerate(all_samples[:3]):
        print(f"[{i}] context: {sample['context']}")
        print(f"    target: {sample['target']}")


if __name__ == "__main__":
    main()
