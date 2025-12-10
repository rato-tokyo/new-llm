#!/usr/bin/env python3
"""
家族関係（親子）CDRデータ生成スクリプト

abstlang/specs/family.abstlang を読み込んでデータを生成。

1ペアから6種類のQ&Aを生成:
  1. 順方向の正解（Bの親は？→ A）
  2. 逆方向の正解（Aの子供は？→ B）
  3. 順方向の情報なし（Aの親は？→ 情報なし）
  4. 逆方向の情報なし（Bの子供は？→ 情報なし）
  5. 未知の人物・順方向（Pの親は？→ Pに関する情報なし）
  6. 未知の人物・逆方向（Pの子供は？→ Pに関する情報なし）

Usage:
    python3 abstlang/generators/family_generator.py --num-pairs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from abstlang.parser import AbstLangSpec, parse_abstlang


ABSTLANG_FILE = Path(__file__).parent.parent / "specs" / "family.abstlang"
SYMBOLS_FILE = Path(__file__).parent.parent / "symbols.json"


def load_symbols() -> List[str]:
    """symbols.jsonから抽象キーワードを読み込み"""
    with open(SYMBOLS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["symbols"]


def create_cdr_data(spec: AbstLangSpec, num_pairs: int, symbols: List[str]) -> dict:
    """
    CDR訓練用データを生成（1ペアから6種類のQ&A）

    Args:
        spec: AbstLangSpec（.abstlangファイルからパースした定義）
        num_pairs: 生成するペア数
        symbols: 抽象記号のリスト

    Returns:
        {"knowledge": str, "samples": list[dict]}
    """
    # ペア数に対して十分な記号があるか確認
    # 各ペアに2個（A, B）+ 未知用に1個（P）
    required = num_pairs * 3
    if required > len(symbols):
        raise ValueError(f"Not enough symbols: need {required}, have {len(symbols)}")

    knowledge_parts = []
    samples = []

    # ペアに使う記号と未知用の記号を分離
    pair_symbols = symbols[: num_pairs * 2]
    unknown_symbols = symbols[num_pairs * 2 : num_pairs * 3]

    for i in range(num_pairs):
        # A: 親、B: 子供、P: 未知の人物
        a = pair_symbols[i * 2]  # 親
        b = pair_symbols[i * 2 + 1]  # 子供
        p = unknown_symbols[i]  # 未知

        # Knowledge: 双方向の関係（abstlangテンプレートを使用）
        knowledge_parts.append(spec.render_knowledge(a, b))

        # 1. 順方向の正解: Bの親は？→ A
        samples.append(
            {
                "question": spec.render_forward_question(b),
                "answer": spec.render_answer(a),
            }
        )

        # 2. 逆方向の正解: Aの子供は？→ B
        samples.append(
            {
                "question": spec.render_backward_question(a),
                "answer": spec.render_answer(b),
            }
        )

        # 3. 順方向の情報なし: Aの親は？→ 情報なし
        samples.append(
            {
                "question": spec.render_forward_question(a),
                "answer": spec.render_no_info(a, spec.forward),
            }
        )

        # 4. 逆方向の情報なし: Bの子供は？→ 情報なし
        samples.append(
            {
                "question": spec.render_backward_question(b),
                "answer": spec.render_no_info(b, spec.backward),
            }
        )

        # 5. 未知の人物・順方向: Pの親は？→ Pに関する情報なし
        samples.append(
            {
                "question": spec.render_forward_question(p),
                "answer": spec.render_unknown(p),
            }
        )

        # 6. 未知の人物・逆方向: Pの子供は？→ Pに関する情報なし
        samples.append(
            {
                "question": spec.render_backward_question(p),
                "answer": spec.render_unknown(p),
            }
        )

    return {
        "knowledge": "".join(knowledge_parts),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="家族関係（親子）CDRデータ生成")
    parser.add_argument("--num-pairs", type=int, default=10, help="生成するペア数")
    parser.add_argument(
        "--abstlang",
        type=str,
        default=str(ABSTLANG_FILE),
        help="AbstLangファイルのパス",
    )
    args = parser.parse_args()

    # AbstLang定義を読み込み
    spec = parse_abstlang(args.abstlang)
    print(f"Loaded AbstLang: {args.abstlang}")
    print(f"  Domain: {spec.domain}")
    print(f"  Relation: {spec.forward} / {spec.backward}")

    # 記号を読み込み
    symbols = load_symbols()
    print(f"  Symbols: {len(symbols)} available")
    print(f"Generating {args.num_pairs} pairs (6 Q&A each)...")

    # CDRデータ生成
    cdr_data = create_cdr_data(spec, args.num_pairs, symbols)

    # 出力データ
    output_data = {
        "description": spec.description,
        "knowledge": cdr_data["knowledge"],
        "samples": cdr_data["samples"],
    }

    # 出力先: abstlang/data/{domain}/cdr.json
    output_dir = Path(__file__).parent.parent / "data" / spec.domain
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cdr.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"  Pairs: {args.num_pairs}")
    print(f"  Samples: {len(cdr_data['samples'])} (6 per pair)")

    # サンプル表示（1ペア分の6種類）
    print("\n--- Knowledge (first 100 chars) ---")
    print(f"{cdr_data['knowledge'][:100]}...")
    print("\n--- Sample Q&A (first pair) ---")
    for i, s in enumerate(cdr_data["samples"][:6]):
        print(f"[{i + 1}] Q: {s['question']}")
        print(f"    A: {s['answer']}")


if __name__ == "__main__":
    main()
