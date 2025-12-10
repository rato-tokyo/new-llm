#!/usr/bin/env python3
"""
家族関係（親子）CDRデータ生成スクリプト

abstlang/specs/family.abstlang の定義に基づいて手動実装。

AbstLang定義:
  非対称人間関係(親)
  非対称人間関係(子)
  対(親, 子)

  親(a, b) = TRUE → 親(b, a) = FALSE
  ∀v ∈ BOOL: 親(a, b) = v ↔ 子(b, a) = v

1ペアから6種類のQ&Aを生成:
  1. 順方向の正解（Bの親は？→ A）
  2. 逆方向の正解（Aの子は？→ B）
  3. 順方向の情報なし（Aの親は？→ 情報なし）
  4. 逆方向の情報なし（Bの子は？→ 情報なし）
  5. 未知の人物・順方向（Pの親は？→ Pに関する情報なし）
  6. 未知の人物・逆方向（Pの子は？→ Pに関する情報なし）

Usage:
    python3 abstlang/generators/family_generator.py --num-pairs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


# AbstLang定義より（family.abstlangを参照）
FORWARD_RELATION = "親"  # 非対称人間関係(親)
BACKWARD_RELATION = "子"  # 非対称人間関係(子)、対(親, 子)
DESCRIPTION = "家族関係データ（CDR訓練用、Reversal Curse実験）"

SYMBOLS_FILE = Path(__file__).parent.parent / "symbols.json"


def load_symbols() -> List[str]:
    """symbols.jsonから抽象キーワードを読み込み"""
    with open(SYMBOLS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["symbols"]


def create_cdr_data(num_pairs: int, symbols: List[str]) -> dict:
    """
    CDR訓練用データを生成（1ペアから6種類のQ&A）

    AbstLang推論規則に基づく:
    - 親(A, B) = TRUE のとき:
      - 子(B, A) = TRUE （対(親, 子)より）
      - 親(B, A) = FALSE （非対称より）
    - 親(A, ?) = NULL → 情報なし
    - 親(P, ?) where P is unknown → Pに関する情報なし

    Args:
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
        # A: 親、B: 子、P: 未知の人物
        a = pair_symbols[i * 2]  # 親
        b = pair_symbols[i * 2 + 1]  # 子
        p = unknown_symbols[i]  # 未知

        # Knowledge: 親(A, B) = TRUE の自然言語表現
        # "aはbの親である" （人間関係(親)の定義より）
        knowledge_parts.append(f"{a}は{b}の{FORWARD_RELATION}である。")
        # 対(親, 子)より: 子(B, A) = TRUE → "bはaの子である"
        knowledge_parts.append(f"{b}は{a}の{BACKWARD_RELATION}である。")

        # 1. 順方向の正解: 親(A, B) = TRUE
        # Q: "Bの親は誰？" A: "A"
        samples.append(
            {
                "question": f"{b}の{FORWARD_RELATION}は誰ですか？",
                "answer": f"{a}です。",
            }
        )

        # 2. 逆方向の正解: 子(B, A) = TRUE （対(親,子)より導出）
        # Q: "Aの子は誰？" A: "B"
        samples.append(
            {
                "question": f"{a}の{BACKWARD_RELATION}は誰ですか？",
                "answer": f"{b}です。",
            }
        )

        # 3. 順方向の情報なし: 親(A, ?) = NULL
        # Q: "Aの親は誰？" A: "Aの親に関する情報がありません"
        samples.append(
            {
                "question": f"{a}の{FORWARD_RELATION}は誰ですか？",
                "answer": f"{a}の{FORWARD_RELATION}に関する情報がありません。",
            }
        )

        # 4. 逆方向の情報なし: 子(B, ?) = NULL
        # Q: "Bの子は誰？" A: "Bの子に関する情報がありません"
        samples.append(
            {
                "question": f"{b}の{BACKWARD_RELATION}は誰ですか？",
                "answer": f"{b}の{BACKWARD_RELATION}に関する情報がありません。",
            }
        )

        # 5. 未知の人物・順方向: P is unknown
        # Q: "Pの親は誰？" A: "Pに関する情報がありません"
        samples.append(
            {
                "question": f"{p}の{FORWARD_RELATION}は誰ですか？",
                "answer": f"{p}に関する情報がありません。",
            }
        )

        # 6. 未知の人物・逆方向: P is unknown
        # Q: "Pの子は誰？" A: "Pに関する情報がありません"
        samples.append(
            {
                "question": f"{p}の{BACKWARD_RELATION}は誰ですか？",
                "answer": f"{p}に関する情報がありません。",
            }
        )

    return {
        "knowledge": "".join(knowledge_parts),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="家族関係（親子）CDRデータ生成")
    parser.add_argument("--num-pairs", type=int, default=10, help="生成するペア数")
    args = parser.parse_args()

    # 記号を読み込み
    symbols = load_symbols()

    print(f"AbstLang: family.abstlang")
    print(f"  Relations: {FORWARD_RELATION} / {BACKWARD_RELATION}")
    print(f"  Symbols: {len(symbols)} available")
    print(f"Generating {args.num_pairs} pairs (6 Q&A each)...")

    # CDRデータ生成
    cdr_data = create_cdr_data(args.num_pairs, symbols)

    # 出力データ
    output_data = {
        "description": DESCRIPTION,
        "knowledge": cdr_data["knowledge"],
        "samples": cdr_data["samples"],
    }

    # 出力先: abstlang/data/family/cdr.json
    output_dir = Path(__file__).parent.parent / "data" / "family"
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
