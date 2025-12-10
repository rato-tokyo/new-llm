#!/usr/bin/env python3
"""
家族関係（親子）CDRデータ生成スクリプト

abstlang/specs/family.abstlang の定義に基づいてデータを生成。

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

import argparse
import json
from pathlib import Path


# AbstLang定義（family.abstlangの内容を反映）
DOMAIN = "family"
FORWARD_RELATION = "親"
BACKWARD_RELATION = "子供"
DESCRIPTION = "家族関係データ（CDR訓練用、Reversal Curse実験）"


def load_symbols() -> list[str]:
    """symbols.jsonから抽象キーワードを読み込み"""
    symbols_path = Path(__file__).parent.parent / "symbols.json"
    with open(symbols_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["symbols"]


def create_cdr_data(num_pairs: int, symbols: list[str]) -> dict:
    """CDR訓練用データを生成（1ペアから6種類のQ&A）"""
    # ペア数に対して十分な記号があるか確認
    # 各ペアに2個（A, B）+ 未知用に1個（P）
    required = num_pairs * 3
    if required > len(symbols):
        raise ValueError(f"Not enough symbols: need {required}, have {len(symbols)}")

    knowledge_parts = []
    samples = []

    # ペアに使う記号と未知用の記号を分離
    pair_symbols = symbols[:num_pairs * 2]
    unknown_symbols = symbols[num_pairs * 2:num_pairs * 3]

    for i in range(num_pairs):
        # A: 親、B: 子供、P: 未知の人物
        a = pair_symbols[i * 2]      # 親
        b = pair_symbols[i * 2 + 1]  # 子供
        p = unknown_symbols[i]        # 未知

        # Knowledge: 双方向の関係
        knowledge_parts.append(f"{a}は{b}の{FORWARD_RELATION}です。")
        knowledge_parts.append(f"{b}は{a}の{BACKWARD_RELATION}です。")

        # 1. 順方向の正解: Bの親は？→ A
        samples.append({
            "question": f"{b}の{FORWARD_RELATION}は誰ですか？",
            "answer": f"{a}です。",
        })

        # 2. 逆方向の正解: Aの子供は？→ B
        samples.append({
            "question": f"{a}の{BACKWARD_RELATION}は誰ですか？",
            "answer": f"{b}です。",
        })

        # 3. 順方向の情報なし: Aの親は？→ 情報なし
        samples.append({
            "question": f"{a}の{FORWARD_RELATION}は誰ですか？",
            "answer": f"{a}の{FORWARD_RELATION}に関する情報がありません。",
        })

        # 4. 逆方向の情報なし: Bの子供は？→ 情報なし
        samples.append({
            "question": f"{b}の{BACKWARD_RELATION}は誰ですか？",
            "answer": f"{b}の{BACKWARD_RELATION}に関する情報がありません。",
        })

        # 5. 未知の人物・順方向: Pの親は？→ Pに関する情報なし
        samples.append({
            "question": f"{p}の{FORWARD_RELATION}は誰ですか？",
            "answer": f"{p}に関する情報がありません。",
        })

        # 6. 未知の人物・逆方向: Pの子供は？→ Pに関する情報なし
        samples.append({
            "question": f"{p}の{BACKWARD_RELATION}は誰ですか？",
            "answer": f"{p}に関する情報がありません。",
        })

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

    print(f"Domain: {DOMAIN}")
    print(f"  Relation: {FORWARD_RELATION} / {BACKWARD_RELATION}")
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
    output_dir = Path(__file__).parent.parent / "data" / DOMAIN
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
        print(f"[{i+1}] Q: {s['question']}")
        print(f"    A: {s['answer']}")


if __name__ == "__main__":
    main()
