#!/usr/bin/env python3
"""
AbstLangパーサー

.abstlangファイルを読み込み、構造化データに変換する。

Usage:
    from abstlang.parser import parse_abstlang

    spec = parse_abstlang("abstlang/specs/family.abstlang")
    print(spec.domain)       # "family"
    print(spec.forward)      # "親"
    print(spec.backward)     # "子供"
    print(spec.templates)    # {"knowledge": "...", "forward_question": "...", ...}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class AbstLangSpec:
    """AbstLang定義の構造化データ"""

    domain: str = ""
    forward: str = ""
    backward: str = ""
    templates: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def render_knowledge(self, a: str, b: str) -> str:
        """知識文を生成（{A}, {B}, {forward}, {backward}を置換）"""
        template = self.templates.get("knowledge", "")
        return self._render(template, a=a, b=b)

    def render_forward_question(self, x: str) -> str:
        """順方向の質問を生成"""
        template = self.templates.get("forward_question", "")
        return self._render(template, x=x)

    def render_backward_question(self, x: str) -> str:
        """逆方向の質問を生成"""
        template = self.templates.get("backward_question", "")
        return self._render(template, x=x)

    def render_answer(self, y: str) -> str:
        """回答を生成"""
        template = self.templates.get("answer", "")
        return self._render(template, y=y)

    def render_no_info(self, x: str, relation: str) -> str:
        """情報なし回答を生成"""
        template = self.templates.get("no_info", "")
        return template.replace("{X}", x).replace("{relation}", relation)

    def render_unknown(self, x: str) -> str:
        """未知の人物回答を生成"""
        template = self.templates.get("unknown", "")
        return template.replace("{X}", x)

    def _render(self, template: str, **kwargs: str) -> str:
        """テンプレートをレンダリング"""
        result = template
        # 小文字プレースホルダー（{a}, {b}, {x}, {y}）
        for key, value in kwargs.items():
            result = result.replace("{" + key.upper() + "}", value)
            result = result.replace("{" + key.lower() + "}", value)
        # 関係プレースホルダー
        result = result.replace("{forward}", self.forward)
        result = result.replace("{backward}", self.backward)
        return result


def parse_abstlang(filepath: Union[str, Path]) -> AbstLangSpec:
    """
    .abstlangファイルをパースしてAbstLangSpecを返す

    Args:
        filepath: .abstlangファイルのパス

    Returns:
        AbstLangSpec: パース結果
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"AbstLang file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8")

    spec = AbstLangSpec()
    current_section = None
    description_lines: list[str] = []

    for line in content.splitlines():
        line = line.strip()

        # 空行・コメント行をスキップ
        if not line or line.startswith("#"):
            continue

        # セクションヘッダー
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            continue

        # キー = 値 の形式
        if "=" in line and current_section != "description":
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if current_section == "domain":
                if key == "name":
                    spec.domain = value

            elif current_section == "relation":
                if key == "forward":
                    spec.forward = value
                elif key == "backward":
                    spec.backward = value

            elif current_section == "templates":
                spec.templates[key] = value

        # descriptionセクションは複数行テキスト
        elif current_section == "description":
            description_lines.append(line)

    spec.description = "\n".join(description_lines)

    return spec


if __name__ == "__main__":
    # テスト
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <abstlang_file>")
        sys.exit(1)

    spec = parse_abstlang(sys.argv[1])
    print(f"Domain: {spec.domain}")
    print(f"Forward relation: {spec.forward}")
    print(f"Backward relation: {spec.backward}")
    print(f"Templates: {spec.templates}")
    print(f"Description: {spec.description}")

    # レンダリングテスト
    print("\n--- Render Test ---")
    print(f"Knowledge: {spec.render_knowledge('Alice', 'Bob')}")
    print(f"Forward Q: {spec.render_forward_question('Bob')}")
    print(f"Backward Q: {spec.render_backward_question('Alice')}")
    print(f"Answer: {spec.render_answer('Alice')}")
    print(f"No info: {spec.render_no_info('Alice', spec.forward)}")
    print(f"Unknown: {spec.render_unknown('Charlie')}")
