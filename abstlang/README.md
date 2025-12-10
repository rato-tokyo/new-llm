# AbstLang - Abstract Language for CDR Training Data

抽象記号を用いたCDR（Context-Dependent Reasoning）訓練データ生成システム。

---

## 概要

AbstLangは、関係性を定義する簡潔なテキスト形式（`.abstlang`）から、
モデル訓練用のCDRデータを生成するためのシステム。

**目的**: Reversal Curse克服のための訓練データ生成

---

## ディレクトリ構造

```
abstlang/
├── README.md              # この仕様書
├── symbols.json           # 抽象記号のプール
├── specs/                 # AbstLang定義ファイル
│   └── family.abstlang    # 家族関係（親子）
├── generators/            # 生成スクリプト
│   └── family_generator.py
└── data/                  # 生成されたデータ
    └── family/
        └── cdr.json
```

---

## AbstLang仕様（.abstlang形式）

### 基本構造

```
# コメント（#で始まる行は無視）

[domain]
name = family

[relation]
forward = 親
backward = 子供

[templates]
knowledge = {A}は{B}の{forward}です。{B}は{A}の{backward}です。
forward_question = {X}の{forward}は誰ですか？
backward_question = {X}の{backward}は誰ですか？
answer = {Y}です。
no_info = {X}の{relation}に関する情報がありません。
unknown = {X}に関する情報がありません。
```

### セクション説明

| セクション | 説明 |
|------------|------|
| `[domain]` | ドメイン名（出力フォルダ名） |
| `[relation]` | 順方向・逆方向の関係名 |
| `[templates]` | 知識・質問・回答のテンプレート |

### プレースホルダ

| 記号 | 意味 |
|------|------|
| `{A}` | 順方向の主体（親） |
| `{B}` | 逆方向の主体（子供） |
| `{X}` | 質問対象 |
| `{Y}` | 回答対象 |
| `{forward}` | 順方向の関係名 |
| `{backward}` | 逆方向の関係名 |
| `{relation}` | 該当する関係名 |

---

## 生成されるQ&Aパターン（1ペアあたり6種類）

| # | 種類 | 質問例 | 回答例 |
|---|------|--------|--------|
| 1 | 順方向の正解 | Bの親は？ | Aです。 |
| 2 | 逆方向の正解 | Aの子供は？ | Bです。 |
| 3 | 順方向の情報なし | Aの親は？ | Aの親に関する情報がありません。 |
| 4 | 逆方向の情報なし | Bの子供は？ | Bの子供に関する情報がありません。 |
| 5 | 未知・順方向 | Pの親は？ | Pに関する情報がありません。 |
| 6 | 未知・逆方向 | Pの子供は？ | Pに関する情報がありません。 |

---

## 使用方法

### 1. AbstLang定義を確認・編集

```bash
cat abstlang/specs/family.abstlang
```

### 2. 生成スクリプトを実行

```bash
python3 abstlang/generators/family_generator.py --num-pairs 10
```

### 3. 生成されたデータを確認

```bash
cat abstlang/data/family/cdr.json
```

---

## symbols.json

抽象記号のプール。具体的な名前（田中、山田）ではなく、
意味を持たない記号（JJ, ZB, ◯）を使用することで、
モデルが関係のパターンを学習することを促す。

```json
{
  "description": "CDRデータ生成用の抽象キーワード",
  "seed": 42,
  "symbols": ["JJ", "ZB", "YD", "◯", "✗", ...]
}
```

---

## 出力形式（cdr.json）

```json
{
  "description": "家族関係データ（CDR訓練用）",
  "knowledge": "JJはZBの親です。ZBはJJの子供です。...",
  "samples": [
    {"question": "ZBの親は誰ですか？", "answer": "JJです。"},
    {"question": "JJの子供は誰ですか？", "answer": "ZBです。"},
    ...
  ]
}
```

---

## 訓練スクリプトとの連携

`scripts/quick_model.py` は `abstlang/data/family/cdr.json` を参照。

```python
cdr_path = Path("abstlang/data/family/cdr.json")
```

---

## バージョン

- v1.0.0 (2025-12-10): 初版、senri-fine-tunerから移行
