# AbstLang - Abstract Language for CDR Training Data

抽象記号を用いたCDR（Context-Dependent Reasoning）訓練データ生成システム。

**仕様書**: [docs/abstlang.md](../docs/abstlang.md)

---

## 概要

AbstLangは、人間の知識や関係性を形式的に定義し、LLMが推論可能な形式で表現するための言語である。

**目的**: Reversal Curse克服のための訓練データ生成

**特徴**:
- 3値論理（TRUE, FALSE, NULL）による知識状態の表現
- 関係の対称性/非対称性の形式的定義
- 自然言語への変換規則
- 推論規則の明示的記述

---

## ディレクトリ構造

```
abstlang/
├── README.md              # この説明書
├── symbols.json           # 抽象記号のプール
├── specs/                 # AbstLang定義ファイル
│   └── family.abstlang    # 家族関係（親子）
├── generators/            # 生成スクリプト
│   └── family_generator.py
└── data/                  # 生成されたデータ（.gitignore）
    └── family/
        └── cdr.json
```

---

## AbstLang記法（概要）

詳細は [docs/abstlang.md](../docs/abstlang.md) を参照。

### 基本構文

```
% コメント

% 型定義
BOOL := {TRUE, FALSE, NULL}
人物 := {太郎, 花子, ...}

% 関係の宣言
非対称人間関係(親)
非対称人間関係(子)
対(親, 子)

% 推論規則
∀a,b ∈ 人物:
  親(a, b) = TRUE → 親(b, a) = FALSE

% 相互変換
∀a,b ∈ 人物:
  ∀v ∈ BOOL: 親(a, b) = v ↔ 子(b, a) = v
```

### 関係の種類

| 種類 | 説明 | 例 |
|------|------|-----|
| 非対称人間関係 | R(a,b)=TRUE → R(b,a)=FALSE | 親, 子, 上司, 部下 |
| 対称人間関係 | R(a,b)=v ↔ R(b,a)=v | 親子関係, 友人関係 |
| 対(R1, R2) | R1(a,b)=v ↔ R2(b,a)=v | 対(親, 子) |

---

## ワークフロー

1. **AbstLang定義を作成**: `specs/*.abstlang` に形式論理で関係を定義
2. **AIがジェネレーターを作成**: AbstLang定義を読み、対応するPythonスクリプトを生成
3. **データ生成**: ジェネレーターを実行してCDRデータを生成

```
[人間/AI] → specs/family.abstlang（形式論理）
    ↓
[AI] → generators/family_generator.py（Pythonスクリプト）
    ↓
[実行] → data/family/cdr.json（訓練データ）
```

**注意**: パーサーによる自動変換ではなく、AIが定義を理解してスクリプトを作成する運用。

---

## 使用方法

### 1. AbstLang定義を確認

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

## 出力形式（cdr.json）

```json
{
  "description": "家族関係データ（CDR訓練用）",
  "knowledge": "JJはZBの親である。ZBはJJの子である。...",
  "samples": [
    {"question": "ZBの親は誰ですか？", "answer": "JJです。"},
    {"question": "JJの子は誰ですか？", "answer": "ZBです。"},
    ...
  ]
}
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

## バージョン

- v1.0.0 (2025-12-10): 初版、senri-fine-tunerから移行
- v1.1.0 (2025-12-10): docs/abstlang.md仕様に準拠、形式論理記法に変更
