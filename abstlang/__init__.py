"""AbstLang - Abstract Language for CDR data generation

仕様書: docs/abstlang.md

ワークフロー:
1. specs/*.abstlang に形式論理で関係を定義
2. AIが定義を読み、generators/*.py を作成
3. ジェネレーターを実行して data/*/cdr.json を生成
"""
