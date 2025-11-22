# New-LLM Refactoring Plan

## 🎯 現状の問題点

### 1. コード重複
- **Train Phase 1とVal Phase 1が完全に別実装**（約150行の重複）
- バグ修正が2箇所必要 → 修正漏れが発生
- bb99c24のCVFPバグ修正がValに適用されていない

### 2. 過度に複雑なファイル構成
- `test_residual.py`: 981行（巨大すぎる）
- 複数の責務が混在：データローディング、Phase 1訓練、Phase 2訓練、分析、評価

### 3. 不明確なアーキテクチャ
- 共通ロジックの抽出不足
- DRY原則違反が多数
- テストしにくい構造

### 4. ドキュメントの肥大化
- CLAUDE.md: 407行
- README.md: 巨大
- 古い情報と新しい情報が混在

---

## ✅ 移行戦略

### Phase A: クリーンな新実装の作成（既存コード保持）

**保持するファイル**:
- `config.py` - 設定は維持
- `src/models/new_llm_residual.py` - モデル定義は維持
- `src/utils/early_stopping.py` - ユーティリティは維持
- `cache/` - キャッシュデータは維持
- `.git/` - Git履歴は維持

**新規作成するファイル**:
```
new-llm/
├── train.py                    # メインエントリーポイント（シンプル）
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── phase1.py          # Phase 1共通実装（Train/Val統合）
│   │   └── phase2.py          # Phase 2実装
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # データローディング統合
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py         # 固定点分析・メトリクス
├── tests/
│   └── test_training.py       # 統合テスト
└── docs/
    ├── ARCHITECTURE.md        # アーキテクチャ概要のみ
    └── EXPERIMENTS.md         # 実験結果記録のみ
```

**削除するファイル**（新実装完了後）:
```
tests/phase2_experiments/test_residual.py  # 981行の巨大ファイル
tests/phase2_experiments/phase1_common.py  # 不完全な共通化
CLAUDE.md                                   # 過度に詳細
README.md                                   # 再構築
```

---

## 📋 新実装の設計方針

### 1. シングルソース原則
```python
# src/training/phase1.py
def train_phase1(model, token_ids, config, device, is_training=True):
    """
    Phase 1: CVFP Fixed-Point Learning

    - is_training=True: Train with backprop
    - is_training=False: Eval only (Val)

    Returns: fixed_contexts
    """
    # 1つの実装で Train/Val 両方カバー
```

### 2. クリーンな責務分離
```python
# train.py - メインフロー
from src.data.loader import load_data
from src.training.phase1 import train_phase1
from src.training.phase2 import train_phase2
from src.evaluation.metrics import analyze_fixed_points

# シンプルなフロー
train_data, val_data = load_data(config)
train_contexts = train_phase1(model, train_data, config, is_training=True)
val_contexts = train_phase1(model, val_data, config, is_training=False)
analyze_fixed_points(train_contexts, val_contexts)
train_phase2(model, train_data, train_contexts, config)
```

### 3. 最小限のドキュメント
- **ARCHITECTURE.md**: CVFP原理とDistribution Regularizationのみ（50行程度）
- **EXPERIMENTS.md**: 実験結果の記録のみ（追記式）
- **README.md**: Quick Start + 基本説明のみ（100行以内）

---

## 🔄 実装手順

### Step 1: 新しいディレクトリ構造を作成
```bash
mkdir -p src/training src/data src/evaluation
touch src/training/__init__.py src/data/__init__.py src/evaluation/__init__.py
```

### Step 2: Phase 1共通実装（最優先）
`src/training/phase1.py`を作成:
- Train/Val統合
- Distribution Regularization正しく実装（`unbiased=False`）
- CVFPロジック修正（前回の固定点と正しく比較）

### Step 3: データローダー統合
`src/data/loader.py`を作成:
- UltraChat、text_file、text_dir、manualを統合
- 重複コード削除

### Step 4: メトリクス・分析
`src/evaluation/metrics.py`を作成:
- Fixed-point分析
- Effective Rank計算
- Singular vector分析

### Step 5: メインスクリプト
`train.py`を作成:
- シンプルなエントリーポイント
- 100行以内

### Step 6: 動作確認
```bash
python train.py --context-dim 16 --num-layers 2 --num-samples 10
```

### Step 7: 古いファイル削除
動作確認後:
```bash
rm tests/phase2_experiments/test_residual.py
rm tests/phase2_experiments/phase1_common.py
rm CLAUDE.md
```

### Step 8: ドキュメント整備
- ARCHITECTURE.md作成（簡潔）
- README.md再構築（簡潔）
- EXPERIMENTS.md作成（実験記録）

---

## 🎯 期待される成果

### コード品質
- ✅ Phase 1実装: 1箇所のみ（約150行）
- ✅ メインスクリプト: 100行以内
- ✅ 総行数: 現在の50%以下

### 保守性
- ✅ バグ修正: 1箇所のみ
- ✅ 新機能追加: 明確な場所
- ✅ テスト: 容易

### 開発速度
- ✅ 新実験: 設定変更のみ
- ✅ デバッグ: シンプルな構造
- ✅ 理解: 新規開発者も容易

---

## ⚠️ リスク管理

### バックアップ
```bash
# 現在のコードを別ブランチに保存
git checkout -b backup-before-refactoring
git commit -am "Backup before major refactoring"
git checkout main
```

### 段階的移行
1. 新実装を`src/`に追加（既存コード保持）
2. 新実装で動作確認
3. 結果が一致することを確認
4. 既存コード削除

### ロールバック計画
問題があれば:
```bash
git checkout backup-before-refactoring
```

---

## 📅 推定作業時間

- **Step 1-2 (Phase 1共通化)**: 30分
- **Step 3-4 (データ・メトリクス)**: 20分
- **Step 5-6 (メイン・テスト)**: 20分
- **Step 7-8 (削除・ドキュメント)**: 10分

**合計**: 約1.5時間

---

## ✅ 完了条件

### 機能要件
- [ ] Phase 1: Train/Val統一実装
- [ ] Distribution Regularization正しく動作（`unbiased=False`）
- [ ] CVFPバグ修正済み
- [ ] UltraChat + manual validation動作
- [ ] layer=2, dim=16, samples=10で動作確認

### 品質要件
- [ ] コード重複なし
- [ ] メインスクリプト100行以内
- [ ] ドキュメント合計200行以内
- [ ] 全機能テスト済み

### 削除確認
- [ ] `test_residual.py`削除
- [ ] `phase1_common.py`削除
- [ ] 古いCLAUDE.md削除
- [ ] 不要なファイル全削除

---

## 🚀 次のステップ

1. **このPlanをレビュー**
2. **`/compact`実行**（コンテキスト圧縮）
3. **Step 1から順次実装**
4. **各Stepで動作確認**
5. **完了条件チェック**

---

**Quality over Quantity. Clean Code is Fast Code.**
