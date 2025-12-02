# 実験サマリー 2025年12月

**最終更新**: 2025-12-02

---

## 🏆 最終採用アーキテクチャ: Cascade Context

### 構成

```
ContextBlock A (cd=500) → ContextBlock B (cd=500) → concat → TokenBlock (cd=1000)
```

### 性能

| 指標 | 値 |
|------|-----|
| **Val PPL** | **111.9** |
| **Val Acc** | **25.6%** |
| 実効次元 | 736/1000 |
| パラメータ | 41.2M |

### 特徴

- 1層固定（multi-layer削除済み）
- Context A → B のカスケード連結
- cd=500×2 = 1000 で高次元を効率活用
- 各ブロックが異なるデータで専門化

---

## 主要な決定事項

### 1. 1層固定（multi-layer廃止）

| 比較 | 結果 |
|------|------|
| C1T1 vs C2T2 | C1T1が優位（PPL 127.2 vs 132.2） |
| 層を増やす効果 | なし（むしろ悪化） |

### 2. context_dim=500

| cd | Val PPL | ER% |
|----|---------|-----|
| **500** | **127.2** | **79.7%** |
| 1000 | 134.0 | 69.3% |

### 3. Early Stopping 90%

| 閾値 | Val PPL | 備考 |
|------|---------|------|
| **90%** | **127.2** | 最適 |
| 99% | 235.1 | 過収束で悪化 |

### 4. OACD アルゴリズム

唯一の多様性損失アルゴリズムとして採用。

---

## 推奨設定

```python
# config.py
context_dim = 500
num_input_tokens = 1
early_stopping_threshold = 0.90
phase2_freeze_embedding = True
use_weight_tying = True
```

---

## 詳細資料

| ファイル | 内容 |
|----------|------|
| EXPERIMENT_RESULTS_SUMMARY.md | 実験結果統合 |
| SUMMARY_scaling_laws.md | スケーリング則 |
| SUMMARY_model_design.md | モデル設計 |
| SUMMARY_algorithms.md | 多様性アルゴリズム |

---

*詳細な実験データは old/ フォルダ参照*
