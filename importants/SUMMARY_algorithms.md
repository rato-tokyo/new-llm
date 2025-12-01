# 多様性アルゴリズムまとめ (Diversity Algorithms Summary)

**作成日**: 2025-12-01
**統合元ファイル**: diversity_algorithm_comparison_20251201.md, cvfp_hypothesis_test_20251201.md, experiment-results-20251201-*.md

---

## 1. 採用アルゴリズム: OACD

**2025-12-01時点で、OACDを唯一のアルゴリズムとして採用。**

### OACD (Origin-Anchored Centroid Dispersion)

```python
def oacd_loss(contexts, centroid_weight=0.1):
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)
    centroid_loss = torch.norm(context_mean, p=2) ** 2
    return dispersion_loss + centroid_weight * centroid_loss
```

**特徴**:
- MCDL（Mean-Centered Dispersion Loss）の拡張版
- **分散最大化** + **重心を原点に固定**
- centroid_weight=0.1 で安定

### OACD実験結果

| 設定 | Val PPL | α値 |
|------|---------|-----|
| dwr=1.0 | 290.1 | -0.509 |
| dwr=0.5 | 289.6 | -0.507 |

---

## 2. 過去のアルゴリズム比較結果（参考）

### 2.1 Phase 2性能比較（通常Layer版, cd=500）

| Algorithm | Best PPL | Best Acc | α値 | 特徴 |
|-----------|----------|----------|-----|------|
| **MCDL** | **289.1** | **20.3%** | -0.423 | 最良PPL、高速 |
| ODCM | 308.5 | 19.6% | **-0.568** | 最良α、VICReg風 |
| NUC | 312.7 | 19.3% | -0.519 | 高速収束 |
| SDL | 343.8 | 18.9% | -0.578 | 最高ER |

### 2.2 アルゴリズムの強み

| 目的 | 推奨 | 理由 |
|------|------|------|
| **最良PPL** | MCDL | PPL=289.1, 最速 |
| **最良スケーリング** | SDL/ODCM | α=-0.57〜-0.58 |
| **最高ER** | SDL | Val ER=95%以上 |
| **バランス** | ODCM | PPL良好、α良好、低コスト |

---

## 3. 重要な発見

### 3.1 CVFPは必須ではない

**発見**: 多様性損失のみ（dist_reg_weight=1.0）で固定点に収束可能

| 設定 | 固定点収束 | 備考 |
|------|-----------|------|
| dwr=1.0（CVFPなし） | ✅ 収束 | 追加伝播1回で完全収束 |
| dwr=0.9（CVFP 10%） | ✅ 収束 | わずかにER向上 |

**理由**:
- 並列処理方式の特性（token i が previous_contexts[i-1] を使用）
- 多様性損失が暗黙的に固定点学習として機能
- ContextBlockの収縮写像的性質

### 3.2 ERとPPLの逆相関（重要）

| 変化 | Val ER | Val PPL |
|------|--------|---------|
| 通常Layer版 | ↓ -5% | ↓ **-18%改善** |

**結論**: **高ERが必ずしも高性能につながらない**

- ERが低下してもPPLは大幅改善することがある
- ERは目標ではなく、副次的指標

### 3.3 MCDLの特異な特性

**MCDLだけがdwr設定で影響を受ける**:

| 設定 | MCDL α | ODCM α |
|------|--------|--------|
| dwr=0.9 | -0.423 | -0.568 |
| dwr=0.5 | **-0.513** | -0.568（変化なし） |

- MCDLはCVFPと相性が良い（収束特性があるため）
- dwr=0.5でα値が改善

---

## 4. dist_reg_weight (dwr) の影響

### 4.1 推奨設定

```python
# 現在CVFPは削除済み。OACDのみ使用。
# 以下は参考情報
dist_reg_weight = 0.9  # 多様性90%, CVFP 10%（旧設定）
```

### 4.2 dwr=1.0 vs dwr=0.9 の違い

| 指標 | dwr=1.0 | dwr=0.9 | 備考 |
|------|---------|---------|------|
| 訓練安定性 | やや不安定 | 安定 | SDLで顕著 |
| ER | 低め | 高め | SDL: 74% → 95% |
| スケーリング(α) | 同等 | やや良い | - |

**dwr=0.9の利点**:
- SDLで劇的な安定化（10 iter → 50 iter継続）
- 安定した訓練曲線

---

## 5. アルゴリズム定義（参考）

### 採用中

| 名称 | 数式 | 計算コスト |
|------|------|-----------|
| **OACD** | `-‖X-mean(X)‖/n + λ‖mean(X)‖²` | O(n×d) 最速 |

### 過去に検討（削除済み）

| 名称 | 概要 | ER | コスト |
|------|------|-----|--------|
| MCDL | 平均からの偏差最大化 | 78% | 最速 |
| ODCM | 非対角共分散最小化（VICReg風） | 89% | 低 |
| SDL | スペクトル多様性（ER直接最大化） | 95% | 高 |
| NUC | 核ノルム最大化 | 92% | 高 |

---

## 6. 実験スクリプト

```bash
# OACD実験
python3 scripts/run_experiment.py -s 50 100 200

# サンプル数指定
python3 scripts/run_experiment.py -s 100 200 500
```

---

*Last Updated: 2025-12-01*
