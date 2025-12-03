# 多様性アルゴリズムまとめ (Diversity Algorithms Summary)

**最終更新**: 2025-12-02

---

## 採用アルゴリズム: OACD

**Origin-Anchored Centroid Dispersion** が唯一の多様性損失アルゴリズム。

### 定義

```python
def oacd_loss(contexts, centroid_weight=0.1):
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)
    centroid_loss = torch.norm(context_mean, p=2) ** 2
    return dispersion_loss + centroid_weight * centroid_loss
```

### 特徴

- **分散最大化**: 重心からの偏差を最大化
- **重心固定**: 重心を原点に引き寄せ → 安定した平衡点
- **計算コスト**: O(n×d) 最速クラス
- **centroid_weight**: 0.1 で安定

---

## 過去のアルゴリズム比較（参考）

| Algorithm | Val PPL | α値 | 特徴 |
|-----------|---------|-----|------|
| **MCDL** | **289.1** | -0.423 | 最良PPL |
| ODCM | 308.5 | **-0.568** | 最良α |
| SDL | 343.8 | -0.578 | 最高ER |
| NUC | 312.7 | -0.519 | - |

### 削除済みアルゴリズム

- MCDL: OACDに統合
- ODCM: メンテナンス性を考慮し削除
- SDL/NUC: 計算コストが高く削除

---

## 重要な発見

### 1. CVFPは不要

多様性損失のみ（dist_reg_weight=1.0）で固定点に収束可能。

| 設定 | 固定点収束 |
|------|-----------|
| dwr=1.0（CVFPなし） | ✅ 収束 |
| dwr=0.9（CVFP 10%） | ✅ 収束 |

### 2. ERとPPLの関係

高ERが必ずしも高性能につながらない。

| 変化 | Val ER | Val PPL |
|------|--------|---------|
| 通常Layer | ↓ -5% | ↓ **-18%改善** |

ERは副次的指標であり、PPL/Accを優先すべき。

---

## 実験スクリプト

```bash
# Cascade Context実験
python3 scripts/experiment_cascade_context.py -s 2000
```

---

*Last Updated: 2025-12-02*
