# 多様性損失アルゴリズム比較実験結果 (2025-12-01)

## 実験概要

Phase 1訓練における12種類の多様性損失アルゴリズムを比較し、Effective Rank（ER）の達成度と計算コストを評価。

### 実験条件

| 項目 | 値 |
|------|-----|
| サンプル数 | 100 |
| トークン数 | 122,795 |
| context_dim | 200 |
| num_layers | 6 |
| dist_reg_weight | 0.9（多様性90%, CVFP 10%） |
| max_iterations | 60 |
| GPU | NVIDIA L4 (22.2GB) |

---

## 結果サマリー（Val ER順）

| 順位 | アルゴリズム | Val ER% | Train ER% | 時間(s) | Loss計算(ms) | 評価 |
|:----:|:------------|--------:|----------:|--------:|-------------:|:-----|
| 🥇1 | **SDL** | **94.9%** | 96.1% | 335.3 | 27.44 | 最高ER、高コスト |
| 🥈2 | **NUC** | **91.9%** | 93.4% | 337.2 | 27.87 | 高ER、高コスト |
| 🥉3 | **ODCM** | **89.2%** | 90.9% | 328.1 | 0.72 | 高ER、低コスト ⭐推奨 |
| 4 | MCDL | 78.4% | 79.9% | 245.6 | 0.20 | 現行、最速 |
| 5 | InfoNCE | 76.9% | 78.7% | 82.4 | 1.33 | バランス良 |
| 6 | UNIF | 75.2% | 77.5% | 107.4 | 18.40 | 中程度 |
| 7 | CTM | 72.0% | 74.1% | 79.9 | 0.31 | 中程度、低コスト |
| 8 | DUE | 58.4% | 60.2% | 52.0 | 14.29 | 低ER、早期停止 |
| 9 | DECORR | 47.4% | 49.0% | 52.0 | 0.68 | 低ER |
| 10 | WMSE | 42.9% | 44.6% | 52.9 | 21.63 | 低ER |
| 11 | UDEL | 41.5% | 43.3% | 52.3 | 0.46 | 低ER |
| 12 | HSIC | 36.8% | 38.7% | 83.8 | 69.25 | 最低ER、最高コスト |

---

## 主要な発見

### 1. トップ3アルゴリズムの特徴

**SDL (Spectral Diversity Loss)** - Val ER: 94.9%
- Effective Rankを直接最大化する設計
- 60イテレーション完走（早期停止なし）
- 計算コスト高（27.44ms/iter）だが最高性能

**NUC (Nuclear Norm Maximization)** - Val ER: 91.9%
- 核ノルムを最大化し、行列のランクを上げる
- 60イテレーション完走
- SDLと同等のコストで若干低いER

**ODCM (Off-Diagonal Covariance Minimization)** - Val ER: 89.2% ⭐
- VICReg風の非対角共分散最小化
- 60イテレーション完走
- **低コスト(0.72ms)で高ER** → **推奨アルゴリズム**

### 2. 現行MCDLとの比較

| 指標 | MCDL（現行） | ODCM（推奨） | 改善率 |
|------|-------------|-------------|--------|
| Val ER | 78.4% | 89.2% | **+13.8%** |
| Train ER | 79.9% | 90.9% | **+13.8%** |
| 時間 | 245.6s | 328.1s | +33.5% |
| Loss計算 | 0.20ms | 0.72ms | +3.6x |

MCDLからODCMへの切り替えで、**Val ERが13.8%向上**。

### 3. 早期停止の影響

多くのアルゴリズムがVal ER低下により早期停止：

| アルゴリズム | 停止イテレーション | 理由 |
|-------------|-------------------|------|
| MCDL | 45 | Val ER改善なし |
| DUE | 10 | Val ER低下 |
| CTM | 15 | Val ER低下 |
| UDEL | 10 | Val ER低下 |
| UNIF | 15 | Val ER低下 |
| DECORR | 10 | Val ER低下 |
| HSIC | 10 | Val ER低下 |
| InfoNCE | 15 | Val ER低下 |
| WMSE | 10 | Val ER低下 |

**SDL, NUC, ODCM**は60イテレーション完走し、Val ERが継続的に改善。

### 4. コストパフォーマンス分析

**低コスト・高ER（推奨）:**
- ODCM: 0.72ms/iter, 89.2% ER

**低コスト・中ER:**
- MCDL: 0.20ms/iter, 78.4% ER
- CTM: 0.31ms/iter, 72.0% ER

**高コスト・高ER:**
- SDL: 27.44ms/iter, 94.9% ER
- NUC: 27.87ms/iter, 91.9% ER

**高コスト・低ER（非推奨）:**
- HSIC: 69.25ms/iter, 36.8% ER

---

## 推奨事項

### 即座に採用可能

1. **ODCM**をデフォルトに変更
   - MCDLより13.8%高いVal ER
   - 計算コスト増加は許容範囲（0.20ms → 0.72ms）
   - VICReg由来の理論的裏付け

### 追加検証が必要

2. **SDL/NUCのコスト削減**
   - サンプリングによる高速化
   - 最高ER達成アルゴリズムなので、コスト削減版の開発価値あり

3. **Phase 2での影響評価**
   - 高ERがVal PPL/Accにどう影響するか
   - ODCM vs MCDLでのPhase 2比較実験

---

## 結論

**ODCMへの切り替えを推奨。** MCDLより13.8%高いVal ERを達成し、計算コスト増加は最小限。Phase 2での検証が次のステップ。

---

## 付録: アルゴリズム説明

| 名称 | 正式名 | 概要 |
|------|--------|------|
| MCDL | Mean-Centered Dispersion Loss | 平均からの偏差最大化（現行） |
| ODCM | Off-Diagonal Covariance Minimization | 非対角共分散最小化（VICReg風） |
| DUE | Dimension Usage Entropy | 次元活性度のエントロピー均一化 |
| CTM | Covariance Trace Maximization | 共分散トレース最大化 |
| UDEL | Uniform Distribution Entropy Loss | 一様分布エントロピー最大化 |
| SDL | Spectral Diversity Loss | スペクトル多様性（ER直接最大化） |
| UNIF | Uniformity Loss | 球面一様分布への誘導 |
| DECORR | Decorrelation Loss | 相関行列対角化 |
| NUC | Nuclear Norm Maximization | 核ノルム最大化 |
| HSIC | Hilbert-Schmidt Independence Criterion | 次元間独立性 |
| InfoNCE | InfoNCE Loss | コントラスティブ学習 |
| WMSE | Whitening MSE | 白色化MSE |
