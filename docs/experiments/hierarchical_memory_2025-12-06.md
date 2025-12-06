# Hierarchical Memory 実験結果

**日付**: 2025-12-06
**環境**: NVIDIA L4 (23.8GB)

---

## 実験概要

Hierarchical Memory（学習可能な展開判断）の性能検証。
Single Memory、Multi-Memory との比較。

### 設定

| 項目 | 値 |
|------|-----|
| サンプル数 | 5,000 |
| シーケンス長 | 256 |
| エポック数 | 30 (early stopping) |
| 学習率 | 1e-4 |
| Fine memories | 4 |
| Delta Rule | 有効 |

---

## 結果サマリー

| モデル | Best PPL | Best Epoch | メモリサイズ | 追加パラメータ |
|--------|----------|------------|--------------|----------------|
| Single Memory | 105.9 | 7 | 133KB | 0 |
| Multi Memory (4) | 105.8 | 7 | 532KB | 0 |
| **Hierarchical (4)** | **105.6** | **7** | **532KB** | **65,793** |

---

## Reversal Curse 比較

| モデル | Forward PPL | Backward PPL | Gap | 改善率 |
|--------|-------------|--------------|-----|--------|
| Single Memory | 1.8 | 604.5 | +602.7 | - |
| Multi Memory (4) | 1.8 | 508.6 | +506.8 | 15.9% |
| **Hierarchical (4)** | **1.7** | **437.8** | **+436.1** | **27.6%** |

**Hierarchical MemoryがReversal Curseを最も改善**

---

## 詳細分析

### 1. 学習曲線 (Hierarchical)

| Epoch | Train PPL | Val PPL |
|-------|-----------|---------|
| 1 | 656.6 | 431.2 |
| 2 | 172.3 | 234.5 |
| 3 | 94.3 | 168.4 |
| 4 | 60.4 | 135.6 |
| 5 | 41.4 | 116.8 |
| 6 | 29.4 | 108.5 |
| 7 | 21.2 | 105.6* |
| 8 | 15.3 | 107.6 |

Epoch 8でearly stopping。

### 2. Position-wise PPL

| Position | Single | Multi | Hierarchical |
|----------|--------|-------|--------------|
| 0-16 | 154.0 | 156.5 | 154.0 |
| 16-32 | 111.3 | 111.4 | 111.6 |
| 32-64 | 102.0 | 102.5 | 101.8 |
| 64-96 | 98.4 | 99.1 | 99.3 |
| 96-256 | 103.7 | 103.1 | 103.1 |

3モデルともほぼ同等の位置依存性を示す。

### 3. パラメータ情報

```
Parameters: 70,484,233
Expansion gate params: 65,793 (0.09%)
Memory: 532,480 bytes
```

Expansion gateの追加パラメータは全体の0.09%と非常に小さい。

---

## 考察

### 1. なぜHierarchicalがReversal Curseに効くのか？

**仮説1: 情報の階層的分離**
- 粗粒度メモリ: 一般的なパターン（「A is B」の構造）
- 細粒度メモリ: 具体的な関係（個別のA, Bの対応）
- 逆方向クエリ時に細粒度メモリから正確な情報を取得

**仮説2: 学習された展開判断**
- Expansion gateが「粗粒度で十分か」を学習
- 曖昧なクエリ → 粗粒度で大まかな回答
- 明確なクエリ → 細粒度で正確な回答
- Reversal queryは細粒度を必要とするケースが多い？

**仮説3: Soft decisionの効果**
- `prob * fine + (1-prob) * coarse` の混合
- 完全な二択ではなく、両方の情報を活用
- 逆方向情報が細粒度に残りやすい

### 2. 改善率の推移

```
Single → Multi:    604.5 → 508.6 (15.9%改善)
Multi → Hierarchical: 508.6 → 437.8 (13.9%改善)
Single → Hierarchical: 604.5 → 437.8 (27.6%改善)
```

Multi-Memoryで16%改善、さらにHierarchicalで14%改善。
両方の効果が加算的に働いている。

### 3. PPL維持の理由

- 3モデルとも105.6〜105.9でほぼ同一
- Expansion gate (65K params) はごくわずかで学習に影響なし
- Soft decisionで学習が安定

---

## アーキテクチャ詳細

```
Hierarchical Memory Pythia:
Token Embedding (512-dim)
       ↓
Layer 0: HierarchicalMemoryAttentionLayer
  ├─ Fine memories: [M_0, M_1, M_2, M_3] (常に保持)
  ├─ Coarse memory: M_0 + M_1 + M_2 + M_3 (動的生成)
  ├─ Expansion gate: MLP(hidden→hidden/4→1)
  └─ Soft decision: prob * fine + (1-prob) * coarse
       ↓
Layer 1-5: PythiaLayer (RoPE)
       ↓
Output Head (512 → vocab)
```

**メモリの加法性**:
```
C = A + B  (統合は可能)
A, B → C  (展開は不可能、事前保存が必要)
```

---

## 結論

1. **Hierarchical MemoryはReversal Curseを最も改善**: 27.6%改善（604.5 → 437.8）
2. **PPLは維持**: 追加パラメータの影響なし（105.6）
3. **学習可能な展開判断が有効**: Soft decisionで安定した学習
4. **Multi-Memoryとの相乗効果**: 両方の改善が加算的

---

## 次のステップ

1. **Long Context評価**: 長文でHierarchicalの優位性を検証
2. **8メモリ実験**: メモリ数増加でさらに改善するか
3. **Expansion gate分析**: 学習されたゲートの振る舞いを可視化
4. **Ablation Study**: Soft vs Hard decision の比較
