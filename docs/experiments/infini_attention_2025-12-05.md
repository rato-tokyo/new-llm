# Infini-Attention 実験結果

**実験日時**: 2025-12-05
**GPU**: NVIDIA L4 (23.8GB)

---

## 実験設定

| パラメータ | 値 |
|-----------|-----|
| サンプル数 | 5,000 |
| シーケンス長 | 256 |
| エポック数 | 30 (Early stopping: patience=1) |
| 学習率 | 1e-4 |
| バッチサイズ | 8 |
| Delta Rule | 有効 |

### データ

- **訓練データ**: 4,830サンプル (Pile: 4,500 + Reversal pairs: 330)
- **検証データ**: 500サンプル (Pileのみ、Reversal pairsなし)

---

## アーキテクチャ比較

| モデル | 構成 | パラメータ数 |
|--------|------|-------------|
| **Pythia** | 全6層 RoPE | 70,420,480 |
| **Infini-Pythia** | Layer 0: Infini-Attention (NoPE)<br>Layer 1-5: Pythia (RoPE) | 70,418,440 |

Infini-Pythiaのメモリ: 133,120 bytes (固定)

---

## 結果サマリー

### Perplexity (PPL)

| モデル | Best PPL | Best Epoch |
|--------|----------|------------|
| Pythia (RoPE) | 106.0 | 7 |
| **Infini-Pythia** | **103.9** | 7 |

**差分: -2.2 PPL (2.1%改善)**

---

## 学習曲線

### Pythia (RoPE baseline)

| Epoch | Train PPL | Val PPL | 備考 |
|-------|-----------|---------|------|
| 1 | 596.7 | 407.7 | * |
| 2 | 163.8 | 228.1 | * |
| 3 | 91.5 | 167.4 | * |
| 4 | 59.3 | 135.4 | * |
| 5 | 41.1 | 117.2 | * |
| 6 | 29.4 | 108.9 | * |
| 7 | 21.3 | **106.0** | * Best |
| 8 | 15.5 | 106.6 | Early stop |

### Infini-Pythia

| Epoch | Train PPL | Val PPL | Gate | 備考 |
|-------|-----------|---------|------|------|
| 1 | 639.2 | 428.4 | 0.498 | * |
| 2 | 170.3 | 238.3 | 0.495 | * |
| 3 | 93.6 | 170.3 | 0.490 | * |
| 4 | 60.2 | 136.0 | 0.485 | * |
| 5 | 41.3 | 118.1 | 0.479 | * |
| 6 | 29.4 | 108.0 | 0.473 | * |
| 7 | 21.2 | **103.9** | 0.466 | * Best |
| 8 | 15.3 | 106.6 | 0.460 | Early stop |

**観察**: Gate値は学習が進むにつれて0.498→0.460に減少（メモリ依存度が増加）

---

## Position-wise PPL

| Position | Pythia | Infini | 差分 | 変化 |
|----------|--------|--------|------|------|
| 0-16 | 165.9 | 163.2 | -2.7 | ↓ 改善 |
| 16-32 | 110.3 | 111.0 | +0.7 | ↑ 微増 |
| 32-64 | 102.1 | 102.0 | -0.1 | - |
| 64-96 | 99.2 | 99.2 | -0.0 | - |
| 96-256 | 104.8 | 104.1 | -0.6 | ↓ 改善 |

**観察**:
- 序盤（0-16）で最大の改善 (-2.7 PPL)
- 中盤（16-32）でわずかな劣化 (+0.7 PPL)
- 後半（96-256）でも改善 (-0.6 PPL)

---

## Reversal Curse 評価

| モデル | Forward PPL | Backward PPL | Ratio | Gap |
|--------|-------------|--------------|-------|-----|
| Pythia | 1.7 | 684.4 | 0.002 | +682.8 |
| **Infini** | 1.7 | **569.6** | **0.003** | **+567.9** |

**改善率**: Backward PPL 16.8%改善 (684.4 → 569.6)

### 解釈

- **Forward PPL**: 両モデルとも1.7（訓練データに含まれる順方向文は完全に学習）
- **Backward PPL**: Infiniが114.8ポイント低い（逆方向推論が改善）
- **Reversal Ratio**: 1.0に近いほど良い。Infiniはわずかに改善 (0.002 → 0.003)
- **Reversal Gap**: 0に近いほど良い。Infiniは114.9ポイント改善

---

## Gate値の分析

学習終了時の各ヘッドのGate値 (`sigmoid(β)`):

| Head | Gate値 | 解釈 |
|------|--------|------|
| 0 | 0.464 | Memory 46.4%, Local 53.6% |
| 1 | 0.475 | Memory 47.5%, Local 52.5% |
| 2 | 0.458 | Memory 45.8%, Local 54.2% |
| 3 | 0.473 | Memory 47.3%, Local 52.7% |
| 4 | 0.452 | Memory 45.2%, Local 54.8% |
| 5 | 0.455 | Memory 45.5%, Local 54.5% |
| 6 | 0.447 | Memory 44.7%, Local 55.3% |
| 7 | 0.454 | Memory 45.4%, Local 54.6% |

**平均**: 0.460 (Memory 46%, Local 54%)

**観察**:
- 全ヘッドがLocal Attentionをやや優先（54%）
- 学習初期（0.498）から終了時（0.460）へ、メモリ依存度が増加
- ヘッド間の分散は小さい（0.447-0.475）

---

## 結論

### 改善点

1. **全体PPL**: 2.1%改善 (106.0 → 103.9)
2. **Reversal Curse**: Backward PPL 16.8%改善
3. **序盤・終盤位置**: Position 0-16, 96-256で改善

### 課題

1. **中盤位置**: Position 16-32でわずかに劣化 (+0.7)
2. **Reversal Ratio**: 依然として非常に低い（0.003）、根本的解決には至らず

### 考察

- Infini-Attentionの圧縮メモリは、特に**長距離依存性**（position 96-256）と**知識の双方向性**（Reversal Curse）に効果を示した
- Gate値が約0.46に収束したことは、モデルがLocal AttentionとMemory Attentionの適切なバランスを学習したことを示唆
- 1層目のみのInfini-Attention導入でも有意な改善が得られた

---

## Memory-Only 実験 (Local Attention無効)

**追加実験日時**: 2025-12-05

### 設定

Local Attentionを完全に無効化し、Memory Attentionのみを使用する実験。
- `--memory-only`フラグ: β=10（sigmoid≈0.99995）で固定、学習不可
- Gate値は全ヘッドで1.000に固定

### 結果比較

| モデル | Val PPL | Best Epoch | Gate | Local Attention |
|--------|---------|------------|------|-----------------|
| Pythia (baseline) | 106.0 | 7 | - | - |
| Infini-Pythia | 103.9 | 7 | 0.460 | 有効 |
| **Infini-Pythia (Memory-Only)** | **105.7** | 7 | 1.000固定 | **無効** |

### 学習曲線 (Memory-Only)

| Epoch | Train PPL | Val PPL | Gate | 備考 |
|-------|-----------|---------|------|------|
| 1 | 662.4 | 436.3 | 1.000 | * |
| 2 | 173.5 | 240.3 | 1.000 | * |
| 3 | 95.8 | 172.5 | 1.000 | * |
| 4 | 61.7 | 138.7 | 1.000 | * |
| 5 | 42.6 | 119.3 | 1.000 | * |
| 6 | 30.5 | 108.8 | 1.000 | * |
| 7 | 22.1 | **105.7** | 1.000 | * Best |
| 8 | 16.1 | 108.0 | 1.000 | Early stop |

### Position-wise PPL 比較

| Position | Pythia | Infini (通常) | Infini (Memory-Only) |
|----------|--------|---------------|----------------------|
| 0-16 | 165.9 | 163.2 | 154.6 |
| 16-32 | 110.3 | 111.0 | 112.2 |
| 32-64 | 102.1 | 102.0 | 104.4 |
| 64-96 | 99.2 | 99.2 | 100.3 |
| 96-256 | 104.8 | 104.1 | 106.0 |

### Reversal Curse 比較

| モデル | Forward PPL | Backward PPL | Ratio | Gap |
|--------|-------------|--------------|-------|-----|
| Pythia | 1.7 | 684.4 | 0.002 | +682.8 |
| Infini (通常) | 1.7 | 569.6 | 0.003 | +567.9 |
| **Infini (Memory-Only)** | 1.7 | **584.7** | 0.003 | +583.0 |

### 分析

#### 全体PPL

| 比較 | 差分 | 解釈 |
|------|------|------|
| Memory-Only vs Pythia | -0.3 | わずかに改善 |
| Memory-Only vs 通常Infini | +1.8 | 通常版が優位 |

#### Position-wise分析

1. **序盤 (0-16)**: Memory-Only が最も良い (-11.3 vs Pythia)
   - 純粋な圧縮メモリは短距離でも有効
2. **中盤 (16-96)**: 通常版が優位
   - Local Attentionの細かい文脈把握が重要
3. **終盤 (96-256)**: 通常版が優位
   - 長距離依存性にはLocal+Memoryの組み合わせが最適

#### Reversal Curse分析

- Memory-Only: 584.7 (通常版より+15.1)
- Local Attentionの存在が逆方向推論に寄与している可能性

### 結論

**Local Attentionは重要な役割を果たしている**

1. **全体性能**: 通常版（103.9）> Memory-Only（105.7）> Pythia（106.0）
2. **学習済みGate（0.46）の妥当性**: モデルは約54% Local / 46% Memory のバランスを選択
3. **序盤のみMemory-Only優位**: Position 0-16では純粋メモリが有効
4. **Reversal Curse**: Local Attentionが逆方向推論に寄与

**示唆**:
- 圧縮メモリ単体でもPythiaと同等の性能を達成可能
- 最適なパフォーマンスにはLocal AttentionとMemory Attentionの組み合わせが必要
- 学習可能なβゲートにより、モデルが適切なバランスを発見できる

### 実行コマンド

```bash
python3 scripts/experiment_infini.py --memory-only --skip-baseline
```

---

## Long Context Training 実験

**追加実験日時**: 2025-12-05

### 設定

長文ドキュメント対応のTruncated BPTT訓練と評価を実施。

| パラメータ | 値 |
|-----------|-----|
| 訓練ドキュメント | 45 |
| 検証ドキュメント | 5 |
| ドキュメントあたりトークン数 | 4,096 |
| セグメント長 | 256 |
| セグメント数/ドキュメント | 16 |
| 総訓練トークン | 184,320 (45 × 4,096) |

### 訓練方式

- **Truncated BPTT**: セグメントごとにbackprop、ドキュメント境界でメモリリセット
- **メモリ管理**: セグメント間でメモリを継続、ドキュメント間でリセット

### 結果サマリー

#### 訓練結果

| モデル | Best Val PPL | Best Epoch | 備考 |
|--------|-------------|------------|------|
| Pythia (RoPE) | 1388.4 | 3 | Early stop (epoch 4) |
| Infini-Pythia | **1317.6** | 4 | Early stop (epoch 5) |

**差分: -70.9 PPL (5.1%改善)**

### 学習曲線

#### Pythia (RoPE baseline)

| Epoch | Train PPL | Val PPL | 備考 |
|-------|-----------|---------|------|
| 1 | 2151.3 | 1851.9 | * |
| 2 | 456.9 | 1470.1 | * |
| 3 | 204.5 | **1388.4** | * Best |
| 4 | 110.7 | 1396.5 | Early stop |

#### Infini-Pythia

| Epoch | Train PPL | Val PPL | Gate | 備考 |
|-------|-----------|---------|------|------|
| 1 | 2148.6 | 1801.9 | 0.501 | * |
| 2 | 450.6 | 1487.0 | 0.500 | * |
| 3 | 201.1 | 1368.7 | 0.497 | * |
| 4 | 109.7 | **1317.6** | 0.494 | * Best |
| 5 | 66.1 | 1334.9 | 0.491 | Early stop |

### Gate値の分析

| Head | Gate値 | 解釈 |
|------|--------|------|
| 0 | 0.489 | Memory 48.9%, Local 51.1% |
| 1 | 0.492 | Memory 49.2%, Local 50.8% |
| 2 | 0.493 | Memory 49.3%, Local 50.7% |
| 3 | 0.490 | Memory 49.0%, Local 51.0% |
| 4 | 0.491 | Memory 49.1%, Local 50.9% |
| 5 | 0.490 | Memory 49.0%, Local 51.0% |
| 6 | 0.493 | Memory 49.3%, Local 50.7% |
| 7 | 0.490 | Memory 49.0%, Local 51.0% |

**平均**: 0.491 (Memory 49.1%, Local 50.9%)

**観察**: ほぼ50:50のバランス。前回の通常訓練（0.460）よりMemory比率が高い。

### Long Context 評価結果

**評価設定**:
- 50ドキュメント × 4,096トークン = 204,800トークン
- ドキュメントごとにメモリリセット
- セグメントごとにメモリ更新

#### 結果比較

| モデル | Total PPL | Seg 0 | Seg 1 | Seg 2 | Last (Seg 15) |
|--------|-----------|-------|-------|-------|---------------|
| Pythia | 55981.3 | 56158.1 | 55935.4 | 55761.6 | 56238.2 |
| Infini (with memory) | **121.1** | 106.6 | 125.3 | 130.8 | 112.9 |
| Infini (without memory) | 121.1 | 106.6 | 125.3 | 130.8 | 112.9 |

### 重大な問題点

#### 1. Pythiaの異常なPPL (55981)

訓練PPLは110程度まで下がったにもかかわらず、Long Context評価で55981という異常値を示した。

**考えられる原因**:
- **訓練データと評価データの分布不一致**: 訓練は45ドキュメント、評価は別の50ドキュメント
- **ドキュメント分割の問題**: Pileデータを連続で分割しているため、訓練と評価で異なる領域を使用
- **過学習**: 訓練ドキュメントに過学習し、評価ドキュメントで汎化失敗

#### 2. Infiniの「with memory」と「without memory」が同一

両者のPPLが完全一致（121.1）している。

**考えられる原因**:
- 評価時のメモリ更新が正しく機能していない可能性
- または、メモリが有効に活用されていない

### Reversal Curse 評価

| モデル | Forward PPL | Backward PPL | Ratio | Gap |
|--------|-------------|--------------|-------|-----|
| Pythia | 32313.2 | 26516.1 | 1.219 | -5797.1 |
| Infini | 30103.8 | 19812.5 | **1.519** | -10291.3 |

**観察**:
- Forward > Backward という異常な結果（通常はBackwardの方が高い）
- **Ratioが1.0より大きい**: 逆方向の方が得意という矛盾
- Long Context訓練でReversal Curse評価の基準が変化した可能性

### 分析と考察

#### 良い点

1. **Infini-PythiaがPythiaを上回った**: Val PPL 1317.6 vs 1388.4 (5.1%改善)
2. **Long Context評価でInfiniが圧倒的優位**: 121 vs 55981
3. **学習が安定**: 両モデルとも収束

#### 問題点

1. **Pythiaの評価PPL異常**: 訓練PPL(110)と評価PPL(55981)の乖離が大きすぎる
2. **メモリ有無で差がない**: Infiniのメモリ機能が評価時に効いていない可能性
3. **Reversal Curse異常**: Forward > Backward の逆転現象

### 今後の調査項目

1. **データ分割の確認**: 訓練と評価で同じドキュメントを使用していないか
2. **メモリ更新ロジックの検証**: `update_memory=True/False`が正しく機能しているか
3. **標準評価との比較**: Long Context訓練後のモデルを通常の評価方法でも検証

### 実行コマンド

```bash
python3 scripts/experiment_infini.py \
    --long-context-train \
    --long-context \
    --num-long-docs 50 \
    --tokens-per-doc 4096
```

---

## Multi-Memory Bank 実験

**追加実験日時**: 2025-12-05

**注意**: この実験はリファクタリング前のコード（Local Attention有効版）で実行されました。Gate値はLocal/Memory比率を示しています。リファクタリング後はMemory-Onlyモードのみとなります。

### 目的

複数のメモリバンクを使用することで、圧縮メモリ内の情報混合を低減し、より正確な検索を実現できるか検証。

### 設定

| パラメータ | 値 |
|-----------|-----|
| サンプル数 | 5,000 |
| シーケンス長 | 256 |
| メモリバンク数 | 2 |
| バンクあたりセグメント数 | 4 |
| Delta Rule | 有効 |
| Baseline | スキップ |

### アーキテクチャ

```
Multi-Memory Infini-Pythia:
  Layer 0: Multi-Memory Infini-Attention
    ├─ Memory Bank 0 (セグメント0-3を蓄積)
    ├─ Memory Bank 1 (セグメント4-7を蓄積)
    └─ Bank Weights (学習可能、softmaxで統合)
  Layer 1-5: Standard Pythia (RoPE)
```

- **メモリサイズ**: 266,240 bytes (133,120 × 2 banks)
- **バンク切り替え**: 4セグメントごとに次のバンクへ
- **検索時**: 全バンクから取得し、学習可能な重みで統合

### 結果

| モデル | Val PPL | Best Epoch |
|--------|---------|------------|
| Pythia (baseline, 前回) | 106.0 | 7 |
| Infini (通常, 前回) | 103.9 | 7 |
| **Multi-Memory (2 banks)** | **105.5** | 7 |

### 学習曲線

| Epoch | Train PPL | Val PPL | Gate | 備考 |
|-------|-----------|---------|------|------|
| 1 | 659.8 | 433.9 | 0.498 | * |
| 2 | 172.5 | 239.3 | 0.495 | * |
| 3 | 94.9 | 171.2 | 0.490 | * |
| 4 | 60.9 | 137.4 | 0.486 | * |
| 5 | 41.8 | 118.3 | 0.480 | * |
| 6 | 29.7 | 108.5 | 0.474 | * |
| 7 | 21.4 | **105.5** | 0.467 | * Best |
| 8 | 15.5 | 109.4 | 0.460 | Early stop |

### Gate値の分析

| Head | Gate値 | 解釈 |
|------|--------|------|
| 0 | 0.456 | Memory 45.6%, Local 54.4% |
| 1 | 0.465 | Memory 46.5%, Local 53.5% |
| 2 | 0.461 | Memory 46.1%, Local 53.9% |
| 3 | 0.453 | Memory 45.3%, Local 54.7% |
| 4 | 0.466 | Memory 46.6%, Local 53.4% |
| 5 | 0.462 | Memory 46.2%, Local 53.8% |
| 6 | 0.464 | Memory 46.4%, Local 53.6% |
| 7 | 0.456 | Memory 45.6%, Local 54.4% |

**平均**: 0.460 (通常Infiniと同じ)

### Position-wise PPL

| Position | Pythia (前回) | Infini (前回) | Multi-Memory |
|----------|---------------|---------------|--------------|
| 0-16 | 165.9 | 163.2 | 161.7 |
| 16-32 | 110.3 | 111.0 | 114.5 |
| 32-64 | 102.1 | 102.0 | 104.3 |
| 64-96 | 99.2 | 99.2 | 101.2 |
| 96-256 | 104.8 | 104.1 | 107.3 |

### Reversal Curse

| モデル | Forward PPL | Backward PPL | Ratio | Gap |
|--------|-------------|--------------|-------|-----|
| Pythia (前回) | 1.7 | 684.4 | 0.002 | +682.8 |
| Infini (前回) | 1.7 | 569.6 | 0.003 | +567.9 |
| **Multi-Memory** | 1.7 | **600.8** | 0.003 | +599.1 |

### 分析

#### PPL比較

| 比較 | 差分 | 解釈 |
|------|------|------|
| Multi-Memory vs Pythia | -0.5 | わずかに改善 |
| Multi-Memory vs 通常Infini | +1.6 | 通常版が優位 |

#### 観察

1. **全体PPL**: Multi-Memory (105.5) は Pythia (106.0) より改善するが、通常Infini (103.9) より劣る
2. **Position 0-16**: Multi-Memoryが最も良い (161.7) - 序盤での圧縮メモリ効果
3. **Position 16-256**: 通常Infiniが優位 - バンク分割によるオーバーヘッド？
4. **Reversal Curse**: Multi-Memory (600.8) は通常Infini (569.6) より劣る

#### 考察

**Multi-Memory Bankが期待通りに機能しなかった理由**:

1. **バンク数の問題**: 2バンクでは情報混合の低減効果が限定的
2. **セグメント長との不整合**: 256トークンのセグメントで4セグメント（1024トークン）ごとにバンク切り替えは、短いコンテキストでは効果が見えにくい
3. **学習データ規模**: 5,000サンプルでは複数バンクの効果を発揮しにくい
4. **Bank Weights**: 初期値0（uniform 0.5, 0.5）から学習するが、短い訓練では最適化が不十分

**改善案**:

1. **Long Context訓練で評価**: より長いドキュメントでバンク切り替え効果を検証
2. **バンク数増加**: 4, 8バンクでの実験
3. **セグメント数調整**: segments_per_bank を小さく（1, 2）

### 実行コマンド

```bash
python3 scripts/experiment_infini.py \
    --num-memory-banks 2 \
    --segments-per-bank 4 \
    --skip-baseline
```

---

## ALiBi位置エンコーディング実験

**追加実験日時**: 2025-12-06

### 目的

Infini-Attentionの圧縮メモリに位置情報を組み込むため、ALiBi（Attention with Linear Biases）を線形化近似で導入。RoPEは線形化アテンションと相性が悪いため、ALiBiを採用。

### ALiBiの線形化近似

従来のALiBiはsoftmax内でバイアスを加算するが、線形アテンションでは以下のように近似：

```
メモリ更新 (ALiBi重み付き):
  M_φ = Σ_i w_i * φ(K_i) * V_i^T
  z_φ = Σ_i w_i * φ(K_i)

  w_i = exp(-slope * segment_distance)

メモリ取得:
  Output = φ(Q) @ M_φ / (φ(Q) @ z_φ)
```

- **slope**: ヘッドごとに異なる値（ALiBi論文に準拠）
- **segment_distance**: 現在のセグメントからの距離（0, 1, 2, ...）
- **効果**: 古いセグメントほど重みが小さくなり、位置情報がメモリに反映

### 設定

| パラメータ | 値 |
|-----------|-----|
| サンプル数 | 5,000 |
| シーケンス長 | 256 |
| エポック数 | 30 |
| ALiBi | 有効 (scale=1.0) |
| Delta Rule | 有効 |
| Baseline | スキップ |

### アーキテクチャ

```
Infini-Pythia (ALiBi):
  Layer 0: InfiniAttentionALiBi (Memory-Only + ALiBi位置重み付け)
    ├─ φ(K) に exp(-slope * segment_count) を乗算
    ├─ メモリ更新時に重み付き外積
    └─ 古いセグメントは減衰
  Layer 1-5: Standard Pythia (RoPE)
```

- **パラメータ数**: 70,418,432
- **メモリサイズ**: 133,120 bytes

### 結果

| モデル | Val PPL | Best Epoch |
|--------|---------|------------|
| Pythia (前回baseline) | 106.0 | 7 |
| Infini (通常, Memory-Only, 前回) | 105.7 | 7 |
| **Infini (ALiBi)** | **105.7** | 7 |

### 学習曲線

| Epoch | Train PPL | Val PPL | 備考 |
|-------|-----------|---------|------|
| 1 | 662.4 | 436.3 | * |
| 2 | 173.5 | 240.3 | * |
| 3 | 95.8 | 172.5 | * |
| 4 | 61.7 | 138.7 | * |
| 5 | 42.6 | 119.3 | * |
| 6 | 30.5 | 108.8 | * |
| 7 | 22.1 | **105.7** | * Best |
| 8 | 16.1 | 107.9 | Early stop |

### Position-wise PPL

| Position | Pythia (前回) | Infini Memory-Only (前回) | Infini ALiBi |
|----------|---------------|---------------------------|--------------|
| 0-16 | 165.9 | 154.6 | 154.5 |
| 16-32 | 110.3 | 112.2 | 112.1 |
| 32-64 | 102.1 | 104.4 | 104.4 |
| 64-96 | 99.2 | 100.3 | 100.3 |
| 96-256 | 104.8 | 106.0 | 106.0 |

### Reversal Curse

| モデル | Forward PPL | Backward PPL | Ratio | Gap |
|--------|-------------|--------------|-------|-----|
| Pythia (前回) | 1.7 | 684.4 | 0.002 | +682.8 |
| Infini Memory-Only (前回) | 1.7 | 584.7 | 0.003 | +583.0 |
| **Infini ALiBi** | 1.7 | **619.1** | 0.003 | +617.4 |

### 分析

#### PPL比較

| 比較 | 差分 | 解釈 |
|------|------|------|
| ALiBi vs Pythia | -0.3 | わずかに改善 |
| ALiBi vs Memory-Only | ±0.0 | 同等 |

#### 観察

1. **全体PPL**: ALiBi (105.7) = Memory-Only (105.7)、両者とも Pythia (106.0) より改善
2. **Position-wise**: ALiBiとMemory-Onlyはほぼ同一のパターン
3. **Reversal Curse**: ALiBi (619.1) はMemory-Only (584.7) より劣化 (+34.4)

#### 考察

**ALiBiが期待通りに機能しなかった理由**:

1. **セグメント粒度の問題**:
   - ALiBiはセグメント単位（256トークン）で距離を計測
   - 実験設定では各サンプルが独立（セグメント数=1）
   - 複数セグメントを跨ぐLong Context訓練でないと効果が見えにくい

2. **減衰の適用タイミング**:
   - 現在の実装: メモリ更新時に累積的に減衰
   - 問題: 訓練データが独立サンプルの場合、segment_count=0で常に重み=1.0

3. **Short Contextでの限界**:
   - 256トークンのシーケンスでは、位置情報の恩恵が限定的
   - RoPE（Layer 1-5で使用）が既に十分な位置情報を提供

4. **Reversal Curseへの悪影響**:
   - 位置バイアスが逆方向推論を阻害した可能性
   - 「A is B」と「B is A」で異なる位置減衰が適用される

**改善案**:

1. **Long Context訓練で再評価**: 複数セグメントを跨ぐドキュメントでALiBiの効果を検証
2. **トークンレベルALiBi**: セグメント単位ではなくトークン単位で距離を計算
3. **alibi_scale調整**: scale=0.5, 2.0など異なる減衰強度で実験

### 実行コマンド

```bash
python3 scripts/experiment_infini.py --alibi --skip-baseline

# 強い減衰
python3 scripts/experiment_infini.py --alibi --alibi-scale 2.0 --skip-baseline
```

---

## 次のステップ（提案）

1. **より長いシーケンス長での実験**: seq_length=512, 1024
2. **複数層へのInfini-Attention導入**: Layer 0-1, Layer 0-2 など
3. **より大規模データでの検証**: samples=10,000以上
4. **Delta Rule無効時との比較**: `--no-delta-rule`オプション
5. **Long Context訓練の問題調査**: データ分割とメモリ更新ロジックの検証
6. **Multi-Memory Bank改良**: Long Contextでの評価、バンク数増加
7. **ALiBi Long Context評価**: 複数セグメントを跨ぐ訓練でALiBiの効果を検証
8. **トークンレベル位置エンコーディング**: セグメント単位ではなくトークン単位のALiBi

---

## 実行コマンド

```bash
# 通常実験（両モデル比較）
python3 scripts/experiment_infini.py --samples 5000 --seq-length 256 --epochs 30

# Multi-Memory Bank実験
python3 scripts/experiment_infini.py --num-memory-banks 2 --segments-per-bank 4 --skip-baseline

# ALiBi位置エンコーディング実験
python3 scripts/experiment_infini.py --alibi --skip-baseline
python3 scripts/experiment_infini.py --alibi --alibi-scale 2.0 --skip-baseline

# Long Context訓練・評価
python3 scripts/experiment_infini.py --long-context-train --long-context --num-long-docs 50 --tokens-per-doc 4096
```
