# HSA vs Memory Norm Landmark比較実験

**実験日**: 2025-12-09
**環境**: Google Colab (NVIDIA L4, 23.8GB)

## 概要

Multi-MemoryレイヤーのLandmark計算方式を比較:
- **HSA方式**: ChunkEncoder（双方向Transformerエンコーダ）で学習可能なLandmarkを生成
- **memory_norm方式**: 書き込み操作の副産物 Σσ(k) をLandmarkとして使用

## 実験設定

| 項目 | 値 |
|------|-----|
| サンプル数 | 5,000 |
| シーケンス長 | 256 |
| メモリ数 | 4 |
| 最大エポック | 30 |
| Early Stopping | patience=5 |
| バッチサイズ | 8 |
| 学習率 | 1e-4 |

## 結果

### パフォーマンス比較

| 方式 | Best PPL | Best Epoch | パラメータ数 | 訓練時間/epoch |
|------|----------|------------|-------------|----------------|
| **HSA** | **494.4** | 4 | 71,514,632 | ~143s |
| memory_norm | 497.7 | 4 | 70,419,464 | ~84s |

### 差分分析

| 指標 | HSA vs memory_norm |
|------|-------------------|
| PPL差 | -3.30 (0.7%改善) |
| パラメータ増 | +1,095,168 (+1.6%) |
| 訓練時間増 | +70% |

### 学習曲線

```
memory_norm方式:
  Epoch 1: train=1046.7, val=869.5
  Epoch 2: train=259.4,  val=610.9
  Epoch 3: train=135.2,  val=528.6
  Epoch 4: train=82.7,   val=497.7  ← Best
  Epoch 5: train=54.4,   val=498.6  (過学習開始)
  ...
  Epoch 9: → Early stop

HSA方式:
  Epoch 1: train=1051.0, val=861.9
  Epoch 2: train=261.8,  val=611.4
  Epoch 3: train=136.0,  val=514.8
  Epoch 4: train=83.2,   val=494.4  ← Best
  Epoch 5: train=54.9,   val=500.1  (過学習開始)
  ...
  Epoch 9: → Early stop
```

## 方式の詳細

### HSA方式 (Hierarchical Sparse Attention)

論文: arXiv:2511.23319v1

```
Landmark = ChunkEncoder([CLS] + key_sequences)[CLS]

ChunkEncoder:
- 入力: [CLS]トークン + メモリ内のキー列
- 処理: 双方向Transformerエンコーダ（2層）
- 出力: [CLS]位置の出力ベクトル → Landmark

検索: Q_slc @ Landmark / sqrt(d)
- Q_slc: Attention用Qとは別の射影（検索専用）
```

**特徴**:
- 学習可能なLandmark
- メモリ内容を動的に要約
- 追加パラメータ: ChunkEncoder × num_heads

### memory_norm方式

```
Landmark = memory_norm = Σσ(k)

- σ: ELU+1活性化関数
- k: メモリに書き込まれたキーベクトル
- 書き込み操作の副産物を再利用

検索: Q @ Landmark / sqrt(d)
```

**特徴**:
- 追加パラメータなし
- 計算コストが低い
- 学習不可能（書き込み内容に依存）

## 考察

### HSA方式の優位性が限定的な理由

1. **データ規模**: 5,000サンプルでは差が出にくい
2. **メモリ数**: 4メモリでは選択の重要性が低い
3. **過学習**: 両方式ともepoch 4でベスト → データ不足の影響

### 今後の検証ポイント

1. **大規模データ**: サンプル数を増やして再実験
2. **メモリ数増加**: num_memories=8, 16で差が拡大するか
3. **長コンテキスト**: seq_length=512, 1024での挙動
4. **事前構築メモリ**: MemoryBuilderで構築したメモリでの検索精度

## 結論

**現段階ではmemory_norm方式を推奨**

- PPL差はわずか0.7%
- 訓練時間は70%増加
- パラメータ数は1.6%増加

HSA方式のコストに見合う効果が現時点では確認できない。

ただし、以下の条件ではHSA方式が有利になる可能性:
- 大規模なメモリ数（16+）
- 意味的に異なるドメインの事前構築メモリ
- より長いコンテキストでの検索タスク

## 実験コード

```bash
python3 scripts/experiment_hsa_vs_memory_norm.py \
    --samples 5000 \
    --seq-length 256 \
    --num-memories 4 \
    --epochs 30 \
    --patience 5
```
