# CVFPLayer vs CVFPBlock - アーキテクチャ解説

## 概要

New-LLMの階層構造は2段階のレイヤー設計になっています：

- **CVFPLayer**: 基本計算単位（1回のコンテキスト更新）
- **CVFPBlock**: 複数のCVFPLayerをグループ化

## CVFPLayer（レイヤー - 基本単位）

### 役割
- **1回のコンテキスト更新**を実行
- FNN処理、Residual接続、EMA統計更新を含む
- 最小の計算ユニット

### 処理内容
```
入力: context [batch, 16], token_embed [batch, 16]
  ↓
[1] 結合: [context + token] → [batch, 32]
  ↓
[2] FNN: Linear(32, 32) + ReLU
  ↓
[3] 分割: delta_context [batch, 16], delta_token [batch, 16]
  ↓
[4] Residual: new_context = context + delta_context
  ↓
[5] EMA統計更新（自動）
  ↓
出力: new_context [batch, 16], new_token [batch, 16]
```

### 特徴
- 分布正則化統計（running_mean, running_var）を内蔵
- 訓練時に自動的にEMA統計を更新
- `get_distribution_loss()` で損失を取得可能

## CVFPBlock（ブロック - グループ化）

### 役割
- **複数のCVFPLayerを順次実行**
- レイヤー間での情報の段階的変換
- 損失の集約

### 処理内容
```
入力: context, token_embed
  ↓
CVFPLayer #0 → context, token_embed
  ↓
CVFPLayer #1 → context, token_embed
  ↓
CVFPLayer #2 → context, token_embed
  ↓
...（num_layers回繰り返し）
  ↓
出力: 最終的な context, token_embed
```

### 特徴
- `num_layers` で内部のレイヤー数を指定
- 全レイヤーの分布損失を平均化
- レイヤーごとに独立したEMA統計を保持

## 実際の構成例

### 例1: layer_structure = [1, 1]（現在のデフォルト）

```
NewLLMResidual
├── CVFPBlock #0
│   └── CVFPLayer #0  → 1回のコンテキスト更新
│
└── CVFPBlock #1
    └── CVFPLayer #0  → 1回のコンテキスト更新

合計: 2ブロック、各1レイヤー = 2回のコンテキスト更新
```

### 例2: layer_structure = [2, 3]（より深い構造）

```
NewLLMResidual
├── CVFPBlock #0
│   ├── CVFPLayer #0  → 1回目
│   └── CVFPLayer #1  → 2回目
│
└── CVFPBlock #1
    ├── CVFPLayer #0  → 3回目
    ├── CVFPLayer #1  → 4回目
    └── CVFPLayer #2  → 5回目

合計: 2ブロック、5レイヤー = 5回のコンテキスト更新
```

### 例3: layer_structure = [4]（1ブロック、深い構造）

```
NewLLMResidual
└── CVFPBlock #0
    ├── CVFPLayer #0  → 1回目
    ├── CVFPLayer #1  → 2回目
    ├── CVFPLayer #2  → 3回目
    └── CVFPLayer #3  → 4回目

合計: 1ブロック、4レイヤー = 4回のコンテキスト更新
```

## なぜこの設計なのか？

### ブロック分割の利点

1. **階層的な特徴抽出**:
   - Block 0: 低レベル特徴
   - Block 1: 高レベル特徴

2. **損失の段階的管理**:
   - ブロックごとに分布損失を集約
   - デバッグ時にブロック単位で分析可能

3. **柔軟なアーキテクチャ**:
   - `layer_structure` を変更するだけで深さを調整
   - 実験的な構造変更が容易

### レイヤー分離の利点

1. **独立したEMA統計**:
   - 各レイヤーが独自の running_mean/var を保持
   - レイヤーごとに異なる分布特性を学習可能

2. **段階的な正規化**:
   - 各レイヤーで N(0, 1) に近づける
   - 深い構造でも勾配が安定

3. **カプセル化**:
   - 各レイヤーが完全に自己完結
   - テスト・デバッグが容易

## コード例

### config.py での設定

```python
# 2ブロック、各1レイヤー（浅い構造）
layer_structure = [1, 1]

# 2ブロック、各3レイヤー（深い構造）
layer_structure = [3, 3]

# 1ブロック、5レイヤー（単一ブロック深い構造）
layer_structure = [5]
```

### モデル作成

```python
from src.models.new_llm_residual import NewLLMResidual

model = NewLLMResidual(
    vocab_size=50257,
    embed_dim=16,
    context_dim=16,
    hidden_dim=32,
    layer_structure=[2, 3],  # Block0: 2層, Block1: 3層
    use_dist_reg=True
)

# ブロック構造を確認
for i, block in enumerate(model.blocks):
    print(f"Block {i}: {len(block.layers)} layers")
```

### 損失計算

```python
# 全ブロックから分布損失を取得
dist_loss = model.get_distribution_loss()

# ブロックごとの損失を個別に確認
for i, block in enumerate(model.blocks):
    block_loss = block.get_distribution_loss()
    print(f"Block {i}: {block_loss.item():.6f}")
```

## まとめ

| 要素 | CVFPLayer | CVFPBlock |
|------|-----------|-----------|
| **役割** | 1回のコンテキスト更新 | 複数レイヤーのグループ化 |
| **スコープ** | 単一の計算ステップ | 複数の計算ステップ |
| **EMA統計** | 独自の統計を保持 | レイヤーの統計を集約 |
| **損失** | 自身の分布損失 | レイヤーの平均損失 |
| **設定** | パラメータで初期化 | `num_layers` で数指定 |

この設計により、柔軟で保守しやすいアーキテクチャを実現しています。
