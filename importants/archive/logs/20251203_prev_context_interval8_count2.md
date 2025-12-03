# prev_context実験: interval=8, count=2

## 実験設定

| 項目 | 値 |
|------|-----|
| Samples | 800 |
| Context dims | [256, 256] |
| prev_context_interval | 8 |
| prev_context_count | 2 |
| Combined dim | 512 × 3 = **1536** |
| 総パラメータ | 40.90M |

### 設定の意味

- `interval=8, count=2`: **8つ前**と**16つ前**のcontextを使用
- 現在のcontext + 8ステップ前 + 16ステップ前 = 3倍の次元

## 結果サマリー

| 指標 | 値 |
|------|-----|
| **Val PPL** | **162.1** |
| **Val Acc** | **22.7%** |
| Effective Rank | 76.7% (1179/1536) |
| Total time | 889.3s |

### Phase 1 結果

| Block | cd | Iterations | Convergence | Time |
|-------|-----|------------|-------------|------|
| 0 | 256 | 18 | 91% | 27.1s |
| 1 | 256 | 17 | 94% | 25.5s |

## 比較分析

### 同じ800 samplesでの比較

| 構成 | combined_dim | Val PPL | Val Acc | 備考 |
|------|--------------|---------|---------|------|
| [256,256] p=0 | 512 | **150.5** | **23.5%** | ベースライン |
| [256,128] p=0 | 384 | 159.6 | 23.2% | 非均等分割 |
| **[256,256] i=8,c=2** | 1536 | 162.1 | 22.7% | **今回** |

### 1600 samplesでのprev_context比較

| 構成 | combined_dim | Val PPL | Val Acc | 備考 |
|------|--------------|---------|---------|------|
| [256,256] p=0 | 512 | 127.4 | 24.5% | ベースライン |
| [256,256] p=1 (i=1,c=1) | 1024 | **118.2** | **-** | 連続1つ前 |

## 分析

### 1. interval=8 は効果なし（悪化）

- **ベースライン（p=0）**: PPL=150.5, Acc=23.5%
- **interval=8, count=2**: PPL=162.1, Acc=22.7%
- **悪化**: PPL +11.6 (+7.7%), Acc -0.8%

### 2. 間隔が広すぎる問題

8トークン前や16トークン前のcontextは：
- 現在のトークンとの関連性が薄い
- 有用な情報よりもノイズになっている可能性
- combined_dimが3倍（512→1536）になり過学習リスク増

### 3. Effective Rankの低下

- **p=0**: ~80%程度（推定）
- **i=8,c=2**: 76.7%
- 次元が増えても有効な表現が増えていない

### 4. interval=1（連続）の方が効果的

1600 samplesでの比較：
- `p=1 (interval=1)`: PPL=118.2（改善）
- 連続した履歴の方が言語モデリングに有効

## 結論

**interval=8 は効果なし。連続した履歴（interval=1）を推奨。**

### 推奨設定

| 目的 | 設定 |
|------|------|
| ベースライン | `--prev-count 0` |
| 精度向上 | `--prev-interval 1 --prev-count 1` |
| さらなる向上 | `--prev-interval 1 --prev-count 2` (要検証) |

### 避けるべき設定

- `--prev-interval 8` など大きなinterval
- 言語の局所的な依存関係を無視することになる

## コマンド

```bash
# 今回の実験
python3 scripts/experiment_cascade_context.py -s 800 -c 256,256 --prev-interval 8 --prev-count 2

# 推奨: 連続履歴
python3 scripts/experiment_cascade_context.py -s 800 -c 256,256 --prev-interval 1 --prev-count 1
```
