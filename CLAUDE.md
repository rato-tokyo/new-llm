# Claude Code Development Guidelines for New-LLM Project

## 🎯 プロジェクト概要

**New-LLM**: 文脈ベクトル固定点特性（CVFP Property）を用いた新しい言語モデル

- **データセット**: UltraChat（対話データ）のみ
- **プラットフォーム**: ローカルGPU（逐次処理のためColab不適）
- **二段階学習**:
  - Phase 1: 文脈ベクトルの固有点学習
  - Phase 2: 出力トークン学習

---

## 🎯 Context Vector Fixed-Point Property (CVFP Property) - 削除不能ルール

**New-LLMの根本原理：文脈ベクトル不動点特性**

### 基本仮説

**十分大きい n に対して、n回繰り返した文脈ベクトルと n+1回繰り返した文脈ベクトルはほとんど同じになる**

- **正式名称**: Context Vector Fixed-Point Property (CVFP Property)
- **略称**: CVFP特性
- **日本語**: 文脈ベクトル不動点特性

### Phase 1 訓練の原理

**各iteration で前回の出力を教師データとして使用し、文脈を引き継いで処理**:

```python
# Iteration 1: ゼロから開始、Forward pass only（学習なし）
context = torch.zeros(1, context_dim)
for t, token_embed in enumerate(token_embeds):
    context = model._update_context_one_step(token_embed, context)
    fixed_contexts[t] = context  # 保存

# Iteration 2+: 前回の出力を教師データとして学習
context = torch.zeros(1, context_dim)  # 毎回ゼロから開始
for t, token_embed in enumerate(token_embeds):
    # 各トークンごとに学習
    optimizer.zero_grad()
    context_new = model._update_context_one_step(token_embed, context)
    loss = mse_loss(context_new, fixed_contexts[t])  # 前回の同じ位置と比較
    loss.backward()
    optimizer.step()

    context = context_new.detach()  # 次のトークンへ引き継ぎ（勾配は切る）
    context.requires_grad = True
    fixed_contexts[t] = context_new.detach()  # 次のiterationのために更新
```

**CVFP特性の実現方法**:
- ✅ **文脈の引き継ぎ**: 各トークン処理時、前のトークンの文脈を引き継ぐ
- ✅ **各トークンで学習**: `zero_grad()` → `backward()` → `step()`のサイクル
- ✅ **勾配は切る**: `context.detach()`で勾配を切るが、値は次のトークンへ
- ✅ **固定点に収束**: 前回iterationの同じ位置の文脈と一致するよう学習

**この特性はNew-LLMの存在意義であり、絶対に削除・変更してはならない**

---

## ⚙️ Distribution Regularization - CRITICAL

**次元崩壊を防ぐための正則化手法**

### 問題の経緯

16次元モデルで次元崩壊が発生：
- Train Effective Rank: 4.55/16 (28%)
- Val Effective Rank: 1.01/16 (6%) - Global Attractor

様々なアプローチを試行したが、**Distribution Regularization**が唯一の成功

### 実装

```python
# 全トークンを処理してから学習
all_contexts = []
for t, token_embed in enumerate(token_embeds):
    context = model._update_context_one_step(token_embed.unsqueeze(0), context)
    all_contexts.append(context)
    fixed_contexts[t] = context.detach().squeeze(0)
    context = context.detach()
    context.requires_grad = True

# 損失計算
if iteration > 0:
    optimizer.zero_grad()

    # Stack all contexts: [num_tokens, context_dim]
    all_contexts_tensor = torch.cat(all_contexts, dim=0)

    # CVFP loss: 固定点への収束
    cvfp_loss = mse_loss(all_contexts_tensor, fixed_contexts)

    # Distribution regularization loss
    # 各次元（全トークンでの）が正規分布N(0,1)に近づく
    dim_mean = all_contexts_tensor.mean(dim=0)  # [context_dim]
    dim_var = all_contexts_tensor.var(dim=0)    # [context_dim]

    mean_penalty = (dim_mean ** 2).mean()
    var_penalty = ((dim_var - 1.0) ** 2).mean()
    dist_loss = mean_penalty + var_penalty

    # 合計
    dist_weight = 0.2  # 20% distribution, 80% CVFP
    total_loss = (1 - dist_weight) * cvfp_loss + dist_weight * dist_loss

    total_loss.backward()
    optimizer.step()
```

### 設定方法

```python
# config.py
use_distribution_reg = True     # 分布正則化を使用（推奨）
dist_reg_weight = 0.2           # 分布正則化の重み
                                # 0.2: 80% CVFP, 20% 分布正則化（推奨）
```

### 実験結果（16次元モデル、k=0.2）

| 指標 | DDR baseline | **Distribution Reg** | 改善率 |
|------|-------------|---------------------|--------|
| **Train ER** | 4.55/16 (28%) | **8.33/16 (52%)** | **1.8倍** |
| **Val ER** | 1.01/16 (6%) | **7.54/16 (47%)** | **7.5倍** |
| **Val L2距離** | 0.007 | **4.199** | **600倍** |
| **Val Cosine** | 0.99997 | **0.37698** | Global Attractor解消 |

### 重要な発見

**1. 即座に収束**
```
Iteration 2: Loss=0.103548 (CVFP=0.000000, Dist=0.517741), Converged=100.0%
```
- CVFP Loss = 0 → すべてのトークンが固定点に収束
- 分布正則化のみがペナルティ

**2. ノルムの統一**
```
Train: Avg Norm: 3.999980 (Range: [3.999978, 3.999982])
Val:   Avg Norm: 4.002080 (Range: [3.997518, 4.006338])
```
- 分散=1への正則化 → √16 = 4.0のノルムに収束

---

## 🚫 Phase 1未解決時のPhase 2実行禁止ポリシー - CRITICAL

**⚠️ Phase 1で次元崩壊が解決しない限り、Phase 2は実行しない**

### 基本原則

**Phase 1（固有点学習）で以下の条件を満たさない限り、Phase 2（トークン予測）を実行してはならない**:

1. **Train Effective Rank**: 最低でも 50/256 (20%) 以上
2. **Val Effective Rank**: 最低でも 20/256 (8%) 以上

### 理由

**Phase 1が失敗している状態でPhase 2を実行しても無意味**:
- Val Effective Rank 1.08/256 = ほぼ1次元に崩壊
- この状態で50エポック訓練しても、表現力がない
- 計算時間の無駄（数時間〜数日）

### 実装ルール

```python
# Phase 1終了後、Val Effective Rankをチェック
MIN_TRAIN_RANK = 50.0  # Minimum 50/256 (20%)
MIN_VAL_RANK = 20.0    # Minimum 20/256 (8%)

if train_effective_rank < MIN_TRAIN_RANK or val_effective_rank < MIN_VAL_RANK:
    print_flush("\n⚠️  PHASE 1 FAILED - DIMENSION COLLAPSE DETECTED")
    print_flush(f"   Phase 2 skipped. Fix dimension collapse first.")
    return

# Effective Rankが十分な場合のみPhase 2実行
print_flush(f"\n✅ Phase 1 successful: Val Effective Rank = {val_effective_rank:.2f}/256")
print_flush(f"   Proceeding to Phase 2...")
```

---

## 🎯 Phase 1とPhase 2のTrain/Val区別 - CRITICAL

**Phase 1とPhase 2で、Train/Valの扱いが異なる**

### Phase 1: 固定点の計算

**目的**: 文脈ベクトル生成NNを学習し、各トークン列に対する固定点を計算する

**Train/Val区別**:
- ✅ **Train**: 文脈ベクトル生成layers（context generation layers）を学習
- ✅ **Val**: 学習済みのモデルで固定点を計算（評価のみ、学習なし）

**理想的な動作**:
- Trainで学習した文脈生成NNが、未知のVal dataに対しても安定した固定点を計算できる
- Valの収束率がTrainと同等なら、Phase 1の学習が成功

### Phase 2: トークン予測

**目的**: 固定点文脈ベクトルから次トークンを予測するtoken_output layerを学習

**Train/Val区別**:
- ✅ **Train**: token_output layerを学習
- ✅ **Val**: 学習なし、評価のみ

---

## 🧪 使用方法

### 基本的な実行例

```bash
# デフォルト設定（256次元、4層）
python3 tests/phase2_experiments/test_residual.py

# 16次元モデル（次元崩壊テスト）
python3 tests/phase2_experiments/test_residual.py \
    --context-dim 16 \
    --embed-dim 16 \
    --hidden-dim 32 \
    --num-samples 10

# 3層モデル
python3 tests/phase2_experiments/test_residual.py \
    --num-layers 3

# Distribution Regularizationの重みを変更
python3 tests/phase2_experiments/test_residual.py \
    --context-dim 16 \
    --embed-dim 16 \
    --hidden-dim 32 \
    --dist-reg-weight 0.5  # 50% distribution, 50% CVFP

# Phase 1のみ実行
python3 tests/phase2_experiments/test_residual.py \
    --context-dim 16 \
    --embed-dim 16 \
    --hidden-dim 32 \
    --skip-phase2
```

### すべての引数

```
モデルアーキテクチャ:
  --context-dim INT       文脈ベクトル次元数（デフォルト: 256）
  --embed-dim INT         トークン埋め込み次元数（デフォルト: 256）
  --hidden-dim INT        中間層次元数（デフォルト: 512）
  --num-layers INT        単層ブロックの数（デフォルト: 4）
                          4なら[1,1,1,1], 3なら[1,1,1]を生成

Phase 1設定:
  --phase1-max-iter INT        最大反復回数（デフォルト: 10）
  --phase1-lr-warmup FLOAT     Warmup LR（デフォルト: 0.002）
  --phase1-lr-medium FLOAT     Medium LR（デフォルト: 0.0005）
  --phase1-lr-finetune FLOAT   Finetune LR（デフォルト: 0.0001）

Distribution Regularization:
  --dist-reg-weight FLOAT  正則化の重み（デフォルト: 0.2）
  --no-dist-reg            分布正則化を無効化

Phase 2設定:
  --phase2-lr FLOAT         学習率（デフォルト: 0.0001）
  --phase2-epochs INT       エポック数（デフォルト: 10）
  --phase2-batch-size INT   バッチサイズ（デフォルト: 32）

データ設定:
  --num-samples INT         訓練サンプル数（デフォルト: 10）
  --train-val-split FLOAT   Train/Val分割比率（デフォルト: 0.8）

その他:
  --device STR           デバイス（cpu/cuda、デフォルト: cpu）
  --skip-phase2          Phase 2をスキップ
  --freeze-context       Phase 2で文脈を固定
```

---

## 📊 実験結果の完全確認ポリシー - CRITICAL

**⚠️ 実験結果を報告する際は、必ず全ての情報を確認すること**

### 必須確認項目

実験結果を分析・報告する際は、**以下の全項目を必ず確認**：

1. **収束過程**
   - 全iterationの収束率とLoss
   - Early stoppingのタイミング

2. **固有点分析（FIXED-POINT ANALYSIS）**
   - ✅ Global Attractor Detection（L2距離、Cosine類似度）
   - ✅ Zero Solution Detection（平均ノルム）
   - ✅ Distribution Statistics（ノルム統計、Pairwise距離）
   - ✅ **Information Content（Effective Rank、特異値）** ← **絶対に見落とすな**

3. **Train/Val両方**
   - Trainの結果だけでなく、**Valの結果も必ず確認**
   - Train/Valの差分を分析

### チェックリスト

実験結果報告前に必ず確認：

- [ ] 収束過程の全iterationを確認したか？
- [ ] 固有点分析の4セクション全て確認したか？
- [ ] **Effective Rankを確認したか？**
- [ ] **特異値（Top 5 Singular Values）を確認したか？**
- [ ] Train/Val両方の結果を比較したか？
- [ ] グローバルアトラクター警告（⚠️ DEGENERATE）を見落としていないか？

**この原則を守らないと、重大な問題を見落とす**

---

## 🔧 コード品質ポリシー

### 1. コード重複の徹底排除 - DRY原則

**✅ 正しいパターン - 共通関数で一元化**:

```python
def _compute_batch_metrics(model, input_ids, device, context_loss_weight):
    """共通のメトリクス計算（trainとval両方で使用）"""
    token_loss = F.cross_entropy(...)
    recon_loss = F.mse_loss(...)
    return {'loss': loss, 'token_loss': token_loss, 'recon_loss': recon_loss}

def train_epoch(...):
    metrics = _compute_batch_metrics(...)  # 共通関数を使用
    optimizer.zero_grad()
    metrics['loss'].backward()
    optimizer.step()

def evaluate(...):
    metrics = _compute_batch_metrics(...)  # 同じ関数を使用
```

### 2. ファイル命名規則 - 完全固定方針

- ✅ **固定ファイル名**: 同じ種類のファイルは常に同じ名前を使う
- ❌ **バージョン接尾辞禁止**: `_v1`, `_v2`, `_old`, `_new`, `_fixed` などは使わない
- ❌ **日付接尾辞禁止**: `_20250117`, `_latest` などは使わない
- ✅ **上書き**: 新しいバージョンは常に同じファイル名で上書き

### 3. Code Cleanup Policy

**古いコードを残すことは厳禁です**

- ✗ 使われなくなったメソッド・関数
- ✗ コメントアウトされたコード
- ✗ デバッグ用の一時コード
- ✗ 不要になったパラメータ・引数

---

## 📁 プロジェクト構成

```
new-llm/
├── config.py                    # デフォルト設定
├── src/
│   ├── models/                  # モデル実装
│   └── utils/                   # ユーティリティ
├── tests/
│   ├── phase1_experiments/      # Phase 1実験スクリプト
│   └── phase2_experiments/      # Phase 2実験スクリプト
│       └── test_residual.py     # メイン実験スクリプト
├── docs/
│   └── experiments/             # 実験結果レポート
└── cache/                       # キャッシュ（削除禁止）
    ├── tokenizer/               # トークナイザーキャッシュ
    └── manual_val_tokens.pt     # 手動Validationデータ
```

---

## 🚨 重要な注意事項

### バックグラウンド実行時のログ出力ポリシー

**⚠️ teeコマンドは絶対に使用禁止**

```bash
# ❌ 絶対禁止 - teeは絶対に使わない
python3 -u script.py 2>&1 | tee /tmp/log.txt &

# ✅ 正しい - リダイレクトのみ使用
python3 -u script.py > /tmp/log.txt 2>&1 &
```

**なぜ禁止か**:
- `tee`はパイプ経由のため、**出力が完全にバッファリングされる**
- プロセスが数時間実行されてもログファイルが更新されない

---

## まとめ

**鉄則**:
- ✅ CVFP特性は絶対に削除・変更しない
- ✅ Distribution Regularizationを使用（次元崩壊対策）
- ✅ Phase 1が成功しない限りPhase 2は実行しない
- ✅ 実験結果は全項目（特にEffective Rank）を必ず確認
- ✅ コマンドライン引数で柔軟に設定変更可能
