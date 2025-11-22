# Claude Code Development Guidelines for New-LLM Project

## 🎯 New Direction: UltraChat Dialogue Training - CRITICAL UPDATE

**⚠️ 方針転換：Colabは使用せず、ローカルGPUで長期訓練**

### 基本方針（2025年更新）

New-LLMは以下の方向性に確定：

1. **データセット**: UltraChat（対話データ）のみ
2. **プラットフォーム**: ローカルGPU（Colab使用せず）
   - 理由：GPU並列化が困難（逐次処理モデルのため）
   - 長期戦を覚悟
3. **柔軟なアーキテクチャ**: layerと文脈ベクトル次元数を頻繁に変更
4. **二段階学習**:
   - Phase 1: 文脈ベクトルの固有点学習
   - Phase 2: 出力トークン学習

### Colab関連ポリシーの廃止

**以下のポリシーは廃止**:
- ❌ 1行curlコマンド完結ポリシー
- ❌ `colab_train_*.sh` スクリプト要件
- ❌ Colab最適化設定

**理由**: New-LLMは逐次処理のため、Colabの並列GPU利用が困難

### 新しい必須実装要件

訓練スクリプト作成時は以下を実装：

1. ✅ 柔軟なアーキテクチャ（layerと文脈次元を簡単に変更可能）
2. ✅ 二段階訓練（Phase 1: 文脈学習、Phase 2: トークン学習）
3. ✅ キャッシュシステム（固有点文脈ベクトルの保存）
4. ✅ 詳細なメトリクスロギング（ステップごとのPPL, Loss, Accuracy）

---

## 🚫 Phase 1未解決時のPhase 2実行禁止ポリシー - CRITICAL

**⚠️ Phase 1で次元崩壊が解決しない限り、Phase 2は実行しない**

### 基本原則

**Phase 1（固有点学習）で以下の条件を満たさない限り、Phase 2（トークン予測）を実行してはならない**:

1. **Train Effective Rank**: 最低でも 50/256 (20%) 以上
2. **Val Effective Rank**: 最低でも 20/256 (8%) 以上
3. **次元崩壊の兆候なし**: 特異値の1位と2位の比が10倍未満

### 理由

**Phase 1が失敗している状態でPhase 2を実行しても無意味**:
- Val Effective Rank 1.08/256 = ほぼ1次元に崩壊
- この状態で50エポック訓練しても、表現力がない
- 計算時間の無駄（数時間〜数日）
- Phase 1の問題を隠蔽してしまう

### 実装ルール

**test_residual.pyでは、Phase 1の結果をチェックしてからPhase 2を実行すること**:

```python
# Phase 1終了後、Val Effective Rankをチェック
if val_effective_rank < 20.0:
    print_flush(f"\n⚠️  WARNING: Val Effective Rank too low ({val_effective_rank:.2f}/256)")
    print_flush(f"   Phase 2 skipped. Fix dimension collapse first.")
    return

# Effective Rankが十分な場合のみPhase 2実行
print_flush(f"\n✅ Phase 1 successful: Val Effective Rank = {val_effective_rank:.2f}/256")
print_flush(f"   Proceeding to Phase 2...")
```

### このルールを破った場合

- ❌ 数時間の計算時間を浪費
- ❌ Phase 1の本質的な問題が隠される
- ❌ 次元崩壊の原因究明が遅れる

**Phase 1の次元崩壊を解決することが最優先。Phase 2は二の次。**

---

## 🔍 実験結果の完全確認ポリシー - CRITICAL

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

### 絶対禁止事項

❌ **部分的な結果のみで判断する**
- 例：「L2距離だけ見て終わり」
- 例：「Trainだけ見てValを見ない」
- 例：「Effective Rankを見落とす」

❌ **最初の数行だけ見て結論を出す**
- grep結果が途中で切れていても、完全な結果を取得せず報告

✅ **正しい手順**:
1. `grep -A 30` で十分な行数を取得
2. すべてのセクション（1. 2. 3. 4.）を確認
3. Train/Val両方の結果を比較
4. 特異値の分布まで確認

### 理由

**Effective Rankの見落としの深刻さ**:
- Train: Rank 17/256（正常）
- Val: Rank 1/256（**次元崩壊**）
- **この差は最重要情報**なのに、見落とすと「問題なし」と誤判断

**特異値の見落としの深刻さ**:
- 1位と2位の比が1000倍 → **完全に1次元に退化**
- これを見落とすと根本的な問題を見逃す

### チェックリスト

実験結果報告前に必ず確認：

- [ ] 収束過程の全iterationを確認したか？
- [ ] 固有点分析の4セクション全て確認したか？
- [ ] **Effective Rankを確認したか？**
- [ ] **特異値（Top 5 Singular Values）を確認したか？**
- [ ] Train/Val両方の結果を比較したか？
- [ ] グローバルアトラクター警告（⚠️ DEGENERATE）を見落としていないか？

**この原則を守らないと、重大な問題を見落とす**

### 二段階訓練の詳細

**Phase 1: 文脈ベクトル学習**:
```
1. サンプルをトークン配列に変換
2. 一周目：文脈ベクトルをただ保存
3. 二周目以降：前回の文脈ベクトルを教師データとして学習
4. 固有点が定まるまで繰り返し
5. 収束判定：95%以上のトークンが収束 → Phase 2へ
6. 収束しない場合：layerや文脈次元を調整
```

**Phase 2: 出力トークン学習**:
```
1. Phase 1で確定した固有点を固定
2. 期待した出力トークンが出力されるように学習
3. 自分の返答のトークンのみ学習（相手の返答は学習しない）
4. ChatML形式で発言者を識別
```

---

## ⚙️ 正則化ポリシー - CRITICAL

**CVFP訓練における正則化の方針**

### 不要な正則化の削除

CVFP訓練では、固有点の学習において**各次元が重要な情報を保持すべき**です。以下の正則化は不要：

- ❌ **Context Clipping**は削除（`use_context_clipping = False`）
  - Layer Normalizationが既にベクトルの大きさを正規化している
  - 冗長な制約を加えるだけ

- ❌ **Dropout**は削除
  - 各次元がランダムに無効化されると、固有点学習が妨げられる
  - 各次元が重要な情報を表現するのは問題ない（むしろ望ましい）

### 使用する正則化

1. **Layer Normalization**（`use_layer_norm = True`）
   - ベクトルの分散を安定化
   - 学習を安定させる

2. **DDR (Dimension Diversity Regularization)**（`use_ddr = True`）
   - **目的**: 次元崩壊を防ぐ
   - **方式**: モデル出力の次元別活性をEMAで追跡し、低活性次元をブースト
   - **効果**: すべての次元が均等に使用され、256次元中1-2次元だけに情報が集中する問題を解決

**DDR (Dimension Diversity Regularization) の詳細**:

**問題**:
- 4層: Effective Rank 1.32 / 256 (0.5%) - 1-2次元のみ使用
- 8層: Effective Rank 1.00 / 256 (0.4%) - 完全に1次元に崩壊
- 254次元がほぼゼロ、すべてのトークンが同じ1-2次元の値に収束

**DDRの仕組み（教師ベクトルブースト方式）**:
```python
# モデル出力の次元別活性を追跡（EMA）
# 重要: 教師ではなく、モデルの実際の出力を追跡
ddr_dim_activity = (
    momentum * ddr_dim_activity +
    (1 - momentum) * abs(model_output)
)

# 低活性次元を検出してブースト計算
mean_activity = ddr_dim_activity.mean()
threshold = mean_activity * threshold_ratio  # 0.5 = 平均の50%未満をブースト

boost_mask = ddr_dim_activity < threshold
boost_amount = (threshold - ddr_dim_activity) / (threshold + eps)

# 教師ベクトルに低活性次元のブーストを追加
target_adjusted = target + boost_weight * boost_amount

# 通常の固定点損失
loss = mse_loss(model_output, target_adjusted)
```

**設定方法**:
```python
# config.py
use_ddr = True                  # DDRを使用（推奨）
ddr_momentum = 0.9              # EMAのモメンタム
ddr_boost_weight = 0.1          # ブースト重み
ddr_threshold_ratio = 0.5       # 平均の50%未満をブースト対象
                                # 1.0 = 平均未満すべて（広範囲、約50%の次元）
                                # 0.5 = 平均の50%未満のみ（厳格、約25%の次元）
```

**重要な設計判断**:

1. **モデル出力を追跡（教師ではない）**:
   - ✅ モデルが実際に使っている次元を把握
   - ❌ 教師ベクトルはiteration毎に変化するため追跡が不安定
   - 目的: モデルの出力が偏っている次元を特定し、修正

2. **threshold_ratioの意味**:
   - 各次元の**平均活性**（EMAで追跡された長期的な活性レベル）を比較
   - `1.0`: 全次元の平均活性の100%未満の次元をブースト → 平均以下の次元すべて
   - `0.5`: 全次元の平均活性の50%未満の次元のみブースト → 極端に低活性な次元のみ
   - **推奨**: `0.5`（極端に低活性な次元のみ修正）
   - **重要**: 直前の出力値ではなく、EMAで追跡された「その次元の典型的な活性レベル」を評価

**命名**:
- **DDR**: Dimension Diversity Regularization（次元別多様性正則化）
- **EMA**: Exponential Moving Average（指数移動平均）
  - `mean_new = momentum * mean_old + (1 - momentum) * current_value`

**実装の特徴**:
- ✅ モデルコードを一切変更しない（訓練ループのみ）
- ✅ 損失計算が複雑にならない（通常のMSE損失のみ）
- ✅ デバッグが容易
- ✅ 教師ベクトル調整により、モデルが低活性次元を使うよう誘導

---

## 🎯 Context Vector Fixed-Point Property (CVFP Property) - 削除不能ルール

**New-LLMの根本原理：文脈ベクトル不動点特性**

### 基本仮説

**十分大きい n に対して、n回繰り返した文脈ベクトルと n+1回繰り返した文脈ベクトルはほとんど同じになる**

- **正式名称**: Context Vector Fixed-Point Property (CVFP Property)
- **略称**: CVFP特性
- **日本語**: 文脈ベクトル不動点特性

### 具体例

```
入力文脈：「赤いリンゴ」を n回繰り返した後の文脈ベクトル
入力トークン：「赤」

→ 出力文脈：入力文脈とほぼ同じ（不動点特性）
```

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
    context_new = model._update_context_one_step(token_embed, context)
    loss = mse_loss(context_new, fixed_contexts[t])  # 前回の同じ位置のcontextと比較
    loss.backward()
    optimizer.step()
    context = context_new.detach()  # 次のトークンへ引き継ぎ（勾配は切る）
    fixed_contexts[t] = context_new  # 次のiterationのために更新
```

**CVFP特性の実現方法**:
- ✅ **文脈の引き継ぎ**: 各トークン処理時、前のトークンの文脈を引き継ぐ
- ✅ **教師データ**: 前回iteration の同じ位置の文脈ベクトルを教師として使用
- ✅ **例**: Iteration 1で「赤いリンゴ」を処理して得た各位置の文脈が、Iteration 2で「赤いリンゴ赤いリンゴ...」を処理した時の同じ位置（0,1,2）の文脈と一致するよう学習
- ✅ 固定点に収束するまで繰り返す

### warmup_iterations（n_warmup）

- **現在の設定**: `n_warmup = 0`（最初から固定点学習）
- **理由**: 十分な試行錯誤（iteration）があれば、ゼロから開始しても固定点に収束する
- **変数名**: `warmup_iterations` または `n_warmup`

### CVFP特性の意義

1. **任意長シーケンス対応**: 同じパターンが繰り返される長文でも安定した表現
2. **メモリ効率**: 固定サイズの文脈ベクトルのみで長文を処理
3. **RNN/LSTM的な性質**: 逐次処理で過去情報を圧縮

**この特性はNew-LLMの存在意義であり、絶対に削除・変更してはならない**

---

## 🚨 Phase 1実装の重大バグと修正 - CRITICAL

**2025-01-22発見：Phase 1の実装バグが収束失敗の原因だった**

### ❌ 間違った実装（バグあり）

```python
for t, token_embed in enumerate(token_embeds):
    context = model._update_context_one_step(token_embed, context)

    if iteration > 0:
        loss = mse_loss(context, fixed_contexts[t])
        loss.backward(retain_graph=True)  # 各トークンでbackward

    context = context.detach()
    context.requires_grad = True

# 最後に1回だけoptimizer.step()
if iteration > 0:
    optimizer.step()  # ❌ 問題：各トークンの学習が独立していない
```

**問題点**:
1. 各トークンで`backward()`するが、`optimizer.step()`は最後に1回のみ
2. 勾配が蓄積されるが、最後のトークンの勾配で上書きされる可能性
3. **各トークンが独立に固定点を学習できない**
4. 結果：4層でも収束が遅い、2層では0%収束

### ✅ 正しい実装（CVFP準拠）

```python
for t, token_embed in enumerate(token_embeds):
    if iteration > 0:
        optimizer.zero_grad()  # ✅ 各トークンで勾配リセット

    context = model._update_context_one_step(token_embed, context)

    if iteration > 0:
        loss = mse_loss(context, fixed_contexts[t])
        loss.backward()        # ✅ 各トークンでbackward
        optimizer.step()       # ✅ 各トークンでstep

    context = context.detach()  # ✅ 次のトークンへ引き継ぎ（勾配は切る）
    context.requires_grad = True
```

**正しい理由**:
1. ✅ **各トークンが独立に学習**: `zero_grad()` → `backward()` → `step()`のサイクル
2. ✅ **文脈は引き継ぐ**: `context.detach()`で勾配を切るが、値は次のトークンへ
3. ✅ **CVFP特性を実現**: 各トークン位置で独立に固定点を学習

### 実験結果の比較（256次元、10サンプル）

#### 4層 [1,1,1,1] の比較

| Implementation | Iteration 10 収束率 | Iteration 18 収束率 | 最終結果 |
|----------------|-------------------|-------------------|---------|
| **バグあり** | 34.4% | - | 99.7% (Iter 20) |
| **CVFP準拠** | **75.1%** | **99.6%** | 99.6% (Iter 18) |

**改善**:
- Iteration 10で**2.2倍速**（34.4% → 75.1%）
- **2 iteration早く収束**（Iter 20 → Iter 18）

#### 2層 [1,1] の比較

| Implementation | Iteration 10 収束率 | 最終結果 | Train Effective Rank |
|----------------|-------------------|---------|---------------------|
| **バグあり** | 0.0% | 0.0% (Iter 50) | - |
| **CVFP準拠** | **37.0%** | **100.0% (Iter 25)** | 7.24 / 256 (2.8%) |

**改善**:
- Iteration 10で**0% → 37.0%**（収束不可能 → 収束可能）
- **完全収束達成**（0% → 100.0%）

#### 4層 vs 2層（CVFP準拠、256次元）

| 指標 | 4層 [1,1,1,1] | 2層 [1,1] | 差 |
|------|--------------|----------|-----|
| **収束速度（Iter 10）** | 75.1% | 37.0% | **4層が2倍速** |
| **最終収束（完全）** | Iter 18 (99.6%) | Iter 25 (100%) | **4層が7 iter速い** |
| **Effective Rank** | 21.38 / 256 (8.4%) | 7.24 / 256 (2.8%) | **4層が3倍多様** |
| **Train L2距離** | 3.495 | 2.676 | **4層が30%大きい** |
| **Train Cosine類似度** | 0.850 | 0.916 | **4層が明確に分離** |

**結論**:
- ✅ 2層でも収束可能（バグ修正により）
- ⚠️ 4層推奨（2倍速く収束、3倍多様、固定点が明確に分離）
- ❌ 2層のEffective Rank 2.8%は表現力不足のリスク

### なぜこのバグが見つからなかったのか

**理由**:
1. 4層では最終的に99%以上収束していた（遅いが成功）
2. バグの影響が「収束速度の低下」として現れ、「完全な失敗」ではなかった
3. 2層で0%収束が発生して初めて、根本的な実装ミスに気づいた

### 教訓

**Phase 1実装の鉄則**:
- ✅ **各トークンごとに**: `optimizer.zero_grad()` → `backward()` → `step()`
- ✅ **文脈は引き継ぐが勾配は切る**: `context.detach()` + `requires_grad = True`
- ❌ **全トークン処理後に1回step()は禁止**: 各トークンの学習が独立しない

**このバグは二度と繰り返してはならない**

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

**評価指標**:
```python
# Train: 学習
train_contexts = phase1_train(model, train_ids)  # 学習あり

# Val: 評価のみ
val_contexts = compute_fixed_contexts(model, val_ids)  # 学習なし、固定点計算のみ

# 期待される結果:
# - Train: 99.5%収束（学習済み）
# - Val: 99.5%収束（汎化成功） ← 理想
# - Val: 収束率が低い → Phase 1の学習失敗（過学習）
```

### Phase 2: トークン予測

**目的**: 固定点文脈ベクトルから次トークンを予測するtoken_output layerを学習

**Train/Val区別**:
- ✅ **Train**: token_output layerを学習
- ✅ **Val**: 学習なし、評価のみ

**汎化性能の評価**:
- Phase 1で計算された固定点文脈ベクトル（Train/Val両方）を使用
- Phase 2でTrainのtoken_output layerを学習
- Valで真の汎化性能を評価

### 実装の注意点

**正しい実装**:
```python
# Phase 1
train_contexts = phase1_train(model, train_ids)           # 学習
val_contexts = compute_fixed_contexts(model, val_ids)     # 評価のみ

# Phase 2
phase2_train(model, train_ids, train_contexts, ...)       # 学習
phase2_evaluate(model, val_ids, val_contexts, ...)        # 評価のみ
```

**間違った実装**:
```python
# ❌ 間違い: Valでも学習してしまう
train_contexts = phase1_train(model, train_ids)
val_contexts = phase1_train(model, val_ids)  # ❌ 学習してしまう
```

**この区別はPhase 1の汎化性能評価に必須であり、絶対に守ること**

---

## 🚨🚨🚨 バックグラウンド実行時のログ出力ポリシー - 最重要・絶対厳守 🚨🚨🚨

**⚠️ 重大問題：ログで進捗が確認できないのは致命的エラーです ⚠️**

このミスを何度も繰り返しています。**絶対にこれを守ってください**。

### 🔴 絶対禁止：`tee`コマンドの使用

```bash
# ❌❌❌ 絶対禁止 - teeは絶対に使わない ❌❌❌
python3 -u script.py 2>&1 | tee /tmp/log.txt &

# ❌ この形式も禁止
python3 script.py | tee log.txt &
python3 script.py 2>&1 | tee -a log.txt &
```

**なぜ禁止か**:
- `tee`はパイプ経由のため、**出力が完全にバッファリングされる**
- プロセスが数時間実行されてもログファイルが更新されない
- **進捗が全く確認できず、ユーザーは何も見えない**
- `python3 -u`（unbuffered）も`tee`経由では無意味

### ✅ 正しい方法：リダイレクトのみ

```bash
# ✅✅✅ これが唯一の正しい方法 ✅✅✅
python3 -u script.py > /tmp/log.txt 2>&1 &

# または stdbuf を使用（推奨）
stdbuf -oL -eL python3 script.py > /tmp/log.txt 2>&1 &
```

**なぜ正しいか**:
- リダイレクト（`>`）は直接ファイルに書き込む
- バッファリングが最小限
- `python3 -u`が正しく機能する
- **リアルタイムでログファイルが更新される**

### 🔍 ログ監視の正しい方法

```bash
# バックグラウンド実行
python3 -u train.py > /tmp/train.log 2>&1 &

# 別のターミナルで進捗監視
tail -f /tmp/train.log

# または定期的にチェック
watch -n 10 "tail -20 /tmp/train.log"
```

### 📋 バックグラウンド実行の必須チェックリスト

**すべてのバックグラウンド実行前に必ず確認**:

- [ ] ❌ `tee`コマンドを使っていないか？ → **絶対禁止**
- [ ] ✅ `>` リダイレクトのみ使用しているか？
- [ ] ✅ `python3 -u`でunbuffered指定しているか？
- [ ] ✅ `2>&1`でstderrもリダイレクトしているか？
- [ ] ✅ ログファイルがリアルタイム更新されるか事前確認したか？

### 🔥 なぜこれが最重要か

1. **ユーザー体験の破壊**: 何時間も実行しても進捗が見えない
2. **繰り返されるミス**: 何度も同じ問題が発生している
3. **時間の浪費**: プロセスを停止・再実行する無駄な時間
4. **信頼性の喪失**: ログが見えないシステムは使えない

### ⚡ 実行例

```bash
# ❌ 間違い - 何度も繰り返された失敗パターン
python3 -u tests/phase2_experiments/test_multi_sample.py --num-samples 10 2>&1 | tee /tmp/test.log &

# ✅ 正しい - 常にこの形式を使う
python3 -u tests/phase2_experiments/test_multi_sample.py --num-samples 10 > /tmp/test.log 2>&1 &

# 進捗確認
tail -f /tmp/test.log
```

---

**🚨 この原則を破ったら、即座に停止してやり直してください 🚨**

---

## 🧪 コード修正時の必須テストポリシー - CRITICAL

**すべてのコード修正後、必ず手元でテストしてからコミット・デプロイすること**

### 必須実行手順

**テストは最小限のデータ・epoch・stepで実施する**

```bash
# ステップ1: 構文チェック
python3 -m py_compile train.py

# ステップ2: 最小限のデータでエンドツーエンドテスト
python3 train.py \
    --max-samples 100 \      # 最小限のデータ（100サンプル）
    --epochs 2 \             # 最小限のepoch（2エポック）
    --batch-size 4 \
    --device cpu

# ステップ3: 動作確認
# - エポックごとのメトリクス表示を確認
# - エラーなく完了することを確認
```

**テスト時間の目安**:
- 100サンプル、2エポック → **4-5分**

**鉄則**: テストするときは**epoch=2、samples=100**で十分

### チェックリスト

コミット前に必ず確認：

- [ ] **構文チェック完了** - `python3 -m py_compile`
- [ ] **テストスクリプトで動作確認**
- [ ] **エラーメッセージを実際に確認**

---

## 🔄 コード品質・保守性の鉄則 - CRITICAL

**コードの重複、パラメータの不整合、キャッシュ不足によるミスを防ぐ**

### 1. コード重複の徹底排除 - DRY原則

**❌ 絶対に避けるべきパターン**:

```python
# ❌ 悪い例 - trainとevalで同じロジックを重複実装
def train_epoch(...):
    token_loss = F.cross_entropy(...)
    recon_loss = F.mse_loss(...)
    # ... 約90行のロジック

def evaluate(...):
    # ← まったく同じロジックをコピペ
    token_loss = F.cross_entropy(...)
    recon_loss = F.mse_loss(...)
    # ... 約90行の重複コード
```

**✅ 正しいパターン - 共通関数で一元化**:

```python
# ✅ 良い例 - 共通ロジックを1箇所に集約
def _compute_batch_metrics(model, input_ids, device, context_loss_weight):
    """共通のメトリクス計算（trainとval両方で使用）"""
    token_loss = F.cross_entropy(...)
    recon_loss = F.mse_loss(...)

    return {
        'loss': loss,
        'token_loss': token_loss,
        'recon_loss': recon_loss,
    }

def train_epoch(...):
    metrics = _compute_batch_metrics(...)  # 共通関数を使用
    optimizer.zero_grad()
    metrics['loss'].backward()
    optimizer.step()

def evaluate(...):
    metrics = _compute_batch_metrics(...)  # 同じ関数を使用
```

**チェックリスト**:
- [ ] trainとevalで同じロジックを重複実装していないか？
- [ ] 共通処理を`_compute_*`などの関数に抽出したか？
- [ ] メトリクス追加時に1箇所修正で済むか？

---

### 2. パラメータ同期の徹底 - CRITICAL

新しいパラメータを追加する際は、**必ず以下の全てを更新**すること：

1. **`train.py`** - `argparse`にパラメータ定義
2. **`scripts/colab_train_*.sh`** - パラメータ解析に追加
3. **ドキュメント** - 使用例を記載

**チェックリスト**:
- [ ] `train.py`にパラメータ定義を追加したか？
- [ ] **`scripts/colab_train_*.sh`にパラメータ解析を追加したか？** ← 最重要
- [ ] ドキュメントに使用例を記載したか？

---

### 3. キャッシュの積極活用 - パフォーマンス最適化

**✅ 正しいパターン - キャッシュを活用**:

```python
def create_tokenizer(texts, vocab_size=10000, output_dir='./tokenizer'):
    """Create or load BPE tokenizer"""
    tokenizer_path = f"{output_dir}/tokenizer.json"

    # 既存のトークナイザーを確認
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}...")
        return Tokenizer.from_file(tokenizer_path)  # 0.1秒で完了

    # 初回のみ訓練
    print(f"Training BPE tokenizer...")
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer
```

**キャッシュすべきもの**:
1. **トークナイザー** - `tokenizer.json`
2. **データセット** - HuggingFace Datasetsが自動キャッシュ
3. **モデルチェックポイント** - 実験再開時に再利用

---

### まとめ

**二度と同じミスをしないための鉄則**:

1. **DRY原則を徹底** - コードを重複させない
2. **パラメータ同期を徹底** - `train.py`と`colab_train_*.sh`の両方を更新
3. **キャッシュを活用** - 同じ処理を繰り返さない

---

## 🚀 GPU最適化設定の確認 - CRITICAL

**スクリプト作成・修正時は、必ずGPU用の設定になっているか確認すること**

### GPU別の推奨batch_size

| GPU | VRAM | 推奨batch_size | 備考 |
|-----|------|---------------|------|
| **T4** | 16GB | 512 | Baseline model |
| **L4** | 24GB | **2048** | **4x T4** |
| **A100** | 40GB | **4096** | **8x T4** |

### Learning Rate Scaling Rule - CRITICAL

**batch_sizeを変更したら、learning_rateも適切にスケールすること**

**大規模batch（>= 256）**: Square Root Scaling（推奨）
```
batch_sizeをk倍 → learning_rateを√k倍
```

**正しい計算（CPU baseline基準）**:
```python
# CPU Baseline
batch_size = 32
learning_rate = 0.0001

# L4 GPU（batch 64倍 = 32→2048）
batch_size = 2048
# Square Root Scaling: 0.0001 * √64 = 0.0001 * 8 = 0.0008
learning_rate = 0.0008  # √64 = 8倍

# A100 GPU（batch 128倍 = 32→4096）
batch_size = 4096
# Square Root Scaling: 0.0001 * √128 ≈ 0.0001 * 11.3 = 0.0011
learning_rate = 0.0011  # √128 ≈ 11.3倍
```

**チェックリスト**:
- [ ] batch >= 256: **Square Root Scaling（√k倍）** ← 推奨
- [ ] 最初のエポックでPPLが急速に減少しているか？

---

### Model Size Scaling Rule - CRITICAL

**モデルサイズを大きくしたら、learning_rateを小さくすること**

**基本原則**:
```
モデルサイズが大きい → learning_rateを小さくする
```

**推奨スケーリング**:
```python
# Baseline (2.74M params)
learning_rate = 0.0004

# Advanced (4.84M params, 1.77x larger)
learning_rate = 0.0002  # 半分に減らす（保守的）
```

**チェックリスト**:
- [ ] learning_rateを小さくしたか？
- [ ] 大きなモデルが小さなモデルより性能が良いか？（そうでなければLR調整不足）

---

### 設定の一元管理ポリシー - CRITICAL

**すべての設定値は `src/utils/config.py` で一元管理すること**

```python
# ✅ 良い例 - config.pyから継承
from src.utils.config import NewLLMGPUConfig

class MyConfig(NewLLMGPUConfig):
    # GPU最適化設定を自動継承
    num_epochs = 100
```

**利用可能な設定クラス**:
- `NewLLMConfig` - CPU訓練（レガシー）
- `NewLLMGPUConfig` - T4 GPU訓練
- **`NewLLML4Config`** - **L4 GPU訓練（推奨）**
- **`NewLLMA100Config`** - **A100 GPU訓練**

---

## New-LLM Architecture Design Principles - CRITICAL

### 🎯 固定メモリ使用量の原則（Fixed Memory Usage Principle）

**New-LLMの根本的設計目標**: シーケンス長に関わらず**メモリ使用量が一定**であること

これはNew-LLMの存在意義であり、**絶対に守るべき原則**です。

### Transformerとの比較

| アーキテクチャ | メモリ使用量 | シーケンス長の制約 |
|--------------|-------------|------------------|
| **Transformer** | O(n²) | Attentionが長いシーケンスで爆発 |
| **New-LLM** | O(1) | 固定サイズ文脈ベクトルで任意長対応可能 |

### 実装で絶対に禁止すること ❌

**1. 位置埋め込み（Positional Embeddings）の使用**

```python
# ❌ 絶対禁止 - max_seq_lengthに制限される
self.position_embedding = nn.Embedding(max_seq_length, embed_dim)

# 問題点:
# - 学習時のmax_seq_length以上のシーケンスを処理できない
# - New-LLMの設計思想（任意長対応）に反する
```

**理由**:
- RNN/LSTMと同様、位置情報は**逐次処理の順序から暗黙的に学習**される
- `context[t] = f(context[t-1], input[t])` の更新順序自体が位置情報を内包
- 明示的な位置埋め込みは不要かつ有害

**2. シーケンス長に依存する操作**

```python
# ❌ 禁止 - 全隠れ状態を保存
hidden_states = []
for t in range(seq_len):
    hidden_states.append(hidden[t])  # メモリが線形増加

# ✅ 正しい - 固定サイズの文脈ベクトルのみ保持
context = torch.zeros(batch_size, context_dim)  # 固定サイズ
for t in range(seq_len):
    context = update(context, input[t])  # 上書き更新
```

### 開発時のチェックリスト

New-LLMの実装・修正時は必ず確認：

- [ ] `nn.Embedding(max_seq_length, ...)` のような位置埋め込みを使っていないか？
- [ ] `hidden_states.append(...)` のような全状態保存をしていないか？
- [ ] メモリ使用量がシーケンス長に依存する操作をしていないか？
- [ ] 任意長のシーケンスを処理できる設計になっているか？

### この原則を守る理由

1. **New-LLMの存在意義**: Transformerの O(n²) 問題を O(1) で解決
2. **スケーラビリティ**: 長いシーケンス（小説、コード全体など）を処理可能
3. **リソース効率**: メモリが限られた環境でも動作

**違反した場合**: New-LLMがただの「重いTransformer」になり、研究的価値を失う

---

## 📁 プロジェクト構成整理ポリシー - CRITICAL

**プロジェクト直下のファイルは最小限に保ち、整理すること**

### 1. ドキュメント配置ルール

**プロジェクト直下に配置可能なmdファイル**:
- ✅ `README.md` - プロジェクト概要（必須）
- ✅ `CLAUDE.md` - 開発ガイドライン（このファイル）

**docsフォルダに配置すべきmdファイル**:
- 📄 `docs/` - すべてのドキュメント（仕様書、設計書、解説など）
- 📊 `docs/experiments/` - すべての実験結果レポート

### 2. テストスクリプト配置ルール

**プロジェクト直下に配置可能なスクリプト**:
- ✅ `train_dialogue.py` - メイン訓練スクリプト（本番用）

**testsフォルダに配置すべきスクリプト**:
- 🧪 `tests/phase1_experiments/` - Phase 1実験スクリプト
  - 収束テスト、トークン数実験、アトラクター検証など
- 🧪 `tests/phase2_experiments/` - Phase 2実験スクリプト
  - トークン予測訓練、アーキテクチャ比較など

### 整理の理由

- **明確な構造**: プロジェクト直下が煩雑にならない
- **検索性向上**: 関連ファイルが一箇所に集約
- **スケーラビリティ**: ファイル数が増えても管理しやすい
- **実験の分類**: Phase 1/2で明確に分離

### チェックリスト

新しいファイルを作成する際は必ず確認：

**ドキュメント**:
- [ ] README.mdまたはCLAUDE.mdか？ → プロジェクト直下に配置
- [ ] 実験結果レポートか？ → `docs/experiments/` に配置
- [ ] その他のドキュメントか？ → `docs/` に配置

**スクリプト**:
- [ ] メイン訓練スクリプトか？ → プロジェクト直下に配置
- [ ] Phase 1実験スクリプトか？ → `tests/phase1_experiments/` に配置
- [ ] Phase 2実験スクリプトか？ → `tests/phase2_experiments/` に配置

### 3. ドキュメント内容ルール

**提案・アイデアはmdファイルに残さない**:
- ❌ ユーザーへの提案をドキュメントとして保存
- ❌ 「次にすべきこと」「推奨事項」などの提案内容
- ✅ 実験結果の事実のみ記録
- ✅ 実装済み機能の仕様・使用方法のみ記録

**理由**:
- 提案は会話で完結すべき
- ドキュメントは事実ベースの記録のみ
- 提案を残すと古い情報が蓄積する

**正しい記載例**:
```markdown
# ✅ 良い例 - 事実ベースの記録
## 実験結果
- Mixed [2,2]: PPL = 31.74
- Layer-wise [1,1,1,1]: PPL = 37.95

## 実装済み機能
- Early Stopping (src/utils/early_stopping.py)
- Phase1EarlyStopping: 収束率ベース
- Phase2EarlyStopping: Validation Lossベース

# ❌ 悪い例 - 提案を記載
## 次にすべきこと
- 10サンプルでテストを実行してください
- 学習率を調整することを推奨します
```

---

## Code and File Cleanup Policy - CRITICAL

**古いコード・ファイルを残すことは厳禁です**

### 重要：無効化ではなく削除

**不要なコードは必ず削除すること。コメントアウトや無効化では不十分です。**

```python
# ❌ 禁止 - コメントアウトで残す
# old_function()

# ✅ 正しい - 完全に削除
# （不要なコードは何も残さない）
```

### 削除対象

- ✗ 使われなくなったメソッド・関数
- ✗ コメントアウトされたコード
- ✗ デバッグ用の一時コード
- ✗ 古いバージョンのファイル
- ✗ 不要になったパラメータ・引数

### 保護対象（削除してはいけないもの）

- ✅ **experiments/**: 実験結果ログ（`.log`, `.png`, `.txt`など）
- ✅ **cache/tokenizer/**: トークナイザーキャッシュ（再作成に時間がかかる）
- ✅ **HuggingFaceデータセットキャッシュ**: 再ダウンロード防止

**実験結果は削除対象ではない**: experiments/フォルダで一元管理し、gitにコミット

### チェックリスト

コミット前に必ず確認：

- [ ] コメントアウトされたコードは削除したか？
- [ ] 未使用のメソッド・関数は削除したか？
- [ ] 古いバージョンのファイルは削除したか？
- [ ] 未使用のインポート文は削除したか？
- [ ] **実験結果（experiments/）は保護したか？**

---

## コミットメッセージ - CRITICAL

**コミットメッセージは1行で簡潔に記述すること**

```bash
# ✅ 正しい - 1行で簡潔に
git commit -m "Add WikiText dataset support"
git commit -m "Fix perplexity calculation bug"
git commit -m "Remove obsolete parameters from train.py"
```

**原則**:
- 1行で何をしたか明確に記述
- 50文字以内を目安
- 英語で記述
- 動詞で始める（Add, Fix, Remove, Update など）

---

## ファイル命名規則 - 完全固定方針

**ファイル名は完全に固定し、常に同じ名前で上書きする**

### 基本原則

- ✅ **固定ファイル名**: 同じ種類のファイルは常に同じ名前を使う
- ❌ **バージョン接尾辞禁止**: `_v1`, `_v2`, `_old`, `_new`, `_fixed` などは使わない
- ❌ **日付接尾辞禁止**: `_20250117`, `_latest` などは使わない
- ✅ **上書き**: 新しいバージョンは常に同じファイル名で上書き

### 例外

固定ファイル名の例外（複数保持が許される場合）:

- ✅ 異なる実験の結果（例: `experiment1_results.md`, `experiment2_results.md`）
- ✅ 異なるモデルアーキテクチャ（例: `new_llm_curves.png`, `transformer_curves.png`）
- ✅ 異なるデータセット（例: `wikitext_results.md`, `bookcorpus_results.md`）

**重要**: 同じ種類のファイルは必ず固定名で上書き

---

## 🔬 CVFPT実験ガイドライン - CRITICAL

**CVFPT (Context Vector Fixed Point Training) 実験時の注意事項**

### 正しいスクリプトの使用

**✅ CVFPT実験には `scripts/train_repetition.py` を使用**

```bash
# ✅ 正しい - トークンIDを直接使用
python3 scripts/train_repetition.py \
    --max-stage 1 \
    --epochs-per-stage 10 \
    --repetitions 10 \
    --device cpu
```

**❌ `train.py` は使わない** - WikiText訓練用（トークナイゼーションが必要）

### 重要な違い

| スクリプト | 用途 | トークナイゼーション |
|----------|------|------------------|
| `train.py` | WikiText訓練 | ✅ 必要（全データセット） |
| **`scripts/train_repetition.py`** | **CVFPT実験** | **❌ 不要（トークンID直接使用）** |

### キャッシュの保護

**`./cache/` ディレクトリは絶対に削除しない**

- ✅ トークナイザーキャッシュ (`tokenizer.json`)
- ✅ HuggingFace Datasetsキャッシュ
- ✅ モデルチェックポイント

**理由**: 再実験時に毎回トークナイゼーションする無駄を防ぐ（数分→0.1秒）

---

## ⚠️ グローバルアトラクター問題 - CRITICAL

**退化解：すべてのトークンが同一の固定点に収束する問題**

### 問題の症状

- ✗ 異なるトークン間のL2距離が異常に小さい（0.000002など）
- ✗ コサイン類似度が異常に高い（0.999以上）
- ✗ 収束ステップが異常に少ない（1ステップ）
- ✗ すべてのトークンが同じ文脈ベクトルに収束

### 根本原因

**Simple Overwrite Updater** が問題：

```python
# ❌ Simple Updater - 以前の文脈を無視
context_new = tanh(W @ hidden)

# 問題点:
# 1. 以前の context を完全に無視
# 2. LayerNorm + Clipping と組み合わさると全トークンが同一点に収束
# 3. トークン固有の情報が失われる
```

### 解決策：Gated Context Updater（必須）

**✅ Gated Updater - LSTM型の更新（標準設定）**

```python
# ✅ Gated Updater - 以前の文脈を保持
context_delta = tanh(W_delta @ hidden)
forget_gate = sigmoid(W_forget @ hidden)
input_gate = sigmoid(W_input @ hidden)
context_new = forget_gate * context + input_gate * context_delta
```

**`src/utils/config.py` での設定**:

```python
context_update_strategy = "gated"  # DEFAULT（変更禁止）
```

### 診断スクリプト

グローバルアトラクター問題を検出：

```bash
# 異なるトークン間の距離をチェック
python3 scripts/check_global_attractor.py

# 期待される結果:
# - Gated Updater: 平均L2距離 > 2.0（トークン固有）
# - Simple Updater: 平均L2距離 < 0.001（グローバルアトラクター）
```

### チェックリスト

CVFPT実験前に必ず確認：

- [ ] `config.context_update_strategy = "gated"` を確認
- [ ] 古い simple updater のチェックポイントを削除
- [ ] 実験結果が「良すぎる」場合は退化解を疑う（cosine > 0.999, L2 < 0.001）

---

## 📊 CVFPT分析結果の解釈

**繰り返し訓練（Fixed-Point）vs 単一パス（Single-Pass）の違い**

### 実験結果（Gated Updater使用時）

| メトリクス | 値 | 解釈 |
|----------|---|------|
| **L2距離** | 3.69 | ノルム（~16）の23% |
| **コサイン類似度** | 0.973 | 方向が97.3%一致 |
| **角度差** | 13.2° | 方向のずれは小さい |
| **ノルム差** | 0.03% | ほぼ同じ大きさ |

### 分解分析

**L2距離の内訳**:
- **方向成分**: 100%（主要な違い）
- **大きさ成分**: 0%（ほぼ同じ）

### 実用的な意味

1. **方向は高精度** - 単一パスでも97.3%方向が一致
2. **大きさも同等** - ノルムの差はわずか0.03%
3. **主な違い** - 次元ごとの微調整（refinement）

**結論**: 単一パスで「本質（essence）」を捉え、繰り返し訓練で「精緻化（refinement）」が進む

---

## 🧪 実装済み機能 - テスト・訓練ツール

### Early Stopping（2025-11-21実装）

**`src/utils/early_stopping.py`**:

1. **Phase1EarlyStopping**
   - 収束率ベースの停止判定
   - デフォルト閾値: 95%

2. **Phase2EarlyStopping**
   - Validation Lossベースの停止判定
   - ベストモデル自動復元機能

3. **CombinedEarlyStopping**
   - Phase 1とPhase 2の統合クラス

### 固有点キャッシュシステム（2025-11-21実装）

**`src/utils/cache_manager.py`**:

- **FixedContextCache**: 固有点文脈ベクトルの自動保存・読み込み
- トークン列のハッシュベース識別
- アーキテクチャ・設定の検証
- インデックスファイルで管理（`cache/fixed_contexts/index.json`）

**主要機能**:
- `save()`: 固有点を保存
- `load()`: 固有点を読み込み（検証付き）
- `exists()`: キャッシュ存在確認
- `clear()`: キャッシュクリア
- `stats()`: キャッシュ統計表示

### 複数サンプルテストスクリプト（2025-11-21実装）

**`tests/phase2_experiments/test_multi_sample.py`**:
- Train/Validation分割（80/20）
- Early Stopping統合（Phase 1とPhase 2）
- 固有点キャッシュ自動管理
- 10/50/100サンプル対応
- カスタムアーキテクチャ対応

**使用例**:
```bash
# 10サンプルでテスト（Mixed [2,2]）
python3 tests/phase2_experiments/test_multi_sample.py --num-samples 10

# 100サンプル、カスタムアーキテクチャ
python3 tests/phase2_experiments/test_multi_sample.py --num-samples 100 --layer-structure 2 2 2

# キャッシュクリアして実行
python3 tests/phase2_experiments/test_multi_sample.py --num-samples 10 --clear-cache
```

### PPL vs Accuracy - 重要

**PPL（Perplexity）が主要指標**:
- 確率分布全体を評価（全語彙）
- モデルの「自信度」を反映
- 汎化性能の指標

**Accuracy**:
- 最大確率トークンが正解かのみ
- 確率分布の質は無視
- 過学習の検出には不十分

**例**: Layer-wise [1,1,1,1]
- Training: PPL=3.43, Acc=62.4%（暗記）
- Validation: PPL=37.95, Acc=43.8%（過学習）
- PPLの劣化（11倍）が本質的問題

---

## まとめ

**鉄則**:
- ✅ 新しいバージョンを作ったら、古いバージョンは即座に削除
- ✅ ファイル名は完全に固定し、常に上書き
- ❌ 「念のため」残すことは禁止
- ✅ Gitの履歴に残っているので、必要なら復元可能
- ✅ **CVFPT実験は `scripts/train_repetition.py` を使用**
- ✅ **Gated Context Updater が標準（変更禁止）**
- ✅ **`./cache/` は保護（削除禁止）**

**徹底的なクリーンアップとファイル名の統一がプロジェクトの品質を保ちます。**
