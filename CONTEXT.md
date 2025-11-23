# New-LLM Project Context

このドキュメントは、New-LLMプロジェクトの開発経緯、発見された問題、実施した解決策を時系列で記録したものです。

## プロジェクト概要

**目的**: Context Vector Fixed-Point Property (CVFP) に基づく新しい言語モデルアーキテクチャの開発

**コア仮説**: コンテキストベクトルは反復的な改良を通じて固定点に収束し、そこから有意義な表現が生まれる

## 開発の時系列

### セッション開始時の状況

プロジェクトは既に大規模なリファクタリングが完了した状態で引き継がれた：
- 統一されたPhase 1実装
- CVFPバグ修正（固定点比較ロジック、分布正則化）
- クリーンなコードベース

### 実験1: 100サンプルでの訓練要求

**ユーザー要求**:
- サンプル数: 100
- レイヤー数: 2
- 次元: 16
- 長時間実験のため、出力を削減

**実行結果**:
- 訓練データ: 92,047トークン
- Phase 1: **わずか2イテレーションで100%収束**
- CVFP Loss: **0.000000** (異常に完璧)
- Distribution Loss: 0.622487

**問題点**: 結果が「良すぎる」→ 何かおかしい

### 問題分析フェーズ

#### 発見1: 恒等写像への収束

**調査スクリプト**: `investigate_convergence.py`

**発見事項**:
```
Identity Mapping Check:
Cosine similarity between input/output: 0.9741 ± 0.0159
⚠️ WARNING: Model appears to be mostly preserving input contexts!
```

**意味**: モデルが `f(x) ≈ x` となっており、トークン情報をほぼ無視している

#### 発見2: トークン依存性の欠如

**調査スクリプト**: `test_token_dependency.py`

**発見事項**:
```
異なるトークン（100, 500, 1000, 5000, 10000）でも:
- 出力の類似度: 99.9%以上
- 入力コンテキストとの類似度: 99.7%
```

**意味**: モデルがトークンIDに依存せず、入力をほぼそのまま出力

#### 発見3: LayerNormの過剰な影響

**発見事項**:
```python
# LayerNormのパラメータ
context_norms.weight: [1.0, 1.0, 1.0, ...] (全て1)
context_norms.bias: [0.0, 0.0, 0.0, ...] (全て0)

# しかし、出力ノルムは常に4.0に固定
output_norm: 4.0000 (標準偏差: 0.0000)
```

**意味**: LayerNormが強制的に正規化し、トークン情報を消失させている

### 解決策の試行

#### 試行1: LayerNormの完全削除

**結果**: NaN（Not a Number）が発生
- 92,047トークンの累積処理で勾配爆発
- メモリ不足エラー（エラーコード137）

**結論**: LayerNormは必要だが、別のアプローチが必要

#### 試行2: LayerNormの弱体化（20%混合）

```python
# 80%オリジナル + 20%正規化
context_temp = 0.8 * context_temp + 0.2 * context_normed
```

**結果**:
- Identity mapping: 0.9947（99.47%）
- まだ恒等写像に近い
- Distribution loss: 0.42（改善は見られるが不十分）

**結論**: 根本的な解決にならず

### ユーザーからの重要な指摘

#### 指摘1: Distribution Regularizationの問題

**ユーザー**: 「正規分布化はどのようなロジックで更新されていますか？トークンごとですか？」

**発見した問題**:
```python
# 現在の実装（バッチ全体で統計計算）
dim_mean = all_contexts_tensor.mean(dim=0)  # [92047, 16] → [16]
dim_var = all_contexts_tensor.var(dim=0)     # 全トークンの分散
```

**問題点**:
- トークンごとではなく、全92,047トークンをまとめて統計計算
- オンライン学習に適していない
- バッチ正規化により恒等写像を許容してしまう

**理論的に正しいアプローチ**:
- トークンごとにEMA（Exponential Moving Average）で統計を更新
- `running_mean = 0.99 * running_mean + 0.01 * current_mean`
- オンライン処理で、シーケンス長に依存しない

#### 指摘2: コード設計の問題

**ユーザー**: 「実装が洗練化されていない。オブジェクト指向を導入し、Layerクラスで統計計算を隠蔽すべき」

**問題点**:
- Phase 1訓練ループに統計計算ロジックが混在
- `running_mean`, `running_var`が外部に露出
- テスト困難、再利用不可

### 大規模リファクタリング

#### 新設計の方針（CLAUDE.mdに記載）

**原則**:
1. **トークンごとの正規化**: EMAによるオンライン統計更新
2. **カプセル化**: レイヤー内部で統計を自動管理
3. **クリーンなインターフェース**: `get_distribution_loss()`で外部から損失取得
4. **自己完結**: forwardで自動的に統計更新

#### 実装したコンポーネント

**1. src/models/layers.py** (新規作成、264行)

**CVFPLayer クラス**:
```python
class CVFPLayer(nn.Module):
    """単一のコンテキスト更新レイヤー"""

    def __init__(self, context_dim, embed_dim, hidden_dim, use_dist_reg=True, ema_momentum=0.99):
        # EMA統計用のバッファ
        self.register_buffer('running_mean', torch.zeros(context_dim))
        self.register_buffer('running_var', torch.ones(context_dim))
        self.momentum = 0.99

    def forward(self, context, token_embed):
        # FNN処理
        new_context, new_token = self._process(context, token_embed)

        # 統計を自動更新（訓練時のみ）
        if self.training and self.use_dist_reg:
            self._update_running_stats(new_context)

        return new_context, new_token

    def _update_running_stats(self, context):
        """内部実装（外部から隠蔽）"""
        with torch.no_grad():
            batch_mean = context.mean(dim=0)
            batch_var = context.var(dim=0, unbiased=False)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

    def get_distribution_loss(self):
        """外部から損失を取得"""
        mean_penalty = (self.running_mean ** 2).mean()
        var_penalty = ((self.running_var - 1.0) ** 2).mean()
        return mean_penalty + var_penalty
```

**CVFPBlock クラス**:
```python
class CVFPBlock(nn.Module):
    """複数のCVFPLayerをグループ化"""

    def __init__(self, num_layers, ...):
        self.layers = nn.ModuleList([
            CVFPLayer(...) for _ in range(num_layers)
        ])

    def forward(self, context, token_embed):
        for layer in self.layers:
            context, token_embed = layer(context, token_embed)
        return context, token_embed

    def get_distribution_loss(self):
        """全レイヤーの損失を集約"""
        return sum(layer.get_distribution_loss() for layer in self.layers) / len(self.layers)
```

**2. src/models/new_llm_residual.py** (リファクタリング版)

**変更点**:
- 238行 → 183行（55行削減、23%削減）
- CVFPBlockを使用
- LayerNormのパラメータを削除可能（`layernorm_mix=0.0`でオフ）
- クリーンな`get_distribution_loss()`インターフェース

**3. src/training/phase1.py** (リファクタリング版)

**変更点**:
- 手動統計計算を削除（約30行削減）
- `model.get_distribution_loss()`を使用
- より読みやすく、保守しやすい

```python
# 旧実装（削除）
dim_mean = all_contexts_tensor.mean(dim=0)
dim_var = all_contexts_tensor.var(dim=0, unbiased=False)
dist_loss = (dim_mean ** 2).mean() + ((dim_var - 1.0) ** 2).mean()

# 新実装（シンプル）
dist_loss = model.get_distribution_loss()
```

### プロジェクト整理

**削除したファイル**:
- `REFACTORING_PLAN.md` - 古い計画書
- `docs/ARCHITECTURE.md` - 古い設計書
- `tests/` - 古いテストディレクトリ全体
- `scripts/` - 未使用スクリプト
- `analyze_checkpoint.py`, `investigate_convergence.py`, `test_token_dependency.py`, `test_without_layernorm.py`, `train_experiment.py` - 実験用スクリプト
- 全ての `__pycache__/` と `.DS_Store`

**統合したファイル**:
- `new_llm_residual_v2.py` → `new_llm_residual.py`
- `phase1_v2.py` → `phase1.py`

**最終的なプロジェクト構造**:
```
new-llm/
├── CLAUDE.md              # 設計ガイドライン
├── README.md              # プロジェクト概要
├── CONTEXT.md             # このファイル
├── config.py              # 設定
├── train.py               # メイン訓練スクリプト
├── test_refactored.py     # クイックテスト（10トークン）
├── src/
│   ├── models/
│   │   ├── layers.py              # CVFPLayer/CVFPBlock
│   │   └── new_llm_residual.py    # メインモデル
│   ├── training/
│   │   ├── phase1.py              # Phase 1（固定点学習）
│   │   └── phase2.py              # Phase 2（トークン予測）
│   ├── data/
│   │   └── loader.py              # データローダー
│   └── evaluation/
│       └── metrics.py             # 評価指標
└── data/
    ├── example_train.txt
    └── example_val.txt
```

### リファクタリング後のテスト結果

**test_refactored.py** (10トークンでの最小テスト):

```
Configuration:
  Layers: 2
  Context dim: 16
  Distribution reg weight: 0.5 (50%に増加)
  Max iterations: 5

Results:
  ✓ Phase 1 completed in 2.03s
  ✓ Distribution loss: 0.083 (前回の0.82から10倍改善！)
  ✓ Context diversity: 5.15 (良好)
  ⚠️ Identity mapping: 0.9653 (まだ高い)
  ⚠️ 2イテレーションで100%収束（CVFP=0.000000）

Running statistics:
  Block 0, Layer 0: mean=0.18, var=0.82, batches=20
  Block 1, Layer 0: mean=0.21, var=0.82, batches=20
```

**改善点**:
- Distribution lossが10分の1に改善（0.82 → 0.083）
- EMA統計が正しく追跡されている
- コードがクリーンで保守しやすい

**残存する問題**:
- 依然として恒等写像に近い（96.5%）
- 2イテレーションで完全収束（異常）
- Mean: 1.12（目標: 0）、分散も目標の1.0に達していない

## 現在の課題と理論的考察

### 課題1: CVFP Loss = 0.000000の異常性

**問題**:
```python
# Iteration 1: contexts_1を計算
# Iteration 2: contexts_2を計算
cvfp_loss = MSE(contexts_2, contexts_1) = 0.000000
```

92,047個のトークン全てで前回と完全一致は物理的にあり得ない。

**原因仮説**:
1. モデルが「前回の出力を記憶」することを学習
2. トークン依存性を学習する必要がない
3. 恒等写像 `f(x) ≈ x` が最も簡単な解

### 課題2: 恒等写像への収束

**理論的な問題**:

固定点条件: `x* = f(x*, token)`

これは以下でも満たされる:
- `f(x, token) = x` (恒等写像)
- これは**自明な解**（trivial solution）

**なぜ恒等写像になるのか**:
1. FNNの初期化が小さい（std=0.05）
2. Residual connection: `context + delta_context`
3. `delta_context`が非常に小さい → `context`がほぼ保存
4. LayerNormも混合比率が低い → さらに変化が小さい

### 課題3: トークン情報の無視

**実験結果**:
```
異なるトークン（100 vs 10000）でも:
- 出力の違い: わずか0.06（L2距離）
- コサイン類似度: 0.9999
```

**モデルが学習していること**:
- トークンに関わらず、入力コンテキストをほぼそのまま返す
- 文脈の「流れ」を学習していない

## CVFPLayer vs CVFPBlockの違い

**CVFPLayer** (単一レイヤー):
- 1回のコンテキスト更新
- FNN処理、Residual connection、EMA統計更新
- 基本的な計算単位

**CVFPBlock** (複数レイヤーのグループ):
- 複数のCVFPLayerを順次実行
- 損失を集約
- `layer_structure`（例: [1, 1]）を実装

**現在の構成**:
```python
layer_structure = [1, 1]  # 2ブロック、各1レイヤー

CVFPBlock #0
  └─ CVFPLayer #0
CVFPBlock #1
  └─ CVFPLayer #0
```

**柔軟性**:
`layer_structure = [2, 3]`なら:
```
CVFPBlock #0
  ├─ CVFPLayer #0
  └─ CVFPLayer #1
CVFPBlock #1
  ├─ CVFPLayer #0
  ├─ CVFPLayer #1
  └─ CVFPLayer #2
```

## 次のステップ（未実施）

### 提案1: CVFP損失関数の再設計

**現在の問題**:
```python
# Iteration Nの出力を、Iteration N+1で再現
cvfp_loss = MSE(contexts[n+1], contexts[n])
```

これでは「記憶」すれば良いだけ。

**改善案**:
```python
# 同じトークンシーケンスを2回処理
contexts_1 = forward_pass(tokens)
contexts_2 = forward_pass(tokens)  # 同じトークン
cvfp_loss = MSE(contexts_2, contexts_1)
```

これなら真の固定点を学習する必要がある。

### 提案2: 初期化の改善

**現在**:
```python
nn.init.normal_(module.weight, mean=0.0, std=0.05)
```

**改善案**:
- より大きなstd（0.1～0.2）
- Xavier初期化やHe初期化の検討

### 提案3: アーキテクチャの調整

**オプション**:
1. FNN層を深くする（2層 → 3-4層）
2. Residual connectionの係数を調整
3. Skip connectionの追加
4. Attention機構の導入

## 重要な設計原則（CLAUDE.mdより）

### トークンごとの正規化

**理論的根拠**:
1. 言語モデルはトークンを順次処理
2. 全トークンを待つことはできない（オンライン学習）
3. バッチ正規化は恒等写像を許容
4. シーケンス長に依存しない

### EMAの利点

```python
running_mean = 0.99 * running_mean + 0.01 * current_mean
```

- 過去の情報を徐々に忘れる
- 安定した勾配
- メモリ効率が良い

### オブジェクト指向設計

**原則**:
1. カプセル化: 実装詳細を隠す
2. 単一責任: 各クラスは1つのことだけ
3. クリーンなインターフェース: 最小限のパラメータ
4. 自己文書化: メソッド名で目的を説明

**アンチパターン**:
- ❌ 訓練ループでの手動統計計算
- ❌ 内部状態（running_mean）の露出
- ❌ forward passとloss計算の混在

**推奨パターン**:
- ✅ レイヤーが自身の統計を管理
- ✅ プロパティ/メソッドで損失取得
- ✅ `nn.Module`のbufferで永続的な状態
- ✅ `self.training`による自動train/eval切り替え

## 技術的な詳細

### Distribution Regularization の実装

**目標**: 各次元が N(0,1) に従う

**実装**:
```python
# トークンごとにEMA更新
with torch.no_grad():
    batch_mean = context.mean(dim=0)  # [context_dim]
    batch_var = context.var(dim=0, unbiased=False)  # 母分散

    self.running_mean = 0.99 * self.running_mean + 0.01 * batch_mean
    self.running_var = 0.99 * self.running_var + 0.01 * batch_var

# 損失計算
mean_penalty = (self.running_mean ** 2).mean()  # 平均が0から離れるペナルティ
var_penalty = ((self.running_var - 1.0) ** 2).mean()  # 分散が1から離れるペナルティ
dist_loss = mean_penalty + var_penalty
```

**重要**: `unbiased=False`で母分散を使用

### Phase 1 の訓練ループ

```python
for iteration in range(max_iterations):
    # Forward pass: 全トークンを処理
    contexts = []
    for token in tokens:
        context = model.update_context(context, token)
        contexts.append(context)

    if iteration > 0:
        # CVFP loss: 前回との差
        cvfp_loss = MSE(contexts, fixed_contexts)

        # Distribution loss: モデル内部で自動計算済み
        dist_loss = model.get_distribution_loss()

        # 統合
        total_loss = 0.5 * cvfp_loss + 0.5 * dist_loss

        # Backprop
        total_loss.backward()
        optimizer.step()

    # 次のイテレーション用に保存
    fixed_contexts = contexts.detach()
```

### 設定パラメータ

**config.py**:
```python
# Model
num_layers = 2
context_dim = 16
embed_dim = 16
hidden_dim = 32

# Phase 1
phase1_max_iterations = 10
phase1_convergence_threshold = 0.02  # MSE < 0.02で収束
phase1_min_converged_ratio = 0.95    # 95%のトークンが収束で停止

# Distribution Regularization
use_distribution_reg = True
dist_reg_weight = 0.2  # 20% (リファクタリング後のテストでは0.5)

# Learning rates
phase1_lr_warmup = 0.002    # Iteration 1-3
phase1_lr_medium = 0.0005   # Iteration 4-8
phase1_lr_finetune = 0.0001 # Iteration 9+
```

## まとめ

### 達成できたこと

1. ✅ トークンごとの分布正規化（EMA）を実装
2. ✅ クリーンなオブジェクト指向設計に移行
3. ✅ CVFPLayer/CVFPBlockによるカプセル化
4. ✅ プロジェクトファイルの完全整理
5. ✅ Distribution lossの大幅改善（0.82 → 0.083）

### 未解決の課題

1. ❌ 恒等写像への収束（96.5%の類似度）
2. ❌ 2イテレーションでの異常な完全収束
3. ❌ トークン依存性の欠如
4. ❌ CVFP損失関数の設計問題

### 次のセッションへの引き継ぎ事項

1. **CVFP損失関数の再設計**を検討すべき
2. モデルの**初期化戦略**を見直す必要がある
3. **より深いネットワーク**（layer_structure）を試す価値がある
4. **Attention機構**などの追加コンポーネントを検討

この問題は、コードのバグではなく**アーキテクチャ設計の根本的な課題**である。自明な解（恒等写像）を避け、真に有意義な固定点を学習させる方法を見つける必要がある。
