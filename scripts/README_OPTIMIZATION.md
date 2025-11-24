# Hyperparameter Optimization Scripts

このディレクトリには、モデルのハイパーパラメータを自動最適化するスクリプトが含まれています。

## optimize_dist_reg_weight.py

**目的**: Optunaを使用して`dist_reg_weight`（多様性正則化の重み）を自動調整

**評価方法**:
1. Phase 1（固定点学習）を実行
2. Phase 2（トークン予測）を実行
3. Validation Perplexity（PPL）で評価
4. Phase 2に到達できない場合は、不適格パラメータとして扱う

**不適格判定条件**:
- 恒等写像検出（Identity Mapping）
- 低いEffective Rank（< 50%）
- Phase 1またはPhase 2でエラー発生

## 使い方

### 基本的な使用方法

```bash
# 20回のトライアルで最適化（デフォルト）
python3 scripts/optimize_dist_reg_weight.py

# トライアル数を指定
python3 scripts/optimize_dist_reg_weight.py --n-trials 50

# タイムアウトを設定（1時間 = 3600秒）
python3 scripts/optimize_dist_reg_weight.py --n-trials 30 --timeout 3600

# より長時間の最適化（3時間、50トライアル）
python3 scripts/optimize_dist_reg_weight.py --n-trials 50 --timeout 10800
```

### オプション

- `--n-trials`: 実行するトライアル数（デフォルト: 20）
- `--timeout`: タイムアウト時間（秒）（デフォルト: なし）
- `--study-name`: Optunaスタディ名（デフォルト: dist_reg_weight_optimization）
- `--storage`: データベースURL（デフォルト: なし = メモリ内）

### 結果の使用

最適化が完了すると、`config_optimized.py`が生成されます。

```python
# 最適化されたconfigを使用
from config_optimized import OptimizedConfig

config = OptimizedConfig()
# config.dist_reg_weight は最適値に設定されています
```

## 最適化のしくみ

### Optuna TPESampler

**Tree-structured Parzen Estimator (TPE)**:
- 過去のトライアル結果から学習
- 有望なパラメータ空間を重点的に探索
- ランダムサーチより効率的

### Pruning（枝刈り）

**MedianPruner**:
- 見込みのないトライアルを早期に打ち切り
- 計算時間を節約
- 有望なパラメータに集中

**Pruning条件**:
- 恒等写像が検出された場合
- Effective Rankが50%未満の場合
- Phase 1/Phase 2でエラーが発生した場合

## 推定時間

**1トライアルあたりの時間**（CPU、5サンプル、128トークン）:
- Phase 1: 約5-10秒
- Phase 2: 約10-20秒（5エポック）
- **合計**: 約15-30秒/トライアル

**20トライアルの推定時間**: 5-10分
**50トライアルの推定時間**: 12-25分

※ データ量が増えると時間も増加します

## 最適化の進行状況

最適化中は以下の情報が表示されます：

```
======================================================================
Trial 5: dist_reg_weight = 0.823
======================================================================

PHASE 1: Fixed-Point Learning
...

PHASE 2: Token Prediction
...

======================================================================
Trial 5 Complete
  dist_reg_weight: 0.823
  Train Effective Rank: 650.2/768 (84.7%)
  Val Perplexity: 125.34
  Trial Time: 22.3s
======================================================================

[推定完了時間]: あと 15分 (残り15トライアル)
```

## 最適化結果の例

```
======================================================================
OPTIMIZATION COMPLETE
======================================================================

Total optimization time: 18.5 minutes
Completed trials: 20
Pruned trials: 5
Failed trials: 0

Best trial:
  Trial number: 12
  Validation Perplexity: 98.45
  Best dist_reg_weight: 0.867

Top 5 trials:
  1. Trial 12: dist_reg_weight=0.867, perplexity=98.45
  2. Trial 18: dist_reg_weight=0.891, perplexity=101.23
  3. Trial 7: dist_reg_weight=0.842, perplexity=105.67
  4. Trial 15: dist_reg_weight=0.778, perplexity=110.34
  5. Trial 3: dist_reg_weight=0.923, perplexity=115.89
```

## トラブルシューティング

### メモリ不足エラー

**対策**: データ量を減らす

```python
# config.py で調整
num_samples = 3  # 5 → 3に削減
max_seq_length = 64  # 128 → 64に削減
```

### 時間がかかりすぎる

**対策**: Phase 2のエポック数を削減（スクリプト内で自動的に5エポックに制限されています）

### すべてのトライアルが失敗する

**原因**: データが不適切、またはモデル設定に問題

**対策**:
1. `python3 train.py --test`で基本的な動作を確認
2. `config.py`の設定を見直す
3. データの品質を確認

## 注意事項

1. **計算リソース**: 長時間の最適化はGPUの使用を推奨
2. **データサイズ**: 最適化中は小規模データ（5-10サンプル）を使用
3. **再現性**: `random_seed`を固定すると結果が再現可能
4. **中断と再開**: `--storage`オプションでデータベースを指定すれば、中断後に再開可能

## 高度な使用方法

### データベースで永続化（中断・再開可能）

```bash
# SQLiteデータベースを使用
python3 scripts/optimize_dist_reg_weight.py \
    --n-trials 100 \
    --storage sqlite:///optuna_studies.db \
    --study-name my_optimization

# 中断後、同じコマンドで再開
python3 scripts/optimize_dist_reg_weight.py \
    --n-trials 100 \
    --storage sqlite:///optuna_studies.db \
    --study-name my_optimization
```

### 複数パラメータの同時最適化

将来的には以下のパラメータも最適化対象に追加可能：
- `phase1_learning_rate`
- `ema_momentum`
- `phase1_convergence_threshold`

## 参考リンク

- [Optuna公式ドキュメント](https://optuna.readthedocs.io/)
- [TPESamplerの説明](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Prunerの説明](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
