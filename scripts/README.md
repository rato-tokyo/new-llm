# Scripts Directory

実験・ユーティリティスクリプト。

## 多様性アルゴリズム実験（2025-12-01追加）

### diversity_algorithm_experiment.py
Phase 1のみで多様性アルゴリズムを比較。Effective Rank (ER) を測定。

```bash
# 全4アルゴリズムを比較
python3 scripts/diversity_algorithm_experiment.py -a MCDL ODCM SDL NUC -s 50 100

# 特定のアルゴリズムのみ
python3 scripts/diversity_algorithm_experiment.py -a MCDL ODCM -s 50

# context_dim指定
python3 scripts/diversity_algorithm_experiment.py -a MCDL -s 50 --context-dim 1000
```

### diversity_full_experiment.py
Phase 1 + Phase 2を実行し、α値（スケーリング指数）を比較。

```bash
# デフォルト設定（4アルゴリズム, samples=[50,100,200], context_dim=1000）
python3 scripts/diversity_full_experiment.py
```

**出力**:
- 各アルゴリズムの Effective Rank
- Val PPL, Val Acc
- α値（PPL = A × tokens^α）
- R²値

## スケーリング実験

### scaling_experiment.py
スケーリング則の実験。α値の推移分析機能あり。

```bash
# 標準実験
python3 scripts/scaling_experiment.py --input-tokens 1 --layers 1 --context-dim 768

# α値推移分析
python3 scripts/scaling_experiment.py --alpha-scaling \
  --init-samples 50 --multiplier 2 --window-size 4 --num-windows 2
```

## ユーティリティ

### check_val_convergence.py
学習済みモデルで検証データの収束性をチェック。

```bash
python3 scripts/check_val_convergence.py --num_trials 10
python3 scripts/check_val_convergence.py --checkpoint_path ./checkpoints/my_model.pt
```

### check_token_overlap.py
訓練データと検証データのトークン重複率を分析。

```bash
python3 scripts/check_token_overlap.py
```

### create_val_from_train.py
訓練データから検証データを生成。

```bash
python3 scripts/create_val_from_train.py
```

### prepare_disk_offload.py
大規模データ用のディスクオフロード準備。

```bash
python3 scripts/prepare_disk_offload.py --output_dir /path/to/nvme --num_samples 200000
```

### train_full_ultrachat.py
UltraChat全データでの訓練。

```bash
python3 scripts/train_full_ultrachat.py
```
