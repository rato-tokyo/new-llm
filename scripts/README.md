# Scripts Directory

実験・ユーティリティスクリプト。

## 利用可能なスクリプト

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

### sample_scaling_experiment.py
サンプル数スケーリング実験。

```bash
python3 scripts/sample_scaling_experiment.py
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
