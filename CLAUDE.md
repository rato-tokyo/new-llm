# Claude Code Development Guidelines for New-LLM Project

## 🎯 Colab実験の1行コマンド完結ポリシー - CRITICAL

**⚠️ 絶対ルール：すべての訓練は1行curlコマンドで開始できること**

### 基本原則

**すべての訓練データセットは、Google Colabで1行のcurlコマンドをコピペするだけで実験が開始できること。**

```bash
# ✅ これが正しい形
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_<dataset>.sh | bash
```

**この方針は例外なく徹底する。**

### 必須実装要件

新しい訓練データセットを追加する際は、**必ず以下の4つを実装**すること：

1. ✅ `scripts/train_<dataset>.py` - Python訓練スクリプト
2. ✅ **`scripts/colab_train_<dataset>.sh`** - 1行実行用bashスクリプト（**最重要・必須**）
3. ✅ `tests/test_<dataset>_training.py` - テストスクリプト
4. ✅ `<DATASET>_TRAINING.md` - ドキュメント（1行コマンドを最初に記載）

**`scripts/colab_train_<dataset>.sh`がない実装は不完全とみなす。**

### 禁止事項

**❌ 絶対にやってはいけないこと**:

```bash
# ❌ 複数ステップが必要（禁止）
!git clone https://...
%cd new-llm
!pip install datasets
!python scripts/train_xxx.py
```

**理由**:
- ユーザーがコピペミスをする
- 実行順序を間違える
- 依存関係のインストール忘れ
- 設定ミス
- エラーが起きやすい

### colab_train_*.sh スクリプトの必須内容

```bash
#!/bin/bash
set -e

# 1. パラメータ解析（必須）
NUM_LAYERS=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_layers) NUM_LAYERS="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# 2. 最新版取得（必須）
cd /content
rm -rf new-llm
git clone https://github.com/rato-tokyo/new-llm
cd new-llm

# 3. 依存関係インストール（必須）
pip install -q datasets tqdm

# 4. バックグラウンド実行（必須）
nohup python scripts/train_<dataset>.py --num_layers $NUM_LAYERS > /content/log.txt 2>&1 &

# 5. 初期ログ表示（必須）
sleep 10
tail -30 /content/log.txt

# 6. モニタリングコマンド表示（必須）
echo "📋 Monitoring: !tail -20 /content/log.txt"
echo "🛑 Stop: !pkill -9 -f train_<dataset>"
```

### 実装チェックリスト

新しいデータセット実装時に必ず確認：

- [ ] `scripts/colab_train_<dataset>.sh` を作成したか？
- [ ] 1行curlコマンドで実行できることをテストしたか？
- [ ] パラメータ（`--num_layers`, `--max_samples`など）に対応しているか？
- [ ] GitHubにpush後、curlコマンドでアクセスできることを確認したか？
- [ ] ドキュメントに1行コマンドを**最初に**記載したか？
- [ ] 構文チェック（`bash -n scripts/colab_train_<dataset>.sh`）を実行したか？

### 既存スクリプトの扱い

**既存の訓練スクリプトも全て1行コマンド化すること**:

- Dolly-15k → `scripts/colab_train_dolly.sh` 作成
- HH-RLHF → `scripts/colab_train_hh_rlhf.sh` 作成
- WikiText → `scripts/colab_train_wikitext.sh` 作成
- UltraChat → ✅ 実装済み（`scripts/colab_train_ultrachat.sh`）

**全てのデータセットで統一すること。**

### Claude AIの対応

**新しい訓練データセット実装を依頼された場合**:

1. `scripts/train_<dataset>.py` を作成
2. **`scripts/colab_train_<dataset>.sh` を必ず作成**（忘れない）
3. テストスクリプト作成
4. ドキュメント作成（1行コマンドを最初に記載）
5. GitHubにpush
6. ユーザーに1行コマンドを提示

**`colab_train_*.sh`の作成を忘れたら、即座に追加すること。**

### 利点

この方針により：

- ✅ **ユーザビリティ向上**: 1回のコピペで完了
- ✅ **エラー防止**: 複数ステップでのミスを排除
- ✅ **一貫性**: すべてのデータセットで同じ体験
- ✅ **メンテナンス性**: スクリプトで一元管理
- ✅ **最新版保証**: 毎回`git clone`で最新版を取得

### 例

**UltraChat訓練**（標準実装）:

```bash
# Layer 1、フルデータセット
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash

# Layer 4、サブセット10万件
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash -s -- --num_layers 4 --max_samples 100000
```

**これが全てのデータセットで実現されるべき形。**

---

## 🧪 コード修正時の必須テストポリシー - CRITICAL

**すべてのコード修正後、必ず手元でテストしてからコミット・デプロイすること**

### 必須実行手順

#### 1. ローカルでのテスト（最重要）

**テストは最小限のデータ・epoch・stepで実施する**

```bash
# ステップ1: 構文チェック
python3 -m py_compile train.py chat.py

# ステップ2: 最小限のデータでエンドツーエンドテスト
python3 train.py \
    --dataset ultrachat \
    --max-samples 100 \      # 最小限のデータ（100サンプル）
    --epochs 2 \             # 最小限のepoch（2エポック）
    --output-dir test_run \
    --no-cuda \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --logging-steps 5

# ステップ3: 動作確認
# - エポックごとのメトリクス表示を確認
# - エラーなく完了することを確認
# - 出力ファイルが正しく生成されることを確認
```

**テスト時間の目安**:
- 100サンプル、2エポック → **4-5分**
- 1000サンプル、5エポック → **30-40分** ← 通常は不要、最小限で十分

**鉄則**: テストするときは**epoch=2、samples=100**で十分

#### 2. チェックリスト

コミット前に必ず確認：

- [ ] **構文チェック完了** - `python3 -m py_compile`
- [ ] **インポートエラーなし** - `python3 -c "import module"`
- [ ] **テストスクリプトで動作確認**
- [ ] **エラーメッセージを実際に確認**
- [ ] **ユーザーが実行する環境を考慮**（Colab = PyTorch 2.6+）

#### 3. 絶対に避けるべき間違い

**❌ やってはいけないこと**:
```
1. コード修正 → すぐコミット → ユーザーがエラー報告 → また修正
2. 「たぶん動くだろう」で推測
3. 手元でテストせずに「修正しました」
4. エラーメッセージを見ずに推測で修正
```

**✅ 正しい手順**:
```
1. コード修正
2. ローカルで構文チェック
3. テストスクリプト作成・実行
4. 動作確認完了
5. コミット・プッシュ
6. ユーザーに実行依頼
```

#### 4. テストスクリプトの例

```python
#!/usr/bin/env python3
"""Test script for checkpoint loading"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the modules you're testing
from src.training.trainer import Trainer

def test_checkpoint_load():
    """Test the specific functionality"""
    try:
        # Test code here
        print("✓ Test passed")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_checkpoint_load()
    sys.exit(0 if success else 1)
```

#### 5. PyTorchバージョン差異への対応

**Colabの環境を考慮**:
- Colab = PyTorch 2.6+
- ローカル環境と異なる可能性

**対策**:
```bash
# PyTorchバージョン確認
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 必要に応じてバージョン固有の対処を追加
```

### このポリシーの重要性

**過去の失敗例**:
1. チェックポイントパス重複 → ローカルテストなし → エラー
2. PyTorch 2.6の`weights_only`変更 → ローカルテストなし → エラー
3. 同じ修正を2-3回繰り返す → ユーザーの時間を無駄にする

**影響**:
- ユーザーの実験時間が無駄になる
- Colab GPUクレジットの浪費
- 信頼性の低下

**鉄則**: **手元でテストしてから、ユーザーに渡す**

---

## 実験管理ポリシー - CRITICAL

### 🛑 実験終了時の必須チェックリスト

**実験を停止・終了する際は、必ず以下を確認すること**

#### 1. プロセスの完全停止確認

```bash
# ステップ1: 訓練プロセスの停止
pkill -f "python.*train"

# ステップ2: プロセスが停止したか確認
ps aux | grep "python.*train" | grep -v grep

# ステップ3: CPU使用率の確認
top -l 1 | head -n 5
```

**確認項目**:
- [ ] `ps aux`でPythonの訓練プロセスが存在しないこと
- [ ] CPU idle（空き）が **50%以上** であること
- [ ] Load Averageが低下していること

#### 2. 実験終了時の必須手順

```bash
# 1. すべての訓練プロセスを停止
pkill -9 -f "python.*train"

# 2. 3秒待機（プロセス終了を確実に）
sleep 3

# 3. CPU使用率を確認
top -l 1 | head -n 5

# 4. 結果が正常であることを確認
# - CPU idle > 50%
# - Python訓練プロセス不在
```

#### 3. CPU使用率の判定基準

| CPU idle | 状態 | 対処 |
|----------|------|------|
| **> 70%** | ✅ 正常 | 問題なし |
| **50-70%** | ⚠️ 注意 | 他のプロセス確認 |
| **< 50%** | ❌ 異常 | **訓練プロセスが残存** |

**異常時の対処**:
```bash
# より強力な停止
pkill -9 -f "python.*train"

# 個別プロセスのkill
kill -9 <PID>
```

#### 4. バックグラウンドシェルの確認

Claude Codeのバックグラウンドシェルも確認：

```bash
# /bashes コマンドで確認
# 不要なシェルは手動で停止
```

### ⚠️ 実験停止を怠った場合の問題

1. **CPU資源の無駄** - 不要な訓練が継続
2. **メモリ圧迫** - 他の作業に影響
3. **結果の不整合** - 意図しない訓練が進行
4. **マシンの過負荷** - システム全体のパフォーマンス低下

### ✅ 実験終了の推奨ワークフロー

```bash
# 1. 訓練停止
pkill -9 -f "python.*train"

# 2. 待機
sleep 3

# 3. 確認（これが最重要！）
top -l 1 | head -n 5

# 4. 結果の保存確認
ls -lh checkpoints/best_*.pt

# 5. 実験サマリーの作成
# experiments/ディレクトリに結果をまとめる
```

**この手順を必ず守ること！**

---

## Google Colab 実験管理ポリシー - CRITICAL

### 🌐 Colab実験の推奨ワークフロー

**基本原則**: リポジトリが既に存在する場合は`git pull origin main`で更新

#### Git更新の推奨方法

**✅ 推奨方法**: リポジトリの有無で自動判定

```bash
# リポジトリが既に存在する場合は git pull で更新
if [ -d "/content/new-llm/.git" ]; then
    cd /content/new-llm
    git fetch origin
    git reset --hard origin/main  # ローカル変更を破棄して最新版に同期
    git pull origin main
else
    # リポジトリが存在しない場合のみ git clone
    cd /content
    git clone https://github.com/rato-tokyo/new-llm
    cd new-llm
fi
```

**理由**:
- ✅ **効率的**: 既存リポジトリがある場合、差分のみダウンロード
- ✅ **確実**: `git reset --hard`でローカル変更を破棄し、クリーンな状態を保証
- ✅ **柔軟**: 初回は`clone`、2回目以降は`pull`を自動選択

**旧方式（非推奨）**: 毎回`rm -rf` → `git clone`

```bash
# ❌ 非効率 - 毎回全ファイルをダウンロード
rm -rf new-llm
git clone https://github.com/rato-tokyo/new-llm
cd new-llm
```

**問題点**:
- ⚠️ 時間がかかる（全ファイルを再ダウンロード）
- ⚠️ 帯域幅の無駄遣い

#### Colabの特性

**Colabはステートレス環境**:
- セッション終了で全ファイルが消失
- 同一セッション内では既存リポジトリを再利用可能
- `git reset --hard`で常にクリーンな状態を保証

→ **条件付きで`git pull`と`git clone`を使い分けるのが最適解**

### 📦 実験スクリプトの設計原則 - CRITICAL

**⚠️ このセクションは冒頭の「Colab実験の1行コマンド完結ポリシー」に統合されました。**

**詳細は冒頭のセクションを参照してください。**

#### 要約

- ✅ すべての訓練は**1行curlコマンド**で開始できること
- ✅ `scripts/colab_train_<dataset>.sh` を必ず作成すること
- ✅ 複数ステップのコマンドは禁止
- ✅ 既存スクリプトも全て1行コマンド化すること

**例**:
```bash
!curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash
```

**詳細な実装ガイドラインは冒頭のセクションを参照。**

### 🔄 実験スクリプトのテンプレート

**`scripts/colab_train_<dataset>.sh`の標準テンプレート**:

```bash
#!/bin/bash
set -e

# Parse arguments (optional)
NUM_LAYERS=1
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_layers) NUM_LAYERS="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 1. 最新版を取得（git pullではなくclone）
cd /content
rm -rf new-llm
git clone https://github.com/rato-tokyo/new-llm
cd new-llm

# 2. 依存関係インストール
pip install -q datasets tqdm

# 3. 訓練コマンド構築
CMD="python scripts/train_<dataset>.py --num_layers $NUM_LAYERS"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi
LOG_FILE="/content/<dataset>_layer${NUM_LAYERS}.log"

# 4. 実験開始（バックグラウンド）
nohup $CMD > $LOG_FILE 2>&1 &

# 5. 初期状態表示
sleep 10
tail -30 $LOG_FILE

# 6. モニタリングコマンド表示
echo "========================================="
echo "✅ Training Started!"
echo "========================================="
echo "📋 Monitoring: !tail -20 $LOG_FILE"
echo "🛑 Stop: !pkill -9 -f train_<dataset>"
echo "========================================="
```

### 📚 実装例

**UltraChat訓練**（実装済み）:
- ファイル: `scripts/colab_train_ultrachat.sh`
- 1行コマンド:
  ```bash
  !curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash
  ```
- パラメータ付き:
  ```bash
  !curl -s https://raw.githubusercontent.com/rato-tokyo/new-llm/main/scripts/colab_train_ultrachat.sh | bash -s -- --num_layers 4 --max_samples 100000
  ```

### ⚡ Colab実験の効率化

#### 並列実験の実行

**GPU余裕がある場合は複数実験を同時実行**:

```bash
# 実験1
nohup python3 experiment1.py > exp1.log 2>&1 &

# 実験2（GPU余裕がある）
nohup python3 experiment2.py > exp2.log 2>&1 &

# GPU使用状況確認
nvidia-smi
```

#### 90分制限への対処

**Colab無料版は90分で切断**:
- 各実験は30-40分以内に完了するよう設計
- 長時間実験はチェックポイント保存機能必須
- Google Driveマウントでチェックポイント保存

### 📋 Colab実験チェックリスト

新しい訓練データセット実装時の確認事項:

#### 必須実装

- [ ] **`scripts/train_<dataset>.py`** - Python訓練スクリプト
- [ ] **`scripts/colab_train_<dataset>.sh`** - 1行実行用bashスクリプト（**必須！**）
- [ ] **テストスクリプト** - `tests/test_<dataset>_training.py`
- [ ] **ドキュメント** - `<DATASET>_TRAINING.md`（Colab実行ガイド）

#### スクリプトの要件

- [ ] `rm -rf` + `git clone`で最新版取得
- [ ] **curlで1行実行可能**（最重要）
- [ ] インデントなし（コピペエラー防止）
- [ ] バックグラウンド実行（`nohup` + `&`）
- [ ] ログファイル出力（進捗確認用）
- [ ] 進捗バー実装（tqdm）
- [ ] モニタリングコマンド表示
- [ ] パラメータ対応（`--num_layers`, `--max_samples`など）

#### 品質チェック

- [ ] 構文チェック: `bash -n scripts/colab_train_<dataset>.sh`
- [ ] ローカルテスト: スクリプトが正常に動作するか
- [ ] GitHubにpush後、curlコマンドでアクセス可能か確認

### 🎯 ベストプラクティスまとめ

**鉄則**:
1. ✅ **`git clone`を使う**（`git pull`は使わない）
2. ✅ **1行curlコマンドで実行できる**スクリプトを提供（**最重要**）
3. ✅ **`scripts/colab_train_*.sh`を必ず作成**する
4. ✅ **インデント不要**な設計（コピペ対応）
5. ✅ **バックグラウンド実行**で複数実験並列化
6. ✅ **ログファイル**で進捗確認可能に
7. ✅ **進捗バー（tqdm）**でユーザーに進捗を表示

**この方針により、Colab実験が確実・効率的になります。**

### 🚀 GPU最適化設定の確認 - CRITICAL

**スクリプト作成・修正時は、必ずGPU用の設定になっているか確認すること**

#### ❌ 発生した問題（2025-11-18）

**問題**: `train_wikitext_fp16.py`が`batch_size=32`（CPU用）のまま
**影響**: L4 GPUの性能を全く活かせず、期待の1/16の速度
**原因**: CPU用設定がそのまま残っていた

#### ✅ GPU用設定の必須確認事項

**新しいスクリプト作成時・修正時に必ず確認**:

```python
# ❌ 悪い例 - CPU用設定
batch_size = 32  # CPUでも動くように32

# ✅ 良い例 - GPU用設定
batch_size = 512  # GPU最適化（L4/A100用）
```

**チェックリスト**:
- [ ] `batch_size`がGPU用（512以上）になっているか？
- [ ] コメントが「CPU」ではなく「GPU」になっているか？
- [ ] T4: 512、L4: 512-1024、A100: 1024-2048が目安

#### GPU別の推奨batch_size

| GPU | VRAM | 推奨batch_size | 実測（Baseline） | 備考 |
|-----|------|---------------|----------------|------|
| **T4** | 16GB | 512 | ~8GB使用 | Baseline model |
| **L4** | 24GB | **2048** | **512→5.5GB、2048→22GB** | **4x T4** |
| **A100** | 40GB | **4096** | 推定35-38GB | **8x T4** |

**重要**: この表は実測値に基づいています（2025-11-18, L4 GPU実測）

**計算式**:
```
batch_size = 512 * (GPU_VRAM / 16GB) * utilization_factor
- T4 (16GB): 512 * 1.0 = 512
- L4 (24GB): 512 * 4.0 = 2048 (実測で検証済み)
- A100 (40GB): 512 * 8.0 = 4096
```

#### スクリプトレビュー時の確認コマンド

```bash
# 全訓練スクリプトのbatch_size確認
grep -n "batch_size" scripts/train_*.py

# CPU用設定が残っていないか確認
grep -n "CPUでも" scripts/train_*.py
```

**この確認を怠ると、GPU性能を全く活かせません！**

#### 🎓 Learning Rate Scaling Rule - CRITICAL

**batch_sizeを変更したら、learning_rateも適切にスケールすること**

##### ❌ 発生した問題（2025-11-18）

**問題1**: batch_sizeを512→2048に4倍したが、learning_rateを調整しなかった
**影響**: Epoch 29でPPL 46.7（以前の実験はEpoch 27でPPL 23.34）

**問題2（発見）**: 実際のCPU Baselineはbatch=**32**だったのに、batch=512を基準にしていた
**影響**: batch 32→2048（64倍）なのに、LR 0.0001→0.0004（4倍のみ）← **16倍不足！**

##### ✅ 正しいScaling Rule

**小規模batch（< 256）**: Linear Scaling
```
batch_sizeをk倍 → learning_rateをk倍
```

**大規模batch（>= 256）**: Square Root Scaling（推奨）
```
batch_sizeをk倍 → learning_rateを√k倍
```

**理由**:
- 大規模batchではLinear Scalingが不安定になる
- Square Root Scalingがより安定（Hoffer et al., 2017）
- batch > 256ではこちらが推奨

**正しい計算（CPU baseline基準）**:
```python
# CPU Baseline（実際の過去実験）
batch_size = 32
learning_rate = 0.0001

# L4 GPU（batch 64倍 = 32→2048）
batch_size = 2048
# Linear Scaling: 0.0001 * 64 = 0.0064（大きすぎて不安定）
# Square Root Scaling: 0.0001 * √64 = 0.0001 * 8 = 0.0008 ✓
learning_rate = 0.0008  # √64 = 8倍

# A100 GPU（batch 128倍 = 32→4096）
batch_size = 4096
# Square Root Scaling: 0.0001 * √128 ≈ 0.0001 * 11.3 = 0.0011
learning_rate = 0.0011  # √128 ≈ 11.3倍
```

##### 実測結果（2025-11-18） - 更新

| 設定 | batch_size | learning_rate | 倍率 | Epoch 50のPPL | 評価 |
|------|-----------|--------------|------|------------|------|
| **CPU Baseline** | **32** | **0.0001** | 基準 | **23.34** (Ep 27) ✓ | 良好 |
| **L4（誤）** | 2048 | 0.0004 | 64倍 batch, 4倍 LR | **30.78** ✗ | 16倍不足 |
| **L4（正）** | 2048 | **0.0008** | 64倍 batch, 8倍 LR (√64) | 期待: **20-23** | 未実験 |

##### チェックリスト

batch_size変更時:
- [ ] **CPU baseline (batch=32)** を基準にしているか？
- [ ] batch < 256: Linear Scaling（k倍）
- [ ] batch >= 256: **Square Root Scaling（√k倍）** ← 推奨
- [ ] 最初のエポックでPPLが急速に減少しているか？

**この原則を怠ると、学習が著しく遅くなります！**

#### 📐 Model Size Scaling Rule - CRITICAL

**モデルサイズを大きくしたら、learning_rateを小さくすること**

##### ❌ 発生した問題（2025-11-18）

**問題**: Advancedモデル（4.84M）でBaselineモデル（2.74M）と同じlearning_rate（0.0004）を使用
**影響**: PPL 40.93（Baseline 30.78より33%悪化）
**原因**: 大きなモデルには小さめのlearning_rateが必要だった

##### ✅ Model Size Scaling Rule

**基本原則**:
```
モデルサイズが大きい → learning_rateを小さくする
```

**理由**:
- 大きなモデル → パラメータ数が多い → より繊細な調整が必要
- 同じlearning_rateだと、大きなモデルほど不安定になりやすい
- **Linear Scaling RuleはBATCH SIZE用であり、MODEL SIZE用ではない**

**推奨スケーリング**:
```python
# Baseline (2.74M params)
learning_rate = 0.0004

# Advanced (4.84M params, 1.77x larger)
learning_rate = 0.0002  # 半分に減らす（保守的）
# または
learning_rate = 0.0001  # 1/4に減らす（より安全）
```

**一般的な経験則**:
```
LR_new = LR_base / sqrt(params_new / params_base)

例:
- Baseline: 2.74M, LR=0.0004
- Advanced: 4.84M (1.77x)
- LR_new = 0.0004 / sqrt(1.77) = 0.0004 / 1.33 ≈ 0.0003
```

##### 実測結果（2025-11-18）

| モデル | Params | learning_rate | PPL @ Epoch 50 | 評価 |
|--------|--------|--------------|---------------|------|
| **Baseline** | 2.74M | 0.0004 | **30.78** ✓ | 良好 |
| **Advanced** | 4.84M | 0.0004 | **40.93** ✗ | 悪化 |
| **Advanced（推奨）** | 4.84M | 0.0002 | 期待: 25以下 | 未実験 |

##### Large Model Paradox

**教訓**: より大きなモデルが必ずしも良い性能を出すとは限らない

**2025-11-18の実験で発見**:
- Advancedモデル（4.84M）はBaselineモデル（2.74M）より**33%性能が悪かった**
- 原因: learning_rateの調整不足
- 解決策: learning_rateを半減させる

##### チェックリスト

モデルサイズ変更時:
- [ ] learning_rateを小さくしたか？
- [ ] パラメータ数が2倍なら、LRを1/√2 ≈ 0.7倍に
- [ ] パラメータ数が4倍なら、LRを1/2に
- [ ] 最初の数エポックでPPLが順調に減少しているか？
- [ ] 大きなモデルが小さなモデルより性能が良いか？（そうでなければLR調整不足）

**この原則を怠ると、大きなモデルが逆に性能が悪くなります！**

### 📦 設定の一元管理ポリシー - CRITICAL

**すべての設定値は `src/utils/config.py` で一元管理すること**

#### 🎯 Single Source of Truth 原則

**基本原則**: 設定値をスクリプト内にハードコードしない

```python
# ❌ 悪い例 - スクリプトで独自に設定を定義
class MyConfig:
    batch_size = 512
    learning_rate = 0.0001
    # ... 他の設定

# ✅ 良い例 - config.pyから継承
from src.utils.config import NewLLMGPUConfig

class MyConfig(NewLLMGPUConfig):
    # GPU最適化設定を自動継承
    # 必要な部分のみ上書き
    num_epochs = 100
```

#### 利用可能な設定クラス

| クラス名 | 用途 | batch_size | learning_rate | Params | Scaling | その他 |
|---------|------|-----------|--------------|--------|---------|--------|
| `NewLLMConfig` | CPU訓練（レガシー） | 16 | 0.0001 | 2.74M | - | device="cpu" |
| `NewLLMGPUConfig` | T4 GPU訓練 | 512 | 0.0001 | 2.74M | - | device="cuda" |
| **`NewLLML4Config`** | **L4 GPU訓練（推奨）** | **2048** | **0.0008** | **2.74M** | **√64=8x** | **device="cuda"** |
| **`NewLLMA100Config`** | **A100 GPU訓練** | **4096** | **0.0011** | **2.74M** | **√128≈11x** | **device="cuda"** |
| **`NewLLMAdvancedL4Config`** | **L4 + 大規模モデル** | **2048** | **0.0004** | **4.84M** | **√64=8x, then /2** | **context=512, layers=12** |
| **`NewLLMAdvancedA100Config`** | **A100 + 大規模モデル** | **4096** | **0.0006** | **4.84M** | **√128≈11x, then /1.8** | **context=512, layers=12** |
| `TransformerConfig` | Transformerベースライン | 16 | 0.0001 | ~3M | - | 比較実験用 |

**推奨**: Colab Proで**L4 GPU**を使う場合は`NewLLML4Config`または`NewLLMAdvancedL4Config`を継承

**重要**:
- **Baseline configs**: **Square Root Scaling Rule**（大規模batch用）を適用
  - CPU baseline (batch=32, lr=0.0001)を基準
  - L4: batch 64x → lr √64 = 8x → 0.0008
  - A100: batch 128x → lr √128 ≈ 11.3x → 0.0011
- **Advanced configs**: さらに**Model Size Scaling Rule**を適用して半減

#### 実装パターン

**新しい実験スクリプトを作成する時（L4 GPU）**:

```python
# 1. L4 GPU用設定クラスをインポート
from src.utils.config import NewLLML4Config  # L4 GPU (24GB)

# 2. 継承して、実験固有の設定のみ上書き
class MyExperimentConfig(NewLLML4Config):
    """My experiment configuration for L4 GPU

    Inherits L4 optimization from NewLLML4Config:
    - batch_size = 2048     ← L4用に最適化済み
    - learning_rate = 0.0004 ← Linear Scaling Rule適用済み
    - device = "cuda"
    """
    # 実験固有の設定のみ記述（batch_size/learning_rate/deviceは継承）
    num_epochs = 100
    # learning_rateは上書きしない！自動継承を使う

# 3. 使用
config = MyExperimentConfig()
```

**大規模モデルの場合（L4 GPU）**:

```python
from src.utils.config import NewLLMAdvancedL4Config

class MyAdvancedConfig(NewLLMAdvancedL4Config):
    """Advanced model for L4 GPU

    Inherits:
    - batch_size = 2048
    - context_vector_dim = 512
    - num_layers = 12
    """
    # 実験固有の設定のみ
    num_epochs = 100
```

#### 設定変更が必要な場合

**新しいGPUタイプに対応する場合**:

1. **`src/utils/config.py`に追加** ← ここだけ変更
2. 全てのスクリプトが自動的に新設定を使える

```python
# src/utils/config.py に追加
class NewLLMH100Config(NewLLMGPUConfig):
    """H100 GPU (80GB VRAM) 用設定"""
    batch_size = 8192        # H100は超大容量（16x T4）
    learning_rate = 0.0016   # Linear Scaling Rule: 16x T4
```

#### チェックリスト

新しいスクリプト作成時:
- [ ] `src/utils.config`から設定クラスをインポートしているか？
- [ ] 適切な設定クラス（GPU用など）を継承しているか？
- [ ] 重複した設定値を定義していないか？
- [ ] **`batch_size`、`learning_rate`、`device`などは自動継承されているか？**
- [ ] スクリプト内で`batch_size`や`learning_rate`を上書きしていないか？

#### この原則の利点

1. **一貫性**: 全スクリプトで同じ設定が使われる
2. **保守性**: 設定変更が1箇所で済む
3. **可読性**: 実験固有の設定のみが記述される
4. **エラー防止**: 設定の不一致によるバグを防ぐ

**違反した場合**: 今回のようなbatch_size不一致が再発します

---

## New-LLM Architecture Design Principles - CRITICAL

### 🎯 固定メモリ使用量の原則（Fixed Memory Usage Principle）

**New-LLMの根本的設計目標**: シーケンス長に関わらず**メモリ使用量が一定**であること

これはNew-LLMの存在意義であり、**絶対に守るべき原則**です。

#### Transformerとの比較

| アーキテクチャ | メモリ使用量 | シーケンス長の制約 |
|--------------|-------------|------------------|
| **Transformer** | O(n²) | Attentionが長いシーケンスで爆発 |
| **New-LLM** | O(1) | 固定サイズ文脈ベクトルで任意長対応可能 |

#### 実装で絶対に禁止すること ❌

**1. 位置埋め込み（Positional Embeddings）の使用**

```python
# ❌ 絶対禁止 - max_seq_lengthに制限される
self.position_embedding = nn.Embedding(max_seq_length, embed_dim)

# 問題点:
# - 学習時のmax_seq_length以上のシーケンスを処理できない
# - メモリ使用量がmax_seq_lengthに依存
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

# ✓ 正しい - 固定サイズの文脈ベクトルのみ保持
context = torch.zeros(batch_size, context_dim)  # 固定サイズ
for t in range(seq_len):
    context = update(context, input[t])  # 上書き更新
```

#### 実装で推奨すること ✅

**1. 固定サイズ文脈ベクトル**

```python
# ✓ 推奨
self.context_dim = 512  # 固定サイズ
context = torch.zeros(batch_size, self.context_dim)  # O(1)メモリ

# どんなに長いシーケンスでもメモリ使用量は一定
for t in range(1000000):  # 100万ステップでもOK
    context = self.context_norm(forget_gate * context + input_gate * delta)
```

**2. 逐次処理での暗黙的位置情報**

```python
# ✓ 推奨 - RNN/LSTM型の処理
for t in range(seq_len):  # 任意長
    # tが増えるほど「後ろの位置」であることを自然に学習
    fnn_input = torch.cat([token_embeds[t], context], dim=-1)
    hidden = self.fnn_layers(fnn_input)
    context = update_context(hidden, context)
```

**3. LayerNormによる安定化（必須）**

```python
# ✓ 必須 - 文脈ベクトルの正規化
self.context_norm = nn.LayerNorm(self.context_dim)

context = forget_g * context + input_g * context_delta
context = self.context_norm(context)  # 毎ステップ正規化
```

#### 開発時のチェックリスト

New-LLMの実装・修正時は必ず確認：

- [ ] `nn.Embedding(max_seq_length, ...)` のような位置埋め込みを使っていないか？
- [ ] `hidden_states.append(...)` のような全状態保存をしていないか？
- [ ] メモリ使用量がシーケンス長に依存する操作をしていないか？
- [ ] `context = self.context_norm(context)` を毎ステップ実行しているか？
- [ ] 任意長のシーケンスを処理できる設計になっているか？

#### この原則を守る理由

1. **New-LLMの存在意義**: Transformerの O(n²) 問題を O(1) で解決
2. **スケーラビリティ**: 長いシーケンス（小説、コード全体など）を処理可能
3. **リソース効率**: メモリが限られた環境でも動作
4. **設計の一貫性**: RNN/LSTMの良い部分を継承

**違反した場合**: New-LLMがただの「重いTransformer」になり、研究的価値を失う

---

## Code and File Cleanup Policy - CRITICAL

**古いコード・ファイルを残すことは厳禁です (Leaving old code/files is strictly prohibited)**

### ⚠️ 重要：無効化ではなく削除

**不要なコードは必ず削除すること。コメントアウトや無効化では不十分です。**

```python
# ❌ 禁止 - コメントアウトで残す
# logging_steps=args.logging_steps,  # 使わないけど残しておく

# ❌ 禁止 - 無効化で残す
if False:  # 古いコード
    old_function()

# ✅ 正しい - 完全に削除
# （不要なコードは何も残さない）
```

**理由**:
- コメントアウトされたコードは混乱の元
- 何が有効で何が無効かわからなくなる
- コードベースが肥大化する
- 後で「なぜこれがあるのか」と疑問に思う

**徹底的に削除すること。**

### 基本原則

新しい実装、バグ修正、ファイル更新を行った際は、**必ず古いバージョンを完全に削除**してください。

### 削除対象

#### 1. 古いコードファイル

- ✗ 使われなくなったメソッド・関数
- ✗ コメントアウトされたコード
- ✗ デバッグ用の一時コード
- ✗ 重複した機能を持つコード
- ✗ 古いバージョンのスクリプト
- ✗ **不要になったパラメータ・引数** ← 重要！

**例**:
```python
# ✗ 悪い例 - 古いコードをコメントアウトで残す
# def old_function():
#     # 古い実装
#     pass

def new_function():
    # 新しい実装
    pass

# ✓ 良い例 - 古いコードは完全に削除
def new_function():
    # 新しい実装
    pass
```

**不要なパラメータの例**:
```python
# ✗ 悪い例 - 使わないパラメータを残す
parser.add_argument('--logging-steps', type=int, default=10,
                   help='Log every N steps')  # ← logging_strategy="no"なので不要
parser.add_argument('--save-steps', type=int, default=500,
                   help='Save checkpoint every N steps')  # ← save_strategy="epoch"なので不要

# ✓ 良い例 - 不要なパラメータは完全に削除
# （上記の--logging-steps、--save-stepsは削除済み）
parser.add_argument('--output-dir', type=str, default='./checkpoints',
                   help='Output directory for checkpoints')
```

#### 2. 古い画像・グラフファイル

- ✗ バグのあるグラフ
- ✗ 古い実験結果の画像
- ✗ 修正前のビジュアライゼーション
- ✗ `_old`, `_backup`, `_v1` などの接尾辞がついたファイル

**具体例**:
```bash
# ✗ 悪い例 - 古い画像を残す
checkpoints/
  ├── graph.png              # 新しい
  ├── graph_old.png          # 古い - 削除すべき
  └── graph_backup.png       # バックアップ - 削除すべき

# ✓ 良い例 - 最新版のみ保持
checkpoints/
  └── graph.png              # 新しい版のみ
```

#### 3. 古いチェックポイント・データファイル

- ✗ 古い実験の `.pt` ファイル
- ✗ 使われていないデータファイル
- ✗ 一時的な中間ファイル

### 後方互換性について

**このプロジェクトでは後方互換性を意識しません。**

- ✗ 古いインターフェースを残す必要なし
- ✗ 古いAPIを維持する必要なし
- ✗ 古いファイル名を保持する必要なし

**理由**:
- 研究プロジェクトであり、プロダクションコードではない
- 古いコードが残ると混乱とバグの温床になる
- クリーンな状態を維持することが最優先

### 実行手順

#### ファイル更新時:

1. **新しいファイルを作成**
2. **動作確認**
3. **古いファイルを完全に削除**
4. **コミット時に削除も含める**

```bash
# 例: グラフファイルの更新
python generate_new_graph.py              # 新しいグラフ生成
# 確認: new_graph.png が正しく生成された

rm old_graph.png                          # 古いグラフを削除
git add new_graph.png                     # 新しいファイルを追加
git rm old_graph.png                      # Gitからも削除
git commit -m "Replace graph with fixed version, remove old buggy graph"
```

#### コード更新時:

1. **新しいコードを実装**
2. **テスト**
3. **古いコード（コメントアウト含む）を削除**
4. **未使用のインポートも削除**

### チェックリスト

コミット前に必ず確認：

- [ ] コメントアウトされたコードは削除したか？
- [ ] 未使用のメソッド・関数は削除したか？
- [ ] 古いバージョンのファイルは削除したか？
- [ ] `_old`, `_backup`, `_temp` などの接尾辞のファイルは削除したか？
- [ ] 未使用のインポート文は削除したか？
- [ ] デバッグ用のprint文は削除したか？

### Git操作

古いファイルを削除する際:

```bash
# ファイルシステムとGitの両方から削除
git rm old_file.py

# または
rm old_file.py
git add -A  # 削除も含めてステージング
```

### 例外

**削除してはいけないもの**:

- ✓ `.gitignore` で除外されているファイル（ローカル環境のみ）
- ✓ ドキュメント内の「悪い例」としての参照コード
- ✓ テストケースとして意図的に残している古い動作

### 違反の影響

古いコード・ファイルを残すことで発生する問題：

1. **混乱** - どれが最新版かわからない
2. **バグ** - 古いコードを誤って使用
3. **メンテナンス困難** - 複数バージョンの同期が必要
4. **リポジトリ肥大化** - 不要なファイルでサイズ増加

## 実装後の必須チェックリスト

機能実装・バグ修正後は以下を必ず確認：

- [ ] 古いメソッドやクラスが残っていないか
- [ ] コメントアウトされたコードが残っていないか
- [ ] 古い画像・グラフファイルが残っていないか
- [ ] 使用されていないファイルが残っていないか
- [ ] 重複した機能を持つコードが複数箇所にないか
- [ ] 未使用のインポートが残っていないか
- [ ] デバッグ用のコードが残っていないか

## コミットメッセージ - CRITICAL

**コミットメッセージは1行で簡潔に記述すること**

```bash
# ✅ 正しい - 1行で簡潔に
git commit -m "Add WikiText dataset support and re-enable tqdm"
git commit -m "Fix perplexity calculation bug in MetricsCallback"
git commit -m "Remove obsolete logging parameters from train.py"

# ❌ 禁止 - 複数行や詳細な説明
git commit -m "Fix graph bug and remove old buggy version

- Fixed: Perplexity plot now uses linear scale
- Removed: old_graph.png (had log scale bug)
- Added: new_graph_fixed.png (correct version)
"
```

**原則**:
- 1行で何をしたか明確に記述
- 50文字以内を目安
- 英語で記述
- 動詞で始める（Add, Fix, Remove, Update など）

## ファイル命名規則 - 完全固定方針

**ファイル名は完全に固定し、常に同じ名前で上書きする**

### 基本原則

- ✓ **固定ファイル名**: 同じ種類のファイルは常に同じ名前を使う
- ✗ **バージョン接尾辞禁止**: `_v1`, `_v2`, `_old`, `_new`, `_fixed` などは使わない
- ✗ **日付接尾辞禁止**: `_20250117`, `_latest` などは使わない
- ✓ **上書き**: 新しいバージョンは常に同じファイル名で上書き

### 実装例

#### グラフ・画像ファイル

```bash
# ✗ 悪い例 - バージョン管理でファイル名を変える
checkpoints/
  ├── new_llm_training_curves.png
  ├── new_llm_training_curves_fixed.png      # ダメ
  ├── new_llm_training_curves_v2.png         # ダメ
  └── transformer_baseline_curves_fixed.png  # ダメ

# ✓ 良い例 - 完全固定ファイル名で上書き
checkpoints/
  ├── new_llm_training_curves.png           # 常にこの名前
  └── transformer_baseline_curves.png       # 常にこの名前
```

#### コード生成ファイル

```python
# trainer.py の plot_and_save_training_curves() メソッド

# ✗ 悪い例
save_path = f"{self.model_name}_training_curves_{timestamp}.png"

# ✓ 良い例 - 常に同じファイル名
save_path = f"checkpoints/{self.model_name}_training_curves.png"
```

### 推奨ファイル名一覧

| ファイル種類 | 固定ファイル名 | 説明 |
|-------------|---------------|------|
| New-LLM訓練曲線 | `new_llm_training_curves.png` | 常にこの名前で上書き |
| Transformerベースライン | `transformer_baseline_curves.png` | 常にこの名前で上書き |
| 実験サマリー | `experiment_summary.md` | 常にこの名前で更新 |
| チェックポイント | `best_{model_name}.pt` | モデルごとに固定名 |

### 利点

1. **混乱防止**: どのファイルが最新版か一目瞭然
2. **シンプル**: ファイル名を考える必要がない
3. **クリーン**: checkpointsディレクトリが散らからない
4. **自動化**: スクリプトが常に同じパスを参照できる

### Git履歴でのバージョン管理

ファイル名を固定しても、Gitの履歴で過去バージョンを参照可能：

```bash
# 過去のバージョンを見る
git log checkpoints/new_llm_training_curves.png
git show HEAD~3:checkpoints/new_llm_training_curves.png

# 過去バージョンを復元（必要な場合のみ）
git checkout HEAD~5 -- checkpoints/new_llm_training_curves.png
```

### 実装時の注意

```python
# 実験スクリプトでの実装例
def save_results():
    # ✓ 固定ファイル名
    GRAPH_PATH = "checkpoints/new_llm_training_curves.png"

    # 上書き保存（古いファイルは自動的に置き換わる）
    plt.savefig(GRAPH_PATH, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {GRAPH_PATH}")
```

### 例外

固定ファイル名の例外（複数保持が許される場合）:

- ✓ 異なる実験の結果（例: `experiment1_results.md`, `experiment2_results.md`）
- ✓ 異なるモデルアーキテクチャ（例: `new_llm_curves.png`, `transformer_curves.png`）
- ✓ 異なるデータセット（例: `wikitext_results.md`, `bookcorpus_results.md`）

**重要**: 同じ種類のファイルは必ず固定名で上書き

## まとめ

**鉄則**:
- ✓ 新しいバージョンを作ったら、古いバージョンは即座に削除
- ✓ ファイル名は完全に固定し、常に上書き
- ✗ 「念のため」残すことは禁止
- ✗ バージョン接尾辞（_v1, _old, _fixed）は禁止
- ✓ Gitの履歴に残っているので、必要なら復元可能

**徹底的なクリーンアップとファイル名の統一がプロジェクトの品質を保ちます。**

---

## Google Colab コマンド実行の正しい方法 - CRITICAL

### ⚠️ IndentationError を繰り返さないために

**重要**: Colabノートブックのセル内でコマンドを実行する際、**絶対にインデントを入れない**

### ❌ 間違った例（IndentationErrorになる）

```python
# ❌ これはエラーになる
  %cd /content
  !rm -rf new-llm
  !git clone https://github.com/rato-tokyo/new-llm
```

```python
# ❌ これもエラーになる（スペースやタブが入っている）
%cd /content
  !rm -rf new-llm  # ← インデントがあるとエラー
```

### ✅ 正しい例

**方法1: 行頭から開始（推奨）**

```python
%cd /content
!rm -rf new-llm
!git clone https://github.com/rato-tokyo/new-llm
%cd new-llm
```

**方法2: %%bash を使う**

```bash
%%bash
cd /content
rm -rf new-llm
git clone https://github.com/rato-tokyo/new-llm
cd new-llm
pip install -r requirements.txt
python scripts/train_wikitext_advanced.py
```

**方法3: 1行で繋げる**

```python
%cd /content && !rm -rf new-llm && !git clone https://github.com/rato-tokyo/new-llm && %cd new-llm
```

### 🎯 Colabでのコマンド実行ルール

1. **`!`で始まるコマンド**: シェルコマンド実行
   ```python
   !pip install torch
   !python train.py
   !nvidia-smi
   ```

2. **`%`で始まるコマンド**: Jupyterマジックコマンド
   ```python
   %cd /content/new-llm
   %pip install torch  # !pip と同じ
   ```

3. **`%%bash`**: セル全体をbashスクリプトとして実行
   ```bash
   %%bash
   cd /content
   ls -la
   ```

4. **複数コマンドを1行で**: `&&` で繋げる
   ```python
   !cd /content && rm -rf new-llm && git clone https://github.com/user/repo
   ```

### 🔧 Colabでの推奨ワークフロー

**セル1: クローン**
```python
%cd /content
!rm -rf new-llm
!git clone https://github.com/rato-tokyo/new-llm
%cd new-llm
```

**セル2: 確認**
```python
!grep "batch_size" scripts/train_wikitext_advanced.py
```

**セル3: インストール**
```python
!pip install -q -r requirements.txt
```

**セル4: 実行**
```python
!python scripts/train_wikitext_advanced.py
```

### ⚠️ よくある間違い

1. **コピペ時にインデントが入る**
   - テキストエディタからコピーするとインデントが入ることがある
   - Colabに直接貼り付ける前に、**行頭にスペースやタブがないか確認**

2. **Markdownセルとコードセルの混同**
   - Markdownセル（説明用）とコードセル（実行用）を間違えない
   - コマンドは必ず「コードセル」で実行

3. **`!` と `%` の混同**
   - `!command`: シェルコマンド
   - `%command`: Jupyterマジックコマンド
   - `%%bash`: セル全体をbash実行

### 💡 トラブルシューティング

**IndentationError が出たら**:
1. セル内の全テキストをコピー
2. テキストエディタに貼り付け
3. 各行の先頭のスペース・タブを削除
4. Colabに貼り直し

**確実な方法**:
```python
# 各コマンドを別々のセルで実行する
# セル1
%cd /content

# セル2
!rm -rf new-llm

# セル3
!git clone https://github.com/rato-tokyo/new-llm

# セル4
%cd new-llm

# セル5
!python scripts/train_wikitext_advanced.py
```

---

**鉄則: Colabのセル内では行頭にスペース・タブを入れない！**
