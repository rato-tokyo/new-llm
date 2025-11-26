# Linux PC 全データ訓練ガイド

## 概要

UltraChat 200kサンプル（約25.6Mトークン）を使用したPhase 1 CVFP学習の実行ガイド。

## 環境要件

| 項目 | 要件 |
|------|------|
| OS | Linux Mint (または任意のLinux) |
| RAM | 32GB（推奨）、最小8GB |
| ストレージ | 外付けNVMe 1TB |
| ファイルシステム | ext4 |
| Python | 3.8以上 |

## ストレージ使用量見積もり

| データ種別 | サイズ |
|------------|--------|
| トークン埋め込みキャッシュ | 39GB |
| コンテキスト（ダブルバッファ） | 78GB |
| トークンID | 0.2GB |
| チェックポイント | ~0.5GB |
| **合計** | **~120GB** |

## メモリ使用量

実行時のメモリ使用は**約5-8GB**。

理由：
- メモリマップファイル(mmap)でディスク上のデータを仮想的にアクセス
- チャンク処理で100万トークンずつ処理
- bf16精度で50%削減

## セットアップ手順

### 1. リポジトリ取得

```bash
cd ~/Desktop/git  # または任意のディレクトリ
git clone https://github.com/YOUR_REPO/new-llm.git
cd new-llm
```

### 2. Python依存関係インストール

```bash
pip install torch transformers datasets numpy tqdm
```

### 3. NVMeのマウント

```bash
# NVMeデバイス確認
lsblk

# ext4でフォーマット（初回のみ、データ消去注意）
sudo mkfs.ext4 /dev/nvme0n1p1

# マウントポイント作成
sudo mkdir -p /mnt/nvme

# マウント
sudo mount /dev/nvme0n1p1 /mnt/nvme

# 権限設定
sudo chown $USER:$USER /mnt/nvme

# マウント確認
df -h /mnt/nvme
```

### 4. 自動マウント設定（オプション、再起動後も維持）

```bash
# UUIDを取得
sudo blkid /dev/nvme0n1p1

# fstabに追加
echo "UUID=YOUR_UUID /mnt/nvme ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

## 実行手順

### Step 1: データ準備（約2-3時間）

```bash
python3 scripts/prepare_disk_offload.py --output_dir /mnt/nvme/cvfp
```

**処理内容**:
1. UltraChat 200kサンプルをストリーミングダウンロード
2. GPT-2トークナイザーでトークン化
3. GPT-2埋め込みを計算
4. ディスクに保存

**出力ファイル**:
```
/mnt/nvme/cvfp/
├── token_ids.npy          # トークンID (0.2GB)
├── embeddings.npy         # GPT-2埋め込み (39GB)
├── context_a.npy          # コンテキストバッファA (39GB)
├── context_b.npy          # コンテキストバッファB (39GB)
└── metadata.json          # メタデータ
```

### Step 2: 訓練開始（約4-5日）

```bash
python3 train_full_ultrachat.py --disk_dir /mnt/nvme/cvfp --num_layers 6
```

**主要パラメータ**:
- `--disk_dir`: データディレクトリ
- `--num_layers`: レイヤー数（6推奨）
- `--max_iterations`: 最大イテレーション（デフォルト10）
- `--lr`: 学習率（デフォルト0.002）

### Step 3: 中断からの再開

```bash
python3 train_full_ultrachat.py --disk_dir /mnt/nvme/cvfp --resume
```

チェックポイントは各イテレーション終了時に自動保存される。

## 進捗確認

### ログファイル

訓練中の進捗はコンソールに表示される：

```
Iteration 1/10
Processing chunk 1/26: 0-1000000
  Effective Rank: 55.3% (425/768)
  CVFP Loss: 0.0234
  Diversity Loss: -0.0156
Processing chunk 2/26: 1000000-2000000
  ...
```

### メモリ使用確認

```bash
# 別ターミナルで実行
watch -n 5 free -h

# または
htop
```

### ディスク使用確認

```bash
df -h /mnt/nvme
du -sh /mnt/nvme/cvfp/*
```

## トラブルシューティング

### メモリ不足エラー

```bash
# スワップ追加（一時的）
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### NVMeが認識されない

```bash
# デバイス確認
sudo fdisk -l

# NVMeドライバ確認
lsmod | grep nvme
```

### 権限エラー

```bash
sudo chown -R $USER:$USER /mnt/nvme/cvfp
```

### データ準備が途中で止まった

```bash
# 既存データを削除して再実行
rm -rf /mnt/nvme/cvfp
python3 scripts/prepare_disk_offload.py --output_dir /mnt/nvme/cvfp
```

## 訓練完了後

### 成果物

```
/mnt/nvme/cvfp/
└── checkpoints/
    └── checkpoint_iter_10.pt  # 最終チェックポイント
```

### Phase 2への引き継ぎ

チェックポイントには以下が含まれる：
- `model_state_dict`: モデルの重み（ContextBlock）
- `iteration`: 完了イテレーション数
- `stats`: 訓練統計

```python
# Phase 2でのロード例
checkpoint = torch.load('/mnt/nvme/cvfp/checkpoints/checkpoint_iter_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 設計詳細

### なぜ32GBで25.6Mトークンを処理できるか

1. **メモリマップファイル (mmap)**
   - numpy.memmapでディスク上のファイルをメモリのように扱う
   - OSが必要な部分だけを自動的にページイン/アウト
   - 39GBのファイルでも実メモリ使用は数GB

2. **チャンク処理**
   - 100万トークンずつ処理
   - 25.6Mトークン → 26チャンク

3. **bf16精度**
   - float32の半分のサイズ
   - numpyではfloat16として保存、PyTorchでbf16に変換

4. **ダブルバッファリング**
   - ポインタ交換のみ、データコピーなし

### アーキテクチャ

```
E案（採用）:
- ContextBlock: 6層（Phase 1で学習）
- TokenBlock: 6層（Phase 2で学習、今回は未使用）
- 各TokenBlockレイヤーiは、ContextBlockレイヤーiの出力を参照
```

## 連絡事項

- 訓練は約4-5日かかる見込み
- 中断しても`--resume`で再開可能
- 最終的な成果物はチェックポイントファイル（モデル重み）
