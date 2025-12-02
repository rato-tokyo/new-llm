# Google Colab Setup Guide

**最終更新**: 2025-12-02

---

## クイックスタート

### Step 1: 環境セットアップ

```python
# Google Drive マウント
from google.colab import drive
drive.mount('/content/drive')

# リポジトリクローン
!git clone https://github.com/your-username/new-llm.git
%cd new-llm

# 依存関係インストール
!pip install -q transformers datasets torch accelerate

# GPU確認
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 2: 永続ストレージ設定

```python
# Google Drive にディレクトリ作成
!mkdir -p /content/drive/MyDrive/new-llm/cache
!mkdir -p /content/drive/MyDrive/new-llm/checkpoints

# シンボリックリンク作成
!ln -sf /content/drive/MyDrive/new-llm/cache ./cache
!ln -sf /content/drive/MyDrive/new-llm/checkpoints ./checkpoints
```

### Step 3: 実験実行

```bash
# Cascade Context 実験（2000サンプル）
python3 scripts/experiment_cascade_context.py -s 2000
```

---

## 推奨設定

### サンプル数の目安

| 環境 | 推奨サンプル数 | 処理時間 |
|------|---------------|----------|
| **ローカル (CPU)** | 2-5 | 数分 |
| **Colab (GPU)** | 2000+ | 30-60分 |

### GPU メモリ

| Colab Tier | GPU | メモリ |
|------------|-----|--------|
| Free | T4 | 15 GB |
| Pro | T4/V100 | 25-40 GB |

2000サンプルで約10-15GB使用。

---

## トラブルシューティング

### Out of Memory (OOM)

```python
# GPUメモリ確認
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# キャッシュクリア
torch.cuda.empty_cache()
```

**対策**:
1. ランタイム再起動
2. サンプル数を減らす（1000サンプルでテスト）
3. High-RAM ランタイムに切り替え

### セッションタイムアウト

**予防**:
- Google Drive に保存（シンボリックリンク設定済み）
- オフピーク時間に実行

**復旧**:
```bash
# リポジトリ更新
cd /content/new-llm && git pull

# 実験再開
python3 scripts/experiment_cascade_context.py -s 2000
```

### 環境リセット対策

Colab は頻繁に環境リセットされます。以下のファイルは自動生成されます：

| ファイル | 用途 |
|----------|------|
| `./data/example_val.txt` | 検証データ |
| `./cache/ultrachat_*samples_full.pt` | 訓練データキャッシュ |

---

## 完全なノートブック例

```python
# === Cell 1: Setup ===
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/your-username/new-llm.git
%cd new-llm
!pip install -q transformers datasets torch accelerate

# === Cell 2: Verify Environment ===
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# === Cell 3: Setup Persistent Storage ===
!mkdir -p /content/drive/MyDrive/new-llm/cache
!mkdir -p /content/drive/MyDrive/new-llm/checkpoints
!ln -sf /content/drive/MyDrive/new-llm/cache ./cache
!ln -sf /content/drive/MyDrive/new-llm/checkpoints ./checkpoints

# === Cell 4: Run Experiment ===
!python3 scripts/experiment_cascade_context.py -s 2000

# === Cell 5: Check Results ===
!ls -lh checkpoints/
```

---

## 期待される結果

### Cascade Context (2000サンプル)

| 指標 | 期待値 |
|------|--------|
| Val PPL | ~110-120 |
| Val Acc | ~25-26% |
| 実効次元 | 70-80% |
| 処理時間 | 30-60分 |

---

*Last Updated: 2025-12-02*
