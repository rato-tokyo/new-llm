# データディレクトリ

独自のテキストデータを使って訓練する場合は、このディレクトリにファイルを配置してください。

## 使い方

### 1. 単一テキストファイルから訓練

```python
# config.py
train_data_source = "text_file"
train_text_file = "./data/my_train.txt"

val_data_source = "text_file"
val_text_file = "./data/my_val.txt"
```

**ファイル形式**:
- UTF-8エンコーディング
- 段落は空行（`\n\n`）で区切る
- または1行1テキストセグメント

**例** (`my_train.txt`):
```
これは最初の段落です。
複数行でも構いません。

これは2番目の段落です。

これは3番目の段落です。
```

### 2. 複数ファイルから訓練

```python
# config.py
train_data_source = "text_dir"
train_text_dir = "./data/train/"

val_data_source = "text_dir"
val_text_dir = "./data/val/"
```

**ディレクトリ構造**:
```
data/
├── train/
│   ├── doc1.txt
│   ├── doc2.txt
│   └── doc3.txt
└── val/
    ├── val1.txt
    └── val2.txt
```

各`.txt`ファイルは1つのテキストセグメントとして扱われます。

### 3. UltraChat（デフォルト）

```python
# config.py
train_data_source = "ultrachat"
num_samples = 10

val_data_source = "manual"  # 手動作成済みの ./cache/manual_val_tokens.pt を使用
```

### 4. 訓練データから自動分割

```python
# config.py
train_data_source = "text_file"  # or "text_dir" or "ultrachat"
train_text_file = "./data/my_data.txt"

val_data_source = "auto_split"
train_val_split = 0.8  # 80% train, 20% val
```

## データソースオプション

### 訓練データ (`train_data_source`)
- `"ultrachat"` - HuggingFace UltraChat データセット
- `"text_file"` - 単一テキストファイル
- `"text_dir"` - ディレクトリ内の全`.txt`ファイル

### 検証データ (`val_data_source`)
- `"manual"` - 事前にトークン化された `./cache/manual_val_tokens.pt`
- `"text_file"` - 単一テキストファイル
- `"text_dir"` - ディレクトリ内の全`.txt`ファイル
- `"auto_split"` - 訓練データから自動分割

## 推奨事項

1. **語彙の一貫性**: 訓練データとvalidationデータで同じ語彙を使用してください
2. **データ量**: 最低でも数百トークンは必要です
3. **ドメイン**: 特定ドメインで学習したい場合、そのドメインのテキストを使用してください
