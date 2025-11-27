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

### 2. UltraChat（デフォルト）

```python
# config.py
train_data_source = "ultrachat"
num_samples = 50  # サンプル数

val_data_source = "text_file"
val_text_file = "./data/ultrachat_50samples_val.txt"
```

## 現在のデータファイル

- `ultrachat_50samples_train.txt` - UltraChat 50サンプルの訓練データ
- `ultrachat_50samples_val.txt` - UltraChat検証データ
- `example_val.txt` - 検証データサンプル

## 推奨事項

1. **語彙の一貫性**: 訓練データと検証データで同じ語彙を使用してください
2. **データ量**: 最低でも数百トークンは必要です
3. **ドメイン**: 特定ドメインで学習したい場合、そのドメインのテキストを使用してください
