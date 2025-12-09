# Senri Fine-tuner

Senriモデルのファインチューニングツール。
カスタム知識データを学習させることで、特定ドメインの知識を持つモデルを作成できます。

## 使い方

```bash
cd senri-fine-tuner

# 基本的な使い方
python3 finetune.py --data data/example_knowledge.json --epochs 10

# ベースモデルを指定（事前学習済みモデルから開始）
python3 finetune.py --data data/custom.json --base-model ../checkpoints/pretrained.pt

# 出力先を指定
python3 finetune.py --data data/custom.json --output checkpoints/finetuned.pt
```

## 入力データ形式

```json
{
  "instances": [
    {
      "knowledge": "東京は日本の首都です。人口は約1400万人。",
      "qa_pairs": [
        {"question": "日本の首都は？", "answer": "東京"},
        {"question": "東京の人口は？", "answer": "約1400万人"}
      ]
    }
  ]
}
```

## 訓練の仕組み（CDR方式）

1. `knowledge`をコンテキストとしてメモリに書き込み
2. `question → answer` の推論パターンを学習
3. `knowledge`部分はloss計算から除外（丸暗記を防止）

## オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--data` | 必須 | 入力JSONファイルパス |
| `--base-model` | None | ベースモデルのチェックポイント |
| `--output` | checkpoints/finetuned.pt | 出力先パス |
| `--epochs` | 10 | エポック数 |
| `--batch-size` | 4 | バッチサイズ |
| `--lr` | 1e-4 | 学習率 |
| `--val-split` | 0.1 | 検証データ割合 |
| `--max-knowledge-len` | 256 | 知識の最大トークン数 |
| `--max-qa-len` | 128 | QAの最大トークン数 |
| `--seed` | 42 | ランダムシード |

## ディレクトリ構造

```
senri-fine-tuner/
├── finetune.py           # メインスクリプト
├── data/
│   └── example_knowledge.json  # サンプルデータ
├── checkpoints/          # 出力先（自動作成）
└── README.md
```
