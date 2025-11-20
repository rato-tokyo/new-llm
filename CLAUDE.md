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

新しい訓練データセットを追加する際は、**必ず以下を実装**すること：

1. ✅ `scripts/train_<dataset>.py` - Python訓練スクリプト
2. ✅ **`scripts/colab_train_<dataset>.sh`** - 1行実行用bashスクリプト（**最重要・必須**）
3. ✅ ドキュメント（1行コマンドを最初に記載）

**`scripts/colab_train_<dataset>.sh`がない実装は不完全とみなす。**

### colab_train_*.sh スクリプトの必須内容

```bash
#!/bin/bash
set -e

# 1. パラメータ解析
NUM_LAYERS=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_layers) NUM_LAYERS="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# 2. 最新版取得
cd /content
rm -rf new-llm
git clone https://github.com/rato-tokyo/new-llm
cd new-llm

# 3. 依存関係インストール
pip install -q datasets tqdm

# 4. バックグラウンド実行
nohup python scripts/train_<dataset>.py --num_layers $NUM_LAYERS > /content/log.txt 2>&1 &

# 5. 初期ログ表示
sleep 10
tail -30 /content/log.txt

# 6. モニタリングコマンド表示
echo "📋 Monitoring: !tail -20 /content/log.txt"
echo "🛑 Stop: !pkill -9 -f train_<dataset>"
```

---

## 🧪 コード修正時の必須テストポリシー - CRITICAL

**すべてのコード修正後、必ず手元でテストしてからコミット・デプロイすること**

### 必須実行手順

**テストは最小限のデータ・epoch・stepで実施する**

```bash
# ステップ1: 構文チェック
python3 -m py_compile train.py

# ステップ2: 最小限のデータでエンドツーエンドテスト
python3 train.py \
    --max-samples 100 \      # 最小限のデータ（100サンプル）
    --epochs 2 \             # 最小限のepoch（2エポック）
    --batch-size 4 \
    --device cpu

# ステップ3: 動作確認
# - エポックごとのメトリクス表示を確認
# - エラーなく完了することを確認
```

**テスト時間の目安**:
- 100サンプル、2エポック → **4-5分**

**鉄則**: テストするときは**epoch=2、samples=100**で十分

### チェックリスト

コミット前に必ず確認：

- [ ] **構文チェック完了** - `python3 -m py_compile`
- [ ] **テストスクリプトで動作確認**
- [ ] **エラーメッセージを実際に確認**

---

## 🔄 コード品質・保守性の鉄則 - CRITICAL

**コードの重複、パラメータの不整合、キャッシュ不足によるミスを防ぐ**

### 1. コード重複の徹底排除 - DRY原則

**❌ 絶対に避けるべきパターン**:

```python
# ❌ 悪い例 - trainとevalで同じロジックを重複実装
def train_epoch(...):
    token_loss = F.cross_entropy(...)
    recon_loss = F.mse_loss(...)
    # ... 約90行のロジック

def evaluate(...):
    # ← まったく同じロジックをコピペ
    token_loss = F.cross_entropy(...)
    recon_loss = F.mse_loss(...)
    # ... 約90行の重複コード
```

**✅ 正しいパターン - 共通関数で一元化**:

```python
# ✅ 良い例 - 共通ロジックを1箇所に集約
def _compute_batch_metrics(model, input_ids, device, context_loss_weight):
    """共通のメトリクス計算（trainとval両方で使用）"""
    token_loss = F.cross_entropy(...)
    recon_loss = F.mse_loss(...)

    return {
        'loss': loss,
        'token_loss': token_loss,
        'recon_loss': recon_loss,
    }

def train_epoch(...):
    metrics = _compute_batch_metrics(...)  # 共通関数を使用
    optimizer.zero_grad()
    metrics['loss'].backward()
    optimizer.step()

def evaluate(...):
    metrics = _compute_batch_metrics(...)  # 同じ関数を使用
```

**チェックリスト**:
- [ ] trainとevalで同じロジックを重複実装していないか？
- [ ] 共通処理を`_compute_*`などの関数に抽出したか？
- [ ] メトリクス追加時に1箇所修正で済むか？

---

### 2. パラメータ同期の徹底 - CRITICAL

新しいパラメータを追加する際は、**必ず以下の全てを更新**すること：

1. **`train.py`** - `argparse`にパラメータ定義
2. **`scripts/colab_train_*.sh`** - パラメータ解析に追加
3. **ドキュメント** - 使用例を記載

**チェックリスト**:
- [ ] `train.py`にパラメータ定義を追加したか？
- [ ] **`scripts/colab_train_*.sh`にパラメータ解析を追加したか？** ← 最重要
- [ ] ドキュメントに使用例を記載したか？

---

### 3. キャッシュの積極活用 - パフォーマンス最適化

**✅ 正しいパターン - キャッシュを活用**:

```python
def create_tokenizer(texts, vocab_size=10000, output_dir='./tokenizer'):
    """Create or load BPE tokenizer"""
    tokenizer_path = f"{output_dir}/tokenizer.json"

    # 既存のトークナイザーを確認
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}...")
        return Tokenizer.from_file(tokenizer_path)  # 0.1秒で完了

    # 初回のみ訓練
    print(f"Training BPE tokenizer...")
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer
```

**キャッシュすべきもの**:
1. **トークナイザー** - `tokenizer.json`
2. **データセット** - HuggingFace Datasetsが自動キャッシュ
3. **モデルチェックポイント** - 実験再開時に再利用

---

### まとめ

**二度と同じミスをしないための鉄則**:

1. **DRY原則を徹底** - コードを重複させない
2. **パラメータ同期を徹底** - `train.py`と`colab_train_*.sh`の両方を更新
3. **キャッシュを活用** - 同じ処理を繰り返さない

---

## 🚀 GPU最適化設定の確認 - CRITICAL

**スクリプト作成・修正時は、必ずGPU用の設定になっているか確認すること**

### GPU別の推奨batch_size

| GPU | VRAM | 推奨batch_size | 備考 |
|-----|------|---------------|------|
| **T4** | 16GB | 512 | Baseline model |
| **L4** | 24GB | **2048** | **4x T4** |
| **A100** | 40GB | **4096** | **8x T4** |

### Learning Rate Scaling Rule - CRITICAL

**batch_sizeを変更したら、learning_rateも適切にスケールすること**

**大規模batch（>= 256）**: Square Root Scaling（推奨）
```
batch_sizeをk倍 → learning_rateを√k倍
```

**正しい計算（CPU baseline基準）**:
```python
# CPU Baseline
batch_size = 32
learning_rate = 0.0001

# L4 GPU（batch 64倍 = 32→2048）
batch_size = 2048
# Square Root Scaling: 0.0001 * √64 = 0.0001 * 8 = 0.0008
learning_rate = 0.0008  # √64 = 8倍

# A100 GPU（batch 128倍 = 32→4096）
batch_size = 4096
# Square Root Scaling: 0.0001 * √128 ≈ 0.0001 * 11.3 = 0.0011
learning_rate = 0.0011  # √128 ≈ 11.3倍
```

**チェックリスト**:
- [ ] batch >= 256: **Square Root Scaling（√k倍）** ← 推奨
- [ ] 最初のエポックでPPLが急速に減少しているか？

---

### Model Size Scaling Rule - CRITICAL

**モデルサイズを大きくしたら、learning_rateを小さくすること**

**基本原則**:
```
モデルサイズが大きい → learning_rateを小さくする
```

**推奨スケーリング**:
```python
# Baseline (2.74M params)
learning_rate = 0.0004

# Advanced (4.84M params, 1.77x larger)
learning_rate = 0.0002  # 半分に減らす（保守的）
```

**チェックリスト**:
- [ ] learning_rateを小さくしたか？
- [ ] 大きなモデルが小さなモデルより性能が良いか？（そうでなければLR調整不足）

---

### 設定の一元管理ポリシー - CRITICAL

**すべての設定値は `src/utils/config.py` で一元管理すること**

```python
# ✅ 良い例 - config.pyから継承
from src.utils.config import NewLLMGPUConfig

class MyConfig(NewLLMGPUConfig):
    # GPU最適化設定を自動継承
    num_epochs = 100
```

**利用可能な設定クラス**:
- `NewLLMConfig` - CPU訓練（レガシー）
- `NewLLMGPUConfig` - T4 GPU訓練
- **`NewLLML4Config`** - **L4 GPU訓練（推奨）**
- **`NewLLMA100Config`** - **A100 GPU訓練**

---

## New-LLM Architecture Design Principles - CRITICAL

### 🎯 固定メモリ使用量の原則（Fixed Memory Usage Principle）

**New-LLMの根本的設計目標**: シーケンス長に関わらず**メモリ使用量が一定**であること

これはNew-LLMの存在意義であり、**絶対に守るべき原則**です。

### Transformerとの比較

| アーキテクチャ | メモリ使用量 | シーケンス長の制約 |
|--------------|-------------|------------------|
| **Transformer** | O(n²) | Attentionが長いシーケンスで爆発 |
| **New-LLM** | O(1) | 固定サイズ文脈ベクトルで任意長対応可能 |

### 実装で絶対に禁止すること ❌

**1. 位置埋め込み（Positional Embeddings）の使用**

```python
# ❌ 絶対禁止 - max_seq_lengthに制限される
self.position_embedding = nn.Embedding(max_seq_length, embed_dim)

# 問題点:
# - 学習時のmax_seq_length以上のシーケンスを処理できない
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

# ✅ 正しい - 固定サイズの文脈ベクトルのみ保持
context = torch.zeros(batch_size, context_dim)  # 固定サイズ
for t in range(seq_len):
    context = update(context, input[t])  # 上書き更新
```

### 開発時のチェックリスト

New-LLMの実装・修正時は必ず確認：

- [ ] `nn.Embedding(max_seq_length, ...)` のような位置埋め込みを使っていないか？
- [ ] `hidden_states.append(...)` のような全状態保存をしていないか？
- [ ] メモリ使用量がシーケンス長に依存する操作をしていないか？
- [ ] 任意長のシーケンスを処理できる設計になっているか？

### この原則を守る理由

1. **New-LLMの存在意義**: Transformerの O(n²) 問題を O(1) で解決
2. **スケーラビリティ**: 長いシーケンス（小説、コード全体など）を処理可能
3. **リソース効率**: メモリが限られた環境でも動作

**違反した場合**: New-LLMがただの「重いTransformer」になり、研究的価値を失う

---

## Code and File Cleanup Policy - CRITICAL

**古いコード・ファイルを残すことは厳禁です**

### 重要：無効化ではなく削除

**不要なコードは必ず削除すること。コメントアウトや無効化では不十分です。**

```python
# ❌ 禁止 - コメントアウトで残す
# old_function()

# ✅ 正しい - 完全に削除
# （不要なコードは何も残さない）
```

### 削除対象

- ✗ 使われなくなったメソッド・関数
- ✗ コメントアウトされたコード
- ✗ デバッグ用の一時コード
- ✗ 古いバージョンのファイル
- ✗ 不要になったパラメータ・引数

### チェックリスト

コミット前に必ず確認：

- [ ] コメントアウトされたコードは削除したか？
- [ ] 未使用のメソッド・関数は削除したか？
- [ ] 古いバージョンのファイルは削除したか？
- [ ] 未使用のインポート文は削除したか？

---

## コミットメッセージ - CRITICAL

**コミットメッセージは1行で簡潔に記述すること**

```bash
# ✅ 正しい - 1行で簡潔に
git commit -m "Add WikiText dataset support"
git commit -m "Fix perplexity calculation bug"
git commit -m "Remove obsolete parameters from train.py"
```

**原則**:
- 1行で何をしたか明確に記述
- 50文字以内を目安
- 英語で記述
- 動詞で始める（Add, Fix, Remove, Update など）

---

## ファイル命名規則 - 完全固定方針

**ファイル名は完全に固定し、常に同じ名前で上書きする**

### 基本原則

- ✅ **固定ファイル名**: 同じ種類のファイルは常に同じ名前を使う
- ❌ **バージョン接尾辞禁止**: `_v1`, `_v2`, `_old`, `_new`, `_fixed` などは使わない
- ❌ **日付接尾辞禁止**: `_20250117`, `_latest` などは使わない
- ✅ **上書き**: 新しいバージョンは常に同じファイル名で上書き

### 例外

固定ファイル名の例外（複数保持が許される場合）:

- ✅ 異なる実験の結果（例: `experiment1_results.md`, `experiment2_results.md`）
- ✅ 異なるモデルアーキテクチャ（例: `new_llm_curves.png`, `transformer_curves.png`）
- ✅ 異なるデータセット（例: `wikitext_results.md`, `bookcorpus_results.md`）

**重要**: 同じ種類のファイルは必ず固定名で上書き

---

## まとめ

**鉄則**:
- ✅ 新しいバージョンを作ったら、古いバージョンは即座に削除
- ✅ ファイル名は完全に固定し、常に上書き
- ❌ 「念のため」残すことは禁止
- ✅ Gitの履歴に残っているので、必要なら復元可能

**徹底的なクリーンアップとファイル名の統一がプロジェクトの品質を保ちます。**
