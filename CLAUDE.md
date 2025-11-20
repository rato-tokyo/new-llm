# Claude Code Development Guidelines for New-LLM Project

## 🎯 New Direction: UltraChat Dialogue Training - CRITICAL UPDATE

**⚠️ 方針転換：Colabは使用せず、ローカルGPUで長期訓練**

### 基本方針（2025年更新）

New-LLMは以下の方向性に確定：

1. **データセット**: UltraChat（対話データ）のみ
2. **プラットフォーム**: ローカルGPU（Colab使用せず）
   - 理由：GPU並列化が困難（逐次処理モデルのため）
   - 長期戦を覚悟
3. **柔軟なアーキテクチャ**: layerと文脈ベクトル次元数を頻繁に変更
4. **二段階学習**:
   - Phase 1: 文脈ベクトルの固有点学習
   - Phase 2: 出力トークン学習

### Colab関連ポリシーの廃止

**以下のポリシーは廃止**:
- ❌ 1行curlコマンド完結ポリシー
- ❌ `colab_train_*.sh` スクリプト要件
- ❌ Colab最適化設定

**理由**: New-LLMは逐次処理のため、Colabの並列GPU利用が困難

### 新しい必須実装要件

訓練スクリプト作成時は以下を実装：

1. ✅ 柔軟なアーキテクチャ（layerと文脈次元を簡単に変更可能）
2. ✅ 二段階訓練（Phase 1: 文脈学習、Phase 2: トークン学習）
3. ✅ キャッシュシステム（固有点文脈ベクトルの保存）
4. ✅ 詳細なメトリクスロギング（ステップごとのPPL, Loss, Accuracy）

### 二段階訓練の詳細

**Phase 1: 文脈ベクトル学習**:
```
1. サンプルをトークン配列に変換
2. 一周目：文脈ベクトルをただ保存
3. 二周目以降：前回の文脈ベクトルを教師データとして学習
4. 固有点が定まるまで繰り返し
5. 収束判定：95%以上のトークンが収束 → Phase 2へ
6. 収束しない場合：layerや文脈次元を調整
```

**Phase 2: 出力トークン学習**:
```
1. Phase 1で確定した固有点を固定
2. 期待した出力トークンが出力されるように学習
3. 自分の返答のトークンのみ学習（相手の返答は学習しない）
4. ChatML形式で発言者を識別
```

---

## 🧪 コード修正時の必須テストポリシー - CRITICAL
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

### 保護対象（削除してはいけないもの）

- ✅ **experiments/**: 実験結果ログ（`.log`, `.png`, `.txt`など）
- ✅ **cache/tokenizer/**: トークナイザーキャッシュ（再作成に時間がかかる）
- ✅ **HuggingFaceデータセットキャッシュ**: 再ダウンロード防止

**実験結果は削除対象ではない**: experiments/フォルダで一元管理し、gitにコミット

### チェックリスト

コミット前に必ず確認：

- [ ] コメントアウトされたコードは削除したか？
- [ ] 未使用のメソッド・関数は削除したか？
- [ ] 古いバージョンのファイルは削除したか？
- [ ] 未使用のインポート文は削除したか？
- [ ] **実験結果（experiments/）は保護したか？**

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

## 🔬 CVFPT実験ガイドライン - CRITICAL

**CVFPT (Context Vector Fixed Point Training) 実験時の注意事項**

### 正しいスクリプトの使用

**✅ CVFPT実験には `scripts/train_repetition.py` を使用**

```bash
# ✅ 正しい - トークンIDを直接使用
python3 scripts/train_repetition.py \
    --max-stage 1 \
    --epochs-per-stage 10 \
    --repetitions 10 \
    --device cpu
```

**❌ `train.py` は使わない** - WikiText訓練用（トークナイゼーションが必要）

### 重要な違い

| スクリプト | 用途 | トークナイゼーション |
|----------|------|------------------|
| `train.py` | WikiText訓練 | ✅ 必要（全データセット） |
| **`scripts/train_repetition.py`** | **CVFPT実験** | **❌ 不要（トークンID直接使用）** |

### キャッシュの保護

**`./cache/` ディレクトリは絶対に削除しない**

- ✅ トークナイザーキャッシュ (`tokenizer.json`)
- ✅ HuggingFace Datasetsキャッシュ
- ✅ モデルチェックポイント

**理由**: 再実験時に毎回トークナイゼーションする無駄を防ぐ（数分→0.1秒）

---

## ⚠️ グローバルアトラクター問題 - CRITICAL

**退化解：すべてのトークンが同一の固定点に収束する問題**

### 問題の症状

- ✗ 異なるトークン間のL2距離が異常に小さい（0.000002など）
- ✗ コサイン類似度が異常に高い（0.999以上）
- ✗ 収束ステップが異常に少ない（1ステップ）
- ✗ すべてのトークンが同じ文脈ベクトルに収束

### 根本原因

**Simple Overwrite Updater** が問題：

```python
# ❌ Simple Updater - 以前の文脈を無視
context_new = tanh(W @ hidden)

# 問題点:
# 1. 以前の context を完全に無視
# 2. LayerNorm + Clipping と組み合わさると全トークンが同一点に収束
# 3. トークン固有の情報が失われる
```

### 解決策：Gated Context Updater（必須）

**✅ Gated Updater - LSTM型の更新（標準設定）**

```python
# ✅ Gated Updater - 以前の文脈を保持
context_delta = tanh(W_delta @ hidden)
forget_gate = sigmoid(W_forget @ hidden)
input_gate = sigmoid(W_input @ hidden)
context_new = forget_gate * context + input_gate * context_delta
```

**`src/utils/config.py` での設定**:

```python
context_update_strategy = "gated"  # DEFAULT（変更禁止）
```

### 診断スクリプト

グローバルアトラクター問題を検出：

```bash
# 異なるトークン間の距離をチェック
python3 scripts/check_global_attractor.py

# 期待される結果:
# - Gated Updater: 平均L2距離 > 2.0（トークン固有）
# - Simple Updater: 平均L2距離 < 0.001（グローバルアトラクター）
```

### チェックリスト

CVFPT実験前に必ず確認：

- [ ] `config.context_update_strategy = "gated"` を確認
- [ ] 古い simple updater のチェックポイントを削除
- [ ] 実験結果が「良すぎる」場合は退化解を疑う（cosine > 0.999, L2 < 0.001）

---

## 📊 CVFPT分析結果の解釈

**繰り返し訓練（Fixed-Point）vs 単一パス（Single-Pass）の違い**

### 実験結果（Gated Updater使用時）

| メトリクス | 値 | 解釈 |
|----------|---|------|
| **L2距離** | 3.69 | ノルム（~16）の23% |
| **コサイン類似度** | 0.973 | 方向が97.3%一致 |
| **角度差** | 13.2° | 方向のずれは小さい |
| **ノルム差** | 0.03% | ほぼ同じ大きさ |

### 分解分析

**L2距離の内訳**:
- **方向成分**: 100%（主要な違い）
- **大きさ成分**: 0%（ほぼ同じ）

### 実用的な意味

1. **方向は高精度** - 単一パスでも97.3%方向が一致
2. **大きさも同等** - ノルムの差はわずか0.03%
3. **主な違い** - 次元ごとの微調整（refinement）

**結論**: 単一パスで「本質（essence）」を捉え、繰り返し訓練で「精緻化（refinement）」が進む

---

## まとめ

**鉄則**:
- ✅ 新しいバージョンを作ったら、古いバージョンは即座に削除
- ✅ ファイル名は完全に固定し、常に上書き
- ❌ 「念のため」残すことは禁止
- ✅ Gitの履歴に残っているので、必要なら復元可能
- ✅ **CVFPT実験は `scripts/train_repetition.py` を使用**
- ✅ **Gated Context Updater が標準（変更禁止）**
- ✅ **`./cache/` は保護（削除禁止）**

**徹底的なクリーンアップとファイル名の統一がプロジェクトの品質を保ちます。**
