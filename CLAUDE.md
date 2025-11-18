# Claude Code Development Guidelines for New-LLM Project

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

### 基本原則

新しい実装、バグ修正、ファイル更新を行った際は、**必ず古いバージョンを完全に削除**してください。

### 削除対象

#### 1. 古いコードファイル

- ✗ 使われなくなったメソッド・関数
- ✗ コメントアウトされたコード
- ✗ デバッグ用の一時コード
- ✗ 重複した機能を持つコード
- ✗ 古いバージョンのスクリプト

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

## コミットメッセージ

古いファイルを削除した場合、コミットメッセージに明記：

```bash
git commit -m "Fix graph bug and remove old buggy version

- Fixed: Perplexity plot now uses linear scale
- Removed: old_graph.png (had log scale bug)
- Added: new_graph_fixed.png (correct version)
"
```

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
