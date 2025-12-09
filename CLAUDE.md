# Senri Project Guidelines

---

## 🧠 メモリ階層の定義 - 削除禁止

**⚠️ この定義は本プロジェクトのメモリアーキテクチャの基盤です。**

### 3種類の圧縮メモリ

| メモリ名 | 役割 | 更新タイミング | 人間の認知との対応 |
|----------|------|----------------|-------------------|
| **Working Memory** | コンテキスト格納、会話内容管理 | トークンごとに更新 | 短期記憶・ワーキングメモリ |
| **Index Memory** | 粗い知識の格納、Detail Memoryへの索引 | 事前に構築、推論時は不変 | 意味記憶の索引 |
| **Detail Memory** | 細かい知識の格納 | 事前に構築、推論時は不変 | 意味記憶の詳細 |

### 各メモリの詳細

#### Working Memory（作業記憶）

- **機能**: 現在のセッション・会話のコンテキストを保持
- **対応**: 既存LLMのKVキャッシュに相当
- **更新**: トークンが増えるたびに更新される
- **特徴**:
  - セッションごとにリセット
  - Linear Attention形式で圧縮
  - 直近の文脈を高速参照

#### Index Memory（索引記憶）

- **機能**: 粗い知識を格納し、Detail Memoryへのルーティングを担当
- **更新**: 推論時は更新されない（事前構築）
- **特徴**:
  - クエリに対して「どのDetail Memoryを参照すべきか」を判定
  - 対応不能と判断した場合のみDetail Memoryを検索
  - 検索方式: memory_norm方式（Landmark = Σσ(k)）を採用

#### Detail Memory（詳細記憶）

- **機能**: 細かい知識を格納
- **更新**: 推論時は更新されない（事前構築）
- **特徴**:
  - Index Memoryから選択されたときのみアクセス
  - 複数のDetail Memoryが存在（Multi-Memory構成）
  - 各メモリは独立した知識ドメインを担当可能

### 処理フロー

```
Query
  ↓
Working Memory 参照（直近コンテキスト）
  ↓
Index Memory 参照（粗い知識）
  ↓
対応可能? ──Yes──→ 回答生成
  │
  No
  ↓
Top-K Detail Memory 選択
  ↓
Detail Memory 検索
  ↓
回答生成
```

### 既存実装との対応

| メモリ名 | 現在の実装 | 備考 |
|----------|-----------|------|
| Working Memory | InfiniLayer のメモリ | トークンごと更新 |
| Index Memory | 未実装 | memory_norm方式Landmarkで選択判定 |
| Detail Memory | MultiMemoryLayer のメモリ群 | memory_norm方式で検索 |

### memory_norm方式（Landmark計算）

メモリ選択のためのLandmark計算方式：

```
Landmark = memory_norm = Σσ(k)

検索スコア = σ(Q) @ Landmark
```

**特徴**:
- **追加パラメータなし**: メモリ書き込み時の副産物を活用
- **シンプル**: 特別な計算なしでメモリの「重要度」を表現
- **効率的**: 訓練時間のオーバーヘッドなし

---

## 🎯 レイヤーベースアーキテクチャ

**レイヤーを組み合わせてモデルを構築する柔軟な設計。**

### コンセプト

```
従来: 複数の固定モデルクラス

新設計: 1つの汎用モデル + 3つのレイヤータイプ
  TransformerLM + [PythiaLayer, InfiniLayer, MultiMemoryLayer]
```

### アーキテクチャ

```
TransformerLM:
  Token Embedding (512-dim)
         ↓
  Layer 0, 1, ..., N-1 (任意のレイヤータイプ)
         ↓
  Final LayerNorm
         ↓
  LM Head (512 → vocab)
```

### レイヤータイプ

| レイヤー | 説明 |
|----------|------|
| `PythiaLayer` | 標準Pythia (RoPE + Softmax Attention) |
| `InfiniLayer` | Infini-Attention (Memory + Linear Attention, NoPE) |
| `MultiMemoryLayer` | 複数独立メモリ + Attention-based選択 |

---

## 🏭 モデル作成

### ModelConfig（推奨）

```python
from src.config import SenriModelConfig, PythiaModelConfig, default_senri_layers

# Senriモデル（デフォルト: 1 Senri + 5 Pythia）
config = SenriModelConfig()
model = config.create_model()

# カスタムSenri構成
config = SenriModelConfig(
    layers=default_senri_layers(
        num_senri=2,
        num_pythia=4,
        use_multi_memory=True,
        num_memories=8,
    )
)
model = config.create_model()

# Pythiaモデル（ベースライン）
config = PythiaModelConfig()
model = config.create_model()
```

### LayerConfigリストを使用

```python
from src.config import SenriLayerConfig, PythiaLayerConfig
from src.models import create_model

# カスタム構成
layers = [
    SenriLayerConfig(use_multi_memory=True, num_memories=8),
    PythiaLayerConfig(),
    PythiaLayerConfig(),
    PythiaLayerConfig(),
]
model = create_model(layers)
```

### 直接レイヤー構築（上級者向け）

```python
from src.models import TransformerLM
from src.models.layers import InfiniLayer, PythiaLayer, MultiMemoryLayer

# 2層Infini + 4層Pythia
layers = [
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    *[PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048) for _ in range(4)]
]
model = TransformerLM(layers=layers, vocab_size=52000, hidden_size=512)
```

### 利用可能なオプション

| オプション | 対象 | デフォルト | 説明 |
|------------|------|------------|------|
| `use_delta_rule` | 全memory系 | `True` | Delta Rule使用 |
| `num_memories` | multi_memory | `4` | メモリ数 |
| `num_memory_banks` | infini | `1` | メモリバンク数 |
| `segments_per_bank` | infini | `4` | バンクあたりセグメント数 |

### 訓練設定のデフォルト値

| 設定 | デフォルト | 説明 |
|------|------------|------|
| `early_stopping_patience` | **1** | 改善なしで訓練終了までのエポック数 |
| `gradient_clip` | 1.0 | 勾配クリッピング |
| `lr` | 1e-4 | 学習率 |

**注意**: patienceのデフォルト値は1。過学習防止のため早めに終了する。

---

## 💾 メモリ状態の保存・転送

```python
import torch
from src.config import SenriModelConfig

# ===== PC A =====
config = SenriModelConfig()
model = config.create_model()
model.reset_memory()

# テキスト処理でメモリを蓄積
for batch in data_loader:
    _ = model(batch, update_memory=True)

# メモリ状態を保存
state = model.get_memory_state()
torch.save(state, "memory.pt")

# ===== PC B =====
state = torch.load("memory.pt")
config = SenriModelConfig()
model = config.create_model()
model.set_memory_state(state)

# メモリが引き継がれた状態で推論
output = model(input_ids)
```

### メモリサイズ

| モデル | サイズ |
|--------|--------|
| Infini (1 bank) | ~135 KB |
| Multi-Memory (4) | ~540 KB |

---

## 📁 ファイル構造

```
src/
├── config/
│   ├── __init__.py          # Public exports
│   ├── constants.py         # OPEN_CALM_TOKENIZER, OPEN_CALM_VOCAB_SIZE
│   ├── layers/              # レイヤー設定
│   │   ├── base.py          # BaseLayerConfig
│   │   ├── pythia.py        # PythiaLayerConfig
│   │   └── senri.py         # SenriLayerConfig
│   ├── models/              # モデル設定
│   │   ├── base.py          # BaseModelConfig
│   │   ├── pythia.py        # PythiaModelConfig
│   │   └── senri.py         # SenriModelConfig
│   └── experiments/         # 実験設定
│       └── base.py          # ExperimentConfig
├── models/
│   ├── __init__.py          # create_model() + exports
│   ├── layers/              # レイヤーパッケージ
│   │   ├── base.py          # BaseLayer 基底クラス
│   │   ├── pythia.py        # PythiaLayer (RoPE + Softmax)
│   │   ├── infini.py        # InfiniLayer (Memory + Linear)
│   │   └── multi_memory.py  # MultiMemoryLayer
│   ├── model.py             # TransformerLM（汎用モデル）
│   ├── base_components.py   # PythiaMLP, init_weights
│   ├── memory_utils.py      # Linear attention utilities
│   └── position_encoding.py # RoPE
└── utils/
    ├── tokenizer_utils.py   # get_tokenizer, get_open_calm_tokenizer
    ├── training.py          # 訓練ユーティリティ
    └── evaluation.py    # 評価ユーティリティ
```

---

## 🧪 実験スクリプト

```bash
# Context Separation Training（Reversal Curse対策）
python3 scripts/experiment_context_reasoning.py
```

---

## 📊 Reversal Curse 評価

| 指標 | 定義 | 解釈 |
|------|------|------|
| Forward PPL | 順方向文のPPL | 訓練データに含まれるため低い |
| Backward PPL | 逆方向文のPPL | 訓練データに含まれないため高い |
| Reversal Gap | Backward - Forward | 0に近いほど良い |

---

## 🚨 CRITICAL: コード品質

### 後方互換性コード禁止

**古い機能を残すことは厳禁。後方互換性を意識したコードは絶対に書かない。**

### ハードコード厳禁

**全ての値はconfigから読み込む。**

### ランダムデータ使用禁止

**実験でランダムデータ（torch.randint等）を使用することは絶対に禁止。**
必ず実データ（Pile）を使用すること。

### 訓練-評価一貫性（Training-Evaluation Consistency）

**訓練時と評価時の条件は必ず揃える。**

```python
# ❌ 悪い例: 離散的なthreshold（訓練時にバイアス発生）
# 訓練: threshold=0.5でスキップ、出力位置のみloss
# → 高確信度トークンのみでloss計算 → 異常に低いPPL
def train():
    output_mask = gate_prob > 0.5  # 離散的判定
    loss = (losses * output_mask).sum() / output_mask.sum()

# ✅ 良い例: 連続的重み（全トークンが学習に寄与）
# gate_probを重みとして使用 → 勾配が常に流れる
def train():
    weighted_loss = (losses * gate_prob).sum() / gate_prob.sum()

def evaluate():
    weighted_ppl = exp(sum(gate_prob * log_loss) / sum(gate_prob))
```

**原則**:
1. 連続的な重みを使用（離散的なthresholdは学習バイアスを生む）
2. 全トークンが学習に寄与するようにする
3. 生成時のみthresholdを使用（訓練・評価には使わない）

---

## 🔄 Lagged Cache Training（LCT）方式 - 削除禁止

**⚠️ この方式は本プロジェクトの核心的手法です。絶対に削除・変更しないこと。**

### 概要

「前のイテレーションの出力を、今回のイテレーションの入力として使用する」訓練方式。

再帰的依存関係を持つモデル（RNN的構造）を並列処理可能にする画期的な手法。

### 問題

再帰的モデルの訓練は本来シーケンシャル:
```
通常の再帰訓練（遅い）:
  for t in range(seq_len):
      h_t = model(h_{t-1}, x_t)  # 1トークンずつ処理
      # → O(seq_len) 回のforward pass
```

### 解決策: Lagged Cache Training

```python
# 基本アルゴリズム
hidden_caches = {}  # バッチごとのキャッシュ

for epoch in epochs:
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        # 1. キャッシュ初期化（初回のみ）
        if batch_idx not in hidden_caches:
            with torch.no_grad():
                hidden_caches[batch_idx] = model.forward(input_ids).detach()

        # 2. 前イテレーションのキャッシュを入力として使用
        prev_cache = hidden_caches[batch_idx]

        # 3. 並列処理で全シーケンスを一度に計算
        # 位置t の入力 = prev_cache[t-1]（前イテレーションでの位置t-1の出力）
        output, new_hidden = model.forward_with_cache(input_ids, prev_cache)
        loss = compute_loss(output, labels)

        # 4. キャッシュを更新（次イテレーション用）
        hidden_caches[batch_idx] = new_hidden.detach()

        loss.backward()
        optimizer.step()
```

### なぜ動作するのか

1. **近似の仮定**: イテレーションごとの変化は微小
   - パラメータ更新量が小さければ、h_t の変化も小さい
   - prev_cache ≈ 真のh_{t-1}（1イテレーション遅れ）

2. **キャッシュの更新**: 学習が進むとキャッシュも追従
   - 各イテレーションでnew_hiddenを保存
   - 次イテレーションでそれを入力として使用
   - → 徐々に真の再帰動作に収束

3. **計算効率**:
   - シーケンシャル: O(seq_len) forward passes per batch
   - LCT: O(1) forward pass per batch
   - 速度向上: ~20倍（seq_len=128の場合）

### 適用可能なモデル

- 任意のRNN的再帰構造（h_{t-1} → h_t）
- Context-Pythia（context_{t-1} → context_t）
- 前の出力を次の入力として使う任意のモデル

### 性能比較（実測値）

| 方式 | 時間/epoch | 速度比 |
|------|-----------|--------|
| シーケンシャル | 450s | 1x |
| LCT | 20s | 22x |

---

## ⚠️ 過去のバグと教訓

### 1. Infini-Attention メモリ勾配バグ

```python
# ❌ バグ: メモリ更新でグラフが残り、二重backwardエラー
self.memory = self.memory + memory_update

# ✅ 修正: detach()でグラフを切断
self.memory = (self.memory + memory_update).detach()
```

### 2. PPL異常値の診断基準

| PPL | 状態 | 対処 |
|-----|------|------|
| < 5 | **異常** - データリーク/因果マスクバグ | コード点検必須 |
| 5-30 | **疑わしい** - 過学習の可能性 | データ量・分割を確認 |
| 30-100 | 正常（小規模データ） | - |
| 100-500 | 正常（スクラッチ訓練） | - |
| > 1000 | 学習不足 | epoch増加/lr調整 |

### 3. Linear Attentionのhead_dim次元数問題（重要）

**head_dimが小さいとLinear Attentionが機能しない。**

```python
# ❌ 問題: head_dim=64（hidden_size=512 / num_heads=8）
# → 64次元空間では異なるキーベクトルが直交しやすい
# → σ(Q) @ σ(K)^T ≈ 0 → メモリから何も取り出せない

# ✅ 解決: シングルヘッド（memory_head_dim=hidden_size=512）
# InfiniLayerは自動的にシングルヘッドを使用
```

**教訓**:
1. Linear Attentionにはhead_dim >= 256が必要
2. 可能ならシングルヘッド（head_dim=hidden_size）を使用
3. alphaが小さいまま → head_dimを疑う

### 4. PPL評価方法によるPPL異常値

**セグメント分割評価で異常に高いPPL（10,000+）が出る原因。**

```python
# ❌ 問題: 各セグメントが独立（コンテキストなし）
for start in range(0, seq_len, segment_length):
    segment = tokens[start:end]
    # → 「文書の途中」を「文書の先頭」として扱うため高PPL

# ✅ 正しい評価: Sliding Window方式
stride = 512
for start in range(0, seq_len - 1, stride):
    input_ids = tokens[start:start+2048]
    labels = input_ids.clone()
    labels[0, :stride] = -100  # コンテキスト部分はloss計算から除外
```

| 評価方法 | PPL |
|----------|-----|
| Sliding window (stride=512) | **40.96** ✓ |
| Segment-based | 14,204 ❌ |

### 5. 短い文のパディング問題（Reversal Curse評価）

**短い文をEOSでパディングすると、訓練と評価の分布が乖離する。**

```python
# ❌ 問題: 短い文をseq_lengthまでEOSでパディング
sentence = "Paris is the capital of France"  # 6トークン
tokens = tokenize(sentence) + [EOS] * 122     # 128トークンにパディング
# → モデルは主にEOS→EOSを学習（94%がEOS）
# → 評価時（6トークンのみ）でPPLが異常に高くなる

# ✅ 正しい方法: 複数の短い文を連結
all_sentences = "Paris is the capital of France EOS Tokyo is the capital of Japan EOS ..."
# → 文の内容が連続して現れる（EOSは12%程度、区切りとしてのみ）
# → 訓練と評価の分布が一致
```

**症状**: Forward PPL > Backward PPL（通常は逆）
**原因**: 訓練データの94%がEOSトークンで、実際の文内容をほとんど学習していない
**対策**: パディングせず、文を連結してスライディングウィンドウでサンプル作成

---

## 🔧 Pretrained LLMへのInfini-Attention導入（失敗）

**⚠️ Layer置き換え方式は全て失敗。**

| 方式 | 結果 |
|------|------|
| Layer 0置き換え | ❌ RoPE損失、PPL大幅劣化 |
| 蒸留+Fine-tune | ❌ PPL 44→1237（28倍劣化） |
| Parallel Adapter | ❌ alphaが学習されない |

**推奨**: スクラッチ訓練（本プロジェクトのアプローチ）

詳細は `docs/experiments/2025-12-06_distill_finetune_failure.md` を参照。

---

## 🔧 Continuous LM（失敗）

**仮説: トークン化による離散化で情報が失われている → 失敗**

### コンセプト

```
通常LM (Discrete):
  h_t → LM Head → token → Embedding → x_{t+1}

Continuous LM:
  h_t → proj → x_{t+1}   (離散化をスキップ)
```

前のトークン処理時の最終隠れ状態を、直接次の入力として使用する。

### 実験結果（2025-12-07）

| モード | Val PPL | Forward PPL | Backward PPL | Gap |
|--------|---------|-------------|--------------|-----|
| discrete | **487.2** | 46534.0 | 11938.4 | -34595.5 |
| continuous | 2031.6 | **413.5** | **1206.2** | **+792.7** |
| continuous_extra | 2573.9 | 1255.3 | 1267.1 | +11.8 |
| continuous_combined | 35062.4 | 32333.1 | 30004.4 | -2328.6 |

### 結論

- **Reversal Curse改善**: Forward/Backward PPLのGapは改善（-34595 → +792）
- **Val PPL大幅悪化**: 487 → 2031（4倍悪化）
- **トレードオフが悪い**: 言語モデリング性能を犠牲にしすぎ

### 失敗の原因

1. **トークン情報の喪失**: 埋め込みベクトルを完全に捨てている
   - 埋め込みには意味的・文法的特徴が含まれる
   - h_{t-1}だけでは現在のトークン情報が失われる

2. **残差加算も効果なし**: `embed + proj(h_{t-1})` も試したが改善せず

**注意**: 実装は削除済み。記録のみ残す。

---

## 🔧 2-Pass Processing による発見（実験記録）

**意図せず発見: 2回処理がReversal Curseを改善**

### 概要

SelectiveOutputLM実装中に、Transformerを2回通す方式がReversal Curseを改善することを発見。

```
1回処理: token → embed → layers → h1 → output
2回処理: token → embed → layers → h1 → proj → layers → h2 → output
```

### 結果

| Model | Val PPL | Forward PPL | Gap |
|-------|---------|-------------|-----|
| 1回処理 | **484.6** | 12868.4 | -1799.1 |
| 2回処理 | 516.8 | **9576.1** | **+1114.1** |

- Val PPLは悪化するが、Reversal CurseのGapが大幅改善
- なぜ2回処理で記憶が定着するのかは未解明

詳細は `docs/experiments/2025-12-07_two_pass_discovery.md` を参照。

**注意**: この発見はメンテナンス性のためコード削除済み。記録のみ残す。

---

## 🧠 Reversal Curse 汎化性能仮説 - 削除禁止

**⚠️ この仮説と実験設計は本プロジェクトの核心です。絶対に削除・変更しないこと。**

### 仮説

Reversal Curseの真の問題は「逆方向を推論できない」ことではなく、**汎化性能の低さ**である。

```
問題の本質:
  "Tom is Alice's parent" と "Alice is Tom's children" を学習しても、
  別のペア "Bob is Jack's parent" から "Who is Bob's children?" に答えられない。

  → 逆方向の「パターン」を学習できていない
  → 個別の事実を丸暗記しているだけ
```

### 実験設計 - 削除禁止

**データ構成**:
- **パターン学習ペア**: 順方向・逆方向の両方を学習（例: Tom-Alice）
- **Valペア**: 順方向のみ学習（例: Bob-Jack）→ **逆方向で評価**

```
目的: パターンペアで学んだ「逆方向推論」をValペアに汎化できるか？
```

### 訓練データ詳細

#### ■ Baseline（従来方式）

**パターン学習（全文学習）**:
```
初期context: 任意
学習対象: "Tom is Alice's parent. Who is Alice's parent? Tom"
学習対象: "Alice is Tom's children. Who is Tom's children? Alice"
```

**Valペア（順方向のみ）**:
```
初期context: 任意
学習対象: "Bob is Jack's parent. Who is Jack's parent? Bob"
```

→ 従来のLLMは全文を丸暗記。パターンの抽象化ができない。

#### ■ Modified（知識分離訓練）

**パターン学習（コンテキスト分離）**:
```
初期context: "Tom is Alice's parent."
学習対象: "Who is Alice's parent? Tom"

初期context: "Alice is Tom's children."
学習対象: "Who is Tom's children? Alice"
```

→ 初期コンテキストをloss計算から除外。丸暗記を防ぎ、推論パターンを学習させる。

**Valペア（順方向のみ、Baselineと同一）**:
```
初期context: 任意
学習対象: "Bob is Jack's parent. Who is Jack's parent? Bob"
```

### 評価方法 - 削除禁止

**公平性**: Baseline/Modified両方とも**全く同じデータ**で評価

**評価項目**:
1. **Pile PPL**: 一般的な言語能力の維持
2. **Reversal Curse評価**: `"Who is Bob's children?"` → 正解は `Jack`

**成功の指標**:
- Modified が Baseline より低いPPLで `"Who is Bob's children?"` に回答できる
- これは「パターン学習ペアで学んだ逆方向推論がValペアに汎化した」ことを意味する

### なぜこの設計が重要か

1. **汎化の直接テスト**: Valペアでは逆方向を**一度も学習していない**
2. **公平な比較**: Valペアの訓練データはBaseline/Modifiedで**完全に同一**
3. **パターン抽出の検証**: Modifiedがコンテキスト分離により抽象パターンを学習できるか

---

## 🇯🇵 Senri - 日本語LLM

**Senri（千里）: OpenCALMトークナイザーを使用した日本語LLM。**

### トークナイザー

| 特徴 | 値 |
|------|-----|
| モデル名 | cyberagent/open-calm-small |
| 語彙サイズ | 52,000 |
| UNKトークン | なし（byte_fallback対応） |
| 日本語 | 完全対応 |
| 英語 | 完全対応（AI, API, GPU等） |
| 絵文字 | 完全対応 |

### 使用方法

```python
from src.utils.tokenizer_utils import get_open_calm_tokenizer

tokenizer = get_open_calm_tokenizer()
# UNKトークンなし！
```

### 採用理由

1. **UNKなし**: byte_fallback対応で任意の入力を処理可能
2. **日本語特化**: 日本語のトークン効率が高い
3. **英語対応**: 日本人がよく使う英単語（AI, API, GPU等）も完全対応
4. **絵文字対応**: 絵文字もUNKにならない

### 比較（採用時の調査結果）

| トークナイザー | UNKテスト | 語彙サイズ |
|---------------|----------|-----------|
| cyberagent/open-calm-small | **PASS** | 52,000 |
| stockmark/gpt-neox-japanese-1.4b | PASS | 50,000 |
| llm-jp/llm-jp-1.3b-v1.0 | PASS | 50,570 |
| rinna/japanese-gpt2-medium | FAIL | 32,000 |

---

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-09 | **config/リファクタリング**: layers/, models/, experiments/の3サブパッケージに分離。PythiaModelConfig追加 |
| 2025-12-09 | **SenriModelConfig追加**: LayerConfigベースのモデル構築。ファクトリパターン廃止 |
| 2025-12-09 | **Senri命名**: プロジェクト名をSenriに決定 |
| 2025-12-09 | **OpenCALM採用**: 日本語LLM対応。OpenCALMトークナイザーを使用 |
| 2025-12-09 | **HSA方式削除**: ChunkEncoder（双方向エンコーダ）を削除。memory_norm方式に一本化。シンプルさ優先 |
| 2025-12-09 | **HSA vs memory_norm比較実験**: ChunkEncoder方式 vs Σσ(k)方式を比較。HSA=494.4 PPL、memory_norm=497.7 PPL。HSA微改善だがコスト増 |
| 2025-12-09 | **リファクタリング**: experiment_landmark_comparison.py削除、test_pythia_pretrained.pyをtests/へ移動 |
| 2025-12-08 | **patience=1に統一**: Early stoppingのデフォルトpatience値を1に変更。過学習防止 |
| 2025-12-08 | **メモリ階層定義追加**: Working Memory / Index Memory / Detail Memory の3層構成を定義 |
| 2025-12-07 | **CDR訓練追加**: Context-Dependent Reasoning Training - 推論をFFNに、知識を外部コンテキストに分離する訓練手法 |
| 2025-12-07 | **Continuous LM失敗**: 離散化スキップ仮説は失敗。Reversal Curse改善するもVal PPL 4倍悪化。実装削除 |
| 2025-12-07 | **LCT方式追加**: Lagged Cache Training - 再帰的モデルを並列訓練可能にする手法。22倍高速化 |
| 2025-12-07 | **2-Pass発見を記録**: Transformerを2回通すとReversal Curseが改善（コードは削除、記録のみ） |
| 2025-12-07 | **訓練-評価一貫性ポリシー追加**: 訓練時と評価時の条件を揃えることを必須化 |
| 2025-12-06 | **SelectiveOutputLM追加**: 学習可能ゲートによる選択的出力モデル（後に失敗と判明） |
| 2025-12-08 | **HierarchicalLayer削除**: MultiMemoryLayerと実質同等のため削除。3レイヤータイプに整理 |
| 2025-12-06 | **レイヤーベースアーキテクチャに移行**: TransformerLM + レイヤータイプ、コード31%削減 |
| 2025-12-06 | **Layer置き換え方式を削除**: 蒸留+Fine-tune等すべて失敗、スクラッチ訓練に集中 |
| 2025-12-06 | **シングルヘッドメモリ導入**: memory_head_dim=512でLinear Attentionの表現力を最大化 |
| 2025-12-06 | **PPL評価方法の教訓追加**: Sliding window方式が正しい |
| 2025-12-06 | **メモリ転送API追加**: get_memory_state/set_memory_stateで圧縮メモリを別PCに転送可能 |
| 2025-12-06 | **モデルファクトリ追加**: create_model()でシンプルにモデル作成 |
| 2025-12-06 | **Multi-Memory Attention追加**: Attention-based選択で複数メモリを動的混合 |
| 2025-12-05 | **Infini-Pythia実装**: 1層目Infini + RoPE |

---

Last Updated: 2025-12-09
