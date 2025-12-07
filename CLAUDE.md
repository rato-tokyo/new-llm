# New-LLM Project Guidelines

---

## 🎯 レイヤーベースアーキテクチャ

**レイヤーを組み合わせてモデルを構築する柔軟な設計。**

### コンセプト

```
従来: 4つの固定モデルクラス
  PythiaModel, InfiniPythiaModel, MultiMemoryPythiaModel, HierarchicalMemoryPythiaModel

新設計: 1つの汎用モデル + 4つのレイヤータイプ
  TransformerLM + [PythiaLayer, InfiniLayer, MultiMemoryLayer, HierarchicalLayer]
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
| `HierarchicalLayer` | 階層的メモリ + 学習可能な展開ゲート |

---

## 🏭 モデル作成

### create_model() ファクトリ

```python
from src.models import create_model

# 基本的な使い方
model = create_model("pythia")       # 標準Pythia（6層）
model = create_model("infini")       # 1層Infini + 5層Pythia
model = create_model("multi_memory") # 1層Multi-Memory + 5層Pythia
model = create_model("hierarchical") # 1層Hierarchical + 5層Pythia

# オプション付き
model = create_model("multi_memory", num_memories=8)
model = create_model("hierarchical", num_memories=4, use_delta_rule=False)
model = create_model("infini", num_memory_banks=2, segments_per_bank=4)
```

### カスタムレイヤー構成

```python
from src.models import TransformerLM
from src.models.layers import InfiniLayer, PythiaLayer, MultiMemoryLayer

# 2層Infini + 4層Pythia
layers = [
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    *[PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048) for _ in range(4)]
]
model = TransformerLM(layers=layers)

# 全層Infini
layers = [InfiniLayer(512, 8, 2048) for _ in range(6)]
model = TransformerLM(layers=layers)

# 混合構成
layers = [
    MultiMemoryLayer(512, 8, 2048, num_memories=4),
    InfiniLayer(512, 8, 2048),
    *[PythiaLayer(512, 8, 2048) for _ in range(4)]
]
model = TransformerLM(layers=layers)
```

### 利用可能なオプション

| オプション | 対象 | デフォルト | 説明 |
|------------|------|------------|------|
| `use_delta_rule` | 全memory系 | `True` | Delta Rule使用 |
| `num_memories` | multi_memory, hierarchical | `4` | メモリ数 |
| `num_memory_banks` | infini | `1` | メモリバンク数 |
| `segments_per_bank` | infini | `4` | バンクあたりセグメント数 |

---

## 💾 メモリ状態の保存・転送

```python
import torch
from src.models import create_model

# ===== PC A =====
model = create_model("infini")
model.reset_memory()

# テキスト処理でメモリを蓄積
for batch in data_loader:
    _ = model(batch, update_memory=True)

# メモリ状態を保存
state = model.get_memory_state()
torch.save(state, "memory.pt")

# ===== PC B =====
state = torch.load("memory.pt")
model = create_model("infini")
model.set_memory_state(state)

# メモリが引き継がれた状態で推論
output = model(input_ids)
```

### メモリサイズ

| モデル | サイズ |
|--------|--------|
| Infini (1 bank) | ~135 KB |
| Multi-Memory (4) | ~540 KB |
| Hierarchical (4) | ~540 KB |

---

## 📁 ファイル構造

```
src/models/
├── __init__.py          # create_model() ファクトリ + exports
├── layers/              # レイヤーパッケージ
│   ├── __init__.py      # exports
│   ├── base.py          # BaseLayer 基底クラス
│   ├── pythia.py        # PythiaLayer (RoPE + Softmax)
│   ├── infini.py        # InfiniLayer (Memory + Linear)
│   ├── multi_memory.py  # MultiMemoryLayer
│   └── hierarchical.py  # HierarchicalLayer
├── model.py             # TransformerLM（汎用モデル）
├── continuous.py        # ContinuousLM（離散化スキップ仮説検証）
├── base_components.py   # PythiaMLP, init_weights
├── memory_utils.py      # elu_plus_one, causal_linear_attention
└── position_encoding.py # RoPE
```

---

## 🧪 統一実験スクリプト

```bash
# 全モデル比較
python3 scripts/experiment.py --models pythia infini multi_memory hierarchical

# Infiniのみ
python3 scripts/experiment.py --models infini

# 設定カスタマイズ
python3 scripts/experiment.py --models infini --samples 10000 --epochs 50 --lr 5e-5
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
# 1. 初期キャッシュを計算（最初のイテレーションのみ）
hidden_cache = model.init_hidden_cache(input_ids)  # [batch, seq_len, hidden]

# 2. 訓練ループ
for epoch in epochs:
    for batch_idx, batch in enumerate(train_loader):
        # 前イテレーションのキャッシュを入力として使用
        prev_cache = hidden_caches[batch_idx]

        # 並列処理で全シーケンスを一度に計算
        # 位置t の入力 = prev_cache[t-1]（前イテレーションでの位置t-1の出力）
        loss, new_hidden = model.forward_with_cache(input_ids, prev_cache)

        # キャッシュを更新（次イテレーション用）
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

- Continuous LM（h_{t-1} → x_t）
- Context-Pythia（context_{t-1} → context_t）
- DProj（prev_proj → new_proj）
- 任意のRNN的再帰構造

### 実装例（ContinuousLM）

```python
# src/models/continuous.py
def forward_with_cache(self, input_ids, prev_hidden_cache):
    # 位置0: 最初のトークンの埋め込み
    first_embed = self.embed_in(input_ids[:, :1])

    # 位置1以降: prev_hidden_cache[t-1] を変換して使用
    rest_input = self.hidden_proj(prev_hidden_cache[:, :-1, :])
    hidden_input = torch.cat([first_embed, rest_input], dim=1)

    # 並列でTransformer処理
    h = self._forward_layers(hidden_input)
    return logits, h  # hを次イテレーション用キャッシュとして返す
```

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

## 🔧 Continuous LM

**仮説: トークン化による離散化で情報が失われている**

### 背景

通常のLMでは、次トークン予測時に離散化が発生する：

```
通常LM (Discrete):
  h_t → LM Head → token → Embedding → x_{t+1}
        ↑                    ↑
        離散化              再埋め込み
        (情報損失)
```

この離散化ステップで情報が失われるのでは？という仮説を検証する。

### Continuous LMのコンセプト

```
Continuous LM:
  h_t → proj → x_{t+1}   (離散化をスキップ、情報保持)
```

前のトークン処理時の最終隠れ状態を、直接次の入力として使用する。

### モード一覧

| モード | 入力方式 | extra_pass | use_h1 | 説明 |
|--------|----------|------------|--------|------|
| discrete | token埋め込み | - | - | 通常のLM（ベースライン） |
| continuous | h_{t-1}を直接使用 | False | - | 離散化スキップ |
| continuous_extra | h_{t-1}を直接使用 | True | False | 1回追加処理、h2のみ使用 |
| continuous_combined | h_{t-1}を直接使用 | True | True | 1回追加処理、h1+h2を使用 |

### 処理フロー

```
discrete (通常LM):
  token_A → embed → layers → h1 → LM Head → "B"予測

continuous:
  h_{t-1} → proj → layers → h1 → LM Head → "B"予測

continuous_extra (extra_pass=True, use_h1=False):
  h_{t-1} → proj → layers → h1 → proj → layers → h2 → LM Head → "B"予測

continuous_combined (extra_pass=True, use_h1=True):
  h_{t-1} → proj → layers → h1 → proj → layers → h2 → combine(h1,h2) → LM Head → "B"予測
```

### 使用方法

```python
from src.models import create_model

# モデル作成
model = create_model("continuous")

# Discrete（通常LM、ベースライン）
loss, stats = model.compute_loss(input_ids, labels, mode="discrete")

# Continuous（離散化スキップ）
loss, stats = model.compute_loss(input_ids, labels, mode="continuous")

# Continuous + 追加処理（h2のみ）
loss, stats = model.compute_loss(input_ids, labels, mode="continuous", extra_pass=True)

# Continuous + 追加処理（h1+h2）
loss, stats = model.compute_loss(input_ids, labels, mode="continuous", extra_pass=True, use_h1=True)
```

### 実験スクリプト

```bash
# 全モード比較
python3 scripts/experiment_continuous.py --models discrete continuous continuous_extra continuous_combined

# NoPE（Position Encodingなし）で実験
python3 scripts/experiment_continuous.py --nope
```

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

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-07 | **LCT方式追加**: Lagged Cache Training - 再帰的モデルを並列訓練可能にする手法。22倍高速化 |
| 2025-12-07 | **Continuous LM実装**: 離散化スキップ仮説の検証。extra_pass/use_h1オプション追加 |
| 2025-12-07 | **2-Pass発見を記録**: Transformerを2回通すとReversal Curseが改善（コードは削除、記録のみ） |
| 2025-12-07 | **訓練-評価一貫性ポリシー追加**: 訓練時と評価時の条件を揃えることを必須化 |
| 2025-12-06 | **SelectiveOutputLM追加**: 学習可能ゲートによる選択的出力モデル（後に失敗と判明） |
| 2025-12-06 | **レイヤーベースアーキテクチャに移行**: TransformerLM + 4レイヤータイプ、コード31%削減 |
| 2025-12-06 | **Layer置き換え方式を削除**: 蒸留+Fine-tune等すべて失敗、スクラッチ訓練に集中 |
| 2025-12-06 | **シングルヘッドメモリ導入**: memory_head_dim=512でLinear Attentionの表現力を最大化 |
| 2025-12-06 | **PPL評価方法の教訓追加**: Sliding window方式が正しい |
| 2025-12-06 | **メモリ転送API追加**: get_memory_state/set_memory_stateで圧縮メモリを別PCに転送可能 |
| 2025-12-06 | **モデルファクトリ追加**: create_model()でシンプルにモデル作成 |
| 2025-12-06 | **Hierarchical Memory追加**: 学習可能な展開判断、Coarse-to-Fine検索 |
| 2025-12-06 | **Multi-Memory Attention追加**: Attention-based選択で複数メモリを動的混合 |
| 2025-12-05 | **Infini-Pythia実装**: 1層目Infini + RoPE |

---

Last Updated: 2025-12-07
