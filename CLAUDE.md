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

## 🔧 Selective Output LM

**仮説: LLMは即座に出力せず、隠れ状態を追加処理してから出力すべき**

### コンセプト

```
extra_passes=0 (Baseline):
  入力A → Transformer処理 → 即座に次トークン"B"を予測（追加処理なし）

extra_passes=1 (Selective):
  入力A → Transformer処理 → 隠れ状態h1（まだ出力しない）
       → h1を追加処理 → 隠れ状態h2 → 次トークン"B"を予測
```

### extra_passes パラメータ

| 値 | 動作 | 説明 |
|----|------|------|
| 0 | 追加処理なし | 従来のContinuousと同等（即座に出力） |
| 1 | 1回追加処理 | トークン入力後、1回追加でTransformer通過してから出力 |

### 使用方法

```python
from src.models import create_model

# モデル作成
model = create_model("selective")

# 訓練（extra_passes=1）
loss, stats = model.compute_loss(input_ids, labels, use_selective=True)

# 訓練（extra_passes=0、Baseline）
loss, stats = model.compute_loss(input_ids, labels, use_selective=False)

# 生成
output, stats = model.generate(input_ids, max_new_tokens=50, use_selective=True)
```

### 実験スクリプト

```bash
# Selective (extra_passes=1)
python3 scripts/experiment_selective.py

# Baselineとの比較
python3 scripts/experiment_selective.py --models baseline selective

# NoPE（Position Encodingなし）で実験
python3 scripts/experiment_selective.py --models baseline selective --nope
```

---

## 🔧 学習可能ゲートによるSelective Output（失敗）

**⚠️ 動的ゲート方式は複雑すぎて失敗。現在は固定extra_passesに簡素化。**

### 試した方式

| 方式 | 結果 |
|------|------|
| OutputGate + threshold | ❌ carry-over 96.3%、ほぼ学習されない |
| max_skip強制 + threshold | ❌ 収束が不安定、PPL改善せず |
| エントロピーベースgate_loss | ❌ ゲートが適切に学習されない |

### 問題点

1. **ゲート学習の不安定性**: gate_probがthresholdに収束しない
2. **carry-over率の制御困難**: 動的に持ち越し判断すると予測困難
3. **勾配の不連続性**: threshold判定で勾配が途切れる
4. **ターゲットアライメントの複雑さ**: 持ち越し時のターゲット計算が非自明

### 教訓

- 動的な判断より**固定パターン**（extra_passes）がシンプル
- 学習可能ゲートは追加パラメータ（65K）に対して効果が見合わない
- OutputGate、エントロピーベース損失は削除済み

詳細は `docs/experiments/2025-12-07_selective_output_gate_failure.md` を参照。

---

## 📜 変更履歴

| 日付 | 内容 |
|------|------|
| 2025-12-07 | **Selective Output LM再設計**: 隠れ状態の追加処理方式に変更（skip_interval=追加処理回数） |
| 2025-12-07 | **学習可能ゲート失敗を記録**: OutputGate方式は複雑すぎて失敗、固定パターンに簡素化 |
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
