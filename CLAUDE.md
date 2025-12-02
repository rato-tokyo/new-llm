# New-LLM Project Guidelines

## 🎯 カスケード連結方式（Cascade Context）採用決定 (2025-12-02)

**1層固定アーキテクチャにカスケード連結方式を採用。複数レイヤーは不要。**

### 決定の背景

実験結果より、カスケード連結方式が最良の結果を達成：

| 構成 | Val PPL | Val Acc | 備考 |
|------|---------|---------|------|
| **Cascade (500×2=1000)** | **111.9** | **25.6%** | **最良** |
| C1T1-500 | 127.2 | 24.7% | 標準構成 |
| C2T2-500 | 132.2 | 24.4% | 2層だが悪化 |
| C1T1-1000 | 134.0 | 23.6% | context_dim増加は非効率 |

### カスケード方式の特徴（Cache-Direct方式、可変ブロック数対応）

**Cache-Direct方式**: Phase 1で得られたキャッシュをそのままPhase 2で使用。再計算不要。

**可変ブロック数対応**: ContextBlockの数は1, 2, 3, ... と柔軟に指定可能。

```
N個のContextBlockを順次学習:

Phase 1[0]: ContextBlock[0] を全データで学習
  → 入力: ゼロベクトル
  → context[0][i] キャッシュ取得

Phase 1[1]: ContextBlock[1] を学習
  → 入力: context[0][i-1]（前のトークン位置のcontext[0]）
  → context[1][i] キャッシュ取得

Phase 1[N-1]: ContextBlock[N-1] を学習
  → 入力: context[N-2][i-1]
  → context[N-1][i] キャッシュ取得

Phase 2: TokenBlock 学習（順伝搬なし）
  → 入力: concat(context[0][i-1], ..., context[N-1][i-1])
  → 予測: token[i]
```

**重要**: 次トークン予測では位置`i`を予測するのに位置`i-1`までのコンテキストを使用。
Phase 1で得たキャッシュを1つシフトして使用するだけで、順伝搬は不要。

**根拠**: 以前の実験で、0ベクトルから順伝搬した出力とキャッシュの差は無視可能と確認済み。

### なぜカスケード方式が良いのか

1. **データ活用効率**: 全データで全ContextBlockを学習
2. **cd=500の効率性維持**: 各ブロックで92%収束を達成
3. **連結による表現力**: N×500次元の表現力を獲得しつつ、各ブロックの効率性を維持
4. **柔軟性**: ブロック数を増やすことで表現力を拡張可能

### 実験の実行

```bash
# Colab（GPU）: 本格実験（デフォルト: 2ブロック）
python3 scripts/experiment_cascade_context.py -s 2000

# ブロック数指定（1, 2, 3, ...）
python3 scripts/experiment_cascade_context.py -s 2000 -n 1  # 1ブロック（カスケードなし）
python3 scripts/experiment_cascade_context.py -s 2000 -n 2  # 2ブロック（デフォルト）
python3 scripts/experiment_cascade_context.py -s 2000 -n 3  # 3ブロック

# context_dim指定（各ContextBlockの次元）
python3 scripts/experiment_cascade_context.py -s 2000 -c 500 -n 2  # 500×2=1000次元
python3 scripts/experiment_cascade_context.py -s 2000 -c 300 -n 3  # 300×3=900次元
```

---

## 🎯 OACDアルゴリズム採用 (2025-12-01)

**Phase 1ではOACD (Origin-Anchored Centroid Dispersion) アルゴリズムを採用。**

### OACDの特徴

```python
def oacd_loss(contexts, centroid_weight=0.1):
    # Term 1: 重心からの分散を最大化
    dispersion_loss = -||X - mean(X)|| / n

    # Term 2: 重心を原点に引き寄せる
    centroid_loss = ||mean(X)||²

    return dispersion_loss + centroid_weight * centroid_loss
```

**特徴**:
- 重心を原点に固定することで、安定した平衡点を実現
- 「自己平衡」効果を維持（相対的目標）
- シンプルな損失関数で高いEffective Rank（80%+）を達成

---

## 🚨 1層固定アーキテクチャ (2025-12-02)

**カスケード連結方式により、複数レイヤーは不要。**

```python
# 各ブロック1層固定
ContextBlock: 1層
TokenBlock: 1層

# カスケード連結で表現力を確保（可変ブロック数対応）
combined_context = concat(context[0], context[1], ..., context[N-1])  # cd=context_dim×N
```

**理由**:
- C2T2（2層）がC1T1（1層）より**悪化**した実験結果
- カスケード連結で十分な表現力を確保
- ブロック数を増やすことで表現力を拡張可能
- コードの大幅な簡素化

---

## 💻 ローカル実験の注意事項 - CPU環境 (2025-12-01)

**ローカル環境（Mac/CPU）では処理が遅いため、サンプル数を最小限に抑える。**

```bash
# ローカル実験（CPU）: 2-5サンプルで十分
python3 scripts/experiment_cascade_context.py -s 2

# Colab（GPU）: 2000サンプルで本格実験
python3 scripts/experiment_cascade_context.py -s 2000
```

---

## 🚨 CPU/GPUテンソル管理 - 重要教訓 (2025-12-01)

**大規模データ（2000サンプル以上）でOOMを防ぐため、テンソルのデバイス管理を徹底。**

### 修正パターン

```python
# ❌ 修正前: CPUテンソルをそのまま使用
batch_contexts = previous_contexts[start_idx:end_idx].detach()

# ✅ 修正後: 明示的にGPU転送
batch_contexts = previous_contexts[start_idx:end_idx].detach().to(self.device)
```

### チェックリスト（OOM対策コード変更時）

- [ ] CPUに保持するテンソルを特定
- [ ] GPU演算に渡す前に`.to(self.device)`を追加
- [ ] ループ内のすべてのテンソル転送を確認
- [ ] `torch.cat`や演算の入力デバイスを統一

---

## ⚠️ COLAB環境リセット対策 (2025-11-29)

**Colabは頻繁に環境がリセットされるため、以下のファイルが消失する可能性がある。**

### 自動生成されるファイル

| ファイル | 用途 | 自動生成元 |
|----------|------|----------|
| `./data/example_val.txt` | 検証データ | `MemoryDataProvider._generate_val_file()` |
| `./cache/ultrachat_*samples_full.pt` | 訓練データキャッシュ | `MemoryDataProvider._load_train_data()` |

### Colabでの推奨手順

```bash
# 1. リポジトリ更新
!cd /content/new-llm && git pull

# 2. 実験実行
!cd /content/new-llm && python3 scripts/experiment_cascade_context.py -s 2000
```

---

## 🔧 開発環境のLint/Type Check (2025-11-29)

**pyenv環境ではruffやmypyを直接実行できないため、`python3 -m` で実行する。**

```bash
# Lint (ruff)
python3 -m ruff check src/

# Type check (mypy)
python3 -m mypy src/ --ignore-missing-imports

# 実験スクリプト
python3 -m ruff check scripts/experiment_cascade_context.py
python3 -m mypy scripts/experiment_cascade_context.py --ignore-missing-imports
```

---

## 🚨 CRITICAL: 後方互換性コード禁止 (2025-11-29)

**古い機能を残すことは厳禁。後方互換性を意識したコードは絶対に書かない。**

### 禁止事項

1. **オプション引数での分岐禁止**
2. **古いメソッドの残存禁止**
3. **「念のため」で残さない**

---

## 🧊 EMBEDDING FREEZE ADOPTED - Embedding凍結採用 (2025-11-27)

**Phase 2でEmbedding凍結を標準採用。**

| 指標 | Embedding学習 | Embedding凍結 | 改善率 |
|------|--------------|--------------|--------|
| Val PPL | 1189.15 | **334.31** | **-71.9%** |
| Val Acc | 11.58% | **18.88%** | **+63.0%** |

---

## 🔗 WEIGHT TYING ADOPTED - 重み共有採用 (2025-11-27)

**Weight Tyingを標準採用。パラメータ数を約38M削減。**

| 項目 | Without Weight Tying | With Weight Tying |
|------|---------------------|-------------------|
| 全体パラメータ | 91.43M | **52.78M** (-42%) |
| Output Head | 38.65M | **0** (共有) |

---

## 📊 MANDATORY: 数値報告ルール

### 絶対遵守: すべての実験結果は具体的な数値で報告する

**必須報告項目**:
- ✅ **収束率**: 具体的なパーセンテージ (例: 92%)
- ✅ Effective Rank: **実数値/総次元数とパーセンテージ** (例: 736/1000 = 73.6%)
- ✅ Val PPL: **実数値** (例: 111.9)
- ✅ Val Acc: **実数値** (例: 25.6%)

---

## 📐 アーキテクチャ仕様

### Core Components（1層固定）

**1. ContextLayer / TokenLayer**
- ContextLayer: 文脈処理専用（単一レイヤー）
- TokenLayer: トークン処理専用（単一レイヤー）

**2. ContextBlock / TokenBlock**
- ContextBlock: 1層固定、Phase 1で学習、Phase 2でfreeze
- TokenBlock: 1層固定、Phase 2で学習

**3. CascadeContextLLM（実験用モデル）**
- ContextBlock[0..N-1]（カスケード連結、可変ブロック数対応）
- TokenBlock（連結されたcontext入力）
- Token Embedding: GPT-2 pretrained (768-dim, frozen)
- Weight Tying: token_output shares weights with token_embedding

### Phase 1: 多様性学習（OACD）

- **学習対象**: ContextBlockのみ
- **損失**: OACD（多様性損失）

### Phase 2: トークン予測

- **ContextBlock**: frozen（重み固定）
- **TokenBlock**: 学習
- **損失**: CrossEntropy（次トークン予測）

---

## Code Quality Standards

### Principles

1. **No Hardcoding**: All hyperparameters in config.py
2. **Single Responsibility**: Each module has one clear purpose
3. **Error Prevention**: Strict validation
4. **Type Hints Required**: 関数・メソッドのパラメータには型注釈を必須

### 🚨 型注釈ポリシー - 重要 (2025-12-02)

**動的な属性アクセスによるAttributeErrorを防ぐため、型注釈を徹底する。**

```python
# ❌ 型注釈なし → mypy で属性不足を検出できない
def __init__(self, base, context_dim):
    self.value = base.some_attribute

# ✅ 型注釈あり → mypy で属性不足を検出可能
def __init__(self, base: Config, context_dim: int):
    self.value = base.some_attribute
```

### Anti-Patterns to Avoid

- ❌ Changing architecture without full retraining
- ❌ Using deprecated features
- ❌ Leaving backward compatibility code
- ❌ 型注釈なしでのConfig属性アクセス

---

## File Structure

**Main Scripts**:
- `scripts/experiment_cascade_context.py` - カスケード連結実験スクリプト

**Core Implementation**:
- `src/trainers/phase1/memory.py` - Phase 1訓練ロジック
- `src/models/blocks.py` - ContextBlock/TokenBlock（1層固定）
- `src/models/layers.py` - ContextLayer/TokenLayer
- `src/models/llm.py` - 基本LLMモデル
- `src/losses/diversity.py` - OACDアルゴリズム

---

Last Updated: 2025-12-02 (可変ContextBlock数対応、カスケード連結方式採用、1層固定アーキテクチャ)
