# New-LLM Project Guidelines

## 🎯 Initial Context Inheritance方式採用 (2025-12-02)

**1層固定アーキテクチャにInitial Context Inheritance方式を採用。複数レイヤーは不要。**

### 決定の背景

Dual方式（前半/後半分割）の効果を全データで実現するため、Initial Context Inheritance方式を採用。

| 構成 | Val PPL | Val Acc | 備考 |
|------|---------|---------|------|
| **Dual (500×2=1000)** | **111.9** | **25.6%** | **前半/後半分割** |
| C1T1-500 | 127.2 | 24.7% | 標準構成 |
| C2T2-500 | 132.2 | 24.4% | 2層だが悪化 |
| C1T1-1000 | 134.0 | 23.6% | context_dim増加は非効率 |

### Initial Context Inheritance方式の特徴

**核心**: 各ContextBlockは**独立してRNN学習**を行う。ブロック1以降は、**最初のトークンの入力として前のブロックの最終出力を使用**。

**可変ブロック数対応**: ContextBlockの数は1, 2, 3, ... と柔軟に指定可能。

```
N個のContextBlockを順次学習:

Phase 1[0]: ContextBlock[0] を全データで学習
  → 初期入力: ゼロベクトル
  → RNN学習: context_0[i] = ContextBlock_0(context_0[i-1], token[i])
  → context_0_final を保存

Phase 1[1]: ContextBlock[1] を全データで学習
  → 初期入力: context_0_final（前のブロックの最終出力）
  → RNN学習: context_1[i] = ContextBlock_1(context_1[i-1], token[i])
  → context_1_final を保存

Phase 1[N-1]: ContextBlock[N-1] を全データで学習
  → 初期入力: context_{N-2}_final
  → RNN学習: 標準的なRNN

Phase 2: TokenBlock 学習
  → 入力: concat(context_0[i-1], ..., context_{N-1}[i-1])
  → 予測: token[i]
```

**Dual方式との違い**:
- Dual: ブロックAは前半データ、ブロックBは後半データで学習
- Initial Context Inheritance: 全ブロックが全データで学習（初期入力のみ継承）

### なぜInitial Context Inheritance方式が良いのか

1. **文脈の継続性**: 前のブロックの最終状態を継承
2. **全データ活用**: 全ブロックが全データで学習
3. **Dual方式と同様の効果**: 前半で学習した文脈を後半に引き継ぐ
4. **並列処理可能**: 標準的なRNN学習と同じ速度

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

Last Updated: 2025-12-02 (Initial Context Inheritance方式採用、可変ContextBlock数対応、1層固定アーキテクチャ)
