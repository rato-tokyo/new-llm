# New-LLM Project Guidelines

## 🎯 Dual方式（前半/後半分割）採用 (2025-12-02)

**⚠️ 重要: Dual方式の成功要因は「異なるデータで学習」することです。**

### ベストプラクティス

| 構成 | Val PPL | Val Acc | 備考 |
|------|---------|---------|------|
| **Dual (500×2=1000)** | **111.9** | **25.6%** | **前半/後半分割** |
| C1T1-500 | 127.2 | 24.7% | 標準構成 |
| C2T2-500 | 132.2 | 24.4% | 2層だが悪化 |
| C1T1-1000 | 134.0 | 23.6% | context_dim増加は非効率 |

### Dual方式の正しい理解

**核心**: 各ContextBlockは**異なるデータ**で学習することで、**異なる表現**を獲得する。

```
2ブロックの場合（Dual方式）:

Phase 1[0]: ContextBlock[0] を前半データで学習
  → 初期入力: ゼロベクトル
  → データ: tokens[0:split]（前半）

Phase 1[1]: ContextBlock[1] を後半データで学習
  → 初期入力: context[0]_final（前のブロックの最終出力）
  → データ: tokens[split:]（後半）
  → Context Continuity Loss: block1の最初の出力 ≈ block0の最終出力

Phase 2: TokenBlock 学習
  → 入力: concat(context_0[i-1], context_1[i-1])
  → 予測: token[i]
```

### ❌ 間違った理解（全データ学習）

以下は**間違い**です:
- 「全ブロックが全データで学習」
- 「初期入力の継承だけで異なる表現を獲得できる」

全データで2つのBlockを学習しても、Initial Contextが違うだけでは**同じような表現**になってしまいます。

### 実験の実行

```bash
# Colab（GPU）: 本格実験
python3 scripts/experiment_cascade_context.py -s 2000

# Context Continuity Lossを無効化（検証用）
python3 scripts/experiment_cascade_context.py -s 2000 --no-continuity-loss

# Phase 1キャッシュを直接使用（時間短縮）
python3 scripts/experiment_cascade_context.py -s 2000 --use-phase1-cache
```

### --use-phase1-cache オプション

**Phase 2 Prepでの全データキャッシュ再収集をスキップし、Phase 1で得たキャッシュを直接結合して使用。**

**前提条件**:
- Context Continuity Lossにより、前半/後半の境界での損失が無視できるほど小さい
- RNN収束後は全トークンの出力が同じ固定点に収束するため、結合しても理論的に問題なし

**動作**:
1. Block A: 前半キャッシュ + 後半は最終値で埋める
2. Block B: 前半はBlock Aの最終値で埋める + 後半キャッシュ
3. Validationキャッシュのみ再収集（学習データではないため）

**効果**: Training dataの再収集（数分〜十数分）をスキップし、大幅な時間短縮

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

## 🔗 Context Continuity Loss - 削除厳禁 (2025-12-02)

**⚠️ このセクションは重要な設計決定を記録しています。削除しないでください。**

**block_idx > 0 のContextBlockでは、Context Continuity Lossを追加。**

### 目的

前のブロックの最終出力（`prev_context_final`）と、現在のブロックの**最初のトークンの出力**を近づける。

```python
# Context Continuity Loss（block_idx > 0の場合のみ）
if block_idx > 0 and prev_context_final is not None:
    if start_idx == 0:  # 最初のバッチの最初の出力
        first_output = batch_output[:1]
        continuity_loss = MSE(first_output, prev_context_final)
        total_loss = diversity_loss + 0.1 * continuity_loss
```

### なぜ「最初の出力」を使うのか

RNN収束後の理論的性質：
- Phase 1のOACD学習が収束すると（conv=90%+）、RNNは**固定点に収束**
- 収束後: `block_first ≈ block_final`（全トークンの出力が同じ値に収束）
- したがって、「最初の出力を近づける」と「最終出力を近づける」は理論的に同等

**「最初の出力」を選んだ理由**:
1. `initial_context`として`prev_context_final`を入力しているため、入力→出力の因果関係が直接的
2. Dual方式の成功（PPL=111.9）と同様の「文脈継続」イメージに合致
3. 実装がシンプル（最初のバッチで計算）

### 重要な注意

- この損失は**全てのcontext出力ではなく、最初の1つだけ**に適用
- OACDの多様性損失と併用（weight=0.1）
- block_idx=0（最初のブロック）では使用しない

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

## 🚨🚨 順次処理禁止 - 削除厳禁 (CRITICAL) 🚨🚨

**⚠️ このセクションは過去に誤って削除されたことがあります。絶対に削除しないでください。**

**順次処理（`for i in range(num_tokens)`でトークンを1つずつ処理）は厳禁。必ずshifted_prev_context方式で並列処理すること。**

### 禁止パターン（絶対に使わない）

```python
# ❌ 禁止: 順次処理（非常に遅い、数百秒〜数千秒かかる）
for i in range(num_tokens):
    token_embed = input_embeds[i:i+1].to(device)
    new_context = model.forward_context(prev_context, token_embed)
    context_cache[i] = new_context.cpu()
    prev_context = new_context  # 前の出力を次の入力に
```

### 推奨パターン（必ずこちらを使う）

```python
# ✅ 推奨: shifted_prev_context方式（並列処理、数秒で完了）
# Phase 1と同様の反復処理で収束させる
previous_contexts = torch.randn(num_tokens, context_dim) * 0.01  # ランダム初期化

for iteration in range(max_iterations):
    # shifted_prev_context: [initial_context, prev_contexts[:-1]]
    shifted_prev_context = torch.cat([initial_context, previous_contexts[:-1]], dim=0)

    # バッチ処理で一括forward
    new_contexts = model.forward_context(shifted_prev_context, input_embeds)

    # 収束判定
    if converged:
        break
    previous_contexts = new_contexts
```

### なぜ並列処理が必要か

| 方式 | 処理時間（2M tokens） | 処理時間（22k tokens） |
|------|---------------------|----------------------|
| 順次処理 | **983秒（16分）** | **9秒** |
| 並列処理 | **5-10秒** | **0.1秒以下** |

**順次処理は100倍以上遅い。Training/Validation両方で並列処理を使うこと。**

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

Last Updated: 2025-12-02 (Context Continuity Loss追加、順次処理禁止ルール追記、Initial Context Inheritance方式採用、可変ContextBlock数対応、1層固定アーキテクチャ)
