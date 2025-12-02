# New-LLM Project Guidelines

## 🎯 Multi Context方式（N分割）採用 (2025-12-02)

**⚠️ 重要: 成功要因は「異なるデータで学習」することです。**

### ベストプラクティス

| 構成 | Val PPL | Val Acc | 備考 |
|------|---------|---------|------|
| **2-block (500×2=1000)** | **111.9** | **25.6%** | **2分割** |
| C1T1-500 | 127.2 | 24.7% | 標準構成 |
| C2T2-500 | 132.2 | 24.4% | 2層だが悪化 |
| C1T1-1000 | 134.0 | 23.6% | context_dim増加は非効率 |

### N分割方式の正しい理解

**核心**: 各ContextBlockは**異なるデータ**で学習することで、**異なる表現**を獲得する。

```
N分割方式（--num-blocks N で指定）:

Phase 1[i]: ContextBlock[i] を i 番目のデータ区間で学習
  → 初期入力: ゼロベクトル
  → データ: tokens[i*split:(i+1)*split]

Phase 2 Prep: 順次処理でキャッシュ収集
  → 全データを順次処理してcontext_0, ..., context_{N-1}を収集

Phase 2: TokenBlock 学習
  → 入力: concat(context_0[i-1], ..., context_{N-1}[i-1])
  → 予測: token[i]
```

### ❌ 間違った理解

以下は**間違い**です:
- 「全ブロックが全データで学習」→ 同じような表現になってしまう
- 「Initial Context Inheritanceで異なる表現を獲得できる」→ PPL悪化（119.5 vs 111.9）

**成功の鍵**: 異なるデータで学習すること。初期入力の違いではなく、学習データの違いが重要。

### 実験の実行

```bash
# Colab（GPU）: 2ブロック（デフォルト）
python3 scripts/experiment_cascade_context.py -s 2000

# Colab（GPU）: 4ブロック
python3 scripts/experiment_cascade_context.py -s 2000 -n 4
```

### Phase 2 Prepのキャッシュ収集

Phase 2では**順次処理**で全データのコンテキストキャッシュを収集します。
これはdual_output.txt（PPL=111.9）と同じ方式です。

**処理時間**: 2M tokens で約983秒（約16分）

**注意**: 並列処理（shifted_prev_context方式）は Phase 1 学習専用です。
Phase 2 Prep では正確なRNN動作を再現するため、順次処理を使用します。

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

## 🚨🚨 Phase 1学習では順次処理禁止 (CRITICAL) 🚨🚨

**⚠️ このセクションは過去に誤って削除されたことがあります。絶対に削除しないでください。**

**Phase 1学習では、順次処理（`for i in range(num_tokens)`でトークンを1つずつ処理）は厳禁。必ずshifted_prev_context方式で並列処理すること。**

### Phase 1学習での禁止パターン

```python
# ❌ 禁止: Phase 1学習で順次処理（非常に遅い）
for i in range(num_tokens):
    token_embed = input_embeds[i:i+1].to(device)
    new_context = model.forward_context(prev_context, token_embed)
    prev_context = new_context
```

### Phase 1学習での推奨パターン

```python
# ✅ 推奨: shifted_prev_context方式（並列処理、数秒で完了）
previous_contexts = torch.randn(num_tokens, context_dim) * 0.01
zero_init = torch.zeros(1, context_dim)  # 常にゼロベクトルから開始

for iteration in range(max_iterations):
    shifted_prev_context = torch.cat([zero_init, previous_contexts[:-1]], dim=0)
    new_contexts = model.forward_context(shifted_prev_context, input_embeds)
    if converged:
        break
    previous_contexts = new_contexts
```

### Phase 2 Prepでは順次処理を使用

**Phase 2 Prepのキャッシュ収集では、正確なRNN動作を再現するため順次処理を使用する。**

```python
# ✅ Phase 2 Prep: 順次処理（正確なRNN動作）
for i in range(num_tokens):
    new_context_a = model.forward_context(0, prev_context_a, token_embed)
    new_context_b = model.forward_context(1, prev_context_b, token_embed)
    prev_context_a = new_context_a
    prev_context_b = new_context_b
```

**処理時間**: 2M tokens で約983秒（約16分）。時間はかかるが正確な結果を得る。

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

## 🚨🚨 Phase 2 Prep: GPUメモリリーク防止 (CRITICAL) (2025-12-03) 🚨🚨

**⚠️ Phase 2 Prepのキャッシュ収集でGPUメモリが15GB+消費される問題が発生した。**

### 問題の原因

```python
# ❌ 禁止: 全token_embedsをGPUに一度にロード
with torch.no_grad():
    token_embeds = model.token_embedding(token_ids.to(device))  # 全データをGPUに！
    token_embeds = model.embed_norm(token_embeds)

for i in range(num_tokens - 1):
    token_embed = token_embeds[i:i+1]  # GPUメモリに全体が残る
    new_context = model.forward_context(prev_context, token_embed)
    prev_context = new_context  # 計算グラフが蓄積
```

**問題点**:
1. 全token_embeds（240万トークン×768次元×4bytes ≈ 7GB）がGPUに常駐
2. `prev_context = new_context` で計算グラフが蓄積
3. ループ中にメモリが増加し続ける

### 正しい実装（チャンク処理）

```python
# ✅ 推奨: チャンク単位でGPUに転送し、即座に解放
# src/utils/cache.py の collect_context_cache_sequential を使用

with torch.no_grad():
    for chunk_start in range(0, num_tokens - 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_tokens - 1)

        # チャンク分だけGPUに転送
        chunk_token_ids = token_ids[chunk_start:chunk_end + 1].to(device)
        chunk_embeds = model.token_embedding(chunk_token_ids)
        chunk_embeds = model.embed_norm(chunk_embeds)

        for i in range(chunk_end - chunk_start):
            token_embed = chunk_embeds[i:i+1]
            new_context = model.forward_context(prev_context, token_embed)
            context_cache[chunk_start + i] = new_context.cpu()
            prev_context = new_context.detach()  # ← 計算グラフを切断！

        # チャンク完了後にGPUメモリを解放
        del chunk_token_ids, chunk_embeds
        clear_gpu_cache(device)
```

### 必須チェックリスト（Phase 2 Prep実装時）

- [ ] **全データを一度にGPUにロードしていないか**
- [ ] **チャンク単位で処理しているか**（デフォルト: 10,000トークン）
- [ ] **`.detach()`で計算グラフを切断しているか**
- [ ] **チャンク完了後に`del`と`clear_gpu_cache()`を呼んでいるか**
- [ ] **共通コード`src/utils/cache.py`を使用しているか**

### 共通コードの使用

```python
# スクリプト固有の実装ではなく、共通コードを使用すること
from src.utils.cache import collect_context_cache_sequential

# 単一ブロック用
context_cache = collect_context_cache_sequential(model, token_ids, device)

# 複数ブロック用
from src.utils.cache import collect_context_cache_sequential_multiblock
context_caches = collect_context_cache_sequential_multiblock(model, token_ids, device, num_blocks)
```

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

Last Updated: 2025-12-03 (Phase 2 PrepのGPUメモリリーク防止、共通コードsrc/utils/cache.py追加)
