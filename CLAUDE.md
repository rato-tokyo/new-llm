# New-LLM Project Guidelines

## 🎯 G案（prev_and_current_context）採用決定 (2025-12-02)

**Context ModeはG案に一本化。E案/A案/F案は削除予定。**

### 決定の背景

4つのContext Modeを比較実験した結果：

| Mode | Val PPL | Val Acc | メモリ効率 | 拡張性 |
|------|---------|---------|-----------|--------|
| E案 (layerwise) | **128.1** | **24.9%** | ❌ 低い | ❌ 低い |
| G案 (prev_and_current) | 132.2 | 24.4% | ✅ 高い | ✅ 高い |
| A案 (final_only) | 136.9 | 24.6% | ✅ 高い | △ |
| F案 (first_layer_only) | 137.9 | 24.4% | ✅ 高い | ❌ |

### E案が理論上は最良だが、G案を選択する理由

**1. メモリ効率**
```
E案: cache = [num_layers, num_tokens, context_dim]  # レイヤー数倍
G案: cache = [num_tokens, context_dim]              # 固定
```
大規模データでE案はメモリが厳しくなる。

**2. 拡張性**
```python
# G案は3層以上に自然に拡張可能
# Layer 1 ← context[i-2]  (2つ前)
# Layer 2 ← context[i-1]  (1つ前)
# Layer 3 ← context[i]    (現在)
```

**3. メンテナンス性**
複数のContext Modeを維持するコストが高い。

### 精度差は許容範囲

- PPL差: 4.1 (+3.2%)
- Acc差: 0.5%

データ量増加でどちらも改善するため、この差は許容可能。

### G案の動作

```python
# 2層の場合
TokenBlock Layer 1 ← context_cache[i-1]  # 前トークン時点
TokenBlock Layer 2 ← context_cache[i]    # 現在トークン時点

# 最初のトークン(i=0)では prev = current
```

### 将来的なE案復活条件

E案が必要になった場合の対応：
1. オンデマンド計算（Phase 2でContextBlock再計算）
2. 大容量GPU使用

詳細: `importants/experiment-results-20251202-context-mode-all-comparison.md`

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

### 実験結果 (context_dim=500)

| サンプル | トークン | Val PPL | Acc | ER% | α値 |
|---------|---------|---------|-----|-----|-----|
| 50 | 62,891 | 573.8 | 17.8% | 81.2% | - |
| 100 | 122,795 | 383.4 | 19.3% | 81.2% | - |
| 200 | 240,132 | **290.1** | **20.2%** | 81.3% | **-0.509** |

### 実験の実行

```bash
# ローカル（CPU）: 動作確認
python3 scripts/run_experiment.py -s 2

# Colab（GPU）: 本格実験
python3 scripts/run_experiment.py -s 50 100 200

# context_dim指定
python3 scripts/run_experiment.py -s 50 100 200 -c 500
```

---

## 🚨 num_layers = 2 推奨 (2025-12-02更新)

**G案採用により、2層以上が標準構成。**

```python
# config.py
num_layers = 2  # G案では2層以上が必要（prev/currentの差分活用）
```

**理由**:
- G案は「前のcontext」と「現在のcontext」を異なるレイヤーに注入
- 1層ではG案の意味がない（prev = currentになる）
- 2層: Layer1=prev, Layer2=current
- 3層: Layer1=prev, Layer2=none, Layer3=current（拡張可能）

---

## 💻 ローカル実験の注意事項 - CPU環境 (2025-12-01)

**ローカル環境（Mac/CPU）では処理が遅いため、サンプル数を最小限に抑える。**

### 推奨設定

```bash
# ローカル実験（CPU）: 2-5サンプルで十分
python3 scripts/run_experiment.py -s 2

# Colab（GPU）: 100サンプル以上で本格実験
python3 scripts/run_experiment.py -s 50 100 200
```

### ローカル vs Colab 比較

| 環境 | 推奨サンプル数 | 処理時間目安 |
|------|--------------|-------------|
| **ローカル（CPU）** | 2-5 | 数分〜十数分 |
| **Colab（GPU）** | 100-500 | 数分 |

---

## 🚨 CPU/GPUテンソル管理 - 重要教訓 (2025-12-01)

**大規模データ（2000サンプル以上）でOOMを防ぐため、テンソルのデバイス管理を徹底。**

### 問題の症状

```
RuntimeError: Expected all tensors to be on the same device, but got tensors is on cpu,
different from other tensors on cuda:0
```

### 根本原因

OOM対策でテンソルをCPUに保持する設計に変更した際、GPU転送漏れが発生：
1. `previous_contexts`: CPUに保持 → バッチ処理時にGPU転送必要
2. `token_embeds`: CPUに保持 → `combine_batch`後にGPU転送必要
3. `last_context`: CPU上で取得 → GPU転送必要

### 修正パターン

```python
# ❌ 修正前: CPUテンソルをそのまま使用
batch_contexts = previous_contexts[start_idx:end_idx].detach()
batch_combined = self._build_combined_tokens_batch(token_embeds, ...)

# ✅ 修正後: 明示的にGPU転送
batch_contexts = previous_contexts[start_idx:end_idx].detach().to(self.device)
batch_combined = self._build_combined_tokens_batch(token_embeds, ...).to(self.device)
```

### チェックリスト（OOM対策コード変更時）

- [ ] CPUに保持するテンソルを特定
- [ ] GPU演算に渡す前に`.to(self.device)`を追加
- [ ] ループ内のすべてのテンソル転送を確認
- [ ] `torch.cat`や演算の入力デバイスを統一

---

## 🚨 Effective Rank計算の整合性 - 重要教訓 (2025-12-01)

**Phase 1 Validation Early StoppingのVal ERと最終評価のERが大幅に乖離する問題を修正。**

### 問題の症状

- `_quick_validate()` が返すVal ER: 3-30%
- 最終評価のVal ER: 64%
- **約2-20倍の乖離**

### 根本原因（3つ）

**1. サンプルサイズの違い（最大の原因）**
- `_quick_validate()`: 500トークン → **ERが低く出る**
- 最終評価: 31,024トークン → ERが正確に出る
- **修正**: `phase1_val_sample_size = 10000` に増加

**2. ER計算方法の違い**
- `_quick_validate()`: 共分散行列の固有値分解を使用
- `analyze_fixed_points()`: SVDの特異値を使用
- **修正**: 両方ともSVDベースに統一

**3. コードパスの違い**
- `_quick_validate()`: `collect_all_layers=False`
- `evaluate()`: `collect_all_layers=True`、`token_embeds[:-1]`で最後のトークンを除く
- **修正**: `_quick_validate()`を`evaluate()`と完全に同じ処理に変更

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

# 2. 実験実行（検証ファイルは自動生成される）
!cd /content/new-llm && python3 scripts/run_experiment.py -s 50 100 200
```

---

## 🔧 開発環境のLint/Type Check (2025-11-29)

**pyenv環境ではruffやmypyを直接実行できないため、`python3 -m` で実行する。**

```bash
# Lint (ruff)
python3 -m ruff check src/

# Type check (mypy)
python3 -m mypy src/ --ignore-missing-imports

# 特定ファイルのみ
python3 -m ruff check src/trainers/phase1/memory.py
python3 -m mypy src/trainers/phase1/memory.py --ignore-missing-imports
```

---

## 🚨 CRITICAL: 後方互換性コード禁止 (2025-11-29)

**古い機能を残すことは厳禁。後方互換性を意識したコードは絶対に書かない。**

### 禁止事項

1. **オプション引数での分岐禁止**
   ```python
   # ❌ 禁止: 古いパスを残す
   def func(cache=None):
       if cache is None:
           cache = build_cache()  # 古いパス

   # ✅ 正解: 必須引数にする
   def func(cache):
       pass  # キャッシュは呼び出し元で必ず準備
   ```

2. **古いメソッドの残存禁止**
   - 新しい設計に置き換えたら、古いメソッドは即座に削除
   - 「念のため」で残すと、誤って古いパスが実行される

---

## 🚀 PHASE 2 CACHE REUSE - Phase 1キャッシュ再利用 (2025-11-29)

**Phase 1で計算した全レイヤー出力をPhase 2で再利用し、627秒のキャッシュ再構築を省略。**

### 新方式: Phase 1からキャッシュを渡す

```python
# Phase 1: return_all_layers=True で全レイヤー出力も取得
train_contexts, train_context_cache, train_token_embeds = phase1_trainer.train(
    ..., return_all_layers=True
)

# Phase 2: キャッシュを受け取り、再構築をスキップ
phase2_trainer.train_full(
    ...,
    train_context_cache=train_context_cache,
    train_token_embeds=train_token_embeds,
)
```

---

## 🧊 EMBEDDING FREEZE ADOPTED - Embedding凍結採用 (2025-11-27)

**Phase 2でEmbedding凍結を標準採用。**

### 実験結果

| 指標 | Embedding学習 | Embedding凍結 | 改善率 |
|------|--------------|--------------|--------|
| Val PPL (500samples) | 1189.15 | **334.31** | **-71.9%** |
| Val Acc (500samples) | 11.58% | **18.88%** | **+63.0%** |

### 設定

```python
# config.py
phase2_freeze_embedding = True  # 推奨（デフォルト）
use_weight_tying = True         # 推奨（デフォルト）
```

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

**禁止事項**:
- ❌ "GOOD", "EXCELLENT" などの抽象的表現での報告
- ❌ 数値を伴わない判定結果の報告

**必須報告項目**:
- ✅ **収束率**: 具体的なパーセンテージ (例: 1.9%)
- ✅ Effective Rank: **実数値/総次元数とパーセンテージ** (例: 406/500 = 81.2%)
- ✅ Val PPL: **実数値** (例: 290.1)
- ✅ Val Acc: **実数値** (例: 20.2%)
- ✅ α値: **実数値** (例: -0.509)

---

## 📐 アーキテクチャ仕様

### Core Components

**1. ContextLayer / TokenLayer**
- ContextLayer: 文脈処理専用（token継ぎ足し方式）
- TokenLayer: トークン処理専用

**2. ContextBlock / TokenBlock**
- ContextBlock: Phase 1で学習、Phase 2でfreeze
- TokenBlock: Phase 2で学習

**3. LLM (Main Model)**
- Token Embedding: GPT-2 pretrained (768-dim, frozen in Phase 2)
- Weight Tying: token_output shares weights with token_embedding

### Phase 1: 多様性学習（OACD）

- **学習対象**: ContextBlockのみ
- **TokenBlock**: 未使用
- **損失**: 多様性損失のみ（OACD）

### Phase 2: トークン予測

- **ContextBlock**: frozen（重み固定）
- **TokenBlock**: 学習
- **損失**: CrossEntropy（次トークン予測）のみ

---

## Code Quality Standards

### Principles

1. **No Hardcoding**: All hyperparameters in config.py
2. **Single Responsibility**: Each module has one clear purpose
3. **Error Prevention**: Strict validation

### Anti-Patterns to Avoid

- ❌ Changing architecture without full retraining
- ❌ Using deprecated features
- ❌ Leaving backward compatibility code

---

## File Structure

**Main Scripts**:
- `scripts/run_experiment.py` - 標準実験スクリプト（Phase 1 + Phase 2）
- `config.py` - 設定ファイル

**Core Implementation**:
- `src/trainers/phase1/memory.py` - Phase 1訓練ロジック
- `src/trainers/phase2.py` - Phase 2訓練ロジック
- `src/models/llm.py` - モデルアーキテクチャ
- `src/losses/diversity.py` - OACDアルゴリズム

---

Last Updated: 2025-12-02 (G案採用決定、num_layers=2推奨)
