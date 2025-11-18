# Training Progression: データセットの難易度順

New-LLMの訓練における、データセットの難易度と進行ステップを説明します。

---

## 📊 現在の進捗状況

| Level | Dataset | Status | PPL | Acc |
|-------|---------|--------|-----|-----|
| ✅ **Level 1** | Dolly-15k | Complete | 15.6 | 46.6% |
| 🔄 **Level 2** | **HH-RLHF** | **Ready** | Expected: 17-20 | Expected: 43-46% |
| 🔄 **Level 3** | **UltraChat** | **Ready** | Expected: 20-25 | Expected: 40-43% |
| ⏭️ Level 4 | Code (CodeAlpaca) | Planned | Expected: 25-30 | Expected: 35-40% |
| ⏭️ Level 5 | Reasoning (MATH) | Future | Expected: 30-40 | Expected: 30-35% |

---

## Level 1: Dolly-15k（構造化Q&A）✅ 完了

### 特徴

```
難易度: ★☆☆☆☆ （最も簡単）
形式: 単発Q&A
データ量: 15,000件
言語: 英語のみ
```

### 学習内容

- 基本的なInstruction-Response理解
- 単純なQ&A応答
- 明確なパターン認識

### 結果

- **PPL 15.6** ✓ 優秀
- **Acc 46.6%** ✓ 高精度

**結論**: **完全に習得済み**

---

## Level 2: HH-RLHF（高品質対話）🔄 現在の目標

### 特徴

```
難易度: ★★☆☆☆
形式: 複数ターン対話
データ量: 43,000件（helpful）+ 42,000件（harmless）
言語: 英語のみ
品質: 非常に高い（人間フィードバック付き）
```

### Dolly-15kとの違い

| 要素 | Dolly-15k | HH-RLHF |
|-----|-----------|---------|
| **会話長** | 短い（1往復） | **長い（複数ターン）** |
| **文脈理解** | 不要 | **必須** |
| **応答品質** | 標準 | **人間が選択した高品質** |
| **安全性** | 一般的 | **有害性を排除** |

### 学習内容

- **複数ターン対話**: 前の発言を踏まえた応答
- **文脈保持**: 会話の流れを理解
- **高品質応答**: 人間が好む応答パターン
- **安全性**: 有害な応答を避ける

### 期待される結果

- **PPL 17-20**: Dollyより少し難しい
- **Acc 43-46%**: 複数ターンで精度低下

### 実装

```bash
# Colab実行コマンド
python scripts/train_hh_rlhf.py --num_layers 1

# Layer 4推奨（より高性能）
python scripts/train_hh_rlhf.py --num_layers 4
```

**これをクリアしたら**: より複雑な対話が可能に

---

## Level 3: UltraChat（大規模対話）🔄 準備完了

### 特徴

```
難易度: ★★★☆☆
形式: 非常に多様な対話
データ量: 1.5M+ 会話
言語: 英語
生成: GPT-3.5ベース
```

### HH-RLHFとの違い

| 要素 | HH-RLHF | UltraChat |
|-----|---------|-----------|
| **データ量** | 85k | **1.5M+（18倍）** |
| **多様性** | 標準 | **非常に高い** |
| **トピック** | 一般的 | **多岐にわたる** |

### 学習内容

- **大規模データ**: より多様なパターン
- **トピック多様性**: 様々な分野の対話
- **長時間訓練**: 大量データの消化

### 期待される結果

- **PPL 20-25**: データ量と多様性で難易度上昇
- **Acc 40-43%**: 多様性により精度低下

### 訓練時間

- Layer 1: 約2-3時間（L4 GPU、フルデータセット）
- Layer 4: 約3-4時間（フルデータセット）
- Layer 1/4: 20-40分（サブセット10万件、推奨）

### 実装

```bash
# Colab実行コマンド（サブセット推奨）
python scripts/train_ultrachat.py --num_layers 4 --max_samples 100000

# フルデータセット（1.5M件）
python scripts/train_ultrachat.py --num_layers 4
```

**実装済み**:
- ✅ `scripts/train_ultrachat.py` - 訓練スクリプト
- ✅ `src/training/ultrachat_dataset.py` - データセットローダー
- ✅ `tests/test_ultrachat_training.py` - テストスイート（全テスト合格）
- ✅ `ULTRACHAT_TRAINING.md` - Colab訓練ガイド

**これをクリアしたら**: 非常に多様な対話に対応可能（**対話能力の完成**）

---

## Level 4: CodeAlpaca（コード生成）⏭️

### 特徴

```
難易度: ★★★★☆
形式: コード生成タスク
データ量: 20,000件
言語: 英語+プログラミング言語
タスク: コード記述、デバッグ、説明
```

### 新しい挑戦

- **コード理解**: 自然言語とコードの混在
- **構文正確性**: プログラミング構文の理解
- **論理的思考**: アルゴリズム設計

### 学習内容

- Python、JavaScript、その他のコード生成
- コードの説明
- デバッグとエラー修正
- アルゴリズム設計

### 期待される結果

- **PPL 25-30**: コードは自然言語より難しい
- **Acc 35-40%**: 構文の厳密性

### 実装

```python
# 新規作成が必要
# scripts/train_code_alpaca.py
# src/training/code_alpaca_dataset.py
```

**これをクリアしたら**: コード生成能力を獲得

---

## Level 5: MATH Dataset（数学的推論）⏭️ 最高難易度

### 特徴

```
難易度: ★★★★★ （最高難度）
形式: 数学問題と証明
データ量: 12,500件
言語: 英語+数学記号
タスク: 数学的推論、証明
```

### 新しい挑戦

- **論理的推論**: 複雑な推論チェーン
- **数学記号**: 特殊な表記法
- **正確性**: 計算の正確性
- **段階的思考**: ステップバイステップ

### 学習内容

- 代数、幾何、微積分
- 論理的推論
- 証明の記述
- Chain-of-Thought（段階的思考）

### 期待される結果

- **PPL 30-40**: 最も難しい
- **Acc 30-35%**: 推論の複雑さ

### 必要な拡張

- **Context Expansion**: 512→1024次元（長い推論チェーン）
- **Layer Increase**: Layer 8-12（より深い思考）

**これをクリアしたら**: 高度な数学的推論が可能に

---

## 🎯 推奨進行順序

### フェーズ1: 対話能力の獲得

1. ✅ **Dolly-15k** - 基礎的なQ&A（完了）
2. 🔄 **HH-RLHF** - 高品質複数ターン対話（現在）
3. ⏭️ **UltraChat** - 大規模多様な対話

### フェーズ2: 専門能力の獲得

4. ⏭️ **CodeAlpaca** - コード生成能力
5. ⏭️ **MATH Dataset** - 数学的推論

---

## 📈 難易度の要因

### データセットを難しくする要因

1. **文脈の長さ**: 複数ターン > 単発
2. **トピックの多様性**: 多様 > 限定的
3. **構造の明確さ**: 不明確 > 明確
4. **専門性**: 専門的（コード、数学） > 一般的
5. **推論の深さ**: 深い > 浅い

### PPL予測の目安

```
PPL 15-17: 構造化された簡単なタスク
PPL 18-22: 複数ターン対話、一般的なトピック
PPL 23-28: 大規模多様データ
PPL 29-35: コード生成、専門知識
PPL 36-45: 数学的推論、高度な論理
```

---

## 🔧 各レベルでの推奨設定

### Level 1-2（対話）

```python
num_layers = 1  # 十分
context_dim = 256  # 標準
max_seq_length = 128  # 複数ターン対応
```

### Level 3（大規模対話）

```python
num_layers = 4  # より深く
context_dim = 256  # 標準
max_seq_length = 128  # 同じ
epochs = 150  # 多めに
```

### Level 4-5（専門タスク）

```python
num_layers = 8-12  # 深い思考
context_dim = 512-1024  # 大容量
max_seq_length = 256  # 長い推論
epochs = 200  # 十分に訓練
```

---

## 💡 成功の指標

### 各レベルをクリアしたと判断する基準

| Level | クリア条件 |
|-------|-----------|
| Level 1 | PPL < 17 |
| Level 2 | PPL < 21 |
| Level 3 | PPL < 26 |
| Level 4 | PPL < 32 |
| Level 5 | PPL < 42 |

---

## 🚀 次の一歩

**現在の位置**: Level 1完了、Level 2-3準備完了

**推奨アクション**:

**オプションA: HH-RLHF（高品質対話、85k件）**
```bash
# Layer 4推奨（20-30分）
python scripts/train_hh_rlhf.py --num_layers 4
```

**オプションB: UltraChat（大規模多様対話、1.5M件）← 推奨**
```bash
# サブセット10万件（20-40分）
python scripts/train_ultrachat.py --num_layers 4 --max_samples 100000

# またはフルデータセット（2-3時間）
python scripts/train_ultrachat.py --num_layers 4
```

**推奨**: HH-RLHFをスキップして、**UltraChat**に直接進む（より実践的）

**成功したら**: 対話能力完成 → Level 4（CodeAlpaca）でコード生成能力を追加

---

## 📚 参考

- Dolly-15k: `experiments/dolly_dialog_experiment_2025-11-19.md`
- HH-RLHF: `HH_RLHF_TRAINING.md`, `scripts/train_hh_rlhf.py`
- UltraChat: `ULTRACHAT_TRAINING.md`, `scripts/train_ultrachat.py`
- Layer Optimization: `experiments/layer_optimization_experiment_2025-11-18.md`

---

**現在のステータス**: Level 2（HH-RLHF）& Level 3（UltraChat）実装完了、テスト済み、訓練準備完了 ✅
