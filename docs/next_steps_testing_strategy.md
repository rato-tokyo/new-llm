# New-LLM 今後のテスト方針

**作成日**: 2025-11-21

## 🎯 現状と目標

### 現状
- ✅ Phase 1: 固有点学習の実装完了
- ✅ Phase 2: トークン予測訓練の実装完了
- ✅ アーキテクチャ比較: Sequential/Layer-wise/Mixed [2,2]
- ✅ 単一サンプル（512トークン）でのテスト完了

### 目標
1. **複数サンプルでの訓練**: スケーラビリティの検証
2. **Early Stopping実装**: 過学習防止と訓練効率化
3. **本格的な対話モデル訓練**: UltraChatデータセットでの実用レベル到達

---

## 📋 提案: 次のテストステップ

### Step 1: 複数サンプル統合訓練（優先度: 高）

**目的**: 単一サンプルから複数サンプルへのスケーリング

**実装内容**:
```python
# tests/phase2_experiments/test_multi_sample_training.py

def load_multiple_samples(num_samples=10, max_length=512):
    """複数のUltraChatサンプルをロード"""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    all_token_ids = []
    for i in range(num_samples):
        sample = dataset[i]
        text = ""
        for msg in sample['messages']:
            text += msg['content'] + " "

        tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
        all_token_ids.extend(tokens)

    return torch.tensor(all_token_ids)

def phase1_multi_sample_train(model, token_ids, batch_size=512):
    """Phase 1: バッチ処理で効率的に固有点学習"""
    # トークン列を batch_size ごとに分割
    # 各バッチで並列に固有点学習
    pass
```

**テスト計画**:
1. 10サンプル（約5,000トークン）
2. 50サンプル（約25,000トークン）
3. 100サンプル（約50,000トークン）

**期待される発見**:
- 固有点の再利用性（異なるサンプル間で似た文脈ベクトル）
- メモリ使用量の線形スケーリング（New-LLMの優位性）
- 訓練時間の推移

---

### Step 2: Early Stopping実装（優先度: 高）

**目的**: 過学習防止と訓練効率化

**実装内容**:

#### 2.1 Phase 1用Early Stopping
```python
class Phase1EarlyStopping:
    """Phase 1固有点学習用Early Stopping

    停止条件:
    - 収束率が閾値以上（例: 95%以上のトークンが収束）
    - または最大エポック数到達
    """
    def __init__(self, convergence_threshold=0.95, patience=3):
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.best_convergence = 0
        self.epochs_no_improve = 0

    def __call__(self, convergence_rate):
        if convergence_rate >= self.convergence_threshold:
            return True  # 十分収束したので停止

        if convergence_rate > self.best_convergence:
            self.best_convergence = convergence_rate
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        return self.epochs_no_improve >= self.patience
```

#### 2.2 Phase 2用Early Stopping
```python
class Phase2EarlyStopping:
    """Phase 2トークン予測用Early Stopping

    停止条件:
    - Validation Lossが patience エポック改善しない
    - または Validation Perplexityが閾値以下
    """
    def __init__(self, patience=5, min_delta=0.001, ppl_threshold=None):
        self.patience = patience
        self.min_delta = min_delta
        self.ppl_threshold = ppl_threshold
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None

    def __call__(self, val_loss, val_ppl, model):
        # PPL閾値チェック
        if self.ppl_threshold and val_ppl <= self.ppl_threshold:
            return True

        # Loss改善チェック
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            # ベストモデルに復元
            model.load_state_dict(self.best_model_state)
            return True

        return False
```

**使用例**:
```python
# Phase 1
early_stop = Phase1EarlyStopping(convergence_threshold=0.95, patience=3)
for epoch in range(max_epochs):
    convergence_rate = train_phase1_epoch(...)
    if early_stop(convergence_rate):
        print(f"Early stopping at epoch {epoch}: convergence = {convergence_rate:.1%}")
        break

# Phase 2
early_stop = Phase2EarlyStopping(patience=5, min_delta=0.001)
for epoch in range(max_epochs):
    train_loss, train_ppl = train_phase2_epoch(...)
    val_loss, val_ppl = evaluate_phase2(...)

    if early_stop(val_loss, val_ppl, model):
        print(f"Early stopping at epoch {epoch}: val_ppl = {val_ppl:.2f}")
        break
```

---

### Step 3: Train/Validation分割（優先度: 高）

**目的**: 過学習の適切な検出

**実装内容**:
```python
def split_samples(num_total_samples=100, train_ratio=0.8):
    """Train/Validation分割"""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    train_size = int(num_total_samples * train_ratio)

    train_samples = dataset[:train_size]
    val_samples = dataset[train_size:num_total_samples]

    return train_samples, val_samples
```

**テスト計画**:
- 100サンプル: Train 80 / Val 20
- Validation Lossでearly stopping
- 過学習検出の確認

---

### Step 4: キャッシュシステムの拡張（優先度: 中）

**目的**: 複数サンプルの固有点を効率的に管理

**実装内容**:
```python
# cache/fixed_contexts/ の構造
# - sample_0001_hash.pt
# - sample_0002_hash.pt
# - ...
# - index.json  # 全サンプルのインデックス

def save_fixed_contexts_batch(samples, fixed_contexts_list, architecture):
    """複数サンプルの固有点をバッチ保存"""
    index = {
        'architecture': architecture,
        'num_samples': len(samples),
        'samples': []
    }

    for i, (sample, contexts) in enumerate(zip(samples, fixed_contexts_list)):
        sample_hash = get_sample_hash(sample)
        cache_path = f"cache/fixed_contexts/{architecture}_sample_{i:04d}_{sample_hash}.pt"
        save_fixed_contexts(cache_path, sample, contexts, ...)

        index['samples'].append({
            'id': i,
            'hash': sample_hash,
            'path': cache_path
        })

    with open('cache/fixed_contexts/index.json', 'w') as f:
        json.dump(index, f)
```

---

### Step 5: アーキテクチャスケーリング実験（優先度: 中）

**目的**: より深いモデルでの性能検証

**テスト構成**:
1. **Mixed [2,2]** - Baseline（現在のベスト）
2. **Mixed [2,2,2]** - 6層、3ブロック
3. **Mixed [2,2,2,2]** - 8層、4ブロック
4. **Mixed [3,3]** - 6層、2ブロック（Transformer標準に近い）

**期待される発見**:
- 最適な層数とブロック数
- Perplexityの改善度合い
- 過学習の傾向

---

### Step 6: 学習率スケジューリング（優先度: 低）

**目的**: 訓練の安定化と高速化

**実装候補**:
```python
# Cosine Annealing with Warm Restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # 最初のリスタートまでのエポック数
    T_mult=2  # リスタートごとにT_0を2倍
)

# ReduceLROnPlateau（Validation Lossベース）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)
```

---

## 🎯 推奨実装順序

### Phase A: 基礎固め（1-2週間）
1. ✅ **Step 2: Early Stopping実装**
   - Phase 1とPhase 2の両方
   - 既存のテストスクリプトに統合

2. ✅ **Step 3: Train/Validation分割**
   - 10サンプルで動作確認
   - Early Stoppingの効果検証

### Phase B: スケーリング（2-3週間）
3. ✅ **Step 1: 複数サンプル統合訓練**
   - 10 → 50 → 100サンプルで段階的にテスト
   - メモリ使用量とPerplexityを記録

4. ✅ **Step 4: キャッシュシステム拡張**
   - 複数サンプルの固有点管理
   - 再実験時の高速化

### Phase C: 最適化（3-4週間）
5. ✅ **Step 5: アーキテクチャスケーリング**
   - Mixed [2,2,2], [2,2,2,2], [3,3] を比較
   - 100サンプルで性能評価

6. ⏸️ **Step 6: 学習率スケジューリング**（オプション）
   - 必要に応じて実装

---

## 📊 成功の指標

### Phase 1（固有点学習）
- ✅ 収束率 > 95%
- ✅ 平均反復数 < 20
- ✅ Phase 1 Loss: 0.1 - 0.5（健全な範囲）

### Phase 2（トークン予測）
- 🎯 **Perplexity < 30**（10サンプル）
- 🎯 **Perplexity < 20**（100サンプル）
- 🎯 Train/Val Perplexityの差 < 5（過学習なし）
- 🎯 Accuracy > 30%

### 最終目標
- 🏆 **Perplexity < 15**（本格訓練後）
- 🏆 **実用的な対話生成能力**

---

## 💡 追加の提案

### データ拡張
- **Multiple Turns**: 対話の複数ターンを1サンプルとして扱う
- **Context Mixing**: 異なる対話を組み合わせて固有点の汎化性を向上

### モデル改善
- **Residual Connection**: 文脈ベクトル更新にResidualを追加
- **Layer Dropout**: ブロック単位でのDropout

### 評価指標
- **BLEU Score**: 生成品質の定量評価
- **Human Evaluation**: 対話の自然さの主観評価

---

## 🚀 次のアクション

### 即座に実装すべきもの
1. **Early Stopping**（Phase 1 & Phase 2）
2. **Train/Validation分割**
3. **10サンプルでの動作確認**

### 準備が整ったら実装
4. **複数サンプル統合訓練**（50-100サンプル）
5. **キャッシュシステム拡張**
6. **アーキテクチャスケーリング実験**

---

**最優先タスク**: Early Stopping + Train/Val分割の実装と検証
