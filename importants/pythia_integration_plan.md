# Pythia統合計画書

## 現状（2025-12-04）

Pythia-70Mへの統合を試みたが、Phase 1のOACD学習が収束しなくなった。
一旦、動作していた`ef338d1`コミットに戻し、再度統合を試みる。

---

## 失敗の原因分析

### 問題点
1. **収束率が0%のまま**: diffの値（平均2.0）がthreshold（0.03）より圧倒的に大きい
2. **学習が発散**: lossが負から正に振動し、収束しない

### 試した修正（全て効果なし）
- `convergence_threshold`を0.001から0.03に変更
- `context_noise = 0.1`追加
- `gradient_clip = 2.0`追加
- ContextBlockを2層FFNから1層FFNに変更
- 初期化をXavierからnormal_(std=0.1)に変更

### 根本原因（推測）
- Pythia統合時にContextBlockの構造は同じにしたつもりだが、何か見落としがある
- token embeddingsの取得方法が異なる可能性
- 学習ループの構造が微妙に異なる可能性

---

## 戻す先のコミット

```
ef338d1 Refactor: remove unused import and vectorize build_batch_context_chunks
```

このコミットでは:
- Phase 1 OACD学習が正常に収束（90%+）
- Context-KV Attentionアーキテクチャが動作
- GPT-2 embeddingsを使用

---

## Pythia統合の目標（再実装時の指針）

### アーキテクチャ目標

```
Context-Pythia:
  Token Embedding (512-dim) ← Pythia-70M
       ↓
  ContextBlock: 512 → 256 (圧縮)
       ↓
  Layer 0-5: 全て context (256-dim) を入力
       ↓
  Output Head (vocab_size=50304)
```

### KVキャッシュ削減目標
- 元: hidden_size (512) × seq_len × num_layers (6)
- 目標: context_dim (256) × seq_len × num_layers (6)
- **削減率: 50%**

### Pythia-70M仕様

| 項目 | 値 |
|------|-----|
| Hidden Size | 512 |
| Layers | 6 |
| Attention Heads | 8 |
| Intermediate Size | 2048 |
| Position Encoding | Rotary (RoPE, 25%) |
| Vocab Size | 50,304 |
| Parallel Attention | Yes |

### 評価指標
- PPL (Perplexity)
- LAMBADA Accuracy
- KV Cache Memory使用量

---

## 再統合時の注意点

### 1. Phase 1の必須機能（削除禁止）

| 機能 | 説明 |
|------|------|
| OACD損失 | `src/losses/diversity.py` |
| 収束率計算 | 各イテレーションで`conv=XX%`表示 |
| Early Stopping | 収束率90%で停止 |
| 勾配累積 | 複数バッチの勾配を累積 |
| CPU/GPUメモリ分離 | contextをCPUに保存 |
| context_noise | 0.1のガウシアンノイズ |
| gradient_clip | 2.0 |

### 2. ContextBlockの実装

```python
# 動作していた実装（ef338d1）
class ContextLayer:
    def __init__(self, context_input_dim, context_output_dim, token_input_dim):
        input_dim = context_input_dim + token_input_dim
        self.fnn = FFN(input_dim, context_output_dim)  # Linear + GELU
        self.context_norm = LayerNorm(context_output_dim)
        # init: normal_(std=0.1), bias: normal_(std=0.01)

    def forward(self, context, token_embeds):
        fnn_input = cat([context, token_embeds])
        delta_context = self.fnn(fnn_input)
        return self.context_norm(context + delta_context)
```

### 3. 段階的統合戦略

**Step 1**: 動作確認（ef338d1に戻す）
- Phase 1が90%収束することを確認

**Step 2**: Embeddingsの置換
- GPT-2 → Pythia embeddingsに置換
- Phase 1の収束を確認

**Step 3**: Attention Layerの統合
- ContextBlock出力をPythia Layerに入力
- Phase 2の動作確認

**Step 4**: 全体評価
- PPL、LAMBADA、メモリ使用量を測定

---

## 参考コード

### 動作していたPhase 1設定（ef338d1）

設定は`config/config.py`にあった：
```python
phase1_learning_rate = 0.002
phase1_max_iterations = 60
phase1_early_stopping_rate = 0.90
phase1_convergence_threshold = 0.03
phase1_batch_size = 5000
phase1_context_noise = 0.1
phase1_gradient_clip = 2.0
```

---

## 今後のTODO

1. [ ] `ef338d1`に戻してPhase 1動作確認
2. [ ] Phase 1が90%収束することを確認
3. [ ] Pythia embeddingsを段階的に統合
4. [ ] 統合後もPhase 1が収束することを確認
5. [ ] Phase 2の実装
6. [ ] 全体評価

---

Last Updated: 2025-12-04
