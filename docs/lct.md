# Lagged Cache Training (LCT)

再帰的依存関係を持つモデルを並列処理可能にする訓練手法。

---

## 概要

「前のイテレーションの出力を、今回のイテレーションの入力として使用する」訓練方式。

RNN的構造（h_{t-1} → h_t）を持つモデルは本来シーケンシャル処理が必要だが、LCTにより並列処理が可能になる。

---

## 核心: OACD（Origin-Anchored Centroid Dispersion）

**LCTの安定性を大幅に向上させる重要なアルゴリズム。**

### 定義

```python
def oacd_loss(contexts, centroid_weight=0.1):
    """
    Origin-Anchored Centroid Dispersion

    Args:
        contexts: [batch, hidden_size] - 各サンプルのコンテキストベクトル
        centroid_weight: 重心固定の強さ（デフォルト: 0.1）

    Returns:
        loss: 多様性損失 + 重心固定損失
    """
    # 重心を計算
    context_mean = contexts.mean(dim=0)

    # 分散最大化: 重心からの偏差を最大化
    deviation = contexts - context_mean
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)

    # 重心固定: 重心を原点に引き寄せ
    centroid_loss = torch.norm(context_mean, p=2) ** 2

    return dispersion_loss + centroid_weight * centroid_loss
```

### なぜOACDが必要か

LCTは「1イテレーション遅れのキャッシュ」を使う近似手法。

**OACDなしの問題**:
- 重心がドリフトして不安定になりやすい
- 学習が不安定になる可能性

**OACDありの効果**:
- **重心が原点に固定** → 安定した平衡点
- **分散が適度に保たれる** → 表現力を維持
- **学習が安定** → チューニングが容易

### 実験結果（重要）

**OACDを適用すると、LCTの学習が安定する。**

| 設定 | 安定性 | 備考 |
|------|--------|------|
| OACDあり | ✅ 安定 | 推奨 |
| OACDなし | △ | 収束可能だが不安定になりやすい |

**centroid_weight=0.1** で安定。これより大きいと表現力が落ち、小さいと不安定になる。

---

## 問題: 再帰的モデルの訓練は遅い

```
通常の再帰訓練（シーケンシャル）:
  for t in range(seq_len):
      h_t = model(h_{t-1}, x_t)  # 1トークンずつ処理
      # → O(seq_len) 回のforward pass
```

各ステップが前のステップの出力に依存するため、並列化できない。

---

## 解決策: Lagged Cache Training

**核心アイデア**: 前のイテレーション（epoch）で計算した隠れ状態を、今回のイテレーションの入力として使用する。

```python
hidden_caches = {}  # バッチごとのキャッシュ

for epoch in epochs:
    for batch_idx, (input_ids, labels) in enumerate(train_loader):

        # 1. キャッシュ初期化（初回のみ）
        if batch_idx not in hidden_caches:
            with torch.no_grad():
                hidden_caches[batch_idx] = model.forward(input_ids).detach()

        # 2. 前イテレーションのキャッシュを入力として使用
        prev_cache = hidden_caches[batch_idx]

        # 3. 並列処理で全シーケンスを一度に計算
        output, new_hidden = model.forward_with_cache(input_ids, prev_cache)

        # 4. タスク損失 + OACD損失
        task_loss = compute_loss(output, labels)
        diversity_loss = oacd_loss(new_hidden, centroid_weight=0.1)
        loss = task_loss + diversity_loss

        # 5. キャッシュを更新（次イテレーション用）
        hidden_caches[batch_idx] = new_hidden.detach()

        loss.backward()
        optimizer.step()
```

---

## なぜ動作するのか

### 1. OACDによる安定化（最重要）

OACDが重心を原点に固定するため、キャッシュが安定した平衡点に収束する。

```
OACD効果:
  重心 → 原点に収束
  分散 → 適度に維持
  → キャッシュが安定した状態に収束
```

### 2. 近似の仮定

イテレーションごとのパラメータ変化は微小。

```
θ_{t+1} = θ_t - lr * ∇L

lr が小さければ、θ の変化も小さい
→ h_t の変化も小さい
→ prev_cache ≈ 真の h_{t-1}（1イテレーション遅れ）
```

### 3. キャッシュの追従

学習が進むとキャッシュも真の値に追従する。

```
Epoch 1: cache = 初期値（ランダム）
Epoch 2: cache = Epoch 1 の出力
Epoch 3: cache = Epoch 2 の出力
...
→ OACDにより安定した平衡点に収束
```

### 4. 計算効率

| 方式 | Forward Passes / Batch | 計算量 |
|------|------------------------|--------|
| シーケンシャル | O(seq_len) | 遅い |
| LCT | O(1) | 速い |

---

## 実装の詳細

### forward_with_cache の実装例

```python
def forward_with_cache(self, input_ids, prev_cache):
    """
    Args:
        input_ids: [batch, seq_len]
        prev_cache: [batch, seq_len, hidden_size] - 前イテレーションの隠れ状態

    Returns:
        output: [batch, seq_len, vocab_size]
        new_hidden: [batch, seq_len, hidden_size] - 今回の隠れ状態（次イテレーション用）
    """
    # 入力埋め込み
    x = self.embed(input_ids)  # [batch, seq_len, hidden_size]

    # 前の隠れ状態をシフトして結合
    shifted_cache = torch.zeros_like(prev_cache)
    shifted_cache[:, 1:, :] = prev_cache[:, :-1, :]  # 1つ右にシフト

    # 入力と前の隠れ状態を結合（加算方式）
    combined = x + shifted_cache

    # Transformer layers
    hidden = combined
    for layer in self.layers:
        hidden = layer(hidden)

    # 出力
    output = self.lm_head(hidden)

    return output, hidden
```

### 重要: detach() の使用

```python
# キャッシュは計算グラフから切り離す
hidden_caches[batch_idx] = new_hidden.detach()

# detach() しないと:
# - メモリリーク（計算グラフが累積）
# - 二重backward エラー
```

---

## HRMへの適用

HRM（Hierarchical Reasoning Model）は H-module と L-module の再帰的やり取りを持つ。

### HRMの再帰構造

```
Step 1: z_L = L(x̃ + z_H)
Step 2: z_L = L(x̃ + z_H + z_L)
...
Step T: z_H = H(z_L)  # H-module更新
Step T+1: z_L = L(x̃ + z_H)  # リセット
...
```

### LCT + OACD 適用

```python
h_caches = {}  # batch_idx → z_H

for epoch in epochs:
    for batch_idx, batch in enumerate(train_loader):

        # 前イテレーションの H-module 状態
        if batch_idx not in h_caches:
            h_caches[batch_idx] = torch.zeros(batch_size, hidden_size)

        prev_z_H = h_caches[batch_idx]

        # L-module を T ステップ実行（並列化可能）
        z_L = run_l_module_parallel(x, prev_z_H, T_steps)

        # H-module 更新
        new_z_H = h_module(z_L)

        # タスク損失 + OACD損失
        output = output_network(new_z_H)
        task_loss = criterion(output, target)
        diversity_loss = oacd_loss(new_z_H, centroid_weight=0.1)
        loss = task_loss + diversity_loss

        # キャッシュ更新
        h_caches[batch_idx] = new_z_H.detach()

        loss.backward()
        optimizer.step()
```

---

## 性能比較（実測値）

Senri プロジェクトでの実測:

| 方式 | 時間/epoch | 速度比 |
|------|-----------|--------|
| シーケンシャル | 450s | 1x |
| LCT + OACD | 20s | **22x** |

---

## 注意点

### 1. OACDは推奨

OACDを使うと学習が安定する。推奨。

```python
# ✅ 推奨: OACDあり
loss = task_loss + oacd_loss(hidden, centroid_weight=0.1)

# △ OACDなし（収束可能だが不安定になりやすい）
loss = task_loss
```

### 2. centroid_weight の調整

| 値 | 効果 |
|----|------|
| 0.01 | 不安定、ドリフトしやすい |
| **0.1** | **推奨**、安定した学習 |
| 1.0 | 表現力低下、すべてが原点に近づく |

### 3. バッチ順序の固定

LCT はバッチごとにキャッシュを管理するため、バッチの順序を固定する必要がある。

```python
# shuffle=False が必須
train_loader = DataLoader(dataset, shuffle=False, ...)
```

### 4. エポック間のキャッシュ引き継ぎ

キャッシュはエポック間で引き継ぐ。リセットしない。

```python
# ❌ エポックごとにリセット（ダメ）
for epoch in epochs:
    hidden_caches = {}  # リセット
    ...

# ✅ エポック間で引き継ぐ（正しい）
hidden_caches = {}
for epoch in epochs:
    ...  # キャッシュは引き継がれる
```

---

## まとめ

| 要素 | 役割 |
|------|------|
| **LCT** | 再帰モデルの並列訓練を可能にする |
| **OACD** | LCTの安定性を向上（推奨） |
| **detach()** | 計算グラフの切断 |
| **shuffle=False** | バッチ順序固定 |

**OACDを使うとLCTの学習が安定する。** 推奨設定。

---

## 適用可能なモデル

- 任意のRNN的再帰構造（h_{t-1} → h_t）
- HRM（Hierarchical Reasoning Model）
- DEQ（Deep Equilibrium Models）
- Context-Pythia（context_{t-1} → context_t）
- 前の出力を次の入力として使う任意のモデル

---

## 参考

- OACD定義: `importants/archive/old/SUMMARY_algorithms.md`
- 実験ログ: `importants/LEGACY_FINDINGS.md`

---

## バージョン

- v1.0.0 (2025-12-10): 初版
- v1.1.0 (2025-12-10): OACD追加、収束の保証について記載
