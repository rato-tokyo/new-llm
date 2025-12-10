# Hierarchical Reasoning Model (HRM)

脳の階層的処理にインスパイアされた再帰的推論アーキテクチャ。

**論文**: "Hierarchical Reasoning Model" (arXiv:2506.21734v3, 2025年8月)
**コード**: [github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)

---

## 概要

HRMは、LLMのChain-of-Thought (CoT)に代わるアプローチとして提案された。

**CoTの問題点**:
- 中間ステップの教師データが必要
- 1ステップのミスで全体が破綻
- 大量のトークン生成 → 遅い

**HRMの解決策**:
- 中間表現の教師データ不要（End-to-End学習）
- 潜在空間での推論（latent reasoning）
- 階層的な再帰処理で計算深度を確保

---

## アーキテクチャ

### 4つのコンポーネント

```
入力 x
  ↓
┌─────────────────────────────────────┐
│  fI (Input Network)                 │  入力を作業表現に変換
│  x̃ = fI(x; θI)                      │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  fL (Low-level Module)              │  高速・詳細な計算
│  fH (High-level Module)             │  低速・抽象的な計画
│                                     │
│  N サイクル × T ステップ            │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  fO (Output Network)                │  隠れ状態から出力を生成
│  ŷ = fO(z_H^NT; θO)                 │
└─────────────────────────────────────┘
  ↓
出力 ŷ
```

### H-module と L-module の関係

```
タイムスケール:
  H-module: ゆっくり更新（Nサイクルで N回更新）
  L-module: 高速に更新（Nサイクルで N×T回更新）

更新ルール:
  z_L^i = fL(z_L^{i-1}, z_H^{i-1}, x̃; θL)    ← 毎ステップ更新

  z_H^i = fH(z_H^{i-1}, z_L^{i-1}; θH)       ← Tステップごとに更新
          z_H^{i-1}                           ← それ以外はそのまま
```

### 処理フロー（N=2, T=3 の例）

```
Step:  0    1    2    3    4    5    6
       │    │    │    │    │    │    │
L:     L₀ → L₁ → L₂ → L₃ → L₄ → L₅ → L₆
       │         │         │         │
H:     H₀ ────→ H₁ ────→ H₂ ────→ H₃
              (T=3で     (T=3で     (最終)
               更新)      更新)

サイクル1: L₀→L₁→L₂ の後に H₀→H₁
サイクル2: L₃→L₄→L₅ の後に H₁→H₂
出力: H₃ から生成
```

---

## 階層的収束（Hierarchical Convergence）

### 標準RNNの問題

```
標準RNN:
  h₀ → h₁ → h₂ → h₃ → h₄ → h₅ → ...

  問題: 早期に収束 → 後半のステップが無駄
  → 計算深度に限界
```

### HRMの解決策

```
HRM:
  L-module: 各サイクル内で収束（局所平衡）
  H-module: サイクル終了時に更新 → L-moduleをリセット

  → 収束と再開の繰り返しで計算深度を維持
```

```
Forward Residual（更新量）の比較:

標準RNN:    ▓▓▓▒▒░░░░░░░░░░░  （早期に減衰）
HRM L:      ▓▓▓░▓▓▓░▓▓▓░▓▓▓░  （リセットで回復）
HRM H:      ▓▒▒▒▓▒▒▒▓▒▒▒▓▒▒▒  （安定して収束）
```

---

## 1-step 勾配近似

### BPTTの問題

```
BPTT (Backpropagation Through Time):
  - メモリ: O(T) — 全ステップの状態を保存
  - 計算コスト: 高い
  - 生物学的に不自然
```

### HRMの解決策: 1-step 勾配

```
勾配パス:
  Output head → H-moduleの最終状態 → L-moduleの最終状態 → Input embedding

実装（PyTorch）:
  with torch.no_grad():
      for i in range(N*T - 1):
          zL = L_net(zL, zH, x)
          if (i + 1) % T == 0:
              zH = H_net(zH, zL)

  # 最後の1ステップのみ勾配を計算
  zL = L_net(zL, zH, x)
  zH = H_net(zH, zL)

  loss = criterion(output_head(zH), target)
  loss.backward()
```

### 理論的背景: Deep Equilibrium Models (DEQ)

```
固定点での勾配:
  z* = F(z*; θ)  （平衡状態）

  ∂z*/∂θ = (I - J_F)^{-1} · ∂F/∂θ

  1-step近似:
  (I - J_F)^{-1} ≈ I

  → ∂z*/∂θ ≈ ∂F/∂θ  （最後のステップの勾配のみ）
```

---

## Deep Supervision

### 概念

```
複数のforward pass（セグメント）を実行し、
各セグメントでlossを計算・更新

for segment in range(M):
    z, ŷ = HRM(z, x; θ)
    loss = criterion(ŷ, y)

    z = z.detach()  # 重要: 勾配を切断

    loss.backward()
    optimizer.step()
```

### 利点

- H-moduleに頻繁なフィードバック
- 正則化効果
- 安定した学習

---

## Adaptive Computational Time (ACT)

### 概念

```
タスクの難易度に応じて計算量を調整

簡単なタスク: 少ないセグメントで終了
難しいタスク: 多くのセグメントを使用
```

### 実装: Q-learning

```
Q-head: z_H から "halt" / "continue" の価値を予測

Q̂ = σ(θ_Q^T · z_H^{mNT})

報酬:
  halt時: 1{ŷ = y}  （正解なら1、不正解なら0）
  continue時: 0

Q-learning targets:
  Ĝ_halt = 1{ŷ = y}
  Ĝ_continue = max(Q̂_halt^{m+1}, Q̂_continue^{m+1})
```

### 効果

```
Sudoku-Extremeでの実測:

M_max=2:  平均1.5ステップ使用、精度85%
M_max=8:  平均3.0ステップ使用、精度98%

→ 難しい問題により多くの計算を割り当て
```

---

## 実験結果

### ベンチマーク

| タスク | HRM (27M) | o3-mini-high | Claude 3.7 | Deepseek R1 |
|--------|-----------|--------------|------------|-------------|
| ARC-AGI-1 | **40.3%** | 34.5% | 21.2% | 15.8% |
| ARC-AGI-2 | **5.0%** | 3.0% | 1.3% | 0.9% |
| Sudoku-Extreme | **55.0%** | 0.0% | 0.0% | 0.0% |
| Maze-Hard (30x30) | **74.5%** | 0.0% | 0.0% | 0.0% |

**注目点**:
- わずか1000サンプルで訓練
- 事前学習なし、CoTデータなし
- 27Mパラメータの小規模モデル

### 深度の重要性

```
Sudoku-Extreme Fullでの比較:

Transformerの幅を増やす（8層固定）:
  27M → 872M: 精度変化なし（~20%）

Transformerの深度を増やす（512 hidden固定）:
  8層 → 256層: 精度向上（20% → 45%）

HRM:
  64ステップ相当で精度95%以上達成
```

---

## 脳との対応

### 次元性階層（Dimensionality Hierarchy）

```
マウス皮質での観察:
  低次領域（感覚野）: 低次元表現
  高次領域（前頭前野）: 高次元表現

HRMでの観察:
  L-module (z_L): PR = 30.22（低次元）
  H-module (z_H): PR = 89.95（高次元）

比率: z_H/z_L ≈ 2.98（マウス皮質 ≈ 2.25 と類似）
```

**PR (Participation Ratio)**: 表現の実効次元数
```
PR = (Σλᵢ)² / Σλᵢ²
```

### 意味

- H-module: 高次元 → 多様なタスクに対応（前頭前野的）
- L-module: 低次元 → 特化した詳細計算（感覚野的）
- 訓練によって自然にこの構造が出現

---

## 実装詳細

### モジュール構成

```python
# H-module, L-module: 同一構造のTransformerブロック
# 入力の統合: element-wise addition

class HRM(nn.Module):
    def __init__(self):
        self.input_embed = Embedding()
        self.L_module = TransformerBlock()  # Llama風
        self.H_module = TransformerBlock()  # Llama風
        self.output_head = nn.Linear()
        self.Q_head = nn.Linear()  # ACT用

    def forward(self, x, z_H, z_L, N, T):
        x̃ = self.input_embed(x)

        for i in range(N * T):
            z_L = self.L_module(z_L + z_H + x̃)
            if (i + 1) % T == 0:
                z_H = self.H_module(z_H + z_L)

        return self.output_head(z_H)
```

### Transformerブロックの改良

- RoPE (Rotary Position Embedding)
- Gated Linear Units (GLU)
- RMSNorm
- バイアスなし
- Post-Norm（Pre-Normではない）
- Truncated LeCun Normal 初期化

### 最適化

- Adam-atan2（scale-invariant Adam）
- 定数学習率 + linear warm-up
- stablemax（小サンプル時、softmaxの代わり）

---

## 限界と今後の方向性

### 現在の限界

1. **直列処理**: H→L→H→L...（並列タスク分割には不向き）
2. **固定構造**: N, T は事前に決定
3. **自然言語生成**: 主にseq2seq形式、自由生成には工夫が必要

### 今後の拡張候補

- Linear Attentionの統合（長文脈対応）
- より洗練されたモジュール間通信
- 階層数の増加（3層以上）
- 言語モデルへの統合

---

## Senriプロジェクトとの関連

| Senri | HRM | 対応 |
|-------|-----|------|
| Working Memory | L-module state | 高速更新、詳細処理 |
| Index/Detail Memory | H-module state | 低速更新、抽象計画 |
| Linear Attention | Full Attention | HRMは効率より推論優先 |
| 圧縮メモリ | 隠れ状態 | 両者とも固定サイズ |

### 統合の可能性

```
Senri + HRM 的アプローチ:
  SenriLayer（圧縮メモリ） → L-module的役割
  追加の計画レイヤー → H-module的役割

  → 長文脈 + 深い推論の両立
```

---

## 参考文献

- Wang et al. (2025). "Hierarchical Reasoning Model". arXiv:2506.21734v3
- Bai et al. (2019). "Deep Equilibrium Models". NeurIPS
- Graves (2016). "Adaptive Computation Time for Recurrent Neural Networks"
- Dehghani et al. (2018). "Universal Transformers"

---

## バージョン

- v0.1.0 (2025-12-10): 初版、論文の要約
