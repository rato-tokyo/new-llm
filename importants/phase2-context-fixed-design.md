# Phase 2: Context-Fixed Token Prediction Design

**作成日**: 2025-11-25
**更新日**: 2025-11-26
**バージョン**: v3.1 (「固定」と「近づく」の区別を明確化)

## 概要

Phase 2はPhase 1で学習した文脈表現を使用して、次トークン予測を学習するフェーズ。

## 「固定」と「近づく」の違い - 重要

**⚠️ これらは全く異なる概念**:

| 概念 | 意味 | context_outの挙動 |
|------|------|-------------------|
| **固定** | context_outがC*から離れない | 変化しない（C*のまま） |
| **近づく** | context_outが変化した後C*に戻る | 一度離れてから近づく（間違い） |

**固定の正しい理解**:
- context_outは**そもそもC*から離れない**
- 「近づく」余地がない（最初からC*に等しい）
- context_stability_lossはこの「離れない」状態を維持する

## 2段階処理

### Stage 1: C*の生成（パラメータ更新なし）

Phase 2開始時に1回だけ実行。訓練データの全トークンを順次処理し、固定文脈ベクトルC*を生成。

```python
with torch.no_grad():
    context = torch.zeros(...)
    C_star = []
    for token_id in token_ids:
        token_embed = get_embedding(token_id)
        context, _ = cvfp_block(context, token_embed)
        C_star.append(context)
```

### Stage 2: 学習（パラメータ更新あり）

```python
for i, token_id in enumerate(input_ids):
    # 入力
    input_context = C_star[i-1] if i > 0 else zero_vector
    token_embed = get_embedding(token_id)

    # CVFPブロック処理
    context_out, token_out = cvfp_block(input_context, token_embed)

    # 予測
    combined = torch.cat([context_out, token_out], dim=-1)
    logits = token_output(combined)

    # 損失（2つの損失を組み合わせ）
    prediction_loss = CrossEntropy(logits, target[i+1])
    context_stability_loss = MSE(context_out, C_star[i])  # C*から離れないようにする
    total_loss = prediction_loss + λ * context_stability_loss
```

## context_stability_loss の役割

**目的**: `context_out = C*[i]` を維持する（離れないようにする）

**なぜ必要か**:
1. パラメータ更新により`context_out`がC*から離れようとする
2. context_stability_lossがこの「離れ」を阻止する
3. 結果: context_outは常にC*に等しいまま

**固定が成功している状態**:
- `context_out = C*[i]`が常に成り立つ
- context_outは変化しない（C*から離れない）
- バッチ並列化が可能

## 記号定義

| 記号 | 意味 |
|------|------|
| `C*[i]` | Stage 1で計算した固定目標文脈（不変） |
| `context_out` | Stage 2でCVFPブロックが出力する文脈（C*[i]に固定される） |
| `token_out` | Stage 2でCVFPブロックが出力するトークン表現（予測に使用） |

## 勾配フロー

```
入力: [C*[i-1], token_embed[i]]
         ↓
    CVFPブロック
         ↓
出力: [context_out, token_out]
         ↓
    prediction_loss = CrossEntropy(logits, target)
    context_stability_loss = MSE(context_out, C*[i])  ← C*から離れないようにする
    total_loss = prediction_loss + λ * context_stability_loss
```

**勾配の流れ**:
- ✅ `token_out` → CVFPブロック（prediction_loss経由）
- ✅ `context_out` → CVFPブロック（context_stability_loss経由 - 離れないように制約）
- ✅ `token_output`層

## 現在の実装状況（要修正）

**問題**: 現在の実装はcontext_stability_lossがなく、context_outを無視してC*で上書きしている。

```python
# 問題のあるコード（現在の実装）
batch_fixed_contexts = C_star[i].detach()  # C*を取得
combined = torch.cat([batch_fixed_contexts, token_out], dim=-1)  # context_outを無視
```

**修正が必要**:
- `context_stability_loss = MSE(context_out, C*[i])`を追加（C*から離れないようにする）
- `combined`にはcontext_outを使用

## 変更履歴

- **2025-11-25**: 初版作成
- **2025-11-26**: v2.0 - 「完全固定」方式（誤った設計）
- **2025-11-26**: v3.0 - context_stability_lossによる固定方式に修正
- **2025-11-26**: v3.1 - 「固定」と「近づく」の区別を明確化
