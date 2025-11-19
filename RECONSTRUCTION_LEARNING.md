# Context Reconstruction Learning - New-LLM v3

## Core Concept

**文脈ベクトルの正解 = 「前の文脈 + 現在のトークン」を圧縮したもの**

これはオートエンコーダー的な学習で、教師データ不要で実装可能。

## Architecture

### Example: "赤いリンゴ"

**t=1**:
- **入力**:
  - 文脈ベクトル: `[0]` (256次元のゼロベクトル)
  - トークン埋め込み: `embed("赤い")` (256次元)
- **出力の正解**:
  - トークン: `"リンゴ"` (次のトークン)
  - 文脈ベクトル: `compress([0, embed("赤い")])` (512次元→256次元に圧縮)

### Loss Functions

1. **Token Loss**: `CrossEntropy(predicted_token, next_token)`
2. **Reconstruction Loss**: `MSE(decoded_context, [prev_context, current_token])`
3. **Total Loss**: `token_loss + λ * reconstruction_loss` (λ=1.0)

## Model Components

### 1. ContextVectorLLM (既存)
- Token embedding (256次元)
- Context vector (256次元)
- FNN layers
- Token output head
- Context update head (with gating)

### 2. Context Decoder (NEW)
```python
context_decoder = nn.Sequential(
    nn.Linear(256, 512),  # 256 (context) → 512 (context + embed)
    nn.ReLU(),
    nn.Linear(512, 512),
)
```

## Training Process

### Forward Pass
```
for t in range(seq_len):
    # 1. 復元ターゲット作成
    target = concat([prev_context, token_embed[t]])  # 512次元

    # 2. FNN forward
    fnn_input = concat([token_embed[t], prev_context])
    hidden = FNN(fnn_input)

    # 3. Token prediction
    token_logits = token_head(hidden)

    # 4. Context update
    new_context = update_context(hidden, prev_context)

    # 5. Reconstruction (for loss)
    reconstructed = context_decoder(new_context)
    reconstruction_loss += MSE(reconstructed, target)
```

### Loss Computation
```python
token_loss = CrossEntropy(predicted_tokens, true_tokens)
reconstruction_loss = MSE(reconstructed, targets)
total_loss = token_loss + reconstruction_loss
```

## Benefits

1. **No External Teacher**: 教師データ不要
2. **Interpretable**: 文脈ベクトルが何を圧縮しているか明確
3. **Flexible**: context_dimの大小で、長文重視 vs 直近重視を調整可能
4. **Simple**: 実装がシンプル

## Implementation (PyTorch Only)

### Dependencies
- `torch` - ニューラルネットワーク
- `tokenizers` (HuggingFace) - BPE tokenizer
- `datasets` (HuggingFace) - データセット読み込み
- `tqdm` - プログレスバー

### Training Script Structure
```python
# 1. Data loading
texts = load_wikitext()
tokenizer = train_bpe_tokenizer(texts)
dataset = tokenize(texts, tokenizer)

# 2. Model
model = ContextVectorLLM(config)

# 3. Training loop
for epoch in range(epochs):
    for batch in dataloader:
        # Forward
        logits, context_trajectory = model(input_ids)

        # Token loss
        token_loss = cross_entropy(logits[:-1], labels[1:])

        # Reconstruction loss
        reconstructed = model.context_decoder(context_trajectory)
        targets = create_targets(context_trajectory, token_embeds)
        recon_loss = mse_loss(reconstructed, targets)

        # Combined loss
        loss = token_loss + recon_loss
        loss.backward()
        optimizer.step()
```

## Expected Results

### Success Criteria
- Token loss: 順調に減少
- Reconstruction loss: 順調に減少
- Perplexity: < 100 (目標)
- Context vectorが意味のある情報を保持

### Monitoring
- Epoch毎のloss, perplexity, accuracy
- Token loss vs Reconstruction loss のバランス

## Parameter Tuning

### Context Loss Weight (λ)
- λ=1.0: Token prediction と reconstruction を同等重視 (推奨)
- λ>1.0: Reconstruction を重視
- λ<1.0: Token prediction を重視

### Context Dimension
- 256: バランス型 (デフォルト)
- 512: 長文重視
- 128: 直近重視

## Next Steps

1. ✅ Context decoder を実装 (完了)
2. ✅ Forward passで reconstruction targets を保存 (完了)
3. 🔄 PyTorchのみの訓練スクリプト作成 (進行中)
4. ⏳ テスト実行 (1 layer, 2 epochs)
5. ⏳ 評価・改善
