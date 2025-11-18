# Training Resume Guide - 訓練再開ガイド

## 概要

訓練の途中で中断した場合、チェックポイントから再開できます。

---

## 🔄 自動保存されるチェックポイント

訓練中、以下のチェックポイントが自動保存されます：

| ファイル名 | 保存タイミング | 内容 |
|-----------|--------------|------|
| `best_{model_name}.pt` | Validation Lossが改善時 | **最良モデル** |
| `{model_name}_epoch_5.pt` | 5エポックごと | 定期チェックポイント |
| `{model_name}_final.pt` | 訓練完了時 | **最終モデル** |

### チェックポイントの内容

各`.pt`ファイルには以下が保存されています：
- モデルパラメータ (`model_state_dict`)
- Optimizer状態 (`optimizer_state_dict`)
- 訓練履歴 (`train_losses`, `val_losses`, `train_ppls`, `val_ppls`)
- 現在のエポック数 (`current_epoch`)
- 設定 (`config`)

---

## 🚀 訓練再開の方法

### 方法1: スクリプト内で指定

```python
# scripts/train_wikitext_advanced.py の例

from src.training.trainer import Trainer

# Trainer作成
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    config=config,
    model_name="new_llm_wikitext_ctx512_layers12"
)

# 訓練再開（チェックポイントから）
trainer.train(
    resume_from="new_llm_wikitext_ctx512_layers12_epoch_25.pt"
)
```

### 方法2: Trainer作成後に手動ロード

```python
from src.training.trainer import Trainer

trainer = Trainer(...)

# チェックポイントから状態を復元
start_epoch = trainer.resume_from_checkpoint("best_new_llm_wikitext.pt")

# 訓練実行（自動的に続きから開始）
trainer.train()
```

---

## 📋 使用例

### 例1: Google Colabで90分制限に引っかかった場合

```python
# 1回目の実行（Epoch 25まで進んだところで切断）
trainer.train()  # → epoch_25.pt が保存される

# 2回目の実行（再接続後）
trainer = Trainer(...)
trainer.train(resume_from="new_llm_wikitext_epoch_25.pt")
# → Epoch 26から再開
```

### 例2: 最良モデルから再訓練

```python
# より長く訓練したい場合
trainer = Trainer(...)
trainer.train(
    num_epochs=100,  # 50 → 100に延長
    resume_from="best_new_llm_wikitext.pt"
)
```

### 例3: ファインチューニング用に最良モデルをロード

```python
# WikiTextで訓練したbestモデルをDailyDialogでファインチューニング
trainer = Trainer(...)
trainer.resume_from_checkpoint("best_new_llm_wikitext.pt")
trainer.train()  # 異なるデータセットで続きから訓練
```

---

## 🔍 チェックポイントの確認

利用可能なチェックポイントを確認：

```bash
ls -lh checkpoints/*.pt
```

最新のチェックポイントを確認：

```bash
ls -lt checkpoints/*.pt | head -5
```

---

## 💡 Tips

### 1. 定期チェックポイントの活用

5エポックごとに保存されるので、任意の時点から再開可能：

```python
# Epoch 15から再開
trainer.train(resume_from="new_llm_wikitext_epoch_15.pt")

# Epoch 20から再開
trainer.train(resume_from="new_llm_wikitext_epoch_20.pt")
```

### 2. bestモデルの優先使用

通常は `best_{model_name}.pt` を使うのが推奨：

```python
# 最良性能のモデルから再開
trainer.train(resume_from="best_new_llm_wikitext.pt")
```

### 3. 訓練完了後の追加訓練

`final.pt` を使って追加訓練が可能：

```python
# 50エポックで終了したモデルを100エポックまで延長
trainer.train(
    num_epochs=100,
    resume_from="new_llm_wikitext_final.pt"
)
```

---

## ⚠️ 注意事項

### 1. モデル構成の一致

チェックポイントをロードする際、**モデルの構成（config）が一致**している必要があります：

```python
# ✓ OK - 同じ構成
config = AdvancedConfig()  # context_vector_dim=512, num_layers=12
model = ContextVectorLLM(config)
trainer = Trainer(model, ...)
trainer.train(resume_from="new_llm_wikitext_ctx512_layers12_epoch_10.pt")

# ✗ NG - 異なる構成
config = AdvancedConfig()
config.context_vector_dim = 1024  # 512 → 1024に変更
model = ContextVectorLLM(config)
trainer = Trainer(model, ...)
trainer.train(resume_from="new_llm_wikitext_ctx512_layers12_epoch_10.pt")
# → エラー: モデル構造が一致しない
```

### 2. デバイスの一致

CPU/GPUは自動的に対応されますが、念のため確認：

```python
# チェックポイントはCPU/GPU間で共有可能
config.device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 3. early stoppingのリセット

訓練再開時、early stoppingのカウンターはリセットされます。

---

## 📊 再開時の表示例

```
============================================================
Resuming from checkpoint: new_llm_wikitext_epoch_25.pt
============================================================
Completed epochs: 25
Best Val Loss so far: 3.1563
Best Val PPL so far: 23.48
============================================================

Resuming from epoch 26

============================================================
Training new_llm_wikitext
============================================================

Epoch 26/50
  Training... 20% 40% 60% 80% 100% | 0.6min | Loss: 3.1200
  Val: Loss=3.14 PPL=23.1 Acc=31.2% ✓
```

---

## 🎯 まとめ

- ✅ **自動保存**: bestモデルと定期チェックポイントが自動保存される
- ✅ **簡単再開**: `resume_from="checkpoint.pt"` で1行で再開
- ✅ **履歴保持**: 訓練履歴も復元されるので、グラフも継続描画
- ✅ **柔軟性**: 任意のエポックから再開可能

**訓練が中断しても安心して再開できます！** 🚀
