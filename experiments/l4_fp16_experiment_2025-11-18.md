# WikiText-2 FP16 Training on L4 GPU - 2025-11-18

## 実験概要

**目的**: L4 GPU（Colab Pro）でFP16混合精度訓練を実行し、batch_size/learning_rateの最適化効果を検証

**実験日**: 2025年11月18日
**Git Commit**: `a458673` (Fix learning rate for L4 GPU batch_size - Linear Scaling Rule)
**実験環境**: Google Colab Pro - L4 GPU

## ハードウェア・ソフトウェア環境

### GPU情報
- **GPU**: NVIDIA L4
- **VRAM**: 22.2 GB
- **Precision**: FP16 Mixed Precision (torch.cuda.amp)

### ソフトウェア
- Python 3.x
- PyTorch with CUDA
- FP16 Automatic Mixed Precision (AMP)

**注意**: Deprecation警告が発生
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated.
Please use `torch.amp.GradScaler('cuda', args...)` instead.
```

## モデル・データ設定

### モデルアーキテクチャ
- **Model**: New-LLM Baseline (Context Vector Propagation)
- **Parameters**: 2,739,432 (2.74M)
- **Embed Dim**: 256
- **Hidden Dim**: 512
- **Num Layers**: 6
- **Context Vector Dim**: 256
- **Dropout**: 0.1

### データセット
- **Dataset**: WikiText-2
- **Vocabulary Size**: 1000
- **Max Sequence Length**: 64
- **Training Sequences**: 51,026
- **Validation Sequences**: 5,272
- **Training Texts**: 23,767
- **Validation Texts**: 2,461

## 訓練ハイパーパラメータ

### 最適化設定（Linear Scaling Rule適用）
- **Batch Size**: 2048（L4用に4倍拡大）
- **Learning Rate**: 0.0004（Linear Scaling Rule: 4x T4 baseline）
- **Optimizer**: Adam
- **Weight Decay**: 0.0
- **Gradient Clipping**: 1.0
- **Total Epochs**: 50
- **Early Stopping Patience**: 15

**Linear Scaling Rule**:
```
T4 (512 batch)  → learning_rate = 0.0001
L4 (2048 batch) → learning_rate = 0.0004 (4x)
```

### GPU使用率
- **Batch Size最適化前**: 5.5GB / 24GB（23%）
- **Batch Size最適化後**: ~22GB / 24GB（92%）

## 訓練結果

### 訓練時間
- **Total Training Time**: 0.30 hours（18分）
- **Time per Epoch**: ~0.36 minutes（21.6秒）
- **Speedup vs CPU**: 推定60-100倍速

### 最終性能
| Metric | Value | Epoch |
|--------|-------|-------|
| **Best Val Loss** | 3.4269 | 50 |
| **Best Val PPL** | **30.78** | 50 |
| **Best Val Accuracy** | 34.4% | 50 |

### 学習曲線サマリー

**初期段階（Epoch 1-5）**:
- Epoch 1: PPL 76.1 → Epoch 5: PPL 68.0
- 急速な初期収束

**中期段階（Epoch 6-20）**:
- Epoch 6: PPL 67.1 → Epoch 20: PPL 47.4
- PPL減少率: 約30%

**後期段階（Epoch 21-50）**:
- Epoch 21: PPL 46.0 → Epoch 50: PPL 30.8
- 安定した収束継続

### エポックごとの詳細

| Epoch | Train Loss | Val Loss | Val PPL | Val Acc | 備考 |
|-------|-----------|----------|---------|---------|------|
| 1 | 5.5457 | 4.33 | 76.1 | 30.5% | 初期状態 |
| 5 | 4.2217 | 4.22 | 68.0 | 30.5% | |
| 10 | 4.2080 | 4.24 | 69.1 | 30.5% | |
| 15 | 4.1633 | 4.14 | 63.0 | 31.0% | Checkpoint |
| 20 | 3.9234 | 3.86 | 47.4 | 30.6% | Checkpoint |
| 25 | 3.8070 | 3.80 | 44.8 | 31.3% | Checkpoint |
| 30 | 3.7268 | 3.71 | 40.8 | 31.7% | Checkpoint |
| 35 | 3.6732 | 3.65 | 38.5 | 32.4% | Checkpoint |
| 40 | 3.5733 | 3.55 | 34.7 | 33.2% | Checkpoint |
| 45 | 3.4984 | 3.48 | 32.3 | 34.0% | Checkpoint |
| **50** | **3.4432** | **3.43** | **30.8** | **34.4%** | **最終** |

### チェックポイント保存
- **Best Checkpoint**: `checkpoints/best_new_llm_wikitext_fp16.pt`
- **Final Checkpoint**: `checkpoints/new_llm_wikitext_fp16_final.pt`

## 過去実験との比較

### Baseline（CPU、2024-11-17）
| 実験 | 環境 | batch_size | learning_rate | Epochs | Best PPL | Best Acc | 備考 |
|------|------|-----------|--------------|--------|----------|----------|------|
| **Baseline** | CPU | 16 | 0.001 | 50 | **23.34** | 31.0% | 以前の最良結果 |
| **L4 FP16** | L4 GPU | 2048 | 0.0004 | 50 | **30.78** | 34.4% | 今回の実験 |

**分析**:
- **PPL**: L4実験は30.78（Baseline 23.34より劣る）
- **Accuracy**: 34.4%（Baseline 31.0%より改善）
- **訓練時間**: 18分（CPU版は推定20-30時間）

**PPLが劣る理由の仮説**:
1. Learning rate調整が不十分（0.0004でも不足？）
2. 大きなbatch_sizeによるgeneralization gap
3. FP16による数値精度の影響
4. Early stopping patienceが短い（15エポック）

## 発見と教訓

### ✅ 成功した点

1. **Linear Scaling Rule適用**: batch_size 4x → learning_rate 4x
   - 以前の実験（PPL 46.7 @ Epoch 29）から大幅改善
   - PPL 30.78達成

2. **GPU使用率最適化**: 23% → 92%
   - batch_size: 512 → 2048（4倍）
   - VRAM使用量: 5.5GB → 22GB

3. **訓練速度**: 18分で50エポック完了
   - 1エポック: 約21.6秒
   - CPU版比で推定60-100倍速

4. **Accuracy改善**: 34.4%（Baseline 31.0%より+3.4%）

### ⚠️ 改善が必要な点

1. **PPL**: 30.78 vs Baseline 23.34
   - まだ約33%劣っている
   - さらなる学習率調整が必要か？

2. **Deprecation警告**: PyTorch AMP APIの更新が必要
   ```python
   # 旧: torch.cuda.amp.GradScaler()
   # 新: torch.amp.GradScaler('cuda')
   ```

3. **学習の頭打ち**: Epoch 40以降の改善が緩やか
   - より長い訓練（100-150 epoch）で改善の余地あり

## 次のステップ

### 提案1: 100エポック延長訓練
- **スクリプト**: `train_wikitext_fp16_extended.py`（作成済み）
- **目標**: PPL 30.78 → 25以下
- **期待**: より長い訓練で収束改善

### 提案2: Learning Rate微調整
- 現在: 0.0004（4x baseline）
- 試行: 0.0002（2x baseline）または warmup追加
- 仮説: 大きすぎるLRがgeneralizationを阻害？

### 提案3: PyTorch API更新
```python
# 修正箇所
# Old: self.scaler = GradScaler()
# New: self.scaler = GradScaler('cuda')
```

### 提案4: より大きなモデル
- Advanced設定（context=512, layers=12）でL4 GPUを活用
- パラメータ数: 2.74M → 4.84M

## 設定ファイル参照

**Config Class**: `NewLLML4Config`（`src/utils/config.py`）
```python
class NewLLML4Config(NewLLMConfig):
    batch_size = 2048
    learning_rate = 0.0004  # Linear Scaling Rule適用済み
    device = "cuda"
```

**実験スクリプト**: `scripts/train_wikitext_fp16.py`

## まとめ

**成果**:
- ✅ L4 GPU最適化により92% VRAM使用率達成
- ✅ Linear Scaling Ruleで学習を正常化（PPL 46.7 → 30.78）
- ✅ 訓練速度60-100倍向上（18分で50エポック）
- ✅ Accuracy 34.4%（過去最高）

**課題**:
- ⚠️ PPL 30.78はBaseline（23.34）に劣る
- ⚠️ 100エポック延長訓練で改善の余地あり

**総合評価**: L4 GPUの性能を最大限活用し、高速訓練を実現。PPLは改善の余地があるが、設定の一元管理とLinear Scaling Ruleの適用により、再現性の高い実験環境を構築できた。

---

**実験者**: Claude Code
**記録日**: 2025-11-18
**Git Commit**: a458673f22ea0705604c4faafb8079931132636d
