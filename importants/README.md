# importants/ ディレクトリ

**最終更新**: 2025-12-03

---

## ファイル構成

| ファイル | 内容 |
|----------|------|
| [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md) | 現在のアーキテクチャ・スケーリング則・推奨設定 |
| [LEGACY_FINDINGS.md](LEGACY_FINDINGS.md) | 過去の設計からの知見・失敗パターン |
| [GUIDE_colab_setup.md](GUIDE_colab_setup.md) | Google Colabセットアップガイド |
| `archive/` | 過去の実験ログ・古いドキュメント |

---

## Quick Reference

### 現在の構成

```
2-Block Cascade Context
- context_dim: 256 × 2 = 512
- 1層固定
- Embedding凍結 + Weight Tying
- 飽和モデル: PPL_min = 95.4 (p=0), 87.3 (p=2)
```

### 主要な実験結果

| Samples | Val PPL | Val Acc |
|---------|---------|---------|
| 1600 | 127.4 | 24.5% |
| 3200 | 114.8 | 25.0% |

### 実験コマンド

```bash
python3 scripts/experiment_cascade_context.py -s 2000
```

---

## 読む順序

1. **CURRENT_ARCHITECTURE.md** - まず現在の設計を理解
2. **LEGACY_FINDINGS.md** - 過去の知見と失敗パターンを参考に

---

*Last Updated: 2025-12-03*
