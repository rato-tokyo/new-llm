# importants/ ディレクトリ構成

**最終更新**: 2025-12-03

---

## メインドキュメント（importants/直下）

| ファイル | 内容 | 重要度 |
|----------|------|--------|
| [CASCADE_CONTEXT_ARCHITECTURE.md](CASCADE_CONTEXT_ARCHITECTURE.md) | アーキテクチャ設計・設計原則 | ★★★ |
| [SCALING_LAW_ANALYSIS.md](SCALING_LAW_ANALYSIS.md) | スケーリング則分析（飽和モデル） | ★★★ |
| [CONTEXT_DIM_SEARCH.md](CONTEXT_DIM_SEARCH.md) | context_dim探索結果（1-block、参考用） | ★★ |
| [GUIDE_colab_setup.md](GUIDE_colab_setup.md) | Google Colabセットアップガイド | ★★ |

---

## ディレクトリ構成

```
importants/
├── README.md                        # このファイル
├── CASCADE_CONTEXT_ARCHITECTURE.md  # アーキテクチャ設計
├── SCALING_LAW_ANALYSIS.md          # スケーリング則分析
├── CONTEXT_DIM_SEARCH.md            # context_dim探索（1-block参考）
├── GUIDE_colab_setup.md             # Colabセットアップ
├── logs/                            # 実験ログ
│   ├── 20251203_2block_p0_complete.md        # p=0完全結果（統合版）
│   ├── 20251203_prev_context_comparison.md   # prev_context比較（統合版）
│   ├── 20251203_prev_context_interval8_count2.md  # interval=8（失敗例）
│   └── old/                         # 統合前の個別ログ
└── old/                             # 古いドキュメント（アーカイブ）
```

---

## 読む順序（推奨）

1. **CASCADE_CONTEXT_ARCHITECTURE.md** - まずアーキテクチャを理解
2. **SCALING_LAW_ANALYSIS.md** - スケーリング則と理論限界値
3. **logs/** - 最新の実験結果

---

## 主要な発見（2025-12-03時点）

### 飽和モデルが最適

```
PPL = PPL_min + A × n^(-a)
```

| 構成 | PPL_min | 1600 samples PPL |
|------|---------|------------------|
| 2-block (p=0) | 95.4 | 127.4 |
| 2-block (p=2) | 87.3 | 114.7 |

### 主要な設計原則

- **1層固定**: multi-layerは効果なし
- **2-block カスケード連結**: 1-blockより優れている
- **prev_context_steps**: 連続履歴（interval=1）が有効
- **飽和モデル**: 理論限界値が存在（データ増加では突破不可）

---

## logs/ ファイル一覧

| ファイル | 内容 |
|----------|------|
| `20251203_2block_p0_complete.md` | p=0の完全結果（200-3200samples、飽和モデル判定含む） |
| `20251203_prev_context_comparison.md` | p=0/p=1/p=2の比較まとめ |
| `20251203_prev_context_interval8_count2.md` | interval=8の実験（効果なし、失敗例として保持） |

---

*Last Updated: 2025-12-03*
