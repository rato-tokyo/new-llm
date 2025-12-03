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
├── logs/                            # 実験ログ（日付別）
│   ├── 20251203_*.md               # 個別実験結果
│   └── old/                        # 古い実験ログ
└── old/                            # 古いドキュメント（アーカイブ）
```

---

## 読む順序（推奨）

1. **CASCADE_CONTEXT_ARCHITECTURE.md** - まずアーキテクチャを理解
2. **SCALING_LAW_ANALYSIS.md** - スケーリング則と理論限界値
3. **logs/20251203_*.md** - 最新の実験結果

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

## logs/ の使い方

個別実験の詳細結果は `logs/` に保存されます。
ファイル名規則: `YYYYMMDD_構成_実験内容.md`

例:
- `20251203_2block_sample_search.md` - 2-block構成のサンプルサイズ探索
- `20251203_2block_p2_sample_search.md` - 2-block + prev_context=2 の実験

---

*Last Updated: 2025-12-03*
