# Scripts Directory

実験スクリプト。

## メイン実験スクリプト

### experiment_cascade_context.py

カスケード連結（N分割方式）での実験。複数のContextBlockを異なるデータで学習。

```bash
# 基本: 2000サンプル、2ブロック
python3 scripts/experiment_cascade_context.py -s 2000 -n 2

# 4ブロック
python3 scripts/experiment_cascade_context.py -s 2000 -n 4

# context_dim指定
python3 scripts/experiment_cascade_context.py -s 2000 -n 2 -c 256
```

### experiment_multiblock_sample_search.py

サンプル数を変化させてPPLの変化を観察。指数減衰モデルでPPL_minを自動推定。

```bash
# 基本: 200-1600サンプル、2ブロック
python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 1600 -c 256

# 前のcontextも連結（prev_context_steps=1）
python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 1600 -c 256 -p 1

# 4ブロック
python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 1600 -c 256 -n 4
```

**オプション**:
- `-c, --context-dim`: 各ブロックのcontext次元（デフォルト: 256）
- `-n, --num-blocks`: ContextBlock数（デフォルト: 2）
- `--start`: 開始サンプル数（デフォルト: 200）
- `--end`: 終了サンプル数（デフォルト: 1600）
- `-p, --prev-context`: 前のcontextを連結する数（0で無効、デフォルト: 0）
- `-o, --output`: 出力ディレクトリ

## 共通モジュール

スクリプトで使用する共通コードは `src/` に移動済み:

- `src/models/cascade.py` - CascadeContextLLM, SingleContextWrapper
- `src/trainers/phase2/cascade.py` - CascadePhase2Trainer
- `src/utils/cache.py` - キャッシュ収集ユーティリティ
