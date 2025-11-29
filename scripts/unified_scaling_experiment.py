#!/usr/bin/env python3
"""
統一設定でのスケーリング実験スクリプト

目的:
- 11/27と11/28の実験条件の違い（トークン化設定）を解消
- 統一した設定（truncation=False）で複数サンプル数での実験を実施
- 正確なα値を導出

使用方法:
  Colab: !python scripts/unified_scaling_experiment.py
  Local: python3 scripts/unified_scaling_experiment.py

設定:
- トークン化: truncation=False（全長使用）
- サンプル数: [50, 100, 200, 500, 1000]
- num_input_tokens: 1
- モデル: 6層/768dim

新設計対応 (2025-11-29):
- MemoryDataProvider: サンプル境界情報を活用
- MemoryPhase1Trainer: CVFP固定点学習
- Phase2Trainer: キャッシュ方式による高速訓練
"""

import sys
import os
import json
import time
import random
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats

# 設定（11/27実験再現用）
SAMPLE_SIZES = [50, 100, 200, 500]  # 11/27実験と同じサンプル数
NUM_LAYERS = 6
CONTEXT_DIM = 768
EMBED_DIM = 768
NUM_INPUT_TOKENS = 1
EMBEDDING_FREEZE = False  # 11/27実験再現: Embedding凍結なし
RANDOM_SEED = 42


def print_flush(msg):
    print(msg, flush=True)


def set_seed(seed=42):
    """全ての乱数生成器のシードを固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_val_data_from_train(train_token_ids: torch.Tensor, tokenizer, val_file_path: str, val_ratio: float = 0.1):
    """
    訓練データから検証データを生成

    Args:
        train_token_ids: 訓練データのトークンID
        tokenizer: トークナイザー
        val_file_path: 検証データの保存先パス
        val_ratio: 検証データの割合（デフォルト10%）
    """
    # 訓練データの最後の一部を検証データとして使用
    val_size = int(len(train_token_ids) * val_ratio)
    val_token_ids = train_token_ids[-val_size:]

    # トークンをテキストに変換
    val_text = tokenizer.decode(val_token_ids.tolist())

    # ディレクトリ作成
    os.makedirs(os.path.dirname(val_file_path), exist_ok=True)

    # ファイルに保存
    with open(val_file_path, 'w', encoding='utf-8') as f:
        f.write(val_text)

    print_flush(f"  Created val data: {val_file_path} ({val_size} tokens)")
    return val_size


def run_experiment(
    num_samples: int,
    device: torch.device,
):
    """
    単一サンプル数での実験を実行

    Args:
        num_samples: 使用するサンプル数
        device: デバイス

    Returns:
        実験結果の辞書
    """
    from config import ResidualConfig
    from src.models.llm import LLM
    from src.providers.data import MemoryDataProvider
    from src.trainers.phase1 import MemoryPhase1Trainer
    from src.trainers.phase2 import Phase2Trainer
    from src.evaluation.metrics import analyze_fixed_points
    from transformers import AutoTokenizer

    print_flush(f"\n{'='*70}")
    print_flush(f"Experiment: {num_samples} samples")
    print_flush(f"{'='*70}")

    # シードを固定（各実験で同じ初期化）
    set_seed(RANDOM_SEED)

    # 設定を作成
    config = ResidualConfig()
    config.num_layers = NUM_LAYERS
    config.context_dim = CONTEXT_DIM
    config.embed_dim = EMBED_DIM
    config.num_input_tokens = NUM_INPUT_TOKENS
    config.phase2_freeze_embedding = EMBEDDING_FREEZE
    config.num_samples = num_samples

    # 検証データファイルパスをサンプル数に応じて設定
    val_file_path = f"./data/ultrachat_{num_samples}samples_val.txt"
    config.val_text_file = val_file_path

    # 検証データが存在しない場合は生成
    if not os.path.exists(val_file_path):
        print_flush(f"\n  Generating validation data...")
        # まず訓練データをロードするために一時的にMemoryDataProviderを使用
        # 但し、val_dataロード前に止める必要があるため、直接UltraChatからロード
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            cache_dir=os.path.join(config.cache_dir, "tokenizer")
        )
        tokenizer.pad_token = tokenizer.eos_token

        # 訓練データをキャッシュからロード（または生成）
        cache_file = os.path.join(
            config.cache_dir,
            f"ultrachat_{num_samples}samples_full.pt"
        )

        if os.path.exists(cache_file):
            cached = torch.load(cache_file)
            train_tokens_for_val = cached['token_ids']
        else:
            # キャッシュがない場合はデータをロードして生成
            from datasets import load_dataset
            dataset = load_dataset(
                config.dataset_name,
                split=config.dataset_split,
                cache_dir=os.path.join(config.cache_dir, "datasets")
            )

            all_token_ids = []
            sample_boundaries = []
            current_pos = 0

            for idx in range(num_samples):
                messages = dataset[idx]["messages"]
                text = "\n".join([msg["content"] for msg in messages])
                tokens = tokenizer(text, truncation=False, return_tensors="pt")
                sample_tokens = tokens["input_ids"].squeeze(0)
                all_token_ids.append(sample_tokens)
                sample_len = len(sample_tokens)
                sample_boundaries.append((current_pos, current_pos + sample_len))
                current_pos += sample_len

            train_tokens_for_val = torch.cat(all_token_ids)

            # キャッシュ保存
            os.makedirs(config.cache_dir, exist_ok=True)
            torch.save({
                'token_ids': train_tokens_for_val,
                'sample_order': list(range(num_samples)),
                'sample_boundaries': sample_boundaries
            }, cache_file)
            print_flush(f"  Cached to: {cache_file}")

        # 検証データを生成
        create_val_data_from_train(train_tokens_for_val, tokenizer, val_file_path)

    # データロード（サンプル数に応じたデータ）
    print_flush(f"\n  Loading {num_samples} samples...")
    data_provider = MemoryDataProvider(config, shuffle_samples=False)
    train_token_ids, val_token_ids = data_provider.load_data()

    train_token_ids = train_token_ids.to(device)
    val_token_ids = val_token_ids.to(device)

    print_flush(f"  Train tokens: {len(train_token_ids):,}")
    print_flush(f"  Val tokens: {len(val_token_ids):,}")

    # モデル作成
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        num_input_tokens=config.num_input_tokens,
        num_context_splits=getattr(config, 'num_context_splits', 1),
        use_pretrained_embeddings=config.use_pretrained_embeddings,
        use_weight_tying=config.use_weight_tying
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"  Model parameters: {total_params:,}")

    # Phase 1: CVFP固定点学習
    print_flush(f"\n  Phase 1 starting...")
    phase1_start = time.time()

    phase1_trainer = MemoryPhase1Trainer(model, config, device)
    train_contexts = phase1_trainer.train(
        train_token_ids,
        label=f"Train ({num_samples} samples)",
        data_provider=data_provider
    )

    # 訓練データのEffective Rank
    train_metrics = analyze_fixed_points(train_contexts, label="Train")

    # 検証データの評価（return_contexts_only=Trueでテンソルを取得）
    val_contexts = phase1_trainer.evaluate(
        val_token_ids,
        label=f"Val ({num_samples} samples)",
        return_contexts_only=True
    )
    val_metrics = analyze_fixed_points(val_contexts, label="Val")

    phase1_time = time.time() - phase1_start
    print_flush(f"\n  Phase 1 completed: {phase1_time/60:.1f}min")
    print_flush(f"  Train ER: {train_metrics['effective_rank_ratio']*100:.1f}%")
    print_flush(f"  Val ER: {val_metrics['effective_rank_ratio']*100:.1f}%")

    # Phase 2: Next-Token Prediction（キャッシュ方式）
    print_flush(f"\n  Phase 2 starting...")
    phase2_start = time.time()

    phase2_trainer = Phase2Trainer(model=model, config=config)

    # train_fullで訓練（早期停止あり）
    phase2_history = phase2_trainer.train_full(
        train_token_ids=train_token_ids,
        val_token_ids=val_token_ids,
        device=device
    )

    phase2_time = time.time() - phase2_start

    # ベストの結果を取得
    best_epoch = phase2_history['best_epoch']
    best_val_ppl = phase2_history['val_ppl'][best_epoch - 1]
    best_val_acc = phase2_history['val_acc'][best_epoch - 1]

    print_flush(f"\n  Phase 2 completed: {phase2_time/60:.1f}min")
    print_flush(f"  Best Val PPL: {best_val_ppl:.2f}")
    print_flush(f"  Best Val Acc: {best_val_acc*100:.2f}%")

    # クリーンアップ
    data_provider.close()
    del model, phase1_trainer, phase2_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'num_samples': num_samples,
        'train_tokens': len(train_token_ids),
        'val_tokens': len(val_token_ids),
        'train_effective_rank': train_metrics['effective_rank_ratio'],
        'val_effective_rank': val_metrics['effective_rank_ratio'],
        'val_ppl': best_val_ppl,
        'val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'early_stopped': phase2_history['early_stopped'],
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
    }


def calculate_scaling_law(results: list):
    """スケーリング則を計算: PPL = A × tokens^α"""
    tokens = np.array([r['train_tokens'] for r in results])
    ppl = np.array([r['val_ppl'] for r in results])

    # 対数変換
    log_tokens = np.log(tokens)
    log_ppl = np.log(ppl)

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_ppl)

    alpha = slope  # 負の値が期待される
    A = np.exp(intercept)
    r_squared = r_value ** 2

    return {
        'alpha': alpha,
        'A': A,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
    }


def main():
    print_flush("=" * 70)
    print_flush("UNIFIED SCALING EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_flush(f"\nSettings:")
    print_flush(f"  Sample sizes: {SAMPLE_SIZES}")
    print_flush(f"  Model: {NUM_LAYERS} layers, {CONTEXT_DIM} dim")
    print_flush(f"  num_input_tokens: {NUM_INPUT_TOKENS}")
    print_flush(f"  Embedding freeze: {EMBEDDING_FREEZE}")
    print_flush(f"  Tokenization: truncation=False (full length)")
    print_flush(f"  Random seed: {RANDOM_SEED}")

    # シード固定
    set_seed(RANDOM_SEED)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print_flush(f"\nGPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print_flush("\nUsing CPU")

    # 実験実行
    print_flush("\n" + "=" * 70)
    print_flush("Running experiments...")
    print_flush("=" * 70)

    results = []
    total_start = time.time()

    # 結果保存ディレクトリ
    output_dir = './results/unified_scaling'
    os.makedirs(output_dir, exist_ok=True)

    for i, num_samples in enumerate(SAMPLE_SIZES):
        print_flush(f"\n[{i+1}/{len(SAMPLE_SIZES)}] {num_samples} samples")
        result = run_experiment(num_samples, device)
        results.append(result)

        # 途中結果をJSON保存（クラッシュ対策）
        partial_file = os.path.join(output_dir, 'partial_results.json')
        with open(partial_file, 'w') as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - total_start

    # スケーリング則計算
    print_flush("\n" + "=" * 70)
    print_flush("SCALING LAW ANALYSIS")
    print_flush("=" * 70)

    scaling = calculate_scaling_law(results)

    print_flush(f"\nPPL = {scaling['A']:.2f} × tokens^({scaling['alpha']:.4f})")
    print_flush(f"α = {scaling['alpha']:.4f}")
    print_flush(f"R² = {scaling['r_squared']:.4f}")
    print_flush(f"p-value = {scaling['p_value']:.6f}")

    # 結果表示
    print_flush("\n" + "=" * 70)
    print_flush("RESULTS SUMMARY")
    print_flush("=" * 70)

    header = f"{'Samples':>8} | {'Tokens':>10} | {'Val PPL':>10} | {'Val Acc':>10}"
    header += f" | {'Train ER':>10} | {'Val ER':>10}"
    print_flush(f"\n{header}")
    print_flush("-" * 80)
    for r in results:
        print_flush(
            f"{r['num_samples']:>8} | "
            f"{r['train_tokens']:>10,} | "
            f"{r['val_ppl']:>10.2f} | "
            f"{r['val_acc']*100:>9.2f}% | "
            f"{r['train_effective_rank']*100:>9.1f}% | "
            f"{r['val_effective_rank']*100:>9.1f}%"
        )

    print_flush(f"\nTotal time: {total_time/60:.1f} minutes")

    # 結果保存
    output = {
        'settings': {
            'sample_sizes': SAMPLE_SIZES,
            'num_layers': NUM_LAYERS,
            'context_dim': CONTEXT_DIM,
            'embed_dim': EMBED_DIM,
            'num_input_tokens': NUM_INPUT_TOKENS,
            'embedding_freeze': EMBEDDING_FREEZE,
            'tokenization': 'truncation=False (full length)',
            'random_seed': RANDOM_SEED,
        },
        'results': results,
        'scaling_law': scaling,
        'total_time_minutes': total_time / 60,
        'timestamp': datetime.now().isoformat(),
    }

    output_file = os.path.join(output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print_flush(f"\nResults saved to: {output_file}")

    # スケーリング則の解釈
    print_flush("\n" + "=" * 70)
    print_flush("SCALING LAW INTERPRETATION")
    print_flush("=" * 70)

    # 2倍のデータでどれだけPPLが下がるか
    ppl_reduction = 1 - 2 ** scaling['alpha']
    print_flush(f"\nα = {scaling['alpha']:.4f}")
    print_flush(f"→ 2倍のデータで {ppl_reduction*100:.1f}% PPL削減")

    # 過去の実験との比較
    print_flush("\n" + "=" * 70)
    print_flush("COMPARISON WITH PREVIOUS EXPERIMENTS")
    print_flush("=" * 70)

    print_flush("\n| Experiment | α | Interpretation |")
    print_flush("|------------|------|----------------|")
    print_flush("| 11/27 (max_length=128) | -0.7463 | 2倍データで40%PPL削減 |")
    print_flush("| 11/28 v2 (truncation=False) | -0.2926 | 2倍データで22%PPL削減 |")
    print_flush(f"| This experiment | {scaling['alpha']:.4f} | 2倍データで{ppl_reduction*100:.1f}%PPL削減 |")

    if scaling['alpha'] < -0.5:
        print_flush("\n✅ 良好なスケーリング効率（α < -0.5）")
    elif scaling['alpha'] < -0.2:
        print_flush("\n⚠️ 標準的なスケーリング効率（-0.5 < α < -0.2）")
    else:
        print_flush("\n⚠️ 低いスケーリング効率（α > -0.2）")


if __name__ == '__main__':
    main()
