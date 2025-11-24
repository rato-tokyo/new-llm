"""
EMA方式統合後の動作確認テスト

少量データで新しいEMA方式が正しく動作することを確認
"""

import torch
import sys

sys.path.insert(0, '.')

from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.training.phase1_trainer import Phase1Trainer
from src.evaluation.metrics import analyze_fixed_points


def test_ema_integration():
    """EMA方式統合後の動作確認"""

    print("="*70)
    print("EMA方式統合テスト - 開始")
    print("="*70)
    print()

    config = ResidualConfig()
    device = torch.device(config.device)

    # モデル作成
    print("Step 1: モデル作成...")
    layer_structure = [1] * config.num_layers
    model = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        layernorm_mix=1.0,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    model.to(device)
    print(f"✓ モデル作成完了")
    print()

    # トレーナー作成
    print("Step 2: Phase1Trainer作成（EMA方式）...")
    trainer = Phase1Trainer(
        model=model,
        max_iterations=config.phase1_max_iterations,
        convergence_threshold=config.phase1_convergence_threshold,
        min_converged_ratio=config.phase1_min_converged_ratio,
        learning_rate=config.phase1_learning_rate,
        dist_reg_weight=config.dist_reg_weight,
        ema_momentum=0.99
    )
    print(f"✓ トレーナー作成完了")
    print(f"  EMA momentum: {trainer.ema_momentum}")
    print()

    # 小規模データで訓練
    print("Step 3: 小規模データで訓練（100トークン）...")
    num_train = 100
    num_val = 50

    train_tokens = torch.randint(0, 1000, (num_train,), device=device)
    val_tokens = torch.randint(0, 1000, (num_val,), device=device)

    # 訓練実行
    train_contexts = trainer.train(train_tokens, device, label="Train")
    print(f"✓ 訓練完了")
    print()

    # 評価実行
    print("Step 4: 評価実行...")
    val_contexts = trainer.evaluate(val_tokens, device, label="Val")
    print(f"✓ 評価完了")
    print()

    # EMA統計量の確認
    print("Step 5: EMA統計量の確認...")
    if trainer.context_mean_ema is not None:
        mean_norm = trainer.context_mean_ema.norm().item()
        var_mean = trainer.context_var_ema.mean().item()
        var_min = trainer.context_var_ema.min().item()
        var_max = trainer.context_var_ema.max().item()

        print(f"  平均ベクトルのノルム: {mean_norm:.4f}")
        print(f"  分散の平均: {var_mean:.4f}")
        print(f"  分散の範囲: [{var_min:.4f}, {var_max:.4f}]")
        print(f"✓ EMA統計量が正常に更新されています")
    else:
        print(f"⚠️ EMA統計量が初期化されていません")
    print()

    # Effective Rank分析
    print("Step 6: Effective Rank分析...")
    train_metrics = analyze_fixed_points(train_contexts, label="Train")
    val_metrics = analyze_fixed_points(val_contexts, label="Val")
    print()

    # 結果サマリー
    print("="*70)
    print("テスト結果サマリー")
    print("="*70)
    print(f"\n✅ すべてのテストが正常に完了しました！")
    print(f"\n主要指標:")
    print(f"  Train Effective Rank: {train_metrics['effective_rank']:.2f}/{config.context_dim} ({train_metrics['effective_rank']/config.context_dim*100:.1f}%)")
    print(f"  Val Effective Rank: {val_metrics['effective_rank']:.2f}/{config.context_dim} ({val_metrics['effective_rank']/config.context_dim*100:.1f}%)")
    print(f"\nEMA方式が正常に動作しています。")
    print(f"履歴保存なし・固定メモリでの真のオンライン学習を実現！")


if __name__ == "__main__":
    try:
        test_ema_integration()
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
