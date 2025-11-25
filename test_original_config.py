"""
89.4%達成時の設定で検証
- LayerNorm無効 (layernorm_mix=0.0)
- 6400トークン (UltraChat 50サンプル)
"""

import torch
from src.models.llm import LLM
from src.training.phase1_trainer import phase1_train
from src.data.loader import load_data
from src.evaluation.metrics import analyze_fixed_points
import config as cfg


def test_original_config():
    """89.4%達成時の設定でテスト"""
    print("=" * 80)
    print("89.4% ER達成時の設定で検証")
    print("=" * 80)

    device = torch.device("cpu")

    # データ読み込み（89.4%達成時と同じ）
    print("\n[1] データ読み込み...")
    train_token_ids, val_token_ids = load_data(cfg.ResidualConfig)
    print(f"✓ 訓練データ: {len(train_token_ids)} トークン")
    print(f"✓ 検証データ: {len(val_token_ids)} トークン")

    # モデル初期化（89.4%達成時と同じ）
    print("\n[2] モデル初期化...")
    torch.manual_seed(42)
    model = LLM(
        vocab_size=cfg.ResidualConfig.vocab_size,
        embed_dim=cfg.ResidualConfig.embed_dim,
        context_dim=cfg.ResidualConfig.context_dim,
        hidden_dim=cfg.ResidualConfig.hidden_dim,
        layer_structure=[1] * cfg.ResidualConfig.num_layers,
        layernorm_mix=0.0,  # LayerNorm無効（89.4%達成時と同じ）
        use_pretrained_embeddings=cfg.ResidualConfig.use_pretrained_embeddings
    ).to(device)

    print(f"✓ モデル初期化完了")
    print(f"  - LayerNorm: 無効 (layernorm_mix=0.0)")
    print(f"  - Layers: {cfg.ResidualConfig.num_layers}")
    print(f"  - Context dim: {cfg.ResidualConfig.context_dim}")

    # Phase 1訓練
    print("\n[3] Phase 1訓練...")
    contexts_train = phase1_train(
        model=model,
        token_ids=train_token_ids,
        device=device,
        max_iterations=cfg.ResidualConfig.phase1_max_iterations,
        learning_rate=cfg.ResidualConfig.phase1_learning_rate,
        dist_reg_weight=cfg.ResidualConfig.dist_reg_weight,
        label="Train"
    )

    # 訓練データ評価
    print("\n[4] 訓練データ評価...")
    metrics_train = analyze_fixed_points(contexts_train, label="Train", verbose=True)
    er_train_pct = (metrics_train["effective_rank"] / cfg.ResidualConfig.context_dim) * 100

    print(f"\n📊 訓練結果:")
    print(f"  Effective Rank: {metrics_train['effective_rank']:.2f}/{cfg.ResidualConfig.context_dim} ({er_train_pct:.1f}%)")
    print(f"  Actual Rank: {metrics_train['actual_rank']}/{cfg.ResidualConfig.context_dim}")

    # 判定
    print("\n" + "=" * 80)
    if er_train_pct >= 85.0:
        print(f"✅ 成功: {er_train_pct:.1f}% (目標: ≥85%)")
        print("   89.4%達成時の設定が正常に動作しています")
    else:
        print(f"❌ 失敗: {er_train_pct:.1f}% (目標: ≥85%)")
        print("   何かが変更されて性能が低下しています")
    print("=" * 80)


if __name__ == "__main__":
    test_original_config()
