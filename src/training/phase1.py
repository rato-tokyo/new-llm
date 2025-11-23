"""
Phase 1 訓練 (バージョン9): Trainerパターン採用

主な改善点:
1. Phase1Trainerクラスで訓練ロジックをカプセル化
2. モデルから訓練状態を分離（責任分離の改善）
3. train() / evaluate() メソッドで訓練/評価を明示的に分離
4. PyTorchの標準的なTrainerパターンに準拠
"""

from .phase1_trainer import Phase1Trainer


def train_phase1(
    model,
    token_ids,
    device,
    max_iterations,
    convergence_threshold,
    min_converged_ratio,
    learning_rate,
    dist_reg_weight,
    label,
    is_training=True
):
    """
    Phase 1: CVFP固定点学習

    Args:
        model: 言語モデル (enable_cvfp_learning=True必須)
        token_ids: 入力トークンID [num_tokens]
        device: torch デバイス
        max_iterations: 最大イテレーション数
        convergence_threshold: 収束判定のMSE閾値
        min_converged_ratio: 早期停止の収束率閾値
        learning_rate: 学習率
        dist_reg_weight: 分布正則化の重み
        label: ログ用ラベル (例: "Train", "Val")
        is_training: 訓練モードかどうか（True=訓練, False=評価）

    Returns:
        final_contexts: 収束したコンテキストベクトル [num_tokens, context_dim]
    """
    # Trainerを作成
    trainer = Phase1Trainer(
        model=model,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        min_converged_ratio=min_converged_ratio,
        learning_rate=learning_rate,
        dist_reg_weight=dist_reg_weight
    )

    # 訓練または評価を実行
    if is_training:
        return trainer.train(token_ids, device, label)
    else:
        return trainer.evaluate(token_ids, device, label)

