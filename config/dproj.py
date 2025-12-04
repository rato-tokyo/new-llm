"""
DProj Training Configuration

Diverse Projection (DProj) の OACD 学習設定。
"""


class DProjTrainingConfig:
    """DProj Training: OACD (Origin-Anchored Centroid Dispersion) の設定"""

    # ========== 学習パラメータ ==========
    max_iterations = 100             # 最大反復回数
    convergence_threshold = 0.03     # 収束判定のMSE閾値（ログ表示用）
    learning_rate = 0.003            # 学習率
    batch_size = 5000                # 並列処理のバッチサイズ（L4 GPU 24GB対応）
    gradient_clip = 2.0              # 勾配クリッピング値

    # ========== ノイズ（汎化性能向上） ==========
    proj_noise = 0.05                # ガウシアンノイズの標準偏差
                                     # 0.0: ノイズなし
                                     # 0.05: 軽いノイズ（収束優先）
                                     # 0.1: 推奨（汎化優先）

    # ========== OACD損失 ==========
    centroid_weight = 0.1            # 重心を原点に引き寄せる損失の重み

    # ========== 収束率Early Stopping ==========
    early_stopping = True            # 収束率による早期停止を有効化
    early_stopping_threshold = 0.95  # 収束率がこの値以上で停止

    # ========== 収束率改善Early Stopping ==========
    min_convergence_improvement = 0.01  # 収束率改善がこの値未満なら早期停止（1%）


# Backward compatibility alias
Phase1Config = DProjTrainingConfig
