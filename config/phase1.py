"""
Phase 1 設定 (Phase 1 Configuration)

多様性学習（OACDアルゴリズム）の設定。
"""


class Phase1Config:
    """Phase 1: 多様性学習（OACD）の設定"""

    # ========== 学習パラメータ ==========
    max_iterations = 100             # 最大反復回数（増加: 60→100）
    convergence_threshold = 0.03     # 収束判定のMSE閾値（ログ表示用）
    learning_rate = 0.003            # 学習率（増加: 0.002→0.003）
    batch_size = 5000                # 並列処理のバッチサイズ（L4 GPU 24GB対応）
    gradient_clip = 2.0              # 勾配クリッピング値

    # ========== コンテキストノイズ（汎化性能向上） ==========
    context_noise = 0.05             # ガウシアンノイズの標準偏差（減少: 0.1→0.05）
                                     # 0.0: ノイズなし
                                     # 0.05: 軽いノイズ（収束優先）
                                     # 0.1: 推奨（汎化優先）

    # ========== 収束率Early Stopping ==========
    early_stopping = True            # 収束率による早期停止を有効化
    early_stopping_threshold = 0.9   # 収束率がこの値以上で停止

    # ========== 収束率改善Early Stopping ==========
    min_convergence_improvement = 0.01  # 収束率改善がこの値未満なら早期停止（1%）
                                        # 例: 30% → 30.5% (0.5%改善) は停止
                                        # 例: 30% → 32% (2%改善) は継続
