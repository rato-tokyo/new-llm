"""
Phase 1 設定 (Phase 1 Configuration)

多様性学習（OACDアルゴリズム）の設定。
"""


class Phase1Config:
    """Phase 1: 多様性学習（OACD）の設定"""

    # ========== 学習パラメータ ==========
    max_iterations = 60              # 最大反復回数
    convergence_threshold = 0.03     # 収束判定のMSE閾値（ログ表示用）
    learning_rate = 0.002            # 学習率（0.001-0.004）
    batch_size = 5000                # 並列処理のバッチサイズ（L4 GPU 24GB対応）
    gradient_clip = 2.0              # 勾配クリッピング値

    # ========== コンテキストノイズ（汎化性能向上） ==========
    context_noise = 0.1              # ガウシアンノイズの標準偏差
                                     # 0.0: ノイズなし
                                     # 0.1: 推奨（軽いノイズ）
                                     # 0.2: 強めのノイズ

    # ========== 収束率Early Stopping ==========
    early_stopping = True            # 収束率による早期停止を有効化
    early_stopping_threshold = 0.99  # 収束率がこの値以上で停止
