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

    # ========== Validation Early Stopping ==========
    val_early_stopping = True        # Validation早期停止を有効化
    val_frequency = 5                # N イテレーションごとに評価
    val_sample_size = 10000          # サンプル数（10000以上推奨）
    val_patience = 1                 # N回連続で改善なし→停止
