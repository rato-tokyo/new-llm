"""
Phase 1 Configuration

ContextBlock OACD学習の設定。
Phase 1の必須機能（削除禁止）の設定値を管理。
"""


class Phase1Config:
    """Phase 1 (OACD) 学習設定

    以下の機能は試行錯誤の末に必須と判明したもの。
    絶対に削除しないこと。
    """

    # ========== イテレーション設定 ==========
    max_iterations = 60             # 最大イテレーション数

    # ========== 学習率 ==========
    learning_rate = 0.002           # Phase 1 学習率

    # ========== 収束判定 ==========
    convergence_threshold = 0.03    # 収束判定の閾値（context変化量MSE）
    early_stopping_rate = 0.90      # 収束率がこの値以上でEarly Stop

    # ========== バッチ処理 ==========
    batch_size = 5000               # 並列処理のバッチサイズ
    batches_per_iteration = 10      # イテレーションあたりのバッチ数（勾配累積）

    # ========== コンテキストノイズ ==========
    context_noise = 0.1             # ガウシアンノイズの標準偏差（汎化性能向上）

    # ========== 勾配クリッピング ==========
    gradient_clip = 2.0             # 勾配クリッピング値

    # ========== Validation ==========
    val_split = 0.1                 # 検証データの割合

    # ========== チェックポイント ==========
    checkpoint_path = "checkpoints/context_block_phase1.pt"

    # ========== 内部設定 ==========
    internal_seq_length = 64        # モデルforward用の内部シーケンス長
