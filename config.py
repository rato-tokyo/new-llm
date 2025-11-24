"""
New-LLM 設定ファイル

プロジェクトの設定値を定義します。
設定を変更する場合は、このファイルを直接編集してください。

使い方:
    python3 tests/phase2_experiments/test_residual.py
"""


class ResidualConfig:
    """
    Residual Standard アーキテクチャの設定

    設定を変更する場合は、このファイルを直接編集してください。
    """

    # ========== モデルアーキテクチャ ==========
    architecture = "residual_standard"
    num_layers = 6                  # 単層ブロックの数（最小対話モデル、固定）
    context_dim = 768               # 文脈ベクトル次元数（GPT-2に合わせて768次元）
    embed_dim = 768                 # トークン埋め込み次元数（GPT-2事前学習済み: 768次元）
    hidden_dim = 1536               # 中間層次元数（embed_dim * 2）
    vocab_size = 50257              # GPT-2トークナイザーの語彙数
    use_pretrained_embeddings = True  # GPT-2事前学習済み埋め込みを使用

    # ========== Diversity Regularization (固定次元割り当て法) ==========
    # 固定次元割り当て + LayerNormによる多様性確保
    dist_reg_weight = 0.99             # 多様性正則化の重み（高く設定して多様性を優先）
                                       # total_loss = (1-w) * cvfp_loss + w * diversity_loss
                                       # 0.5: 50% CVFP, 50% Diversity
                                       # 0.8: 20% CVFP, 80% Diversity
                                       # 0.99: 1% CVFP, 99% Diversity（現在の設定）

    # ========== Phase 1: 固有点学習 ==========
    phase1_max_iterations = 10           # 固有点探索の最大反復回数
    phase1_convergence_threshold = 0.05  # 収束判定のMSE閾値（0.05=かなり緩い, 0.02=緩い, 0.01=厳格）
                                         # 意味: 前回iterationとのMSE < 0.05なら収束と判定
                                         # √0.05 ≈ 0.224 = L2距離の閾値
    phase1_min_converged_ratio = 0.95    # 全トークンの95%が収束したら停止

    # 学習率（固定）
    phase1_learning_rate = 0.002         # Phase 1の学習率（固定値）
                                         # 0.002: 推奨（高速収束）
                                         # 0.001: 安定的
                                         # 0.0005: 慎重

    # ========== Phase 2: トークン予測 ==========
    skip_phase2 = True              # Phase 2をスキップ（Phase 1のみ実行）
    freeze_context = True           # Phase 2で文脈を固定（block_outputsのみ学習）
    phase2_learning_rate = 0.0001   # トークン予測の学習率
    phase2_epochs = 10              # 訓練エポック数
    phase2_batch_size = 32          # バッチサイズ
    phase2_gradient_clip = 1.0      # 勾配クリッピング値

    # ========== データ ==========
    # Training data source
    train_data_source = "ultrachat"                        # "ultrachat", "text_file", or "text_dir"
    train_text_file = "./data/example_train.txt"           # text_file使用時のパス
    train_text_dir = "./data/train/"                       # text_dir使用時のディレクトリ

    # Validation data source
    val_data_source = "manual"                             # "manual", "text_file", "text_dir", or "auto_split"
    val_text_file = "./data/example_val.txt"               # text_file使用時のパス
    val_text_dir = "./data/val/"                           # text_dir使用時のディレクトリ
    manual_val_path = "./cache/manual_val_tokens.pt"       # manual使用時のパス

    # Common settings
    max_seq_length = 128                                   # 最大シーケンス長（短縮: 高速テスト用）
    num_samples = 5                                        # 訓練サンプル数（少量: 高速テスト用）
    train_val_split = 0.8                                  # Train/Val分割比率（auto_split使用時）

    # UltraChat specific (train_data_source="ultrachat"の場合)
    dataset_name = "HuggingFaceH4/ultrachat_200k"          # HuggingFaceデータセット
    dataset_split = "train_sft"                            # 使用するデータセット分割
    cache_dir = "./cache"                                  # キャッシュディレクトリ

    # ========== デバイス ==========
    device = "cpu"        # "cpu" または "cuda"
    random_seed = 42      # 再現性のためのランダムシード

    # ========== 診断設定 ==========
    identity_mapping_threshold = 0.95  # 恒等写像検出の閾値（コサイン類似度）
                                       # 0.95: 推奨（95%以上の類似度で警告）
                                       # 0.90: 緩い基準
                                       # 0.98: 厳しい基準
    identity_check_samples = 100       # 恒等写像チェックのサンプル数

    # ========== チェックポイント ==========
    checkpoint_dir = "./checkpoints"                    # チェックポイント保存ディレクトリ
    checkpoint_path = "./checkpoints/model_latest.pt"   # 最新モデルのパス
    load_checkpoint = True                              # 訓練開始時にチェックポイントを読み込む
    save_checkpoint = True                              # 訓練終了時にチェックポイントを保存

    # ========== ログ出力 ==========
    log_every_steps = 1
    save_every_samples = 10
