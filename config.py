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
    num_layers = 2                  # 単層ブロックの数
    context_dim = 16                # 文脈ベクトル次元数
    embed_dim = 16                  # トークン埋め込み次元数
    hidden_dim = 32                 # 中間層次元数
    vocab_size = 50257              # GPT-2トークナイザーの語彙数

    # ========== Distribution Regularization ==========
    # 分布正則化：各次元の出力が正規分布N(0,1)に近づくよう制約
    use_distribution_reg = True        # 分布正則化を使用（推奨：True）
    dist_reg_weight = 0.2              # 分布正則化の重み
                                       # total_loss = (1-w) * cvfp_loss + w * dist_loss
                                       # 0.2: 80% CVFP, 20% Dist（推奨）
                                       # 0.5: 50% CVFP, 50% Dist
                                       # 0.7: 30% CVFP, 70% Dist

    # ========== Phase 1: 固有点学習 ==========
    phase1_max_iterations = 10           # 固有点探索の最大反復回数
    phase1_convergence_threshold = 0.02  # 収束判定のMSE閾値（0.02=緩い, 0.01=厳格）
                                         # 意味: 前回iterationとのMSE < 0.02なら収束と判定
                                         # √0.02 ≈ 0.141 = L2距離の閾値
    phase1_min_converged_ratio = 0.95    # 全トークンの95%が収束したら停止
    # Early Stopping: 収束率が2回連続で低下したら停止

    # LRスケジュール
    phase1_lr_warmup = 0.002      # 反復1-3回目: 高めのLR（高速収束）
    phase1_lr_medium = 0.0005     # 反復4-8回目: 中程度のLR
    phase1_lr_finetune = 0.0001   # 反復9回目以降: 低いLR（微調整）

    # ========== Phase 2: トークン予測 ==========
    skip_phase2 = True              # Phase 2をスキップ（Phase 1のみ実行）
    freeze_context = False          # Phase 2で文脈を固定（token_outputのみ学習）
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
    max_seq_length = 1024                                  # 最大シーケンス長
    num_samples = 100                                      # 訓練サンプル数（ultrachat使用時）
    train_val_split = 0.8                                  # Train/Val分割比率（auto_split使用時）

    # UltraChat specific (train_data_source="ultrachat"の場合)
    dataset_name = "HuggingFaceH4/ultrachat_200k"          # HuggingFaceデータセット
    dataset_split = "train_sft"                            # 使用するデータセット分割
    cache_dir = "./cache"                                  # キャッシュディレクトリ

    # ========== デバイス ==========
    device = "cpu"        # "cpu" または "cuda"
    random_seed = 42      # 再現性のためのランダムシード

    # ========== ログ出力 ==========
    log_every_steps = 1
    save_every_samples = 10
