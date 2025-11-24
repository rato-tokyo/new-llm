"""
New-LLM 設定ファイル

プロジェクトの設定値を定義します。
設定を変更する場合は、このファイルを直接編集してください。

使い方:
    python3 test.py
"""

import torch


class ResidualConfig:
    """
    CVFP (Context Vector Fixed-Point) アーキテクチャの設定

    設定を変更する場合は、このファイルを直接編集してください。
    """

    # ========== モデルアーキテクチャ ==========
    architecture = "residual_standard"
    num_layers = 6                  # CVFPブロック数（6層固定）
    context_dim = 768               # コンテキストベクトル次元数（GPT-2に合わせて768次元）
    embed_dim = 768                 # トークン埋め込み次元数（GPT-2事前学習済み: 768次元）
    hidden_dim = 1536               # 中間層次元数（embed_dim * 2）
    vocab_size = 50257              # GPT-2トークナイザーの語彙数
    use_pretrained_embeddings = True  # GPT-2事前学習済み埋め込みを使用

    # ========== Diversity Regularization (Per-Dimension Usage Tracking) ==========
    # LayerNorm + Per-Dimension Variance Tracking (EMA-based) による多様性確保
    # 実装: 各次元の使用頻度を追跡し、使用頻度が低い次元を優先的に活性化
    dist_reg_weight = 0.5              # 多様性正則化の重み
                                       # total_loss = (1-w) * cvfp_loss + w * diversity_loss
                                       # 0.5: 50% CVFP, 50% Diversity（実験用設定）
                                       # 0.8: 20% CVFP, 80% Diversity
                                       # 0.99: 1% CVFP, 99% Diversity（標準設定）

    # ========== Phase 1: CVFP学習（固定点学習） ==========
    phase1_max_iterations = 10           # 固定点探索の最大反復回数
    phase1_convergence_threshold = 2.0   # 収束判定のMSE閾値
                                         # 意味: 前回iterationとのMSE < 2.0なら収束と判定
                                         # 実測値: 初期MSE≈1.43、収束後MSE≈0.5-1.0
                                         # 2.0: 適切な閾値（実測値に基づく）
                                         # 1.0: やや厳格
                                         # 0.5: 厳格
    phase1_min_converged_ratio = 0.95    # 早期停止: 全トークンの95%が収束したら停止

    # 学習率
    phase1_learning_rate = 0.002         # Phase 1の学習率
                                         # 0.002: 推奨（高速収束）
                                         # 0.001: 安定的
                                         # 0.0005: 慎重

    # ========== Phase 2: トークン予測 ==========
    skip_phase1 = True              # Phase 1をスキップ（チェックポイントから続行する場合）
    skip_phase2 = False             # Phase 2を実行（実装完了）
    freeze_context = False          # Phase 2で全層を微調整（予測精度向上のため）
    phase2_learning_rate = 0.002    # トークン予測の学習率 (Phase 1と同じ)
    phase2_epochs = 10              # 訓練エポック数
    # NOTE: No batch_size - all tokens processed at once (each token is independent)
    phase2_gradient_clip = 1.0      # 勾配クリッピング値

    # ========== データ ==========
    # Training data source
    train_data_source = "ultrachat"                        # "ultrachat", "text_file", or "text_dir"
    train_text_file = "./data/example_train.txt"           # text_file使用時のパス
    train_text_dir = "./data/train/"                       # text_dir使用時のディレクトリ

    # Validation data source
    # ⚠️ CRITICAL: Must use "text_file" only! auto_split is FORBIDDEN!
    # Validation data must contain ONLY tokens from training data
    # Generate using: python3 scripts/create_val_from_train.py
    val_data_source = "text_file"                          # MUST be "text_file" (auto_split is FORBIDDEN)
    val_text_file = "./data/example_val.txt"               # text_file使用時のパス
    val_text_dir = "./data/val/"                           # text_dir使用時のディレクトリ
    manual_val_path = "./cache/manual_val_tokens.pt"       # manual使用時のパス

    # Common settings
    max_seq_length = 128                                   # 最大シーケンス長（高速テスト用に短縮）
    num_samples = 50                                       # 訓練サンプル数（→ 6400トークン生成）
    train_val_split = 0.8                                  # Train/Val分割比率（auto_split使用時のみ）

    # UltraChat specific (train_data_source="ultrachat"の場合)
    dataset_name = "HuggingFaceH4/ultrachat_200k"          # HuggingFaceデータセット
    dataset_split = "train_sft"                            # 使用するデータセット分割
    cache_dir = "./cache"                                  # キャッシュディレクトリ

    # ========== デバイス ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自動GPU検出
    random_seed = 42      # 再現性のためのランダムシード（完全な再現性保証）

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
