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
    num_layers = 3                  # CVFPブロック数（6層固定）
    context_dim = 768               # コンテキストベクトル次元数（GPT-2に合わせて768次元）
    embed_dim = 768                 # トークン埋め込み次元数（GPT-2事前学習済み: 768次元）
    hidden_dim = 1536               # 中間層次元数（embed_dim * 2）
    vocab_size = 50257              # GPT-2トークナイザーの語彙数
    use_pretrained_embeddings = True  # GPT-2事前学習済み埋め込みを使用

    # ========== Diversity Regularization (Per-Dimension Usage Tracking) ==========
    # LayerNorm + Per-Dimension Variance Tracking (EMA-based) による多様性確保
    # 実装: 各次元の使用頻度を追跡し、使用頻度が低い次元を優先的に活性化
    dist_reg_weight = 0.5              # 多様性正則化の重み (PARALLEL OPTIMIZED: 90% diversity)
                                       # total_loss = (1-w) * cvfp_loss + w * diversity_loss
                                       # 0.9: 10% CVFP, 90% Diversity（並列版最適設定: 55.9% ER達成）
                                       # 並列版の情報遅延を多様性強化で補償

    # ========== Phase 1: CVFP学習（固定点学習） ==========
    phase1_max_iterations = 20           # 固定点探索の最大反復回数
    phase1_convergence_threshold = 0.03   # 収束判定のMSE閾値
                                         # 意味: 前回iterationとのMSE < 0.1なら収束と判定
                                         # 実測値: 初期MSE≈1.43、学習後MSE≈0.001-0.1
                                         # 0.1: バランスの取れた閾値
                                         # 0.05: やや厳格（以前の設定）
                                         # 0.01: 非常に厳格
    phase1_min_converged_ratio = 1.01    # 早期停止を無効化（101%以上 = 不可能）

    # 学習率
    phase1_learning_rate = 0.002         # Phase 1の学習率
                                         # 0.002: 推奨（高速収束）
                                         # 0.001: 安定的
                                         # 0.0005: 慎重

    # ========== Phase 2: トークン予測（Context-Fixed Learning） ==========
    skip_phase1 = False             # Phase 1を実行（Colab実験用）
    skip_phase2 = False             # Phase 2を実行（実装完了）
    freeze_context = False          # Phase 2でCVFP層も微調整（token_out経由で勾配が流れる）
    phase2_learning_rate = 0.002    # トークン予測の学習率 (Phase 1と同じ)
    phase2_epochs = 10              # 訓練エポック数
    phase2_patience = 3             # Early stopping patience
    phase2_batch_size = 2048         # ミニバッチサイズ（GPUメモリ節約のため）
                                    # 512: 推奨（~4GB VRAM削減）
                                    # 256: メモリ不足時
                                    # 1024: 高速化重視
    phase2_gradient_clip = 1.0      # 勾配クリッピング値
    # Context-Fixed Learning: context_out = C*[i] に完全固定（MSE制約ではない）

    # ========== データ ==========
    # ⚠️ UltraChat専用設定 (高速化により大規模データセットに対応)
    # データ生成: cd ../new-llm-data && python3 generate_ultrachat_data.py --num_samples 50 --max_seq_length 128

    # Training data source
    train_data_source = "ultrachat"                        # UltraChat専用（キャッシュ使用で高速）

    # Validation data source
    # ⚠️ CRITICAL: Must use "text_file" only! auto_split is FORBIDDEN!
    # Validation data must contain ONLY tokens from training data
    # 自動生成: データ生成スクリプトが自動的に作成（訓練データの最後20%）
    val_data_source = "text_file"                          # MUST be "text_file" (auto_split is FORBIDDEN)
    val_text_file = "./data/ultrachat_50samples_val.txt"   # 自動生成される検証データ

    # UltraChat settings
    max_seq_length = 128                                   # 最大シーケンス長
    num_samples = 500                                      # 訓練サンプル数（→ ~64000トークン）
    dataset_name = "HuggingFaceH4/ultrachat_200k"          # HuggingFaceデータセット
    dataset_split = "train_sft"                            # 使用するデータセット分割
    cache_dir = "./cache"                                  # キャッシュディレクトリ

    # 推奨データセットサイズ（3層モデル + ミニバッチ処理版）
    # - テスト: num_samples=100, max_seq_length=128 (~12800トークン)
    # - 中規模: num_samples=500, max_seq_length=128 (~64000トークン) ← 現在
    # - 大規模: num_samples=1000, max_seq_length=128 (~128000トークン) ※メモリ注意

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
    load_checkpoint = False                             # 新規モデルから開始（Colab実験用）
    save_checkpoint = True                              # 訓練終了時にチェックポイントを保存

    # ========== ログ出力 ==========
    log_every_steps = 1
    save_every_samples = 10
