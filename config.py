"""
New-LLM 設定ファイル

プロジェクトのデフォルト設定値を定義します。
実際の使用時はコマンドライン引数で上書きできます。

使い方:
    python3 tests/phase2_experiments/test_residual.py \
        --context-dim 16 \
        --num-samples 10 \
        --dist-reg-weight 0.2

    # 層数を変更（単層ブロックのみ）
    python3 tests/phase2_experiments/test_residual.py --num-layers 3  # [1,1,1]
"""


class ResidualConfig:
    """
    Residual Standard アーキテクチャのデフォルト設定

    これらの値はコマンドライン引数で上書き可能です。
    """

    # ========== モデルアーキテクチャ ==========
    architecture = "residual_standard"
    num_layers = 4                  # 単層ブロックの数（--num-layers で変更可能）
    context_dim = 256               # 文脈ベクトル次元数
    embed_dim = 256                 # トークン埋め込み次元数
    hidden_dim = 512                # 中間層次元数
    vocab_size = 50257              # GPT-2トークナイザーの語彙数

    # ========== Distribution Regularization ==========
    # 分布正則化：各次元の出力が正規分布N(0,1)に近づくよう制約
    use_distribution_reg = True        # 分布正則化を使用（推奨：True）
    dist_reg_weight = 0.2              # 分布正則化の重み（0.2 = 20%、0.5 = 50%）
                                       # total_loss = (1-w) * cvfp_loss + w * dist_loss
                                       # 0.2: 80% CVFP, 20% 分布正則化（推奨）
                                       # 0.5: 50% CVFP, 50% 分布正則化（より強い制約）

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
    phase2_learning_rate = 0.0001   # トークン予測の学習率
    phase2_epochs = 10              # 訓練エポック数
    phase2_batch_size = 32          # バッチサイズ
    phase2_gradient_clip = 1.0      # 勾配クリッピング値

    # ========== データ ==========
    max_seq_length = 1024                          # 最大シーケンス長
    dataset_name = "HuggingFaceH4/ultrachat_200k"  # HuggingFaceデータセット
    dataset_split = "train_sft"                    # 使用するデータセット分割
    cache_dir = "./cache"                          # キャッシュディレクトリ
    num_samples = 10                               # 訓練サンプル数
    train_val_split = 0.8                          # Train/Val分割比率（80/20）

    # ========== デバイス ==========
    device = "cpu"        # "cpu" または "cuda"
    random_seed = 42      # 再現性のためのランダムシード

    # ========== ログ出力 ==========
    log_every_steps = 1
    save_every_samples = 10
