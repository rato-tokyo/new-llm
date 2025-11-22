"""
New-LLM 設定ファイル

プロジェクトのメイン設定ファイルです。
モデルアーキテクチャ、訓練設定、データパラメータはここで編集します。

使い方:
    python3 tests/phase2_experiments/test_residual.py --config Residual4Layer
    python3 tests/phase2_experiments/test_residual.py --config Residual2Layer
"""


# ============================================================================
# デフォルト設定 (Residual4Layer - 推奨)
# ============================================================================

class ResidualConfig:
    """
    Residual Standard アーキテクチャの基本設定クラス

    カスタム設定の作り方:
    1. ResidualConfigを継承した新しいクラスを作成
    2. 変更したいパラメータのみ上書き
    3. --config YourConfigName で使用
    """

    # ========== モデルアーキテクチャ ==========
    architecture = "residual_standard"
    layer_structure = [1, 1, 1, 1]  # 4層のFNNブロック（推奨）
    context_dim = 256               # 文脈ベクトル次元数
    embed_dim = 256                 # トークン埋め込み次元数
    hidden_dim = 512                # 中間層次元数（自動計算される場合あり）
    vocab_size = 50257              # GPT-2トークナイザーの語彙数

    # ========== CVFP設定（理解していない場合は変更しないこと） ==========
    context_update_strategy = "gated"  # CVFPには"gated"が必須
    use_layer_norm = True              # Layer Normalizationを使用（必須）
    use_context_clipping = False       # Context Clippingは不要（Layer Normで十分）

    # ========== DDR (Dimension Diversity Regularization) ==========
    # 次元別多様性正則化：低活性次元をブーストして次元崩壊を防ぐ
    use_ddr = True                     # DDRを使用（推奨：True）
    ddr_momentum = 0.9                 # 次元別活性のEMAモメンタム
    ddr_boost_weight = 0.1             # 低活性次元へのブースト重み

    # ⚠️ CRITICAL: ddr_threshold は「固定値」であり、「全次元の平均活性の比率」ではない
    # 各次元の平均活性（EMA）が threshold 未満の次元をブースト
    ddr_threshold = 0.2                # 閾値（固定値）
                                       # 0.2 = 各次元の平均活性が0.2未満の次元をブースト（推奨）
                                       # 0.3 = 各次元の平均活性が0.3未満の次元をブースト（やや広範囲）
                                       # 0.5 = 各次元の平均活性が0.5未満の次元をブースト（広範囲）
                                       # 注: 各次元のEMA（長期的な典型値）が threshold 未満かどうかで判定
                                       # 全次元の平均を計算することはない（これは頻繁に誤解されるポイント）

    # ========== Phase 1: 固有点学習 ==========
    phase1_max_iterations = 10           # 固有点探索の最大反復回数
    phase1_convergence_threshold = 0.02  # 収束判定のMSE閾値（0.02=緩い, 0.01=厳格）
                                         # 意味: 前回iterationとのMSE < 0.02なら収束と判定
                                         # √0.02 ≈ 0.141 = L2距離の閾値
    phase1_min_converged_ratio = 0.95    # 全トークンの95%が収束したら停止
    # Early Stopping: 収束率が2回連続で低下したら停止

    # LRスケジュール（DDR対応）
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


# ============================================================================
# 事前定義済み設定
# ============================================================================

class Residual2Layer(ResidualConfig):
    """2層アーキテクチャ（最小構成、テスト用に高速）"""
    layer_structure = [1, 1]


class Residual4Layer(ResidualConfig):
    """4層アーキテクチャ（本番環境推奨）"""
    layer_structure = [1, 1, 1, 1]


class Residual8Layer(ResidualConfig):
    """8層アーキテクチャ（高性能、訓練は遅い）"""
    layer_structure = [1, 1, 1, 1, 1, 1, 1, 1]


class Residual4Layer16Ctx(ResidualConfig):
    """4層・16次元文脈（超極小次元、次元崩壊対策）"""
    layer_structure = [1, 1, 1, 1]
    context_dim = 16
    embed_dim = 16
    hidden_dim = 32
    num_samples = 10  # 10サンプルで訓練


class Residual4Layer32Ctx(ResidualConfig):
    """4層・32次元文脈（極小次元、次元崩壊対策）"""
    layer_structure = [1, 1, 1, 1]
    context_dim = 32
    embed_dim = 32
    hidden_dim = 64
    num_samples = 50  # より多くのサンプルで訓練


class Residual4Layer64Ctx(ResidualConfig):
    """4層・64次元文脈（最小次元、次元崩壊対策）"""
    layer_structure = [1, 1, 1, 1]
    context_dim = 64
    embed_dim = 64
    hidden_dim = 128


class Residual4Layer128Ctx(ResidualConfig):
    """4層・128次元文脈（次元削減、次元崩壊対策）"""
    layer_structure = [1, 1, 1, 1]
    context_dim = 128
    embed_dim = 128
    hidden_dim = 256


class Residual4Layer512Ctx(ResidualConfig):
    """4層・512次元文脈（大きな文脈、より表現力が高い）"""
    layer_structure = [1, 1, 1, 1]
    context_dim = 512
    hidden_dim = 1024


# ============================================================================
# カスタム設定の例
# ============================================================================

# コメントを外して変更すれば、独自の設定を作成できます:
#
# class MyCustomConfig(ResidualConfig):
#     """独自の設定例"""
#     layer_structure = [2, 2]         # 2ブロック、各2層
#     context_dim = 384                # カスタム文脈次元数
#     num_samples = 100                # 100サンプルで訓練
#     phase1_lr_warmup = 0.002         # より高い初期LR
