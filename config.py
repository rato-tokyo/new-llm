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
    num_layers = 6                  # ContextBlock と TokenBlock の両方のレイヤー数
    embed_dim = 768                 # トークン埋め込み次元数（GPT-2事前学習済み: 768次元、固定）
    context_multiplier = 1          # context_dim = embed_dim × この値（1, 2, 3, ...）
    context_dim = embed_dim * context_multiplier  # コンテキストベクトル次元数（自動計算）
    vocab_size = 50257              # GPT-2トークナイザーの語彙数
    use_pretrained_embeddings = True  # GPT-2事前学習済み埋め込みを使用
    use_weight_tying = True         # Weight Tying: Token EmbeddingとOutput Headで重み共有（標準採用）
                                    # パラメータ約38M削減（91M → 53M）
    # LayerNorm: 常に有効（数値安定性のため必須）

    # ========== 複数トークン入力 ==========
    num_input_tokens = 1            # 入力するトークン数
                                    # 1: 現在のトークンのみ（現状維持）
                                    # 2: 直前2トークン [token_{t-1}, token_t]
                                    # N: 直前Nトークン [token_{t-N+1}, ..., token_t]

    # ========== ContextBlock構造 ==========
    # token継ぎ足し方式（全レイヤーでtoken入力）に一本化
    # PPL 334 vs 536（38%改善）、Acc 18.9% vs 15.4%（23%向上）
    # 等差減少設計は廃止（2025-11-29）

    # ========== ContextBlock分割 ==========
    num_context_splits = 1          # ContextBlockの分割数
                                    # 1: 分割なし（従来動作）
                                    # N: N分割（各ブロックがcontext_dim/Nを出力、連結して結合）
                                    # 効果: 計算量・パラメータが約1/Nに削減
                                    # 訓練: 各ブロックは異なるサンプルで訓練（sample_id % N）

    @property
    def split_context_dim(self):
        """分割後の各ContextBlockのcontext_dim"""
        if self.context_dim % self.num_context_splits != 0:
            raise ValueError(
                f"context_dim ({self.context_dim}) must be divisible by "
                f"num_context_splits ({self.num_context_splits})"
            )
        return self.context_dim // self.num_context_splits

    # ========== Diversity Regularization (Per-Dimension Usage Tracking) ==========
    # LayerNorm + Per-Dimension Variance Tracking (EMA-based) による多様性確保
    # 実装: 各次元の使用頻度を追跡し、使用頻度が低い次元を優先的に活性化
    dist_reg_weight = 0.8              # 多様性正則化の重み (PARALLEL OPTIMIZED: 90% diversity)
                                       # total_loss = (1-w) * cvfp_loss + w * diversity_loss
                                       # 0.9: 10% CVFP, 90% Diversity（並列版最適設定: 55.9% ER達成）
                                       # 並列版の情報遅延を多様性強化で補償

    # ========== Phase 1: CVFP学習（固定点学習） ==========
    phase1_min_iterations = 5            # 固定点探索の最小反復回数（早期停止の最低保証）
    phase1_max_iterations = 40           # 固定点探索の最大反復回数
    phase1_convergence_threshold = 0.03   # 収束判定のMSE閾値
                                         # 意味: 前回iterationとのMSE < 0.1なら収束と判定
                                         # 実測値: 初期MSE≈1.43、学習後MSE≈0.001-0.1
                                         # 0.1: バランスの取れた閾値
                                         # 0.05: やや厳格（以前の設定）
                                         # 0.01: 非常に厳格
    phase1_min_converged_ratio = 0.99    # 早期停止を無効化（101%以上 = 不可能）

    # 検証データ収束判定
    val_convergence_trials = 4              # 検証データの収束判定イテレーション回数
                                             # 複数回順伝播してCVFP損失の推移を確認

    # コンテキストノイズ（汎化性能向上）
    phase1_context_noise = 0.1           # コンテキストに追加するガウシアンノイズの標準偏差
                                         # 0.0: ノイズなし
                                         # 0.1: 推奨（軽いノイズ）
                                         # 0.2: 強めのノイズ

    # 学習率
    phase1_learning_rate = 0.002         # Phase 1の学習率（0.001-0.004）
    phase1_batch_size = 8000            # 並列処理のバッチサイズ（L4 GPU 24GB対応）
    phase1_gradient_clip = 1.0           # 勾配クリッピング値

    # ========== Phase 2: トークン予測（キャッシュ方式） ==========
    # キャッシュ方式による高速化:
    # - ContextBlock出力を事前計算してキャッシュ（エポックごとに1回のみ）
    # - TokenBlockをバッチ並列処理（真のバッチ処理）
    # - 従来比 5〜20倍の高速化
    #
    # 分離アーキテクチャ: ContextBlock(frozen+cached) + TokenBlock(学習)
    # - ContextBlockがfreezeされているため、context_out = C*が自動的に保証
    # - C*の事前計算不要、context_stability_loss不要
    skip_phase1 = False             # Phase 1を実行（Colab実験用）
    skip_phase2 = False             # Phase 2を実行（実装完了）
    phase2_learning_rate = 0.001    # トークン予測の学習率
    phase2_epochs = 10              # 訓練エポック数
    phase2_patience = 1             # Early stopping patience
    phase2_batch_size = None        # ミニバッチサイズ（Noneで自動計算）
                                    # None: GPUメモリに基づいて自動設定
                                    #       キャッシュ構築後の空きメモリから最適値を計算
                                    # 手動指定も可能（ただし自動調整される場合あり）
    phase2_gradient_clip = 1.0      # 勾配クリッピング値
    phase2_memory_safety_factor = 0.5  # メモリ安全係数（0.0-1.0）
                                       # 0.5: 推奨（空きメモリの50%を使用）
                                       # 0.7: 積極的（OOMリスク増）
                                       # 0.3: 保守的（遅いが安全）
    phase2_min_batch_size = 256     # 最小バッチサイズ
    phase2_max_batch_size = 16384   # 最大バッチサイズ
    phase2_freeze_embedding = True  # Embedding凍結（標準採用）
                                    # PPL 66-72%改善、Accuracy 53-63%改善
                                    # Weight Tying時はOutput Headも凍結される

    @property
    def effective_phase2_batch_size(self):
        """Phase 2のバッチサイズを取得（GPUメモリベース自動計算）"""
        if self.phase2_batch_size is not None:
            return self.phase2_batch_size

        # GPUメモリに基づく自動計算
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # 目安: 1GB あたり 500トークン、安全係数0.8
            batch_size = int(gpu_memory_gb * 500 * 0.8)
        else:
            # CPU: 固定値
            batch_size = 512

        return batch_size

    # ========== データ ==========
    # ⚠️ UltraChat専用設定 (高速化により大規模データセットに対応)

    # トークナイザー設定
    tokenizer_name = "gpt2"                                # トークナイザーモデル名

    # Training data source
    train_data_source = "ultrachat"                        # UltraChat専用（キャッシュ使用で高速）
    train_val_split_ratio = 0.9                            # 訓練データの割合（storage mode用）

    # Validation data source
    # ⚠️ CRITICAL: Must use "text_file" only! auto_split is FORBIDDEN!
    # Validation data must contain ONLY tokens from training data
    # 自動生成: データ生成スクリプトが自動的に作成（訓練データの最後20%）
    val_data_source = "text_file"                          # MUST be "text_file" (auto_split is FORBIDDEN)
    val_text_file = "./data/ultrachat_50samples_val.txt"   # 自動生成される検証データ

    # UltraChat settings
    num_samples = 500                                      # 訓練サンプル数
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
    load_checkpoint = False                             # 新規モデルから開始（Colab実験用）
    save_checkpoint = True                              # 訓練終了時にチェックポイントを保存

    # ========== ログ出力 ==========
    log_every_steps = 1
    save_every_samples = 10
