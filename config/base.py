"""
基本設定 (Base Configuration)

モデルアーキテクチャ、データ、デバイスの基本設定。
"""

import torch


class BaseConfig:
    """基本設定クラス"""

    # ========== モデルアーキテクチャ ==========
    num_layers = 1                  # ContextBlock と TokenBlock のレイヤー数（固定）
    embed_dim = 768                 # トークン埋め込み次元数（GPT-2: 768固定）
    context_dim = 500               # コンテキストベクトル次元数
    vocab_size = 50257              # GPT-2トークナイザーの語彙数
    use_pretrained_embeddings = True  # GPT-2事前学習済み埋め込みを使用
    use_weight_tying = True         # Weight Tying（パラメータ約38M削減）

    # ========== FFN (Feed-Forward Network) 設定 ==========
    fnn_type = "standard"           # FFNタイプ: "standard", "swiglu"
    fnn_expand_factor = 1           # 中間層の拡張率
    fnn_num_layers = 1              # FFN内の層数
    fnn_activation = "gelu"         # 活性化関数: "relu", "gelu"

    # ========== 複数トークン入力 ==========
    num_input_tokens = 1            # 入力するトークン数

    # ========== データ ==========
    tokenizer_name = "gpt2"
    train_data_source = "ultrachat"
    train_val_split_ratio = 0.9
    val_data_source = "text_file"
    val_text_file = "./cache/example_val.txt"  # data/ → cache/ に変更
    num_samples = 500
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    dataset_split = "train_sft"
    cache_dir = "./cache"

    # ========== デバイス ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed = 42

    # ========== 診断設定 ==========
    identity_mapping_threshold = 0.95
    identity_check_samples = 100

    # ========== チェックポイント ==========
    checkpoint_dir = "./checkpoints"
    checkpoint_path = "./checkpoints/model_latest.pt"
    load_checkpoint = False
    save_checkpoint = True

    # ========== ログ出力 ==========
    log_every_steps = 1
    save_every_samples = 10
