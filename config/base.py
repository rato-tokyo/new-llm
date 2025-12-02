"""
基本設定 (Base Configuration)

モデルアーキテクチャ、データ、デバイスの基本設定。
"""

import torch


class BaseConfig:
    """基本設定クラス"""

    # ========== モデルアーキテクチャ ==========
    num_layers = 2                  # ContextBlock と TokenBlock のレイヤー数（G案は2以上推奨）
    embed_dim = 768                 # トークン埋め込み次元数（GPT-2: 768固定）
    context_dim = 500               # コンテキストベクトル次元数
    vocab_size = 50257              # GPT-2トークナイザーの語彙数
    use_pretrained_embeddings = True  # GPT-2事前学習済み埋め込みを使用
    use_weight_tying = True         # Weight Tying（パラメータ約38M削減）

    # ========== 複数トークン入力 ==========
    num_input_tokens = 1            # 入力するトークン数

    # ========== データ ==========
    tokenizer_name = "gpt2"
    val_data_source = "text_file"
    val_text_file = "./cache/example_val.txt"
    num_samples = 500
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    dataset_split = "train_sft"
    cache_dir = "./cache"

    # ========== デバイス ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed = 42
