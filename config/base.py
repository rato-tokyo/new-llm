"""
基本設定 (Base Configuration)

モデルアーキテクチャ、データ、デバイスの基本設定。

1層固定アーキテクチャ（2025-12-02）:
- カスケード連結方式により複数レイヤーは不要
- num_layers は削除
"""

import torch


class BaseConfig:
    """基本設定クラス"""

    # ========== モデルアーキテクチャ（1層固定） ==========
    embed_dim = 768                 # トークン埋め込み次元数（GPT-2: 768固定）
    context_dim = 256               # コンテキストベクトル次元数
    vocab_size = 50257              # GPT-2トークナイザーの語彙数
    use_pretrained_embeddings = True  # GPT-2事前学習済み埋め込みを使用

    # ========== Context-KV Attention ==========
    context_interval = 32           # Contextを取得する間隔
                                    # Position i から i, i-interval, i-2*interval, ... を使用
    num_heads = 8                   # Attention head数
    max_contexts = 32               # 使用するcontext数の上限（context window）
                                    # OOM防止のため、通常LLMのcontext windowと同様の制限

    # ========== データ ==========
    tokenizer_name = "gpt2"
    val_data_source = "text_file"
    val_text_file = "./cache/example_val.txt"
    num_samples = 1600
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    dataset_split = "train_sft"
    cache_dir = "./cache"

    # ========== デバイス ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed = 42
