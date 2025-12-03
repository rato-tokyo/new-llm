"""
Pythia-70M Configuration

Pythia-70Mモデルの設定。
https://huggingface.co/EleutherAI/pythia-70m
"""


class PythiaConfig:
    """Pythia-70M モデル設定"""

    # ========== モデル構造 ==========
    vocab_size = 50304              # Pythiaの語彙サイズ
    hidden_size = 512               # 隠れ層の次元
    num_layers = 6                  # レイヤー数
    num_attention_heads = 8         # アテンションヘッド数
    intermediate_size = 2048        # FFNの中間層サイズ
    max_position_embeddings = 2048  # 最大シーケンス長
    rotary_pct = 0.25               # Rotary embeddingの割合

    # ========== 学習設定 ==========
    learning_rate = 1e-4            # Phase 2 学習率
    batch_size = 8                  # バッチサイズ

    # ========== トークナイザー ==========
    tokenizer_name = "EleutherAI/pythia-70m"

    # ========== チェックポイント ==========
    phase1_checkpoint_path = "checkpoints/context_block_pythia_phase1.pt"


class ContextPythiaConfig(PythiaConfig):
    """Context-Pythia モデル設定（KVキャッシュ圧縮版）"""

    # ========== Context次元 ==========
    context_dim = 300               # 圧縮後のcontext次元（50%削減）

    # ========== KVキャッシュ削減 ==========
    # 元: hidden_size (512) × seq_len × num_layers (6)
    # 圧縮後: context_dim (256) × seq_len × num_layers (6)
    # 削減率: 50%

    @classmethod
    def kv_cache_reduction(cls) -> float:
        """KVキャッシュ削減率を計算"""
        return 1.0 - (cls.context_dim / cls.hidden_size)
