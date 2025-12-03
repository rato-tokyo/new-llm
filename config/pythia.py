"""
Pythia-70M Configuration

Pythia-70Mモデルの設定。
https://huggingface.co/EleutherAI/pythia-70m
"""


class PythiaConfig:
    """Pythia-70M モデル設定 (Baseline)"""

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


class ContextPythiaConfig(PythiaConfig):
    """
    Context-Pythia モデル設定（KVキャッシュ圧縮版）

    アーキテクチャ:
    - Token Embedding: vocab → embed_dim (512)
    - ContextBlock: embed_dim (512) → context_dim (256)  ← 圧縮
    - PythiaLayer × 6: context_dim (256) で動作  ← Baselineと同じ構造
    - Output Head: context_dim (256) → vocab

    KVキャッシュ削減:
    - Baseline: hidden_size (512) × seq_len × num_layers
    - Context-Pythia: context_dim (256) × seq_len × num_layers
    - 削減率: 50%
    """

    # ========== Embedding次元 ==========
    embed_dim = 512                 # Token Embedding dimension (Pythiaと同じ)

    # ========== Context次元（圧縮後）==========
    # ⚠️ 重要: context_dimはnum_attention_heads (8) で割り切れる値にすること
    # 例: 256, 320, 384, 448, 512 など
    context_dim = 320               # 圧縮後の次元（PythiaLayerはこの次元で動作）

    # ========== Transformer設定（context_dimに合わせてスケール）==========
    # intermediate_size: 2048 * (256/512) = 1024
    intermediate_size = 1024

    # ========== チェックポイント ==========
    phase1_checkpoint_path = "checkpoints/context_block_pythia_phase1.pt"

    @classmethod
    def kv_cache_reduction(cls) -> float:
        """KVキャッシュ削減率を計算"""
        return 1.0 - (cls.context_dim / cls.embed_dim)
