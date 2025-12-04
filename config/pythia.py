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
    learning_rate = 1e-4            # 学習率
    batch_size = 8                  # バッチサイズ
    num_epochs = 10                 # 最大エポック数
    early_stopping_patience = 3    # Early Stopping: 何エポック改善しなければ停止

    # ========== トークナイザー ==========
    tokenizer_name = "EleutherAI/pythia-70m"


class DProjPythiaConfig(PythiaConfig):
    """
    DProj-Pythia モデル設定（KVキャッシュ圧縮版）

    アーキテクチャ:
    - Token Embedding: vocab → embed_dim (512)
    - DiverseProjection: embed_dim (512) → proj_dim (320)  ← 圧縮
    - PythiaLayer × 6: proj_dim (320) で動作  ← Baselineと同じ構造
    - Output Head: proj_dim (320) → vocab

    KVキャッシュ削減:
    - Baseline: hidden_size (512) × seq_len × num_layers
    - DProj-Pythia: proj_dim (320) × seq_len × num_layers
    - 削減率: 37.5%
    """

    # ========== Embedding次元 ==========
    embed_dim = 512                 # Token Embedding dimension (Pythiaと同じ)

    # ========== Projection次元（圧縮後）==========
    # ⚠️ 重要: proj_dimはnum_attention_heads (8) で割り切れる値にすること
    # 例: 256, 320, 384, 448, 512 など
    proj_dim = 320                  # 圧縮後の次元（PythiaLayerはこの次元で動作）

    # ========== Transformer設定（proj_dimに合わせてスケール）==========
    # intermediate_size: 2048 * (320/512) = 1280
    intermediate_size = 1280

    # ========== チェックポイント ==========
    dproj_checkpoint_path = "checkpoints/dproj_pythia.pt"

    @classmethod
    def kv_cache_reduction(cls) -> float:
        """KVキャッシュ削減率を計算"""
        return 1.0 - (cls.proj_dim / cls.embed_dim)


# Backward compatibility alias
ContextPythiaConfig = DProjPythiaConfig
