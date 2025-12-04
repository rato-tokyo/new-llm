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
    num_epochs = 30                 # 最大エポック数
    early_stopping_patience = 3     # Early Stopping: 何エポック改善しなければ停止

    # ========== トークナイザー ==========
    tokenizer_name = "EleutherAI/pythia-70m"
