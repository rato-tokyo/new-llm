"""
CVFPLayer - コンテキスト更新レイヤー（基本単位）

Context Vector Fixed-Point (CVFP) 学習のための基本計算ユニット
Exponential Moving Average (EMA) を使用した分布正則化を内蔵
"""

import torch
import torch.nn as nn


class CVFPLayer(nn.Module):
    """
    コンテキスト更新レイヤー（基本単位）

    このレイヤーは以下をカプセル化:
    1. FNN（フィードフォワードニューラルネットワーク）によるコンテキスト更新
    2. トークン埋め込みの統合
    3. Residual接続
    4. オプションのLayerNorm正規化

    Args:
        context_dim: コンテキストベクトルの次元数
        embed_dim: トークン埋め込みの次元数
        hidden_dim: 隠れ層の次元数（context_dim + embed_dimと等しい必要がある）
        layernorm_mix: LayerNormの混合比率 (0.0 = 無効, 1.0 = 完全適用)
    """

    def __init__(
        self,
        context_dim,
        embed_dim,
        hidden_dim,
        layernorm_mix=0.0  # デフォルトで無効
    ):
        super().__init__()

        # 次元の検証
        if hidden_dim != context_dim + embed_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) は "
                f"context_dim ({context_dim}) + embed_dim ({embed_dim}) と等しい必要があります"
            )

        # 設定を保存
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layernorm_mix = layernorm_mix

        # FNN: [context + token] -> [hidden_dim]
        self.fnn = nn.Sequential(
            nn.Linear(context_dim + embed_dim, hidden_dim),
            nn.ReLU()
        )

        # オプションのLayerNorm
        if layernorm_mix > 0:
            self.context_norm = nn.LayerNorm(context_dim)
            self.token_norm = nn.LayerNorm(embed_dim)

        # 重みの初期化
        self._init_weights()

    def _init_weights(self):
        """レイヤーの重みを初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 恒等写像を防ぎつつ、値の爆発も防ぐバランスの取れた初期化
                # std=0.1で適度な多様性を維持
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    # バイアスも小さなランダム値で初期化（多様性向上）
                    nn.init.normal_(module.bias, mean=0.0, std=0.01)

    def forward(self, context, token_embed):
        """
        順伝播: コンテキストとトークンを更新

        Args:
            context: 現在のコンテキスト [batch, context_dim]
            token_embed: トークン埋め込み [batch, embed_dim]

        Returns:
            new_context: 更新されたコンテキスト [batch, context_dim]
            new_token: 更新されたトークン埋め込み [batch, embed_dim]
        """
        # 入力を結合
        fnn_input = torch.cat([context, token_embed], dim=-1)

        # FNN順伝播
        fnn_output = self.fnn(fnn_input)

        # 出力を分割
        delta_context = fnn_output[:, :self.context_dim]
        delta_token = fnn_output[:, self.context_dim:]

        # Residual接続
        new_context = context + delta_context
        new_token = token_embed + delta_token

        # オプションのLayerNorm混合
        if self.layernorm_mix > 0:
            context_normed = self.context_norm(new_context)
            token_normed = self.token_norm(new_token)

            mix = self.layernorm_mix
            new_context = (1 - mix) * new_context + mix * context_normed
            new_token = (1 - mix) * new_token + mix * token_normed

        return new_context, new_token

    def extra_repr(self):
        """デバッグ用の文字列表現"""
        return (
            f'context_dim={self.context_dim}, '
            f'embed_dim={self.embed_dim}, '
            f'hidden_dim={self.hidden_dim}, '
            f'layernorm_mix={self.layernorm_mix}'
        )
