"""
CVFPBlock - 複数レイヤーのグループ化

複数のCVFPLayerを順次実行し、損失を集約するブロック
"""

import torch.nn as nn
from .layer import CVFPLayer


class CVFPBlock(nn.Module):
    """
    複数のCVFPLayerインスタンスから構成される多層CVFPブロック（グループ化）

    **CVFPLayerとの違い**:
    - CVFPLayer: 1回のコンテキスト更新（基本単位）
    - CVFPBlock: 複数のCVFPLayerを順次実行（グループ化）

    **例**:
    layer_structure = [2, 3] の場合:
    - Block 0: 2つのCVFPLayerを順次実行
    - Block 1: 3つのCVFPLayerを順次実行

    Args:
        num_layers: このブロック内のCVFPレイヤー数
        context_dim: コンテキストベクトルの次元数
        embed_dim: トークン埋め込みの次元数
        hidden_dim: 各レイヤーの隠れ層次元数
        use_dist_reg: 分布正則化を有効化
        ema_momentum: EMAのモメンタム
        layernorm_mix: LayerNorm混合比率
    """

    def __init__(
        self,
        num_layers,
        context_dim,
        embed_dim,
        hidden_dim,
        use_dist_reg=True,
        ema_momentum=0.99,
        layernorm_mix=0.0,
        enable_cvfp_learning=False  # CVFP学習を有効化
    ):
        super().__init__()

        self.num_layers = num_layers

        # CVFPレイヤーのスタックを作成
        self.layers = nn.ModuleList([
            CVFPLayer(
                context_dim=context_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                use_dist_reg=use_dist_reg,
                ema_momentum=ema_momentum,
                layernorm_mix=layernorm_mix,
                enable_cvfp_learning=enable_cvfp_learning
            )
            for _ in range(num_layers)
        ])

    def forward(self, context, token_embed):
        """
        ブロック内の全レイヤーを順次実行

        Args:
            context: [batch, context_dim]
            token_embed: [batch, embed_dim]

        Returns:
            context: 更新されたコンテキスト [batch, context_dim]
            token_embed: 更新されたトークン [batch, embed_dim]
        """
        for layer in self.layers:
            context, token_embed = layer(context, token_embed)

        return context, token_embed

    def get_distribution_loss(self):
        """全レイヤーからの分布損失を集約"""
        total_loss = 0.0
        for layer in self.layers:
            total_loss += layer.get_distribution_loss()
        return total_loss / len(self.layers)  # レイヤー間で平均

    def get_cvfp_loss(self):
        """全レイヤーからのCVFP損失を集約"""
        total_loss = 0.0
        for layer in self.layers:
            total_loss += layer.get_cvfp_loss()
        return total_loss / len(self.layers)  # レイヤー間で平均

    def reset_running_stats(self):
        """全レイヤーの統計をリセット"""
        for layer in self.layers:
            layer.reset_running_stats()

    def reset_cvfp_state(self):
        """全レイヤーのCVFP学習状態をリセット"""
        for layer in self.layers:
            layer.reset_cvfp_state()
