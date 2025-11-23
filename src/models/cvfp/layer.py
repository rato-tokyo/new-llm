"""
CVFPLayer - コンテキスト更新レイヤー（基本単位）

Context Vector Fixed-Point (CVFP) 学習のための基本計算ユニット
Exponential Moving Average (EMA) を使用した分布正則化を内蔵
"""

import torch
import torch.nn as nn


class CVFPLayer(nn.Module):
    """
    分布追跡機能を内蔵したコンテキスト更新レイヤー（基本単位）

    このレイヤーは以下をカプセル化:
    1. FNN（フィードフォワードニューラルネットワーク）によるコンテキスト更新
    2. トークン埋め込みの統合
    3. Residual接続
    4. 分布正則化のためのExponential Moving Average (EMA) 統計

    Args:
        context_dim: コンテキストベクトルの次元数
        embed_dim: トークン埋め込みの次元数
        hidden_dim: 隠れ層の次元数（context_dim + embed_dimと等しい必要がある）
        use_dist_reg: 分布正則化を有効化 (デフォルト: True)
        ema_momentum: EMA統計のモメンタム (デフォルト: 0.99)
        layernorm_mix: LayerNormの混合比率 (0.0 = 無効, 1.0 = 完全適用)
    """

    def __init__(
        self,
        context_dim,
        embed_dim,
        hidden_dim,
        use_dist_reg=True,
        ema_momentum=0.99,
        layernorm_mix=0.0,  # デフォルトで無効
        enable_cvfp_learning=False  # CVFP学習を有効化
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
        self.use_dist_reg = use_dist_reg
        self.ema_momentum = ema_momentum
        self.layernorm_mix = layernorm_mix
        self.enable_cvfp_learning = enable_cvfp_learning

        # FNN: [context + token] -> [hidden_dim]
        self.fnn = nn.Sequential(
            nn.Linear(context_dim + embed_dim, hidden_dim),
            nn.ReLU()
        )

        # オプションのLayerNorm
        if layernorm_mix > 0:
            self.context_norm = nn.LayerNorm(context_dim)
            self.token_norm = nn.LayerNorm(embed_dim)

        # CVFP学習はPhase1Trainerで管理されるため、ここでは保存しない
        # （enable_cvfp_learningフラグは後方互換性のため残すが使用しない）

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

        # CVFP学習はPhase1Trainerで管理（ここでは何もしない）

        return new_context, new_token

    def get_cvfp_loss(self):
        """
        CVFP損失を取得

        注: CVFP損失はPhase1Trainerで計算されるため、ここでは常に0を返す。
        このメソッドは後方互換性のため残している。

        Returns:
            cvfp_loss: 常に0.0
        """
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=False)

    def reset_cvfp_state(self):
        """
        CVFP学習状態をリセット

        注: CVFP状態はPhase1Trainerで管理されるため、ここでは何もしない。
        このメソッドは後方互換性のため残している。
        """
        pass  # 何もしない

    def get_distribution_loss(self):
        """
        分布正則化損失を計算

        注: 共分散正則化はPhase1Trainerで直接計算されるため、ここでは常に0を返す。
        このメソッドは後方互換性のため残している。

        Returns:
            dist_loss: 常に0.0
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def reset_running_stats(self):
        """ランニング統計をリセット（互換性のため残すが何もしない）"""
        pass

    def extra_repr(self):
        """デバッグ用の文字列表現"""
        return (
            f'context_dim={self.context_dim}, '
            f'embed_dim={self.embed_dim}, '
            f'hidden_dim={self.hidden_dim}, '
            f'use_dist_reg={self.use_dist_reg}, '
            f'ema_momentum={self.ema_momentum}, '
            f'layernorm_mix={self.layernorm_mix}'
        )
