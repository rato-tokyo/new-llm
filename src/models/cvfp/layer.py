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

        # 分布正則化のためのEMA統計
        if use_dist_reg:
            # 次元ごとのランニング平均と分散
            self.register_buffer('running_mean', torch.zeros(context_dim))
            self.register_buffer('running_var', torch.ones(context_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # CVFP学習はPhase1Trainerで管理されるため、ここでは保存しない
        # （enable_cvfp_learningフラグは後方互換性のため残すが使用しない）

        # 重みの初期化
        self._init_weights()

    def _init_weights(self):
        """レイヤーの重みを初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 恒等写像を防ぐため、やや大きめに初期化
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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

        # ランニング統計を更新（訓練モードのみ）
        if self.training and self.use_dist_reg:
            self._update_running_stats(new_context)

        # CVFP学習はPhase1Trainerで管理（ここでは何もしない）

        return new_context, new_token

    def _update_running_stats(self, context):
        """
        EMAを使用してランニング平均と分散を更新

        これは訓練モードでの順伝播時に自動的に呼ばれる。
        クリーンなカプセル化のため、外部の呼び出し元からは隠蔽されている。

        Args:
            context: 現在のバッチのコンテキスト [batch, context_dim]
        """
        with torch.no_grad():
            # バッチ統計を計算
            batch_mean = context.mean(dim=0)  # [context_dim]
            batch_var = context.var(dim=0, unbiased=False)  # [context_dim]

            # EMA更新
            momentum = self.ema_momentum
            self.running_mean = momentum * self.running_mean + (1 - momentum) * batch_mean
            self.running_var = momentum * self.running_var + (1 - momentum) * batch_var

            # 更新回数を追跡
            self.num_batches_tracked += 1

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

        注: この実装ではEMA統計（勾配なし）を使用。
        Phase1Trainerで現在のコンテキストから直接計算する必要がある。

        Returns:
            dist_loss: 常に0.0（後方互換性のため残す）
        """
        if not self.use_dist_reg:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # EMA統計からの損失（勾配なし）
        # これは情報提供のみで、最適化には使用されない
        with torch.no_grad():
            mean_penalty = (self.running_mean ** 2).mean()
            var_penalty = ((self.running_var - 1.0) ** 2).mean()
            return mean_penalty + var_penalty

    def reset_running_stats(self):
        """ランニング統計をリセット（新しい訓練実行時に有用）"""
        if self.use_dist_reg:
            self.running_mean.zero_()
            self.running_var.fill_(1.0)
            self.num_batches_tracked.zero_()

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
