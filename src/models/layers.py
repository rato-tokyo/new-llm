"""
CVFPレイヤーモジュール

Context Vector Fixed-Point (CVFP) 学習のためのカプセル化されたレイヤー
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
        self.use_dist_reg = use_dist_reg
        self.ema_momentum = ema_momentum
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

        # 分布正則化のためのEMA統計
        if use_dist_reg:
            # 次元ごとのランニング平均と分散
            self.register_buffer('running_mean', torch.zeros(context_dim))
            self.register_buffer('running_var', torch.ones(context_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

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

    def get_distribution_loss(self):
        """
        分布正則化損失を計算

        目標: 各次元がN(0, 1)に従うべき
        損失 = 平均のペナルティ + 分散のペナルティ

        Returns:
            dist_loss: 分布損失を表すスカラーテンソル
        """
        if not self.use_dist_reg:
            return torch.tensor(0.0, device=self.running_mean.device)

        # N(0, 1)からの逸脱にペナルティ
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
        layernorm_mix=0.0
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
                layernorm_mix=layernorm_mix
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

    def reset_running_stats(self):
        """全レイヤーの統計をリセット"""
        for layer in self.layers:
            layer.reset_running_stats()
