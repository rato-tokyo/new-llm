"""
New-LLM: Residual接続アーキテクチャ（リファクタリング版2）

CVFPLayerを使用したクリーンな実装。

主な改善点:
1. CVFPLayer/CVFPBlockを使用（cvfp/モジュールから）
2. 分布正則化はレイヤー内部で自動処理
3. よりクリーンな順伝播
4. 関心の分離の改善
"""

import torch
import torch.nn as nn
from .cvfp import CVFPBlock


class NewLLMResidual(nn.Module):
    """
    ResNetスタイルのResidual接続を持つNew-LLM

    このバージョンはCVFPLayerを使用して以下をクリーンにカプセル化:
    - コンテキスト更新
    - トークン更新
    - 分布正則化（EMAベース）

    Args:
        vocab_size: 語彙サイズ
        embed_dim: トークン埋め込みの次元数
        context_dim: コンテキストベクトルの次元数
        hidden_dim: 隠れ層の次元数（embed_dim + context_dimと等しい必要がある）
        layer_structure: ブロックごとのレイヤー数を指定するリスト
        use_dist_reg: 分布正則化を有効化 (デフォルト: True)
        ema_momentum: ランニング統計のEMAモメンタム (デフォルト: 0.99)
        layernorm_mix: LayerNorm混合比率、0.0=無効 (デフォルト: 0.0)
        enable_cvfp_learning: CVFP自己学習を有効化 (デフォルト: False)
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        context_dim,
        hidden_dim,
        layer_structure,
        use_dist_reg=True,
        ema_momentum=0.99,
        layernorm_mix=0.0,
        enable_cvfp_learning=False
    ):
        super().__init__()

        # 次元の検証
        if hidden_dim != embed_dim + context_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) は "
                f"embed_dim ({embed_dim}) + context_dim ({context_dim}) と等しい必要があります"
            )

        # 設定を保存
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.layer_structure = layer_structure
        self.use_dist_reg = use_dist_reg
        self.enable_cvfp_learning = enable_cvfp_learning


        # ========== トークン埋め込み ==========
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)

        # ========== CVFPブロック ==========
        self.blocks = nn.ModuleList([
            CVFPBlock(
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                use_dist_reg=use_dist_reg,
                ema_momentum=ema_momentum,
                layernorm_mix=layernorm_mix,
                enable_cvfp_learning=enable_cvfp_learning
            )
            for num_layers in layer_structure
        ])

        # ========== 出力ヘッド ==========
        # コンテキストから次のトークンを予測
        self.token_output = nn.Linear(context_dim, vocab_size)

        # 埋め込みの初期化
        self._init_weights()

    def _init_weights(self):
        """埋め込み重みを初期化"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def _update_context_one_step(self, token_vec, context, return_token=False):
        """
        1トークンステップのコンテキスト更新

        Args:
            token_vec: トークンベクトル [batch, embed_dim]
            context: 現在のコンテキスト [batch, context_dim]
            return_token: Trueの場合、更新されたトークンも返す

        Returns:
            new_context: 更新されたコンテキスト [batch, context_dim]
            new_token: 更新されたトークン (return_token=Trueの場合)
        """
        current_context = context
        current_token = token_vec

        # 全ブロックを通して処理
        for block in self.blocks:
            current_context, current_token = block(current_context, current_token)

        if return_token:
            return current_context, current_token
        return current_context

    def forward(self, input_ids, return_context_trajectory=False):
        """
        モデルの順伝播

        Args:
            input_ids: 入力トークンID [batch, seq_len]
            return_context_trajectory: Trueの場合、全中間コンテキストを返す

        Returns:
            logits: 出力ロジット [batch, seq_len, vocab_size]
            context_trajectory: (オプション) 全コンテキスト [batch, seq_len, context_dim]
        """
        batch_size, seq_len = input_ids.shape

        # トークン埋め込みを取得
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        token_embeds = self.embed_norm(token_embeds)

        # コンテキストを初期化
        context = torch.zeros(
            batch_size, self.context_dim,
            device=input_ids.device,
            dtype=token_embeds.dtype
        )

        # シーケンスを処理
        contexts = []
        for t in range(seq_len):
            token_vec = token_embeds[:, t, :]  # [batch, embed_dim]
            context = self._update_context_one_step(token_vec, context)
            contexts.append(context)

        # コンテキストをスタック
        all_contexts = torch.stack(contexts, dim=1)  # [batch, seq_len, context_dim]

        # 次のトークンを予測
        logits = self.token_output(all_contexts)  # [batch, seq_len, vocab_size]

        if return_context_trajectory:
            return logits, all_contexts
        return logits
