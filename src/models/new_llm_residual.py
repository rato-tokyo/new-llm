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

        # Phase 1訓練用の内部状態
        self.phase1_optimizer = None
        self.dist_reg_weight = 0.2  # デフォルト値
        self.phase1_config = None  # Phase 1設定を保存
        self.current_iteration = 0  # 現在のイテレーション番号

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

        # 自動最適化（enable_cvfp_learning=Trueの場合）
        self.auto_optimize_if_enabled()

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

    def get_distribution_loss(self):
        """
        全ブロックから集約された分布正則化損失を取得

        このメソッドは実装詳細を公開せずに、
        内部統計へのクリーンなアクセスを提供します。

        Returns:
            dist_loss: スカラーテンソル
        """
        if not self.use_dist_reg:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_loss = 0.0
        for block in self.blocks:
            total_loss += block.get_distribution_loss()

        return total_loss / len(self.blocks)  # ブロック間で平均

    def get_cvfp_loss(self):
        """
        全ブロックから集約されたCVFP損失を取得

        CVFP自己学習が有効な場合、各レイヤーが自動的に計算した
        前回出力との差（MSE）を集約して返す。

        Returns:
            cvfp_loss: スカラーテンソル
        """
        total_loss = 0.0
        for block in self.blocks:
            total_loss += block.get_cvfp_loss()

        return total_loss / len(self.blocks)  # ブロック間で平均

    def get_total_loss(self, dist_reg_weight=0.2):
        """
        CVFP損失と分布正則化損失を統合して取得

        Args:
            dist_reg_weight: 分布正則化の重み（0.0~1.0）

        Returns:
            total_loss: 統合損失（勾配あり）
            loss_dict: 損失の内訳 {'cvfp': float, 'dist': float, 'total': float}
        """
        cvfp_loss = self.get_cvfp_loss()

        if self.use_dist_reg:
            dist_loss = self.get_distribution_loss()
            total_loss = (1 - dist_reg_weight) * cvfp_loss + dist_reg_weight * dist_loss

            # 損失の内訳（.item()で勾配なしのfloatに変換）
            with torch.no_grad():
                loss_dict = {
                    'cvfp': cvfp_loss.item() if isinstance(cvfp_loss, torch.Tensor) else cvfp_loss,
                    'dist': dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss,
                    'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                }
        else:
            total_loss = cvfp_loss

            with torch.no_grad():
                loss_dict = {
                    'cvfp': cvfp_loss.item() if isinstance(cvfp_loss, torch.Tensor) else cvfp_loss,
                    'dist': 0.0,
                    'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                }

        return total_loss, loss_dict

    def reset_running_stats(self):
        """全ランニング統計をリセット（新しい訓練実行時用）"""
        for block in self.blocks:
            block.reset_running_stats()

    def reset_cvfp_state(self):
        """全CVFP学習状態をリセット（新しいイテレーション開始時用）"""
        for block in self.blocks:
            block.reset_cvfp_state()

    def setup_phase1_training(self, context_params, learning_rate, dist_reg_weight):
        """
        Phase 1訓練の初期化（全てをカプセル化）

        Args:
            context_params: 訓練対象パラメータのリスト
            learning_rate: 学習率
            dist_reg_weight: 分布正則化の重み
        """
        self.reset_running_stats()
        self.current_iteration = 0

        # Optimizerを作成
        self.phase1_optimizer = torch.optim.Adam(
            context_params,
            lr=learning_rate
        )
        self.dist_reg_weight = dist_reg_weight

    def start_phase1_iteration(self, iteration):
        """
        Phase 1の各イテレーション開始時の処理

        Args:
            iteration: 現在のイテレーション番号（0-indexed）
        """
        self.current_iteration = iteration

        # CVFP状態リセット（iteration 0以降）
        if iteration > 0:
            self.reset_cvfp_state()

    def auto_optimize_if_enabled(self):
        """
        自動最適化が有効な場合、backward + stepを実行

        enable_cvfp_learning=True かつ phase1_optimizerが設定されている場合、
        get_total_loss()で取得した損失に対してbackward + stepを実行。

        Returns:
            loss_dict: 損失の内訳（学習した場合）、None（学習しなかった場合）
        """
        if not self.enable_cvfp_learning or self.phase1_optimizer is None or not self.training:
            return None

        # 統合損失を取得
        total_loss, loss_dict = self.get_total_loss(self.dist_reg_weight)

        # CVFP損失が0の場合はskip（previous_context未設定）
        if loss_dict['cvfp'] > 0:
            self.phase1_optimizer.zero_grad()
            total_loss.backward()
            self.phase1_optimizer.step()
            return loss_dict

        return None
