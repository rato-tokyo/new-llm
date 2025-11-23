"""
New-LLM: Residual接続アーキテクチャ（リファクタリング版2）

CVFPLayerを使用したクリーンな実装。

主な改善点:
1. CVFPLayer/CVFPBlockを使用（cvfp/モジュールから）
2. 分布正則化はレイヤー内部で自動処理
3. よりクリーンな順伝播
4. 関心の分離の改善
5. GPT-2事前学習済み埋め込みのサポート
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
        layernorm_mix: LayerNorm混合比率、0.0=無効 (デフォルト: 0.0)
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        context_dim,
        hidden_dim,
        layer_structure,
        layernorm_mix=0.0,
        use_pretrained_embeddings=False
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
        self.use_pretrained_embeddings = use_pretrained_embeddings


        # ========== トークン埋め込み ==========
        if use_pretrained_embeddings:
            # GPT-2事前学習済み埋め込みを読み込み
            self._load_pretrained_embeddings()
        else:
            # ランダム初期化
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_norm = nn.LayerNorm(embed_dim)

        # ========== CVFPブロック ==========
        self.blocks = nn.ModuleList([
            CVFPBlock(
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                layernorm_mix=layernorm_mix
            )
            for num_layers in layer_structure
        ])

        # ========== 出力ヘッド ==========
        # コンテキストから次のトークンを予測
        self.token_output = nn.Linear(context_dim, vocab_size)

        # 埋め込みの初期化（事前学習済みでない場合のみ）
        if not use_pretrained_embeddings:
            self._init_weights()

    def _load_pretrained_embeddings(self):
        """GPT-2事前学習済み埋め込みを読み込み"""
        try:
            from transformers import GPT2Model
            print("Loading GPT-2 pretrained embeddings...")

            # GPT-2モデルから埋め込み層のみ取得
            gpt2 = GPT2Model.from_pretrained('gpt2')
            pretrained_embeddings = gpt2.wte.weight.data  # [vocab_size, 768]

            # 埋め込み層を作成してコピー
            self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.token_embedding.weight.data.copy_(pretrained_embeddings)

            # 埋め込みを固定（Phase 1では学習しない）
            self.token_embedding.weight.requires_grad = False

            print(f"✓ Loaded GPT-2 embeddings: {pretrained_embeddings.shape}")

        except Exception as e:
            print(f"Warning: Failed to load GPT-2 embeddings: {e}")
            print("Falling back to random initialization...")
            self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

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
