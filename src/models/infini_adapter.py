"""
Infini-Attention Adapter for Pretrained LLMs

既存のLLM（Pythia等）のLayer 0をInfini-Attentionに置き換えるためのアダプター。

使用方法:
1. 蒸留: 既存Layer 0の入出力ペアでInfini Layerを訓練
2. 置き換え: 訓練済みInfini LayerでLayer 0を置き換え
3. ファインチューニング: 長文データでEnd-to-end訓練
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from src.models.infini_attention import InfiniAttention, InfiniAttentionALiBi


class InfiniAdapterLayer(nn.Module):
    """
    Infini-Attention Layer compatible with GPTNeoX/Pythia

    GPTNeoXLayerと同じインターフェースを持つ。
    Pre-LayerNorm + Parallel Attention/MLP (Pythia style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        use_delta_rule: bool = True,
        use_alibi: bool = False,
        alibi_scale: float = 1.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_alibi = use_alibi

        # Infini-Attention (declare type for mypy)
        self.attention: Union[InfiniAttention, InfiniAttentionALiBi]

        # Layer norms (Pythia style: pre-norm)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Dropouts (for compatibility, but we set to 0)
        self.post_attention_dropout = nn.Dropout(0.0)
        self.post_mlp_dropout = nn.Dropout(0.0)

        # Infini-Attention
        if use_alibi:
            self.attention = InfiniAttentionALiBi(
                hidden_size=hidden_size,
                num_heads=num_heads,
                use_delta_rule=use_delta_rule,
                alibi_scale=alibi_scale,
            )
        else:
            self.attention = InfiniAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                use_delta_rule=use_delta_rule,
            )

        # MLP (Pythia style)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        update_memory: bool = True,
        position_embeddings: Optional[tuple] = None,  # For compatibility with newer transformers
        cache_position: Optional[torch.Tensor] = None,  # For compatibility with newer transformers
        **kwargs,  # Catch any other arguments
    ) -> tuple:
        """
        GPTNeoXLayerと同じシグネチャ

        Returns:
            tuple: (hidden_states, present, attentions)
        """
        residual = hidden_states

        # Pre-LayerNorm
        ln_out = self.input_layernorm(hidden_states)

        # Parallel Attention + MLP (Pythia style)
        # Note: Infini Attention does not use position embeddings (NoPE)
        attn_output = self.attention(ln_out, attention_mask, update_memory=update_memory)
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))

        # Combine
        hidden_states = residual + self.post_attention_dropout(attn_output) + self.post_mlp_dropout(mlp_output)

        # Return format compatible with GPTNeoXLayer
        outputs: tuple = (hidden_states,)

        if use_cache:
            outputs = outputs + (None,)  # No KV cache for Infini

        if output_attentions:
            outputs = outputs + (None,)  # No attention weights

        return outputs

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """メモリをリセット"""
        self.attention.reset_memory(device)

    def get_memory_state(self) -> dict:
        """メモリ状態を取得"""
        return self.attention.get_memory_state()

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        """メモリ状態を設定"""
        self.attention.set_memory_state(state, device)


class PythiaWithInfiniLayer(nn.Module):
    """
    Pythia model with Layer 0 replaced by Infini-Attention

    HuggingFace GPTNeoXForCausalLMをラップし、Layer 0を置き換える。
    """

    def __init__(
        self,
        base_model,
        use_delta_rule: bool = True,
        use_alibi: bool = False,
        alibi_scale: float = 1.0,
        freeze_other_layers: bool = True,
    ):
        """
        Args:
            base_model: GPTNeoXForCausalLM instance
            use_delta_rule: Delta Ruleを使用するか
            use_alibi: ALiBiを使用するか
            alibi_scale: ALiBiスロープスケール
            freeze_other_layers: Layer 0以外をfreezeするか
        """
        super().__init__()

        self.base_model = base_model
        self.config = base_model.config

        # Create Infini Layer
        self.infini_layer = InfiniAdapterLayer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            use_delta_rule=use_delta_rule,
            use_alibi=use_alibi,
            alibi_scale=alibi_scale,
            layer_norm_eps=self.config.layer_norm_eps,
        )

        # Store original layer 0 for distillation
        self.original_layer0 = base_model.gpt_neox.layers[0]

        # Freeze other layers if specified
        if freeze_other_layers:
            self._freeze_other_layers()

    def _freeze_other_layers(self) -> None:
        """Layer 0以外をfreeze"""
        # Freeze embeddings
        for param in self.base_model.gpt_neox.embed_in.parameters():
            param.requires_grad = False

        # Freeze layers 1-N
        for i, layer in enumerate(self.base_model.gpt_neox.layers):
            if i > 0:
                for param in layer.parameters():
                    param.requires_grad = False

        # Freeze final layer norm
        for param in self.base_model.gpt_neox.final_layer_norm.parameters():
            param.requires_grad = False

        # Freeze LM head
        for param in self.base_model.embed_out.parameters():
            param.requires_grad = False

    def use_infini_layer(self, use: bool = True) -> None:
        """Infini LayerをLayer 0として使用するかどうか"""
        if use:
            self.base_model.gpt_neox.layers[0] = self.infini_layer
        else:
            self.base_model.gpt_neox.layers[0] = self.original_layer0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ):
        """
        Forward pass

        Note: update_memoryはInfini Layer使用時のみ有効
        """
        # Set update_memory for infini layer if it's active
        if isinstance(self.base_model.gpt_neox.layers[0], InfiniAdapterLayer):
            # Temporarily modify the layer's forward behavior
            # This is a bit hacky but necessary for compatibility
            pass

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def reset_memory(self) -> None:
        """Infini Layerのメモリをリセット"""
        self.infini_layer.reset_memory()

    def get_memory_state(self) -> dict:
        """メモリ状態を取得"""
        return self.infini_layer.get_memory_state()

    def set_memory_state(self, state: dict) -> None:
        """メモリ状態を設定"""
        device = next(self.parameters()).device
        self.infini_layer.set_memory_state(state, device)

    def distill_layer0(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Layer 0の入出力を取得（蒸留用）

        Returns:
            layer0_input: Layer 0への入力
            layer0_output: Layer 0からの出力
        """
        with torch.no_grad():
            # Get embeddings
            hidden_states = self.base_model.gpt_neox.embed_in(input_ids)
            layer0_input = hidden_states.clone()

            # Get position embeddings (required for GPTNeoX)
            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

            # Get rotary embeddings
            position_embeddings = self.base_model.gpt_neox.rotary_emb(
                hidden_states, position_ids
            )

            # Run through original layer 0
            outputs = self.original_layer0(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            layer0_output = outputs[0]

        return layer0_input, layer0_output

    def compute_distillation_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        蒸留損失を計算

        Returns:
            MSE loss between original and infini layer outputs
        """
        # Get target from original layer
        layer0_input, target = self.distill_layer0(input_ids, attention_mask)

        # Get prediction from infini layer
        outputs = self.infini_layer(
            layer0_input,
            attention_mask=attention_mask,
            update_memory=True,
        )
        prediction = outputs[0]

        # MSE loss
        loss = torch.nn.functional.mse_loss(prediction, target)

        return loss


def create_pythia_with_infini(
    model_name: str = "EleutherAI/pythia-70m",
    use_delta_rule: bool = True,
    use_alibi: bool = False,
    alibi_scale: float = 1.0,
    freeze_other_layers: bool = True,
) -> PythiaWithInfiniLayer:
    """
    Infini Layer付きPythiaモデルを作成

    Args:
        model_name: HuggingFaceモデル名
        use_delta_rule: Delta Ruleを使用するか
        use_alibi: ALiBiを使用するか
        alibi_scale: ALiBiスロープスケール
        freeze_other_layers: Layer 0以外をfreezeするか

    Returns:
        PythiaWithInfiniLayer instance
    """
    from transformers import GPTNeoXForCausalLM

    base_model = GPTNeoXForCausalLM.from_pretrained(model_name)

    return PythiaWithInfiniLayer(
        base_model=base_model,
        use_delta_rule=use_delta_rule,
        use_alibi=use_alibi,
        alibi_scale=alibi_scale,
        freeze_other_layers=freeze_other_layers,
    )
