"""
Pretrained MKA-Attention: Two-Stage Attention with Frozen Pretrained Pythia

事前学習済みPythiaのAttentionを凍結してStage 1として使用し、
Stage 2（KA部分）のみを学習する方式。

Stage 1 (Frozen): 事前学習済みPythiaのAttention → local A
Stage 2 (Trainable): KA attention（A同士のattention）→ global output

メリット:
- Stage 1は既に言語モデリングに最適化済み
- Stage 2のみを学習すればよい
- 学習が速く、収束しやすい
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPTNeoXForCausalLM


class PretrainedMKALayer(nn.Module):
    """
    Pretrained MKA Layer

    - Stage 1: 事前学習済みPythiaのAttention（凍結）
    - Stage 2: KA attention（学習可能）
    """

    def __init__(
        self,
        pretrained_layer: nn.Module,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Stage 1: 事前学習済みのlayer（凍結）
        self.pretrained_layer = pretrained_layer
        for param in self.pretrained_layer.parameters():
            param.requires_grad = False

        # Stage 2用のprojection（学習可能）
        self.query_a = nn.Linear(hidden_size, hidden_size)
        self.key_a = nn.Linear(hidden_size, hidden_size)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ===== Stage 1: Pretrained Pythia Layer (Frozen) =====
        # 事前学習済みの層を通してlocal Aを取得
        # pretrained_layerはGPTNeoXLayerで、attention + MLPを含む
        # Note: パラメータはrequires_grad=Falseで凍結済みなので
        # torch.no_grad()は使わない（勾配が流れなくなる問題を回避）
        A_local = self.pretrained_layer(hidden_states)[0]
        # GPTNeoXLayerは(hidden_states, present_key_value)を返す

        # ===== Stage 2: KA Attention (Trainable) =====
        # A同士のattentionを計算

        # Stage 2用のQ, K
        query_a = self.query_a(A_local)  # [batch, seq, hidden]
        key_a = self.key_a(A_local)  # [batch, seq, hidden]

        query_a = query_a.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_a = key_a.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query_a = query_a.transpose(1, 2)  # [batch, heads, seq, head_dim]
        key_a = key_a.transpose(1, 2)  # [batch, heads, seq, head_dim]

        # A_localをheads形式に変換
        A_local_heads = A_local.view(batch_size, seq_len, self.num_heads, self.head_dim)
        A_local_heads = A_local_heads.transpose(1, 2)  # [batch, heads, seq, head_dim]

        # Global attention scores
        attn_weights = torch.matmul(query_a, key_a.transpose(-1, -2)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask

        attn_probs = F.softmax(attn_weights, dim=-1)

        # Global attention output
        attn_output = torch.matmul(attn_probs, A_local_heads)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return attn_output


class PretrainedMKAModel(nn.Module):
    """
    Pretrained MKA-Pythia Model

    事前学習済みPythia-70Mをベースに、KA attentionを追加。
    PythiaのAttention部分は凍結し、KA部分のみを学習。

    構造:
    - Embedding: 事前学習済み（凍結）
    - Layers: 事前学習済みAttention（凍結） + KA projection（学習可能）
    - LM Head: 事前学習済み（凍結）
    """

    def __init__(
        self,
        pretrained_model_name: str = "EleutherAI/pythia-70m",
        freeze_pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained model
        self.pretrained = GPTNeoXForCausalLM.from_pretrained(pretrained_model_name)
        config = self.pretrained.config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads

        # Freeze pretrained parts if specified
        if freeze_pretrained:
            for param in self.pretrained.parameters():
                param.requires_grad = False

        # Create MKA layers with pretrained attention
        self.mka_layers = nn.ModuleList([
            PretrainedMKALayer(
                pretrained_layer=self.pretrained.gpt_neox.layers[i],
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
            )
            for i in range(self.num_layers)
        ])

        # Output projection (trainable)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embedding (frozen pretrained)
        with torch.no_grad():
            hidden_states = self.pretrained.gpt_neox.embed_in(input_ids)

        # MKA Layers
        for layer in self.mka_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Output projection
        hidden_states = self.output_proj(hidden_states)

        # Final layer norm (frozen pretrained)
        with torch.no_grad():
            hidden_states = self.pretrained.gpt_neox.final_layer_norm(hidden_states)

        # LM Head (frozen pretrained)
        with torch.no_grad():
            logits = self.pretrained.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }


class PretrainedMKAModelV2(nn.Module):
    """
    Pretrained MKA-Pythia Model V2

    より単純な設計:
    - Stage 1: 事前学習済みPythia全体を通す（凍結）→ A
    - Stage 2: A同士のKA attention（学習可能）

    Pythia全体を1つのブロックとして扱い、その出力Aに対してKA attentionを適用。
    """

    def __init__(
        self,
        pretrained_model_name: str = "EleutherAI/pythia-70m",
    ):
        super().__init__()

        # Load pretrained model
        self.pretrained = GPTNeoXForCausalLM.from_pretrained(pretrained_model_name)
        config = self.pretrained.config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Freeze all pretrained parameters
        for param in self.pretrained.parameters():
            param.requires_grad = False

        # Stage 2: KA attention (trainable)
        self.query_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Layer norm for KA output
        self.ka_layer_norm = nn.LayerNorm(self.hidden_size)

        self.scale = self.head_dim ** -0.5

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize trainable weights"""
        for module in [self.query_a, self.key_a, self.output_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.ones_(self.ka_layer_norm.weight)
        nn.init.zeros_(self.ka_layer_norm.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # ===== Stage 1: Pretrained Pythia (Frozen) =====
        # 事前学習済みPythia全体を通してAを取得
        # Note: パラメータはrequires_grad=Falseで凍結済みなので
        # torch.no_grad()は不要（勾配が流れなくなる問題を回避）
        outputs = self.pretrained.gpt_neox(input_ids)
        A = outputs.last_hidden_state  # [batch, seq, hidden]

        # ===== Stage 2: KA Attention (Trainable) =====
        # A同士のattentionを計算

        # Q, K for KA attention
        query_a = self.query_a(A)
        key_a = self.key_a(A)

        query_a = query_a.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_a = key_a.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query_a = query_a.transpose(1, 2)  # [batch, heads, seq, head_dim]
        key_a = key_a.transpose(1, 2)

        # A in heads format
        A_heads = A.view(batch_size, seq_len, self.num_heads, self.head_dim)
        A_heads = A_heads.transpose(1, 2)

        # Attention scores
        attn_weights = torch.matmul(query_a, key_a.transpose(-1, -2)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float("-inf"),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask

        attn_probs = F.softmax(attn_weights, dim=-1)

        # KA output
        ka_output = torch.matmul(attn_probs, A_heads)
        ka_output = ka_output.transpose(1, 2).contiguous()
        ka_output = ka_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection + residual + layer norm
        ka_output = self.output_proj(ka_output)
        hidden_states = self.ka_layer_norm(A + ka_output)

        # LM Head
        # Note: embed_outのパラメータは凍結済みだが、
        # hidden_statesから勾配を流すためにtorch.no_grad()は使わない
        logits = self.pretrained.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }
