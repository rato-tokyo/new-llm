"""
KA Training Model (案2)

学習時からKAキャッシュ方式を使用するモデル。
推論時も同じ方式を使うことでtrain-inference mismatchを解消。

方式:
  A[0] = V[0]                                   (ブートストラップ)
  A[i] = renormalize(weights[:i]) @ A[:i]       (過去のAのみ、現在のVは不使用)

注意:
  この方式では誤差が再帰的に蓄積する。
  A[j] が A[1:j-1] に依存し、A[j+1] が A[j] に依存するため。
  ただし、学習と推論で一貫しているため、モデルはこの誤差を学習で吸収できる。
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.rope import RotaryEmbedding


class KATrainAttention(nn.Module):
    """
    学習時からKAキャッシュを使用するAttention

    並列計算版とautoregressive版の両方を提供。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        # RoPE (shared implementation)
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
            base=base,
        )

    def forward_parallel_ka(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        並列KA計算（学習用）

        全トークンでAを使用する方式（Vは使わない）:
          A[0] = V[0]                   (最初だけV、ブートストラップ)
          A[i] = weights[:i] @ A[:i]    (過去のAのみ使用、現在のVは不使用)

        仮説: Attention OutputはResidual Connectionで加算されるため、
        現在位置のVを直接使用する必要はない。
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        query, key = self.rope(query, key, position_ids)

        # Attention weights (full causal)
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        # Causal mask: 現在位置も見えないようにする (diagonal=0 ではなく diagonal=1 のまま)
        # ただし、過去位置のみを使うため、weightsを再正規化
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # KA方式（全てA統一、現在位置のVは不使用）
        # A[0] = V[0]  (ブートストラップ: 最初のトークンはVで初期化)
        # A[i] = renormalize(weights[:i]) @ A[:i]  (過去のAのみ)

        attn_output = torch.zeros_like(value)

        for i in range(seq_len):
            if i == 0:
                # 最初のトークン: Vで初期化（他に参照できるAがない）
                attn_output[:, :, i, :] = value[:, :, i, :]
            else:
                # 過去のAのみ使用（現在位置のVは使わない）
                # weights[:i] を再正規化
                past_weights = attn_weights[:, :, i, :i]  # [batch, heads, i]
                past_weights_sum = past_weights.sum(dim=-1, keepdim=True)
                past_weights_normalized = past_weights / (past_weights_sum + 1e-8)

                # A[i] = normalized_weights @ A[:i]
                # past_weights_normalized: [batch, heads, i] -> [batch, heads, 1, i]
                # attn_output[:, :, :i, :]: [batch, heads, i, head_dim]
                # result: [batch, heads, 1, head_dim] -> squeeze -> [batch, heads, head_dim]
                past_a = attn_output[:, :, :i, :]  # [batch, heads, i, head_dim]
                weights_expanded = past_weights_normalized.unsqueeze(2)  # [batch, heads, 1, i]
                attn_output[:, :, i, :] = (weights_expanded @ past_a).squeeze(2)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.dense(attn_output)

        return output

    def forward_with_ka_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        ka_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        KAキャッシュを使用したforward（推論用）

        全てAを使用する方式（現在のVは不使用）:
          A[i] = renormalize(weights[:cached_len]) @ cached_A
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        query, key = self.rope(query, key, position_ids)

        # Handle KA cache
        if ka_cache is not None:
            cached_k, cached_a = ka_cache
            full_key = torch.cat([cached_k, key], dim=2)
            cached_len = cached_k.shape[2]
        else:
            full_key = key
            cached_a = None
            cached_len = 0

        total_len = full_key.shape[2]

        # Attention scores
        attn_weights = torch.matmul(query, full_key.transpose(-1, -2)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, total_len, device=hidden_states.device) * float("-inf"),
            diagonal=total_len - seq_len + 1
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # KA-Attention（全てA、現在のVは不使用）
        if cached_a is not None and cached_len > 0:
            # 過去のAのみ使用（現在位置のVは使わない）
            weights_for_cached = attn_weights[..., :cached_len]

            # 再正規化
            weights_sum = weights_for_cached.sum(dim=-1, keepdim=True)
            weights_normalized = weights_for_cached / (weights_sum + 1e-8)

            attn_output = torch.matmul(weights_normalized, cached_a)
        else:
            # 最初のトークン: Vを使用（ブートストラップ）
            attn_output = torch.matmul(attn_weights, value)

        # Update KA cache
        new_a = attn_output
        if cached_a is not None:
            new_cached_a = torch.cat([cached_a, new_a], dim=2)
        else:
            new_cached_a = new_a

        new_ka_cache = (full_key, new_cached_a)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.dense(attn_output)

        return output, new_ka_cache

    def forward_with_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """標準KVキャッシュforward（比較用）"""
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query, key = self.rope(query, key, position_ids)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            key = torch.cat([cached_k, key], dim=2)
            value = torch.cat([cached_v, value], dim=2)

        new_kv_cache = (key, value)

        total_len = key.shape[2]
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        causal_mask = torch.triu(
            torch.ones(seq_len, total_len, device=hidden_states.device) * float("-inf"),
            diagonal=total_len - seq_len + 1
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.dense(attn_output)

        return output, new_kv_cache


class KATrainPythiaModel(nn.Module):
    """
    学習時からKAキャッシュを使用するPythiaモデル (案2)

    学習と推論で同じKA方式を使用することで、
    train-inference mismatchを解消。
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        rotary_pct: float = 0.25,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Attention layers
        self.attentions = nn.ModuleList([
            KATrainAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_layers)
        ])

        # MLPs
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size),
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.input_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """学習用forward（KA方式で並列計算）"""
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_in(input_ids)

        for i in range(self.num_layers):
            residual = hidden_states
            hidden_states = self.input_layernorms[i](hidden_states)

            # KA方式のAttention
            attn_output = self.attentions[i].forward_parallel_ka(hidden_states, position_ids)
            mlp_output = self.mlps[i](hidden_states)

            hidden_states = residual + attn_output + mlp_output

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: Optional[List] = None,
        use_ka_cache: bool = True,
    ) -> Tuple[torch.Tensor, List]:
        """推論用forward（KAまたはKVキャッシュ）"""
        batch_size, seq_len = input_ids.shape

        if cache is None:
            cache = [None] * self.num_layers
            past_len = 0
        else:
            past_len = cache[0][0].shape[2] if cache[0] is not None else 0

        position_ids = torch.arange(
            past_len, past_len + seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_in(input_ids)
        new_cache = []

        for i in range(self.num_layers):
            residual = hidden_states
            hidden_states = self.input_layernorms[i](hidden_states)

            if use_ka_cache:
                attn_output, layer_cache = self.attentions[i].forward_with_ka_cache(
                    hidden_states, position_ids, cache[i]
                )
            else:
                attn_output, layer_cache = self.attentions[i].forward_with_kv_cache(
                    hidden_states, position_ids, cache[i]
                )

            mlp_output = self.mlps[i](hidden_states)
            hidden_states = residual + attn_output + mlp_output

            new_cache.append(layer_cache)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        use_ka_cache: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """テキスト生成"""
        cache = None

        for _ in range(max_new_tokens):
            if cache is None:
                logits, cache = self.forward_with_cache(input_ids, None, use_ka_cache)
            else:
                logits, cache = self.forward_with_cache(
                    input_ids[:, -1:], cache, use_ka_cache
                )

            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def num_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}
