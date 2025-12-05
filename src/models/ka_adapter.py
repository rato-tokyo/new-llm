"""
KA Cache with Adapter

Adapter方式のKAキャッシュ実装。
既存モデルの重みを凍結し、A→V変換用のAdapterのみを学習する。

方式:
  1. 既存モデルでK, Q, Vを計算（凍結）
  2. Attention Output (A) を計算
  3. 推論時: 過去のAをAdapterで変換してVの代わりに使用
     A[i] = weights @ [Adapter(A[1:i-1]), V[i]]

Adapterアーキテクチャ:
  - Bottleneck構造: dim → bottleneck → dim
  - 残差接続: output = A + Adapter(A)
  - 小パラメータ（bottleneck=64がデフォルト）
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KAAdapter(nn.Module):
    """
    A→V変換用のAdapter

    Bottleneck構造で少パラメータでAをVに近づける。
    """

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.layer_norm = nn.LayerNorm(dim)

        # 初期化: 小さい値で開始（残差接続のため）
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.zeros_(self.up.bias)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        A (Attention Output) をV-likeな表現に変換

        Args:
            a: [batch, heads, seq, head_dim]

        Returns:
            transformed: [batch, heads, seq, head_dim]
        """
        # Bottleneck変換
        delta = self.up(F.gelu(self.down(a)))
        # 残差接続 + LayerNorm
        return self.layer_norm(a + delta)


class KAAdapterAttention(nn.Module):
    """
    Adapter付きKAキャッシュAttention

    既存のAttention重みは凍結し、Adapterのみ学習可能。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        adapter_bottleneck: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = int(self.head_dim * rotary_pct)
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections (will be frozen)
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        # Adapter (trainable)
        self.adapter = KAAdapter(self.head_dim, adapter_bottleneck)

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._init_rope_cache(max_position_embeddings)

    def _init_rope_cache(self, max_len: int) -> None:
        t = torch.arange(max_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        self.max_seq_len_cached = max_len

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key"""
        seq_len = int(position_ids.max().item()) + 1
        if seq_len > self.max_seq_len_cached:
            self._init_rope_cache(seq_len)

        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)

        q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
        k_rot, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        k_rot = k_rot * cos + rotate_half(k_rot) * sin

        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    def forward_with_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Standard KV cache forward (for baseline comparison)"""
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query, key = self._apply_rotary(query, key, position_ids)

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

    def forward_with_ka_adapter(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        ka_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """KA cache with Adapter forward"""
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query, key = self._apply_rotary(query, key, position_ids)

        if ka_cache is not None:
            cached_k, cached_a = ka_cache
            full_key = torch.cat([cached_k, key], dim=2)
            cached_len = cached_k.shape[2]
        else:
            full_key = key
            cached_a = None
            cached_len = 0

        total_len = full_key.shape[2]

        attn_weights = torch.matmul(query, full_key.transpose(-1, -2)) * self.scale

        causal_mask = torch.triu(
            torch.ones(seq_len, total_len, device=hidden_states.device) * float("-inf"),
            diagonal=total_len - seq_len + 1
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # KA-Attention with Adapter
        if cached_a is not None:
            weights_for_cached = attn_weights[..., :cached_len]
            weights_for_current = attn_weights[..., cached_len:]

            # Apply adapter to cached A
            adapted_a = self.adapter(cached_a)

            # Compute attention output
            cached_contribution = torch.matmul(weights_for_cached, adapted_a)
            current_contribution = torch.matmul(weights_for_current, value)

            attn_output = cached_contribution + current_contribution
        else:
            attn_output = torch.matmul(attn_weights, value)

        # Update KA cache
        new_a = attn_output
        if cached_a is not None:
            new_cached_a = torch.cat([cached_a, new_a], dim=2)
        else:
            new_cached_a = new_a

        new_ka_cache = (full_key, new_cached_a)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.dense(attn_output)

        return output, new_ka_cache


class KAAdapterPythiaModel(nn.Module):
    """
    Adapter付きKAキャッシュPythiaモデル

    既存モデルの重みを凍結し、Adapterのみ学習可能。
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
        adapter_bottleneck: int = 64,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.adapter_bottleneck = adapter_bottleneck

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Attention layers with adapters
        self.attentions = nn.ModuleList([
            KAAdapterAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
                adapter_bottleneck=adapter_bottleneck,
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

    def freeze_base_model(self) -> None:
        """Freeze all parameters except adapters"""
        for name, param in self.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True

    def get_adapter_parameters(self) -> List[nn.Parameter]:
        """Get only adapter parameters (for optimizer)"""
        return [p for n, p in self.named_parameters() if "adapter" in n]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard forward (no cache)"""
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_in(input_ids)

        for i in range(self.num_layers):
            residual = hidden_states
            hidden_states = self.input_layernorms[i](hidden_states)

            attn_output, _ = self.attentions[i].forward_with_kv_cache(hidden_states, position_ids, None)
            mlp_output = self.mlps[i](hidden_states)

            hidden_states = residual + attn_output + mlp_output

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: Optional[List] = None,
        use_ka_adapter: bool = False,
    ) -> Tuple[torch.Tensor, List]:
        """Forward with cache (KV or KA with Adapter)"""
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

            if use_ka_adapter:
                attn_output, layer_cache = self.attentions[i].forward_with_ka_adapter(
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
        use_ka_adapter: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens with KV or KA adapter cache"""
        cache = None

        for _ in range(max_new_tokens):
            if cache is None:
                logits, cache = self.forward_with_cache(input_ids, None, use_ka_adapter)
            else:
                logits, cache = self.forward_with_cache(
                    input_ids[:, -1:], cache, use_ka_adapter
                )

            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def forward_adapter_training(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        並列Adapter学習用forward

        通常の並列forward + Adapter損失を返す。
        Adapter(A) が V に近づくように学習。

        Returns:
            logits: 通常のlogits
            adapter_loss: Adapter(A) と V の MSE損失
        """
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_in(input_ids)

        total_adapter_loss: torch.Tensor = torch.tensor(0.0, device=input_ids.device)
        num_layers = 0

        for i in range(self.num_layers):
            residual = hidden_states
            hidden_states = self.input_layernorms[i](hidden_states)

            # 並列計算でQ, K, V, Aを全て計算
            attn = self.attentions[i]
            qkv = attn.query_key_value(hidden_states)
            qkv = qkv.view(batch_size, seq_len, 3, attn.num_heads, attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            query, key = attn._apply_rotary(query, key, position_ids)

            # Attention計算
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) * attn.scale

            # 因果マスク
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Attention Output (A)
            attn_output = torch.matmul(attn_weights, value)  # [batch, heads, seq, head_dim]

            # Adapter損失: Adapter(A) が V に近づくように
            # ただし、最初のトークンはキャッシュがないので除外
            if seq_len > 1:
                adapted_a = attn.adapter(attn_output[:, :, :-1, :])  # 過去のA
                target_v = value[:, :, 1:, :]  # 次のステップのV
                layer_loss = F.mse_loss(adapted_a, target_v.detach())
                total_adapter_loss += layer_loss
                num_layers += 1

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
            attn_output = attn.dense(attn_output)

            mlp_output = self.mlps[i](hidden_states)
            hidden_states = residual + attn_output + mlp_output

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        adapter_loss = total_adapter_loss / max(num_layers, 1)

        return logits, adapter_loss

    def num_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        adapter = sum(p.numel() for n, p in self.named_parameters() if "adapter" in n)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "adapter": adapter,
            "trainable": trainable,
        }
