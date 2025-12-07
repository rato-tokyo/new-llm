"""
Continuous Language Model

仮説: トークン化による離散化で情報が失われている

通常LM (Discrete):
  h_t → LM Head → token → Embedding → x_{t+1}
        ↑                    ↑
        離散化              再埋め込み（情報損失）

Continuous LM:
  h_t → proj → x_{t+1}   (離散化をスキップ、情報保持)

モード:
- discrete: 通常のLM（トークン埋め込みを入力）
- continuous: 前の隠れ状態を直接入力として使用
- continuous + extra_pass: 追加処理あり
- continuous + extra_pass + use_h1: h1とh2の両方を使用
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import init_weights
from src.models.layers import BaseLayer


class ContinuousLM(nn.Module):
    """
    Continuous Language Model

    離散化をスキップして、隠れ状態を直接次の入力として使用する。
    """

    def __init__(
        self,
        layers: list[BaseLayer],
        vocab_size: int = 50304,
        hidden_size: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers)

        # Token Embedding（discrete mode用）
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Hidden → Hidden projection（continuous mode: h_{t-1} → x_t）
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)

        # Extra pass用の投影
        self.extra_proj = nn.Linear(hidden_size, hidden_size)

        # Combined mode: h1 + h2 を結合するための投影
        self.combine_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(layers)

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        # LM Head
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self.apply(init_weights)

    def _forward_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """Transformer layers を通過"""
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                update_memory=update_memory,
            )
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "discrete",
        extra_pass: bool = False,
        use_h1: bool = False,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            mode: "discrete" or "continuous"
            extra_pass: 追加処理を行うかどうか
            use_h1: extra_pass時にh1も使用するか（combined）
            update_memory: メモリ更新フラグ

        Returns:
            logits: [batch, seq_len, vocab_size]
            final_hidden: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        if mode == "discrete":
            # 通常のLM: トークン埋め込みを入力
            hidden_states = self.embed_in(input_ids)
            h1 = self._forward_layers(hidden_states, attention_mask, update_memory)

            if extra_pass:
                h2 = self.extra_proj(h1)
                h2 = self._forward_layers(h2, attention_mask, update_memory)
                if use_h1:
                    combined = torch.cat([h1, h2], dim=-1)
                    final_hidden = self.combine_proj(combined)
                else:
                    final_hidden = h2
            else:
                final_hidden = h1

        elif mode == "continuous":
            # Continuous: 前の隠れ状態を直接入力として使用
            # 最初のトークンはembeddingを使用、以降は前の隠れ状態を使用

            # 最初のトークンの埋め込み
            first_embed = self.embed_in(input_ids[:, :1])  # [batch, 1, hidden]

            # 全シーケンスを処理
            all_hidden = []
            prev_hidden = None

            for t in range(seq_len):
                if t == 0:
                    # 最初のトークンは埋め込みを使用
                    x_t = first_embed
                else:
                    # 以降は前の隠れ状態を変換して使用
                    x_t = self.hidden_proj(prev_hidden)

                # Transformer処理（1トークンずつ）
                h_t = self._forward_layers(x_t, attention_mask=None, update_memory=update_memory)
                all_hidden.append(h_t)
                prev_hidden = h_t

            # 結合
            h1 = torch.cat(all_hidden, dim=1)  # [batch, seq_len, hidden]

            if extra_pass:
                h2 = self.extra_proj(h1)
                h2 = self._forward_layers(h2, attention_mask, update_memory)
                if use_h1:
                    combined = torch.cat([h1, h2], dim=-1)
                    final_hidden = self.combine_proj(combined)
                else:
                    final_hidden = h2
            else:
                final_hidden = h1

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Final layer norm + LM Head
        final_hidden = self.final_layer_norm(final_hidden)
        logits = self.embed_out(final_hidden)

        return logits, final_hidden

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mode: str = "discrete",
        extra_pass: bool = False,
        use_h1: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        損失計算

        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len]
            mode: "discrete" or "continuous"
            extra_pass: 追加処理を行うか
            use_h1: extra_pass時にh1も使用するか

        Returns:
            loss: スカラー損失
            stats: 訓練統計
        """
        logits, _ = self.forward(
            input_ids, mode=mode, extra_pass=extra_pass, use_h1=use_h1
        )

        # Next-token prediction: logits[:-1] -> labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        # モード名を生成
        mode_name = mode
        if extra_pass:
            mode_name += "_extra"
        if use_h1:
            mode_name += "_combined"

        stats = {
            "lm_loss": loss.item(),
            "ppl": torch.exp(loss).item(),
            "mode": mode_name,
        }

        return loss, stats

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        mode: str = "discrete",
        extra_pass: bool = False,
        use_h1: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        テキスト生成
        """
        self.eval()

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.forward(
                    generated, mode=mode, extra_pass=extra_pass, use_h1=use_h1
                )
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        mode_name = mode
        if extra_pass:
            mode_name += "_extra"
        if use_h1:
            mode_name += "_combined"

        stats = {
            "num_tokens_generated": max_new_tokens,
            "mode": mode_name,
        }

        return generated, stats

    def reset_memory(self, keep_frozen: bool = True) -> None:
        """メモリ系レイヤーのメモリをリセット"""
        device = self.embed_in.weight.device
        for layer in self.layers:
            if hasattr(layer, 'reset_memory'):
                import inspect
                sig = inspect.signature(layer.reset_memory)
                if 'keep_frozen' in sig.parameters:
                    layer.reset_memory(device, keep_frozen)
                else:
                    layer.reset_memory(device)

    def get_memory_state(self) -> dict:
        """メモリ状態を取得"""
        states = {}
        for i, layer in enumerate(self.layers):
            state = layer.get_memory_state()
            if state is not None:
                states[i] = state
        return states

    def set_memory_state(self, states: dict) -> None:
        """メモリ状態を設定"""
        device = self.embed_in.weight.device
        for i, state in states.items():
            self.layers[i].set_memory_state(state, device)

    def num_parameters(self) -> dict[str, int]:
        """パラメータ数をカウント"""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()
        hidden_proj = sum(p.numel() for p in self.hidden_proj.parameters())
        extra_proj = sum(p.numel() for p in self.extra_proj.parameters())
        combine_proj = sum(p.numel() for p in self.combine_proj.parameters())

        layer_params = [
            sum(p.numel() for p in layer.parameters())
            for layer in self.layers
        ]

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "hidden_proj": hidden_proj,
            "extra_proj": extra_proj,
            "combine_proj": combine_proj,
            "transformer": sum(layer_params),
            "per_layer": layer_params,
        }
