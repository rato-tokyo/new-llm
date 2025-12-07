"""
Selective Output Language Model

仮説: LLMは即座に出力せず、隠れ状態を追加処理してから出力すべき

動作:
- extra_passes=0 (use_selective=False): トークン入力 → 即座に次トークン予測
- extra_passes=1 (use_selective=True): トークン入力 → 1回追加処理 → 次トークン予測

例 (入力: "A B C D"):
  extra_passes=0: embed(A) → layers → 即座に"B"予測
  extra_passes=1: embed(A) → layers → proj → layers → "B"予測
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import init_weights
from src.models.layers import BaseLayer


class SelectiveOutputLM(nn.Module):
    """
    Selective Output Language Model (効率化版)

    extra_passes=0 (use_selective=False): 通常のContinuous（追加処理なし）
    extra_passes=1 (use_selective=True): 1回追加処理してから出力

    バッチ処理で高速化: forループなしで全トークンを一括処理
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

        # Token Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Hidden → Hidden projection (追加処理時に使用)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)

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
        use_selective: bool = True,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (バッチ処理で高速)

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            use_selective: True=extra_passes=1、False=extra_passes=0
            update_memory: メモリ更新フラグ

        Returns:
            logits: [batch, seq_len, vocab_size]
            final_hidden: [batch, hidden_size]
        """
        # Pass 1: トークン埋め込み → Transformer
        hidden_states = self.embed_in(input_ids)
        hidden_states = self._forward_layers(hidden_states, attention_mask, update_memory)

        if use_selective:
            # Pass 2: 隠れ状態を投影 → 再度Transformer
            hidden_states = self.hidden_proj(hidden_states)
            hidden_states = self._forward_layers(hidden_states, attention_mask, update_memory)

        # Final layer norm + LM Head
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        final_hidden = hidden_states[:, -1, :]
        return logits, final_hidden

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        use_selective: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        損失計算

        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] - input_idsと同じ（内部でshift）
            use_selective: True=extra_passes=1、False=extra_passes=0

        Returns:
            loss: スカラー損失
            stats: 訓練統計
        """
        logits, _ = self.forward(input_ids, use_selective=use_selective)

        # Next-token prediction: logits[:-1] -> labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        extra_passes = 1 if use_selective else 0
        stats = {
            "lm_loss": loss.item(),
            "ppl": torch.exp(loss).item(),
            "extra_passes": extra_passes,
        }

        return loss, stats

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        use_selective: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        テキスト生成

        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: 最大生成トークン数
            temperature: サンプリング温度
            use_selective: True=extra_passes=1、False=extra_passes=0

        Returns:
            generated_ids: [batch, output_len]
            stats: 統計情報
        """
        self.eval()

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.forward(generated, use_selective=use_selective)

                # サンプリング
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

        extra_passes = 1 if use_selective else 0
        stats = {
            "num_tokens_generated": max_new_tokens,
            "extra_passes": extra_passes,
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

        layer_params = [
            sum(p.numel() for p in layer.parameters())
            for layer in self.layers
        ]

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "hidden_proj": hidden_proj,
            "transformer": sum(layer_params),
            "per_layer": layer_params,
        }
