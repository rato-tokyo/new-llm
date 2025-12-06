"""
Continuous Representation Language Model

仮説: 既存LLMはtoken_idへの離散化で情報を損失している
提案: 前ステップの連続的な隠れ表現を次の入力として使用

既存LLM:
  hidden → logits → argmax → token_id → embed(token_id) → 次の入力
  (情報がtoken_idに圧縮 = 情報損失)

ContinuousLM:
  hidden → logits → 出力（表示用）
         → hidden自体を次の入力として使用
  (連続表現がそのまま伝播 = 情報保持)
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base_components import init_weights
from src.models.layers import BaseLayer


class ContinuousLM(nn.Module):
    """
    Continuous Representation Language Model

    通常のLMと異なり、推論時に前ステップの最終隠れ表現を
    次ステップの入力として使用できる。

    訓練時:
      - 通常のTeacher Forcing (token embeddingを使用)
      - または連続表現モードで訓練

    推論時:
      - use_continuous=True で連続表現モードを有効化
      - 最初のトークンのみembeddingを使用
      - 以降は前ステップの隠れ表現を使用
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

        # Token Embedding (最初のトークン用)
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Hidden → Hidden projection (連続表現を次の入力に変換)
        # 最終LayerNorm後の表現を、LayerNorm前の空間に戻す
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(layers)

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        # LM Head
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self.apply(init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_continuous: bool = False,
        prev_hidden: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len] - トークンID
            attention_mask: Optional attention mask
            use_continuous: 連続表現モードを使用するか
            prev_hidden: [batch, hidden_size] - 前ステップの最終隠れ表現
            update_memory: メモリを更新するか（メモリ系レイヤーのみ）

        Returns:
            logits: [batch, seq_len, vocab_size]
            final_hidden: [batch, hidden_size] - 最後のトークンの隠れ表現
        """
        batch_size, seq_len = input_ids.shape

        if use_continuous and prev_hidden is not None:
            # 連続表現モード: 前の隠れ表現を変換して最初の入力に使用
            # prev_hidden: [batch, hidden_size] → [batch, 1, hidden_size]
            first_hidden = self.hidden_proj(prev_hidden).unsqueeze(1)

            if seq_len > 1:
                # 残りはtoken embedding
                rest_hidden = self.embed_in(input_ids[:, 1:])
                hidden_states = torch.cat([first_hidden, rest_hidden], dim=1)
            else:
                hidden_states = first_hidden
        else:
            # 通常モード: token embeddingを使用
            hidden_states = self.embed_in(input_ids)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                update_memory=update_memory,
            )

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # LM Head
        logits = self.embed_out(hidden_states)

        # 最後のトークンの隠れ表現を返す（次ステップの入力用）
        final_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]

        return logits, final_hidden

    def generate_continuous(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        連続表現モードでテキスト生成

        Args:
            input_ids: [batch, seq_len] - 初期トークン
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度

        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 初期トークンを処理
        with torch.no_grad():
            logits, prev_hidden = self.forward(input_ids, use_continuous=False)

        generated = [input_ids]

        for _ in range(max_new_tokens):
            # 最後のトークンのlogitsからサンプリング
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)

            # 連続表現モードで次を予測
            with torch.no_grad():
                logits, prev_hidden = self.forward(
                    next_token,
                    use_continuous=True,
                    prev_hidden=prev_hidden,
                )

        return torch.cat(generated, dim=1)

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
