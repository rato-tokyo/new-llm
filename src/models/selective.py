"""
Selective Output Language Model (Simplified)

仮説: LLMの全出力がトークン化されるべきではない
- 固定パターンで持ち越し（例: 2回に1回）
- 持ち越し位置では損失なし、出力位置でのみ損失計算

簡素化版:
- 学習可能ゲートを削除
- 固定スキップパターン（skip_interval回に1回出力）
- 例: skip_interval=2 → 出力、スキップ、出力、スキップ...
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base_components import init_weights
from src.models.layers import BaseLayer


class SelectiveOutputLM(nn.Module):
    """
    Selective Output Language Model (Simplified)

    固定パターンで持ち越しを行う。
    skip_interval=2: 2位置ごとに1回出力（50%持ち越し）
    skip_interval=3: 3位置ごとに1回出力（66%持ち越し）

    訓練時:
    - 出力位置のみで損失計算
    - 持ち越し位置は損失なし

    推論時:
    - 同じ固定パターンで出力
    """

    def __init__(
        self,
        layers: list[BaseLayer],
        vocab_size: int = 50304,
        hidden_size: int = 512,
        skip_interval: int = 2,  # N回に1回出力（デフォルト: 2回に1回=50%出力）
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers)
        self.skip_interval = skip_interval

        # Token Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Hidden → Hidden projection (スキップ時に使用)
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
        use_selective: bool = False,
        prev_hidden: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            use_selective: 選択的出力モードを使用するか
            prev_hidden: [batch, hidden_size] - 前ステップの隠れ表現（スキップ時）
            update_memory: メモリ更新フラグ

        Returns:
            logits: [batch, seq_len, vocab_size]
            final_hidden: [batch, hidden_size] - 最終隠れ表現
        """
        if use_selective and prev_hidden is not None:
            # 選択的モード: 前の隠れ表現を最初の入力に使用
            batch_size, seq_len = input_ids.shape
            first_hidden = self.hidden_proj(prev_hidden).unsqueeze(1)

            if seq_len > 1:
                rest_hidden = self.embed_in(input_ids[:, 1:])
                hidden_states = torch.cat([first_hidden, rest_hidden], dim=1)
            else:
                hidden_states = first_hidden
        else:
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

        # Final hidden for next step
        final_hidden = hidden_states[:, -1, :]

        return logits, final_hidden

    def compute_fixed_skip_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        固定スキップパターンによる損失計算

        skip_interval=2の場合:
        - pos 0: 出力 → logits[0]とlabels[1]比較
        - pos 1: スキップ → 損失なし（labels[2]を持ち越し）
        - pos 2: 出力 → logits[2]とlabels[2]比較（持ち越されたターゲット）
        - pos 3: スキップ → 損失なし（labels[3]を持ち越し）
        ...

        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]

        Returns:
            loss: スカラー損失
            stats: 訓練統計
        """
        batch_size, seq_len = labels.shape

        # Next-token prediction の shift
        shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab]

        # 固定出力マスク: skip_interval回に1回出力
        # pos 0, skip_interval, 2*skip_interval, ... で出力
        output_mask = torch.zeros(seq_len - 1, device=logits.device)
        for i in range(0, seq_len - 1, self.skip_interval):
            output_mask[i] = 1.0
        output_mask = output_mask.unsqueeze(0).expand(batch_size, -1)  # [batch, seq_len-1]

        # 持ち越しターゲットの計算
        # 累積出力数（exclusive）
        cumsum_before = torch.zeros_like(output_mask)
        cumsum_before[:, 1:] = output_mask[:, :-1].cumsum(dim=1)

        # ターゲットインデックス = 1 + cumsum_before
        target_indices = (1 + cumsum_before.long()).clamp(0, seq_len - 1)

        # バッチごとにターゲットを gather
        carryover_targets = torch.gather(labels, 1, target_indices)  # [batch, seq_len-1]

        # Per-token cross entropy
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            carryover_targets.view(-1)
        ).view(batch_size, seq_len - 1)

        # 出力位置のみで損失を計算
        masked_loss = per_token_loss * output_mask
        output_count = output_mask.sum()

        if output_count > 0:
            loss = masked_loss.sum() / output_count
        else:
            loss = per_token_loss.mean()

        # 統計情報
        with torch.no_grad():
            output_ratio = output_count.item() / output_mask.numel()
            carryover_ratio = 1.0 - output_ratio

        stats = {
            "lm_loss": loss.item(),
            "output_ratio": output_ratio,
            "carryover_ratio": carryover_ratio,
            "output_count": output_count.item(),
        }

        return loss, stats

    def generate_selective(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        固定スキップパターンでテキスト生成

        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: 最大生成トークン数
            temperature: サンプリング温度

        Returns:
            generated_ids: [batch, output_len]
            stats: 統計情報
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        # 初期処理
        with torch.no_grad():
            logits, prev_hidden = self.forward(input_ids, use_selective=False)

        generated = [input_ids]
        total_steps = 0
        skip_count = 0
        output_count = 0

        while output_count < max_new_tokens:
            total_steps += 1

            if total_steps % self.skip_interval == 1:
                # 出力
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated.append(next_token)
                output_count += 1

                # 次のステップ（トークン埋め込みを使用）
                with torch.no_grad():
                    logits, prev_hidden = self.forward(
                        next_token,
                        use_selective=False,
                    )
            else:
                # スキップ（hiddenを継続）
                skip_count += 1

                # ダミートークン
                dummy_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

                with torch.no_grad():
                    logits, prev_hidden = self.forward(
                        dummy_token,
                        use_selective=True,
                        prev_hidden=prev_hidden,
                    )

        stats = {
            "total_steps": total_steps,
            "output_count": output_count,
            "skip_count": skip_count,
            "skip_ratio": skip_count / total_steps if total_steps > 0 else 0,
        }

        return torch.cat(generated, dim=1), stats

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
