"""
Selective Output Language Model

仮説: LLMは即座に出力せず、隠れ状態を追加処理してから出力すべき

動作:
- 従来のContinuous: トークン入力 → 即座に次トークン予測
- Selective: トークン入力 → 隠れ状態を追加処理 → 次トークン予測

skip_interval (追加処理回数):
- 0: 追加処理なし = 従来のContinuousと同等
- 1: 1回追加処理後に出力（入力→処理→追加処理→出力）
- 2: 2回追加処理後に出力

例 (skip_interval=1, 入力: "A B C D"):
  Step 1: 入力A → 隠れ状態h1（まだ出力しない）
  Step 2: h1を追加処理 → 隠れ状態h2 → "B"を予測・出力
  Step 3: 入力B → 隠れ状態h3（まだ出力しない）
  Step 4: h3を追加処理 → 隠れ状態h4 → "C"を予測・出力
  ...
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import init_weights
from src.models.layers import BaseLayer


class SelectiveOutputLM(nn.Module):
    """
    Selective Output Language Model

    トークン入力後、skip_interval回の追加処理を経てから出力する。
    skip_interval=0: 追加処理なし（Continuousと同等）
    skip_interval=1: 1回追加処理後に出力
    skip_interval=2: 2回追加処理後に出力
    """

    def __init__(
        self,
        layers: list[BaseLayer],
        vocab_size: int = 50304,
        hidden_size: int = 512,
        skip_interval: int = 1,  # 追加処理回数（0=Continuous、1=1回追加処理）
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers)
        self.skip_interval = max(0, skip_interval)  # 0以上

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

    def _process_single_step(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """1ステップの処理（Transformer layers通過）"""
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                update_memory=update_memory,
            )
        return hidden_states

    def forward_selective(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Selective forward pass - 各トークンをskip_interval回処理してから出力

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            update_memory: メモリ更新フラグ

        Returns:
            logits: [batch, num_outputs, vocab_size] - 出力位置のlogitsのみ
            final_hidden: [batch, hidden_size] - 最終隠れ表現
            stats: 統計情報
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        all_output_logits = []
        total_process_steps = 0

        # 各入力トークンに対して処理
        for t in range(seq_len):
            # Step 1: トークンを埋め込み
            token_embed = self.embed_in(input_ids[:, t:t+1])  # [batch, 1, hidden]

            # 初回処理（トークン埋め込み → Transformer）
            hidden = self._process_single_step(token_embed, None, update_memory)
            total_process_steps += 1

            # skip_interval回の追加処理
            for _ in range(self.skip_interval):
                # 隠れ状態を投影して再処理
                hidden = self.hidden_proj(hidden)
                hidden = self._process_single_step(hidden, None, update_memory)
                total_process_steps += 1

            # 出力
            normed = self.final_layer_norm(hidden)
            logits = self.embed_out(normed)  # [batch, 1, vocab]
            all_output_logits.append(logits)

        # 全出力を結合
        output_logits = torch.cat(all_output_logits, dim=1)  # [batch, seq_len, vocab]
        final_hidden = hidden[:, -1, :]

        stats = {
            "num_outputs": seq_len,
            "total_process_steps": total_process_steps,
            "avg_steps_per_output": total_process_steps / seq_len if seq_len > 0 else 0,
        }

        return output_logits, final_hidden, stats

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_selective: bool = True,
        update_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            use_selective: Selectiveモードを使用するか（False=通常のforward）
            update_memory: メモリ更新フラグ

        Returns:
            logits: [batch, seq_len, vocab_size]
            final_hidden: [batch, hidden_size]
        """
        if use_selective and self.skip_interval > 0:
            logits, final_hidden, _ = self.forward_selective(
                input_ids, attention_mask, update_memory
            )
            return logits, final_hidden

        # 通常のforward（skip_interval=0 または use_selective=False）
        hidden_states = self.embed_in(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                update_memory=update_memory,
            )

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

        Selective mode:
        - forward_selectiveで各トークンをskip_interval回処理
        - 各出力で次トークンを予測

        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] - input_idsと同じ（内部でshift）

        Returns:
            loss: スカラー損失
            stats: 訓練統計
        """
        if use_selective and self.skip_interval > 0:
            logits, _, proc_stats = self.forward_selective(input_ids)
        else:
            logits, _ = self.forward(input_ids, use_selective=False)
            proc_stats = {"num_outputs": input_ids.size(1), "total_process_steps": input_ids.size(1)}

        # Next-token prediction: logits[:-1] -> labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        stats = {
            "lm_loss": loss.item(),
            "ppl": torch.exp(loss).item(),
            **proc_stats,
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
            use_selective: Selectiveモードを使用するか

        Returns:
            generated_ids: [batch, output_len]
            stats: 統計情報
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        generated = [input_ids]
        total_process_steps = 0

        # 初期コンテキスト処理
        with torch.no_grad():
            if use_selective and self.skip_interval > 0:
                logits, prev_hidden, init_stats = self.forward_selective(input_ids)
                total_process_steps += init_stats["total_process_steps"]
            else:
                logits, prev_hidden = self.forward(input_ids, use_selective=False)
                total_process_steps += input_ids.size(1)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                if use_selective and self.skip_interval > 0:
                    # 最後に生成したトークンをSelectiveモードで処理
                    last_token = generated[-1][:, -1:]
                    logits, prev_hidden, step_stats = self.forward_selective(last_token)
                    total_process_steps += step_stats["total_process_steps"]
                else:
                    last_token = generated[-1][:, -1:]
                    logits, prev_hidden = self.forward(last_token, use_selective=False)
                    total_process_steps += 1

                # サンプリング
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated.append(next_token)

        stats = {
            "num_tokens_generated": max_new_tokens,
            "total_process_steps": total_process_steps,
            "avg_steps_per_token": total_process_steps / (input_ids.size(1) + max_new_tokens),
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
