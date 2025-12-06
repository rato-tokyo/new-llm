"""
Selective Output Language Model

仮説: LLMの全出力がトークン化されるべきではない
- 高確信度の出力 → トークン化して外部に出力
- 低確信度の出力 → 内部処理として継続（スキップ）

人間の「えーと」「うーん」のような思考時間に相当。
内部で推論を続け、確信が持てたときだけ出力する。

学習可能ゲート:
- hidden state から「出力すべきか」を判断
- ゲートが開いている → トークン出力
- ゲートが閉じている → hidden をそのまま次へ
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import init_weights
from src.models.layers import BaseLayer


class OutputGate(nn.Module):
    """
    学習可能な出力ゲート

    hidden state を入力として、出力すべきかどうかを判断。

    判断基準（学習で最適化）:
    - hidden の情報量
    - 予測の確信度
    - 文脈の完結度
    """

    def __init__(self, hidden_size: int):
        super().__init__()

        # Multi-layer gate for complex decision
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [batch, seq_len, hidden_size] or [batch, hidden_size]

        Returns:
            gate_prob: [batch, seq_len, 1] or [batch, 1] - 出力確率
        """
        return torch.sigmoid(self.gate_net(hidden))


class SelectiveOutputLM(nn.Module):
    """
    Selective Output Language Model

    通常のLMと異なり、各ステップで「出力すべきか」を判断。

    訓練時:
    - 全ステップでlogitsを計算
    - ゲートの出力確率も計算
    - ゲート損失: 確信度の高い予測で出力、低い予測でスキップを学習

    推論時:
    - ゲートが開く（> threshold）まで内部処理を継続
    - 最大スキップ数に達したら強制出力
    """

    def __init__(
        self,
        layers: list[BaseLayer],
        vocab_size: int = 50304,
        hidden_size: int = 512,
        max_skip: int = 3,  # 連続スキップの最大数
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers)
        self.max_skip = max_skip

        # Token Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Hidden → Hidden projection (スキップ時に使用)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)

        # Output gate (学習可能)
        self.output_gate = OutputGate(hidden_size)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            gate_probs: [batch, seq_len, 1] - 各位置の出力確率
            final_hidden: [batch, hidden_size] - 最終隠れ表現
        """
        batch_size, seq_len = input_ids.shape

        if use_selective and prev_hidden is not None:
            # 選択的モード: 前の隠れ表現を最初の入力に使用
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

        # Output gate
        gate_probs = self.output_gate(hidden_states)

        # Final hidden for next step
        final_hidden = hidden_states[:, -1, :]

        return logits, gate_probs, final_hidden

    def compute_gate_loss(
        self,
        logits: torch.Tensor,
        gate_probs: torch.Tensor,
        labels: torch.Tensor,
        gate_target_mode: str = "entropy",
    ) -> torch.Tensor:
        """
        ゲートの学習用損失を計算

        目標: 確信度が高いときに出力、低いときにスキップを学習

        Args:
            logits: [batch, seq_len, vocab_size]
            gate_probs: [batch, seq_len, 1]
            labels: [batch, seq_len]
            gate_target_mode: "entropy" or "correctness"

        Returns:
            gate_loss: スカラー
        """
        # logitsからエントロピーを計算
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]

        # エントロピーを[0, 1]に正規化
        # 低エントロピー（高確信度）→ 1に近い、高エントロピー（低確信度）→ 0に近い
        max_entropy = torch.log(torch.tensor(self.vocab_size, dtype=logits.dtype, device=logits.device))
        normalized_entropy = entropy / max_entropy

        # ゲートのターゲット: 低エントロピーなら出力(1)、高エントロピーならスキップ(0)
        gate_target = 1.0 - normalized_entropy  # [batch, seq_len]

        # Binary cross entropy
        gate_probs_squeezed = gate_probs.squeeze(-1)  # [batch, seq_len]
        gate_loss = F.binary_cross_entropy(gate_probs_squeezed, gate_target.detach())

        return gate_loss

    def compute_selective_loss(
        self,
        logits: torch.Tensor,
        gate_probs: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5,
        gate_loss_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict]:
        """
        訓練-評価一貫性を保った損失計算

        訓練時と評価時で同じthresholdを使用し、出力位置のみでLM lossを計算。
        これにより訓練と評価の条件が揃う。

        Args:
            logits: [batch, seq_len, vocab_size]
            gate_probs: [batch, seq_len, 1]
            labels: [batch, seq_len]
            threshold: 出力閾値（訓練・評価で同じ値を使用）
            gate_loss_weight: ゲート損失の重み

        Returns:
            total_loss: LM loss + gate_loss_weight * gate_loss
            stats: 訓練統計
        """
        batch_size, seq_len = labels.shape
        device = logits.device

        # 出力マスク: gate_prob > threshold の位置のみ出力
        gate_probs_squeezed = gate_probs.squeeze(-1)  # [batch, seq_len]
        output_mask = (gate_probs_squeezed > threshold).float()  # [batch, seq_len]

        # LM Loss計算（出力位置のみ）
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab]
        shift_labels = labels[:, 1:].contiguous()  # [batch, seq_len-1]
        shift_mask = output_mask[:, :-1].contiguous()  # [batch, seq_len-1]

        # Per-token cross entropy
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        ).view(batch_size, seq_len - 1)  # [batch, seq_len-1]

        # 出力位置のlossのみ平均
        masked_loss = per_token_loss * shift_mask
        num_output_tokens = shift_mask.sum()

        if num_output_tokens > 0:
            lm_loss = masked_loss.sum() / num_output_tokens
        else:
            # 出力トークンがない場合、全体の平均を使用（まれなケース）
            lm_loss = per_token_loss.mean()

        # Gate Loss（エントロピーベース）
        gate_loss = self.compute_gate_loss(logits, gate_probs, labels)

        # Total Loss
        total_loss = lm_loss + gate_loss_weight * gate_loss

        # 統計情報
        with torch.no_grad():
            output_ratio = output_mask.mean().item()
            skip_ratio = 1.0 - output_ratio
            avg_gate_prob = gate_probs_squeezed.mean().item()

        stats = {
            "lm_loss": lm_loss.item(),
            "gate_loss": gate_loss.item(),
            "total_loss": total_loss.item(),
            "output_ratio": output_ratio,
            "skip_ratio": skip_ratio,
            "avg_gate_prob": avg_gate_prob,
            "num_output_tokens": num_output_tokens.item(),
        }

        return total_loss, stats

    def generate_selective(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, dict]:
        """
        選択的出力モードでテキスト生成

        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: 最大生成トークン数
            temperature: サンプリング温度
            threshold: 出力閾値

        Returns:
            generated_ids: [batch, output_len]
            stats: 統計情報（スキップ数など）
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        # 初期処理
        with torch.no_grad():
            logits, gate_probs, prev_hidden = self.forward(input_ids, use_selective=False)

        generated = [input_ids]
        total_steps = 0
        skip_count = 0
        output_count = 0
        consecutive_skips = 0

        while output_count < max_new_tokens and total_steps < max_new_tokens * (self.max_skip + 1):
            total_steps += 1

            # 最後の位置のゲート確率（バッチ平均）
            gate_prob = gate_probs[:, -1, 0].mean().item()

            if gate_prob > threshold or consecutive_skips >= self.max_skip:
                # 出力
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated.append(next_token)
                output_count += 1
                consecutive_skips = 0

                # 次のステップ（トークン埋め込みを使用）
                with torch.no_grad():
                    logits, gate_probs, prev_hidden = self.forward(
                        next_token,
                        use_selective=False,
                    )
            else:
                # スキップ（hiddenを継続）
                skip_count += 1
                consecutive_skips += 1

                # ダミートークン（使わないが形式上必要）
                dummy_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

                with torch.no_grad():
                    logits, gate_probs, prev_hidden = self.forward(
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
        gate = sum(p.numel() for p in self.output_gate.parameters())

        layer_params = [
            sum(p.numel() for p in layer.parameters())
            for layer in self.layers
        ]

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "hidden_proj": hidden_proj,
            "output_gate": gate,
            "transformer": sum(layer_params),
            "per_layer": layer_params,
        }
