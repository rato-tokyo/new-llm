"""
MemoryPhase1Trainer - メモリ展開型Phase 1トレーナー

全データをメモリに展開して処理（小〜中規模データ向け）
"""

import time
from typing import Optional, Callable, Union
import torch
import torch.nn.functional as F

from .base import Phase1Trainer, print_flush
from src.evaluation import check_convergence, ConvergenceResult


class MemoryPhase1Trainer(Phase1Trainer):
    """メモリ展開型Phase 1トレーナー"""

    def train_epochs(
        self,
        data_provider,
        num_epochs: int = 1,
        label: str = "Train"
    ) -> torch.Tensor:
        """
        複数エポックのPhase1学習（シャッフル学習対応）

        Args:
            data_provider: MemoryDataProvider（reshuffle機能付き）
            num_epochs: エポック数（1以上）
            label: ラベル

        Returns:
            最終エポックのコンテキスト
        """
        for epoch in range(num_epochs):
            if epoch > 0:
                # 2エポック目以降はreshuffle
                token_ids = data_provider.reshuffle()
            else:
                token_ids = data_provider.get_all_train_tokens(self.device)

            epoch_label = f"{label} (epoch {epoch+1}/{num_epochs})"
            contexts = self.train(token_ids, label=epoch_label)

        return contexts

    def train(self, token_ids: torch.Tensor, label: str = "Train") -> torch.Tensor:
        self.model.train()
        num_tokens = len(token_ids)

        print_flush(f"\n{'='*70}")
        print_flush(f"PHASE 1: 固定点コンテキスト学習 - {label}")
        print_flush(f"{'='*70}")
        print_flush(f"  Mode: Memory")
        print_flush(f"  Tokens: {num_tokens:,}")
        print_flush(f"  Max iterations: {self.config.phase1_max_iterations}")
        print_flush(f"  Learning rate: {self.config.phase1_learning_rate}")
        print_flush(f"  Diversity weight: {self.config.dist_reg_weight}")

        # ContextBlockのパラメータのみ学習
        context_params = list(self.model.context_block.parameters())
        print_flush(f"  ContextBlock params: {sum(p.numel() for p in context_params):,}")
        optimizer = torch.optim.Adam(context_params, lr=self.config.phase1_learning_rate)

        # トークン埋め込み（1回のみ計算）
        with torch.no_grad():
            token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)

        previous_contexts = None
        final_convergence_rate = 0.0

        for iteration in range(self.config.phase1_max_iterations):
            start_time = time.time()

            if iteration == 0:
                # Iteration 0: シーケンシャル（学習なし）
                contexts = self._forward_sequential(token_embeds, None)
                previous_contexts = contexts.detach()
                elapsed = time.time() - start_time
                print_flush(f"Iteration 1/{self.config.phase1_max_iterations}: シーケンシャル [{elapsed:.2f}s]")
                continue

            # Iteration 1+: 並列処理
            contexts = self._forward_parallel(token_embeds, previous_contexts)

            # 損失計算
            cvfp_loss = F.mse_loss(contexts, previous_contexts)
            diversity_loss = self._compute_diversity_loss(contexts)
            total_loss = (1 - self.config.dist_reg_weight) * cvfp_loss + self.config.dist_reg_weight * diversity_loss

            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.phase1_gradient_clip)
                optimizer.step()

            # 収束率
            convergence_rate = self._compute_convergence_rate(contexts, previous_contexts, num_tokens)
            previous_contexts = contexts.detach()
            final_convergence_rate = convergence_rate

            elapsed = time.time() - start_time
            print_flush(
                f"Iteration {iteration+1}/{self.config.phase1_max_iterations}: "
                f"収束={convergence_rate*100:.1f}% | "
                f"Loss={total_loss.item():.6f} | "
                f"CVFP={cvfp_loss.item():.6f} | "
                f"Div={diversity_loss.item():.6f} [{elapsed:.2f}s]"
            )

            # 早期停止: 最小イテレーション数を超え、かつ収束率が閾値以上
            min_iterations = getattr(self.config, 'phase1_min_iterations', 3)
            if iteration + 1 >= min_iterations and convergence_rate >= self.config.phase1_min_converged_ratio:
                print_flush(f"  → Early stopping (min_iterations={min_iterations} satisfied)")
                break

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        self._training_stats = {
            'iterations': iteration + 1,
            'convergence_rate': final_convergence_rate,
            'num_tokens': num_tokens,
        }

        print_flush(f"\nPhase 1 完了: {int(final_convergence_rate * num_tokens)}/{num_tokens} トークン収束\n")
        return contexts.detach()

    def evaluate(
        self,
        token_ids: torch.Tensor,
        label: str = "Val",
        num_trials: int = None,
        return_contexts_only: bool = False
    ) -> Union[ConvergenceResult, torch.Tensor]:
        """
        検証データを評価（収束判定）

        複数回イテレーションして、CVFP損失が収束するかを判定。

        Args:
            token_ids: トークンID
            label: ラベル
            num_trials: イテレーション回数（Noneの場合はconfig.val_convergence_trialsを使用）
            return_contexts_only: Trueの場合はコンテキストのみ返す（後方互換性）

        Returns:
            ConvergenceResult または contexts
        """
        if num_trials is None:
            num_trials = getattr(self.config, 'val_convergence_trials', 10)

        print_flush(f"\nEvaluating {label} data...")

        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)

        result = check_convergence(
            model=self.model,
            token_ids=token_ids,
            device=self.device,
            num_trials=num_trials,
            verbose=True,
            num_input_tokens=num_input_tokens
        )

        if return_contexts_only:
            return result.contexts

        return result

    def _forward_sequential(self, token_embeds: torch.Tensor, previous_contexts: Optional[torch.Tensor]) -> torch.Tensor:
        """シーケンシャル処理（Iteration 0用）- 勾配なし"""
        num_tokens = len(token_embeds)
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)

        # 結果を格納するテンソルを事前確保（メモリ効率）
        contexts = torch.zeros(num_tokens, self.model.context_dim, device=self.device)

        if previous_contexts is None:
            context = torch.zeros(1, self.model.context_dim, device=self.device)
        else:
            context = previous_contexts[-1].unsqueeze(0).detach()

        # トークン履歴を初期化（ゼロベクトルで埋める）
        token_history = [torch.zeros(self.model.embed_dim, device=self.device)
                         for _ in range(num_input_tokens - 1)]

        # 勾配なしで処理（Iteration 0は学習なし）
        with torch.no_grad():
            for i, token_embed in enumerate(token_embeds):
                # 履歴 + 現在のトークンを結合
                token_history.append(token_embed)
                combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)

                context = self.model.context_block(context, combined_tokens.unsqueeze(0))
                contexts[i] = context.squeeze(0)

                # メモリ効率: 古い履歴を削除
                if len(token_history) > num_input_tokens:
                    token_history = token_history[-num_input_tokens:]

        return contexts

    def _forward_parallel(self, token_embeds: torch.Tensor, previous_contexts: torch.Tensor) -> torch.Tensor:
        """並列処理（Iteration 1+用）"""
        num_tokens = len(token_embeds)
        batch_size = self.config.phase1_batch_size
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)

        # コンテキスト準備（既存ロジック）
        contexts_for_batch = torch.zeros(num_tokens, self.model.context_dim, device=self.device)
        contexts_for_batch[1:] = previous_contexts[:-1].detach()
        contexts_for_batch[0] = previous_contexts[-1].detach()

        if self.config.phase1_context_noise > 0 and self.model.training:
            noise = torch.randn_like(contexts_for_batch) * self.config.phase1_context_noise
            contexts_for_batch = contexts_for_batch + noise

        # バッチ処理（メモリ効率: combined_tokensをバッチごとに作成）
        all_contexts = []
        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)

            # バッチ分のcombined_tokensを作成（メモリ効率化）
            batch_combined = self._build_combined_tokens_batch(
                token_embeds, num_input_tokens, start_idx, end_idx
            )

            batch_output = self.model.context_block(
                contexts_for_batch[start_idx:end_idx],
                batch_combined
            )
            all_contexts.append(batch_output)

            # メモリ解放
            del batch_combined

        return torch.cat(all_contexts, dim=0)

    def _build_combined_tokens_batch(
        self, token_embeds: torch.Tensor, num_input_tokens: int,
        start_idx: int, end_idx: int
    ) -> torch.Tensor:
        """
        バッチ範囲のみの結合テンソルを作成（メモリ効率版）

        Args:
            token_embeds: [num_tokens, embed_dim] - 全トークン
            num_input_tokens: 入力トークン数
            start_idx: バッチ開始インデックス
            end_idx: バッチ終了インデックス

        Returns:
            combined_tokens: [batch_size, embed_dim * num_input_tokens]
        """
        if num_input_tokens == 1:
            return token_embeds[start_idx:end_idx]

        batch_size = end_idx - start_idx
        embed_dim = self.model.embed_dim

        combined_tokens = torch.zeros(
            batch_size,
            embed_dim * num_input_tokens,
            device=self.device
        )

        for batch_i, global_i in enumerate(range(start_idx, end_idx)):
            for j in range(num_input_tokens):
                # j=0 が最も古いトークン、j=num_input_tokens-1 が現在のトークン
                src_idx = global_i - (num_input_tokens - 1 - j)
                if src_idx >= 0:
                    start = j * embed_dim
                    end = (j + 1) * embed_dim
                    combined_tokens[batch_i, start:end] = token_embeds[src_idx]
                # src_idx < 0 の場合はゼロベクトル（初期化済み）

        return combined_tokens

    def _compute_diversity_loss(self, contexts: torch.Tensor) -> torch.Tensor:
        context_mean = contexts.mean(dim=0)
        deviation = contexts - context_mean
        return -torch.norm(deviation, p=2) / len(contexts)

    def _compute_convergence_rate(self, current: torch.Tensor, previous: torch.Tensor, num_tokens: int) -> float:
        with torch.no_grad():
            token_losses = ((current - previous) ** 2).mean(dim=1)
            converged = token_losses < self.config.phase1_convergence_threshold
            return converged.sum().item() / num_tokens

    @property
    def is_streaming(self) -> bool:
        return False
