"""
MemoryPhase1Trainer - メモリ展開型Phase 1トレーナー

全データをメモリに展開して処理（小〜中規模データ向け）
"""

import time
from typing import Optional, Any
import torch

from .base import Phase1Trainer, Phase1Result, ContextCache
from src.losses.diversity import oacd_loss
from src.utils.io import print_flush
from src.utils.device import is_cuda_device, clear_gpu_cache
from src.utils.token_combiner import TokenCombiner


class MemoryPhase1Trainer(Phase1Trainer):
    """メモリ展開型Phase 1トレーナー"""

    def train_epochs(
        self,
        data_provider: Any,
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
        result: Optional[Phase1Result] = None
        for epoch in range(num_epochs):
            if epoch > 0:
                # 2エポック目以降はreshuffle
                token_ids = data_provider.reshuffle()
            else:
                token_ids = data_provider.get_all_train_tokens(self.device)

            epoch_label = f"{label} (epoch {epoch+1}/{num_epochs})"
            result = self.train(token_ids, label=epoch_label)

        assert result is not None
        return result.contexts

    def train(
        self,
        token_ids: torch.Tensor,
        label: str = "Train",
        return_all_layers: bool = False,
        val_token_ids: Optional[torch.Tensor] = None
    ) -> Phase1Result:
        """
        Phase 1訓練

        Args:
            token_ids: トークンID
            label: ラベル
            return_all_layers: Trueの場合、全レイヤー出力も返す（Phase 2キャッシュ用）
            val_token_ids: 検証用トークンID（早期停止用、オプション）

        Returns:
            Phase1Result: contexts必須、cache/token_embedsはreturn_all_layers=True時のみ
        """
        return self._train_single(
            token_ids, label,
            return_all_layers=return_all_layers,
            val_token_ids=val_token_ids
        )

    def _train_single(
        self,
        token_ids: torch.Tensor,
        label: str = "Train",
        return_all_layers: bool = False,
        val_token_ids: Optional[torch.Tensor] = None
    ) -> Phase1Result:
        """
        通常の単一訓練

        分割された処理:
        - _train_iteration(): 単一イテレーションの処理
        - _check_validation_early_stop(): 早期停止判定
        - _collect_layer_cache(): 全レイヤーキャッシュ収集
        """
        self.model.train()
        num_tokens = len(token_ids)

        # ContextBlockのパラメータのみ学習
        context_params = list(self.model.context_block.parameters())
        print_flush(f"\n[Phase 1] {label}: {num_tokens:,} tokens, {self.config.phase1_max_iterations} iterations")
        optimizer = torch.optim.Adam(context_params, lr=self.config.phase1_learning_rate)

        # トークン埋め込み（1回のみ計算、CPUに保存）
        token_embeds = self._compute_token_embeddings(token_ids)

        # 訓練ループ
        previous_contexts, final_convergence_rate, val_early_stopped, best_val_er, final_iter = (
            self._run_training_loop(token_embeds, num_tokens, optimizer, val_token_ids)
        )

        self._training_stats = {
            'iterations': final_iter + 1,
            'convergence_rate': final_convergence_rate,
            'num_tokens': num_tokens,
            'val_early_stopped': val_early_stopped,
            'best_val_er': best_val_er,
        }

        print_flush(f"  Done: {final_convergence_rate*100:.0f}% converged")

        # 最終結果をGPUに戻す
        assert previous_contexts is not None
        final_contexts = previous_contexts.to(self.device)

        if not return_all_layers:
            return Phase1Result(contexts=final_contexts)

        # G案: 最終レイヤー出力のみキャッシュ [num_tokens, context_dim]
        context_cache, token_embeds_combined = self._collect_layer_cache(
            token_embeds, previous_contexts
        )

        return Phase1Result(
            contexts=final_contexts,
            cache=context_cache,
            token_embeds=token_embeds_combined
        )

    def _compute_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """トークン埋め込みを計算（CPUに保存）"""
        with torch.no_grad():
            token_embeds_gpu = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds_gpu = self.model.embed_norm(token_embeds_gpu).squeeze(0)
            token_embeds = token_embeds_gpu.cpu()
            del token_embeds_gpu
            clear_gpu_cache(self.device)
        return token_embeds

    def _run_training_loop(
        self,
        token_embeds: torch.Tensor,
        num_tokens: int,
        optimizer: torch.optim.Optimizer,
        val_token_ids: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, float, bool, float, int]:
        """
        訓練ループを実行

        Returns:
            (previous_contexts, final_convergence_rate, early_stopped, best_convergence_rate, final_iter)
        """
        # 収束率Early Stoppingの設定
        early_stopping = getattr(self.config, 'phase1_early_stopping', True)
        early_stopping_threshold = getattr(self.config, 'phase1_early_stopping_threshold', 0.30)
        min_convergence_improvement = getattr(self.config, 'phase1_min_convergence_improvement', 0.01)

        previous_contexts: Optional[torch.Tensor] = None
        final_convergence_rate = 0.0
        best_convergence_rate = 0.0
        prev_convergence_rate = 0.0  # 前回の収束率（改善幅チェック用）
        no_improvement_count = 0     # 改善なしカウント
        early_stopped = False
        final_iter = 0

        for iteration in range(self.config.phase1_max_iterations):
            final_iter = iteration

            if iteration == 0:
                # Iteration 0: 小さなランダム値で初期化
                previous_contexts = torch.randn(num_tokens, self.model.context_dim) * 0.01
                print_flush("  Iter 1: random init")
                continue

            # Iteration 1+: 勾配累積付き並列処理
            assert previous_contexts is not None
            contexts, total_loss, convergence_rate, elapsed = self._train_iteration(
                token_embeds, previous_contexts, num_tokens, optimizer
            )

            previous_contexts = contexts.detach().cpu()
            final_convergence_rate = convergence_rate
            best_convergence_rate = max(best_convergence_rate, convergence_rate)

            # 改善幅計算
            improvement = convergence_rate - prev_convergence_rate
            improvement_marker = ""
            if iteration >= 2:  # Iter 2以降で改善幅チェック
                if improvement < min_convergence_improvement:
                    no_improvement_count += 1
                    improvement_marker = f" (↑{improvement*100:.1f}%)"
                else:
                    no_improvement_count = 0

            print_flush(
                f"  Iter {iteration+1}: conv={convergence_rate*100:.0f}% "
                f"loss={total_loss:.4f} [{elapsed:.1f}s]{improvement_marker}"
            )

            prev_convergence_rate = convergence_rate

            # 収束率Early Stopping（閾値達成）
            if early_stopping and convergence_rate >= early_stopping_threshold:
                early_stopped = True
                print_flush(f"  → Early stop: conv {convergence_rate*100:.0f}% >= {early_stopping_threshold*100:.0f}%")
                break

            # 収束率改善Early Stopping（改善幅不足）
            if early_stopping and no_improvement_count >= 1 and iteration >= 2:
                early_stopped = True
                print_flush(
                    f"  → Early stop: improvement {improvement*100:.1f}% < {min_convergence_improvement*100:.0f}%"
                )
                break

        assert previous_contexts is not None
        return previous_contexts, final_convergence_rate, early_stopped, best_convergence_rate, final_iter

    def _train_iteration(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: torch.Tensor,
        num_tokens: int,
        optimizer: torch.optim.Optimizer
    ) -> tuple[torch.Tensor, float, float, float]:
        """
        単一イテレーションの処理

        大規模データでのOOMを防ぐため、token_embedsとprevious_contextsは
        CPUに保持し、バッチ処理内で必要な分だけGPUに転送する。

        Returns:
            (contexts, total_loss, convergence_rate, elapsed_time)
        """
        start_time = time.time()

        # CPUのまま渡す（_forward_parallel_with_grad_accum内でバッチごとにGPU転送）
        contexts, total_loss, _ = self._forward_parallel_with_grad_accum(
            token_embeds, previous_contexts, optimizer
        )

        # 収束率計算（バッチ処理で省メモリ）
        # contextsはGPU上、previous_contextsはCPU上なのでGPUに転送してバッチ計算
        convergence_rate = self._compute_convergence_rate_batched(
            contexts, previous_contexts, num_tokens
        )

        elapsed = time.time() - start_time

        return contexts, total_loss, convergence_rate, elapsed

    def _collect_layer_cache(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 2用のコンテキストキャッシュを収集（G案: 最終レイヤー出力のみ）

        大規模データでのOOMを防ぐため、token_embedsとprevious_contextsは
        CPUに保持し、バッチ処理内で必要な分だけGPUに転送する。

        Returns:
            (context_cache, token_embeds_combined)
            - context_cache: [num_tokens, context_dim] - 最終レイヤー出力
            - token_embeds_combined: [num_tokens, embed_dim * num_input_tokens]
        """
        cache_start = time.time()
        print_flush("  Collecting cache (parallel)...")
        self.model.eval()

        # Phase 2用に最後のトークンを除く（CPUのまま）
        input_token_embeds = token_embeds[:-1]
        num_input_tokens_total = len(input_token_embeds)

        # Phase 2では token i の処理に previous_contexts[i-1] を使用
        # shifted_contextsはCPUに保持（バッチごとにGPU転送）
        initial_context = torch.zeros(1, self.model.context_dim, device='cpu')
        shifted_contexts = torch.cat([
            initial_context,
            previous_contexts[:-2]
        ], dim=0)

        # バッチサイズ決定
        cache_batch_size = self._compute_cache_batch_size()

        # G案: 最終レイヤー出力のみ保持 [num_tokens, context_dim]
        context_cache = torch.zeros(
            num_input_tokens_total, self.model.context_dim,
            device='cpu', dtype=torch.float32
        )

        # token_embeds_combined を準備（CPUで）
        print_flush(f"    Preparing combined tokens ({num_input_tokens_total:,} tokens)...")
        combine_start = time.time()
        token_embeds_combined = self._prepare_combined_token_embeds(
            input_token_embeds, num_input_tokens_total
        )
        print_flush(f"    Combined tokens ready [{time.time() - combine_start:.1f}s]")

        # バッチ処理（バッチごとにGPU転送）
        with torch.no_grad():
            for batch_start in range(0, num_input_tokens_total, cache_batch_size):
                batch_end = min(batch_start + cache_batch_size, num_input_tokens_total)

                # バッチ分だけGPUに転送
                batch_contexts = shifted_contexts[batch_start:batch_end].to(self.device)
                batch_embeds = token_embeds_combined[batch_start:batch_end].to(self.device)

                # G案: forward_batchで最終レイヤー出力のみ取得
                batch_results = self.model.context_block.forward_batch(
                    batch_contexts, batch_embeds
                )

                context_cache[batch_start:batch_end, :] = batch_results.cpu()

                del batch_results, batch_embeds, batch_contexts
                clear_gpu_cache(self.device)

        cache_elapsed = time.time() - cache_start
        print_flush(f"  Cache collected (parallel) [{cache_elapsed:.1f}s]")

        self.model.train()
        clear_gpu_cache(self.device)

        return context_cache, token_embeds_combined

    def _compute_cache_batch_size(self) -> int:
        """キャッシュ収集用のバッチサイズを計算"""
        cache_batch_size = 50000
        if is_cuda_device(self.device):
            clear_gpu_cache(self.device)
            free_mem_gb = (torch.cuda.get_device_properties(0).total_memory
                          - torch.cuda.memory_allocated()) / (1024**3)
            cache_batch_size = min(cache_batch_size, int(free_mem_gb * 1024 * 1024 / 30))
            cache_batch_size = max(cache_batch_size, 10000)
        return cache_batch_size

    def _prepare_combined_token_embeds(
        self,
        input_token_embeds: torch.Tensor,
        num_input_tokens_total: int
    ) -> torch.Tensor:
        """num_input_tokens対応のtoken_embeds_combinedを準備"""
        combiner = TokenCombiner(self.config.num_input_tokens, self.model.embed_dim)
        return combiner.combine_all(input_token_embeds, device=torch.device('cpu'))

    def evaluate(
        self,
        token_ids: torch.Tensor,
        label: str = "Val",
        return_all_layers: bool = True
    ) -> Phase1Result:
        """
        検証データのPhase 2キャッシュを収集

        Args:
            token_ids: トークンID
            label: ラベル
            return_all_layers: 必ずTrue（Phase 2キャッシュ用）

        Returns:
            Phase1Result: contexts, cache, token_embedsすべて含む
        """

        self.model.eval()

        # トークン埋め込みを計算
        with torch.no_grad():
            token_embeds_gpu = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds_gpu = self.model.embed_norm(token_embeds_gpu).squeeze(0)

        # Phase 2用に最後のトークンを除く
        input_token_embeds = token_embeds_gpu[:-1]

        # Phase 2キャッシュ用に全レイヤー出力を収集
        contexts, all_layer_contexts, token_embeds_combined = self._forward_sequential(
            input_token_embeds, None, collect_all_layers=True
        )

        del token_embeds_gpu
        clear_gpu_cache(self.device)

        assert all_layer_contexts is not None
        assert token_embeds_combined is not None
        return Phase1Result(
            contexts=contexts,
            cache=all_layer_contexts,
            token_embeds=token_embeds_combined
        )

    def _forward_sequential(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: Optional[torch.Tensor],
        collect_all_layers: bool = False
    ) -> tuple[torch.Tensor, Optional[ContextCache], Optional[torch.Tensor]]:
        """
        シーケンシャル処理（Iteration 0用 or キャッシュ収集用）- 勾配なし

        G案: 最終レイヤー出力のみ保持

        Args:
            token_embeds: トークン埋め込み [num_tokens, embed_dim]
            previous_contexts: 前回のコンテキスト（Noneの場合はゼロ初期化）
            collect_all_layers: キャッシュを収集するか

        Returns:
            collect_all_layers=False: (contexts, None, None)
            collect_all_layers=True: (contexts, context_cache, token_embeds_combined)
                - context_cache: [num_tokens, context_dim] (G案: 最終レイヤーのみ)
        """
        num_tokens = len(token_embeds)
        num_input_tokens = self.config.num_input_tokens
        combiner = TokenCombiner(num_input_tokens, self.model.embed_dim)

        # 結果を格納するテンソルを事前確保（メモリ効率）
        contexts = torch.zeros(num_tokens, self.model.context_dim, device=self.device)

        if previous_contexts is None:
            context = torch.zeros(1, self.model.context_dim, device=self.device)
        else:
            context = previous_contexts[-1].unsqueeze(0).detach()

        # トークン履歴を初期化（空リスト）
        token_history: list[torch.Tensor] = []

        # キャッシュ収集用の変数を初期化
        context_cache: Optional[ContextCache] = None
        token_embeds_combined: Optional[torch.Tensor] = None

        if collect_all_layers:
            # G案: 最終レイヤー出力のみ [num_tokens, context_dim]
            context_cache = torch.zeros(
                num_tokens, self.model.context_dim,
                device=self.device, dtype=torch.float32
            )

            token_embeds_combined = torch.zeros(
                num_tokens, combiner.combined_dim,
                device=self.device, dtype=torch.float32
            )

        # 勾配なしで処理
        with torch.no_grad():
            for i, token_embed in enumerate(token_embeds):
                # TokenCombinerを使用してトークンを結合
                combined_tokens = combiner.combine_single(token_history, token_embed)

                # 履歴を更新（現在のトークンを追加し、古いものを削除）
                token_history.append(token_embed)
                if len(token_history) >= num_input_tokens:
                    token_history = token_history[-(num_input_tokens - 1):]

                # forward処理
                context = self.model.context_block(context, combined_tokens.unsqueeze(0))
                contexts[i] = context.squeeze(0)

                if collect_all_layers:
                    # G案: 最終レイヤー出力をキャッシュ
                    assert context_cache is not None
                    assert token_embeds_combined is not None
                    context_cache[i] = context.squeeze(0)
                    token_embeds_combined[i] = combined_tokens

        return contexts, context_cache, token_embeds_combined

    def _forward_parallel_with_grad_accum(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> tuple[torch.Tensor, float, float]:
        """
        勾配累積付き並列処理（メモリ効率版）

        バッチごとに勾配を計算・累積し、最後にまとめてパラメータ更新。
        これにより、大規模データでもOOMを回避できる。

        Returns:
            (contexts, total_loss, diversity_loss)
        """
        num_tokens = len(token_embeds)
        batch_size = self.config.phase1_batch_size
        num_input_tokens = self.config.num_input_tokens
        num_batches = (num_tokens + batch_size - 1) // batch_size
        context_noise = self.config.phase1_context_noise

        # 勾配をゼロに
        optimizer.zero_grad()

        # 結果を格納（CPUに保存してGPUメモリ節約）
        all_contexts_cpu = []
        total_diversity_loss = 0.0
        total_loss_sum = 0.0

        # 最終コンテキスト（ラップアラウンド用）- GPUに転送
        last_context = previous_contexts[-1].detach().to(self.device)

        # バッチ処理（勾配累積）
        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)
            current_batch_size = end_idx - start_idx

            # バッチ分のコンテキストをその場で作成（GPUに転送）
            if start_idx == 0:
                # 最初のバッチ: index 0 は last_context を使用
                batch_contexts = torch.zeros(current_batch_size, self.model.context_dim, device=self.device)
                batch_contexts[0] = last_context
                if current_batch_size > 1:
                    # CPUからGPUに転送
                    batch_contexts[1:] = previous_contexts[:end_idx-1].detach().to(self.device)
            else:
                # CPUからGPUに転送
                batch_contexts = previous_contexts[start_idx-1:end_idx-1].detach().to(self.device)

            # ノイズ追加
            if context_noise > 0 and self.model.training:
                noise = torch.randn_like(batch_contexts) * context_noise
                batch_contexts = batch_contexts + noise

            # バッチ分のcombined_tokensを作成（GPUに転送）
            batch_combined = self._build_combined_tokens_batch(
                token_embeds, num_input_tokens, start_idx, end_idx
            ).to(self.device)

            # Forward pass
            batch_output = self.model.context_block(batch_contexts, batch_combined)

            # バッチごとの損失計算（多様性損失のみ）
            diversity_loss = self._compute_diversity_loss(batch_output)
            batch_loss = diversity_loss

            # 勾配累積のためにバッチ数で割る
            scaled_loss = batch_loss / num_batches

            # Backward（勾配累積）
            if not torch.isnan(scaled_loss) and not torch.isinf(scaled_loss):
                scaled_loss.backward()

            # 結果をCPUに保存（GPUメモリ節約）
            all_contexts_cpu.append(batch_output.detach().cpu())

            # 損失の記録
            total_diversity_loss += diversity_loss.item() * current_batch_size
            total_loss_sum += batch_loss.item() * current_batch_size

            # メモリ解放
            del batch_combined, batch_output, batch_loss, scaled_loss, batch_contexts
            clear_gpu_cache(self.device)

        # 勾配クリッピングとパラメータ更新
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.phase1_gradient_clip)
        optimizer.step()

        # コンテキストを結合（CPUに保持してメモリ節約）
        contexts_cpu = torch.cat(all_contexts_cpu, dim=0)
        del all_contexts_cpu

        # 平均損失を計算
        avg_diversity_loss = total_diversity_loss / num_tokens
        avg_total_loss = total_loss_sum / num_tokens

        return contexts_cpu, avg_total_loss, avg_diversity_loss

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
        combiner = TokenCombiner(num_input_tokens, self.model.embed_dim)
        return combiner.combine_batch(token_embeds, start_idx, end_idx, self.device)

    def _compute_diversity_loss(self, contexts: torch.Tensor) -> torch.Tensor:
        """多様性損失計算（デフォルト: OACD）"""
        return oacd_loss(contexts)

    def _compute_convergence_rate_batched(
        self, current: torch.Tensor, previous: torch.Tensor, num_tokens: int
    ) -> float:
        """
        収束率を計算（CPUテンソル対応、バッチ処理でメモリ効率化）

        大規模データでのOOMを防ぐため、バッチごとにGPU転送して処理。
        currentとpreviousはどちらもCPUテンソルを想定。

        Args:
            current: 現在のコンテキスト（CPUテンソル）
            previous: 前回のコンテキスト（CPUテンソル）
            num_tokens: トークン数

        Returns:
            収束率（0.0-1.0）
        """
        with torch.no_grad():
            # バッチサイズ（メモリ効率のため分割処理）
            batch_size = 100000  # 10万トークンずつ処理

            converged_count = 0
            threshold = self.config.phase1_convergence_threshold

            for start_idx in range(0, num_tokens, batch_size):
                end_idx = min(start_idx + batch_size, num_tokens)

                # バッチ分だけGPUに転送して計算
                current_batch = current[start_idx:end_idx].to(self.device)
                previous_batch = previous[start_idx:end_idx].to(self.device)

                token_losses = ((current_batch - previous_batch) ** 2).mean(dim=1)
                converged_count += (token_losses < threshold).sum().item()

                del current_batch, previous_batch, token_losses
                clear_gpu_cache(self.device)

            return converged_count / num_tokens

    @property
    def is_streaming(self) -> bool:
        return False
