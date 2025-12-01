"""
MemoryPhase1Trainer - メモリ展開型Phase 1トレーナー

全データをメモリに展開して処理（小〜中規模データ向け）
"""

import time
from typing import Optional, Any
import torch
import torch.nn.functional as F

from .base import Phase1Trainer, TrainResult, EvalResult, ContextCache
from src.utils.io import print_flush
from src.utils.device import is_cuda_device, clear_gpu_cache


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
        for epoch in range(num_epochs):
            if epoch > 0:
                # 2エポック目以降はreshuffle
                token_ids = data_provider.reshuffle()
            else:
                token_ids = data_provider.get_all_train_tokens(self.device)

            epoch_label = f"{label} (epoch {epoch+1}/{num_epochs})"
            result = self.train(token_ids, label=epoch_label)
            # train_epochsは常にreturn_all_layers=Falseなので、結果はTensor
            contexts = result if isinstance(result, torch.Tensor) else result[0]

        return contexts

    def train(
        self,
        token_ids: torch.Tensor,
        label: str = "Train",
        data_provider: Any = None,
        return_all_layers: bool = False,
        val_token_ids: Optional[torch.Tensor] = None
    ) -> TrainResult:
        """
        Phase 1訓練

        Args:
            token_ids: トークンID
            label: ラベル
            data_provider: データプロバイダー（未使用、互換性のため残す）
            return_all_layers: Trueの場合、全レイヤー出力も返す（Phase 2キャッシュ用）
            val_token_ids: 検証用トークンID（早期停止用、オプション）

        Returns:
            return_all_layers=False: コンテキスト [num_tokens, context_dim]
            return_all_layers=True: (コンテキスト, 全レイヤー出力, トークン埋め込み)
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
    ) -> TrainResult:
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
            return final_contexts

        # 全レイヤーキャッシュ収集
        all_layer_contexts, token_embeds_combined = self._collect_layer_cache(
            token_embeds, previous_contexts
        )

        return final_contexts, all_layer_contexts, token_embeds_combined

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
            (previous_contexts, final_convergence_rate, val_early_stopped, best_val_er, final_iter)
        """
        # Validation早期停止の設定
        val_early_stopping = getattr(self.config, 'phase1_val_early_stopping', False)
        val_frequency = getattr(self.config, 'phase1_val_frequency', 5)
        val_patience = getattr(self.config, 'phase1_val_patience', 2)
        best_val_er = 0.0
        val_patience_counter = 0
        val_early_stopped = False

        previous_contexts: Optional[torch.Tensor] = None
        final_convergence_rate = 0.0
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

            print_flush(
                f"  Iter {iteration+1}: conv={convergence_rate*100:.0f}% "
                f"loss={total_loss:.4f} [{elapsed:.1f}s]"
            )

            # Validation早期停止チェック
            if (val_early_stopping and
                val_token_ids is not None and
                (iteration + 1) % val_frequency == 0):

                should_stop, best_val_er, val_patience_counter = self._check_validation_early_stop(
                    val_token_ids, best_val_er, val_patience_counter, val_patience
                )
                if should_stop:
                    val_early_stopped = True
                    break

        assert previous_contexts is not None
        return previous_contexts, final_convergence_rate, val_early_stopped, best_val_er, final_iter

    def _train_iteration(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: torch.Tensor,
        num_tokens: int,
        optimizer: torch.optim.Optimizer
    ) -> tuple[torch.Tensor, float, float, float]:
        """
        単一イテレーションの処理

        Returns:
            (contexts, total_loss, convergence_rate, elapsed_time)
        """
        start_time = time.time()

        token_embeds_gpu = token_embeds.to(self.device)
        previous_contexts_gpu = previous_contexts.to(self.device)

        contexts, total_loss, _, _ = self._forward_parallel_with_grad_accum(
            token_embeds_gpu, previous_contexts_gpu, optimizer
        )

        convergence_rate = self._compute_convergence_rate(contexts, previous_contexts_gpu, num_tokens)

        del token_embeds_gpu, previous_contexts_gpu
        clear_gpu_cache(self.device)

        elapsed = time.time() - start_time

        return contexts, total_loss, convergence_rate, elapsed

    def _check_validation_early_stop(
        self,
        val_token_ids: torch.Tensor,
        best_val_er: float,
        patience_counter: int,
        patience: int
    ) -> tuple[bool, float, int]:
        """
        Validation早期停止チェック

        Returns:
            (should_stop, best_val_er, patience_counter)
        """
        val_er = self._quick_validate(val_token_ids)
        val_er_percent = val_er / self.model.context_dim * 100
        print_flush(f"    Val ER: {val_er_percent:.1f}%")

        if val_er > best_val_er:
            best_val_er = val_er
            patience_counter = 0
        else:
            patience_counter += 1

        should_stop = patience_counter >= patience
        if should_stop:
            print_flush(f"  → Val early stop: ER not improving for {patience} checks")

        return should_stop, best_val_er, patience_counter

    def _collect_layer_cache(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 2用の全レイヤーキャッシュを収集

        Returns:
            (all_layer_contexts, token_embeds_combined)
        """
        cache_start = time.time()
        print_flush("  Collecting cache (parallel)...")
        self.model.eval()
        token_embeds_gpu = token_embeds.to(self.device)

        # Phase 2用に最後のトークンを除く
        input_token_embeds = token_embeds_gpu[:-1]
        num_input_tokens_total = len(input_token_embeds)

        # Phase 2では token i の処理に previous_contexts[i-1] を使用
        initial_context = torch.zeros(1, self.model.context_dim, device=self.device)
        shifted_contexts = torch.cat([
            initial_context,
            previous_contexts[:-2].to(self.device)
        ], dim=0)

        # バッチサイズ決定
        cache_batch_size = self._compute_cache_batch_size()

        num_layers = self.model.num_layers
        all_layer_contexts = torch.zeros(
            num_layers, num_input_tokens_total, self.model.context_dim,
            device='cpu', dtype=torch.float32
        )

        # token_embeds_combined を準備
        token_embeds_combined = self._prepare_combined_token_embeds(
            input_token_embeds, num_input_tokens_total
        )

        # バッチ処理
        with torch.no_grad():
            for batch_start in range(0, num_input_tokens_total, cache_batch_size):
                batch_end = min(batch_start + cache_batch_size, num_input_tokens_total)
                batch_contexts = shifted_contexts[batch_start:batch_end]
                batch_embeds = token_embeds_combined[batch_start:batch_end].to(self.device)

                batch_results = self.model.context_block.forward_with_intermediates_batch(
                    batch_contexts, batch_embeds
                )

                all_layer_contexts[:, batch_start:batch_end, :] = batch_results.cpu()

                del batch_results, batch_embeds
                clear_gpu_cache(self.device)

        cache_elapsed = time.time() - cache_start
        print_flush(f"  Cache collected (parallel) [{cache_elapsed:.1f}s]")

        self.model.train()
        del token_embeds_gpu
        clear_gpu_cache(self.device)

        return all_layer_contexts, token_embeds_combined

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
        num_input_tokens_config = self.config.num_input_tokens
        if num_input_tokens_config == 1:
            return input_token_embeds.cpu()

        # 複数トークン入力の場合（履歴を結合）
        token_embeds_combined = torch.zeros(
            num_input_tokens_total, self.model.embed_dim * num_input_tokens_config,
            device='cpu', dtype=input_token_embeds.dtype
        )
        for i in range(num_input_tokens_total):
            start_idx = max(0, i - num_input_tokens_config + 1)
            token_window = input_token_embeds[start_idx:i+1]
            if len(token_window) < num_input_tokens_config:
                padding = torch.zeros(
                    num_input_tokens_config - len(token_window),
                    self.model.embed_dim,
                    device=self.device
                )
                token_window = torch.cat([padding, token_window], dim=0)
            token_embeds_combined[i] = token_window.flatten().cpu()

        return token_embeds_combined

    def evaluate(
        self,
        token_ids: torch.Tensor,
        label: str = "Val",
        num_trials: Optional[int] = None,
        return_contexts_only: bool = False,
        return_all_layers: bool = True
    ) -> EvalResult:
        """
        検証データのPhase 2キャッシュを収集

        Args:
            token_ids: トークンID
            label: ラベル
            num_trials: 未使用
            return_contexts_only: 未使用
            return_all_layers: 必ずTrue（Phase 2キャッシュ用）

        Returns:
            (contexts, all_layer_contexts, token_embeds)
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
        return contexts, all_layer_contexts, token_embeds_combined

    def _forward_sequential(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: Optional[torch.Tensor],
        collect_all_layers: bool = False
    ) -> tuple[torch.Tensor, Optional[ContextCache], Optional[torch.Tensor]]:
        """
        シーケンシャル処理（Iteration 0用 or 全レイヤー収集用）- 勾配なし

        Args:
            token_embeds: トークン埋め込み [num_tokens, embed_dim]
            previous_contexts: 前回のコンテキスト（Noneの場合はゼロ初期化）
            collect_all_layers: 全レイヤー出力を収集するか

        Returns:
            collect_all_layers=False: (contexts, None, None)
            collect_all_layers=True: (contexts, context_cache, token_embeds_combined)
        """
        num_tokens = len(token_embeds)
        num_input_tokens = self.config.num_input_tokens

        # 結果を格納するテンソルを事前確保（メモリ効率）
        contexts = torch.zeros(num_tokens, self.model.context_dim, device=self.device)

        if previous_contexts is None:
            context = torch.zeros(1, self.model.context_dim, device=self.device)
        else:
            context = previous_contexts[-1].unsqueeze(0).detach()

        # トークン履歴を初期化（ゼロベクトルで埋める）
        token_history = [torch.zeros(self.model.embed_dim, device=self.device)
                         for _ in range(num_input_tokens - 1)]

        # 全レイヤー収集用の変数を初期化
        context_cache: Optional[ContextCache] = None
        token_embeds_combined: Optional[torch.Tensor] = None

        if collect_all_layers:
            # Phase 2用キャッシュ形式で初期化
            # token継ぎ足し方式: 全レイヤー同じcontext_dim
            num_layers = self.model.num_layers
            context_dim = self.model.context_dim

            context_cache = torch.zeros(
                num_layers, num_tokens, context_dim,
                device=self.device, dtype=torch.float32
            )

            token_embeds_combined = torch.zeros(
                num_tokens, self.model.embed_dim * num_input_tokens,
                device=self.device, dtype=torch.float32
            )

        # 勾配なしで処理
        with torch.no_grad():
            for i, token_embed in enumerate(token_embeds):
                # 履歴 + 現在のトークンを結合
                token_history.append(token_embed)
                combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)

                if collect_all_layers:
                    # 全レイヤー出力を取得
                    assert token_embeds_combined is not None
                    token_embeds_combined[i] = combined_tokens

                    context_outputs = self.model.forward_context_with_intermediates(
                        context, combined_tokens.unsqueeze(0)
                    )

                    for layer_idx, ctx_out in enumerate(context_outputs):
                        assert context_cache is not None
                        context_cache[layer_idx, i] = ctx_out.squeeze(0)

                    context = context_outputs[-1]
                    contexts[i] = context.squeeze(0)
                else:
                    # 通常処理（最終レイヤーのみ）
                    context = self.model.context_block(context, combined_tokens.unsqueeze(0))
                    contexts[i] = context.squeeze(0)

                # メモリ効率: 古い履歴を削除
                if len(token_history) > num_input_tokens:
                    token_history = token_history[-num_input_tokens:]

        return contexts, context_cache, token_embeds_combined

    def _forward_parallel_with_grad_accum(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> tuple[torch.Tensor, float, float, float]:
        """
        勾配累積付き並列処理（メモリ効率版）

        バッチごとに勾配を計算・累積し、最後にまとめてパラメータ更新。
        これにより、大規模データでもOOMを回避できる。

        Returns:
            (contexts, total_loss, cvfp_loss, diversity_loss)
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
        total_cvfp_loss = 0.0
        total_diversity_loss = 0.0
        total_loss_sum = 0.0

        # 最終コンテキスト（ラップアラウンド用）
        last_context = previous_contexts[-1].detach()

        # バッチ処理（勾配累積）
        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)
            current_batch_size = end_idx - start_idx

            # バッチ分のコンテキストをその場で作成（メモリ効率化）
            if start_idx == 0:
                # 最初のバッチ: index 0 は last_context を使用
                batch_contexts = torch.zeros(current_batch_size, self.model.context_dim, device=self.device)
                batch_contexts[0] = last_context
                if current_batch_size > 1:
                    batch_contexts[1:] = previous_contexts[:end_idx-1].detach()
            else:
                batch_contexts = previous_contexts[start_idx-1:end_idx-1].detach().clone()

            # ノイズ追加
            if context_noise > 0 and self.model.training:
                noise = torch.randn_like(batch_contexts) * context_noise
                batch_contexts = batch_contexts + noise

            # バッチ分のcombined_tokensを作成
            batch_combined = self._build_combined_tokens_batch(
                token_embeds, num_input_tokens, start_idx, end_idx
            )

            # Forward pass
            batch_output = self.model.context_block(batch_contexts, batch_combined)

            # バッチごとの損失計算
            batch_prev = previous_contexts[start_idx:end_idx].detach()
            cvfp_loss = F.mse_loss(batch_output, batch_prev)
            diversity_loss = self._compute_diversity_loss(batch_output)
            batch_loss = (1 - self.config.dist_reg_weight) * cvfp_loss + self.config.dist_reg_weight * diversity_loss

            # 勾配累積のためにバッチ数で割る
            scaled_loss = batch_loss / num_batches

            # Backward（勾配累積）
            if not torch.isnan(scaled_loss) and not torch.isinf(scaled_loss):
                scaled_loss.backward()

            # 結果をCPUに保存（GPUメモリ節約）
            all_contexts_cpu.append(batch_output.detach().cpu())

            # 損失の記録
            total_cvfp_loss += cvfp_loss.item() * current_batch_size
            total_diversity_loss += diversity_loss.item() * current_batch_size
            total_loss_sum += batch_loss.item() * current_batch_size

            # メモリ解放
            del batch_combined, batch_output, batch_loss, scaled_loss, batch_contexts, batch_prev
            clear_gpu_cache(self.device)

        # 勾配クリッピングとパラメータ更新
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.phase1_gradient_clip)
        optimizer.step()

        # コンテキストを結合してGPUに戻す
        contexts = torch.cat(all_contexts_cpu, dim=0).to(self.device)
        del all_contexts_cpu

        # 平均損失を計算
        avg_cvfp_loss = total_cvfp_loss / num_tokens
        avg_diversity_loss = total_diversity_loss / num_tokens
        avg_total_loss = total_loss_sum / num_tokens

        return contexts, avg_total_loss, avg_cvfp_loss, avg_diversity_loss

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
        result: torch.Tensor = -torch.norm(deviation, p=2) / len(contexts)
        return result

    def _compute_convergence_rate(self, current: torch.Tensor, previous: torch.Tensor, num_tokens: int) -> float:
        with torch.no_grad():
            token_losses = ((current - previous) ** 2).mean(dim=1)
            converged = token_losses < self.config.phase1_convergence_threshold
            return float(converged.sum().item()) / num_tokens

    def _quick_validate(self, val_token_ids: torch.Tensor) -> float:
        """
        高速Validation評価（シーケンシャル処理、サンプリング）

        最終評価と同じシーケンシャル処理を使用してERを計算。
        これにより、最終評価のVal ERと整合性のある値が得られる。

        Args:
            val_token_ids: 検証用トークンID

        Returns:
            effective_rank: Effective Rank値
        """
        self.model.eval()

        # サンプリング（固定数、ただし検証データが少なければ全量）
        # 注意: 500トークンではERが不正確になるため、10000以上を推奨
        sample_size = min(
            len(val_token_ids),
            getattr(self.config, 'phase1_val_sample_size', 10000)
        )
        sample_ids = val_token_ids[:sample_size]

        with torch.no_grad():
            # トークン埋め込み取得（evaluate()と同じ処理）
            token_embeds = self.model.token_embedding(sample_ids.unsqueeze(0).to(self.device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)

            # Phase 2用に最後のトークンを除く（evaluate()と同じ）
            input_token_embeds = token_embeds[:-1]

            # シーケンシャル処理（evaluate()と同じ方式）
            # collect_all_layers=Trueで処理することで、完全に同じコードパスを通る
            contexts, _, _ = self._forward_sequential(input_token_embeds, None, collect_all_layers=True)

            # Effective Rank計算
            effective_rank = self._compute_effective_rank(contexts)

        self.model.train()
        return effective_rank

    def _compute_effective_rank(self, contexts: torch.Tensor) -> float:
        """
        Effective Rank計算（SVDベース - analyze_fixed_pointsと同じ方法）

        Args:
            contexts: コンテキストテンソル [num_tokens, context_dim]

        Returns:
            effective_rank: Effective Rank値（0 〜 context_dim）
        """
        # SVDで特異値を計算（analyze_fixed_pointsと同じ方法）
        _, S, _ = torch.svd(contexts)

        # Effective rank (entropy-based)
        S_normalized = S / S.sum()
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()

        return float(torch.exp(entropy).item())

    @property
    def is_streaming(self) -> bool:
        return False
