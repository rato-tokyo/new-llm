"""
MemoryPhase1Trainer - メモリ展開型Phase 1トレーナー

全データをメモリに展開して処理（小〜中規模データ向け）
"""

import time
from typing import Optional, Any
import torch
import torch.nn.functional as F

from .base import Phase1Trainer, print_flush, TrainResult, EvalResult, ContextCache


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
            result = self.train(token_ids, label=epoch_label)
            # train_epochsは常にreturn_all_layers=Falseなので、結果はTensor
            contexts = result if isinstance(result, torch.Tensor) else result[0]

        return contexts

    def train(
        self,
        token_ids: torch.Tensor,
        label: str = "Train",
        data_provider: Any = None,
        return_all_layers: bool = False
    ) -> TrainResult:
        """
        Phase 1訓練

        Args:
            token_ids: トークンID（分割なしの場合）
            label: ラベル
            data_provider: データプロバイダー（分割訓練用）
            return_all_layers: Trueの場合、全レイヤー出力も返す（Phase 2キャッシュ用）

        Returns:
            return_all_layers=False: コンテキスト [num_tokens, context_dim]
            return_all_layers=True: (コンテキスト, 全レイヤー出力, トークン埋め込み)
        """
        # 分割数を取得
        num_splits = getattr(self.config, 'num_context_splits', 1)

        if num_splits > 1 and data_provider is not None:
            # 分割訓練モード
            return self._train_split_mode(data_provider, label)
        else:
            # 通常モード
            return self._train_single(token_ids, label, return_all_layers=return_all_layers)

    def _train_single(
        self,
        token_ids: torch.Tensor,
        label: str = "Train",
        return_all_layers: bool = False
    ) -> TrainResult:
        """通常の単一訓練"""
        self.model.train()
        num_tokens = len(token_ids)

        print_flush(f"\n{'='*70}")
        print_flush(f"PHASE 1: 固定点コンテキスト学習 - {label}")
        print_flush(f"{'='*70}")
        print_flush("  Mode: Memory (GPU-optimized)")
        print_flush(f"  Tokens: {num_tokens:,}")
        print_flush(f"  Max iterations: {self.config.phase1_max_iterations}")
        print_flush(f"  Learning rate: {self.config.phase1_learning_rate}")
        print_flush(f"  Diversity weight: {self.config.dist_reg_weight}")

        # ContextBlockのパラメータのみ学習
        context_params = list(self.model.context_block.parameters())
        print_flush(f"  ContextBlock params: {sum(p.numel() for p in context_params):,}")
        optimizer = torch.optim.Adam(context_params, lr=self.config.phase1_learning_rate)

        # トークン埋め込み（1回のみ計算、CPUに保存）
        # デバイス判定用（文字列またはtorch.device両対応）
        is_cuda = str(self.device) == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda')

        with torch.no_grad():
            # GPUで計算してCPUに移動（大規模データ対応）
            token_embeds_gpu = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds_gpu = self.model.embed_norm(token_embeds_gpu).squeeze(0)
            token_embeds = token_embeds_gpu.cpu()  # CPUに保存
            del token_embeds_gpu
            if is_cuda:
                torch.cuda.empty_cache()

        previous_contexts: Optional[torch.Tensor] = None  # CPUに保存
        final_convergence_rate = 0.0

        for iteration in range(self.config.phase1_max_iterations):
            start_time = time.time()

            if iteration == 0:
                # Iteration 0: シーケンシャル（学習なし）
                # token_embedsをGPUに移動して処理
                token_embeds_gpu = token_embeds.to(self.device)
                contexts, _, _ = self._forward_sequential(token_embeds_gpu, None)
                previous_contexts = contexts.detach().cpu()  # CPUに保存
                del token_embeds_gpu, contexts
                if is_cuda:
                    torch.cuda.empty_cache()
                elapsed = time.time() - start_time
                print_flush(f"Iteration 1/{self.config.phase1_max_iterations}: シーケンシャル [{elapsed:.2f}s]")
                continue

            # Iteration 1+: 勾配累積付き並列処理
            # token_embedsとprevious_contextsをGPUに移動
            assert previous_contexts is not None  # iteration 0で必ず設定される
            token_embeds_gpu = token_embeds.to(self.device)
            previous_contexts_gpu = previous_contexts.to(self.device)

            contexts, total_loss, cvfp_loss, diversity_loss = self._forward_parallel_with_grad_accum(
                token_embeds_gpu, previous_contexts_gpu, optimizer
            )

            # 収束率（GPUで計算）
            convergence_rate = self._compute_convergence_rate(contexts, previous_contexts_gpu, num_tokens)

            # CPUに保存してGPUメモリ解放
            previous_contexts = contexts.detach().cpu()
            final_convergence_rate = convergence_rate

            del token_embeds_gpu, previous_contexts_gpu, contexts
            if is_cuda:
                torch.cuda.empty_cache()

            elapsed = time.time() - start_time
            print_flush(
                f"Iteration {iteration+1}/{self.config.phase1_max_iterations}: "
                f"収束={convergence_rate*100:.1f}% | "
                f"Loss={total_loss:.6f} | "
                f"CVFP={cvfp_loss:.6f} | "
                f"Div={diversity_loss:.6f} [{elapsed:.2f}s]"
            )

            # 早期停止: 最小イテレーション数を超え、かつ収束率が閾値以上
            min_iterations = getattr(self.config, 'phase1_min_iterations', 3)
            if iteration + 1 >= min_iterations and convergence_rate >= self.config.phase1_min_converged_ratio:
                print_flush(f"  → Early stopping (min_iterations={min_iterations} satisfied)")
                break

        self._training_stats = {
            'iterations': iteration + 1,
            'convergence_rate': final_convergence_rate,
            'num_tokens': num_tokens,
        }

        print_flush(f"\nPhase 1 完了: {int(final_convergence_rate * num_tokens)}/{num_tokens} トークン収束\n")

        # 最終結果をGPUに戻す
        assert previous_contexts is not None  # 必ず1回以上iterationが実行される
        final_contexts = previous_contexts.to(self.device)

        if not return_all_layers:
            return final_contexts

        # return_all_layers=True: Phase 2キャッシュ用に全レイヤー出力を収集
        # 訓練完了後、推論モードで1回だけシーケンシャル処理して全レイヤー出力を取得
        print_flush("Collecting all-layer outputs for Phase 2 cache...")
        self.model.eval()
        token_embeds_gpu = token_embeds.to(self.device)

        # Phase 2用に最後のトークンを除く
        input_token_embeds = token_embeds_gpu[:-1]

        _, all_layer_contexts, token_embeds_combined = self._forward_sequential(
            input_token_embeds, None, collect_all_layers=True
        )

        self.model.train()
        del token_embeds_gpu
        if is_cuda:
            torch.cuda.empty_cache()

        assert all_layer_contexts is not None
        assert token_embeds_combined is not None
        cache_layers = len(all_layer_contexts) if isinstance(all_layer_contexts, list) else all_layer_contexts.shape[0]
        print_flush(f"Cache collected: {cache_layers} layers")

        return final_contexts, all_layer_contexts, token_embeds_combined

    def evaluate(
        self,
        token_ids: torch.Tensor,
        label: str = "Val",
        num_trials: Optional[int] = None,
        return_contexts_only: bool = False,
        return_all_layers: bool = False
    ) -> EvalResult:
        """
        検証データを評価

        1回のシーケンシャル処理でコンテキストを計算。
        return_all_layers=Trueの場合、Phase 2キャッシュ用に全レイヤー出力も収集。

        Args:
            token_ids: トークンID
            label: ラベル
            num_trials: 未使用（後方互換性のため残す）
            return_contexts_only: 未使用（後方互換性のため残す）
            return_all_layers: Trueの場合、全レイヤー出力も返す（Phase 2キャッシュ用）

        Returns:
            return_all_layers=True: (contexts, all_layer_contexts, token_embeds)
            return_all_layers=False: contexts
        """
        print_flush(f"\nEvaluating {label} data ({len(token_ids):,} tokens)...")

        self.model.eval()

        # トークン埋め込みを計算
        with torch.no_grad():
            token_embeds_gpu = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds_gpu = self.model.embed_norm(token_embeds_gpu).squeeze(0)

        # Phase 2用に最後のトークンを除く
        input_token_embeds = token_embeds_gpu[:-1]

        if return_all_layers:
            # Phase 2キャッシュ用に全レイヤー出力を収集
            print_flush("Collecting all-layer outputs for Phase 2 cache (val)...")

            contexts, all_layer_contexts, token_embeds_combined = self._forward_sequential(
                input_token_embeds, None, collect_all_layers=True
            )

            del token_embeds_gpu
            is_cuda = str(self.device) == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda')
            if is_cuda:
                torch.cuda.empty_cache()

            assert all_layer_contexts is not None
            assert token_embeds_combined is not None
            cache_layers = len(all_layer_contexts) if isinstance(all_layer_contexts, list) else all_layer_contexts.shape[0]
            print_flush(f"Cache collected: {cache_layers} layers")
            return contexts, all_layer_contexts, token_embeds_combined

        # return_all_layers=False: コンテキストのみ計算
        contexts, _, _ = self._forward_sequential(
            input_token_embeds, None, collect_all_layers=False
        )

        del token_embeds_gpu
        is_cuda = str(self.device) == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda')
        if is_cuda:
            torch.cuda.empty_cache()

        print_flush(f"Evaluation complete: {len(contexts):,} contexts")
        return contexts

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

        # 全レイヤー収集用の変数を初期化
        context_cache: Optional[ContextCache] = None
        token_embeds_combined: Optional[torch.Tensor] = None

        if collect_all_layers:
            # Phase 2用キャッシュ形式で初期化
            num_layers = self.model.num_layers
            context_dim = self.model.context_dim
            token_input_all_layers = getattr(self.model, 'token_input_all_layers', True)

            if hasattr(self.model, 'context_block'):
                context_dims = getattr(self.model.context_block, 'context_dims', None)
            else:
                context_dims = None

            if token_input_all_layers or context_dims is None:
                context_cache = torch.zeros(
                    num_layers, num_tokens, context_dim,
                    device=self.device, dtype=torch.float32
                )
            else:
                context_cache = [
                    torch.zeros(num_tokens, context_dims[i + 1], device=self.device, dtype=torch.float32)
                    for i in range(num_layers)
                ]

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
                        if isinstance(context_cache, list):
                            context_cache[layer_idx][i] = ctx_out.squeeze(0)
                        else:
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
    ) -> tuple:
        """
        勾配累積付き並列処理（メモリ効率版）

        バッチごとに勾配を計算・累積し、最後にまとめてパラメータ更新。
        これにより、大規模データでもOOMを回避できる。

        Returns:
            (contexts, total_loss, cvfp_loss, diversity_loss)
        """
        num_tokens = len(token_embeds)
        batch_size = self.config.phase1_batch_size
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)
        num_batches = (num_tokens + batch_size - 1) // batch_size
        context_noise = self.config.phase1_context_noise

        # デバイス判定用（文字列またはtorch.device両対応）
        is_cuda = str(self.device) == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda')

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

            if is_cuda:
                torch.cuda.empty_cache()

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

    def _forward_parallel(self, token_embeds: torch.Tensor, previous_contexts: torch.Tensor) -> torch.Tensor:
        """並列処理（Iteration 1+用）- 推論専用（勾配なし）"""
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
        with torch.no_grad():
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

    def _train_split_mode(self, data_provider, label: str = "Train") -> torch.Tensor:
        """
        分割訓練モード: 各splitを順番に訓練

        Args:
            data_provider: MemoryDataProvider（get_split_token_ids対応）
            label: ラベル

        Returns:
            結合されたコンテキスト [num_tokens, context_dim]
        """
        num_splits = self.config.num_context_splits
        split_context_dim = self.config.split_context_dim

        print_flush(f"\n{'='*70}")
        print_flush(f"PHASE 1: 分割訓練モード - {label}")
        print_flush(f"{'='*70}")
        print_flush(f"  num_context_splits: {num_splits}")
        print_flush(f"  split_context_dim: {split_context_dim}")

        # 各splitのコンテキストを保存
        all_split_contexts = []

        for split_id in range(num_splits):
            print_flush(f"\n--- Split {split_id + 1}/{num_splits} ---")

            # このsplit用のトークンを取得
            split_token_ids = data_provider.get_split_token_ids(split_id, num_splits, self.device)
            num_tokens = len(split_token_ids)
            print_flush(f"  Tokens for split {split_id}: {num_tokens:,}")

            # このsplit用のパラメータのみを学習
            split_block = self.model.context_block.blocks[split_id]
            split_params = list(split_block.parameters())
            optimizer = torch.optim.Adam(split_params, lr=self.config.phase1_learning_rate)
            print_flush(f"  Split ContextBlock params: {sum(p.numel() for p in split_params):,}")

            # トークン埋め込み（1回のみ計算）
            is_cuda = str(self.device) == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda')

            with torch.no_grad():
                token_embeds_gpu = self.model.token_embedding(split_token_ids.unsqueeze(0).to(self.device))
                token_embeds_gpu = self.model.embed_norm(token_embeds_gpu).squeeze(0)
                token_embeds = token_embeds_gpu.cpu()
                del token_embeds_gpu
                if is_cuda:
                    torch.cuda.empty_cache()

            previous_contexts: Optional[torch.Tensor] = None
            final_convergence_rate = 0.0

            for iteration in range(self.config.phase1_max_iterations):
                import time
                start_time = time.time()

                if iteration == 0:
                    # Iteration 0: シーケンシャル（学習なし）
                    token_embeds_gpu = token_embeds.to(self.device)
                    contexts = self._forward_sequential_split(
                        token_embeds_gpu, None, split_block, split_context_dim
                    )
                    previous_contexts = contexts.detach().cpu()
                    del token_embeds_gpu, contexts
                    if is_cuda:
                        torch.cuda.empty_cache()
                    elapsed = time.time() - start_time
                    print_flush(f"  Iteration 1/{self.config.phase1_max_iterations}: シーケンシャル [{elapsed:.2f}s]")
                    continue

                # Iteration 1+: 勾配累積付き並列処理
                assert previous_contexts is not None
                token_embeds_gpu = token_embeds.to(self.device)
                previous_contexts_gpu = previous_contexts.to(self.device)

                contexts, total_loss, cvfp_loss, diversity_loss = self._forward_parallel_split(
                    token_embeds_gpu, previous_contexts_gpu, optimizer,
                    split_block, split_context_dim
                )

                # 収束率
                convergence_rate = self._compute_convergence_rate(contexts, previous_contexts_gpu, num_tokens)

                # CPUに保存
                previous_contexts = contexts.detach().cpu()
                final_convergence_rate = convergence_rate

                del token_embeds_gpu, previous_contexts_gpu, contexts
                if is_cuda:
                    torch.cuda.empty_cache()

                elapsed = time.time() - start_time
                print_flush(
                    f"  Iteration {iteration+1}/{self.config.phase1_max_iterations}: "
                    f"収束={convergence_rate*100:.1f}% | "
                    f"Loss={total_loss:.6f} [{elapsed:.2f}s]"
                )

                # 早期停止
                min_iterations = getattr(self.config, 'phase1_min_iterations', 3)
                if iteration + 1 >= min_iterations and convergence_rate >= self.config.phase1_min_converged_ratio:
                    print_flush("  → Early stopping")
                    break

            print_flush(f"  Split {split_id} 完了: {int(final_convergence_rate * num_tokens)}/{num_tokens} トークン収束")

            # このsplitのコンテキストを保存
            assert previous_contexts is not None
            all_split_contexts.append(previous_contexts)

        # 全データに対して推論モードでコンテキストを計算（結合用）
        print_flush("\n--- 全データで推論（結合用） ---")
        all_token_ids = data_provider.get_all_train_tokens(self.device)
        contexts = self._inference_all_splits(all_token_ids, data_provider)

        print_flush(f"\nPhase 1 分割訓練完了: {len(contexts)} トークン\n")
        return contexts

    def _forward_sequential_split(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: Optional[torch.Tensor],
        split_block,
        split_context_dim: int
    ) -> torch.Tensor:
        """分割ブロック用シーケンシャル処理"""
        num_tokens = len(token_embeds)
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)

        contexts = torch.zeros(num_tokens, split_context_dim, device=self.device)

        if previous_contexts is None:
            context = torch.zeros(1, split_context_dim, device=self.device)
        else:
            context = previous_contexts[-1].unsqueeze(0).detach()

        token_history = [torch.zeros(self.model.embed_dim, device=self.device)
                         for _ in range(num_input_tokens - 1)]

        with torch.no_grad():
            for i, token_embed in enumerate(token_embeds):
                token_history.append(token_embed)
                combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)

                context = split_block(context, combined_tokens.unsqueeze(0))
                contexts[i] = context.squeeze(0)

                if len(token_history) > num_input_tokens:
                    token_history = token_history[-num_input_tokens:]

        return contexts

    def _forward_parallel_split(
        self,
        token_embeds: torch.Tensor,
        previous_contexts: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        split_block,
        split_context_dim: int
    ) -> tuple:
        """分割ブロック用並列処理（勾配累積）"""
        num_tokens = len(token_embeds)
        batch_size = self.config.phase1_batch_size
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)
        num_batches = (num_tokens + batch_size - 1) // batch_size
        context_noise = self.config.phase1_context_noise

        is_cuda = str(self.device) == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda')

        optimizer.zero_grad()

        all_contexts_cpu = []
        total_cvfp_loss = 0.0
        total_diversity_loss = 0.0
        total_loss_sum = 0.0

        last_context = previous_contexts[-1].detach()

        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)
            current_batch_size = end_idx - start_idx

            if start_idx == 0:
                batch_contexts = torch.zeros(current_batch_size, split_context_dim, device=self.device)
                batch_contexts[0] = last_context
                if current_batch_size > 1:
                    batch_contexts[1:] = previous_contexts[:end_idx-1].detach()
            else:
                batch_contexts = previous_contexts[start_idx-1:end_idx-1].detach().clone()

            if context_noise > 0 and self.model.training:
                noise = torch.randn_like(batch_contexts) * context_noise
                batch_contexts = batch_contexts + noise

            batch_combined = self._build_combined_tokens_batch(
                token_embeds, num_input_tokens, start_idx, end_idx
            )

            batch_output = split_block(batch_contexts, batch_combined)

            batch_prev = previous_contexts[start_idx:end_idx].detach()
            cvfp_loss = F.mse_loss(batch_output, batch_prev)
            diversity_loss = self._compute_diversity_loss(batch_output)
            batch_loss = (1 - self.config.dist_reg_weight) * cvfp_loss + self.config.dist_reg_weight * diversity_loss

            scaled_loss = batch_loss / num_batches

            if not torch.isnan(scaled_loss) and not torch.isinf(scaled_loss):
                scaled_loss.backward()

            all_contexts_cpu.append(batch_output.detach().cpu())

            total_cvfp_loss += cvfp_loss.item() * current_batch_size
            total_diversity_loss += diversity_loss.item() * current_batch_size
            total_loss_sum += batch_loss.item() * current_batch_size

            del batch_combined, batch_output, batch_loss, scaled_loss, batch_contexts, batch_prev

            if is_cuda:
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(split_block.parameters(), max_norm=self.config.phase1_gradient_clip)
        optimizer.step()

        contexts = torch.cat(all_contexts_cpu, dim=0).to(self.device)
        del all_contexts_cpu

        avg_cvfp_loss = total_cvfp_loss / num_tokens
        avg_diversity_loss = total_diversity_loss / num_tokens
        avg_total_loss = total_loss_sum / num_tokens

        return contexts, avg_total_loss, avg_cvfp_loss, avg_diversity_loss

    def _inference_all_splits(self, token_ids: torch.Tensor, data_provider) -> torch.Tensor:
        """
        全splitを使って全データのコンテキストを計算（結合モード）

        推論時は全splitを実行して出力を連結
        """
        self.model.eval()
        num_tokens = len(token_ids)
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)

        with torch.no_grad():
            token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(self.device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)

        # 結合されたコンテキストを計算
        contexts = torch.zeros(num_tokens, self.model.context_dim, device=self.device)
        context = torch.zeros(1, self.model.context_dim, device=self.device)

        token_history = [torch.zeros(self.model.embed_dim, device=self.device)
                         for _ in range(num_input_tokens - 1)]

        with torch.no_grad():
            for i, token_embed in enumerate(token_embeds):
                token_history.append(token_embed)
                combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)

                # SplitContextBlock.forward (split_id=None) で全splitを実行して連結
                context = self.model.context_block(context, combined_tokens.unsqueeze(0))
                contexts[i] = context.squeeze(0)

                if len(token_history) > num_input_tokens:
                    token_history = token_history[-num_input_tokens:]

        self.model.train()
        return contexts
