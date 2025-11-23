"""
Phase 1 Trainer: CVFP固定点学習のトレーナークラス

責任分離:
- モデル: アーキテクチャと順伝播のみ
- Trainer: 訓練ループ、収束判定、状態管理、最適化
"""

import torch


class Phase1Trainer:
    """
    Phase 1 (CVFP固定点学習) のトレーナー

    モデルから訓練ロジックを分離し、クリーンな設計を実現。
    """

    def __init__(
        self,
        model,
        max_iterations,
        convergence_threshold,
        min_converged_ratio,
        learning_rate,
        dist_reg_weight
    ):
        """
        Args:
            model: 言語モデル (enable_cvfp_learning=True必須)
            max_iterations: 最大イテレーション数
            convergence_threshold: 収束判定のMSE閾値
            min_converged_ratio: 早期停止の収束率閾値
            learning_rate: 学習率
            dist_reg_weight: 分布正則化の重み
        """
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_converged_ratio = min_converged_ratio
        self.learning_rate = learning_rate
        self.dist_reg_weight = dist_reg_weight

        # 収束判定用の状態（Trainerが管理）
        self.previous_contexts = None
        self.num_tokens = 0
        self.num_converged_tokens = 0
        self.current_iteration = 0

        # Optimizer（訓練時のみ作成）
        self.optimizer = None

        # 最後のCVFPロスとDistロスを記録（ログ出力用）
        self._last_cvfp_loss = 0.0
        self._last_dist_loss = 0.0

    def train(self, token_ids, device, label="Train"):
        """
        訓練実行（最適化あり）

        Args:
            token_ids: 入力トークンID [num_tokens]
            device: torch デバイス
            label: ログ用ラベル

        Returns:
            final_contexts: 収束したコンテキストベクトル [num_tokens, context_dim]
        """
        # 訓練モードに設定
        self.model.train()

        # Optimizerをセットアップ（初回のみ）
        if self.optimizer is None:
            context_params = [
                p for name, p in self.model.named_parameters()
                if 'token_output' not in name and 'token_embedding' not in name
            ]
            self.optimizer = torch.optim.Adam(context_params, lr=self.learning_rate)

        # 訓練実行
        return self._run(token_ids, device, label, is_training=True)

    def evaluate(self, token_ids, device, label="Val"):
        """
        評価実行（最適化なし）

        Args:
            token_ids: 入力トークンID [num_tokens]
            device: torch デバイス
            label: ログ用ラベル

        Returns:
            final_contexts: 収束したコンテキストベクトル [num_tokens, context_dim]
        """
        # 評価モードに設定
        self.model.eval()

        # 評価実行
        return self._run(token_ids, device, label, is_training=False)

    def _run(self, token_ids, device, label, is_training):
        """
        訓練/評価の共通ロジック

        Args:
            token_ids: 入力トークンID [num_tokens]
            device: torch デバイス
            label: ログ用ラベル
            is_training: 訓練モードかどうか

        Returns:
            final_contexts: 収束したコンテキストベクトル [num_tokens, context_dim]
        """
        self._print_header(label)

        self.model.to(device)
        self.num_tokens = len(token_ids)

        # トークン埋め込みを計算（1回のみ）
        with torch.no_grad():
            token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)

        # 状態リセット
        self.previous_contexts = None
        self.num_converged_tokens = 0

        # 反復改善ループ
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration

            # イテレーション開始時の処理（CVFP状態はPhase1Trainerで管理）
            # reset_cvfp_state()は不要（previous_contextsで管理）

            # トークンごとの処理
            current_contexts = self._process_tokens(token_embeds, device, is_training)

            # 収束状態を更新
            if is_training:
                self._update_convergence_state(current_contexts)

            # ログ出力
            self._log_iteration(iteration)

            # Early stopping判定
            if is_training and self._is_converged() and iteration > 0:
                convergence_rate = self._get_convergence_rate()
                self._print_flush(f"  → Early stopping: 収束率 = {convergence_rate*100:.1f}%")
                break

        # 最終サマリー
        self._print_summary()

        return current_contexts

    def _process_tokens(self, token_embeds, device, is_training):
        """
        全トークンを処理

        Args:
            token_embeds: トークン埋め込み [num_tokens, embed_dim]
            device: torch デバイス
            is_training: 訓練モードかどうか

        Returns:
            current_contexts: コンテキストベクトル [num_tokens, context_dim]
        """
        import time

        total_tokens = len(token_embeds)

        # 大量データ警告
        if total_tokens > 10000:
            self._print_flush(f"⚠️ Processing {total_tokens:,} tokens - this may take several minutes")

        context = torch.zeros(1, self.model.context_dim, device=device)
        current_contexts = []

        start_time = time.time()

        for t, token_embed in enumerate(token_embeds):
            # 進捗表示（100トークンごと）
            if t > 0 and t % 100 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = t / elapsed if elapsed > 0 else 0
                remaining_tokens = total_tokens - t
                eta_seconds = remaining_tokens / tokens_per_sec if tokens_per_sec > 0 else 0

                progress = (t / total_tokens) * 100
                self._print_flush(
                    f"  Progress: {t:,}/{total_tokens:,} ({progress:.1f}%) | "
                    f"Speed: {tokens_per_sec:.1f} tok/s | "
                    f"ETA: {eta_seconds:.0f}s"
                )

            if is_training:
                # 訓練モード: 最適化あり
                context = self._train_one_token(
                    token_embed.unsqueeze(0),
                    context.detach() if t > 0 else context,
                    token_idx=t
                )
            else:
                # 評価モード: 最適化なし
                with torch.no_grad():
                    context = self.model._update_context_one_step(
                        token_embed.unsqueeze(0),
                        context
                    )

            current_contexts.append(context.detach())

        # 全コンテキストをスタック
        return torch.cat(current_contexts, dim=0)

    def _train_one_token(self, token_embed, context, token_idx):
        """
        1トークンの訓練（最適化実行）

        Args:
            token_embed: トークン埋め込み [1, embed_dim]
            context: 現在のコンテキスト [1, context_dim]
            token_idx: トークンインデックス（CVFP損失計算用）

        Returns:
            new_context: 更新されたコンテキスト [1, context_dim]
        """
        import torch.nn.functional as F

        # 順伝播
        new_context = self.model._update_context_one_step(token_embed, context)

        # CVFP損失を計算（前回イテレーションとの比較）
        if self.current_iteration > 0 and self.previous_contexts is not None:
            # 同じトークン位置の前回イテレーション出力と比較
            # previous_contextsはdetachされているので、requires_grad=Trueにする
            previous_token_context = self.previous_contexts[token_idx:token_idx+1].detach()
            cvfp_loss = F.mse_loss(new_context, previous_token_context)
        else:
            # 初回イテレーションはCVFP損失なし
            cvfp_loss = torch.tensor(0.0, device=new_context.device, requires_grad=True)

        # 分布正則化損失を現在のコンテキストから直接計算（勾配あり）
        if self.model.use_dist_reg:
            # N(0, 1)からの逸脱にペナルティ
            # new_contextは[1, context_dim]なので、次元0で平均・分散を計算
            context_mean = new_context.mean(dim=-1, keepdim=True)  # [1, 1]
            context_var = new_context.var(dim=-1, unbiased=False, keepdim=True)  # [1, 1]

            mean_penalty = (context_mean ** 2).mean()
            var_penalty = ((context_var - 1.0) ** 2).mean()
            dist_loss = mean_penalty + var_penalty
        else:
            dist_loss = torch.tensor(0.0, device=new_context.device, requires_grad=True)

        # 総合損失を計算
        total_loss = (1 - self.dist_reg_weight) * cvfp_loss + self.dist_reg_weight * dist_loss

        # CVFPロスとDistロスを記録（ログ出力用）
        cvfp_loss_val = cvfp_loss.item() if isinstance(cvfp_loss, torch.Tensor) else cvfp_loss
        dist_loss_val = dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss
        self._last_cvfp_loss = cvfp_loss_val
        self._last_dist_loss = dist_loss_val

        # 損失が0より大きい場合のみ最適化
        if total_loss.item() > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return new_context

    def _update_convergence_state(self, current_contexts):
        """
        収束状態を更新

        Args:
            current_contexts: 現在のコンテキストベクトル [num_tokens, context_dim]
        """
        if self.current_iteration == 0 or self.previous_contexts is None:
            # 初回イテレーションは前回コンテキストを保存するのみ
            self.previous_contexts = current_contexts.detach()
            self.num_converged_tokens = 0
            return

        # 前回との差分を計算
        with torch.no_grad():
            token_losses = ((current_contexts - self.previous_contexts) ** 2).mean(dim=1)
            converged_tokens = token_losses < self.convergence_threshold
            self.num_converged_tokens = converged_tokens.sum().item()

        # 前回コンテキストを更新
        self.previous_contexts = current_contexts.detach()

    def _is_converged(self):
        """
        収束判定

        Returns:
            converged: 収束したかどうか（bool）
        """
        if self.current_iteration == 0 or self.num_tokens == 0:
            return False

        convergence_rate = self.num_converged_tokens / self.num_tokens
        return convergence_rate >= self.min_converged_ratio

    def _get_convergence_rate(self):
        """
        現在の収束率を取得

        Returns:
            convergence_rate: 収束率 (0.0~1.0)
        """
        if self.num_tokens == 0:
            return 0.0
        return self.num_converged_tokens / self.num_tokens

    def _print_header(self, label):
        """ヘッダーを出力"""
        self._print_flush(f"\n{'='*70}")
        self._print_flush(f"PHASE 1: 固定点コンテキスト学習 (CVFP){' - ' + label if label else ''}")
        self._print_flush(f"{'='*70}")

    def _log_iteration(self, iteration):
        """イテレーションログを出力"""
        if iteration == 0:
            self._print_flush(f"Iteration 1/{self.max_iterations}: 順伝播のみ（コンテキスト保存）")
        else:
            convergence_rate = self._get_convergence_rate()

            # 最後のトークンで記録された損失を使用
            cvfp_loss = self._last_cvfp_loss
            dist_loss = self._last_dist_loss

            self._print_flush(
                f"Iteration {iteration+1}/{self.max_iterations}: "
                f"収束={convergence_rate*100:.1f}% | "
                f"CVFP loss={cvfp_loss:.6f} | "
                f"Dist loss={dist_loss:.6f}"
            )

    def _print_summary(self):
        """最終サマリーを出力"""
        self._print_flush(f"\nPhase 1 完了: {self.num_converged_tokens}/{self.num_tokens} トークンが収束\n")

    def _print_flush(self, msg):
        """リアルタイム出力"""
        print(msg, flush=True)
