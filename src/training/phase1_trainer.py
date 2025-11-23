"""
Phase 1 Trainer: CVFP固定点学習のトレーナークラス

多様性確保アプローチ:
- LayerNorm: 値の爆発を防止し、スケールを正規化
- 固定次元割り当て: 各トークンに固定の次元セットを割り当て
- 使用頻度追跡: 使用頻度が低い次元を優先的に活性化

責任分離:
- モデル: アーキテクチャと順伝播のみ
- Trainer: 訓練ループ、収束判定、状態管理、最適化、多様性制約
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

        # 最後のCVFPロスと多様性ロスを記録（ログ出力用）
        self._last_cvfp_loss = 0.0
        self._last_diversity_loss = 0.0

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

        # 次元統計のリセット（各イテレーションごとに初期化）
        if is_training:
            self._dim_stats = torch.zeros(self.model.context_dim, device=device)

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
        1トークンの訓練（固定次元割り当て + 分散最大化）

        Args:
            token_embed: トークン埋め込み [1, embed_dim]
            context: 現在のコンテキスト [1, context_dim]
            token_idx: トークンインデックス

        Returns:
            new_context: 更新されたコンテキスト [1, context_dim]
        """
        import torch.nn.functional as F

        # 順伝播
        new_context = self.model._update_context_one_step(token_embed, context)

        # ========== 固定次元割り当て法 ==========
        # 各トークンに固定の次元セットを割り当てる
        context_dim = self.model.context_dim
        dims_per_token = max(4, context_dim // 4)  # 各トークンが使用する次元数

        # トークンIDのハッシュで使用する次元を決定
        token_hash = hash(token_idx) % context_dim
        assigned_dims = [(token_hash + i) % context_dim for i in range(dims_per_token)]

        # 割り当てマスクを作成
        mask = torch.zeros_like(new_context)
        mask[0, assigned_dims] = 1.0

        # 割り当てられた次元のみを強化、他は抑制
        masked_context = new_context * mask
        suppression_loss = (new_context * (1 - mask)).abs().mean()

        # ========== 次元間分散最大化 ==========
        if not hasattr(self, '_dim_stats'):
            # 各次元の使用統計を初期化
            self._dim_stats = torch.zeros(context_dim, device=new_context.device)

        # 次元の使用頻度を更新
        with torch.no_grad():
            self._dim_stats += mask.squeeze(0)

        # 使用頻度が低い次元を優先的に活性化
        dim_weights = 1.0 / (self._dim_stats + 1.0)  # 使用頻度の逆数
        weighted_loss = (dim_weights * new_context.abs().squeeze(0)).mean()

        # ========== 値のスケール制御 ==========
        # 値が爆発しないように制約
        scale_penalty = torch.relu(new_context.abs() - 5.0).mean()  # 5を超える値にペナルティ

        # CVFP損失を計算
        if self.current_iteration > 0 and self.previous_contexts is not None:
            previous_token_context = self.previous_contexts[token_idx:token_idx+1].detach()
            # CVFPも正規化して計算（スケール不変）
            cvfp_loss = F.mse_loss(
                F.normalize(new_context, p=2, dim=1),
                F.normalize(previous_token_context, p=2, dim=1)
            )
        else:
            cvfp_loss = torch.tensor(0.0, device=new_context.device, requires_grad=True)

        # 総合損失（スケール制御を重視）
        diversity_loss = suppression_loss + weighted_loss * 0.1 + scale_penalty * 2.0
        total_loss = (1 - self.dist_reg_weight) * cvfp_loss + self.dist_reg_weight * diversity_loss

        # 損失を記録
        self._last_cvfp_loss = cvfp_loss.item() if isinstance(cvfp_loss, torch.Tensor) else cvfp_loss
        self._last_diversity_loss = diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss

        # 勾配クリッピングを追加
        if total_loss.item() > 0 and not torch.isnan(total_loss):
            self.optimizer.zero_grad()
            total_loss.backward()
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            diversity_loss = self._last_diversity_loss

            self._print_flush(
                f"Iteration {iteration+1}/{self.max_iterations}: "
                f"収束={convergence_rate*100:.1f}% | "
                f"CVFP={cvfp_loss:.6f} | "
                f"Diversity={diversity_loss:.6f}"
            )

    def _print_summary(self):
        """最終サマリーを出力"""
        self._print_flush(f"\nPhase 1 完了: {self.num_converged_tokens}/{self.num_tokens} トークンが収束\n")

    def _print_flush(self, msg):
        """リアルタイム出力"""
        print(msg, flush=True)
