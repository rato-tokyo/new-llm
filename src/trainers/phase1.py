"""
Phase 1 Trainer: CVFP固定点学習のトレーナークラス（リファクタリング版）

責任分離による設計:
- Phase1Trainer: イテレーションループ制御のみ
- cvfp.Layer: 1トークンの基本処理
- cvfp.Network: 全トークン系列の処理
- cvfp.Optimizer: 損失計算と最適化
"""

import torch
from src.algorithms.fixpoint import Layer, Network, Optimizer


class Phase1Trainer:
    """
    Phase 1 (CVFP固定点学習) のトレーナー

    責任:
    - イテレーションループ制御
    - トレーニング/評価モードの切り替え
    - ログ出力

    使い方:
        trainer = Phase1Trainer(model, ...)
        contexts = trainer.train(token_ids, device)
        val_contexts = trainer.evaluate(val_token_ids, device)
    """

    def __init__(
        self,
        model,
        max_iterations,
        convergence_threshold,
        min_converged_ratio,
        learning_rate,
        dist_reg_weight,
        ema_momentum=0.99
    ):
        """
        Args:
            model: 言語モデル
            max_iterations: 最大イテレーション数
            convergence_threshold: 収束判定のMSE閾値
            min_converged_ratio: 早期停止の収束率閾値
            learning_rate: 学習率
            dist_reg_weight: 多様性正則化の重み
            ema_momentum: EMA係数（デフォルト: 0.99）
        """
        self.model = model
        self.max_iterations = max_iterations
        self.min_converged_ratio = min_converged_ratio
        self.learning_rate = learning_rate

        # CVFP Layer（1トークンの基本処理）
        self.layer = Layer(model)

        # CVFP Network（全トークン系列の処理）
        self.network = Network(self.layer, convergence_threshold)

        # CVFP Optimizer（訓練時のみ作成）
        self.cvfp_optimizer = None
        self.dist_reg_weight = dist_reg_weight
        self.ema_momentum = ema_momentum

        # Torch Optimizer（訓練時のみ作成）
        self.torch_optimizer = None

        # 収束状態（test.pyとの互換性のため公開）
        self.num_converged_tokens = 0
        self.num_tokens = 0

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
        if self.torch_optimizer is None:
            context_params = [
                p for name, p in self.model.named_parameters()
                if 'token_output' not in name and 'token_embedding' not in name
            ]
            self.torch_optimizer = torch.optim.Adam(context_params, lr=self.learning_rate)

            # CVFP Optimizer作成
            self.cvfp_optimizer = Optimizer(
                model=self.model,
                optimizer=self.torch_optimizer,
                dist_reg_weight=self.dist_reg_weight,
                ema_momentum=self.ema_momentum
            )

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

        # Networkをリセット
        self.network.reset()

        # CVFP Optimizerのシーケンスリセット（訓練時のみ）
        if is_training:
            self.cvfp_optimizer.reset_ema_for_sequence()

        # Iteration 0の出力を保存（これが固定点の目標）
        target_contexts = None

        # イテレーションループ
        for iteration in range(self.max_iterations):
            # CVFP Optimizerに新しいイテレーション開始を通知（訓練時、2回目以降のみ）
            if is_training and iteration > 0:
                self.cvfp_optimizer.start_new_iteration(
                    iteration,
                    target_contexts  # 固定された目標を渡す
                )

            # 全トークンを処理
            contexts = self.network.forward_all(
                token_embeds,
                device,
                optimizer=self.cvfp_optimizer if is_training else None,
                target_contexts=target_contexts if (is_training and iteration > 0) else None,
                verbose=(iteration == 0)  # 初回のみ進捗表示
            )

            # Iteration 0の出力を保存（以降は変更しない）
            if iteration == 0:
                target_contexts = contexts.detach().clone()

            # 収束状態を更新（収束判定用 - これは毎回更新してよい）
            self.network.update_convergence(contexts)

            # 公開属性を更新（test.pyとの互換性）
            self.num_converged_tokens = self.network.num_converged_tokens

            # ログ出力
            self._log_iteration(iteration, is_training)

            # Early stopping判定（訓練時のみ、2回目以降のイテレーション）
            if is_training and iteration > 0:
                if self.network.is_converged(self.min_converged_ratio):
                    convergence_rate = self.network.get_convergence_rate()
                    self._print_flush(f"  → Early stopping: 収束率 = {convergence_rate*100:.1f}%")
                    break

            # GPU MEMORY OPTIMIZATION: イテレーション間でメモリ断片化防止
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # 最終サマリー
        self._print_summary()

        return contexts

    def _print_header(self, label):
        """ヘッダーを出力"""
        self._print_flush(f"\n{'='*70}")
        self._print_flush(f"PHASE 1: 固定点コンテキスト学習 (CVFP){' - ' + label if label else ''}")
        self._print_flush(f"{'='*70}")

    def _log_iteration(self, iteration, is_training):
        """イテレーションログを出力"""
        convergence_rate = self.network.get_convergence_rate()

        if iteration == 0:
            self._print_flush(f"Iteration 1/{self.max_iterations}: 順伝播のみ（コンテキスト保存）")
        else:
            if is_training:
                # 訓練時: 損失情報を表示
                cvfp_loss, diversity_loss = self.cvfp_optimizer.get_last_losses()
                self._print_flush(
                    f"Iteration {iteration+1}/{self.max_iterations}: "
                    f"収束={convergence_rate*100:.1f}% | "
                    f"CVFP={cvfp_loss:.6f} | "
                    f"Diversity={diversity_loss:.6f}"
                )
            else:
                # 検証時: 収束率のみ表示
                self._print_flush(
                    f"Iteration {iteration+1}/{self.max_iterations}: "
                    f"収束={convergence_rate*100:.1f}%"
                )

    def _print_summary(self):
        """最終サマリーを出力"""
        self._print_flush(f"\nPhase 1 完了: {self.num_converged_tokens}/{self.num_tokens} トークンが収束\n")

    def _print_flush(self, msg):
        """リアルタイム出力"""
        print(msg, flush=True)
