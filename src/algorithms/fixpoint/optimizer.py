"""
CVFP Optimizer - 損失計算と最適化（バッチ処理版）

責任:
- CVFP損失計算（固定点への収束）
- 多様性損失計算（バッチ全体）
- 総合損失計算と最適化実行
"""

import torch
import torch.nn.functional as F


class Optimizer:
    """
    CVFP最適化器（バッチ処理版）

    トークンごとではなく、全トークン処理後に一括最適化を実行。
    これにより optimizer.step() 呼び出し回数を 64,000回 → 10回 に削減。
    """

    def __init__(
        self,
        model,
        optimizer,
        dist_reg_weight,
        ema_momentum=0.99
    ):
        """
        Args:
            model: NewLLMResidualモデル
            optimizer: torch.optim.Optimizer
            dist_reg_weight: 多様性正則化の重み
            ema_momentum: EMA係数（デフォルト: 0.99）
        """
        self.model = model
        self.optimizer = optimizer
        self.dist_reg_weight = dist_reg_weight
        self.ema_momentum = ema_momentum

        # 現在のイテレーション番号
        self.current_iteration = 0

        # 前回イテレーションのコンテキスト（CVFP損失計算用）
        self.target_contexts = None

        # ログ用
        self.last_cvfp_loss = 0.0
        self.last_diversity_loss = 0.0

    def compute_batch_loss(self, contexts, target_contexts):
        """
        全トークンの損失を一括計算

        Args:
            contexts: 現在のコンテキスト [num_tokens, context_dim]
            target_contexts: 固定点目標（Iteration 0の出力） [num_tokens, context_dim]

        Returns:
            total_loss: 総合損失（勾配あり）
            cvfp_loss: CVFP損失値（float）
            diversity_loss: 多様性損失値（float）
        """
        # CVFP損失: 固定点目標との MSE
        cvfp_loss = F.mse_loss(contexts, target_contexts)

        # 多様性損失: 全トークンの平均からの偏差（負の損失で最大化）
        context_mean = contexts.mean(dim=0)  # [context_dim]
        deviation = contexts - context_mean  # [num_tokens, context_dim]
        diversity_loss = -torch.norm(deviation, p=2) / len(contexts)

        # 総合損失
        total_loss = (
            (1 - self.dist_reg_weight) * cvfp_loss +
            self.dist_reg_weight * diversity_loss
        )

        # ログ用に記録
        self.last_cvfp_loss = cvfp_loss.item()
        self.last_diversity_loss = diversity_loss.item()

        return total_loss, cvfp_loss.item(), diversity_loss.item()

    def optimize_batch(self, total_loss):
        """
        バッチ最適化実行

        Args:
            total_loss: 総合損失テンソル
        """
        # 最適化実行（NaN/Infチェック付き）
        if not torch.isnan(total_loss) and not torch.isinf(total_loss):
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    def start_new_iteration(self, iteration, target_contexts):
        """
        新しいイテレーション開始

        Args:
            iteration: イテレーション番号
            target_contexts: 固定点目標（Iteration 0の出力） [num_tokens, context_dim]
        """
        self.current_iteration = iteration
        self.target_contexts = target_contexts

    def reset_ema_for_sequence(self):
        """
        シーケンス開始時にEMA統計をリセット

        注意: バッチ処理版では不要だが、phase1.pyとの互換性のため残す
        """
        pass

    def get_last_losses(self):
        """
        最後の損失値を取得（ログ用）

        Returns:
            (cvfp_loss, diversity_loss): タプル
        """
        return self.last_cvfp_loss, self.last_diversity_loss
