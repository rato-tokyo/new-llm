"""
CVFP Network - 全トークン系列の処理

責任:
- トークン系列の順次処理
- コンテキスト引き継ぎ管理
- 収束判定
"""

import torch
import time


class Network:
    """
    CVFP Network - 全トークン系列を処理

    旧実装の_process_tokens()に相当。
    Layerを使ってトークンを順次処理し、収束判定を行う。
    """

    def __init__(self, layer, convergence_threshold):
        """
        Args:
            layer: Layerインスタンス
            convergence_threshold: 収束判定のMSE閾値
        """
        self.layer = layer
        self.convergence_threshold = convergence_threshold

        # 収束状態
        self.previous_contexts = None
        self.num_converged_tokens = 0
        self.num_tokens = 0

    def forward_all(self, token_embeds, device, optimizer=None, target_contexts=None, verbose=True):
        """
        全トークンを処理（バッチ最適化版）

        Args:
            token_embeds: [num_tokens, embed_dim]
            device: torch device
            optimizer: Optimizerインスタンス（訓練時のみ）
            target_contexts: 固定点目標（Iteration 0の出力） [num_tokens, context_dim]
            verbose: 進捗表示するか

        Returns:
            contexts: [num_tokens, context_dim]
        """
        num_tokens = len(token_embeds)
        self.num_tokens = num_tokens

        # 大量データ警告
        if verbose and num_tokens > 10000:
            print(f"⚠️ Processing {num_tokens:,} tokens - this may take several minutes", flush=True)

        # CRITICAL: イテレーション間でコンテキストを引き継ぐ
        if self.previous_contexts is None:
            # 初回イテレーション: ゼロ初期化
            context = torch.zeros(1, self.layer.model.context_dim, device=device)
        else:
            # 前イテレーションの最終コンテキストから開始
            context = self.previous_contexts[-1].unsqueeze(0).detach()

        # メモリ最適化: 事前確保（勾配保持）
        contexts = torch.zeros(num_tokens, self.layer.model.context_dim, device=device, requires_grad=True if optimizer is not None else False)

        start_time = time.time()

        # 全トークンを順次処理（勾配グラフを保持）
        context_list = []
        for t, token_embed in enumerate(token_embeds):
            # 進捗表示（100トークンごと）
            if verbose and t > 0 and t % 100 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = t / elapsed if elapsed > 0 else 0
                remaining_tokens = num_tokens - t
                eta_seconds = remaining_tokens / tokens_per_sec if tokens_per_sec > 0 else 0

                progress = (t / num_tokens) * 100
                print(
                    f"  Progress: {t:,}/{num_tokens:,} ({progress:.1f}%) | "
                    f"Speed: {tokens_per_sec:.1f} tok/s | "
                    f"ETA: {eta_seconds:.0f}s",
                    flush=True
                )

            if optimizer is not None:
                # 訓練モード: トークン間勾配遮断、勾配グラフ保持
                context = self.layer.forward(
                    token_embed.unsqueeze(0),
                    context.detach(),  # トークン間で勾配遮断
                    detach_context=False
                )
                context_list.append(context.squeeze(0))
            else:
                # 評価モード: 最適化なし
                with torch.no_grad():
                    context = self.layer.forward(
                        token_embed.unsqueeze(0),
                        context,
                        detach_context=False
                    )
                context_list.append(context.squeeze(0))

        # contextsテンソルを構築
        contexts = torch.stack(context_list)

        # バッチ最適化（訓練時のみ）
        if optimizer is not None and target_contexts is not None:
            total_loss, cvfp_loss, diversity_loss = optimizer.compute_batch_loss(
                contexts, target_contexts
            )
            optimizer.optimize_batch(total_loss)

        return contexts

    def update_convergence(self, current_contexts):
        """
        収束状態を更新

        Args:
            current_contexts: [num_tokens, context_dim]
        """
        if self.previous_contexts is None:
            # 初回イテレーション: 保存のみ
            self.previous_contexts = current_contexts.detach()
            self.num_converged_tokens = 0
            return

        # 前回との差分を計算
        with torch.no_grad():
            token_losses = ((current_contexts - self.previous_contexts) ** 2).mean(dim=1)
            converged_tokens = token_losses < self.convergence_threshold
            self.num_converged_tokens = converged_tokens.sum().item()

        # 収束判定用の前回値を更新（Phase1Trainerのtarget_contextsとは別物）
        self.previous_contexts = current_contexts.detach()

    def is_converged(self, min_converged_ratio):
        """
        収束判定

        Args:
            min_converged_ratio: 必要な収束率閾値

        Returns:
            converged: 収束したかどうか
        """
        if self.num_tokens == 0:
            return False

        convergence_rate = self.num_converged_tokens / self.num_tokens
        return convergence_rate >= min_converged_ratio

    def get_convergence_rate(self):
        """
        収束率を取得

        Returns:
            convergence_rate: 0.0~1.0
        """
        if self.num_tokens == 0:
            return 0.0

        return self.num_converged_tokens / self.num_tokens

    def reset(self):
        """
        状態をリセット（新しい訓練/評価開始時）
        """
        self.previous_contexts = None
        self.num_converged_tokens = 0
        self.num_tokens = 0
