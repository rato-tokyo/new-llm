"""
Diversity Regularization方式の比較実験

方式1: 共分散行列EMA (Covariance Matrix EMA)
方式2: 次元ごとの分散追跡 (Per-Dimension Variance Tracking)

小規模データで性能を比較し、最適な方式を決定する。
"""

import torch
import torch.nn.functional as F
import sys
import time
from datetime import datetime

sys.path.insert(0, '.')

from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.evaluation.metrics import analyze_fixed_points


class Phase1TrainerVariance:
    """方式2: 次元ごとの分散追跡（指数平均的）"""

    def __init__(self, model, max_iterations, convergence_threshold,
                 min_converged_ratio, learning_rate, dist_reg_weight,
                 ema_momentum=0.99):
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_converged_ratio = min_converged_ratio
        self.learning_rate = learning_rate
        self.dist_reg_weight = dist_reg_weight
        self.ema_momentum = ema_momentum

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )

        # EMA統計量（各次元ごと）
        self.context_mean_ema = None  # [context_dim]
        self.context_var_ema = None   # [context_dim]

        # 訓練状態
        self.previous_contexts = None
        self.num_converged_tokens = 0
        self.current_iteration = 0
        self.train_convergence_rate = 0.0
        self.num_tokens = 0

        # 損失記録
        self._last_cvfp_loss = 0.0
        self._last_diversity_loss = 0.0

    def train(self, token_ids, device, label="Train"):
        """訓練実行"""
        self.model.train()
        return self._run(token_ids, device, label, is_training=True)

    def evaluate(self, token_ids, device, label="Val"):
        """評価実行"""
        self.model.eval()
        return self._run(token_ids, device, label, is_training=False)

    def _run(self, token_ids, device, label, is_training):
        """訓練/評価の共通ロジック"""
        self.model.to(device)
        self.num_tokens = len(token_ids)

        # トークン埋め込み
        with torch.no_grad():
            token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)

        # 状態リセット
        self.previous_contexts = None
        self.num_converged_tokens = 0

        # 反復改善ループ
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            current_contexts = self._process_tokens(token_embeds, device, is_training)

            if is_training:
                self._update_convergence_state(current_contexts)

            # Early stopping
            if is_training and self._is_converged() and iteration > 0:
                break

        return current_contexts

    def _process_tokens(self, token_embeds, device, is_training):
        """全トークンを処理"""
        context = torch.zeros(1, self.model.context_dim, device=device)
        current_contexts = []

        for t, token_embed in enumerate(token_embeds):
            if is_training:
                context = self._train_one_token(
                    token_embed.unsqueeze(0),
                    context.detach() if t > 0 else context,
                    token_idx=t
                )
            else:
                with torch.no_grad():
                    context = self.model._update_context_one_step(
                        token_embed.unsqueeze(0),
                        context
                    )

            current_contexts.append(context.detach())

        return torch.cat(current_contexts, dim=0)

    def _train_one_token(self, token_embed, context, token_idx):
        """1トークンの訓練（次元ごとの分散追跡版）"""
        # 順伝播
        new_context = self.model._update_context_one_step(token_embed, context)

        # CVFP損失
        if self.current_iteration > 0 and self.previous_contexts is not None:
            previous_token_context = self.previous_contexts[token_idx:token_idx+1].detach()
            cvfp_loss = F.mse_loss(
                F.normalize(new_context, p=2, dim=1),
                F.normalize(previous_token_context, p=2, dim=1)
            )
        else:
            cvfp_loss = torch.tensor(0.0, device=new_context.device, requires_grad=True)

        # 多様性損失（次元ごとの分散）
        new_context_flat = new_context.squeeze(0)  # [context_dim]

        if self.context_mean_ema is None:
            # 初期化
            self.context_mean_ema = new_context_flat.detach()
            self.context_var_ema = torch.ones_like(new_context_flat)
            diversity_loss = torch.tensor(0.0, device=new_context.device, requires_grad=True)
        else:
            # EMA更新（勾配計算のため、更新前の値を使用）
            old_mean = self.context_mean_ema.detach()
            old_var = self.context_var_ema.detach()

            # 偏差
            deviation = new_context_flat - old_mean

            # 分散が低い = 多様性不足 = 高損失
            # 損失 = 1 / (分散の平均 + epsilon)
            diversity_loss = 1.0 / (old_var.mean() + 1e-6)

            # EMA更新（学習後に実行）
            with torch.no_grad():
                self.context_mean_ema = (
                    self.ema_momentum * old_mean +
                    (1 - self.ema_momentum) * new_context_flat
                )
                deviation_sq = deviation ** 2
                self.context_var_ema = (
                    self.ema_momentum * old_var +
                    (1 - self.ema_momentum) * deviation_sq
                )

        # 総合損失
        total_loss = (
            (1 - self.dist_reg_weight) * cvfp_loss +
            self.dist_reg_weight * diversity_loss
        )

        # 損失記録
        self._last_cvfp_loss = cvfp_loss.item() if isinstance(cvfp_loss, torch.Tensor) else cvfp_loss
        self._last_diversity_loss = diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss

        # 最適化
        if total_loss.item() > 0 and not torch.isnan(total_loss):
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return new_context

    def _update_convergence_state(self, current_contexts):
        """収束状態を更新"""
        if self.current_iteration == 0 or self.previous_contexts is None:
            self.previous_contexts = current_contexts.detach()
            self.num_converged_tokens = 0
            return

        # 各トークンの収束判定
        mse_per_token = F.mse_loss(
            current_contexts,
            self.previous_contexts,
            reduction='none'
        ).mean(dim=1)

        converged_mask = mse_per_token < self.convergence_threshold
        self.num_converged_tokens = converged_mask.sum().item()
        self.train_convergence_rate = self.num_converged_tokens / self.num_tokens

        self.previous_contexts = current_contexts.detach()

    def _is_converged(self):
        """収束判定"""
        return self.train_convergence_rate >= self.min_converged_ratio

    def _get_convergence_rate(self):
        """収束率を取得"""
        return self.train_convergence_rate


class Phase1TrainerCovariance:
    """方式1: 共分散行列EMA（指数平均的）"""

    def __init__(self, model, max_iterations, convergence_threshold,
                 min_converged_ratio, learning_rate, dist_reg_weight,
                 ema_momentum=0.99):
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_converged_ratio = min_converged_ratio
        self.learning_rate = learning_rate
        self.dist_reg_weight = dist_reg_weight
        self.ema_momentum = ema_momentum

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )

        # EMA統計量（共分散行列）
        self.context_mean_ema = None      # [context_dim]
        self.context_cov_ema = None       # [context_dim, context_dim]

        # 訓練状態
        self.previous_contexts = None
        self.num_converged_tokens = 0
        self.current_iteration = 0
        self.train_convergence_rate = 0.0
        self.num_tokens = 0

        # 損失記録
        self._last_cvfp_loss = 0.0
        self._last_diversity_loss = 0.0

    def train(self, token_ids, device, label="Train"):
        """訓練実行"""
        self.model.train()
        return self._run(token_ids, device, label, is_training=True)

    def evaluate(self, token_ids, device, label="Val"):
        """評価実行"""
        self.model.eval()
        return self._run(token_ids, device, label, is_training=False)

    def _run(self, token_ids, device, label, is_training):
        """訓練/評価の共通ロジック"""
        self.model.to(device)
        self.num_tokens = len(token_ids)

        # トークン埋め込み
        with torch.no_grad():
            token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(device))
            token_embeds = self.model.embed_norm(token_embeds).squeeze(0)

        # 状態リセット
        self.previous_contexts = None
        self.num_converged_tokens = 0

        # 反復改善ループ
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            current_contexts = self._process_tokens(token_embeds, device, is_training)

            if is_training:
                self._update_convergence_state(current_contexts)

            # Early stopping
            if is_training and self._is_converged() and iteration > 0:
                break

        return current_contexts

    def _process_tokens(self, token_embeds, device, is_training):
        """全トークンを処理"""
        context = torch.zeros(1, self.model.context_dim, device=device)
        current_contexts = []

        for t, token_embed in enumerate(token_embeds):
            if is_training:
                context = self._train_one_token(
                    token_embed.unsqueeze(0),
                    context.detach() if t > 0 else context,
                    token_idx=t
                )
            else:
                with torch.no_grad():
                    context = self.model._update_context_one_step(
                        token_embed.unsqueeze(0),
                        context
                    )

            current_contexts.append(context.detach())

        return torch.cat(current_contexts, dim=0)

    def _train_one_token(self, token_embed, context, token_idx):
        """1トークンの訓練（共分散行列EMA版）"""
        # 順伝播
        new_context = self.model._update_context_one_step(token_embed, context)

        # CVFP損失
        if self.current_iteration > 0 and self.previous_contexts is not None:
            previous_token_context = self.previous_contexts[token_idx:token_idx+1].detach()
            cvfp_loss = F.mse_loss(
                F.normalize(new_context, p=2, dim=1),
                F.normalize(previous_token_context, p=2, dim=1)
            )
        else:
            cvfp_loss = torch.tensor(0.0, device=new_context.device, requires_grad=True)

        # 多様性損失（共分散行列の有効ランク）
        new_context_flat = new_context.squeeze(0)  # [context_dim]

        if self.context_mean_ema is None:
            # 初期化
            self.context_mean_ema = new_context_flat.detach()
            self.context_cov_ema = torch.eye(
                self.model.context_dim,
                device=new_context.device
            )
            diversity_loss = torch.tensor(0.0, device=new_context.device, requires_grad=True)
        else:
            # EMA更新前の値を使用
            old_mean = self.context_mean_ema.detach()
            old_cov = self.context_cov_ema.detach()

            # 中心化
            centered = new_context_flat - old_mean

            # 共分散行列の有効ランク（固有値の和）
            # ランクが低い = 多様性不足
            eigenvalues = torch.linalg.eigvalsh(old_cov)
            eigenvalues = torch.clamp(eigenvalues, min=0.0)  # 数値安定性

            # 損失 = 1 / (固有値の平均 + epsilon)
            # または: -log(固有値の平均)
            diversity_loss = -torch.log(eigenvalues.mean() + 1e-6)

            # EMA更新（学習後）
            with torch.no_grad():
                self.context_mean_ema = (
                    self.ema_momentum * old_mean +
                    (1 - self.ema_momentum) * new_context_flat
                )

                # 共分散行列の更新: Cov = E[(X - μ)(X - μ)^T]
                outer_product = torch.outer(centered, centered)
                self.context_cov_ema = (
                    self.ema_momentum * old_cov +
                    (1 - self.ema_momentum) * outer_product
                )

        # 総合損失
        total_loss = (
            (1 - self.dist_reg_weight) * cvfp_loss +
            self.dist_reg_weight * diversity_loss
        )

        # 損失記録
        self._last_cvfp_loss = cvfp_loss.item() if isinstance(cvfp_loss, torch.Tensor) else cvfp_loss
        self._last_diversity_loss = diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss

        # 最適化
        if total_loss.item() > 0 and not torch.isnan(total_loss):
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return new_context

    def _update_convergence_state(self, current_contexts):
        """収束状態を更新"""
        if self.current_iteration == 0 or self.previous_contexts is None:
            self.previous_contexts = current_contexts.detach()
            self.num_converged_tokens = 0
            return

        # 各トークンの収束判定
        mse_per_token = F.mse_loss(
            current_contexts,
            self.previous_contexts,
            reduction='none'
        ).mean(dim=1)

        converged_mask = mse_per_token < self.convergence_threshold
        self.num_converged_tokens = converged_mask.sum().item()
        self.train_convergence_rate = self.num_converged_tokens / self.num_tokens

        self.previous_contexts = current_contexts.detach()

    def _is_converged(self):
        """収束判定"""
        return self.train_convergence_rate >= self.min_converged_ratio

    def _get_convergence_rate(self):
        """収束率を取得"""
        return self.train_convergence_rate


def run_comparison_experiment():
    """比較実験を実行"""

    print("="*70)
    print("Diversity Regularization 方式比較実験")
    print("="*70)
    print()

    config = ResidualConfig()
    device = torch.device(config.device)

    # データ生成
    num_train = 5000
    num_val = 1000

    print(f"データセット: Train={num_train}, Val={num_val}")
    print()

    train_tokens = torch.randint(0, 1000, (num_train,), device=device)
    val_tokens = torch.randint(0, 1000, (num_val,), device=device)

    results = {}

    # ========== 方式2: 次元ごとの分散追跡 ==========
    print("="*70)
    print("方式2: 次元ごとの分散追跡（Per-Dimension Variance）")
    print("="*70)
    print()

    # モデル作成
    layer_structure = [1] * config.num_layers
    model_variance = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        layernorm_mix=1.0,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    model_variance.to(device)

    # トレーナー作成
    trainer_variance = Phase1TrainerVariance(
        model=model_variance,
        max_iterations=config.phase1_max_iterations,
        convergence_threshold=config.phase1_convergence_threshold,
        min_converged_ratio=config.phase1_min_converged_ratio,
        learning_rate=config.phase1_learning_rate,
        dist_reg_weight=config.dist_reg_weight,
        ema_momentum=0.99
    )

    # 訓練
    start_time = time.time()
    train_contexts_var = trainer_variance.train(train_tokens, device, label="Train")
    train_time_var = time.time() - start_time

    # 評価
    val_contexts_var = trainer_variance.evaluate(val_tokens, device, label="Val")

    # メトリクス
    train_metrics_var = analyze_fixed_points(train_contexts_var, label="Train")
    val_metrics_var = analyze_fixed_points(val_contexts_var, label="Val")

    results['variance'] = {
        'train_time': train_time_var,
        'iterations': trainer_variance.current_iteration + 1,
        'convergence_rate': trainer_variance.train_convergence_rate,
        'train_metrics': train_metrics_var,
        'val_metrics': val_metrics_var
    }

    print(f"\n訓練時間: {train_time_var:.2f}秒")
    print(f"イテレーション数: {trainer_variance.current_iteration + 1}")
    print(f"収束率: {trainer_variance.train_convergence_rate*100:.1f}%")
    print()

    # メモリ使用量
    variance_memory = (
        model_variance.context_dim * 2 * 4  # mean + var, float32
    ) / 1024  # KB
    print(f"EMA統計量メモリ: {variance_memory:.2f} KB")
    print()

    # ========== 方式1: 共分散行列EMA ==========
    print("="*70)
    print("方式1: 共分散行列EMA（Covariance Matrix EMA）")
    print("="*70)
    print()

    # モデル作成
    model_cov = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        layernorm_mix=1.0,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    model_cov.to(device)

    # トレーナー作成
    trainer_cov = Phase1TrainerCovariance(
        model=model_cov,
        max_iterations=config.phase1_max_iterations,
        convergence_threshold=config.phase1_convergence_threshold,
        min_converged_ratio=config.phase1_min_converged_ratio,
        learning_rate=config.phase1_learning_rate,
        dist_reg_weight=config.dist_reg_weight,
        ema_momentum=0.99
    )

    # 訓練
    start_time = time.time()
    train_contexts_cov = trainer_cov.train(train_tokens, device, label="Train")
    train_time_cov = time.time() - start_time

    # 評価
    val_contexts_cov = trainer_cov.evaluate(val_tokens, device, label="Val")

    # メトリクス
    train_metrics_cov = analyze_fixed_points(train_contexts_cov, label="Train")
    val_metrics_cov = analyze_fixed_points(val_contexts_cov, label="Val")

    results['covariance'] = {
        'train_time': train_time_cov,
        'iterations': trainer_cov.current_iteration + 1,
        'convergence_rate': trainer_cov.train_convergence_rate,
        'train_metrics': train_metrics_cov,
        'val_metrics': val_metrics_cov
    }

    print(f"\n訓練時間: {train_time_cov:.2f}秒")
    print(f"イテレーション数: {trainer_cov.current_iteration + 1}")
    print(f"収束率: {trainer_cov.train_convergence_rate*100:.1f}%")
    print()

    # メモリ使用量
    cov_memory = (
        model_cov.context_dim * 4 +  # mean, float32
        model_cov.context_dim * model_cov.context_dim * 4  # covariance matrix
    ) / 1024  # KB
    print(f"EMA統計量メモリ: {cov_memory:.2f} KB")
    print()

    # ========== 結果比較 ==========
    print("="*70)
    print("結果比較")
    print("="*70)
    print()

    print("【訓練時間】")
    print(f"  方式2（分散）: {train_time_var:.2f}秒")
    print(f"  方式1（共分散）: {train_time_cov:.2f}秒")
    print(f"  速度比: {train_time_cov / train_time_var:.2f}x")
    print()

    print("【メモリ使用量】")
    print(f"  方式2（分散）: {variance_memory:.2f} KB")
    print(f"  方式1（共分散）: {cov_memory:.2f} KB")
    print(f"  メモリ比: {cov_memory / variance_memory:.2f}x")
    print()

    print("【Effective Rank（訓練）】")
    print(f"  方式2（分散）: {train_metrics_var['effective_rank']:.2f}/{config.context_dim} ({train_metrics_var['effective_rank']/config.context_dim*100:.1f}%)")
    print(f"  方式1（共分散）: {train_metrics_cov['effective_rank']:.2f}/{config.context_dim} ({train_metrics_cov['effective_rank']/config.context_dim*100:.1f}%)")
    print()

    print("【Effective Rank（検証）】")
    print(f"  方式2（分散）: {val_metrics_var['effective_rank']:.2f}/{config.context_dim} ({val_metrics_var['effective_rank']/config.context_dim*100:.1f}%)")
    print(f"  方式1（共分散）: {val_metrics_cov['effective_rank']:.2f}/{config.context_dim} ({val_metrics_cov['effective_rank']/config.context_dim*100:.1f}%)")
    print()

    print("【収束性能】")
    print(f"  方式2（分散）: {results['variance']['convergence_rate']*100:.1f}% in {results['variance']['iterations']} iterations")
    print(f"  方式1（共分散）: {results['covariance']['convergence_rate']*100:.1f}% in {results['covariance']['iterations']} iterations")
    print()

    # 推奨
    print("="*70)
    print("推奨方式")
    print("="*70)
    print()

    var_rank_ratio = train_metrics_var['effective_rank'] / config.context_dim
    cov_rank_ratio = train_metrics_cov['effective_rank'] / config.context_dim

    if var_rank_ratio > cov_rank_ratio and train_time_var < train_time_cov:
        print("✅ 推奨: 方式2（次元ごとの分散追跡）")
        print("   理由: より高い多様性、より高速、より低メモリ")
    elif cov_rank_ratio > var_rank_ratio:
        print("✅ 推奨: 方式1（共分散行列EMA）")
        print(f"   理由: より高い多様性 ({cov_rank_ratio*100:.1f}% vs {var_rank_ratio*100:.1f}%)")
    else:
        print("✅ 推奨: 方式2（次元ごとの分散追跡）")
        print("   理由: 同等の多様性、より高速、より低メモリ")

    print()

    return results


if __name__ == "__main__":
    try:
        results = run_comparison_experiment()
    except Exception as e:
        print(f"\n❌ 実験失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
