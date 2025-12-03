"""
Config Wrappers for Experiment Scripts

Phase 1/Phase 2 Trainer用の設定ラッパー。
スクリプト間でコードを共有するための共通実装。

Usage:
    from config import Config
    from src.config.wrappers import Phase1ConfigWrapper, Phase2ConfigWrapper

    base = Config()
    p1_cfg = Phase1ConfigWrapper(base, context_dim=500)
    p2_cfg = Phase2ConfigWrapper(base)
"""

from config import Config


class Phase1ConfigWrapper:
    """
    Phase1Trainer用のConfig wrapper

    MemoryPhase1Trainerが必要とする属性を
    base Configから抽出して保持する。
    """

    def __init__(
        self,
        base: Config,
        context_dim: int,
        patience: int = 2
    ):
        """
        Args:
            base: 基本Config
            context_dim: コンテキスト次元
            patience: Early stopping patience（デフォルト: 2）
        """
        # Phase 1 学習パラメータ
        self.phase1_max_iterations = base.phase1_max_iterations
        self.phase1_convergence_threshold = base.phase1_convergence_threshold
        self.phase1_learning_rate = base.phase1_learning_rate
        self.phase1_batch_size = base.phase1_batch_size
        self.phase1_gradient_clip = base.phase1_gradient_clip
        self.phase1_context_noise = base.phase1_context_noise

        # Early stopping
        self.phase1_early_stopping = base.phase1_early_stopping
        self.phase1_early_stopping_threshold = base.phase1_early_stopping_threshold
        self.phase1_min_convergence_improvement = base.phase1_min_convergence_improvement
        self.phase1_no_improvement_patience = patience

        # アーキテクチャ
        self.context_dim = context_dim
        self.embed_dim = base.embed_dim
        self.vocab_size = base.vocab_size
        self.num_input_tokens = base.num_input_tokens


class Phase2ConfigWrapper:
    """
    Phase2Trainer用のConfig wrapper

    Phase 2学習で必要なパラメータを保持する。
    """

    def __init__(self, base: Config):
        """
        Args:
            base: 基本Config
        """
        self.phase2_learning_rate = base.phase2_learning_rate
        self.phase2_epochs = base.phase2_epochs
        self.phase2_patience = base.phase2_patience
        self.phase2_gradient_clip = base.phase2_gradient_clip
        self.phase2_min_ppl_improvement = base.phase2_min_ppl_improvement

        # batch_sizeがNoneの場合はデフォルト値を使用
        self.phase2_batch_size: int = base.effective_phase2_batch_size
