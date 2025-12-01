"""
評価モジュール (Evaluation Module)

評価指標と訓練診断機能:
- metrics: 固定点分析などの評価指標
- convergence: CVFP収束判定
- diagnostics: 恒等写像検出などの訓練診断
"""

from .metrics import analyze_fixed_points, compute_effective_rank
from .convergence import check_convergence, forward_sequential, ConvergenceResult
from .diagnostics import check_identity_mapping, print_identity_mapping_warning

__all__ = [
    # 評価指標
    'analyze_fixed_points',
    'compute_effective_rank',
    # 収束判定
    'check_convergence',
    'forward_sequential',
    'ConvergenceResult',
    # 訓練診断
    'check_identity_mapping',
    'print_identity_mapping_warning',
]
