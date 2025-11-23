"""
評価モジュール (Evaluation Module)

評価指標と訓練診断機能:
- metrics: 固定点分析、特異値分析などの評価指標
- diagnostics: 恒等写像検出などの訓練診断
"""

from .metrics import analyze_fixed_points, analyze_singular_vectors
from .diagnostics import check_identity_mapping, print_identity_mapping_warning

__all__ = [
    # 評価指標
    'analyze_fixed_points',
    'analyze_singular_vectors',
    # 訓練診断
    'check_identity_mapping',
    'print_identity_mapping_warning',
]
