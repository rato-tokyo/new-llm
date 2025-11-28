"""
Trainers - Phase 1/Phase 2 トレーナー

Phase 1: CVFP固定点学習（ContextBlock）
Phase 2: トークン予測学習（TokenBlock）
"""

from .phase1 import Phase1Trainer, MemoryPhase1Trainer
from .phase2 import Phase2Trainer

__all__ = [
    # Phase 1
    "Phase1Trainer",
    "MemoryPhase1Trainer",
    # Phase 2
    "Phase2Trainer",
]
