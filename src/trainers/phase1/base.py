"""
Phase1Trainer - CVFP固定点学習トレーナーの抽象基底クラス
"""

import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch

from src.evaluation import ConvergenceResult


def print_flush(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


class Phase1Trainer(ABC):
    """Phase 1トレーナーの抽象基底クラス"""

    def __init__(self, model: torch.nn.Module, config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self._training_stats: Dict[str, Any] = {}

    @abstractmethod
    def train(self, token_ids: torch.Tensor, label: str = "Train") -> torch.Tensor:
        pass

    @abstractmethod
    def evaluate(
        self,
        token_ids: torch.Tensor,
        label: str = "Val",
        num_trials: Optional[int] = None,
        return_contexts_only: bool = False
    ) -> Union[ConvergenceResult, torch.Tensor]:
        pass

    def get_training_stats(self) -> Dict[str, Any]:
        return self._training_stats

    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        pass

    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': 'phase1_complete',
            'config': {
                'num_layers': self.config.num_layers,
                'embed_dim': self.config.embed_dim,
                'context_dim': self.config.context_dim,
                'vocab_size': self.config.vocab_size,
            },
            'training_stats': self._training_stats,
        }
        torch.save(checkpoint, path)
        print_flush(f"✓ Checkpoint saved: {path}")
