"""
DataProvider - データローダーの抽象基底クラス
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch

from src.utils.io import print_flush


class DataProvider(ABC):
    """データプロバイダーの抽象基底クラス"""

    @abstractmethod
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """訓練・検証データをロード"""
        pass

    @abstractmethod
    def get_num_train_tokens(self) -> int:
        pass

    @abstractmethod
    def get_num_val_tokens(self) -> int:
        pass

    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        pass

    @abstractmethod
    def get_all_train_tokens(self, device: torch.device) -> torch.Tensor:
        pass

    @abstractmethod
    def get_all_val_tokens(self, device: torch.device) -> torch.Tensor:
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
