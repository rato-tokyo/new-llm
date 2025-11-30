"""
DataProvider - データローダーの抽象基底クラス
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Type
from types import TracebackType
import torch


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

    def close(self) -> None:
        pass

    def __enter__(self) -> "DataProvider":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        self.close()
