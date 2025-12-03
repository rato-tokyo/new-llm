"""
Phase1Trainer - 多様性学習トレーナーの抽象基底クラス
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Protocol
import torch

from src.utils.io import print_flush


class Phase1ConfigProtocol(Protocol):
    """Phase1Trainerが要求するConfig属性の型定義"""
    embed_dim: int
    context_dim: int
    vocab_size: int
    phase1_max_iterations: int
    phase1_convergence_threshold: float
    phase1_learning_rate: float
    phase1_batch_size: int
    phase1_gradient_clip: float
    phase1_context_noise: float
    phase1_early_stopping: bool
    phase1_early_stopping_threshold: float
    phase1_min_convergence_improvement: float

# Phase 2キャッシュ用の型定義
# 1層固定: [num_tokens, context_dim]
ContextCache = torch.Tensor


@dataclass
class Phase1Result:
    """
    Phase 1 訓練/評価の結果を格納するデータクラス

    型安全なデータ受け渡しを保証し、条件付き戻り値の脆弱性を解消。

    1層固定アーキテクチャ（2025-12-02）:
    - cacheはレイヤー出力 [num_tokens, context_dim]
    - カスケード連結方式により複数レイヤーは不要

    Attributes:
        contexts: コンテキストベクトル [num_tokens, context_dim]
        cache: コンテキストキャッシュ（Phase 2用）[num_tokens, context_dim]
        token_embeds: トークン埋め込み（Phase 2用）[num_tokens, embed_dim]
    """
    contexts: torch.Tensor
    cache: Optional[ContextCache] = None
    token_embeds: Optional[torch.Tensor] = None

    @property
    def has_cache(self) -> bool:
        """Phase 2用キャッシュを持っているか"""
        return self.cache is not None and self.token_embeds is not None


class Phase1Trainer(ABC):
    """Phase 1トレーナーの抽象基底クラス"""

    def __init__(self, model: torch.nn.Module, config: Phase1ConfigProtocol, device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        self._training_stats: Dict[str, Any] = {}

    @abstractmethod
    def train(
        self,
        token_ids: torch.Tensor,
        label: str = "Train",
        return_all_layers: bool = False,
        val_token_ids: Optional[torch.Tensor] = None
    ) -> Phase1Result:
        """
        Phase 1訓練を実行

        Args:
            token_ids: トークンID [num_tokens]
            label: ログ用ラベル
            return_all_layers: Trueの場合、Phase 2用キャッシュも返す（1層固定）
            val_token_ids: 検証用トークンID（早期停止用）

        Returns:
            Phase1Result: contexts必須、cache/token_embedsはreturn_all_layers=True時のみ
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        token_ids: torch.Tensor,
        label: str = "Val",
        return_all_layers: bool = True
    ) -> Phase1Result:
        """
        検証データを評価しPhase 2用キャッシュを収集

        Args:
            token_ids: トークンID [num_tokens]
            label: ログ用ラベル
            return_all_layers: 通常True（Phase 2キャッシュ用、1層固定）

        Returns:
            Phase1Result: contexts, cache, token_embedsすべて含む
        """
        pass

    def get_training_stats(self) -> Dict[str, Any]:
        return self._training_stats

    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        pass

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': 'phase1_complete',
            'config': {
                'embed_dim': self.config.embed_dim,
                'context_dim': self.config.context_dim,
                'vocab_size': self.config.vocab_size,
            },
            'training_stats': self._training_stats,
        }
        torch.save(checkpoint, path)
        print_flush(f"✓ Checkpoint saved: {path}")
