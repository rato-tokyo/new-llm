"""
ExperimentRunner - 実験実行の統一インターフェース

スクリプトからは設定のみを渡し、データ読み込み・訓練・評価を一括実行。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

import torch

from config import ResidualConfig
from src.models import LLM
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed
from src.experiments.config import DataConfig, Phase1Config, Phase2Config


@dataclass
class ExperimentConfig:
    """実験設定"""
    # アーキテクチャ
    num_layers: int = 6
    context_dim: int = 768
    embed_dim: int = 768
    num_input_tokens: int = 1

    # データ
    num_samples: int = 500
    val_ratio: float = 0.1

    # 訓練（Noneの場合はResidualConfigのデフォルト値を使用）
    phase1_learning_rate: Optional[float] = None
    phase1_max_iterations: Optional[int] = None
    phase2_learning_rate: Optional[float] = None
    phase2_epochs: Optional[int] = None
    dist_reg_weight: Optional[float] = None

    # その他
    random_seed: int = 42
    verbose: bool = True

    def get_phase1_config(
        self, base_config: ResidualConfig, device: Union[str, torch.device]
    ) -> Phase1Config:
        """Phase 1用の設定オブジェクトを生成"""
        return Phase1Config.from_base(
            base_config, device,
            context_dim=self.context_dim,
            num_layers=self.num_layers,
            num_input_tokens=self.num_input_tokens,
            phase1_learning_rate=self.phase1_learning_rate,
            phase1_max_iterations=self.phase1_max_iterations,
            dist_reg_weight=self.dist_reg_weight,
        )

    def get_phase2_config(
        self, base_config: ResidualConfig, device: Union[str, torch.device]
    ) -> Phase2Config:
        """Phase 2用の設定オブジェクトを生成"""
        return Phase2Config.from_base(
            base_config, device,
            context_dim=self.context_dim,
            num_layers=self.num_layers,
            num_input_tokens=self.num_input_tokens,
            phase2_learning_rate=self.phase2_learning_rate,
            phase2_epochs=self.phase2_epochs,
        )

    def get_data_config(self, base_config: ResidualConfig) -> DataConfig:
        """データ読み込み用の設定オブジェクトを生成"""
        return DataConfig.from_base(base_config, num_samples=self.num_samples)


class ExperimentRunner:
    """実験実行クラス"""

    def __init__(self, device: Optional[str] = None, base_config: Optional[ResidualConfig] = None):
        """
        Args:
            device: 'cuda' or 'cpu' (Noneで自動検出)
            base_config: ベース設定（Noneで新規作成）
        """
        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.base_config = base_config or ResidualConfig()

        if self.device.type == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print_flush(f"Device: {self.device} ({gpu_name}, {gpu_mem:.1f}GB)")
        else:
            print_flush(f"Device: {self.device}")

    def run(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        単一の実験を実行

        Args:
            config: 実験設定

        Returns:
            結果辞書（PPL, Accuracy, Effective Rank等）
        """
        set_seed(config.random_seed)

        if config.verbose:
            print_flush(f"\n--- Experiment: {config.num_samples} samples, "
                       f"{config.num_layers}L, {config.context_dim}d, "
                       f"{config.num_input_tokens}tok ---")

        # データ読み込み
        data_config = config.get_data_config(self.base_config)
        provider = MemoryDataProvider(data_config)
        train_token_ids, val_token_ids = provider.load_data()
        train_token_ids = train_token_ids.to(self.device)
        val_token_ids = val_token_ids.to(self.device)

        train_tokens = len(train_token_ids)
        val_tokens = len(val_token_ids)

        if config.verbose:
            print_flush(f"  Data: {train_tokens:,} train, {val_tokens:,} val tokens")

        # モデル作成
        model = LLM(
            vocab_size=self.base_config.vocab_size,
            embed_dim=config.embed_dim,
            context_dim=config.context_dim,
            num_layers=config.num_layers,
            num_input_tokens=config.num_input_tokens,
            use_pretrained_embeddings=self.base_config.use_pretrained_embeddings,
            use_weight_tying=self.base_config.use_weight_tying,
            config=self.base_config
        ).to(self.device)

        # Phase 1 実行
        phase1_config = config.get_phase1_config(self.base_config, self.device)
        phase1_trainer = MemoryPhase1Trainer(model, phase1_config, self.device)

        train_result = phase1_trainer.train(
            train_token_ids,
            return_all_layers=True,
            val_token_ids=val_token_ids
        )
        train_contexts, train_context_cache, train_token_embeds = train_result

        # Phase 1 訓練統計を取得
        phase1_stats = getattr(phase1_trainer, '_training_stats', {})
        phase1_iterations = phase1_stats.get('iterations', 0)

        val_result = phase1_trainer.evaluate(val_token_ids, return_all_layers=True)
        assert isinstance(val_result, tuple), "evaluate with return_all_layers=True must return tuple"
        val_contexts, val_context_cache, val_token_embeds = val_result

        # Effective Rank計算
        train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
        val_metrics = analyze_fixed_points(val_contexts, label="Val", verbose=False)
        train_er = train_metrics['effective_rank'] / config.context_dim
        val_er = val_metrics['effective_rank'] / config.context_dim

        if config.verbose:
            print_flush(f"  ER: train={train_er*100:.1f}%, val={val_er*100:.1f}%")

        # Phase 2 実行
        phase2_config = config.get_phase2_config(self.base_config, self.device)
        phase2_trainer = Phase2Trainer(model, phase2_config)

        history = phase2_trainer.train_full(
            train_token_ids=train_token_ids,
            val_token_ids=val_token_ids,
            device=self.device,
            train_context_cache=train_context_cache,
            train_token_embeds=train_token_embeds,
            val_context_cache=val_context_cache,
            val_token_embeds=val_token_embeds
        )

        best_epoch = history['best_epoch']
        best_ppl = history['val_ppl'][best_epoch - 1]
        best_acc = history['val_acc'][best_epoch - 1]
        best_train_ppl = history['train_ppl'][best_epoch - 1]

        if config.verbose:
            print_flush(f"  Result: PPL={best_ppl:.1f}, Acc={best_acc*100:.1f}%")

        # メモリ解放
        del model, phase1_trainer, phase2_trainer
        del train_contexts, val_contexts
        del train_context_cache, val_context_cache
        del train_token_embeds, val_token_embeds
        provider.close()
        clear_gpu_cache(self.device)

        return {
            'num_samples': config.num_samples,
            'num_layers': config.num_layers,
            'context_dim': config.context_dim,
            'num_input_tokens': config.num_input_tokens,
            'train_tokens': train_tokens,
            'val_tokens': val_tokens,
            'phase1_iterations': phase1_iterations,
            'train_effective_rank': train_er,
            'val_effective_rank': val_er,
            'best_epoch': best_epoch,
            'train_ppl': best_train_ppl,
            'val_ppl': best_ppl,
            'val_acc': best_acc,
            'history': {
                'train_loss': history['train_loss'],
                'train_ppl': history['train_ppl'],
                'val_ppl': history['val_ppl'],
                'val_acc': history['val_acc'],
            }
        }

    def calculate_params(self, config: ExperimentConfig) -> Dict[str, int]:
        """パラメータ数を計算"""
        token_input_dim = config.embed_dim * config.num_input_tokens
        vocab_size = self.base_config.vocab_size

        # ContextBlock
        context_block_total = 0
        for i in range(config.num_layers):
            layer_input_dim = config.context_dim + token_input_dim
            layer_output_dim = config.context_dim
            fnn_params = layer_input_dim * layer_output_dim + layer_output_dim
            layernorm_params = layer_output_dim * 2
            context_block_total += fnn_params + layernorm_params

        # TokenBlock
        token_block_total = 0
        for i in range(config.num_layers):
            token_in = config.embed_dim
            token_out = config.embed_dim
            ctx_dim = config.context_dim
            fnn_params = (ctx_dim + token_in) * token_out + token_out
            layernorm_params = token_out * 2
            token_block_total += fnn_params + layernorm_params

        embedding = vocab_size * config.embed_dim
        embed_norm = config.embed_dim * 2
        total = embedding + embed_norm + context_block_total + token_block_total

        return {
            'total': total,
            'context_block': context_block_total,
            'token_block': token_block_total,
            'trainable_phase1': context_block_total,
            'trainable_phase2': token_block_total,
        }
