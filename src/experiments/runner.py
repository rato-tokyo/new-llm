"""
ExperimentRunner - 実験実行の統一インターフェース

スクリプトからは設定のみを渡し、データ読み込み・訓練・評価を一括実行。
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch

from config import Config
from src.models import LLM
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed
from config.experiment import DataConfig, Phase1TrainerConfig, Phase2TrainerConfig


@dataclass
class ExperimentConfig:
    """実験設定"""
    # データ
    num_samples: int = 2000

    # アーキテクチャ
    context_dim: int = 500
    context_layers: Optional[int] = None  # Noneの場合はbase_configから
    token_layers: Optional[int] = None    # Noneの場合はbase_configから

    # その他
    seed: int = 42
    verbose: bool = True


@dataclass
class ExperimentResult:
    """実験結果"""
    config_name: str
    context_layers: int
    token_layers: int
    context_dim: int
    num_samples: int
    train_tokens: int
    val_tokens: int
    total_params: int
    context_block_params: int
    token_block_params: int
    phase1_iterations: int
    phase1_time: float
    train_er: float
    train_er_pct: float
    val_er: float
    val_er_pct: float
    convergence_rate: float
    phase2_time: float
    best_epoch: int
    train_ppl: float
    val_ppl: float
    val_acc: float
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            'config_name': self.config_name,
            'context_layers': self.context_layers,
            'token_layers': self.token_layers,
            'context_dim': self.context_dim,
            'num_samples': self.num_samples,
            'train_tokens': self.train_tokens,
            'val_tokens': self.val_tokens,
            'total_params': self.total_params,
            'context_block_params': self.context_block_params,
            'token_block_params': self.token_block_params,
            'phase1_iterations': self.phase1_iterations,
            'phase1_time': self.phase1_time,
            'train_er': self.train_er,
            'train_er_pct': self.train_er_pct,
            'val_er': self.val_er,
            'val_er_pct': self.val_er_pct,
            'convergence_rate': self.convergence_rate,
            'phase2_time': self.phase2_time,
            'best_epoch': self.best_epoch,
            'train_ppl': self.train_ppl,
            'val_ppl': self.val_ppl,
            'val_acc': self.val_acc,
            'total_time': self.total_time,
        }


class ExperimentRunner:
    """実験実行クラス"""

    def __init__(
        self,
        device: Optional[str] = None,
        base_config: Optional[Config] = None,
        verbose: bool = True
    ):
        """
        Args:
            device: 'cuda' or 'cpu' (Noneで自動検出)
            base_config: ベース設定（Noneで新規作成）
            verbose: 詳細出力
        """
        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.base_config = base_config or Config()
        self.verbose = verbose

        if verbose:
            if self.device.type == "cuda" and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print_flush(f"Device: {self.device} ({gpu_name}, {gpu_mem:.1f}GB)")
            else:
                print_flush(f"Device: {self.device}")

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        単一の実験を実行

        Args:
            config: 実験設定

        Returns:
            ExperimentResult: 実験結果
        """
        set_seed(config.seed)

        # レイヤー数を決定
        context_layers = (
            config.context_layers
            if config.context_layers is not None
            else self.base_config.num_layers
        )
        token_layers = (
            config.token_layers
            if config.token_layers is not None
            else self.base_config.num_layers
        )
        config_name = f"C{context_layers}T{token_layers}"

        if config.verbose:
            print_flush(f"\n{'='*60}")
            print_flush(f"Running {config_name}: Context {context_layers}L, Token {token_layers}L")
            print_flush(f"{'='*60}")

        # データ読み込み
        data_config = DataConfig.from_base(self.base_config, num_samples=config.num_samples)
        data_provider = MemoryDataProvider(data_config)
        train_token_ids, val_token_ids = data_provider.load_data()
        train_token_ids = train_token_ids.to(self.device)
        val_token_ids = val_token_ids.to(self.device)

        num_train_tokens = len(train_token_ids)
        num_val_tokens = len(val_token_ids)

        if config.verbose:
            print_flush(f"Data: {num_train_tokens:,} train, {num_val_tokens:,} val tokens")

        # モデル作成
        set_seed(config.seed)
        model = LLM(
            vocab_size=self.base_config.vocab_size,
            embed_dim=self.base_config.embed_dim,
            context_dim=config.context_dim,
            context_layers=context_layers,
            token_layers=token_layers,
            num_input_tokens=self.base_config.num_input_tokens,
            use_pretrained_embeddings=self.base_config.use_pretrained_embeddings,
            use_weight_tying=self.base_config.use_weight_tying,
        )
        model.to(self.device)

        params = model.num_params()
        if config.verbose:
            print_flush(f"Parameters: {params['total']:,} total")
            print_flush(f"  ContextBlock: {params['context_block']:,}")
            print_flush(f"  TokenBlock: {params['token_block']:,}")

        # Phase 1
        phase1_config = Phase1TrainerConfig.from_base(
            self.base_config, self.device,
            context_dim=config.context_dim,
            num_layers=context_layers,
        )
        phase1_trainer = MemoryPhase1Trainer(model, phase1_config, self.device)

        phase1_start = time.time()
        train_result = phase1_trainer.train(
            train_token_ids,
            label="OACD",
            return_all_layers=True,
            val_token_ids=val_token_ids
        )
        phase1_time = time.time() - phase1_start

        assert train_result.cache is not None
        assert train_result.token_embeds is not None
        train_contexts = train_result.contexts
        train_context_cache = train_result.cache[-1]
        train_token_embeds = train_result.token_embeds

        phase1_stats = phase1_trainer._training_stats
        phase1_iterations = phase1_stats.get('iterations', 0)
        convergence_rate = phase1_stats.get('convergence_rate', 0.0)

        # 検証データのキャッシュ
        val_result = phase1_trainer.evaluate(val_token_ids, return_all_layers=True)
        assert val_result.cache is not None
        assert val_result.token_embeds is not None
        val_contexts = val_result.contexts
        val_context_cache = val_result.cache[-1]
        val_token_embeds = val_result.token_embeds

        # Effective Rank計算
        train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
        val_metrics = analyze_fixed_points(val_contexts, label="Val", verbose=False)
        train_er = train_metrics['effective_rank']
        val_er = val_metrics['effective_rank']
        train_er_pct = train_er / config.context_dim * 100
        val_er_pct = val_er / config.context_dim * 100

        if config.verbose:
            print_flush(f"Phase 1: {phase1_time:.1f}s, {phase1_iterations} iter, "
                        f"conv={convergence_rate*100:.0f}%, ER={train_er_pct:.1f}%/{val_er_pct:.1f}%")

        # Phase 2
        phase2_config = Phase2TrainerConfig.from_base(
            self.base_config, self.device,
            context_dim=config.context_dim,
            num_layers=token_layers,
        )
        phase2_trainer = Phase2Trainer(model, phase2_config)

        phase2_start = time.time()
        history = phase2_trainer.train_full(
            train_token_ids=train_token_ids,
            val_token_ids=val_token_ids,
            device=self.device,
            train_context_cache=train_context_cache,
            train_token_embeds=train_token_embeds,
            val_context_cache=val_context_cache,
            val_token_embeds=val_token_embeds
        )
        phase2_time = time.time() - phase2_start

        best_epoch = history['best_epoch']
        best_ppl = history['val_ppl'][best_epoch - 1]
        best_acc = history['val_acc'][best_epoch - 1]
        best_train_ppl = history['train_ppl'][best_epoch - 1]
        total_time = phase1_time + phase2_time

        if config.verbose:
            print_flush(f"Phase 2: {phase2_time:.1f}s, Best epoch {best_epoch}")
            print_flush(f"Result: PPL={best_ppl:.1f}, Acc={best_acc*100:.1f}%")
            print_flush(f"Total time: {total_time:.1f}s")

        # メモリ解放
        del model, phase1_trainer, phase2_trainer
        del train_contexts, val_contexts
        del train_context_cache, val_context_cache
        del train_token_embeds, val_token_embeds
        data_provider.close()
        clear_gpu_cache(self.device)

        return ExperimentResult(
            config_name=config_name,
            context_layers=context_layers,
            token_layers=token_layers,
            context_dim=config.context_dim,
            num_samples=config.num_samples,
            train_tokens=num_train_tokens,
            val_tokens=num_val_tokens,
            total_params=params['total'],
            context_block_params=params['context_block'],
            token_block_params=params['token_block'],
            phase1_iterations=phase1_iterations,
            phase1_time=phase1_time,
            train_er=train_er,
            train_er_pct=train_er_pct,
            val_er=val_er,
            val_er_pct=val_er_pct,
            convergence_rate=convergence_rate,
            phase2_time=phase2_time,
            best_epoch=best_epoch,
            train_ppl=best_train_ppl,
            val_ppl=best_ppl,
            val_acc=best_acc,
            total_time=total_time,
        )


def print_results_summary(results: List[ExperimentResult]) -> None:
    """結果サマリーを表示"""
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush(f"\n{'Config':<10} {'Context':<8} {'Token':<8} {'Params':<12} "
                f"{'Val PPL':<10} {'Acc':<8} {'ER%':<8} {'Time':<8}")
    print_flush("-" * 80)

    for r in results:
        print_flush(f"{r.config_name:<10} {r.context_layers:<8} {r.token_layers:<8} "
                   f"{r.total_params:,}  {r.val_ppl:<10.1f} "
                   f"{r.val_acc*100:<8.1f} {r.val_er_pct:<8.1f} {r.total_time:<8.1f}")


def save_results(
    results: List[ExperimentResult],
    output_file: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """結果をファイルに保存"""
    with open(output_file, 'w') as f:
        f.write("Experiment Results\n")
        f.write("=" * 50 + "\n\n")

        if metadata:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

        for r in results:
            f.write(f"\n{r.config_name}: Context {r.context_layers}L, Token {r.token_layers}L\n")
            f.write(f"  Parameters: {r.total_params:,}\n")
            f.write(f"    ContextBlock: {r.context_block_params:,}\n")
            f.write(f"    TokenBlock: {r.token_block_params:,}\n")
            f.write(f"  Train tokens: {r.train_tokens:,}\n")
            f.write(f"  Val PPL: {r.val_ppl:.2f}\n")
            f.write(f"  Val Acc: {r.val_acc*100:.2f}%\n")
            f.write(f"  Val ER: {r.val_er_pct:.1f}%\n")
            f.write(f"  Convergence: {r.convergence_rate*100:.0f}%\n")
            f.write(f"  Phase 1: {r.phase1_iterations} iter, {r.phase1_time:.1f}s\n")
            f.write(f"  Phase 2: epoch {r.best_epoch}, {r.phase2_time:.1f}s\n")
            f.write(f"  Total time: {r.total_time:.1f}s\n")
