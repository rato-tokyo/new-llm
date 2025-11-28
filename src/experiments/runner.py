"""
実験ランナー
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from calculate_params import calculate_params
from config import ResidualConfig
from src.evaluation import analyze_fixed_points
from src.experiments.config import ExperimentConfig, TrainingConfig
from src.experiments.data_loader import UltraChatDataLoader
from src.experiments.utils import (
    cleanup_memory,
    format_number,
    format_time,
    get_device,
    print_flush,
    set_seed,
)
from src.models import LLM
from src.trainers.phase1 import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer


class ExperimentRunner:
    """
    実験ランナー

    複数の実験設定を順番に実行し、結果を保存する。

    使用例:
        experiments = [
            ExperimentConfig(name="A", num_layers=6, context_multiplier=1),
            ExperimentConfig(name="B", num_layers=3, context_multiplier=2),
        ]
        runner = ExperimentRunner(
            experiments=experiments,
            sample_sizes=[500, 1000],
            output_dir="./results/my_experiment"
        )
        results = runner.run()
    """

    def __init__(
        self,
        experiments: List[ExperimentConfig],
        sample_sizes: List[int],
        output_dir: str,
        val_samples: int = 50,
        skip_existing: bool = True,
    ):
        """
        Args:
            experiments: 実験設定のリスト
            sample_sizes: 訓練サンプル数のリスト
            output_dir: 結果の出力ディレクトリ
            val_samples: 検証サンプル数
            skip_existing: 既存結果がある場合スキップするか
        """
        self.experiments = experiments
        self.sample_sizes = sample_sizes
        self.output_dir = output_dir
        self.val_samples = val_samples
        self.skip_existing = skip_existing

        self.training_config = TrainingConfig()
        self.device = get_device()
        self.data_loader: Optional[UltraChatDataLoader] = None
        self.results: List[Dict[str, Any]] = []

    def _get_experiment_id(self, exp: ExperimentConfig, num_samples: int) -> str:
        """実験IDを生成"""
        return f"exp{exp.name}_s{num_samples}"

    def _get_result_path(self, exp_id: str) -> str:
        """個別結果ファイルのパスを取得"""
        return os.path.join(self.output_dir, f"{exp_id}.json")

    def _result_exists(self, exp_id: str) -> bool:
        """既存結果があるかチェック"""
        return os.path.exists(self._get_result_path(exp_id))

    def _save_result(self, result: Dict[str, Any], exp_id: str) -> None:
        """個別結果を保存"""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self._get_result_path(exp_id), "w") as f:
            json.dump(result, f, indent=2)

    def _load_existing_results(self) -> List[Dict[str, Any]]:
        """既存の結果をロード"""
        results: List[Dict[str, Any]] = []
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.startswith("exp") and filename.endswith(".json"):
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath) as f:
                        results.append(json.load(f))
        return results

    def _get_params_info(
        self, num_layers: int, context_dim: int, num_input_tokens: int
    ) -> Dict[str, int]:
        """パラメータ数を計算（等差減少設計対応）"""
        params = calculate_params(
            num_layers=num_layers,
            context_dim=context_dim,
            embed_dim=self.training_config.embed_dim,
            num_input_tokens=num_input_tokens,
            vocab_size=self.training_config.vocab_size,
        )
        return {
            "total": params["total"],
            "context_block": params["context_block"],
            "token_block": params["token_block"],
            "embedding": params["embedding"],
            "trainable_phase1": params["trainable_phase1"],
            "trainable_phase2": params["trainable_phase2"],
        }

    def _run_single_experiment(
        self,
        exp: ExperimentConfig,
        num_samples: int,
        max_train_samples: int,
    ) -> Dict[str, Any]:
        """単一実験を実行"""
        if self.data_loader is None:
            raise RuntimeError("Data loader not initialized")

        exp_id = self._get_experiment_id(exp, num_samples)
        experiment_start = time.time()

        tc = self.training_config
        context_dim = exp.get_context_dim(tc.embed_dim)
        num_layers = exp.num_layers
        num_input_tokens = exp.num_input_tokens or tc.num_input_tokens

        print_flush(f"\n  Experiment {exp.name}: {exp.description}")
        print_flush(f"    num_layers={num_layers}, context_dim={context_dim}")
        print_flush(f"    num_input_tokens={num_input_tokens}")
        print_flush(f"    num_samples={num_samples}")

        set_seed(42)

        # パラメータ数計算
        params_info = self._get_params_info(num_layers, context_dim, num_input_tokens)
        print_flush(f"    ContextBlock params: {format_number(params_info['context_block'])}")
        print_flush(f"    TokenBlock params: {format_number(params_info['token_block'])}")
        print_flush(f"    Phase 1 学習対象: {format_number(params_info['trainable_phase1'])}")
        print_flush(f"    Phase 2 学習対象: {format_number(params_info['trainable_phase2'])}")

        # モデル初期化
        model = LLM(
            vocab_size=tc.vocab_size,
            embed_dim=tc.embed_dim,
            context_dim=context_dim,
            num_layers=num_layers,
            num_input_tokens=num_input_tokens,
            use_pretrained_embeddings=True,
            use_weight_tying=True,
        )
        model.to(self.device)

        # データ取得
        train_tokens = self.data_loader.get_train(num_samples)
        val_tokens = self.data_loader.get_val(max_train_samples, self.val_samples)

        num_train_tokens = len(train_tokens)
        num_val_tokens = len(val_tokens)

        print_flush(f"    Train tokens: {num_train_tokens:,}")
        print_flush(f"    Val tokens: {num_val_tokens:,}")

        result: Dict[str, Any] = {
            "experiment_id": exp_id,
            "experiment_name": exp.name,
            "description": exp.description,
            "num_layers": num_layers,
            "context_multiplier": exp.context_multiplier,
            "context_dim": context_dim,
            "num_input_tokens": num_input_tokens,
            "num_samples": num_samples,
            "num_train_tokens": num_train_tokens,
            "num_val_tokens": num_val_tokens,
            "context_block_params": params_info["context_block"],
            "token_block_params": params_info["token_block"],
            "trainable_phase1": params_info["trainable_phase1"],
            "trainable_phase2": params_info["trainable_phase2"],
        }

        # ========== Phase 1 ==========
        print_flush("    Phase 1 starting...")
        phase1_start = time.time()

        config = ResidualConfig()
        config.context_dim = context_dim
        config.num_layers = num_layers
        config.num_input_tokens = num_input_tokens
        config.phase1_max_iterations = tc.phase1_max_iterations
        config.phase1_learning_rate = tc.phase1_learning_rate
        config.dist_reg_weight = tc.dist_reg_weight

        trainer1 = MemoryPhase1Trainer(model, config, self.device)
        train_contexts = trainer1.train(train_tokens, label="Train")

        phase1_time = time.time() - phase1_start
        train_stats = trainer1.get_training_stats()

        # Phase 1評価
        val_result = trainer1.evaluate(val_tokens, label="Val")
        train_analysis = analyze_fixed_points(train_contexts, label="Train", verbose=False)
        # val_resultはConvergenceResultを期待
        val_contexts = val_result.contexts if hasattr(val_result, "contexts") else val_result
        val_analysis = analyze_fixed_points(val_contexts, label="Val", verbose=False)

        result.update({
            "phase1_time_sec": phase1_time,
            "phase1_iterations": train_stats.get("iterations", 0),
            "train_effective_rank": train_analysis["effective_rank"],
            "train_effective_rank_percent": train_analysis["effective_rank"] / context_dim * 100,
            "val_effective_rank": val_analysis["effective_rank"],
            "val_effective_rank_percent": val_analysis["effective_rank"] / context_dim * 100,
        })

        print_flush(f"    Phase 1 done: {format_time(phase1_time)}")
        print_flush(
            f"      Train ER: {result['train_effective_rank']:.1f}/{context_dim} "
            f"({result['train_effective_rank_percent']:.1f}%)"
        )
        print_flush(
            f"      Val ER: {result['val_effective_rank']:.1f}/{context_dim} "
            f"({result['val_effective_rank_percent']:.1f}%)"
        )

        # ========== Phase 2 ==========
        print_flush(f"    Phase 2 starting (freeze_embedding={tc.phase2_freeze_embedding})...")
        phase2_start = time.time()

        config.phase2_epochs = tc.phase2_epochs
        config.phase2_batch_size = tc.phase2_batch_size
        config.phase2_patience = tc.phase2_patience
        config.phase2_freeze_embedding = tc.phase2_freeze_embedding

        trainer2 = Phase2Trainer(model, config)
        history = trainer2.train_full(
            train_tokens,
            val_tokens,
            self.device,
            epochs=tc.phase2_epochs,
            patience=tc.phase2_patience,
            batch_size=tc.phase2_batch_size,
        )

        phase2_time = time.time() - phase2_start

        # Phase 2結果
        best_epoch = history["best_epoch"]
        result.update({
            "phase2_time_sec": phase2_time,
            "phase2_epochs_run": len(history["train_loss"]),
            "phase2_best_epoch": best_epoch,
            "val_loss": history["val_loss"][best_epoch - 1],
            "val_ppl": history["val_ppl"][best_epoch - 1],
            "val_accuracy": history["val_acc"][best_epoch - 1],
        })

        experiment_time = time.time() - experiment_start
        result["total_time_sec"] = experiment_time
        result["timestamp"] = datetime.now().isoformat()

        print_flush(f"    Phase 2 done: {format_time(phase2_time)}")
        print_flush(
            f"    RESULT: Val PPL={result['val_ppl']:.2f}, "
            f"Val Acc={result['val_accuracy'] * 100:.2f}%"
        )

        # クリーンアップ
        del model, trainer1, trainer2, train_contexts, train_tokens, val_tokens, val_result
        cleanup_memory(self.device)

        return result

    def run(self) -> List[Dict[str, Any]]:
        """全実験を実行"""
        total_start = time.time()

        print_flush(f"\n{'=' * 70}")
        print_flush("EXPERIMENT RUNNER")
        print_flush(f"{'=' * 70}")
        print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 実験設定表示
        print_flush("\nExperiments:")
        for exp in self.experiments:
            context_dim = exp.get_context_dim(self.training_config.embed_dim)
            print_flush(f"  {exp.name}: {exp.description} (context_dim={context_dim})")

        print_flush(f"\nSample sizes: {self.sample_sizes}")
        print_flush(f"num_input_tokens: {self.training_config.num_input_tokens}")
        print_flush(f"Embedding freeze: {self.training_config.phase2_freeze_embedding}")

        total_experiments = len(self.experiments) * len(self.sample_sizes)
        print_flush(f"Total experiments: {total_experiments}")

        # 既存結果をロード
        existing_results = self._load_existing_results() if self.skip_existing else []
        existing_ids = {r["experiment_id"] for r in existing_results}

        if existing_results:
            print_flush(f"\nFound {len(existing_results)} existing results")

        # データローダー初期化
        max_samples = max(self.sample_sizes)
        total_samples = max_samples + self.val_samples

        print_flush(f"\n{'=' * 70}")
        print_flush("Loading data...")
        print_flush(f"{'=' * 70}")
        self.data_loader = UltraChatDataLoader(self.device)
        self.data_loader.load_all(total_samples)

        # 実験実行
        self.results = list(existing_results)

        print_flush(f"\n{'=' * 70}")
        print_flush("Running experiments...")
        print_flush(f"{'=' * 70}")

        exp_count = 0
        for exp in self.experiments:
            for num_samples in self.sample_sizes:
                exp_id = self._get_experiment_id(exp, num_samples)

                if self.skip_existing and exp_id in existing_ids:
                    print_flush(f"\n  Skipping {exp_id} (already exists)")
                    continue

                exp_count += 1
                print_flush(f"\n[{exp_count}/{total_experiments}] Running {exp_id}...")

                try:
                    result = self._run_single_experiment(exp, num_samples, max_samples)
                    self.results.append(result)
                    self._save_result(result, exp_id)
                except Exception as e:
                    print_flush(f"ERROR in {exp_id}: {e}")
                    import traceback

                    traceback.print_exc()
                    cleanup_memory(self.device)

        # 結果保存
        total_time = time.time() - total_start

        print_flush(f"\n{'=' * 70}")
        print_flush("ALL EXPERIMENTS COMPLETE")
        print_flush(f"{'=' * 70}")
        print_flush(f"Total time: {format_time(total_time)}")

        self._save_all_results()
        self._save_summary_markdown()
        self._print_comparison_table()

        return self.results

    def _save_all_results(self) -> None:
        """全結果をJSON保存"""
        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, "all_results.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "training_config": self.training_config.to_dict(),
                    "experiments": [e.to_dict() for e in self.experiments],
                    "sample_sizes": self.sample_sizes,
                    "results": self.results,
                },
                f,
                indent=2,
            )
        print_flush(f"\nResults saved to: {json_path}")

    def _save_summary_markdown(self) -> None:
        """サマリーをMarkdown保存"""
        os.makedirs(self.output_dir, exist_ok=True)
        md_path = os.path.join(self.output_dir, "summary.md")

        with open(md_path, "w") as f:
            f.write("# 実験結果サマリー\n\n")
            f.write(f"実験日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 実験設定\n\n")
            f.write("| 実験 | レイヤー数 | context_dim | 説明 |\n")
            f.write("|------|-----------|-------------|------|\n")
            for exp in self.experiments:
                context_dim = exp.get_context_dim(self.training_config.embed_dim)
                f.write(f"| {exp.name} | {exp.num_layers} | {context_dim} | {exp.description} |\n")

            f.write("\n## 結果比較\n\n")
            f.write("| 実験 | Samples | Val PPL | Val Acc | Train ER% | Val ER% | Time |\n")
            f.write("|------|---------|---------|---------|-----------|---------|------|\n")

            for r in sorted(self.results, key=lambda x: (x["experiment_name"], x["num_samples"])):
                f.write(
                    f"| {r['experiment_name']} | "
                    f"{r['num_samples']} | "
                    f"{r['val_ppl']:.2f} | "
                    f"{r['val_accuracy'] * 100:.2f}% | "
                    f"{r['train_effective_rank_percent']:.1f}% | "
                    f"{r['val_effective_rank_percent']:.1f}% | "
                    f"{format_time(r['total_time_sec'])} |\n"
                )

        print_flush(f"Summary saved to: {md_path}")

    def _print_comparison_table(self) -> None:
        """比較結果を表示"""
        print_flush(f"\n{'=' * 100}")
        print_flush("COMPARISON RESULTS")
        print_flush(f"{'=' * 100}")

        header = (
            f"{'Exp':>4} | {'Layers':>6} | {'CtxDim':>7} | {'Samples':>7} | "
            f"{'Val PPL':>10} | {'Val Acc%':>10} | {'Train ER%':>10} | {'Val ER%':>10}"
        )
        print_flush(f"\n{header}")
        print_flush("-" * len(header))

        for r in sorted(self.results, key=lambda x: (x["num_samples"], x["experiment_name"])):
            print_flush(
                f"{r['experiment_name']:>4} | "
                f"{r['num_layers']:>6} | "
                f"{r['context_dim']:>7} | "
                f"{r['num_samples']:>7} | "
                f"{r['val_ppl']:>10.2f} | "
                f"{r['val_accuracy'] * 100:>9.2f}% | "
                f"{r['train_effective_rank_percent']:>9.1f}% | "
                f"{r['val_effective_rank_percent']:>9.1f}%"
            )
