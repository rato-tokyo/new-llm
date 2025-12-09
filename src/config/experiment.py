"""
Experiment Configuration

実験固有の設定を管理。モデル構造とは分離。

Usage:
    from src.config import ExperimentConfig

    # デフォルト設定
    config = ExperimentConfig()

    # カスタム設定
    config = ExperimentConfig(num_epochs=50, batch_size=16)

    # CLIから設定
    config.add_to_parser(parser)
    config = ExperimentConfig.from_args(args)
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """実験設定

    訓練、評価、データに関する設定を管理。
    モデル構造（hidden_size等）は含まない。
    """

    # ========== 訓練設定 ==========
    num_epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    patience: int = 1  # Early stopping patience

    # ========== シーケンス設定 ==========
    seq_length: int = 128

    # ========== データ設定 ==========
    pile_tokens: int = 500_000
    num_pile_samples: int = 2000

    # ========== Reversal Curse実験設定 ==========
    num_pairs: int = 500  # 総ペア数
    num_val_pairs: int = 100  # Valペア数（汎化テスト用）

    # ========== 評価設定 ==========
    num_docs: int = 10  # 評価用ドキュメント数
    tokens_per_doc: int = 4096  # ドキュメントあたりトークン数

    # ========== 位置エンコーディング ==========
    use_nope: bool = False  # NoPE (No Position Encoding) を使用

    def add_to_parser(self, parser) -> None:
        """argparserに設定を追加"""
        # 訓練設定
        parser.add_argument(
            "--epochs", type=int, default=self.num_epochs,
            help=f"Number of epochs (default: {self.num_epochs})"
        )
        parser.add_argument(
            "--batch-size", type=int, default=self.batch_size,
            help=f"Batch size (default: {self.batch_size})"
        )
        parser.add_argument(
            "--lr", type=float, default=self.learning_rate,
            help=f"Learning rate (default: {self.learning_rate})"
        )
        parser.add_argument(
            "--patience", type=int, default=self.patience,
            help=f"Early stopping patience (default: {self.patience})"
        )

        # シーケンス設定
        parser.add_argument(
            "--seq-length", type=int, default=self.seq_length,
            help=f"Sequence length (default: {self.seq_length})"
        )

        # データ設定
        parser.add_argument(
            "--pile-tokens", type=int, default=self.pile_tokens,
            help=f"Number of Pile tokens (default: {self.pile_tokens})"
        )
        parser.add_argument(
            "--num-pile-samples", type=int, default=self.num_pile_samples,
            help=f"Number of Pile samples (default: {self.num_pile_samples})"
        )

        # Reversal Curse設定
        parser.add_argument(
            "--num-pairs", type=int, default=self.num_pairs,
            help=f"Total number of family pairs (default: {self.num_pairs})"
        )
        parser.add_argument(
            "--num-val-pairs", type=int, default=self.num_val_pairs,
            help=f"Number of val pairs (default: {self.num_val_pairs})"
        )

        # 評価設定
        parser.add_argument(
            "--num-docs", type=int, default=self.num_docs,
            help=f"Number of documents for evaluation (default: {self.num_docs})"
        )
        parser.add_argument(
            "--tokens-per-doc", type=int, default=self.tokens_per_doc,
            help=f"Tokens per document (default: {self.tokens_per_doc})"
        )

        # 位置エンコーディング
        parser.add_argument(
            "--nope", action="store_true", default=self.use_nope,
            help="Use NoPE (No Position Encoding)"
        )

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """argparseの結果から設定を作成"""
        return cls(
            num_epochs=getattr(args, "epochs", cls.num_epochs),
            batch_size=getattr(args, "batch_size", cls.batch_size),
            learning_rate=getattr(args, "lr", cls.learning_rate),
            patience=getattr(args, "patience", cls.patience),
            seq_length=getattr(args, "seq_length", cls.seq_length),
            pile_tokens=getattr(args, "pile_tokens", cls.pile_tokens),
            num_pile_samples=getattr(args, "num_pile_samples", cls.num_pile_samples),
            num_pairs=getattr(args, "num_pairs", cls.num_pairs),
            num_val_pairs=getattr(args, "num_val_pairs", cls.num_val_pairs),
            num_docs=getattr(args, "num_docs", cls.num_docs),
            tokens_per_doc=getattr(args, "tokens_per_doc", cls.tokens_per_doc),
            use_nope=getattr(args, "nope", cls.use_nope),
        )
