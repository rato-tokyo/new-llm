"""
Senri Configuration

Senri: Japanese LLM with Compressive Memory
OpenCALMトークナイザーを使用した日本語LLM。

すべての設定を一元管理:
- モデル構造（hidden_size, num_layers等）
- メモリ設定（Infini, MultiMemory）
- 実験設定（num_epochs, patience等）

Usage:
    from src.config import SenriConfig
    from src.models import create_model

    # デフォルト設定でモデル作成
    config = SenriConfig()
    model = create_model("infini", config)

    # カスタム設定
    config = SenriConfig(num_memories=8, use_delta_rule=False)
    model = create_model("multi_memory", config)
"""

from dataclasses import dataclass, field
from typing import Literal

from .open_calm import OPEN_CALM_VOCAB_SIZE, OPEN_CALM_TOKENIZER


# Type alias for model types
ModelTypeLiteral = Literal["pythia", "infini", "multi_memory"]


@dataclass
class SenriConfig:
    """Senri 日本語LLM設定

    Compressive Memoryを持つ日本語LLM。
    OpenCALMトークナイザーを使用。

    モデル構造、メモリ設定、実験設定を一元管理。

    Examples:
        # デフォルト設定
        config = SenriConfig()

        # Infini設定をカスタマイズ
        config = SenriConfig(num_memory_banks=2, segments_per_bank=8)

        # MultiMemory設定をカスタマイズ
        config = SenriConfig(num_memories=8, top_k=3)

        # 実験設定をカスタマイズ
        config = SenriConfig(num_epochs=50, patience=2)
    """

    # ========== モデル構造 ==========
    vocab_size: int = OPEN_CALM_VOCAB_SIZE   # OpenCALMの語彙サイズ (52,000)
    hidden_size: int = 512                   # 隠れ層の次元
    num_layers: int = 6                      # レイヤー数
    num_attention_heads: int = 8             # アテンションヘッド数
    intermediate_size: int = 2048            # FFNの中間層サイズ
    max_position_embeddings: int = 2048      # 最大シーケンス長
    rotary_pct: float = 0.25                 # Rotary embeddingの割合

    # ========== トークナイザー ==========
    tokenizer_name: str = OPEN_CALM_TOKENIZER

    # ========== Infini-Attention設定 ==========
    # Layer 0がInfiniLayerの場合に使用
    num_memory_banks: int = 1                # メモリバンク数
    segments_per_bank: int = 4               # バンクあたりのセグメント数

    # ========== MultiMemory設定 ==========
    # Layer 0がMultiMemoryLayerの場合に使用
    num_memories: int = 4                    # メモリ数（Detail Memory）
    top_k: int = 2                           # 選択するメモリ数

    # ========== 共通メモリ設定 ==========
    use_delta_rule: bool = True              # Delta Rule更新を使用

    # ========== 訓練設定 ==========
    num_epochs: int = 30                     # 最大エポック数
    batch_size: int = 8                      # バッチサイズ
    learning_rate: float = 1e-4              # 学習率
    gradient_clip: float = 1.0               # 勾配クリッピング
    patience: int = 1                        # Early stopping patience

    # ========== シーケンス設定 ==========
    seq_length: int = 128                    # 訓練時のシーケンス長

    # ========== データ設定 ==========
    pile_tokens: int = 500_000               # Pileからのトークン数
    num_pile_samples: int = 2000             # Pileサンプル数

    # ========== Reversal Curse実験設定 ==========
    num_pairs: int = 500                     # 総ペア数
    num_val_pairs: int = 100                 # Valペア数（汎化テスト用）

    # ========== 評価設定 ==========
    num_docs: int = 10                       # 評価用ドキュメント数
    tokens_per_doc: int = 4096               # ドキュメントあたりトークン数

    # ========== 位置エンコーディング ==========
    use_nope: bool = False                   # NoPE (No Position Encoding) を使用

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

        # メモリ設定
        parser.add_argument(
            "--num-memories", type=int, default=self.num_memories,
            help=f"Number of memories for MultiMemory (default: {self.num_memories})"
        )
        parser.add_argument(
            "--num-memory-banks", type=int, default=self.num_memory_banks,
            help=f"Number of memory banks for Infini (default: {self.num_memory_banks})"
        )
        parser.add_argument(
            "--no-delta-rule", action="store_true",
            help="Disable Delta Rule for memory update"
        )

    @classmethod
    def from_args(cls, args) -> "SenriConfig":
        """argparseの結果から設定を作成"""
        return cls(
            # 訓練設定
            num_epochs=getattr(args, "epochs", cls.num_epochs),
            batch_size=getattr(args, "batch_size", cls.batch_size),
            learning_rate=getattr(args, "lr", cls.learning_rate),
            patience=getattr(args, "patience", cls.patience),
            # シーケンス設定
            seq_length=getattr(args, "seq_length", cls.seq_length),
            # データ設定
            pile_tokens=getattr(args, "pile_tokens", cls.pile_tokens),
            num_pile_samples=getattr(args, "num_pile_samples", cls.num_pile_samples),
            num_pairs=getattr(args, "num_pairs", cls.num_pairs),
            num_val_pairs=getattr(args, "num_val_pairs", cls.num_val_pairs),
            # 評価設定
            num_docs=getattr(args, "num_docs", cls.num_docs),
            tokens_per_doc=getattr(args, "tokens_per_doc", cls.tokens_per_doc),
            # 位置エンコーディング
            use_nope=getattr(args, "nope", cls.use_nope),
            # メモリ設定
            num_memories=getattr(args, "num_memories", cls.num_memories),
            num_memory_banks=getattr(args, "num_memory_banks", cls.num_memory_banks),
            use_delta_rule=not getattr(args, "no_delta_rule", False),
        )
