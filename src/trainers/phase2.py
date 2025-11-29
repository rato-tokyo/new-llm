"""
Phase 2 Trainer: Next-Token Prediction with Separated Architecture (E案)

Phase 1で学習したContextBlockを固定（freeze）し、TokenBlockのみを学習する。

## キャッシュ方式による高速化

### 従来の問題点
- 各トークンごとにContextBlockをforward（遅い）
- バッチ処理が1トークンずつ（GPU効率が悪い）

### 新方式: ContextBlockキャッシュ
1. 全トークンのContextBlock出力を事前計算しキャッシュ
2. TokenBlockはキャッシュを参照してバッチ並列処理

### 期待される高速化
- ContextBlock計算: エポックごとに1回のみ（従来: トークン数×エポック回）
- TokenBlock: 真のバッチ並列処理（GPU効率大幅向上）

## 分離アーキテクチャ（E案）- レイヤー対応版

### ContextBlock（Phase 1で学習済み、Phase 2でfreeze）
- 入力: [context_in, token_embed]
- 出力: 各レイヤーのcontext出力 [context_1, context_2, ..., context_N]
- Phase 2では重みが固定されているため、同じ入力 → 同じ出力が保証される

### TokenBlock（Phase 2で学習）
- **E案**: TokenBlock Layer i は ContextBlock Layer i の出力を参照
- 入力: [context_i, token_{i-1}]
- 出力: token_i
- 最終token_Nから次トークンを予測

## 自動メモリ管理 (2025-11-29)

- サンプル数に応じたキャッシュサイズの事前見積もり
- GPU空きメモリに基づくバッチサイズ自動調整
- OOM防止のための安全係数
"""

import torch
import torch.nn as nn
import time
import sys
from typing import Dict, Any

# メモリユーティリティをインポート（オプショナル）
try:
    from src.utils.memory import can_fit_in_memory, calculate_optimal_batch_size
    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    MEMORY_UTILS_AVAILABLE = False


def print_flush(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


class Phase2Trainer:
    """
    Phase 2 Trainer: TokenBlockのみを学習（キャッシュ方式）

    分離アーキテクチャ:
    - ContextBlock: frozen（Phase 1で学習済み）
    - TokenBlock: 学習対象
    - token_output: 学習対象

    Args:
        model: LLMモデル（分離アーキテクチャ）
        config: ResidualConfig（設定）
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.learning_rate = config.phase2_learning_rate
        self.gradient_clip = config.phase2_gradient_clip

        # キャッシュ（一度構築したら再利用）
        self._train_cache = None
        self._val_cache = None

        # ContextBlockをfreeze
        self.model.freeze_context_block()

        # Embedding凍結オプション
        freeze_embedding = getattr(config, 'phase2_freeze_embedding', False)

        # token_outputを有効化（Embedding凍結オプション付き）
        self.model.unfreeze_token_output(freeze_embedding=freeze_embedding)

        # 学習対象パラメータの確認
        if model.use_separated_architecture:
            # 実際に学習されるパラメータを計算
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            num_trainable = sum(p.numel() for p in trainable_params)
            total_params = sum(p.numel() for p in model.parameters())

            if freeze_embedding:
                print_flush(f"✓ Training TokenBlock only: {num_trainable:,}/{total_params:,} parameters")
            else:
                print_flush(f"✓ Training TokenBlock + token_output: {num_trainable:,}/{total_params:,} parameters")
        else:
            # Legacy: token_outputのみ
            trainable_params = list(model.token_output.parameters())
            num_trainable = sum(p.numel() for p in trainable_params)
            total_params = sum(p.numel() for p in model.parameters())
            print_flush(f"✓ Training token_output only (legacy): {num_trainable:,}/{total_params:,} parameters")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def _build_context_cache(self, token_ids, device):
        """
        全トークンのContextBlock出力をキャッシュ構築（メモリ効率化版）

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス

        Returns:
            context_cache: レイヤーごとのテンソルリスト
                - token_input_all_layers=True: テンソル [num_layers, num_tokens, context_dim]
                - token_input_all_layers=False: リスト [num_tokens, layer_dim] × num_layers
            token_embeds: 全トークンの埋め込み [num_tokens, embed_dim * num_input_tokens]
        """
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)
        input_ids = token_ids[:-1]  # 最後のトークン以外
        num_tokens = len(input_ids)
        num_layers = self.model.num_layers
        context_dim = self.model.context_dim

        # token_input_all_layersかどうかでキャッシュ形式を分岐
        token_input_all_layers = getattr(self.model, 'token_input_all_layers', True)

        # context_blockの次元リストを取得
        if hasattr(self.model, 'context_block'):
            context_dims = getattr(self.model.context_block, 'context_dims', None)
        else:
            context_dims = None

        if token_input_all_layers or context_dims is None:
            # 旧構造: 全レイヤー同じ次元 → 単一テンソル
            context_cache = torch.zeros(
                num_layers, num_tokens, context_dim,
                device=device, dtype=torch.float32
            )
        else:
            # 等差減少設計: 各レイヤーで次元が異なる → リスト
            # context_dims[1:]は各レイヤーの出力次元（context_dims[0]は入力次元）
            context_cache = [
                torch.zeros(num_tokens, context_dims[i + 1], device=device, dtype=torch.float32)
                for i in range(num_layers)
            ]

        token_embeds = torch.zeros(
            num_tokens, self.model.embed_dim * num_input_tokens,
            device=device, dtype=torch.float32
        )

        with torch.no_grad():
            context = torch.zeros(1, context_dim, device=device)

            # トークン履歴を初期化（ゼロベクトルで埋める）
            token_history = [torch.zeros(1, self.model.embed_dim, device=device)
                             for _ in range(num_input_tokens - 1)]

            for i in range(num_tokens):
                token_id = input_ids[i]

                # トークン埋め込み
                token_embed = self.model.token_embedding(token_id.unsqueeze(0))
                if isinstance(token_embed, tuple):
                    token_embed = token_embed[0]
                token_embed = self.model.embed_norm(token_embed)

                # 履歴に追加
                token_history.append(token_embed)

                # 最新の num_input_tokens 個を結合
                combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)
                token_embeds[i] = combined_tokens.squeeze(0)

                # ContextBlock forward
                context_outputs = self.model.forward_context_with_intermediates(
                    context, combined_tokens
                )

                # 各レイヤーの出力をテンソルに格納
                for layer_idx, ctx_out in enumerate(context_outputs):
                    if isinstance(context_cache, list):
                        context_cache[layer_idx][i] = ctx_out.squeeze(0)
                    else:
                        context_cache[layer_idx, i] = ctx_out.squeeze(0)

                # コンテキスト更新
                context = context_outputs[-1]

        return context_cache, token_embeds

    def train_epoch(self, token_ids, device, batch_size=None, context_cache=None, token_embeds=None):
        """
        1エポックの訓練（キャッシュ方式）

        処理フロー:
        1. ContextBlock出力を全トークン分キャッシュ（初回のみ、以降は再利用）
        2. TokenBlockをバッチ並列処理

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス
            batch_size: ミニバッチサイズ
            context_cache: 事前構築済みキャッシュ（オプション）
            token_embeds: 事前構築済みトークン埋め込み（オプション）

        Returns:
            avg_loss: 平均損失
            perplexity: パープレキシティ
        """
        self.model.train()

        if batch_size is None:
            # effective_phase2_batch_sizeプロパティを使用（自動計算対応）
            batch_size = getattr(self.config, 'effective_phase2_batch_size', None)
            if batch_size is None:
                batch_size = self.config.phase2_batch_size or 1024

        target_ids = token_ids[1:]  # 最初のトークン以外（次トークン予測の正解）
        num_tokens = len(target_ids)

        # Step 1: ContextBlockキャッシュ構築（キャッシュがない場合のみ）
        if context_cache is None or token_embeds is None:
            context_cache, token_embeds = self._build_context_cache(token_ids, device)

        # Step 2: TokenBlockバッチ並列処理
        total_loss = 0.0
        num_layers = self.model.num_layers

        for batch_start in range(0, num_tokens, batch_size):
            batch_end = min(batch_start + batch_size, num_tokens)
            batch_len = batch_end - batch_start

            # バッチ内のデータを取得
            batch_targets = target_ids[batch_start:batch_end]
            batch_token_embeds = token_embeds[batch_start:batch_end]  # [batch, embed_dim * num_input_tokens]

            # バッチ内のcontext_outputsを構築
            # context_cache: テンソル [num_layers, num_tokens, context_dim] またはリスト
            batch_context_list = []
            for layer_idx in range(num_layers):
                if isinstance(context_cache, list):
                    layer_contexts = context_cache[layer_idx][batch_start:batch_end]
                else:
                    layer_contexts = context_cache[layer_idx, batch_start:batch_end]
                batch_context_list.append(layer_contexts)

            # TokenBlock forward（バッチ並列）
            batch_token_out = self.model.forward_token_e(batch_context_list, batch_token_embeds)

            # 予測
            batch_logits = self.model.token_output(batch_token_out)

            # 損失計算と更新
            self.optimizer.zero_grad()
            batch_loss = self.criterion(batch_logits, batch_targets)
            batch_loss.backward()

            # 勾配クリッピング
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.gradient_clip
                )

            self.optimizer.step()

            # 損失を累積
            total_loss += batch_loss.item() * batch_len

        # 平均損失とパープレキシティ
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def evaluate(self, token_ids, device, context_cache=None, token_embeds=None):
        """
        評価（検証データ）- キャッシュ方式

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス
            context_cache: 事前構築済みキャッシュ（オプション）
            token_embeds: 事前構築済みトークン埋め込み（オプション）

        Returns:
            avg_loss: 平均損失
            perplexity: パープレキシティ
            accuracy: 正解率
        """
        self.model.eval()

        target_ids = token_ids[1:]
        num_tokens = len(target_ids)

        # ContextBlockキャッシュ構築（キャッシュがない場合のみ）
        if context_cache is None or token_embeds is None:
            context_cache, token_embeds = self._build_context_cache(token_ids, device)

        total_loss = 0.0
        correct = 0
        num_layers = self.model.num_layers

        # バッチサイズを計算（評価時も大きなデータでOOMを防ぐ）
        # 評価時はbackwardがないので、訓練時より大きくできる
        eval_batch_size = getattr(self, '_eval_batch_size', None)
        if eval_batch_size is None:
            # 訓練バッチサイズの2倍を評価バッチサイズとして使用
            train_batch = getattr(self.config, 'phase2_batch_size', None) or 4096
            eval_batch_size = min(train_batch * 2, 8192)
            self._eval_batch_size = eval_batch_size

        with torch.no_grad():
            # バッチ処理で評価（大量トークンでのOOM防止）
            for batch_start in range(0, num_tokens, eval_batch_size):
                batch_end = min(batch_start + eval_batch_size, num_tokens)

                # context_cache: テンソル [num_layers, num_tokens, context_dim] またはリスト
                context_list = []
                for layer_idx in range(num_layers):
                    if isinstance(context_cache, list):
                        layer_contexts = context_cache[layer_idx][batch_start:batch_end]
                    else:
                        layer_contexts = context_cache[layer_idx, batch_start:batch_end]
                    context_list.append(layer_contexts)

                # バッチのトークン埋め込み
                batch_token_embeds = token_embeds[batch_start:batch_end]

                # TokenBlock forward（バッチ並列）
                token_out = self.model.forward_token_e(context_list, batch_token_embeds)

                # 予測
                logits = self.model.token_output(token_out)  # [batch_size, vocab_size]

                # バッチのターゲット
                batch_targets = target_ids[batch_start:batch_end]

                # 損失（バッチ）
                batch_loss = self.criterion(logits, batch_targets).item()
                total_loss += batch_loss * (batch_end - batch_start)

                # 正解率
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch_targets).sum().item()

        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = correct / num_tokens

        return avg_loss, perplexity, accuracy

    def _estimate_memory_requirements(
        self,
        train_tokens: int,
        val_tokens: int,
        device
    ) -> Dict[str, Any]:
        """
        メモリ要件を事前見積もり

        Args:
            train_tokens: 訓練トークン数
            val_tokens: 検証トークン数
            device: デバイス

        Returns:
            dict: メモリ見積もり情報
        """
        num_layers = self.model.num_layers
        context_dim = self.model.context_dim
        embed_dim = self.model.embed_dim
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)

        if MEMORY_UTILS_AVAILABLE:
            return can_fit_in_memory(
                train_tokens, val_tokens,
                num_layers, context_dim, embed_dim, num_input_tokens
            )

        # フォールバック: 簡易計算
        # context_cache: [num_layers, num_tokens, context_dim] (float32)
        train_cache_gb = (num_layers * train_tokens * context_dim * 4) / (1024**3)
        val_cache_gb = (num_layers * val_tokens * context_dim * 4) / (1024**3)

        if torch.cuda.is_available():
            total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available = total_gpu * 0.8
        else:
            available = float('inf')

        total_required = train_cache_gb + val_cache_gb + 1.0  # +1GB for model/batch

        return {
            'fits': total_required <= available,
            'total_required_gb': total_required,
            'available_gb': available,
            'train_cache_gb': train_cache_gb,
            'val_cache_gb': val_cache_gb,
            'recommendation': f"Required: {total_required:.1f}GB, Available: {available:.1f}GB"
        }

    def _calculate_optimal_batch_size(
        self,
        device,
        initial_batch_size: int = 4096,
        actual_cache_gb: float = 0.0
    ) -> int:
        """
        現在のGPUメモリ状態から最適なバッチサイズを計算

        memory.pyのcalculate_optimal_batch_size関数に委譲。

        Args:
            device: デバイス
            initial_batch_size: 初期バッチサイズ
            actual_cache_gb: 実際に使用しているキャッシュサイズ (GB) - 未使用

        Returns:
            int: 最適なバッチサイズ
        """
        if MEMORY_UTILS_AVAILABLE:
            # memory.pyの関数を使用（一元化）
            return calculate_optimal_batch_size(
                vocab_size=getattr(self.config, 'vocab_size', 50257),
                safety_factor=getattr(self.config, 'phase2_memory_safety_factor', 0.5),
                min_batch_size=getattr(self.config, 'phase2_min_batch_size', 256),
                max_batch_size=getattr(self.config, 'phase2_max_batch_size', 16384),
                initial_batch_size=initial_batch_size,
                verbose=True
            )

        # フォールバック: memory.pyがない場合の簡易実装
        if not torch.cuda.is_available():
            return min(initial_batch_size, 512)

        # 簡易計算
        safety_factor = getattr(self.config, 'phase2_memory_safety_factor', 0.5)
        min_batch = getattr(self.config, 'phase2_min_batch_size', 256)
        max_batch = getattr(self.config, 'phase2_max_batch_size', 16384)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        free_gb = total_gb - allocated_gb

        vocab_size = getattr(self.config, 'vocab_size', 50257)
        per_token_mb = vocab_size * 4 / (1024**2) * 3.5
        available_mb = free_gb * 1024 * safety_factor * 0.5
        safe_batch_size = int(available_mb / per_token_mb)

        return max(min_batch, min(safe_batch_size, min(initial_batch_size, max_batch)))

    def train_full(
        self,
        train_token_ids,
        val_token_ids,
        device,
        epochs=None,
        patience=None,
        batch_size=None,
        train_context_cache=None,
        train_token_embeds=None,
        val_context_cache=None,
        val_token_embeds=None
    ):
        """
        フル訓練ループ（早期停止あり + 自動メモリ管理）

        Args:
            train_token_ids: 訓練トークンID
            val_token_ids: 検証トークンID
            device: デバイス
            epochs: エポック数
            patience: 早期停止の忍耐回数
            batch_size: ミニバッチサイズ（Noneで自動計算）
            train_context_cache: Phase 1から渡されたキャッシュ（オプション）
            train_token_embeds: Phase 1から渡されたトークン埋め込み（オプション）
            val_context_cache: Phase 1から渡されたキャッシュ（オプション）
            val_token_embeds: Phase 1から渡されたトークン埋め込み（オプション）

        Returns:
            history: 訓練履歴
        """
        if epochs is None:
            epochs = self.config.phase2_epochs
        if patience is None:
            patience = self.config.phase2_patience

        # 初期バッチサイズ（後で調整される可能性あり）
        if batch_size is None:
            batch_size = getattr(self.config, 'effective_phase2_batch_size', None)
            if batch_size is None:
                batch_size = self.config.phase2_batch_size or 4096

        train_tokens = len(train_token_ids)
        val_tokens = len(val_token_ids)

        print_flush(f"\n{'='*70}")
        print_flush("PHASE 2: Next-Token Prediction Training (キャッシュ方式)")
        print_flush(f"{'='*70}\n")

        # メモリ事前見積もり
        memory_estimate = self._estimate_memory_requirements(train_tokens, val_tokens, device)
        print_flush("Memory Estimation:")
        print_flush(f"  Train cache: {memory_estimate['train_cache_gb']:.2f}GB")
        print_flush(f"  Val cache: {memory_estimate['val_cache_gb']:.2f}GB")
        print_flush(f"  Available GPU: {memory_estimate['available_gb']:.1f}GB")
        if not memory_estimate['fits']:
            print_flush(f"  ⚠️ WARNING: {memory_estimate['recommendation']}")
        else:
            print_flush(f"  ✓ {memory_estimate['recommendation']}")
        print_flush("")

        print_flush(f"Training tokens: {train_tokens:,}")
        print_flush(f"Validation tokens: {val_tokens:,}")
        print_flush(f"Epochs: {epochs}")
        print_flush(f"Initial batch size: {batch_size}")
        print_flush(f"Learning rate: {self.learning_rate}")
        print_flush(f"Gradient clip: {self.gradient_clip}")
        print_flush(f"Early stopping patience: {patience}")
        print_flush("")

        if self.model.use_separated_architecture:
            print_flush("Architecture: E案 (ContextBlock + TokenBlock Layer-wise)")
            print_flush("  - ContextBlock: FROZEN + CACHED (エポックごとに1回計算)")
            print_flush("  - TokenBlock: TRAINING (バッチ並列処理)")
            print_flush("  - token_output: TRAINING")
            print_flush("  - E案: TokenBlock Layer i は ContextBlock Layer i の出力を参照")
        else:
            print_flush("Architecture: Legacy CVFP")
            print_flush("  - CVFP blocks: FROZEN")
            print_flush("  - token_output: TRAINING")
        print_flush("")

        # キャッシュを事前構築（全エポックで再利用）
        # Phase 1からキャッシュが渡された場合はスキップ
        cache_reused = train_context_cache is not None and train_token_embeds is not None
        if cache_reused:
            print_flush("Using pre-built context cache from Phase 1 (skipping cache build)")
            cache_time = 0.0
            # 検証キャッシュも確認
            if val_context_cache is None or val_token_embeds is None:
                print_flush("Building val context cache...")
                cache_start = time.time()
                val_context_cache, val_token_embeds = self._build_context_cache(val_token_ids, device)
                cache_time = time.time() - cache_start
                print_flush(f"Val cache built in {cache_time:.1f}s")
        else:
            print_flush("Building context cache (one-time computation)...")
            cache_start = time.time()
            train_context_cache, train_token_embeds = self._build_context_cache(train_token_ids, device)
            val_context_cache, val_token_embeds = self._build_context_cache(val_token_ids, device)
            cache_time = time.time() - cache_start
            print_flush(f"Cache built in {cache_time:.1f}s")

        # 実際のキャッシュサイズを計算
        def _calc_cache_size(cache):
            """キャッシュサイズを計算（テンソルまたはリスト対応）"""
            if isinstance(cache, list):
                return sum(t.numel() for t in cache) * 4 / (1024 * 1024)
            else:
                return cache.numel() * 4 / (1024 * 1024)

        train_cache_mb = _calc_cache_size(train_context_cache)
        val_cache_mb = _calc_cache_size(val_context_cache)
        total_cache_gb = (train_cache_mb + val_cache_mb) / 1024
        print_flush(f"Actual cache size: train={train_cache_mb:.1f}MB, val={val_cache_mb:.1f}MB (total={total_cache_gb:.2f}GB)")

        # キャッシュ構築後にバッチサイズを自動調整
        print_flush("\nCalculating optimal batch size...")
        optimal_batch_size = self._calculate_optimal_batch_size(device, batch_size, total_cache_gb)
        if optimal_batch_size != batch_size:
            print_flush(f"⚠️ Auto-adjusting batch_size: {batch_size} → {optimal_batch_size}")
        batch_size = optimal_batch_size
        print_flush(f"Effective batch size: {batch_size}")
        print_flush("")

        history = {
            'train_loss': [],
            'train_ppl': [],
            'val_loss': [],
            'val_ppl': [],
            'val_acc': [],
            'early_stopped': False,
            'stopped_epoch': None,
            'best_epoch': None
        }

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train（キャッシュを再利用）
            train_loss, train_ppl = self.train_epoch(
                train_token_ids, device, batch_size=batch_size,
                context_cache=train_context_cache, token_embeds=train_token_embeds
            )

            # Validate（キャッシュを再利用）
            val_loss, val_ppl, val_acc = self.evaluate(
                val_token_ids, device,
                context_cache=val_context_cache, token_embeds=val_token_embeds
            )

            elapsed = time.time() - start_time

            # Record history
            history['train_loss'].append(train_loss)
            history['train_ppl'].append(train_ppl)
            history['val_loss'].append(val_loss)
            history['val_ppl'].append(val_ppl)
            history['val_acc'].append(val_acc)

            # Print progress
            print_flush(f"Epoch {epoch}/{epochs} [{elapsed:.1f}s]:")
            print_flush(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print_flush(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val Acc: {val_acc*100:.2f}%")

            # Track best model and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                print_flush(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print_flush(f"  ⚠️ No improvement ({patience_counter}/{patience})")

            print_flush("")

            # Early stopping check
            if patience_counter >= patience:
                print_flush(f"⛔ Early stopping triggered at epoch {epoch}")
                print_flush(f"   Val loss hasn't improved for {patience} epochs")
                history['early_stopped'] = True
                history['stopped_epoch'] = epoch
                break

        history['best_epoch'] = best_epoch

        print_flush(f"{'='*70}")
        print_flush("Phase 2 Training Complete")
        print_flush(f"{'='*70}\n")
        print_flush(f"Best epoch: {best_epoch}")
        print_flush(f"Best validation loss: {best_val_loss:.4f}")
        print_flush(f"Best validation PPL: {history['val_ppl'][best_epoch-1]:.2f}")
        print_flush(f"Best validation accuracy: {history['val_acc'][best_epoch-1]*100:.2f}%")
        if history['early_stopped']:
            print_flush(f"Early stopped at epoch: {history['stopped_epoch']}\n")
        else:
            print_flush(f"Completed all {epochs} epochs\n")

        return history
