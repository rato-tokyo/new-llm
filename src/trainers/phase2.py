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

    def train_epoch(self, token_ids, device, batch_size, context_cache, token_embeds):
        """
        1エポックの訓練（キャッシュ方式）

        処理フロー:
        TokenBlockをバッチ並列処理（キャッシュはPhase 1から渡される）

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス
            batch_size: ミニバッチサイズ（必須）
            context_cache: Phase 1から渡されたキャッシュ（必須）
            token_embeds: Phase 1から渡されたトークン埋め込み（必須）

        Returns:
            avg_loss: 平均損失
            perplexity: パープレキシティ
        """
        self.model.train()

        target_ids = token_ids[1:]  # 最初のトークン以外（次トークン予測の正解）
        num_tokens = len(target_ids)

        # TokenBlockバッチ並列処理
        total_loss = 0.0
        num_layers = self.model.num_layers

        for batch_start in range(0, num_tokens, batch_size):
            batch_end = min(batch_start + batch_size, num_tokens)
            batch_len = batch_end - batch_start

            # バッチ内のデータを取得（GPUに転送）
            batch_targets = target_ids[batch_start:batch_end].to(device)
            batch_token_embeds = token_embeds[batch_start:batch_end].to(device)  # [batch, embed_dim * num_input_tokens]

            # バッチ内のcontext_outputsを構築（GPUに転送）
            # context_cache: テンソル [num_layers, num_tokens, context_dim] またはリスト
            batch_context_list = []
            for layer_idx in range(num_layers):
                if isinstance(context_cache, list):
                    layer_contexts = context_cache[layer_idx][batch_start:batch_end].to(device)
                else:
                    layer_contexts = context_cache[layer_idx, batch_start:batch_end].to(device)
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

    def evaluate(self, token_ids, device, context_cache, token_embeds):
        """
        評価（検証データ）- キャッシュ方式

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス
            context_cache: Phase 1から渡されたキャッシュ（必須）
            token_embeds: Phase 1から渡されたトークン埋め込み（必須）

        Returns:
            avg_loss: 平均損失
            perplexity: パープレキシティ
            accuracy: 正解率
        """
        self.model.eval()

        target_ids = token_ids[1:]
        num_tokens = len(target_ids)

        total_loss = 0.0
        correct = 0
        num_layers = self.model.num_layers

        # バッチサイズを取得（train_fullで設定された値を使用）
        # 評価時はbackwardがないので訓練より少しだけ大きくできるが、安全のため同じ値を使用
        eval_batch_size = getattr(self, '_effective_batch_size', None)
        if eval_batch_size is None:
            # train_fullで設定されていない場合のフォールバック
            eval_batch_size = getattr(self.config, 'phase2_batch_size', None) or 2048

        with torch.no_grad():
            # バッチ処理で評価（大量トークンでのOOM防止）
            for batch_start in range(0, num_tokens, eval_batch_size):
                batch_end = min(batch_start + eval_batch_size, num_tokens)

                # context_cache: テンソル [num_layers, num_tokens, context_dim] またはリスト（GPUに転送）
                context_list = []
                for layer_idx in range(num_layers):
                    if isinstance(context_cache, list):
                        layer_contexts = context_cache[layer_idx][batch_start:batch_end].to(device)
                    else:
                        layer_contexts = context_cache[layer_idx, batch_start:batch_end].to(device)
                    context_list.append(layer_contexts)

                # バッチのトークン埋め込み（GPUに転送）
                batch_token_embeds = token_embeds[batch_start:batch_end].to(device)

                # TokenBlock forward（バッチ並列）
                token_out = self.model.forward_token_e(context_list, batch_token_embeds)

                # 予測
                logits = self.model.token_output(token_out)  # [batch_size, vocab_size]

                # バッチのターゲット（GPUに転送）
                batch_targets = target_ids[batch_start:batch_end].to(device)

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
        train_context_cache,
        train_token_embeds,
        val_context_cache,
        val_token_embeds,
        epochs=None,
        patience=None,
        batch_size=None
    ):
        """
        フル訓練ループ（早期停止あり + 自動メモリ管理）

        Args:
            train_token_ids: 訓練トークンID
            val_token_ids: 検証トークンID
            device: デバイス
            train_context_cache: Phase 1から渡されたキャッシュ（必須）
            train_token_embeds: Phase 1から渡されたトークン埋め込み（必須）
            val_context_cache: Phase 1から渡されたキャッシュ（必須）
            val_token_embeds: Phase 1から渡されたトークン埋め込み（必須）
            epochs: エポック数
            patience: 早期停止の忍耐回数
            batch_size: ミニバッチサイズ（Noneで自動計算）

        Returns:
            history: 訓練履歴

        Note:
            キャッシュは必ずPhase 1から渡す必要があります。
            Phase 1でreturn_all_layers=Trueを指定してください。
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

        print_flush(f"\n[Phase 2] {train_tokens:,} train / {val_tokens:,} val tokens, {epochs} epochs")

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

        # キャッシュ構築後にバッチサイズを自動調整
        optimal_batch_size = self._calculate_optimal_batch_size(device, batch_size, total_cache_gb)
        batch_size = optimal_batch_size

        # evaluateメソッドでも同じバッチサイズを使用
        self._effective_batch_size = batch_size

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

            # Print progress (1行で)
            best_marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_marker = " ★"
            else:
                patience_counter += 1

            print_flush(
                f"  Epoch {epoch}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                f"acc={val_acc*100:.1f}% [{elapsed:.1f}s]{best_marker}"
            )

            # Early stopping check
            if patience_counter >= patience:
                print_flush(f"  → Early stop at epoch {epoch}")
                history['early_stopped'] = True
                history['stopped_epoch'] = epoch
                break

        history['best_epoch'] = best_epoch

        print_flush(
            f"  Best: epoch {best_epoch}, ppl={history['val_ppl'][best_epoch-1]:.1f}, "
            f"acc={history['val_acc'][best_epoch-1]*100:.1f}%"
        )

        return history
