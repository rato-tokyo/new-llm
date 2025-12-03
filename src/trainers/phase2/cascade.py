"""
Cascade Context Phase 2 Trainer

CascadeContextLLM の Phase 2（TokenBlock）学習用トレーナー
"""

import time
from typing import Dict, Any

import torch
import torch.nn as nn

from src.utils.io import print_flush


class CascadePhase2Trainer:
    """Cascade Context 用の Phase 2 トレーナー"""

    def __init__(self, model: nn.Module, config: Any, device: torch.device):
        """
        Args:
            model: CascadeContextLLM モデル
            config: Phase2ConfigWrapper
            device: デバイス
        """
        self.model = model
        self.config = config
        self.device = device

    def train(
        self,
        train_token_ids: torch.Tensor,
        val_token_ids: torch.Tensor,
        train_context_cache: torch.Tensor,
        train_token_embeds: torch.Tensor,
        val_context_cache: torch.Tensor,
        val_token_embeds: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Phase 2 学習を実行

        Args:
            train_token_ids: 訓練トークンID [num_tokens]
            val_token_ids: 検証トークンID [num_tokens]
            train_context_cache: 訓練コンテキストキャッシュ [num_tokens-1, combined_dim]
            train_token_embeds: 訓練トークン埋め込み [num_tokens-1, embed_dim]
            val_context_cache: 検証コンテキストキャッシュ
            val_token_embeds: 検証トークン埋め込み

        Returns:
            学習履歴（train_ppl, val_ppl, val_acc, best_epoch など）
        """
        self.model.to(self.device)
        self.model.freeze_all_context_blocks()
        self.model.token_embedding.weight.requires_grad = False
        print_flush("✓ Embedding frozen")

        # 学習対象のパラメータ
        trainable_params = [
            p for p in self.model.token_block.parameters() if p.requires_grad
        ]
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = self.model.num_params()['total']
        print_flush(f"✓ Training TokenBlock only: {total_trainable:,}/{total_params:,} parameters")

        optimizer = torch.optim.Adam(trainable_params, lr=self.config.phase2_learning_rate)
        criterion = nn.CrossEntropyLoss()

        # ターゲット
        train_targets = train_token_ids[1:].to(self.device)
        val_targets = val_token_ids[1:].to(self.device)

        history: Dict[str, Any] = {
            'train_ppl': [],
            'val_ppl': [],
            'val_acc': [],
            'best_epoch': 1,
            'best_val_ppl': float('inf'),
        }

        best_val_ppl = float('inf')
        patience_counter = 0

        num_train = len(train_targets)
        num_val = len(val_targets)
        batch_size = self.config.phase2_batch_size

        print_flush(f"\n[Phase 2] {num_train:,} train / {num_val:,} val tokens, "
                    f"{self.config.phase2_epochs} epochs")

        for epoch in range(1, self.config.phase2_epochs + 1):
            epoch_start = time.time()

            # === Training ===
            self.model.train()
            total_loss = 0.0

            for start_idx in range(0, num_train, batch_size):
                end_idx = min(start_idx + batch_size, num_train)

                batch_token_embeds = train_token_embeds[start_idx:end_idx].to(self.device)
                batch_targets = train_targets[start_idx:end_idx]
                batch_context = train_context_cache[start_idx:end_idx].to(self.device)

                optimizer.zero_grad()
                token_out = self.model.forward_token(batch_context, batch_token_embeds)
                logits = self.model.token_output(token_out)

                loss = criterion(logits, batch_targets)
                loss.backward()

                if self.config.phase2_gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.config.phase2_gradient_clip
                    )
                optimizer.step()

                total_loss += loss.item() * (end_idx - start_idx)

            train_ppl = min(torch.exp(torch.tensor(total_loss / num_train)).item(), 1e7)

            # === Validation ===
            self.model.eval()
            val_loss = 0.0
            correct = 0

            with torch.no_grad():
                for start_idx in range(0, num_val, batch_size):
                    end_idx = min(start_idx + batch_size, num_val)

                    batch_token_embeds = val_token_embeds[start_idx:end_idx].to(self.device)
                    batch_targets = val_targets[start_idx:end_idx]
                    batch_context = val_context_cache[start_idx:end_idx].to(self.device)

                    token_out = self.model.forward_token(batch_context, batch_token_embeds)
                    logits = self.model.token_output(token_out)

                    val_loss += criterion(logits, batch_targets).item() * (end_idx - start_idx)
                    correct += (logits.argmax(dim=-1) == batch_targets).sum().item()

            val_ppl = min(torch.exp(torch.tensor(val_loss / num_val)).item(), 1e7)
            val_acc = correct / num_val

            history['train_ppl'].append(train_ppl)
            history['val_ppl'].append(val_ppl)
            history['val_acc'].append(val_acc)

            is_best = val_ppl < best_val_ppl - self.config.phase2_min_ppl_improvement
            marker = " *" if is_best else ""

            if is_best:
                best_val_ppl = val_ppl
                history['best_epoch'] = epoch
                history['best_val_ppl'] = best_val_ppl
                patience_counter = 0
            else:
                patience_counter += 1

            elapsed = time.time() - epoch_start
            print_flush(f"    Epoch {epoch}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                        f"acc={val_acc*100:.1f}% [{elapsed:.1f}s]{marker}")

            # Early stopping
            if patience_counter >= self.config.phase2_patience:
                print_flush(f"    → Early stop at epoch {epoch}")
                break

        print_flush(f"    Best: epoch {history['best_epoch']}, ppl={best_val_ppl:.1f}, "
                    f"acc={history['val_acc'][history['best_epoch']-1]*100:.1f}%")

        return history
