"""
Phase 1 Training (Version 3): Token-wise online learning with CVFP

Key improvements:
1. Token-wise loss computation and backpropagation
2. True online learning (not batch processing)
3. Memory efficient (no need to store all contexts)
4. Layers automatically track previous outputs (enable_cvfp_learning=True)
"""

import torch


def train_phase1(model, token_ids, config, device, is_training=True, label=""):
    """
    Phase 1: CVFP Fixed-Point Learning (Token-wise Online Version)

    各トークン処理後に即座に損失計算・バックプロパゲーションを行う
    真のオンライン学習実装。

    Args:
        model: The language model (must have enable_cvfp_learning capability)
        token_ids: Input token IDs [num_tokens]
        config: Configuration object
        device: torch device
        is_training: If True, train parameters; if False, eval only
        label: Label for logging (e.g., "Train", "Val")

    Returns:
        final_contexts: Converged context vectors [num_tokens, context_dim]
    """
    print_flush(f"\n{'='*70}")
    print_flush(f"PHASE 1: Fixed-Point Context Learning (CVFP){' - ' + label if label else ''}")
    print_flush(f"{'='*70}")

    model.to(device)

    # Setup training or evaluation mode
    if is_training:
        model.train()

        # Reset running statistics at start of training
        model.reset_running_stats()

        # Only train context generation layers
        context_params = []
        for name, param in model.named_parameters():
            if 'token_output' not in name and 'token_embedding' not in name:
                param.requires_grad = True
                context_params.append(param)
            else:
                param.requires_grad = False

        optimizer = torch.optim.Adam(context_params, lr=config.phase1_lr_warmup)
        early_stopping = Phase1EarlyStopping(
            convergence_threshold=config.phase1_min_converged_ratio,
            min_delta=0.01
        )
    else:
        model.eval()

    # Compute token embeddings (once)
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # Initialize previous contexts for convergence checking
    prev_contexts = None

    # Iterative refinement loop
    for iteration in range(config.phase1_max_iterations):
        # LR Schedule (only for training)
        if is_training:
            if iteration <= 3:
                lr = config.phase1_lr_warmup
            elif iteration <= 8:
                lr = config.phase1_lr_medium
            else:
                lr = config.phase1_lr_finetune

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Reset CVFP state at start of each iteration
            model.reset_cvfp_state()

        # Token-wise processing
        context = torch.zeros(1, model.context_dim, device=device)
        current_contexts = []
        iteration_cvfp_loss = 0.0
        iteration_dist_loss = 0.0
        num_tokens_processed = 0

        for t, token_embed in enumerate(token_embeds):
            if is_training:
                # Forward pass for this token (with gradients)
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context.detach() if t > 0 else context  # 最初のトークン以外は前のトークンからの勾配を切断
                )

                # CVFP loss: レイヤーが自動計算（前回の同じトークン位置との差）
                # iteration > 0 かつ previous_context が設定されている場合のみ学習
                if iteration > 0:
                    cvfp_loss = model.get_cvfp_loss()

                    # CVFP損失が0でない場合のみbackprop（初回イテレーションは0）
                    if cvfp_loss.item() > 0:
                        optimizer.zero_grad()

                        # Distribution regularization
                        if config.use_distribution_reg:
                            dist_loss = model.get_distribution_loss()
                            dist_weight = config.dist_reg_weight
                            total_loss = (1 - dist_weight) * cvfp_loss + dist_weight * dist_loss
                            iteration_dist_loss += dist_loss.item()
                        else:
                            total_loss = cvfp_loss

                        iteration_cvfp_loss += cvfp_loss.item()

                        # Backprop for this token
                        total_loss.backward()
                        optimizer.step()

                        num_tokens_processed += 1

                current_contexts.append(context.detach())

            else:
                # Evaluation: no gradients
                with torch.no_grad():
                    context = model._update_context_one_step(
                        token_embed.unsqueeze(0),
                        context
                    )
                    current_contexts.append(context.clone())

        # Stack all contexts
        current_contexts_tensor = torch.cat(current_contexts, dim=0)

        # Check convergence and log (iteration > 0 only)
        if iteration > 0 and prev_contexts is not None:
            with torch.no_grad():
                token_losses = ((current_contexts_tensor - prev_contexts) ** 2).mean(dim=1)
                converged_tokens = token_losses < config.phase1_convergence_threshold

            # Logging
            convergence_rate = converged_tokens.float().mean().item()

            if is_training:
                avg_cvfp_loss = iteration_cvfp_loss / num_tokens_processed
                log_msg = f"Iteration {iteration+1}/{config.phase1_max_iterations}: "

                if config.use_distribution_reg:
                    avg_dist_loss = iteration_dist_loss / num_tokens_processed
                    avg_total_loss = (1 - config.dist_reg_weight) * avg_cvfp_loss + \
                                   config.dist_reg_weight * avg_dist_loss
                    log_msg += f"Loss={avg_total_loss:.6f} (CVFP={avg_cvfp_loss:.6f}, Dist={avg_dist_loss:.6f}), "
                else:
                    log_msg += f"Loss={avg_cvfp_loss:.6f}, "

                log_msg += f"Converged={convergence_rate*100:.1f}%, LR={lr:.4f}"
            else:
                log_msg = f"Iteration {iteration+1}/{config.phase1_max_iterations}: "
                log_msg += f"Converged={convergence_rate*100:.1f}%"

            print_flush(log_msg)

            # Early stopping
            if is_training and early_stopping(convergence_rate):
                print_flush(f"  → Early stopping: Convergence = {convergence_rate*100:.1f}%")
                break
            elif not is_training and convergence_rate >= config.phase1_min_converged_ratio:
                print_flush(f"  → Early stopping: Convergence = {convergence_rate*100:.1f}%")
                break
        else:
            # Iteration 0: just save contexts
            print_flush(f"Iteration 1/{config.phase1_max_iterations}: Forward pass only (saving contexts)")
            converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool, device=device)

        # Update previous contexts for next iteration
        prev_contexts = current_contexts_tensor

    print_flush(f"\nPhase 1 Complete: {converged_tokens.sum()}/{len(token_ids)} tokens converged\n")

    return current_contexts_tensor


class Phase1EarlyStopping:
    """Early stopping for Phase 1 based on convergence rate"""

    def __init__(self, convergence_threshold=0.95, min_delta=0.01):
        self.convergence_threshold = convergence_threshold
        self.min_delta = min_delta
        self.best_rate = 0
        self.counter = 0
        self.patience = 2

    def __call__(self, convergence_rate):
        if convergence_rate >= self.convergence_threshold:
            return True

        if convergence_rate < self.best_rate - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_rate = max(self.best_rate, convergence_rate)
            self.counter = 0

        return False


def print_flush(msg):
    """Print with immediate flush for real-time output"""
    print(msg, flush=True)
