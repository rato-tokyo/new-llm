"""
Phase 1 Training (Version 5): True self-learning implementation

Key improvements:
1. Model handles all optimization internally
2. Phase1 only feeds tokens to the model
3. Dramatically simplified training loop
4. True separation of concerns
"""

import torch


def train_phase1(model, token_ids, config, device, is_training=True, label=""):
    """
    Phase 1: CVFP Fixed-Point Learning (True Self-Learning Version)

    モデルが完全に自己学習するため、このスクリプトはトークンを順次入力するだけ。

    Args:
        model: The language model (must have enable_cvfp_learning=True)
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
        model.reset_running_stats()

        # Only train context generation layers
        context_params = []
        for name, param in model.named_parameters():
            if 'token_output' not in name and 'token_embedding' not in name:
                param.requires_grad = True
                context_params.append(param)
            else:
                param.requires_grad = False

        # Optimizerを作成し、モデルに設定
        optimizer = torch.optim.Adam(context_params, lr=config.phase1_lr_warmup)
        model.set_phase1_optimizer(optimizer, dist_reg_weight=config.dist_reg_weight)

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

        for t, token_embed in enumerate(token_embeds):
            # Forward pass (モデルが自動で学習する)
            if is_training:
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context.detach() if t > 0 else context
                )
            else:
                with torch.no_grad():
                    context = model._update_context_one_step(
                        token_embed.unsqueeze(0),
                        context
                    )

            current_contexts.append(context.detach())

        # Stack all contexts
        current_contexts_tensor = torch.cat(current_contexts, dim=0)

        # Check convergence and log
        if iteration > 0 and prev_contexts is not None:
            with torch.no_grad():
                token_losses = ((current_contexts_tensor - prev_contexts) ** 2).mean(dim=1)
                converged_tokens = token_losses < config.phase1_convergence_threshold

            # Logging
            convergence_rate = converged_tokens.float().mean().item()

            log_msg = f"Iteration {iteration+1}/{config.phase1_max_iterations}: "
            log_msg += f"Converged={convergence_rate*100:.1f}%"
            if is_training:
                log_msg += f", LR={lr:.4f}"

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
