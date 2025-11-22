"""
Phase 1: Context Vector Fixed-Point Property (CVFP) Training

Unified implementation for both training and validation.
Fixes critical bugs from the old implementation:
- Proper CVFP iteration logic (compare with previous iteration's final contexts)
- Correct distribution regularization (unbiased=False for population variance)
- Single implementation for Train/Val (no code duplication)
"""

import torch
import torch.nn.functional as F


def train_phase1(model, token_ids, config, device, is_training=True, label=""):
    """
    Phase 1: CVFP Fixed-Point Learning

    Learn context generation parameters so that contexts converge to fixed points.

    Args:
        model: The language model
        token_ids: Input token IDs [num_tokens]
        config: Configuration object
        device: torch device
        is_training: If True, train parameters; if False, eval only
        label: Label for logging (e.g., "Train", "Val")

    Returns:
        fixed_contexts: Converged context vectors [num_tokens, context_dim]
    """
    print_flush(f"\n{'='*70}")
    print_flush(f"PHASE 1: Fixed-Point Context Learning (CVFP){' - ' + label if label else ''}")
    print_flush(f"{'='*70}")

    model.to(device)

    # Setup training or evaluation mode
    if is_training:
        model.train()

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

    # Compute token embeddings
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # Initialize storage for fixed contexts
    fixed_contexts = torch.zeros(len(token_ids), model.context_dim).to(device)
    converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool).to(device)

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

        # Forward pass: compute all contexts
        context = torch.zeros(1, model.context_dim).to(device)
        all_contexts = []

        for t, token_embed in enumerate(token_embeds):
            if is_training:
                # Training: enable gradients
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context
                )
                all_contexts.append(context)
                context = context.detach()
                context.requires_grad = True
            else:
                # Evaluation: no gradients
                with torch.no_grad():
                    context = model._update_context_one_step(
                        token_embed.unsqueeze(0),
                        context
                    )
                    all_contexts.append(context.clone())

        # Stack all contexts
        all_contexts_tensor = torch.cat(all_contexts, dim=0)

        # Compute loss and update (iteration > 0 only)
        if iteration > 0:
            if is_training:
                optimizer.zero_grad()

            # CVFP loss: match previous iteration's contexts
            cvfp_loss = F.mse_loss(all_contexts_tensor, fixed_contexts)
            total_cvfp_loss = cvfp_loss.item()

            # Distribution regularization (if enabled and training)
            if getattr(config, 'use_distribution_reg', True) and is_training:
                # Goal: each dimension (across all tokens) follows N(0,1)
                dim_mean = all_contexts_tensor.mean(dim=0)  # [context_dim]
                # CRITICAL: Use unbiased=False for population variance
                dim_var = all_contexts_tensor.var(dim=0, unbiased=False)  # [context_dim]

                # Penalize deviation from N(0,1)
                mean_penalty = (dim_mean ** 2).mean()
                var_penalty = ((dim_var - 1.0) ** 2).mean()
                dist_loss = mean_penalty + var_penalty
                total_dist_loss = dist_loss.item()

                # Combine losses
                dist_weight = getattr(config, 'dist_reg_weight', 0.2)
                total_loss = (1 - dist_weight) * cvfp_loss + dist_weight * dist_loss
            else:
                total_dist_loss = 0.0
                total_loss = cvfp_loss

            total_loss_value = total_loss.item()

            # Backprop (training only)
            if is_training:
                total_loss.backward()
                optimizer.step()

            # Check convergence
            with torch.no_grad():
                token_losses = ((all_contexts_tensor - fixed_contexts) ** 2).mean(dim=1)
                converged_tokens = token_losses < config.phase1_convergence_threshold

            # Logging
            convergence_rate = converged_tokens.float().mean().item()

            log_msg = f"Iteration {iteration+1}/{config.phase1_max_iterations}: "
            if is_training and getattr(config, 'use_distribution_reg', True):
                log_msg += f"Loss={total_loss_value:.6f} (CVFP={total_cvfp_loss:.6f}, Dist={total_dist_loss:.6f}), "
            else:
                log_msg += f"Loss={total_cvfp_loss:.6f}, "
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

        # Update fixed contexts for next iteration (CRITICAL: after loss computation)
        fixed_contexts = all_contexts_tensor.detach()

    print_flush(f"\nPhase 1 Complete: {converged_tokens.sum()}/{len(token_ids)} tokens converged\n")

    return fixed_contexts.detach()


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