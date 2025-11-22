"""
Test Residual Standard Architecture with CVFP

Tests Residual Standard architecture with configurable layer structure.
Supports Phase 1 (fixed-point context learning) and Phase 2 (token prediction).
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

from transformers import AutoTokenizer
from src.models.new_llm_residual import NewLLMResidual
from src.utils.early_stopping import Phase1EarlyStopping, Phase2EarlyStopping

# Import config from project root
import config

# Conditional import for datasets (only needed for UltraChat)
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def print_flush(*args, **kwargs):
    """Print with immediate flush to ensure real-time output"""
    print(*args, **kwargs)
    sys.stdout.flush()


def load_text_file(file_path):
    """Load text from a single file"""
    print_flush(f"Loading text from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split by double newlines to get paragraphs, or by single newlines
    texts = [t.strip() for t in text.split('\n\n') if t.strip()]
    if not texts:
        texts = [t.strip() for t in text.split('\n') if t.strip()]

    print_flush(f"  Loaded {len(texts)} text segments")
    return texts


def load_text_directory(dir_path, max_files=None):
    """Load all text files from a directory"""
    print_flush(f"Loading text files from: {dir_path}")

    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    texts = []
    txt_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.txt')])

    if max_files:
        txt_files = txt_files[:max_files]

    for filename in txt_files:
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                texts.append(text)

    print_flush(f"  Loaded {len(texts)} files")
    return texts


def load_ultrachat_samples(num_samples=10):
    """Load UltraChat samples"""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets module not available. Install with: pip install datasets")

    print_flush(f"Loading {num_samples} samples from UltraChat...")

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    samples = dataset.select(range(num_samples))

    texts = []
    for sample in samples:
        for msg in sample['messages']:
            texts.append(msg['content'])

    return texts


def load_train_data(cfg):
    """Load training data based on config"""
    if cfg.train_data_source == "ultrachat":
        return load_ultrachat_samples(cfg.num_samples)
    elif cfg.train_data_source == "text_file":
        return load_text_file(cfg.train_text_file)
    elif cfg.train_data_source == "text_dir":
        return load_text_directory(cfg.train_text_dir, cfg.num_samples)
    else:
        raise ValueError(f"Unknown train_data_source: {cfg.train_data_source}")


def load_val_data(cfg, tokenizer):
    """Load validation data based on config

    Returns:
        torch.Tensor: Token IDs for validation data
    """
    if cfg.val_data_source == "manual":
        # Load pre-tokenized manual validation data
        if os.path.exists(cfg.manual_val_path):
            print_flush(f"Loading manual validation data: {cfg.manual_val_path}")
            val_ids = torch.load(cfg.manual_val_path)
            print_flush(f"  Loaded {len(val_ids)} validation tokens")
            return val_ids
        else:
            raise FileNotFoundError(f"Manual validation file not found: {cfg.manual_val_path}")

    elif cfg.val_data_source == "text_file":
        texts = load_text_file(cfg.val_text_file)
    elif cfg.val_data_source == "text_dir":
        texts = load_text_directory(cfg.val_text_dir)
    elif cfg.val_data_source == "auto_split":
        # This will be handled by caller (split from training data)
        return None
    else:
        raise ValueError(f"Unknown val_data_source: {cfg.val_data_source}")

    # Tokenize validation texts
    token_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)

    return torch.tensor(token_ids, dtype=torch.long)


def tokenize_texts(texts):
    """Tokenize texts using GPT-2 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    token_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)

    return torch.tensor(token_ids, dtype=torch.long), tokenizer


def phase1_train(model, token_ids, config, device='cpu'):
    """Phase 1: Fixed-point context learning with CVFP (Context Vector Fixed-Point Property)

    CVFP Principle:
    - Iteration 1: Forward pass only, save contexts
    - Iteration 2+: Learn so that context[t] matches previous context[t-period]
    - Example: "red apple" → "red apple red" should output same context as "red"

    Optimizations:
    - LR Schedule: Higher LR early, lower later (from config)
    - Batch processing: Process multiple tokens in parallel after convergence
    - Relaxed threshold: Configurable (default 0.02)
    """
    print_flush("\n" + "="*70)
    print_flush("PHASE 1: Fixed-Point Context Learning (CVFP) - Train")
    print_flush("="*70)

    model.to(device)
    model.train()

    # Optimizer for Phase 1 - ONLY train context generation layers (exclude token_output)
    context_params = [p for name, p in model.named_parameters() if 'token_output' not in name]
    optimizer = torch.optim.Adam(context_params, lr=config.phase1_lr_warmup)

    # Early stopping - from config
    # Stops if convergence >= threshold OR drops twice
    early_stopping = Phase1EarlyStopping(
        convergence_threshold=config.phase1_min_converged_ratio,
        min_delta=0.01  # 1% drop threshold
    )

    # Token embeddings
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # Fixed contexts storage
    fixed_contexts = torch.zeros(len(token_ids), model.context_dim).to(device)

    # Train until convergence
    converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool)

    for iteration in range(config.phase1_max_iterations):
        total_loss_value = 0
        total_cvfp_loss = 0
        total_dist_loss = 0

        # Save previous iteration's contexts for comparison
        prev_fixed_contexts = fixed_contexts.clone()

        # LR Schedule
        if iteration <= 3:
            lr = config.phase1_lr_warmup
        elif iteration <= 8:
            lr = config.phase1_lr_medium
        else:
            lr = config.phase1_lr_finetune

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Process entire sequence with context carry-over
        context = torch.zeros(1, model.context_dim).to(device)

        # Collect all contexts for distribution regularization
        all_contexts = []

        for t, token_embed in enumerate(token_embeds):
            # Update context one step
            context = model._update_context_one_step(
                token_embed.unsqueeze(0),
                context
            )

            # Collect contexts for distribution regularization
            all_contexts.append(context)

            # Carry over to next token (no saving yet)
            context = context.detach()  # Detach to prevent growing computation graph
            context.requires_grad = True  # Re-enable gradients for next token

        # Compute loss and backprop (after processing all tokens)
        if iteration > 0:
            optimizer.zero_grad()

            # Stack all contexts: [num_tokens, context_dim]
            all_contexts_tensor = torch.cat(all_contexts, dim=0)  # [num_tokens, context_dim]

            # CVFP loss: match previous iteration's contexts
            cvfp_loss = torch.nn.functional.mse_loss(all_contexts_tensor, prev_fixed_contexts)
            total_cvfp_loss = cvfp_loss.item()

            # Distribution regularization loss (if enabled)
            if getattr(config, 'use_distribution_reg', True):
                # Goal: each dimension (across all tokens) should have mean=0, var=1
                dim_mean = all_contexts_tensor.mean(dim=0)  # [context_dim]
                dim_var = all_contexts_tensor.var(dim=0)    # [context_dim]

                # Penalize deviation from N(0,1)
                mean_penalty = (dim_mean ** 2).mean()
                var_penalty = ((dim_var - 1.0) ** 2).mean()
                dist_loss = mean_penalty + var_penalty
                total_dist_loss = dist_loss.item()

                # Combine losses
                dist_weight = getattr(config, 'dist_reg_weight', 0.2)
                total_loss = (1 - dist_weight) * cvfp_loss + dist_weight * dist_loss
            else:
                # Pure CVFP (no distribution regularization)
                total_dist_loss = 0.0
                total_loss = cvfp_loss

            total_loss_value = total_loss.item()

            # Backprop and update
            total_loss.backward()
            optimizer.step()

            # Save contexts for next iteration
            fixed_contexts = all_contexts_tensor.detach()

            # Check convergence for each token
            with torch.no_grad():
                token_losses = ((all_contexts_tensor - prev_fixed_contexts) ** 2).mean(dim=1)
                converged_tokens = token_losses < config.phase1_convergence_threshold

        # Compute convergence rate
        convergence_rate = converged_tokens.float().mean().item()

        if iteration == 0:
            # Save contexts from first iteration
            all_contexts_tensor = torch.cat(all_contexts, dim=0)
            fixed_contexts = all_contexts_tensor.detach()
            print_flush(f"Iteration 1/{config.phase1_max_iterations}: Forward pass only (saving contexts)")
        else:
            log_msg = f"Iteration {iteration+1}/{config.phase1_max_iterations}: "
            log_msg += f"Loss={total_loss_value:.6f} (CVFP={total_cvfp_loss:.6f}, Dist={total_dist_loss:.6f}), "
            log_msg += f"Converged={convergence_rate*100:.1f}%, LR={lr:.4f}"

            print_flush(log_msg)

            # Early stopping check
            if early_stopping(convergence_rate):
                print_flush(f"  → Early stopping: Convergence = {convergence_rate*100:.1f}%")
                break

        # Reset convergence tracking for next iteration
        converged_tokens.fill_(False)

    print_flush(f"\nPhase 1 Complete: {converged_tokens.sum()}/{len(token_ids)} tokens converged\n")

    # Save model checkpoint
    checkpoint_path = '/Users/sakajiritomoyoshi/Desktop/git/new-llm/cache/phase1_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'fixed_contexts': fixed_contexts,
    }, checkpoint_path)
    print_flush(f"Model checkpoint saved: {checkpoint_path}\n")

    # Automatic fixed-point analysis
    analyze_fixed_points(fixed_contexts, label="Train")

    return fixed_contexts.detach()


def phase1_train_batch(model, batch_token_ids, config, device='cpu'):
    """Phase 1: Fixed-point context learning with batch processing

    Args:
        model: The model to train
        batch_token_ids: List of token_ids tensors (variable length)
        config: Configuration object
        device: Device to use

    Returns:
        List of fixed_contexts for each sample
    """
    print_flush("\n" + "="*70)
    print_flush("PHASE 1: Fixed-Point Context Learning (CVFP) - Train (BATCH)")
    print_flush("="*70)

    model.to(device)
    model.train()

    # Only train context generation layers
    context_params = [p for name, p in model.named_parameters() if 'token_output' not in name]
    optimizer = torch.optim.Adam(context_params, lr=config.phase1_lr_warmup)

    # Early stopping - from config
    # Stops if convergence >= threshold OR drops twice
    early_stopping = Phase1EarlyStopping(
        convergence_threshold=config.phase1_min_converged_ratio,
        min_delta=0.01  # 1% drop threshold
    )

    batch_size = len(batch_token_ids)
    print_flush(f"Batch size: {batch_size} samples")

    # Pad sequences to same length
    max_len = max(len(ids) for ids in batch_token_ids)
    print_flush(f"Max sequence length: {max_len} tokens")

    # Prepare batch data
    batch_embeds = []
    batch_masks = []  # Track valid positions (not padding)

    for token_ids in batch_token_ids:
        # Pad to max_len
        padded_ids = torch.cat([
            token_ids,
            torch.zeros(max_len - len(token_ids), dtype=torch.long)
        ]).to(device)

        # Create mask (1 = valid, 0 = padding)
        mask = torch.cat([
            torch.ones(len(token_ids)),
            torch.zeros(max_len - len(token_ids))
        ]).bool()

        # Get embeddings
        with torch.no_grad():
            embeds = model.token_embedding(padded_ids.unsqueeze(0))
            embeds = model.embed_norm(embeds).squeeze(0)

        batch_embeds.append(embeds)
        batch_masks.append(mask)

    # Stack into batch tensors
    batch_embeds = torch.stack(batch_embeds)  # [batch_size, max_len, embed_dim]
    batch_masks = torch.stack(batch_masks)    # [batch_size, max_len]

    # Fixed contexts storage for all samples
    batch_fixed_contexts = torch.zeros(batch_size, max_len, model.context_dim).to(device)

    # LDR: Running mean for target adjustment
    if config.use_ldr:
        ldr_running_mean = torch.zeros(model.context_dim).to(device)

    # Convergence tracking for all samples
    batch_converged = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Training loop
    for iteration in range(config.phase1_max_iterations):
        total_loss_value = 0
        total_valid_tokens = 0

        # LR Schedule (unified for DDR)
        if iteration <= 3:
            lr = config.phase1_lr_warmup
        elif iteration <= 8:
            lr = config.phase1_lr_medium
        else:
            lr = config.phase1_lr_finetune

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Process all sequences in parallel
        contexts = torch.zeros(batch_size, 1, model.context_dim).to(device)

        for t in range(max_len):
            # Get current token embeddings for all samples
            token_embeds_t = batch_embeds[:, t, :]  # [batch_size, embed_dim]

            # Zero gradients
            if iteration > 0:
                optimizer.zero_grad()

            # Update contexts for all samples in parallel
            # _update_context_one_step expects [batch, embed_dim] and [batch, context_dim]
            contexts = model._update_context_one_step(token_embeds_t, contexts.squeeze(1)).unsqueeze(1)

            # Compute loss only for valid (non-padding) tokens
            if iteration > 0:
                # Get valid samples for this position
                valid_mask = batch_masks[:, t]  # [batch_size]

                if valid_mask.any():
                    # Compute loss for valid samples
                    valid_contexts = contexts[valid_mask]
                    target_contexts = batch_fixed_contexts[valid_mask, t].unsqueeze(1)

                    # LDR: Adjust targets to push away from running mean
                    if config.use_ldr:
                        with torch.no_grad():
                            # Update running mean (EMA) - average across batch
                            batch_mean = valid_contexts.mean(dim=0).squeeze(0)
                            ldr_running_mean = (
                                config.ldr_momentum * ldr_running_mean +
                                (1 - config.ldr_momentum) * batch_mean
                            )

                        # Push targets away from mean
                        deviation = target_contexts.squeeze(1) - ldr_running_mean
                        deviation_norm = torch.norm(deviation, p=2, dim=-1, keepdim=True)
                        deviation_unit = deviation / (deviation_norm + 1e-6)
                        target_contexts = target_contexts + config.ldr_weight * deviation_unit.unsqueeze(1)

                    loss = torch.nn.functional.mse_loss(valid_contexts, target_contexts)
                    total_loss_value += loss.item() * valid_mask.sum().item()
                    total_valid_tokens += valid_mask.sum().item()

                    # Backprop and update
                    loss.backward()
                    optimizer.step()

                    # Check convergence for valid tokens (use original targets, not LDR-adjusted)
                    original_targets = batch_fixed_contexts[valid_mask, t].unsqueeze(1)
                    per_sample_loss = torch.nn.functional.mse_loss(
                        valid_contexts, original_targets, reduction='none'
                    ).mean(dim=(1, 2))

                    converged_mask = per_sample_loss < config.phase1_convergence_threshold
                    batch_converged[valid_mask, t] = converged_mask

            # Save contexts for next iteration
            batch_fixed_contexts[:, t, :] = contexts.detach().squeeze(1)

            # Carry over to next token (detach to prevent graph growth)
            contexts = contexts.detach()
            contexts.requires_grad = True

        # Compute convergence rate (only for valid tokens)
        valid_tokens_mask = batch_masks.flatten()
        converged_valid = batch_converged.flatten()[valid_tokens_mask]
        convergence_rate = converged_valid.float().mean().item()

        if iteration == 0:
            print_flush(f"Iteration 1/{config.phase1_max_iterations}: Forward pass only (saving contexts)")
        else:
            avg_loss = total_loss_value / total_valid_tokens if total_valid_tokens > 0 else 0
            print_flush(f"Iteration {iteration+1}/{config.phase1_max_iterations}: Loss={avg_loss:.6f}, Converged={convergence_rate*100:.1f}%, LR={lr:.4f}")

            # Early stopping check
            if early_stopping(convergence_rate):
                print_flush(f"  → Early stopping: Convergence = {convergence_rate*100:.1f}%")
                break

        # Reset convergence tracking
        batch_converged.fill_(False)

    print_flush(f"\nPhase 1 Complete (Batch)\n")

    # Extract fixed contexts for each sample (remove padding)
    result_contexts = []
    for i, token_ids in enumerate(batch_token_ids):
        contexts = batch_fixed_contexts[i, :len(token_ids)].detach()
        result_contexts.append(contexts)

    return result_contexts


def compute_fixed_contexts(model, token_ids, config, device='cpu'):
    """Compute fixed-point contexts without training (evaluation only)

    This function computes fixed-point contexts for validation data
    using the already-trained context generation layers.
    """
    print_flush("\n" + "="*70)
    print_flush("PHASE 1: Fixed-Point Context Learning (CVFP) - Val")
    print_flush("="*70)

    model.to(device)
    model.eval()

    # Token embeddings
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # Fixed contexts storage
    fixed_contexts = torch.zeros(len(token_ids), model.context_dim).to(device)

    # Convergence tracking
    converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool)

    for iteration in range(config.phase1_max_iterations):
        total_loss_value = 0.0  # Track total loss across all tokens

        # Save previous iteration's contexts for comparison
        prev_fixed_contexts = fixed_contexts.clone()

        # Process entire sequence with context carry-over
        context = torch.zeros(1, model.context_dim).to(device)

        for t, token_embed in enumerate(token_embeds):
            with torch.no_grad():
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context
                )

                # Check convergence (compare to previous iteration)
                if iteration > 0:
                    loss = torch.nn.functional.mse_loss(context, prev_fixed_contexts[t].unsqueeze(0))
                    total_loss_value += loss.item()  # Accumulate loss
                    if loss.item() < config.phase1_convergence_threshold:
                        converged_tokens[t] = True

                # Update fixed context for next iteration
                fixed_contexts[t] = context.squeeze(0)

        convergence_rate = converged_tokens.float().mean().item()

        if iteration == 0:
            print_flush(f"Iteration 1/{config.phase1_max_iterations}: Forward pass only (saving contexts)")
        else:
            avg_loss = total_loss_value / len(token_ids)
            print_flush(f"Iteration {iteration+1}/{config.phase1_max_iterations}: Loss={avg_loss:.6f}, Converged={convergence_rate*100:.1f}%")

            # Stop if converged
            if convergence_rate >= config.phase1_min_converged_ratio:
                print_flush(f"  → Early stopping: Convergence = {convergence_rate*100:.1f}%")
                break

        # Reset for next iteration
        converged_tokens.fill_(False)

    print_flush(f"\nPhase 1 Complete: {converged_tokens.sum()}/{len(token_ids)} tokens converged\n")

    # Automatic fixed-point analysis
    analyze_fixed_points(fixed_contexts, label="Val")

    return fixed_contexts.detach()


def analyze_fixed_points(contexts, label="", token_ids=None):
    """Analyze fixed-point contexts for degenerate solutions and statistics

    Args:
        contexts: Fixed-point context vectors
        label: Label for this analysis (Train/Val)
        token_ids: Token IDs corresponding to contexts (for singular vector analysis)

    Returns:
        effective_rank (float): Effective rank of the context matrix
    """
    print_flush(f"\n{'='*70}")
    print_flush(f"FIXED-POINT ANALYSIS{' - ' + label if label else ''}")
    print_flush(f"{'='*70}")

    # 1. Global Attractor Detection
    print_flush(f"\n1. Global Attractor Detection:")
    n_tokens = len(contexts)
    sample_size = min(100, n_tokens * (n_tokens - 1) // 2)
    indices = torch.randperm(n_tokens)[:min(sample_size * 2, n_tokens)]

    l2_distances = []
    cosine_sims = []
    for i in range(0, len(indices) - 1, 2):
        idx1, idx2 = indices[i], indices[i + 1]
        ctx1, ctx2 = contexts[idx1], contexts[idx2]
        l2_dist = torch.norm(ctx1 - ctx2).item()
        l2_distances.append(l2_dist)
        cos_sim = torch.nn.functional.cosine_similarity(
            ctx1.unsqueeze(0), ctx2.unsqueeze(0)
        ).item()
        cosine_sims.append(cos_sim)

    avg_l2 = np.mean(l2_distances)
    avg_cosine = np.mean(cosine_sims)

    print_flush(f"  Avg L2 Distance: {avg_l2:.6f} (Range: [{min(l2_distances):.6f}, {max(l2_distances):.6f}])")
    print_flush(f"  Avg Cosine Sim:  {avg_cosine:.6f} (Range: [{min(cosine_sims):.6f}, {max(cosine_sims):.6f}])")

    if avg_l2 < 0.01 and avg_cosine > 0.999:
        print_flush(f"  ⚠️  DEGENERATE: Global Attractor detected")
    else:
        print_flush(f"  ✅ Token-specific fixed points")

    # 2. Zero Solution Detection
    print_flush(f"\n2. Zero Solution Detection:")
    norms = torch.norm(contexts, dim=1)
    avg_norm = norms.mean().item()
    print_flush(f"  Avg Norm: {avg_norm:.6f} (Range: [{norms.min().item():.6f}, {norms.max().item():.6f}])")

    if avg_norm < 0.1:
        print_flush(f"  ⚠️  DEGENERATE: Zero Solution detected")
    else:
        print_flush(f"  ✅ Non-zero contexts")

    # 3. Distribution Statistics
    print_flush(f"\n3. Distribution Statistics:")
    print_flush(f"  Norm - Mean: {norms.mean().item():.4f}, Median: {norms.median().item():.4f}, Std: {norms.std().item():.4f}")

    # Pairwise distances
    pairwise_dists = []
    for i in range(0, len(indices) - 1, 2):
        idx1, idx2 = indices[i], indices[i + 1]
        dist = torch.norm(contexts[idx1] - contexts[idx2]).item()
        pairwise_dists.append(dist)
    print_flush(f"  Pairwise Dist - Mean: {np.mean(pairwise_dists):.4f}, Median: {np.median(pairwise_dists):.4f}")

    # Sparsity
    near_zero = (torch.abs(contexts) < 0.01).float().mean().item()
    print_flush(f"  Sparsity: {near_zero*100:.2f}% of values < 0.01")

    # 4. Information Content (Effective Rank)
    print_flush(f"\n4. Information Content:")
    U, S, V = torch.svd(contexts)
    S_norm = S / S.sum()
    effective_rank = 1.0 / (S_norm ** 2).sum().item()
    print_flush(f"  Effective Rank: {effective_rank:.2f} / {len(S)} (max)")
    print_flush(f"  Top 5 Singular Values: {S[:5].tolist()}")

    if effective_rank < len(S) * 0.1:
        print_flush(f"  ⚠️  Low diversity in fixed points")
    else:
        print_flush(f"  ✅ Good diversity")

    # 5. Singular Vector Analysis (if token_ids provided)
    if token_ids is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        print_flush(f"\n5. Singular Vector Analysis:")
        print_flush(f"  Analyzing which tokens contribute to top singular vectors...")

        # Analyze top 2 singular vectors
        for vec_idx in range(min(2, len(S))):
            print_flush(f"\n  --- Singular Vector {vec_idx+1} (Value: {S[vec_idx].item():.2f}) ---")

            # Project contexts onto this singular vector
            projections = contexts @ V[:, vec_idx]  # Shape: [num_tokens]

            # Find tokens with highest projections (absolute value)
            abs_projections = torch.abs(projections)
            top_indices = torch.topk(abs_projections, k=min(5, len(projections))).indices

            print_flush(f"  Top 5 tokens contributing to SV{vec_idx+1}:")
            for rank, idx in enumerate(top_indices):
                token_id = token_ids[idx].item()
                token_text = tokenizer.decode([token_id])
                projection = projections[idx].item()
                print_flush(f"    {rank+1}. Token {token_id} '{token_text}': proj={projection:.3f}")

    print_flush(f"{'='*70}\n")

    return effective_rank


def phase2_train(model, token_ids, fixed_contexts, val_ids, val_contexts,
                 num_epochs, batch_size, freeze_context=False, device='cpu'):
    """Phase 2: Token prediction training with validation"""
    print_flush("\n" + "="*70)
    print_flush("PHASE 2: Token Prediction Training")
    if freeze_context:
        print_flush("  Mode: Freeze context (only train token output)")
    else:
        print_flush("  Mode: Train all layers")
    print_flush("="*70)

    model.to(device)
    model.train()

    # Freeze context generation layers if requested
    if freeze_context:
        for name, param in model.named_parameters():
            if 'token_output' not in name:
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_flush(f"  Trainable parameters: {trainable_params:,} (token_output only)")
    else:
        trainable_params = model.count_parameters()
        print_flush(f"  Trainable parameters: {trainable_params:,} (all layers)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    # Early stopping
    early_stopping = Phase2EarlyStopping(patience=5, min_delta=0.0, restore_best=True)

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total_tokens = 0

        for i in range(0, len(token_ids) - 1, batch_size):
            batch_end = min(i + batch_size, len(token_ids) - 1)
            batch_contexts = fixed_contexts[i:batch_end].to(device)
            batch_targets = token_ids[i+1:batch_end+1].to(device)

            optimizer.zero_grad()

            logits = model.token_output(batch_contexts)
            loss = nn.functional.cross_entropy(logits, batch_targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_contexts)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_targets).sum().item()
            total_tokens += len(batch_contexts)

        train_loss = total_loss / total_tokens
        train_acc = correct / total_tokens * 100
        train_ppl = np.exp(train_loss)

        # Validation
        model.eval()
        val_total_loss = 0
        val_correct = 0
        val_total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(val_ids) - 1, batch_size):
                batch_end = min(i + batch_size, len(val_ids) - 1)
                batch_contexts = val_contexts[i:batch_end].to(device)
                batch_targets = val_ids[i+1:batch_end+1].to(device)

                logits = model.token_output(batch_contexts)
                loss = nn.functional.cross_entropy(logits, batch_targets)

                val_total_loss += loss.item() * len(batch_contexts)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_targets).sum().item()
                val_total_tokens += len(batch_contexts)

        val_loss = val_total_loss / val_total_tokens
        val_acc = val_correct / val_total_tokens * 100
        val_ppl = np.exp(val_loss)

        print_flush(f"  Epoch {epoch+1}/{num_epochs}:")
        print_flush(f"    Train: Loss={train_loss:.4f}, PPL={train_ppl:.2f}, Acc={train_acc:.1f}%")
        print_flush(f"    Val:   Loss={val_loss:.4f}, PPL={val_ppl:.2f}, Acc={val_acc:.1f}%")

        # Early stopping check
        if early_stopping(val_loss, val_ppl, model):
            print_flush(f"  → Early stopping: Best Val PPL = {early_stopping.best_ppl:.2f}")
            break

    # Restore best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    print_flush("\nPhase 2 Complete\n")

    # Return best validation metrics
    return {
        'loss': early_stopping.best_loss,
        'ppl': early_stopping.best_ppl,
        'acc': val_acc  # Use last val_acc (approximation)
    }




def main():
    # Load configuration from config.py
    cfg = config.ResidualConfig()

    print_flush("="*70)
    print_flush("Residual Standard Architecture Test")
    print_flush("="*70)
    print_flush(f"\n📋 Configuration: config.py")
    print_flush(f"   Edit settings in: config.py (project root)\n")

    print_flush("🏗️  Model Architecture:")
    print_flush(f"   Num layers: {cfg.num_layers} (structure: {[1] * cfg.num_layers})")
    print_flush(f"   Context dim: {cfg.context_dim}")
    print_flush(f"   Embed dim: {cfg.embed_dim}")
    print_flush(f"   Hidden dim: {cfg.hidden_dim}\n")

    print_flush("⚙️  Phase 1 Settings (CVFP):")
    print_flush(f"   Max iterations: {cfg.phase1_max_iterations}")
    print_flush(f"   Convergence threshold: {cfg.phase1_convergence_threshold}")
    print_flush(f"   Min converged ratio: {cfg.phase1_min_converged_ratio}")

    # Display LR schedule and Distribution Regularization settings
    print_flush(f"   LR schedule: {cfg.phase1_lr_warmup} → {cfg.phase1_lr_medium} → {cfg.phase1_lr_finetune}")
    if getattr(cfg, 'use_distribution_reg', True):
        dist_weight = getattr(cfg, 'dist_reg_weight', 0.2)
        print_flush(f"   Distribution Reg: weight={dist_weight} ({int((1-dist_weight)*100)}% CVFP, {int(dist_weight*100)}% Dist)\n")
    else:
        print_flush("   Pure CVFP (no distribution regularization)\n")

    print_flush("⚙️  Phase 2 Settings:")
    print_flush(f"   Skip Phase 2: {cfg.skip_phase2}")
    if not cfg.skip_phase2:
        print_flush(f"   Freeze context: {cfg.freeze_context}")
        print_flush(f"   Learning rate: {cfg.phase2_learning_rate}")
        print_flush(f"   Epochs: {cfg.phase2_epochs}")
        print_flush(f"   Batch size: {cfg.phase2_batch_size}\n")
    else:
        print_flush("")

    print_flush("📊 Data Settings:")
    print_flush(f"   Train source: {cfg.train_data_source}")
    print_flush(f"   Val source: {cfg.val_data_source}")
    if cfg.train_data_source == "ultrachat":
        print_flush(f"   Num samples: {cfg.num_samples}")
    print_flush(f"   Device: {cfg.device}\n")

    # Load training data
    print_flush("Loading training data...")
    train_texts = load_train_data(cfg)
    all_token_ids, tokenizer = tokenize_texts(train_texts)
    print_flush(f"  Loaded {len(train_texts)} text segments → {len(all_token_ids)} tokens")

    # Load validation data
    val_ids = load_val_data(cfg, tokenizer)

    if val_ids is None:
        # auto_split mode: split from training data
        print_flush("Using auto-split for validation data...")
        split_idx = int(len(all_token_ids) * cfg.train_val_split)
        train_ids = all_token_ids[:split_idx]
        val_ids = all_token_ids[split_idx:]
        print_flush(f"  Train: {len(train_ids)} tokens")
        print_flush(f"  Val:   {len(val_ids)} tokens (auto-split)")
    else:
        # Separate validation data loaded
        train_ids = all_token_ids
        print_flush(f"  Train: {len(train_ids)} tokens")
        print_flush(f"  Val:   {len(val_ids)} tokens ({cfg.val_data_source})")

    print_flush(f"\n{'='*70}")
    print_flush(f"Starting Training: Residual Standard ({cfg.num_layers} layers)")
    print_flush(f"{'='*70}")

    # Create model
    hidden_dim = cfg.embed_dim + cfg.context_dim
    layer_structure = [1] * cfg.num_layers  # Convert num_layers to structure

    model = NewLLMResidual(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        context_dim=cfg.context_dim,
        hidden_dim=hidden_dim,
        layer_structure=layer_structure
    )

    print_flush(f"  Parameters: {model.count_parameters():,}")

    # Phase 1: Fixed-point learning
    # Train: Learn context generation layers
    # Val: Compute fixed contexts only (evaluation of generalization)
    train_contexts = phase1_train(model, train_ids, cfg, device=cfg.device)
    val_contexts = compute_fixed_contexts(model, val_ids, cfg, device=cfg.device)

    # Analyze fixed points for degenerate solutions
    train_effective_rank = analyze_fixed_points(train_contexts, label="Train", token_ids=train_ids)
    val_effective_rank = analyze_fixed_points(val_contexts, label="Val", token_ids=val_ids)

    # Check if Phase 1 succeeded (minimum Effective Rank requirements)
    MIN_TRAIN_RANK = 50.0  # Minimum 50/256 (20%)
    MIN_VAL_RANK = 20.0    # Minimum 20/256 (8%)

    phase1_success = (train_effective_rank >= MIN_TRAIN_RANK and
                      val_effective_rank >= MIN_VAL_RANK)

    if not phase1_success:
        print_flush("\n" + "="*70)
        print_flush("⚠️  PHASE 1 FAILED - DIMENSION COLLAPSE DETECTED")
        print_flush("="*70)
        print_flush(f"\n  Train Effective Rank: {train_effective_rank:.2f}/256 (required: >= {MIN_TRAIN_RANK})")
        print_flush(f"  Val Effective Rank:   {val_effective_rank:.2f}/256 (required: >= {MIN_VAL_RANK})")
        print_flush(f"\n  ❌ Phase 2 skipped. Fix dimension collapse first.")
        print_flush(f"  ❌ See CLAUDE.md for Phase 1/2 execution policy.")
        print_flush("\n" + "="*70)
        return

    # Phase 1 succeeded
    print_flush("\n" + "="*70)
    print_flush("✅ PHASE 1 SUCCESSFUL")
    print_flush("="*70)
    print_flush(f"\n  Train Effective Rank: {train_effective_rank:.2f}/256")
    print_flush(f"  Val Effective Rank:   {val_effective_rank:.2f}/256")
    print_flush(f"\n  Proceeding to Phase 2...")
    print_flush("="*70)

    # Phase 2: Token prediction (skip if requested)
    if not cfg.skip_phase2:
        val_metrics = phase2_train(model, train_ids, train_contexts, val_ids, val_contexts,
                                    num_epochs=cfg.phase2_epochs,
                                    batch_size=cfg.phase2_batch_size,
                                    freeze_context=cfg.freeze_context,
                                    device=cfg.device)

        # Final results
        print_flush("\n" + "="*70)
        print_flush("FINAL RESULTS")
        print_flush("="*70)
        print_flush(f"\nResidual Standard ({cfg.num_layers} layers):")
        print_flush(f"  Context dim: {cfg.context_dim}, Embed dim: {cfg.embed_dim}")
        print_flush(f"  Parameters: {model.count_parameters():,}")
        print_flush(f"  Best Val Loss: {val_metrics['loss']:.2f}")
        print_flush(f"  Best Val PPL:  {val_metrics['ppl']:.2f}")
        print_flush(f"  Best Val Acc:  {val_metrics['acc']:.1f}%")
    else:
        print_flush("\n" + "="*70)
        print_flush("PHASE 1 ONLY - SKIPPING PHASE 2")
        print_flush("="*70)


if __name__ == '__main__':
    main()
