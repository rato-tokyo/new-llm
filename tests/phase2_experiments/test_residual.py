"""
Test Residual Standard Architecture with CVFP

Tests Residual Standard architecture with configurable layer structure.
Supports Phase 1 (fixed-point context learning) and Phase 2 (token prediction).
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

from transformers import AutoTokenizer
from src.models.new_llm_residual import NewLLMResidual
from src.utils.early_stopping import Phase1EarlyStopping, Phase2EarlyStopping

# Import config from project root
import config


def print_flush(*args, **kwargs):
    """Print with immediate flush to ensure real-time output"""
    print(*args, **kwargs)
    sys.stdout.flush()


def load_ultrachat_samples(num_samples=10):
    """Load UltraChat samples"""
    print_flush(f"Loading {num_samples} samples from UltraChat...")

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    samples = dataset.select(range(num_samples))

    texts = []
    for sample in samples:
        for msg in sample['messages']:
            texts.append(msg['content'])

    return texts


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

    # DDR: Dimension-wise Diversity Regularization
    if config.use_ddr:
        ddr_dim_activity = torch.ones(model.context_dim).to(device)  # 各次元の活性度（EMA）、初期値=1.0
        ddr_boost_count = 0  # ブースト発生回数

    # Train until convergence
    converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool)

    for iteration in range(config.phase1_max_iterations):
        total_loss_value = 0

        # Reset boost count for this iteration
        if config.use_ddr:
            ddr_boost_count = 0

        # LR Schedule (unified for DDR)
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

        for t, token_embed in enumerate(token_embeds):
            # Zero gradients before each token
            if iteration > 0:
                optimizer.zero_grad()

            # Update context one step
            context = model._update_context_one_step(
                token_embed.unsqueeze(0),
                context
            )

            # Loss: Match previous iteration's context (CVFP learning)
            if iteration > 0:
                target_context = fixed_contexts[t].unsqueeze(0)

                # DDR: Dimension-wise Diversity Regularization
                # 低活性次元をブーストして次元崩壊を防ぐ
                if config.use_ddr:
                    with torch.no_grad():
                        # Update dimension-wise activity (EMA of model output's absolute values)
                        # モデルの出力を追跡（教師ではなく、実際の出力の活性を見る）
                        ddr_dim_activity = (
                            config.ddr_momentum * ddr_dim_activity +
                            (1 - config.ddr_momentum) * torch.abs(context.squeeze(0))
                        )

                        # ⚠️ CRITICAL: threshold is FIXED value, NOT mean of all dimensions
                        # 各次元の平均活性（EMA）が threshold 未満の次元をブースト
                        # threshold = ddr_threshold（固定値、例: 0.2）
                        threshold = config.ddr_threshold

                        # 活性が閾値未満の次元にブースト適用
                        boost_mask = ddr_dim_activity < threshold

                        # 修正ベクトル = (1 / 平均活性) * 係数k * 対象ベクトル
                        # 平均活性が低いほど修正が強くなる（最大10倍に制限）
                        inverse_activity = torch.clamp(1.0 / (ddr_dim_activity + 1e-6), max=10.0)
                        boost_amount = torch.where(
                            boost_mask,
                            inverse_activity * target_context.squeeze(0),
                            torch.zeros_like(ddr_dim_activity)
                        )

                        # ブースト発生回数をカウント
                        ddr_boost_count += boost_mask.sum().item()

                    # Apply boost to target (encourage low-activity dimensions)
                    # 低活性次元を使うように教師ベクトルを調整
                    target_context = target_context + config.ddr_boost_weight * boost_amount.unsqueeze(0)

                loss = torch.nn.functional.mse_loss(context, target_context)
                total_loss_value += loss.item()

                # Backprop and update weights for each token
                loss.backward()
                optimizer.step()

                # Check convergence (from config)
                if loss.item() < config.phase1_convergence_threshold:
                    converged_tokens[t] = True

            # Save context for next iteration and carry over to next token
            fixed_contexts[t] = context.detach().squeeze(0)
            context = context.detach()  # Detach to prevent growing computation graph
            context.requires_grad = True  # Re-enable gradients for next token

        # Compute convergence rate
        convergence_rate = converged_tokens.float().mean().item()

        if iteration == 0:
            print_flush(f"Iteration 1/{config.phase1_max_iterations}: Forward pass only (saving contexts)")
        else:
            avg_loss = total_loss_value / len(token_ids)
            log_msg = f"Iteration {iteration+1}/{config.phase1_max_iterations}: Loss={avg_loss:.6f}, Converged={convergence_rate*100:.1f}%, LR={lr:.4f}"

            # DDR boost count (total dimensions boosted across all tokens)
            if config.use_ddr:
                avg_boost_per_token = ddr_boost_count / len(token_ids)
                log_msg += f", DDR_Boost={avg_boost_per_token:.1f}dims/token"

            print_flush(log_msg)

            # Early stopping check
            if early_stopping(convergence_rate):
                print_flush(f"  → Early stopping: Convergence = {convergence_rate*100:.1f}%")
                break

        # Reset convergence tracking for next iteration
        converged_tokens.fill_(False)

    print_flush(f"\nPhase 1 Complete: {converged_tokens.sum()}/{len(token_ids)} tokens converged\n")

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
                    loss = torch.nn.functional.mse_loss(context, fixed_contexts[t].unsqueeze(0))
                    total_loss_value += loss.item()  # Accumulate loss
                    if loss.item() < config.phase1_convergence_threshold:
                        converged_tokens[t] = True

                # Update fixed context
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
                 num_epochs=50, batch_size=32, freeze_context=False, device='cpu'):
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
    parser = argparse.ArgumentParser(description='Test Residual Standard Architecture')
    parser.add_argument('--config', type=str, default='Residual4Layer',
                        help='Config class name (Residual2Layer, Residual4Layer, Residual8Layer)')
    parser.add_argument('--skip-phase2', action='store_true',
                        help='Skip Phase 2 (only run Phase 1)')
    parser.add_argument('--freeze-context', action='store_true',
                        help='Freeze context in Phase 2 (only train token output)')
    args = parser.parse_args()

    # Load configuration from project root config.py
    ConfigClass = getattr(config, args.config, config.Residual4Layer)
    cfg = ConfigClass()

    print_flush("="*70)
    print_flush("Residual Standard Architecture Test")
    print_flush("="*70)
    print_flush(f"\n📋 Configuration: {args.config}")
    print_flush(f"   Edit settings in: config.py (project root)\n")

    print_flush("🏗️  Model Architecture:")
    print_flush(f"   Layer structure: {cfg.layer_structure}")
    print_flush(f"   Context dim: {cfg.context_dim}")
    print_flush(f"   Embed dim: {cfg.embed_dim}")
    print_flush(f"   Hidden dim: {cfg.hidden_dim}\n")

    print_flush("⚙️  Phase 1 Settings (CVFP):")
    print_flush(f"   Max iterations: {cfg.phase1_max_iterations}")
    print_flush(f"   Convergence threshold: {cfg.phase1_convergence_threshold}")
    print_flush(f"   Min converged ratio: {cfg.phase1_min_converged_ratio}")

    # Display LR schedule and DDR settings
    print_flush(f"   LR schedule: {cfg.phase1_lr_warmup} → {cfg.phase1_lr_medium} → {cfg.phase1_lr_finetune}")
    if cfg.use_ddr:
        print_flush(f"   DDR: boost_weight={cfg.ddr_boost_weight}, threshold={cfg.ddr_threshold}\n")
    else:
        print_flush("")

    print_flush("📊 Data Settings:")
    print_flush(f"   Num samples: {cfg.num_samples}")
    print_flush(f"   Train/Val split: {cfg.train_val_split}")
    print_flush(f"   Device: {cfg.device}\n")

    # Load data
    print_flush("Loading data...")
    texts = load_ultrachat_samples(cfg.num_samples)
    all_token_ids, tokenizer = tokenize_texts(texts)

    # Use manual validation data (tokens from training set only)
    train_ids = all_token_ids
    manual_val_path = '/Users/sakajiritomoyoshi/Desktop/git/new-llm/cache/manual_val_tokens.pt'
    if os.path.exists(manual_val_path):
        val_ids = torch.load(manual_val_path)
        print_flush(f"  Train: {len(train_ids)} tokens")
        print_flush(f"  Val:   {len(val_ids)} tokens (manual data, uses training vocab only)")
    else:
        # Fallback to split if manual data not found
        split_idx = int(len(all_token_ids) * cfg.train_val_split)
        train_ids = all_token_ids[:split_idx]
        val_ids = all_token_ids[split_idx:]
        print_flush(f"  Train: {len(train_ids)} tokens")
        print_flush(f"  Val:   {len(val_ids)} tokens (auto-split)")

    print_flush(f"\n{'='*70}")
    print_flush(f"Starting Training: Residual Standard {cfg.layer_structure}")
    print_flush(f"{'='*70}")

    # Create model
    hidden_dim = cfg.embed_dim + cfg.context_dim

    model = NewLLMResidual(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        context_dim=cfg.context_dim,
        hidden_dim=hidden_dim,
        layer_structure=cfg.layer_structure
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
    if not args.skip_phase2:
        val_metrics = phase2_train(model, train_ids, train_contexts, val_ids, val_contexts,
                                    freeze_context=args.freeze_context, device=cfg.device)

        # Final results
        print_flush("\n" + "="*70)
        print_flush("FINAL RESULTS")
        print_flush("="*70)
        print_flush(f"\nResidual Standard {cfg.layer_structure}:")
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
