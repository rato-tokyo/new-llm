"""
Standard Phase 1 Test - 81.7% Effective Rank (Validation)

Uses fixed datasets:
- Training: 6400 tokens (UltraChat 50 samples)
- Validation: 1280 tokens (from training data)

Expected Results:
- Training Effective Rank: 88.7% (681/768)
- Validation Effective Rank: 81.7% (627/768)

CRITICAL CHECKS (絶対必要):
1. Effective Rank: 多様性確認
2. Identity Check: 恒等写像になっていないか確認
3. CVFP Convergence: 固定点学習ができているか確認
"""

from config import ResidualConfig
import torch
import numpy as np
import random
from src.models.new_llm_residual import NewLLMResidual
from src.data.loader import load_data
from src.training.phase1_trainer import Phase1Trainer
from src.evaluation.metrics import analyze_fixed_points, check_identity_mapping, check_cvfp_convergence

# ============================================================
# SEED FIXING - 完全な再現性保証
# ============================================================
def set_seed(seed=42):
    """全ての乱数生成器のシードを固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 決定的動作を保証
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("\n✅ Random seed fixed: 42 (完全な再現性保証)")

# Configuration
config = ResidualConfig()
config.num_samples = 50  # 6400 tokens
config.max_seq_length = 128

print("\n" + "="*70)
print("Phase 1 Test: 81.7% Effective Rank (Validation)")
print("="*70 + "\n")

device = torch.device(config.device if torch.cuda.is_available() else "cpu")

# Load data
print("Loading data...")
train_token_ids, val_token_ids = load_data(config)

print(f"Total training tokens: {len(train_token_ids)}")
print(f"Total validation tokens: {len(val_token_ids)}")

# Ensure we have 5000+ training tokens
if len(train_token_ids) < 5000:
    print(f"\n⚠️ Warning: Only {len(train_token_ids)} training tokens available")
    print("89.3% Effective Rank requires 5000+ tokens")
else:
    print(f"\n✅ Sufficient training tokens for 89.3% target")

# Create model
print("\nCreating model...")
layer_structure = [1] * config.num_layers
model = NewLLMResidual(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    context_dim=config.context_dim,
    hidden_dim=config.hidden_dim,
    layer_structure=layer_structure,
    layernorm_mix=1.0,
    use_pretrained_embeddings=True
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Create trainer
print("\nStarting Phase 1 training...")
trainer = Phase1Trainer(
    model=model,
    max_iterations=config.phase1_max_iterations,
    convergence_threshold=config.phase1_convergence_threshold,
    min_converged_ratio=config.phase1_min_converged_ratio,
    learning_rate=config.phase1_learning_rate,
    dist_reg_weight=config.dist_reg_weight
)

# Train with 5000 tokens
print(f"\nTraining with {len(train_token_ids)} tokens...")
train_contexts = trainer.train(train_token_ids, device, label="Train")

# Evaluate on validation data
print(f"\nEvaluating on validation data ({len(val_token_ids)} tokens)...")
val_contexts = trainer.evaluate(val_token_ids, device)

# ============================================================
# CRITICAL CHECKS (絶対必要な3つのチェック)
# ============================================================

# Check 1: Effective Rank - 多様性確認
print("\n" + "="*70)
print("CHECK 1: EFFECTIVE RANK (多様性確認)")
print("="*70)
print("\nTRAINING DATA:")
print("-"*70 + "\n")

train_metrics = analyze_fixed_points(train_contexts, label="Train")

print("\n" + "="*70)
print("VALIDATION DATA (THIS IS WHAT MATTERS):")
print("-"*70 + "\n")

val_metrics = analyze_fixed_points(val_contexts, label="Validation")

# Check 2: Identity Mapping Check - 恒等写像になっていないか確認
print("\n" + "="*70)
print("CHECK 2: IDENTITY MAPPING (恒等写像チェック)")
print("="*70)

# トークン埋め込みを取得
with torch.no_grad():
    train_token_embeds = model.token_embedding(train_token_ids.unsqueeze(0).to(device))
    train_token_embeds = model.embed_norm(train_token_embeds).squeeze(0)

identity_check = check_identity_mapping(model, train_token_embeds, train_contexts, device)

# Check 3: CVFP Convergence Check - 固定点学習確認
print("\n" + "="*70)
print("CHECK 3: CVFP CONVERGENCE (固定点学習確認)")
print("="*70)

# サンプルデータで収束チェック（全データは時間がかかるため）
sample_size = min(100, len(train_token_ids))
sample_token_ids = train_token_ids[:sample_size]

convergence_check = check_cvfp_convergence(trainer, sample_token_ids, device, max_test_iterations=5)

# ============================================================
# FINAL SUMMARY - 3つのチェック結果まとめ
# ============================================================
print(f"\n" + "="*70)
print("FINAL SUMMARY - 81.7% Implementation Verification")
print("="*70)

print(f"\nTraining tokens: {len(train_token_ids)}")
print(f"Train Effective Rank: {train_metrics['effective_rank']:.2f}/{config.context_dim} ({train_metrics['effective_rank']/config.context_dim*100:.1f}%)")

print(f"\nValidation tokens: {len(val_token_ids)}")
print(f"Val Effective Rank: {val_metrics['effective_rank']:.2f}/{config.context_dim} ({val_metrics['effective_rank']/config.context_dim*100:.1f}%)")

# Check results
all_passed = True

print("\n" + "-"*70)
print("CRITICAL CHECKS:")
print("-"*70)

# 1. Effective Rank
print("\n1. Effective Rank (多様性):")
if val_metrics['effective_rank']/config.context_dim >= 0.80:
    print(f"   ✅ PASSED: {val_metrics['effective_rank']/config.context_dim*100:.1f}% (Expected: ~81.7%)")
else:
    print(f"   ❌ FAILED: {val_metrics['effective_rank']/config.context_dim*100:.1f}% (Expected: ~81.7%)")
    all_passed = False

# 2. Identity Mapping
print("\n2. Identity Mapping (恒等写像):")
if not identity_check['is_identity']:
    print(f"   ✅ PASSED: Not identity mapping (diff={identity_check['context_diff_from_zero']:.4f})")
else:
    print(f"   ❌ FAILED: Identity mapping detected (diff={identity_check['context_diff_from_zero']:.4f})")
    all_passed = False

# 3. CVFP Convergence
print("\n3. CVFP Convergence (固定点学習):")
if convergence_check['quality'] in ['excellent', 'good', 'moderate']:
    print(f"   ✅ PASSED: {convergence_check['quality'].upper()} (final_diff={convergence_check['final_diff']:.6f})")
else:
    print(f"   ❌ FAILED: {convergence_check['quality'].upper()} (final_diff={convergence_check['final_diff']:.6f})")
    all_passed = False

print("\n" + "="*70)
if all_passed:
    print("✅ ALL CHECKS PASSED - Implementation is valid!")
else:
    print("❌ SOME CHECKS FAILED - Implementation may have issues!")
print("="*70 + "\n")