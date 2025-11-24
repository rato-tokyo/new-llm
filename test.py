"""
Standard Phase 1 Test - 89.4% Effective Rank (Validation)

Uses fixed datasets:
- Training: 6400 tokens (UltraChat 50 samples)
- Validation: 1280 tokens (from training data)

Expected Results:
- Training Effective Rank: 89.7% (689/768)
- Validation Effective Rank: 89.4% (687/768)
- Convergence Rate: >50% (fixed-point learning)

CRITICAL CHECKS (4つの必須チェック):
1. Effective Rank: 多様性確認（89.4%目標）
2. Identity Mapping Check: 恒等写像になっていないか確認
3. Gradient Flow Check: トークン間勾配伝播確認
4. Convergence Rate: 収束率確認（>50%目標）
"""

from config import ResidualConfig
import torch
import numpy as np
import random
from src.models.new_llm_residual import NewLLMResidual
from src.data.loader import load_data
from src.training.phase1_trainer import Phase1Trainer
from src.evaluation.metrics import analyze_fixed_points, check_identity_mapping
from src.evaluation.diagnostics import check_gradient_flow, print_gradient_flow_result

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

# Check 3: Gradient Flow Check - トークン間勾配伝播確認
print("\n" + "="*70)
print("CHECK 3: GRADIENT FLOW (勾配伝播チェック)")
print("="*70)

# 勾配フローチェック（100トークンでテスト）
gradient_flow_check = check_gradient_flow(trainer, train_token_ids, device, num_tokens_to_check=100)
print_gradient_flow_result(gradient_flow_check)

# ============================================================
# CONVERGENCE CHECK - 収束状況の確認
# ============================================================
print("\n" + "="*70)
print("CONVERGENCE CHECK (収束状況)")
print("="*70)

# 訓練の収束結果を表示
print(f"\n訓練収束率: {trainer.num_converged_tokens}/{len(train_token_ids)} = {trainer.num_converged_tokens/len(train_token_ids)*100:.1f}%")
print(f"収束閾値: {config.phase1_convergence_threshold}")
print(f"必要収束率: {config.phase1_min_converged_ratio*100:.0f}%")

if trainer.num_converged_tokens == 0:
    print("\n⚠️  警告: 収束率0% - 固定点学習が機能していません")
    print("    原因: コンテキスト変化量が閾値より遥かに大きい")
    print("    対策: 学習率調整またはアーキテクチャの見直しが必要")
elif trainer.num_converged_tokens/len(train_token_ids) < config.phase1_min_converged_ratio:
    print(f"\n⚠️  警告: 収束率が低い ({trainer.num_converged_tokens/len(train_token_ids)*100:.1f}% < {config.phase1_min_converged_ratio*100:.0f}%)")
else:
    print(f"\n✅ 収束率良好: {trainer.num_converged_tokens/len(train_token_ids)*100:.1f}%")

# ============================================================
# FINAL SUMMARY - 3つのチェック結果まとめ
# ============================================================
print(f"\n" + "="*70)
print("FINAL SUMMARY - 89.4% Implementation Verification")
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
    print(f"   ✅ PASSED: {val_metrics['effective_rank']/config.context_dim*100:.1f}% (Target: ~89.4%)")
else:
    print(f"   ❌ FAILED: {val_metrics['effective_rank']/config.context_dim*100:.1f}% (Target: ~89.4%)")
    all_passed = False

# 2. Identity Mapping
print("\n2. Identity Mapping (恒等写像):")
if not identity_check['is_identity']:
    print(f"   ✅ PASSED: Not identity mapping (diff={identity_check['context_diff_from_zero']:.4f})")
else:
    print(f"   ❌ FAILED: Identity mapping detected (diff={identity_check['context_diff_from_zero']:.4f})")
    all_passed = False

# 3. Gradient Flow
print("\n3. Gradient Flow (勾配伝播):")
if gradient_flow_check['has_gradient_flow']:
    print(f"   ✅ PASSED: Gradient flows between tokens (norm_ratio={gradient_flow_check['norm_ratio']:.2f})")
else:
    print(f"   ❌ FAILED: Gradient flow blocked (norm_ratio={gradient_flow_check['norm_ratio']:.2f})")
    all_passed = False

# 4. Convergence Rate (収束率)
print("\n4. Convergence Rate (収束率):")
convergence_rate = trainer.num_converged_tokens/len(train_token_ids)
if convergence_rate > 0.5:  # 50%以上が収束すれば良好とする
    print(f"   ✅ PASSED: {convergence_rate*100:.1f}% converged")
else:
    print(f"   ❌ FAILED: {convergence_rate*100:.1f}% converged (expected >50%)")
    all_passed = False

print("\n" + "="*70)
if all_passed:
    print("✅ ALL CHECKS PASSED - Implementation is valid!")
else:
    print("❌ SOME CHECKS FAILED - Implementation may have issues!")
print("="*70 + "\n")