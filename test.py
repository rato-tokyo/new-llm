"""
Standard Phase 1 Test - 並列処理版（55.9% Effective Rank達成）

並列化により23x高速化（265秒 → 11秒）
dist_reg_weight=0.9により多様性を維持

Uses fixed datasets:
- Training: 6400 tokens (UltraChat 50 samples)
- Validation: 1280 tokens (from training data)

Expected Results (並列版):
- Training Effective Rank: ~60% (460/768)
- Validation Effective Rank: 55.9% (429/768)
- Convergence Rate: ~27% (多様性優先のためCVFP収束率は低め)
- Processing Time: ~11秒（シーケンシャル版の23x高速化）

CRITICAL CHECKS (3つの必須チェック):
1. Effective Rank: 多様性確認（55.9%目標）
2. Identity Mapping Check: 恒等写像になっていないか確認
3. Processing Speed: 高速化の確認（<15秒）
"""

from config import ResidualConfig
import torch
import numpy as np
import random
from src.models.llm import LLM
from src.data.loader import load_data
from src.trainers.phase1 import phase1_train
from src.evaluation.metrics import analyze_fixed_points, check_identity_mapping
import time

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
print("Phase 1 Test: 並列処理版（55.9% ER, 23x高速化）")
print("="*70 + "\n")

device = torch.device(config.device if torch.cuda.is_available() else "cpu")

# Load data
print("Loading data...")
train_token_ids, val_token_ids = load_data(config)

print(f"Total training tokens: {len(train_token_ids)}")
print(f"Total validation tokens: {len(val_token_ids)}")

# Create model
print("\nCreating model...")
model = LLM(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    context_dim=config.context_dim,
    context_layers=config.context_layers,
    token_layers=config.token_layers,
    layernorm_mix=config.layernorm_mix,
    use_pretrained_embeddings=config.use_pretrained_embeddings
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Train with parallel processing
print("\nStarting Phase 1 training (並列処理版)...")
print(f"設定: dist_reg_weight={config.dist_reg_weight}, max_iterations={config.phase1_max_iterations}")

train_start = time.time()
train_contexts = phase1_train(
    model=model,
    token_ids=train_token_ids,
    device=device,
    max_iterations=config.phase1_max_iterations,
    convergence_threshold=config.phase1_convergence_threshold,
    min_converged_ratio=config.phase1_min_converged_ratio,
    learning_rate=config.phase1_learning_rate,
    dist_reg_weight=config.dist_reg_weight,
    label="Train"
)
train_time = time.time() - train_start

# Evaluate on validation data (並列処理)
print(f"\nEvaluating on validation data ({len(val_token_ids)} tokens)...")

val_start = time.time()
# 検証時は訓練なし（順伝播のみ）
model.eval()
with torch.no_grad():
    val_token_embeds = model.token_embedding(val_token_ids.unsqueeze(0).to(device))
    val_token_embeds = model.embed_norm(val_token_embeds).squeeze(0)

from src.trainers.phase1 import forward_all_tokens_sequential
val_contexts = forward_all_tokens_sequential(model, val_token_embeds, None, device)
val_time = time.time() - val_start

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

# Check 3: Processing Speed - 高速化確認
print("\n" + "="*70)
print("CHECK 3: PROCESSING SPEED (高速化確認)")
print("="*70)

print(f"\n訓練時間: {train_time:.2f}秒")
print(f"検証時間: {val_time:.2f}秒")
print(f"合計時間: {train_time + val_time:.2f}秒")
print(f"\nシーケンシャル版: ~265秒")
print(f"並列版（現在）: {train_time:.2f}秒")
print(f"高速化率: {265/train_time:.1f}x")

# ============================================================
# FINAL SUMMARY - 3つのチェック結果まとめ
# ============================================================
print(f"\n" + "="*70)
print("FINAL SUMMARY - 並列処理版性能検証")
print("="*70)

print(f"\nTraining tokens: {len(train_token_ids)}")
print(f"Train Effective Rank: {train_metrics['effective_rank']:.2f}/{config.context_dim} ({train_metrics['effective_rank']/config.context_dim*100:.1f}%)")

print(f"\nValidation tokens: {len(val_token_ids)}")
print(f"Val Effective Rank: {val_metrics['effective_rank']:.2f}/{config.context_dim} ({val_metrics['effective_rank']/config.context_dim*100:.1f}%)")

print(f"\n処理時間: {train_time:.2f}秒（訓練） + {val_time:.2f}秒（検証）= {train_time + val_time:.2f}秒")

# Check results
all_passed = True

print("\n" + "-"*70)
print("CRITICAL CHECKS:")
print("-"*70)

# 1. Effective Rank（並列版目標: 55.9%）
print("\n1. Effective Rank (多様性):")
if val_metrics['effective_rank']/config.context_dim >= 0.50:
    print(f"   ✅ PASSED: {val_metrics['effective_rank']/config.context_dim*100:.1f}% (Target: ~55.9%)")
else:
    print(f"   ❌ FAILED: {val_metrics['effective_rank']/config.context_dim*100:.1f}% (Target: ~55.9%)")
    all_passed = False

# 2. Identity Mapping
print("\n2. Identity Mapping (恒等写像):")
if not identity_check['is_identity']:
    print(f"   ✅ PASSED: Not identity mapping (diff={identity_check['context_diff_from_zero']:.4f})")
else:
    print(f"   ❌ FAILED: Identity mapping detected (diff={identity_check['context_diff_from_zero']:.4f})")
    all_passed = False

# 3. Processing Speed（並列版目標: <15秒）
print("\n3. Processing Speed (高速化):")
if train_time < 15:
    print(f"   ✅ PASSED: {train_time:.2f}秒 < 15秒（{265/train_time:.1f}x高速化）")
else:
    print(f"   ⚠️ SLOW: {train_time:.2f}秒 > 15秒（目標: <15秒）")

print("\n" + "="*70)
if all_passed:
    print("✅ ALL CHECKS PASSED - 並列処理版は正常動作しています！")
    print(f"   Effective Rank: {val_metrics['effective_rank']/config.context_dim*100:.1f}%")
    print(f"   処理時間: {train_time:.2f}秒（{265/train_time:.1f}x高速化）")
else:
    print("❌ SOME CHECKS FAILED - 実装に問題がある可能性があります！")
print("="*70 + "\n")
