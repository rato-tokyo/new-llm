"""
Standard Phase 1 Test - 複数トークン入力対応版

Uses fixed datasets:
- Training: UltraChat from config.num_samples
- Validation: From training data

CRITICAL CHECKS (3つの必須チェック):
1. Effective Rank: 多様性確認
2. Identity Mapping Check: 恒等写像になっていないか確認
3. Processing Speed: 高速化の確認
"""

from config import ResidualConfig
import torch
import time
from src.models.llm import LLM
from src.providers.data.memory import MemoryDataProvider
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.evaluation.metrics import analyze_fixed_points, check_identity_mapping
from src.evaluation.convergence import forward_sequential
from src.utils.seed import set_seed


# ============================================================
# SEED FIXING - 完全な再現性保証
# ============================================================
set_seed(42)
print("\n✅ Random seed fixed: 42 (完全な再現性保証)")

# Configuration
config = ResidualConfig()
config.num_samples = 10  # 小サンプルで動作確認

print("\n" + "="*70)
print("Phase 1 Test: 複数トークン入力対応版")
print("="*70 + "\n")

print("Config:")
print(f"  - num_samples: {config.num_samples}")
print(f"  - num_layers: {config.num_layers}")
print(f"  - context_dim: {config.context_dim}")
print(f"  - embed_dim: {config.embed_dim}")
print(f"  - num_input_tokens: {config.num_input_tokens}")
print(f"  - dist_reg_weight: {config.dist_reg_weight}")

device = torch.device(config.device if torch.cuda.is_available() else "cpu")
print(f"  - device: {device}")

# Load data
print("\nLoading data...")
data_provider = MemoryDataProvider(config)
data_provider.load_data()

train_token_ids = data_provider.get_all_train_tokens(device)
val_token_ids = data_provider.get_all_val_tokens(device)

print(f"Total training tokens: {len(train_token_ids)}")
print(f"Total validation tokens: {len(val_token_ids)}")

# Create model
print("\nCreating model...")
model = LLM(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    context_dim=config.context_dim,
    num_layers=config.num_layers,
    num_input_tokens=config.num_input_tokens,
    use_pretrained_embeddings=config.use_pretrained_embeddings,
    use_weight_tying=config.use_weight_tying,
    config=config
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Create trainer
trainer = MemoryPhase1Trainer(model, config, device)

# Train with parallel processing
print("\nStarting Phase 1 training...")
print(f"設定: dist_reg_weight={config.dist_reg_weight}, max_iterations={config.phase1_max_iterations}")

train_start = time.time()
train_result = trainer.train(train_token_ids, label="Train", data_provider=data_provider)
assert isinstance(train_result, torch.Tensor), "Expected Tensor from train()"
train_contexts: torch.Tensor = train_result
train_time = time.time() - train_start

# Evaluate on validation data
print(f"\nEvaluating on validation data ({len(val_token_ids)} tokens)...")

val_start = time.time()
model.eval()
with torch.no_grad():
    val_token_embeds = model.token_embedding(val_token_ids.unsqueeze(0).to(device))
    val_token_embeds = model.embed_norm(val_token_embeds).squeeze(0)

num_input_tokens = config.num_input_tokens
val_contexts = forward_sequential(model, val_token_embeds, None, device, num_input_tokens)
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
print("CHECK 3: PROCESSING SPEED (処理時間)")
print("="*70)

print(f"\n訓練時間: {train_time:.2f}秒")
print(f"検証時間: {val_time:.2f}秒")
print(f"合計時間: {train_time + val_time:.2f}秒")

# ============================================================
# FINAL SUMMARY - 3つのチェック結果まとめ
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
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

# 1. Effective Rank
print("\n1. Effective Rank (多様性):")
if val_metrics['effective_rank']/config.context_dim >= 0.30:
    print(f"   ✅ PASSED: {val_metrics['effective_rank']/config.context_dim*100:.1f}%")
else:
    print(f"   ❌ FAILED: {val_metrics['effective_rank']/config.context_dim*100:.1f}%")
    all_passed = False

# 2. Identity Mapping
print("\n2. Identity Mapping (恒等写像):")
if not identity_check['is_identity']:
    print(f"   ✅ PASSED: Not identity mapping (diff={identity_check['context_diff_from_zero']:.4f})")
else:
    print(f"   ❌ FAILED: Identity mapping detected (diff={identity_check['context_diff_from_zero']:.4f})")
    all_passed = False

# 3. Processing Speed
print("\n3. Processing Speed:")
print(f"   ✓ 訓練: {train_time:.2f}秒")
print(f"   ✓ 検証: {val_time:.2f}秒")

print("\n" + "="*70)
if all_passed:
    print("✅ ALL CHECKS PASSED - 複数トークン入力対応版は正常動作しています！")
    print(f"   Effective Rank: {val_metrics['effective_rank']/config.context_dim*100:.1f}%")
    print(f"   処理時間: {train_time + val_time:.2f}秒")
else:
    print("❌ SOME CHECKS FAILED - 実装に問題がある可能性があります！")
print("="*70 + "\n")
