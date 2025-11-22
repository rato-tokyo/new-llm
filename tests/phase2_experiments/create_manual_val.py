"""Create manual validation data using only tokens from training data"""
import torch

# Most common tokens from training data (from analyze_train_tokens.py)
# Creating simple sequences using these tokens
val_sequences = [
    # Sequence 1: "the , and ."
    [262, 11, 290, 13],

    # Sequence 2: "a to the of"
    [257, 284, 262, 286],

    # Sequence 3: "the for is in"
    [262, 329, 318, 287],

    # Sequence 4: ". , - :"
    [13, 11, 12, 25],

    # Sequence 5: "or with as on"
    [393, 351, 355, 319],

    # Sequence 6: "the and to in for"
    [262, 290, 284, 287, 329],

    # Sequence 7: "a the of is"
    [257, 262, 286, 318],

    # Sequence 8: ". the , and"
    [13, 262, 11, 290],
]

# Flatten all sequences
all_val_tokens = []
for seq in val_sequences:
    all_val_tokens.extend(seq)

print(f"Created {len(val_sequences)} validation sequences")
print(f"Total validation tokens: {len(all_val_tokens)}")
print(f"Validation token IDs: {all_val_tokens}")

# Save as tensor
val_tensor = torch.tensor(all_val_tokens, dtype=torch.long)
torch.save(val_tensor, '/Users/sakajiritomoyoshi/Desktop/git/new-llm/cache/manual_val_tokens.pt')
print(f"\nSaved to: cache/manual_val_tokens.pt")
