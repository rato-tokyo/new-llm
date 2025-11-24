"""
Model Save/Load Utility for New-LLM

学習済みモデルの保存とロード機能を提供
"""

import torch
import os
from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual


def save_model(model, filepath="checkpoints/model_checkpoint.pth"):
    """
    モデルの重みを保存

    Args:
        model: 保存するモデル
        filepath: 保存先ファイルパス
    """
    # ディレクトリが存在しない場合作成
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # state_dictを保存
    # blocksの各ブロック内のレイヤー数をカウント
    num_blocks = len(model.blocks)
    if num_blocks > 0:
        num_layers = len(model.blocks[0].layers)
    else:
        num_layers = 6  # デフォルト値

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'embed_dim': model.embed_dim,
            'context_dim': model.context_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': num_layers,
            'num_blocks': num_blocks,
            'layernorm_mix': model.layernorm_mix
        }
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath="checkpoints/model_checkpoint.pth", device="cpu"):
    """
    保存されたモデルの重みをロード

    Args:
        filepath: 読み込むファイルパス
        device: デバイス (cpu/cuda)

    Returns:
        model: ロードされたモデル
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

    # チェックポイントをロード
    checkpoint = torch.load(filepath, map_location=device)
    config_dict = checkpoint['config']

    # モデルを作成
    # layer_structureを再構成（ブロックごとのレイヤー数）
    num_blocks = config_dict.get('num_blocks', 1)
    num_layers_per_block = config_dict['num_layers']
    layer_structure = [num_layers_per_block] * num_blocks

    model = NewLLMResidual(
        vocab_size=config_dict['vocab_size'],
        embed_dim=config_dict['embed_dim'],
        context_dim=config_dict['context_dim'],
        hidden_dim=config_dict['hidden_dim'],
        layer_structure=layer_structure,
        layernorm_mix=config_dict['layernorm_mix'],
        use_pretrained_embeddings=True
    )

    # 重みをロード
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # 簡単なテスト
    config = ResidualConfig()

    # モデル作成
    layer_structure = [1] * config.num_layers
    test_model = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        layernorm_mix=1.0,
        use_pretrained_embeddings=True
    )

    # 保存テスト
    save_model(test_model)

    # ロードテスト
    loaded_model = load_model()

    # パラメータ数確認
    original_params = sum(p.numel() for p in test_model.parameters())
    loaded_params = sum(p.numel() for p in loaded_model.parameters())

    print(f"Original model params: {original_params:,}")
    print(f"Loaded model params: {loaded_params:,}")
    assert original_params == loaded_params, "Parameter count mismatch!"

    print("Save/Load test passed!")