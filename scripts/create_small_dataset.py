"""
小規模データセット作成スクリプト（1000トークン）
並列化実験用の高速検証データ
"""

import torch
from transformers import AutoTokenizer

def create_small_dataset(num_tokens=1000, output_path="./data/small_train.txt"):
    """
    UltraChatから1000トークンを抽出

    Args:
        num_tokens: 目標トークン数
        output_path: 出力ファイルパス
    """
    # Tokenizerロード
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        cache_dir="./cache/tokenizer"
    )

    # UltraChatデータロード
    from datasets import load_dataset
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        cache_dir="./cache/datasets"
    )

    # トークン収集
    collected_tokens = []
    for sample in dataset:
        messages = sample["messages"]
        for message in messages:
            content = message["content"]
            tokens = tokenizer.encode(content, add_special_tokens=False)
            collected_tokens.extend(tokens)

            if len(collected_tokens) >= num_tokens:
                break

        if len(collected_tokens) >= num_tokens:
            break

    # ちょうど1000トークンに切り詰め
    collected_tokens = collected_tokens[:num_tokens]

    # テキストにデコード
    text = tokenizer.decode(collected_tokens)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Created small dataset: {len(collected_tokens)} tokens")
    print(f"   Saved to: {output_path}")

    return output_path

if __name__ == "__main__":
    create_small_dataset(num_tokens=1000, output_path="./data/small_train.txt")
