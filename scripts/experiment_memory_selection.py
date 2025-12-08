#!/usr/bin/env python3
"""
Memory Selection Experiment - メモリ選択精度の評価

異なるドメインのテキストを各メモリに格納し、
クエリに対して正しいメモリを選択できるかを評価する。

使用例:
    python3 scripts/experiment_memory_selection.py
    python3 scripts/experiment_memory_selection.py --num-memories 8
"""

import argparse
import sys
from pathlib import Path

import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.pythia import PythiaConfig
from src.models import create_model
from src.models.layers import MultiMemoryLayer
from src.utils import (
    set_seed,
    get_device,
    get_tokenizer,
    print_flush,
    MemoryBuilder,
)


def create_domain_data() -> dict[str, dict]:
    """ドメイン別のテキストとクエリを作成

    Returns:
        domain_name -> {
            "context": メモリに格納するテキスト,
            "queries": クエリのリスト,
        }
    """
    return {
        "science": {
            "context": """
Physics is the natural science that studies matter, energy, and the fundamental forces of nature.
Quantum mechanics describes the behavior of particles at atomic and subatomic scales.
Einstein's theory of relativity revolutionized our understanding of space, time, and gravity.
The Standard Model explains the electromagnetic, weak, and strong nuclear forces.
Thermodynamics studies heat, work, and energy transfer between systems.
Chemistry is the scientific study of the composition, structure, and properties of matter.
Biology examines living organisms and their vital processes.
""",
            "queries": [
                "What is quantum mechanics?",
                "Who developed the theory of relativity?",
                "What does the Standard Model explain?",
                "What is thermodynamics about?",
                "What is physics?",
            ],
        },
        "history": {
            "context": """
World War II was a global conflict that lasted from 1939 to 1945.
The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century.
The Industrial Revolution began in Britain in the late 18th century.
Ancient Rome was a civilization that grew from a small town to control much of Europe.
The French Revolution of 1789 dramatically changed France and inspired revolutions worldwide.
The Cold War was a period of geopolitical tension between the United States and Soviet Union.
The printing press was invented by Johannes Gutenberg around 1440.
""",
            "queries": [
                "When did World War II end?",
                "What was the Renaissance?",
                "Where did the Industrial Revolution begin?",
                "What was the French Revolution?",
                "Who invented the printing press?",
            ],
        },
        "technology": {
            "context": """
Machine learning is a subset of artificial intelligence that enables computers to learn from data.
The internet is a global network connecting millions of computers worldwide.
Python is a popular programming language known for its simplicity and versatility.
Cloud computing delivers computing services over the internet.
Blockchain is a decentralized, distributed ledger technology.
Neural networks are computing systems inspired by biological neural networks.
Cybersecurity protects computer systems and networks from digital attacks.
""",
            "queries": [
                "What is machine learning?",
                "What programming language is known for simplicity?",
                "What is cloud computing?",
                "What is blockchain?",
                "What are neural networks?",
            ],
        },
        "geography": {
            "context": """
Mount Everest is the highest mountain on Earth, located in the Himalayas.
The Amazon River is the largest river by volume, flowing through South America.
The Sahara Desert is the largest hot desert in the world, covering much of North Africa.
Australia is both a continent and a country, known for its unique wildlife.
The Pacific Ocean is the largest and deepest ocean on Earth.
Antarctica is the coldest continent, located at the South Pole.
The Great Barrier Reef is the world's largest coral reef system.
""",
            "queries": [
                "What is the highest mountain on Earth?",
                "Where is the Sahara Desert?",
                "What is the largest ocean?",
                "What is special about Antarctica?",
                "Where is the Great Barrier Reef?",
            ],
        },
    }


def evaluate_memory_selection(
    model,
    memory_layer: MultiMemoryLayer,
    tokenizer,
    domain_data: dict[str, dict],
    domain_to_memory: dict[str, int],
    device: torch.device,
) -> dict:
    """メモリ選択の精度を評価

    Args:
        model: モデル
        memory_layer: MultiMemoryLayer
        tokenizer: トークナイザー
        domain_data: ドメイン別データ
        domain_to_memory: ドメイン名 → メモリインデックス
        device: デバイス

    Returns:
        評価結果
    """
    model.eval()
    attn = memory_layer.attention

    results: dict = {
        "total": 0,
        "correct": 0,
        "per_domain": {},
    }

    with torch.no_grad():
        for domain_name, data in domain_data.items():
            expected_memory = domain_to_memory[domain_name]
            domain_results = {"total": 0, "correct": 0}

            for query in data["queries"]:
                # クエリをトークン化
                tokens = tokenizer.encode(query)
                if isinstance(tokens, list):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                input_ids = tokens.unsqueeze(0).to(device)

                # Embeddingを通して隠れ状態を取得
                hidden_states = model.embed_in(input_ids)
                hidden_states = memory_layer.input_layernorm(hidden_states)

                # Q/K/V計算
                batch_size, seq_len, _ = hidden_states.shape
                q = attn.w_q(hidden_states).view(
                    batch_size, seq_len, attn.num_heads, attn.head_dim
                ).transpose(1, 2)

                # 各メモリとの関連度を計算
                relevances = []
                assert attn.landmarks is not None and attn.key_counts is not None
                for idx, landmark in enumerate(attn.landmarks):
                    if attn.key_counts[idx].sum() < 1e-6:
                        relevances.append(float('-inf'))
                    else:
                        rel = attn._compute_relevance(q, landmark)
                        relevances.append(rel.mean().item())

                # 最も関連度が高いメモリを選択
                selected_memory = max(range(len(relevances)), key=lambda x: relevances[x])

                # 正解判定
                correct = selected_memory == expected_memory
                domain_results["total"] += 1
                domain_results["correct"] += int(correct)
                results["total"] += 1
                results["correct"] += int(correct)

            results["per_domain"][domain_name] = domain_results

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Memory Selection Experiment")
    parser.add_argument("--num-memories", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = get_tokenizer()

    print_flush("=" * 60)
    print_flush("Memory Selection Experiment")
    print_flush("=" * 60)
    print_flush(f"Device: {device}")
    print_flush(f"Memories: {args.num_memories}")
    print_flush(f"Seq length: {args.seq_length}")
    print_flush()

    # ドメインデータ作成
    domain_data = create_domain_data()
    domain_names = list(domain_data.keys())

    if len(domain_names) > args.num_memories:
        print_flush(f"Warning: {len(domain_names)} domains but only {args.num_memories} memories")
        domain_names = domain_names[:args.num_memories]

    # モデル作成
    config = PythiaConfig()
    model = create_model("multi_memory", config, num_memories=args.num_memories)
    model = model.to(device)
    model.eval()

    # MultiMemoryLayerを取得
    memory_layer = None
    for layer in model.layers:
        if isinstance(layer, MultiMemoryLayer):
            memory_layer = layer
            break

    if memory_layer is None:
        raise RuntimeError("MultiMemoryLayer not found")

    # MemoryBuilderでメモリを構築
    print_flush("[Building memories]")
    builder = MemoryBuilder(model, tokenizer, args.seq_length, device)
    builder.reset_all_memories()

    domain_to_memory = {}
    for i, domain_name in enumerate(domain_names):
        context = domain_data[domain_name]["context"]
        stats = builder.build_memory(i, context)
        domain_to_memory[domain_name] = i
        print_flush(f"  Memory {i} ({domain_name}): {stats['num_tokens']} tokens")

    print_flush()
    builder.print_memory_info()
    print_flush()

    # 評価
    print_flush("[Evaluating memory selection]")
    results = evaluate_memory_selection(
        model, memory_layer, tokenizer,
        {k: v for k, v in domain_data.items() if k in domain_to_memory},
        domain_to_memory, device
    )

    print_flush()
    print_flush("=" * 60)
    print_flush("Results")
    print_flush("=" * 60)
    print_flush(f"Overall Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    print_flush()
    print_flush("Per-domain:")
    for domain_name, domain_results in results["per_domain"].items():
        acc = domain_results["correct"] / domain_results["total"] if domain_results["total"] > 0 else 0
        print_flush(f"  {domain_name}: {acc:.1%} ({domain_results['correct']}/{domain_results['total']})")

    print_flush()
    print_flush("=" * 60)
    print_flush("Analysis")
    print_flush("=" * 60)

    if results["accuracy"] < 0.5:
        print_flush("Low accuracy suggests:")
        print_flush("  - Landmarks are not sufficiently distinct")
        print_flush("  - Need more diverse domain texts")
        print_flush("  - Model may need fine-tuning for memory selection")
    elif results["accuracy"] < 0.8:
        print_flush("Moderate accuracy:")
        print_flush("  - HSA-style landmarks are working partially")
        print_flush("  - Consider increasing text diversity")
    else:
        print_flush("High accuracy:")
        print_flush("  - HSA-style landmarks effectively distinguish domains")
        print_flush("  - Memory selection is working well")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
