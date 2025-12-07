"""
Reversal Curse Evaluation Data

順方向と逆方向のペアを定義し、Reversal Curseを測定する。

例:
  順方向: "The capital of France is Paris"
  逆方向: "Paris is the capital of France"

モデルが順方向で学習した知識を逆方向でも活用できるかを評価する。
"""

from typing import List, Dict


# 事実ペア: (主語, 関係, 目的語)
# 順方向: "{subject} {relation} {object}"
# 逆方向: "{object} {reverse_relation} {subject}"
FACT_TRIPLES = [
    # 首都
    ("The capital of France", "is", "Paris", "is the capital of"),
    ("The capital of Japan", "is", "Tokyo", "is the capital of"),
    ("The capital of Germany", "is", "Berlin", "is the capital of"),
    ("The capital of Italy", "is", "Rome", "is the capital of"),
    ("The capital of Spain", "is", "Madrid", "is the capital of"),
    ("The capital of China", "is", "Beijing", "is the capital of"),
    ("The capital of Russia", "is", "Moscow", "is the capital of"),
    ("The capital of Brazil", "is", "Brasilia", "is the capital of"),
    ("The capital of Australia", "is", "Canberra", "is the capital of"),
    ("The capital of Canada", "is", "Ottawa", "is the capital of"),

    # 発明者
    ("The inventor of the telephone", "was", "Alexander Graham Bell", "invented the telephone"),
    ("The inventor of the light bulb", "was", "Thomas Edison", "invented the light bulb"),
    ("The inventor of the airplane", "was", "the Wright brothers", "invented the airplane"),
    ("The inventor of the printing press", "was", "Johannes Gutenberg", "invented the printing press"),
    ("The inventor of the steam engine", "was", "James Watt", "improved the steam engine"),

    # 著者
    ("The author of Hamlet", "was", "William Shakespeare", "wrote Hamlet"),
    ("The author of Pride and Prejudice", "was", "Jane Austen", "wrote Pride and Prejudice"),
    ("The author of 1984", "was", "George Orwell", "wrote 1984"),
    ("The author of The Great Gatsby", "was", "F. Scott Fitzgerald", "wrote The Great Gatsby"),
    ("The author of War and Peace", "was", "Leo Tolstoy", "wrote War and Peace"),

    # 科学的発見
    ("The discoverer of penicillin", "was", "Alexander Fleming", "discovered penicillin"),
    ("The discoverer of gravity", "was", "Isaac Newton", "discovered gravity"),
    ("The discoverer of radioactivity", "was", "Marie Curie", "discovered radioactivity"),
    ("The discoverer of DNA structure", "was", "Watson and Crick", "discovered DNA structure"),

    # 創設者
    ("The founder of Microsoft", "was", "Bill Gates", "founded Microsoft"),
    ("The founder of Apple", "was", "Steve Jobs", "founded Apple"),
    ("The founder of Amazon", "was", "Jeff Bezos", "founded Amazon"),
    ("The founder of Tesla", "is", "Elon Musk", "founded Tesla"),

    # 合成データ（架空の事実）- より純粋なテスト用
    ("Zephyr Moonstone", "wrote", "The Crimson Dawn", "was written by"),
    ("Aria Windfall", "invented", "the quantum compass", "was invented by"),
    ("The city of Luminara", "is ruled by", "King Aldric", "rules the city of Luminara"),
    ("Professor Thorne", "discovered", "element Novarite", "was discovered by"),
    ("The Silverbrook Academy", "was founded by", "Lady Elara", "founded the Silverbrook Academy"),
]


def get_reversal_pairs() -> List[Dict[str, str]]:
    """
    順方向・逆方向のペアを取得

    Returns:
        List of dicts with 'forward' and 'backward' keys
    """
    pairs = []
    for subject, relation, obj, reverse_relation in FACT_TRIPLES:
        forward = f"{subject} {relation} {obj}"
        backward = f"{obj} {reverse_relation} {subject}"
        pairs.append({
            "forward": forward,
            "backward": backward,
            "subject": subject,
            "object": obj,
        })
    return pairs


def get_training_sentences(include_backward: bool = False) -> List[str]:
    """
    訓練用の文を取得

    Args:
        include_backward: 逆方向も含めるか（デフォルト: False）

    Returns:
        訓練用文のリスト
    """
    pairs = get_reversal_pairs()
    sentences = [p["forward"] for p in pairs]

    if include_backward:
        sentences.extend([p["backward"] for p in pairs])

    return sentences


