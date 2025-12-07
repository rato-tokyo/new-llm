"""
Family Relation Data Generator

知識と推論を分離した訓練データを生成する。
- コンテキスト付き: 関係性が与えられた状態で推論
- コンテキストなし: 知らない人物には「わからない」と回答

目的: FFNに関係性パターン（母親↔子供など）を学習させ、Reversal Curseを軽減
"""

import random
from dataclasses import dataclass


# 名前のプール（英語）
FIRST_NAMES = [
    # 女性名
    "Alice", "Emma", "Olivia", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amelia", "Harper", "Evelyn", "Abigail", "Emily", "Elizabeth", "Sofia",
    "Avery", "Ella", "Scarlett", "Grace", "Chloe", "Victoria", "Riley",
    "Aria", "Lily", "Aurora", "Zoey", "Nora", "Hannah", "Lily", "Eleanor",
    "Hazel", "Violet", "Luna", "Stella", "Natalie", "Zoe", "Leah", "Penelope",
    # 男性名
    "James", "William", "Oliver", "Benjamin", "Elijah", "Lucas", "Mason",
    "Ethan", "Alexander", "Henry", "Sebastian", "Jack", "Aiden", "Owen",
    "Samuel", "Ryan", "Nathan", "Leo", "Isaac", "Luke", "Gabriel", "Anthony",
    "Dylan", "Lincoln", "Jaxon", "Asher", "Christopher", "Joshua", "Andrew",
    "Theodore", "Caleb", "Thomas", "Charles", "Daniel", "Matthew", "Joseph",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
]


@dataclass
class FamilyPair:
    """親子ペア"""
    parent_name: str
    child_name: str
    relation: str  # "mother" or "father"


def generate_unique_name(used_names: set) -> str:
    """ユニークな名前を生成"""
    for _ in range(100):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        full_name = f"{first} {last}"
        if full_name not in used_names:
            used_names.add(full_name)
            return full_name
    raise ValueError("Could not generate unique name")


def generate_family_pairs(num_pairs: int, seed: int = 42) -> list[FamilyPair]:
    """親子ペアを生成"""
    random.seed(seed)
    used_names: set[str] = set()
    pairs = []

    for _ in range(num_pairs):
        parent_name = generate_unique_name(used_names)
        child_name = generate_unique_name(used_names)
        relation = random.choice(["mother", "father"])
        pairs.append(FamilyPair(parent_name, child_name, relation))

    return pairs


def create_context_qa_samples(
    pairs: list[FamilyPair],
    include_negative: bool = True,
) -> list[dict]:
    """
    コンテキスト付きQAサンプルを生成

    Args:
        pairs: 親子ペア
        include_negative: コンテキストなし（わからない）サンプルを含めるか

    Returns:
        list of {"input": str, "target": str, "has_context": bool}
    """
    samples = []

    for pair in pairs:
        # 1. コンテキスト付き: 順方向（親→子を推論）
        # Context: "Alice Smith is Tom Johnson's mother."
        # Question: "Who is Tom Johnson's parent?"
        # Answer: "Alice Smith"
        context = f"{pair.parent_name} is {pair.child_name}'s {pair.relation}."
        question_parent = f"Who is {pair.child_name}'s parent?"

        samples.append({
            "input": f"Context: {context} Question: {question_parent} Answer:",
            "target": f" {pair.parent_name}",
            "has_context": True,
            "direction": "forward",
        })

        # 2. コンテキスト付き: 逆方向（子→親を推論）
        # Context: "Alice Smith is Tom Johnson's mother."
        # Question: "Who is Alice Smith's child?"
        # Answer: "Tom Johnson"
        question_child = f"Who is {pair.parent_name}'s child?"

        samples.append({
            "input": f"Context: {context} Question: {question_child} Answer:",
            "target": f" {pair.child_name}",
            "has_context": True,
            "direction": "backward",
        })

    if include_negative:
        # 3. コンテキストなし: 知らない人物
        # 別のペアから名前を借りて、コンテキストなしで質問
        for pair in pairs:
            # この人物についてのコンテキストがない状態で質問
            question = f"Who is {pair.child_name}'s parent?"
            samples.append({
                "input": f"Context: None. Question: {question} Answer:",
                "target": " I don't know.",
                "has_context": False,
                "direction": "unknown",
            })

            question = f"Who is {pair.parent_name}'s child?"
            samples.append({
                "input": f"Context: None. Question: {question} Answer:",
                "target": " I don't know.",
                "has_context": False,
                "direction": "unknown",
            })

    return samples


def create_baseline_samples(pairs: list[FamilyPair]) -> list[dict]:
    """
    ベースライン用サンプルを生成（コンテキストなし、直接学習）

    通常のReversal Curse実験と同様:
    - 順方向: "Tom Johnson's mother is Alice Smith."
    - 評価時に逆方向を問う
    """
    samples = []

    for pair in pairs:
        # 順方向の事実のみ学習（通常のLM訓練）
        statement = f"{pair.child_name}'s {pair.relation} is {pair.parent_name}."
        samples.append({
            "input": statement,
            "target": "",  # 文全体を学習
            "direction": "forward_only",
        })

    return samples


def create_test_pairs(num_pairs: int, seed: int = 12345) -> list[FamilyPair]:
    """
    テスト用ペアを生成（訓練データとは別の名前）
    """
    return generate_family_pairs(num_pairs, seed=seed)


def split_pairs(
    pairs: list[FamilyPair],
    train_ratio: float = 0.8,
) -> tuple[list[FamilyPair], list[FamilyPair]]:
    """訓練/検証に分割"""
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    return pairs[:split_idx], pairs[split_idx:]


# 有名人テストペア（実在の親子関係）
CELEBRITY_TEST_PAIRS = [
    FamilyPair("Mary Lee Pfeiffer", "Tom Cruise", "mother"),
    FamilyPair("Gloria Henry", "Ron Howard", "mother"),
    FamilyPair("Jada Pinkett Smith", "Jaden Smith", "mother"),
    FamilyPair("Goldie Hawn", "Kate Hudson", "mother"),
    FamilyPair("Judy Garland", "Liza Minnelli", "mother"),
    FamilyPair("Janet Sheen", "Charlie Sheen", "mother"),
    FamilyPair("Donna Summer", "Brooklyn Sudano", "mother"),
    FamilyPair("Debbie Reynolds", "Carrie Fisher", "mother"),
]


def get_celebrity_test_pairs() -> list[FamilyPair]:
    """有名人テストペアを取得"""
    return CELEBRITY_TEST_PAIRS.copy()
