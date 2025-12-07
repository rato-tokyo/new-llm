"""
Family Relation Data Generator

Reversal Curse 汎化性能仮説の検証用データ生成。

設計:
- パターン学習ペア: 順方向・逆方向の両方を学習（汎化パターンを獲得）
- Valペア: 順方向のみ学習 → 逆方向で評価（汎化テスト）
"""

import random
from dataclasses import dataclass


# 名前のプール（英語）
FIRST_NAMES = [
    # 女性名
    "Alice", "Emma", "Olivia", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amelia", "Harper", "Evelyn", "Abigail", "Emily", "Elizabeth", "Sofia",
    "Avery", "Ella", "Scarlett", "Grace", "Chloe", "Victoria", "Riley",
    "Aria", "Lily", "Aurora", "Zoey", "Nora", "Hannah", "Eleanor",
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


# =============================================================================
# Baseline用サンプル生成
# =============================================================================


def create_baseline_pattern_samples(pairs: list[FamilyPair]) -> list[dict]:
    """
    Baseline: パターン学習ペア用サンプル（順方向・逆方向両方）

    全文を学習対象とする従来のLM学習方式。
    """
    samples = []

    for pair in pairs:
        # 順方向: "X is Y's parent. Who is Y's parent? X"
        forward_text = (
            f"{pair.parent_name} is {pair.child_name}'s {pair.relation}. "
            f"Who is {pair.child_name}'s parent? {pair.parent_name}"
        )
        samples.append({
            "text": forward_text,
            "type": "pattern_forward",
        })

        # 逆方向: "Y is X's child. Who is X's child? Y"
        # 注: childrenではなくchildを使用（文法的に正確）
        backward_text = (
            f"{pair.child_name} is {pair.parent_name}'s child. "
            f"Who is {pair.parent_name}'s child? {pair.child_name}"
        )
        samples.append({
            "text": backward_text,
            "type": "pattern_backward",
        })

    return samples


def create_baseline_val_samples(pairs: list[FamilyPair]) -> list[dict]:
    """
    Baseline: Valペア用サンプル（順方向のみ）

    このペアでは逆方向を学習しない。評価時に逆方向で汎化テスト。
    """
    samples = []

    for pair in pairs:
        # 順方向のみ
        forward_text = (
            f"{pair.parent_name} is {pair.child_name}'s {pair.relation}. "
            f"Who is {pair.child_name}'s parent? {pair.parent_name}"
        )
        samples.append({
            "text": forward_text,
            "type": "val_forward",
        })

    return samples


# =============================================================================
# Modified用サンプル生成（知識分離訓練）
# =============================================================================


def create_modified_pattern_samples(pairs: list[FamilyPair]) -> list[dict]:
    """
    Modified: パターン学習ペア用サンプル（コンテキスト分離）

    初期コンテキストをloss計算から除外し、推論パターンのみを学習。
    """
    samples = []

    for pair in pairs:
        # 順方向: context="X is Y's parent." → 学習対象="Who is Y's parent? X"
        samples.append({
            "context": f"{pair.parent_name} is {pair.child_name}'s {pair.relation}.",
            "question": f"Who is {pair.child_name}'s parent?",
            "answer": f" {pair.parent_name}",
            "type": "pattern_forward",
        })

        # 逆方向: context="Y is X's child." → 学習対象="Who is X's child? Y"
        samples.append({
            "context": f"{pair.child_name} is {pair.parent_name}'s child.",
            "question": f"Who is {pair.parent_name}'s child?",
            "answer": f" {pair.child_name}",
            "type": "pattern_backward",
        })

    return samples


def create_modified_no_context_samples(pairs: list[FamilyPair]) -> list[dict]:
    """
    Modified: コンテキストなしサンプル（知識分離確認）

    コンテキストがない場合は「わからない」と答えることを学習。
    """
    samples = []

    for pair in pairs:
        # 順方向の質問（コンテキストなし）
        samples.append({
            "context": "",
            "question": f"Who is {pair.child_name}'s parent?",
            "answer": " I don't know.",
            "type": "no_context",
        })

        # 逆方向の質問（コンテキストなし）
        samples.append({
            "context": "",
            "question": f"Who is {pair.parent_name}'s child?",
            "answer": " I don't know.",
            "type": "no_context",
        })

    return samples


def create_modified_val_samples(pairs: list[FamilyPair]) -> list[dict]:
    """
    Modified: Valペア用サンプル（順方向のみ、全文学習）

    Baselineと同一のデータ。公平な比較のため。
    """
    samples = []

    for pair in pairs:
        # 順方向のみ（Baselineと同一形式）
        forward_text = (
            f"{pair.parent_name} is {pair.child_name}'s {pair.relation}. "
            f"Who is {pair.child_name}'s parent? {pair.parent_name}"
        )
        samples.append({
            "text": forward_text,
            "type": "val_forward",
        })

    return samples


# =============================================================================
# ユーティリティ
# =============================================================================


def split_pairs_for_experiment(
    all_pairs: list[FamilyPair],
    num_val_pairs: int = 20,
) -> tuple[list[FamilyPair], list[FamilyPair]]:
    """
    パターン学習ペアとValペアに分割

    Args:
        all_pairs: 全ペア
        num_val_pairs: Valペア数

    Returns:
        (pattern_pairs, val_pairs)
    """
    # シャッフルせず、最後のnum_val_pairsをValペアとする
    # （再現性のため）
    pattern_pairs = all_pairs[:-num_val_pairs]
    val_pairs = all_pairs[-num_val_pairs:]
    return pattern_pairs, val_pairs


# 後方互換性のため残す（使用非推奨）
def create_baseline_samples(pairs: list[FamilyPair]) -> list[dict]:
    """非推奨: create_baseline_pattern_samplesを使用してください"""
    return create_baseline_pattern_samples(pairs)


def create_cdr_samples(pairs: list[FamilyPair]) -> list[dict]:
    """非推奨: create_modified_pattern_samplesを使用してください"""
    return create_modified_pattern_samples(pairs)
