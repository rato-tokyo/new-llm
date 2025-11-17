"""Dialog dataset handling for conversation fine-tuning"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import random


class DialogDataset(Dataset):
    """Dataset for dialogue/conversation modeling

    DailyDialogなどの対話データを処理。
    ターン単位の会話を連結して学習する。
    """

    def __init__(self, dialogues: List[List[str]], tokenizer, max_length: int = 64):
        """
        Args:
            dialogues: List of dialogues, each dialogue is a list of utterances
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        for dialogue in dialogues:
            # 対話全体を連結（ターン区切りは自然に学習される）
            # 形式: [BOS] utterance1 utterance2 utterance3 ... [EOS]
            full_dialogue = " ".join(dialogue)

            tokens = tokenizer.encode(full_dialogue)

            # 長い対話は分割
            if len(tokens) > max_length:
                # スライディングウィンドウで分割
                for i in range(0, len(tokens) - max_length + 1, max_length // 2):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) == max_length:
                        self.sequences.append(chunk)
            else:
                # 短い対話はパディング
                padded = tokens + [0] * (max_length - len(tokens))
                self.sequences.append(padded[:max_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def load_dailydialog_data(config) -> Tuple[DialogDataset, DialogDataset, object]:
    """Load DailyDialog dataset

    DailyDialog: 13k対話、日常会話、10ターン程度
    高品質で多様性のあるデータセット

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with:\n"
            "pip install datasets"
        )

    print("Loading DailyDialog dataset...")
    dataset = load_dataset("daily_dialog")

    # dialogフィールドから対話を取得
    train_dialogues = [item['dialog'] for item in dataset['train']]
    val_dialogues = [item['dialog'] for item in dataset['validation']]

    print(f"Loaded {len(train_dialogues)} training dialogues")
    print(f"Loaded {len(val_dialogues)} validation dialogues")

    # Tokenizerを訓練データから構築
    from .dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

    # 全対話を平坦化してvocab構築
    all_train_texts = []
    for dialogue in train_dialogues:
        all_train_texts.extend(dialogue)

    tokenizer.build_vocab(all_train_texts)

    # データセット作成
    train_dataset = DialogDataset(train_dialogues, tokenizer, config.max_seq_length)
    val_dataset = DialogDataset(val_dialogues, tokenizer, config.max_seq_length)

    print(f"Created {len(train_dataset)} training sequences")
    print(f"Created {len(val_dataset)} validation sequences")
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")

    return train_dataset, val_dataset, tokenizer


def load_personachat_data(config) -> Tuple[DialogDataset, DialogDataset, object]:
    """Load PersonaChat dataset

    PersonaChat: キャラクター性のある対話データ
    DailyDialogより複雑な対話を含む
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("HuggingFace datasets library required")

    print("Loading PersonaChat dataset...")
    # PersonaChatの正確なデータセット名を使用
    dataset = load_dataset("bavard/personachat_truecased")

    # PersonaChatは形式が異なるので注意
    train_dialogues = []
    val_dialogues = []

    # 訓練データから対話を抽出
    for item in dataset['train']:
        if 'utterances' in item and item['utterances']:
            # utterancesフィールドから対話を抽出
            dialogue = []
            for utterance in item['utterances']:
                if 'history' in utterance:
                    dialogue.extend(utterance['history'])
                if 'candidates' in utterance and utterance['candidates']:
                    dialogue.append(utterance['candidates'][0])
            if dialogue:
                train_dialogues.append(dialogue)

    # バリデーションデータから抽出
    for item in dataset['validation']:
        if 'utterances' in item and item['utterances']:
            dialogue = []
            for utterance in item['utterances']:
                if 'history' in utterance:
                    dialogue.extend(utterance['history'])
                if 'candidates' in utterance and utterance['candidates']:
                    dialogue.append(utterance['candidates'][0])
            if dialogue:
                val_dialogues.append(dialogue)

    print(f"Loaded {len(train_dialogues)} training dialogues")
    print(f"Loaded {len(val_dialogues)} validation dialogues")

    # Tokenizer構築
    from .dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

    all_train_texts = []
    for dialogue in train_dialogues:
        all_train_texts.extend(dialogue)
    tokenizer.build_vocab(all_train_texts)

    # データセット作成
    train_dataset = DialogDataset(train_dialogues, tokenizer, config.max_seq_length)
    val_dataset = DialogDataset(val_dialogues, tokenizer, config.max_seq_length)

    print(f"Created {len(train_dataset)} training sequences")
    print(f"Created {len(val_dataset)} validation sequences")

    return train_dataset, val_dataset, tokenizer
