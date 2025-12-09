#!/usr/bin/env python3
"""
Senri Fine-tuning Script

カスタム知識データでSenriモデルをファインチューニング。
知識をメモリに固定し、QAパターンのみを学習する。

Usage:
    # 基本的な使い方
    python3 scripts/finetune.py --data data/custom_knowledge.json --epochs 10

    # ベースモデルを指定
    python3 scripts/finetune.py --data data/custom_knowledge.json --base-model models/pretrained.pt

    # 出力先を指定
    python3 scripts/finetune.py --data data/custom_knowledge.json --output models/finetuned.pt

Input JSON format:
{
  "instances": [
    {
      "knowledge": "東京は日本の首都です。人口は約1400万人。",
      "qa_pairs": [
        {"question": "日本の首都は？", "answer": "東京"},
        {"question": "東京の人口は？", "answer": "約1400万人"}
      ]
    },
    {
      "knowledge": "富士山は日本最高峰の山です。標高は3776m。",
      "qa_pairs": [
        {"question": "日本で一番高い山は？", "answer": "富士山"},
        {"question": "富士山の標高は？", "answer": "3776m"}
      ]
    }
  ]
}
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import SENRI_MODEL
from src.models import SenriModel
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_open_calm_tokenizer
from src.utils.training import get_device


class KnowledgeQADataset(Dataset):
    """知識 + QAペアのデータセット"""

    def __init__(
        self,
        instances: list[dict],
        tokenizer,
        max_knowledge_len: int = 256,
        max_qa_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_knowledge_len = max_knowledge_len
        self.max_qa_len = max_qa_len
        self.samples = []

        for instance in instances:
            knowledge = instance["knowledge"]
            knowledge_tokens = tokenizer.encode(knowledge, add_special_tokens=False)

            # 知識が長すぎる場合は切り詰め
            if len(knowledge_tokens) > max_knowledge_len:
                knowledge_tokens = knowledge_tokens[:max_knowledge_len]

            for qa in instance["qa_pairs"]:
                question = qa["question"]
                answer = qa["answer"]

                # QA部分をトークン化
                qa_text = f" Q: {question} A: {answer}"
                qa_tokens = tokenizer.encode(qa_text, add_special_tokens=False)

                if len(qa_tokens) > max_qa_len:
                    qa_tokens = qa_tokens[:max_qa_len]

                self.samples.append({
                    "knowledge_tokens": knowledge_tokens,
                    "qa_tokens": qa_tokens,
                    "knowledge_len": len(knowledge_tokens),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """バッチをまとめる（パディング付き）"""
    max_knowledge_len = max(s["knowledge_len"] for s in batch)
    max_qa_len = max(len(s["qa_tokens"]) for s in batch)
    max_total_len = max_knowledge_len + max_qa_len

    input_ids_list = []
    labels_list = []
    knowledge_mask_list = []

    for sample in batch:
        knowledge_tokens = sample["knowledge_tokens"]
        qa_tokens = sample["qa_tokens"]

        # 入力: knowledge + qa
        tokens = knowledge_tokens + qa_tokens

        # パディング
        pad_len = max_total_len - len(tokens)
        tokens = tokens + [0] * pad_len  # 0 = pad token

        # ラベル: knowledge部分は-100（無視）、qa部分のみ学習
        labels = [-100] * len(knowledge_tokens) + qa_tokens + [-100] * pad_len

        # 知識マスク（メモリ書き込み対象）
        knowledge_mask = [1] * len(knowledge_tokens) + [0] * (max_qa_len + pad_len)

        input_ids_list.append(tokens)
        labels_list.append(labels)
        knowledge_mask_list.append(knowledge_mask)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "knowledge_mask": torch.tensor(knowledge_mask_list, dtype=torch.bool),
    }


def train_step(
    model: SenriModel,
    batch: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    """1バッチの訓練ステップ"""
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # メモリリセット
    model.reset_memory()

    # Forward pass（メモリ更新あり）
    logits = model(input_ids, update_memory=True)

    # Loss計算（knowledge部分は-100で無視される）
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def evaluate(
    model: SenriModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """評価"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            model.reset_memory()
            logits = model(input_ids, update_memory=True)

            # 有効トークン数をカウント
            valid_mask = labels != -100
            if valid_mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='sum',
            )

            total_loss += loss.item()
            total_tokens += valid_mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return {"loss": avg_loss, "ppl": ppl}


def test_generation(
    model: SenriModel,
    knowledge: str,
    question: str,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 50,
) -> str:
    """知識を与えて質問に回答"""
    model.eval()
    model.reset_memory()

    # 知識をコンテキストとして処理
    knowledge_ids = tokenizer.encode(knowledge, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(knowledge_ids, update_memory=True)

    # 質問を入力して生成
    prompt = f" Q: {question} A:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated_ids)
            next_token = logits[0, -1, :].argmax().item()
            generated_ids = torch.cat([
                generated_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

            # EOSまたは改行で停止
            if next_token == tokenizer.eos_token_id:
                break
            decoded = tokenizer.decode([next_token])
            if '\n' in decoded or 'Q:' in tokenizer.decode(generated_ids[0][-10:]):
                break

    # 回答部分のみ抽出
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "A:" in full_text:
        answer = full_text.split("A:")[-1].strip()
        # 次の質問が始まったら切る
        if "Q:" in answer:
            answer = answer.split("Q:")[0].strip()
        return answer
    return full_text


def main():
    parser = argparse.ArgumentParser(description="Senri Fine-tuning")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to JSON data file"
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Path to base model checkpoint (optional)"
    )
    parser.add_argument(
        "--output", type=str, default="models/finetuned.pt",
        help="Output path for finetuned model"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--max-knowledge-len", type=int, default=256,
        help="Maximum knowledge token length"
    )
    parser.add_argument(
        "--max-qa-len", type=int, default=128,
        help="Maximum QA token length"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    print_flush("=" * 70)
    print_flush("SENRI FINE-TUNING")
    print_flush("=" * 70)

    # データ読み込み
    print_flush(f"\n[1] Loading data: {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instances = data.get("instances", [data])  # 単一インスタンスにも対応
    print_flush(f"    Loaded {len(instances)} knowledge instances")

    total_qa = sum(len(inst.get("qa_pairs", [])) for inst in instances)
    print_flush(f"    Total QA pairs: {total_qa}")

    # トークナイザー
    print_flush("\n[2] Loading tokenizer")
    tokenizer = get_open_calm_tokenizer()

    # データセット作成
    print_flush("\n[3] Creating dataset")
    dataset = KnowledgeQADataset(
        instances,
        tokenizer,
        max_knowledge_len=args.max_knowledge_len,
        max_qa_len=args.max_qa_len,
    )
    print_flush(f"    Total samples: {len(dataset)}")

    # Train/Val分割
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print_flush(f"    Train: {len(train_dataset)} samples")
    print_flush(f"    Val: {len(val_dataset)} samples")

    # モデル作成
    print_flush("\n[4] Creating model")
    if args.base_model:
        print_flush(f"    Loading base model: {args.base_model}")
        checkpoint = torch.load(args.base_model, map_location=device)
        model = SENRI_MODEL()
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print_flush("    Creating new Senri model")
        model = SENRI_MODEL()

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"    Parameters: {total_params:,}")

    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 訓練
    print_flush(f"\n[5] Training ({args.epochs} epochs)")
    best_val_ppl = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            loss = train_step(model, batch, device, optimizer)
            train_loss += loss

        train_loss /= len(train_loader)
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Eval
        val_metrics = evaluate(model, val_loader, device)
        val_ppl = val_metrics["ppl"]

        epoch_time = time.time() - epoch_start

        improved = val_ppl < best_val_ppl
        marker = " *" if improved else ""
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            # ベストモデルを保存
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_ppl": val_ppl,
            }, args.output)

        print_flush(
            f"    Epoch {epoch:2d}: train_ppl={train_ppl:.1f}, "
            f"val_ppl={val_ppl:.1f} [{epoch_time:.1f}s]{marker}"
        )

    print_flush(f"\n    Best: epoch {best_epoch}, val_ppl={best_val_ppl:.1f}")
    print_flush(f"    Saved to: {args.output}")

    # 生成テスト
    print_flush("\n[6] Generation Test")

    # 最初のインスタンスでテスト
    if instances:
        test_instance = instances[0]
        knowledge = test_instance["knowledge"]
        print_flush(f"    Knowledge: {knowledge[:50]}...")

        for qa in test_instance.get("qa_pairs", [])[:2]:
            question = qa["question"]
            expected = qa["answer"]
            generated = test_generation(model, knowledge, question, tokenizer, device)
            print_flush(f"    Q: {question}")
            print_flush(f"    Expected: {expected}")
            print_flush(f"    Generated: {generated}")
            print_flush("")

    print_flush("=" * 70)
    print_flush("DONE")


if __name__ == "__main__":
    main()
