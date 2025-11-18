#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation and Comparison

訓練済みモデルを評価し、生成品質を比較する。

評価項目:
1. Perplexity (定量評価)
2. テキスト生成品質 (定性評価)
3. 推論速度
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils.config import NewLLMConfig
from src.models.context_vector_llm import ContextVectorLLM
from src.models.gpt2_baseline import create_gpt2_baseline
from src.training.dialog_dataset import load_dailydialog_data
from src.training.wikitext_dataset import load_wikitext_data
import time
import math


class EvaluationConfig(NewLLMConfig):
    """評価用設定"""
    max_seq_length = 64
    vocab_size = 1000
    batch_size = 16
    device = "cpu"


def load_model(model_type: str, checkpoint_path: str, config):
    """モデルをロード

    Args:
        model_type: 'new_llm' or 'tinygpt2'
        checkpoint_path: チェックポイントファイルのパス
        config: 設定オブジェクト

    Returns:
        モデルインスタンス
    """
    if model_type == 'new_llm':
        model = ContextVectorLLM(config)
    elif model_type == 'tinygpt2':
        model = create_gpt2_baseline(config, tiny=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # チェックポイントをロード
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.to(config.device)
    model.eval()
    return model


def calculate_perplexity(model, dataloader, device):
    """Perplexityを計算

    Args:
        model: 評価するモデル
        dataloader: データローダー
        device: デバイス

    Returns:
        perplexity, loss
    """
    model.eval()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            # Loss計算
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0  # PADを無視
            )

            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def generate_text(model, tokenizer, prompt: str, max_length: int = 50, device='cpu'):
    """テキスト生成

    Args:
        model: 生成モデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        max_length: 最大生成長
        device: デバイス

    Returns:
        生成されたテキスト
    """
    model.eval()

    # プロンプトをトークン化
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens]).to(device)

    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(max_length):
            # 現在のシーケンスで予測
            if input_ids.size(1) > model.max_seq_length:
                # 長すぎる場合は最新部分のみ使用
                input_ids = input_ids[:, -model.max_seq_length:]

            logits = model(input_ids)

            # 最後のトークンの予測
            next_token_logits = logits[0, -1, :]

            # Greedy sampling（最も確率の高いトークンを選択）
            next_token = torch.argmax(next_token_logits).item()

            # EOSで終了
            if next_token == 3:  # <EOS>
                break

            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)

    # デコード
    generated_text = tokenizer.decode(generated)
    return generated_text


def measure_inference_speed(model, dataloader, device, num_batches=10):
    """推論速度を測定

    Args:
        model: モデル
        dataloader: データローダー
        device: デバイス
        num_batches: 測定するバッチ数

    Returns:
        平均推論時間（秒/バッチ）
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs = inputs.to(device)

            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()

            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    return avg_time


def main():
    """メイン評価スクリプト"""
    print("="*80)
    print("Model Evaluation and Comparison")
    print("="*80)

    config = EvaluationConfig()

    # データロード
    print("\nLoading evaluation dataset (DailyDialog)...")
    _, val_dataset, tokenizer = load_dailydialog_data(config)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Validation samples: {len(val_dataset)}")

    # モデルをロード
    print("\n" + "="*80)
    print("Loading Models")
    print("="*80)

    new_llm_path = "checkpoints/best_new_llm_dialog.pt"
    tinygpt2_path = "checkpoints/best_tinygpt2_dialog.pt"

    try:
        print("\n[1/2] Loading New-LLM...")
        new_llm = load_model('new_llm', new_llm_path, config)
    except FileNotFoundError:
        print(f"⚠ New-LLM checkpoint not found. Skipping.")
        new_llm = None

    try:
        print("\n[2/2] Loading TinyGPT2...")
        tinygpt2 = load_model('tinygpt2', tinygpt2_path, config)
    except FileNotFoundError:
        print(f"⚠ TinyGPT2 checkpoint not found. Skipping.")
        tinygpt2 = None

    # 評価1: Perplexity
    print("\n" + "="*80)
    print("Evaluation 1: Perplexity on DailyDialog")
    print("="*80)

    results = {}

    if new_llm:
        print("\nEvaluating New-LLM...")
        ppl, loss = calculate_perplexity(new_llm, val_dataloader, config.device)
        results['new_llm'] = {'perplexity': ppl, 'loss': loss}
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Loss: {loss:.4f}")

    if tinygpt2:
        print("\nEvaluating TinyGPT2...")
        ppl, loss = calculate_perplexity(tinygpt2, val_dataloader, config.device)
        results['tinygpt2'] = {'perplexity': ppl, 'loss': loss}
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Loss: {loss:.4f}")

    # 評価2: テキスト生成
    print("\n" + "="*80)
    print("Evaluation 2: Text Generation Quality")
    print("="*80)

    test_prompts = [
        "hello how are you",
        "what is your name",
        "tell me about yourself",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 60)

        if new_llm:
            generated = generate_text(new_llm, tokenizer, prompt, max_length=30, device=config.device)
            print(f"New-LLM:   {generated}")

        if tinygpt2:
            generated = generate_text(tinygpt2, tokenizer, prompt, max_length=30, device=config.device)
            print(f"TinyGPT2:  {generated}")

    # 評価3: 推論速度
    print("\n" + "="*80)
    print("Evaluation 3: Inference Speed")
    print("="*80)

    if new_llm:
        speed = measure_inference_speed(new_llm, val_dataloader, config.device)
        results['new_llm']['speed'] = speed
        print(f"\nNew-LLM:   {speed*1000:.2f} ms/batch")

    if tinygpt2:
        speed = measure_inference_speed(tinygpt2, val_dataloader, config.device)
        results['tinygpt2']['speed'] = speed
        print(f"TinyGPT2:  {speed*1000:.2f} ms/batch")

    # 最終まとめ
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    if 'new_llm' in results and 'tinygpt2' in results:
        new_llm_ppl = results['new_llm']['perplexity']
        tinygpt2_ppl = results['tinygpt2']['perplexity']

        print(f"\nPerplexity:")
        print(f"  New-LLM:   {new_llm_ppl:.2f}")
        print(f"  TinyGPT2:  {tinygpt2_ppl:.2f}")

        ppl_diff = ((new_llm_ppl - tinygpt2_ppl) / tinygpt2_ppl) * 100
        print(f"  Difference: {ppl_diff:+.1f}%")

        if new_llm_ppl < tinygpt2_ppl:
            print("  ✓ New-LLM achieves better perplexity!")
        else:
            print("  ✓ TinyGPT2 achieves better perplexity")

        print(f"\nInference Speed:")
        print(f"  New-LLM:   {results['new_llm']['speed']*1000:.2f} ms/batch")
        print(f"  TinyGPT2:  {results['tinygpt2']['speed']*1000:.2f} ms/batch")

        speed_diff = ((results['new_llm']['speed'] - results['tinygpt2']['speed']) /
                      results['tinygpt2']['speed']) * 100
        print(f"  Difference: {speed_diff:+.1f}%")

        if results['new_llm']['speed'] < results['tinygpt2']['speed']:
            print("  ✓ New-LLM is faster!")
        else:
            print("  ✓ TinyGPT2 is faster")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
