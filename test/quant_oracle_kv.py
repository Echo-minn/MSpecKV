#!/usr/bin/env python3
"""
Main evaluation script for QuantSpec + SpeCache hybrid system.
Compares hybrid approach with baseline and logs detailed metrics.
"""

import torch
import time
import sys
import os
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import ModelConfig, load_model_and_tokenizer
from utils.generate import hybrid_quantspec_specache_generate, forward_one_step

TARGET_MODEL_ID = "NousResearch/Yarn-Llama-2-7b-128k"
DRAFT_MODEL_ID = "JackFram/llama-68m"  

DATASET_ID     = "ccdv/cnn_dailymail"


@torch.no_grad()
def baseline_generate(
    model_target,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
):
    """Baseline generation: target model only, no speculation."""
    device = next(model_target.parameters()).device
    
    # Encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    batch_size = input_ids.size(0)
    assert batch_size == 1, "single batch only"
    
    # Initial forward on prompt
    logits, past_key_values = forward_one_step(
        model_target,
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True
    )
    
    # Generate tokens with progress bar
    generated_ids = []
    last_token_id = input_ids[:, -1]
    
    pbar = tqdm(total=max_new_tokens, desc="Baseline Generating", unit="token", ncols=100)
    
    for _ in range(max_new_tokens):
        # Forward one step
        logits, past_key_values = forward_one_step(
            model_target,
            input_ids=last_token_id.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True
        )
        
        # Sample
        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        
        generated_ids.append(next_id)
        last_token_id = torch.tensor([next_id], dtype=torch.long, device=device)
        
        # Update progress bar
        pbar.update(1)
        
        if next_id == tokenizer.eos_token_id:
            break
    
    pbar.close()
    
    # Decode
    output_ids = torch.cat([input_ids[0], torch.tensor(generated_ids, device=device)], dim=0)
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return text, len(generated_ids)


def evaluate_hybrid(
    model_target,
    model_draft,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    draft_ahead_num: int = 4,
    num_bits_kv: int = 8,
    temperature: float = 0.0,
):
    """Run hybrid QuantSpec + SpeCache generation with KV cache quantization."""
    start_time = time.time()
    text, stats = hybrid_quantspec_specache_generate(
        model_target,
        model_draft,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        draft_ahead_num=draft_ahead_num,
        num_bits_kv=num_bits_kv,
        temperature=temperature
    )
    elapsed_time = time.time() - start_time
    
    stats["wall_clock_time"] = elapsed_time
    stats["tokens_per_sec"] = stats["total_generated"] / max(elapsed_time, 1e-6)
    
    return text, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QuantSpec + SpeCache hybrid evaluation")
    parser.add_argument("--model_path", type=str, default=TARGET_MODEL_ID, help="Model path")
    parser.add_argument("--draft_model_path", type=str, default=DRAFT_MODEL_ID, help="Draft model path")
    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--draft_ahead_num", type=int, default=4, help="Number of speculative tokens per block")
    parser.add_argument("--num_bits_kv", type=int, default=8, help="Number of bits for KV cache quantization (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Data type")
    parser.add_argument("--use_flash_attention", action="store_true", default=True, help="Use Flash Attention 2 (default: True)")
    parser.add_argument("--no_flash_attention", dest="use_flash_attention", action="store_false", help="Disable Flash Attention")
    parser.add_argument("--run_baseline", action="store_true", help="Also run baseline for comparison")
    parser.add_argument("--output_file", type=str, default=None, help="Save results to JSON file")
    
    args = parser.parse_args()

    # Read prompt from file
    with open("prompt.txt", "r") as f:
        prompt = f.read()
    
    # Load models
    print(f"Loading target model from {args.model_path}...")
    print(f"Loading draft model from {args.draft_model_path}...")
    model_cfg = ModelConfig(
        target_model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        dtype=args.dtype,
        device=args.device,
        use_flash_attention=args.use_flash_attention
    )
    target_model, draft_model, tokenizer = load_model_and_tokenizer(model_cfg)
    print("Models loaded.")
    
    results = {}
    
    # Run hybrid approach
    print(f"\n{'='*80}")
    print("HYBRID QUANTSPEC + SPECACHE GENERATION")
    print(f"{'='*80}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Draft ahead num: {args.draft_ahead_num}")
    print(f"KV cache quantization bits: {args.num_bits_kv}")
    
    hybrid_text, hybrid_stats = evaluate_hybrid(
        target_model,
        draft_model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        draft_ahead_num=args.draft_ahead_num,
        num_bits_kv=args.num_bits_kv,
        temperature=args.temperature
    )
    
    print("\n" + "="*80)
    print("GENERATED TEXT (Hybrid):")
    print("="*80)
    print(hybrid_text)
    print("="*80)
    
    print("\nHYBRID STATISTICS:")
    print(f"  Total tokens generated: {hybrid_stats['total_generated']}")
    print(f"  Total tokens proposed (draft): {hybrid_stats['total_proposed']}")
    print(f"  Total tokens accepted: {hybrid_stats['total_accepted']}")
    print(f"  Acceptance rate: {hybrid_stats['acceptance_rate']:.2%}")
    print(f"  Target model forwards: {hybrid_stats['total_target_forwards']}")
    print(f"  Target forwards per token: {hybrid_stats['target_forwards_per_token']:.3f}")
    print(f"  Wall-clock time: {hybrid_stats['wall_clock_time']:.3f}s")
    print(f"  Throughput: {hybrid_stats['tokens_per_sec']:.2f} tokens/sec")
    
    results["hybrid"] = {
        "text": hybrid_text,
        "stats": hybrid_stats
    }
    
    # Run baseline if requested
    if args.run_baseline:
        print(f"\n{'='*80}")
        print("BASELINE GENERATION (Target Only)")
        print(f"{'='*80}")
        
        baseline_start = time.time()
        baseline_text, baseline_tokens = baseline_generate(
            target_model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        baseline_time = time.time() - baseline_start
        baseline_tokens_per_sec = baseline_tokens / max(baseline_time, 1e-6)
        
        print("\nBASELINE STATISTICS:")
        print(f"  Total tokens generated: {baseline_tokens}")
        print(f"  Wall-clock time: {baseline_time:.3f}s")
        print(f"  Throughput: {baseline_tokens_per_sec:.2f} tokens/sec")
        print(f"  Target model forwards: {baseline_tokens} (1 per token)")
        
        # Compare
        speedup = baseline_time / hybrid_stats['wall_clock_time']
        print(f"\n{'='*80}")
        print("COMPARISON:")
        print(f"{'='*80}")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Baseline throughput: {baseline_tokens_per_sec:.2f} tokens/sec")
        print(f"  Hybrid throughput: {hybrid_stats['tokens_per_sec']:.2f} tokens/sec")
        print(f"  Target forwards saved: {baseline_tokens - hybrid_stats['total_target_forwards']} "
              f"({(1 - hybrid_stats['total_target_forwards']/baseline_tokens)*100:.1f}% reduction)")
        
        results["baseline"] = {
            "text": baseline_text,
            "stats": {
                "total_generated": baseline_tokens,
                "wall_clock_time": baseline_time,
                "tokens_per_sec": baseline_tokens_per_sec,
                "total_target_forwards": baseline_tokens
            }
        }
        results["comparison"] = {
            "speedup": speedup,
            "target_forwards_saved": baseline_tokens - hybrid_stats['total_target_forwards'],
            "target_forwards_reduction_pct": (1 - hybrid_stats['total_target_forwards']/baseline_tokens)*100
        }
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

