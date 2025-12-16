# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 test/compare_methods.py --budget 4096 --prefill 8192 --dataset demo --target llama-7B-128K --on_chip 4 --gamma 8 --output_csv comparison.csv

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch.distributed as dist
import torch
import argparse
from termcolor import colored
from utils.decoding import TriForce_Dist
from models.TP_llama import distributed_init, DistributedLlama
from models.modeling_llama import LlamaForCausalLM
from models.modeling_llama_68m import LlamaForCausalLM as LlamaForCausalLM_68M
from models.cache import StreamingLLMEvictionCache
from transformers import AutoTokenizer
import numpy as np
import time
from tqdm import tqdm
import csv
import json
from datetime import datetime

local_rank, world_size = distributed_init()
device = torch.device("cuda", local_rank)


def parse_arguments():
    parser = argparse.ArgumentParser(description='TriForce with KV Cache Quantization')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--prefill', type=int, default=8192, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=64, help='generation length')
    parser.add_argument('--gamma', type=int, default=8, help='speculation depth')
    parser.add_argument('--budget', type=int, default=4096, help='retrieval budget for TriForce')
    parser.add_argument('--on_chip', type=int, default=12, help='on chip layers')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset')
    parser.add_argument('--resident_layers', type=int, default=8, help='resident layers')
    args = parser.parse_args()
    
    return args

args = parse_arguments()
torch.manual_seed(args.seed)
prefill = args.prefill
gen_len = args.gen_len
temperature = args.temp
top_p = args.top_p
retrieval_budget = args.budget
gamma = args.gamma

# Model setup
if args.target == 'llama-13B-128K':
    model_name_or_path = "NousResearch/Yarn-Llama-2-13b-128k"
elif args.target == 'llama-7B-128K':
    model_name_or_path = "NousResearch/Yarn-Llama-2-7b-128k"
elif args.target == 'lwm-128K':
   model_name_or_path = "LargeWorldModel/LWM-Text-Chat-128K"
elif args.target == 'lwm-128K-base':
   model_name_or_path = "LargeWorldModel/LWM-Text-128K"
else:
    raise NotImplementedError

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, legacy=False)

from data.dataset import get_dataset
tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=32768)
tokenized_prompts = tokenized_prompts[:1]

os.makedirs("results", exist_ok=True)

# Store all results
comparison_results = {
    'config': {
        'model': args.target,
        'prefill_length': prefill,
        'gen_length': gen_len,
        'temperature': temperature,
        'top_p': top_p,
        'on_chip_layers': args.on_chip,
        'gamma': gamma,
        'world_size': world_size,
        'dataset': args.dataset,
        'resident_layers': args.resident_layers,
        'num_samples': len(tokenized_prompts),
        'timestamp': datetime.now().isoformat()
    },
    'methods': {}
}
# =============================================================================
# TriForce with KV Cache Quantization
# =============================================================================

draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map=device)
draft = draft.eval()
draft_cache_budget = 256
recent_size = draft_cache_budget - 16 - gamma

if local_rank == 0:
    print(colored("\n" + "=" * 80, "yellow"))
    print(colored("METHOD 4: TRIFORCE WITH KV CACHE QUANTIZATION", "yellow"))
    print(colored("=" * 80, "yellow"))

draft_cache_triforce_quant = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

llm_triforce_quant = DistributedLlama(
    model_name_or_path=model_name_or_path,
    local_rank=local_rank,
    world_size=world_size,
    prefill=prefill,
    gen_len=gen_len,
    temperature=temperature,
    top_p=top_p,
    flash_attn=True,
    retrieval_budget=retrieval_budget,  # With retrieval
    kv_offload=True,
    on_chip_layers=args.on_chip,
    draft=draft,
    draft_cache=draft_cache_triforce_quant,
    gamma=gamma,
    kv_cache_quant=True,
    kv_cache_quant_bits=8,
    kv_fp16_tail=0,
    kv_resident_gpu=True,
    kv_resident_max_layers=args.resident_layers
)

for rank in range(world_size):
    if local_rank == rank:
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
        llm_triforce_quant.init_parameters(hf_model=hf_model)
        del hf_model
    dist.barrier()

triforce_quant_latencies = []
triforce_quant_latency_per_token_s = []  # seconds/token (as returned by TriForce_Dist)
triforce_quant_accepted_tokens = []

for idx, input_ids in enumerate(tqdm(tokenized_prompts, desc="TriForce_KVQuant", disable=(local_rank != 0))):
    input_ids = input_ids[:,:prefill].to(llm_triforce_quant.device)
    
    avg_tokens, latency = TriForce_Dist(tokenizer, llm_triforce_quant, input_ids, gamma=gamma, max_len=gen_len, top_k=-1, top_p=top_p, temperature=temperature, verbose=False, file_path=None, dataset=args.dataset)
    
    # Convert to Python scalars if they're tensors
    if torch.is_tensor(avg_tokens):
        avg_tokens = avg_tokens.item()
    if torch.is_tensor(latency):
        latency = latency.item()
    
    # NOTE: TriForce_Dist returns **seconds per generated token**: (time2-time1)/n.
    # Convert to total sample latency to make it comparable to Baseline/Vanilla.
    latency_per_token_s = float(latency)
    total_latency_s = latency_per_token_s * gen_len
    triforce_quant_latency_per_token_s.append(latency_per_token_s)
    triforce_quant_latencies.append(total_latency_s)
    triforce_quant_accepted_tokens.append(avg_tokens)
    
    if local_rank == 0:
        print(colored(
            f"  Sample {idx}: {total_latency_s:.4f}s total, "
            f"{(1.0/max(latency_per_token_s, 1e-9)):.2f} tok/s, "
            f"{avg_tokens:.2f} accepted",
            "green"
        ))

if local_rank == 0:
    mean_triforce_quant_latency = np.mean(triforce_quant_latencies)
    std_triforce_quant_latency = np.std(triforce_quant_latencies)
    mean_triforce_quant_throughput = gen_len / mean_triforce_quant_latency
    mean_triforce_quant_accepted = np.mean(triforce_quant_accepted_tokens)
    
    mean_triforce_quant_latency_per_token_s = float(np.mean(triforce_quant_latency_per_token_s)) if len(triforce_quant_latency_per_token_s) > 0 else None
    comparison_results['methods']['triforce_kv_quant'] = {
        'name': 'TriForce with KV Cache Quantization',
        'retrieval_budget': retrieval_budget,
        'mean_latency_s': float(mean_triforce_quant_latency),
        'std_latency_s': float(std_triforce_quant_latency),
        'mean_throughput_tok_per_s': float(mean_triforce_quant_throughput),
        'mean_accepted_tokens': float(mean_triforce_quant_accepted),
        'mean_latency_s_per_token': mean_triforce_quant_latency_per_token_s,
        'samples': [{'latency_s': float(lat), 'throughput': float(gen_len/lat), 'accepted_tokens': float(acc)} 
                    for lat, acc in zip(triforce_quant_latencies, triforce_quant_accepted_tokens)]
    }
    
    print(colored(f"\n[MSpecKV] Mean Latency: {mean_triforce_quant_latency:.4f} ± {std_triforce_quant_latency:.4f} s", "green"))
    print(colored(f"[MSpecKV] Mean Throughput: {mean_triforce_quant_throughput:.2f} tok/s", "green"))
    print(colored(f"[MSpecKV] Mean Accepted Tokens: {mean_triforce_quant_accepted:.2f}", "green"))

dist.barrier()
del llm_triforce_quant
dist.barrier()
torch.cuda.empty_cache()

# =============================================================================
# METHOD 3: TriForce (Speculative Decoding + Retrieval Cache)
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "green"))
    print(colored("METHOD 3: TRIFORCE (SpecDec + Retrieval Cache)", "green"))
    print(colored(f"Retrieval Budget: {retrieval_budget}", "green"))
    print(colored("=" * 80, "green"))

draft_cache_triforce = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

llm_triforce = DistributedLlama(
    model_name_or_path=model_name_or_path,
    local_rank=local_rank,
    world_size=world_size,
    prefill=prefill,
    gen_len=gen_len,
    temperature=temperature,
    top_p=top_p,
    flash_attn=True,
    retrieval_budget=retrieval_budget,  # With retrieval
    kv_offload=True,
    on_chip_layers=args.on_chip,
    draft=draft,
    draft_cache=draft_cache_triforce,
    gamma=gamma
)

for rank in range(world_size):
    if local_rank == rank:
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
        llm_triforce.init_parameters(hf_model=hf_model)
        del hf_model
    dist.barrier()

triforce_latencies = []
triforce_latency_per_token_s = []  # seconds/token (as returned by TriForce_Dist)
triforce_accepted_tokens = []

for idx, input_ids in enumerate(tqdm(tokenized_prompts, desc="TriForce", disable=(local_rank != 0))):
    input_ids = input_ids[:,:prefill].to(llm_triforce.device)
    
    avg_tokens, latency = TriForce_Dist(tokenizer, llm_triforce, input_ids, gamma=gamma, max_len=gen_len, top_k=-1, top_p=top_p, temperature=temperature, verbose=False, file_path=None, dataset=args.dataset)
    
    # Convert to Python scalars if they're tensors
    if torch.is_tensor(avg_tokens):
        avg_tokens = avg_tokens.item()
    if torch.is_tensor(latency):
        latency = latency.item()
    
    # NOTE: TriForce_Dist returns **seconds per generated token**: (time2-time1)/n.
    # Convert to total sample latency to make it comparable to Baseline/Vanilla.
    latency_per_token_s = float(latency)
    total_latency_s = latency_per_token_s * gen_len
    triforce_latency_per_token_s.append(latency_per_token_s)
    triforce_latencies.append(total_latency_s)
    triforce_accepted_tokens.append(avg_tokens)
    
    if local_rank == 0:
        print(colored(
            f"  Sample {idx}: {total_latency_s:.4f}s total, "
            f"{(1.0/max(latency_per_token_s, 1e-9)):.2f} tok/s, "
            f"{avg_tokens:.2f} accepted",
            "green"
        ))

if local_rank == 0:
    mean_triforce_latency = np.mean(triforce_latencies)
    std_triforce_latency = np.std(triforce_latencies)
    mean_triforce_throughput = gen_len / mean_triforce_latency
    mean_triforce_accepted = np.mean(triforce_accepted_tokens)
    mean_triforce_latency_per_token_s = float(np.mean(triforce_latency_per_token_s)) if len(triforce_latency_per_token_s) > 0 else None
    
    comparison_results['methods']['triforce'] = {
        'name': 'TriForce (SpecDec + Retrieval)',
        'retrieval_budget': retrieval_budget,
        'mean_latency_s': float(mean_triforce_latency),
        'std_latency_s': float(std_triforce_latency),
        'mean_throughput_tok_per_s': float(mean_triforce_throughput),
        'mean_accepted_tokens': float(mean_triforce_accepted),
        'mean_latency_s_per_token': mean_triforce_latency_per_token_s,
        'samples': [{'latency_s': float(lat), 'throughput': float(gen_len/lat), 'accepted_tokens': float(acc)} 
                    for lat, acc in zip(triforce_latencies, triforce_accepted_tokens)]
    }
    
    print(colored(f"\n[TriForce] Mean Latency: {mean_triforce_latency:.4f} ± {std_triforce_latency:.4f} s", "green"))
    print(colored(f"[TriForce] Mean Throughput: {mean_triforce_throughput:.2f} tok/s", "green"))
    print(colored(f"[TriForce] Mean Accepted Tokens: {mean_triforce_accepted:.2f}", "green"))
    print(colored(f"[TriForce] Speedup vs non-quantized SpecDec: {mean_triforce_latency / mean_triforce_quant_latency:.2f}x", "green"))

dist.barrier()
del llm_triforce
dist.barrier()
torch.cuda.empty_cache()

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("COMPARISON SUMMARY", "magenta"))
    print(colored("=" * 80, "magenta"))
    
    print(f"\n{'Method':<35} {'Latency (s)':<15} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 80)
    print(f"{'TriForce (SpecDec + Retrieval)':<35} {mean_triforce_latency:>6.4f} ± {std_triforce_latency:<5.4f} {mean_triforce_throughput:>8.2f} tok/s")
    print(f"{'TriForce (KV Cache Quantization)':<35} {mean_triforce_quant_latency:>6.4f} ± {std_triforce_quant_latency:<5.4f} {mean_triforce_quant_throughput:>8.2f} tok/s")
    
    print(f"\n{'Method':<35} {'Accepted Tokens':<20}")
    print("-" * 55)    
    print(f"{'TriForce (SpecDec + Retrieval)':<35} {mean_triforce_accepted:>8.2f}")
    print(f"{'TriForce (KV Cache Quantization)':<35} {mean_triforce_quant_accepted:>8.2f}")
    
    # Save JSON
    with open(f"results/triforce_quant_{args.target}_p{args.prefill}_g{args.gamma}_b{args.budget}.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(colored(f"\n✓ Detailed results saved to results/triforce_quant_{args.target}_p{args.prefill}_g{args.gamma}_b{args.budget}_r{args.resident_layers}.json", "cyan"))
    
    # Save CSV
    csv_file = f"results/triforce_quant_{args.target}_p{args.prefill}_g{args.gamma}_b{args.budget}_r{args.resident_layers}.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            'timestamp', 'model', 'prefill_length', 'gen_length', 'gamma', 'budget', 'on_chip_layers',
            'resident_layers',
            'triforce_latency_s', 'triforce_throughput', 'triforce_accepted_tokens',
            'triforce_quant_latency_s', 'triforce_quant_throughput', 'triforce_quant_accepted_tokens'
        ])
        
        writer.writerow([
            comparison_results['config']['timestamp'],
            args.target,
            prefill,
            gen_len,
            gamma,
            retrieval_budget,
            args.on_chip,
            args.resident_layers,
            f"{mean_triforce_latency:.4f}",
            f"{mean_triforce_throughput:.2f}",
            f"{mean_triforce_accepted:.2f}",
            f"{mean_triforce_quant_latency:.4f}",
            f"{mean_triforce_quant_throughput:.2f}",
            f"{mean_triforce_quant_accepted:.2f}",
        ])
    
    print(colored(f"✓ Comparison appended to {csv_file}", "cyan"))
    print(colored("\n" + "=" * 80 + "\n", "magenta"))

dist.destroy_process_group()
