# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 test/compare_methods.py --budget 4096 --prefill 8192 --dataset demo --target llama-7B-128K --on_chip 4 --gamma 8 --output_csv comparison.csv

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import argparse
from termcolor import colored
from utils.decoding import Baseline_Dist, VanillaSpecDec_Dist, MSpecKV_Dist
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
    parser = argparse.ArgumentParser(description='Compare: Baseline vs Vanilla SpecDec vs MSpecKV')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--prefill', type=int, default=8192, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset')
    parser.add_argument('--on_chip', type=int, default=4, help='on chip layers')
    parser.add_argument('--budget', type=int, default=4096, help='retrieval budget for MSpecKV')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--gamma', type=int, default=8, help='speculation depth')
    parser.add_argument('--resident_layers', type=int, default=8, help='resident layers')
    parser.add_argument('--output_csv', type=str, default='comparison.csv', help='output CSV file (use "auto" for automatic naming)')
    parser.add_argument('--output_json', type=str, default='comparison.json', help='output JSON file (use "auto" for automatic naming)')
    parser.add_argument('--separate_files', action='store_true', help='create separate files per experiment (timestamp-based)')
    args = parser.parse_args()
    
    # Auto-generate filenames if requested
    if args.output_csv == 'auto' or args.separate_files:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{args.target}_p{args.prefill}_g{args.gamma}_b{args.budget}"
        args.output_csv = f"results/comparison_{base_name}_{timestamp_str}.csv"
        args.output_json = f"results/comparison_{base_name}_{timestamp_str}.json"
        # Create results directory if needed
        os.makedirs('results', exist_ok=True)
    
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
# Draft model setup (used by Vanilla SpecDec + MSpecKV)
# =============================================================================
draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map=device)
draft = draft.eval()
draft_cache_budget = 256
recent_size = draft_cache_budget - 16 - gamma

# =============================================================================
# METHOD 4: MSpecKV with KV Cache Quantization
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "yellow"))
    print(colored("METHOD 4: TRIFORCE WITH KV CACHE QUANTIZATION", "yellow"))
    print(colored("=" * 80, "yellow"))

draft_cache_mspeckv_quant = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

llm_mspeckv_quant = DistributedLlama(
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
    draft_cache=draft_cache_mspeckv_quant,
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
        llm_mspeckv_quant.init_parameters(hf_model=hf_model)
        del hf_model
    dist.barrier()

mspeckv_quant_latencies = []
mspeckv_quant_latency_per_token_s = []  # seconds/token (as returned by MSpecKV_Dist)
mspeckv_quant_accepted_tokens = []

for idx, input_ids in enumerate(tqdm(tokenized_prompts, desc="MSpecKV_KVQuant", disable=(local_rank != 0))):
    input_ids = input_ids[:,:prefill].to(llm_mspeckv_quant.device)
    
    avg_tokens, latency = MSpecKV_Dist(tokenizer, llm_mspeckv_quant, input_ids, gamma=gamma, max_len=gen_len, top_k=-1, top_p=top_p, temperature=temperature, verbose=False, file_path=None, dataset=args.dataset)
    
    if torch.is_tensor(avg_tokens):
        avg_tokens = avg_tokens.item()
    if torch.is_tensor(latency):
        latency = latency.item()
    
    latency_per_token_s = float(latency)
    total_latency_s = latency_per_token_s * gen_len
    mspeckv_quant_latency_per_token_s.append(latency_per_token_s)
    mspeckv_quant_latencies.append(total_latency_s)
    mspeckv_quant_accepted_tokens.append(avg_tokens)
    
    if local_rank == 0:
        print(colored(
            f"  Sample {idx}: {total_latency_s:.4f}s total, "
            f"{(1.0/max(latency_per_token_s, 1e-9)):.2f} tok/s, "
            f"{avg_tokens:.2f} accepted",
            "green"
        ))

if local_rank == 0:
    mean_mspeckv_quant_latency = np.mean(mspeckv_quant_latencies)
    std_mspeckv_quant_latency = np.std(mspeckv_quant_latencies)
    mean_mspeckv_quant_throughput = gen_len / mean_mspeckv_quant_latency
    mean_mspeckv_quant_accepted = np.mean(mspeckv_quant_accepted_tokens)
    mean_mspeckv_quant_latency_per_token_s = float(np.mean(mspeckv_quant_latency_per_token_s)) if len(mspeckv_quant_latency_per_token_s) > 0 else None
    
    comparison_results['methods']['mspeckv_kv_quant'] = {
        'name': 'MSpecKV with KV Cache Quantization',
        'retrieval_budget': retrieval_budget,
        'mean_latency_s': float(mean_mspeckv_quant_latency),
        'std_latency_s': float(std_mspeckv_quant_latency),
        'mean_throughput_tok_per_s': float(mean_mspeckv_quant_throughput),
        'mean_accepted_tokens': float(mean_mspeckv_quant_accepted),
        'mean_latency_s_per_token': mean_mspeckv_quant_latency_per_token_s,
        'speedup_vs_baseline': None,  # filled after baseline runs
        'speedup_vs_vanilla': None,   # filled after vanilla runs
        'samples': [{'latency_s': float(lat), 'throughput': float(gen_len/lat), 'accepted_tokens': float(acc)} 
                    for lat, acc in zip(mspeckv_quant_latencies, mspeckv_quant_accepted_tokens)]
    }
    
    print(colored(f"\n[MSpecKV] Mean Latency: {mean_mspeckv_quant_latency:.4f} ± {std_mspeckv_quant_latency:.4f} s", "green"))
    print(colored(f"[MSpecKV] Mean Throughput: {mean_mspeckv_quant_throughput:.2f} tok/s", "green"))
    print(colored(f"[MSpecKV] Mean Accepted Tokens: {mean_mspeckv_quant_accepted:.2f}", "green"))

dist.barrier()
del llm_mspeckv_quant
dist.barrier()
torch.cuda.empty_cache()

# =============================================================================
# METHOD 1: Baseline (Autoregressive - No Speculation)
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "yellow"))
    print(colored("METHOD 1: BASELINE (Autoregressive - No Speculation)", "yellow"))
    print(colored("=" * 80, "yellow"))

llm_baseline = DistributedLlama(
    model_name_or_path=model_name_or_path, 
    local_rank=local_rank, 
    world_size=world_size, 
    prefill=prefill, 
    gen_len=gen_len, 
    temperature=temperature, 
    top_p=top_p, 
    flash_attn=True, 
    retrieval_budget=0,  # No retrieval
    kv_offload=True, 
    on_chip_layers=args.on_chip
)

for rank in range(world_size):
    if local_rank == rank:
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
        llm_baseline.init_parameters(hf_model=hf_model)
        del hf_model
    dist.barrier()

baseline_latencies = []
baseline_tokens = []

for idx, input_ids in enumerate(tqdm(tokenized_prompts, desc="Baseline", disable=(local_rank != 0))):
    input_ids = input_ids[:,:prefill].to(device)
    latency_per_token_ms, gen_tokens_tensor = Baseline_Dist(tokenizer, llm_baseline, input_ids, max_len=gen_len, temperature=temperature, top_p=top_p, local_rank=local_rank)
    
    # latency_per_token_ms is already a scalar (ms per token)
    # gen_tokens_tensor is the actual generated tokens, we need the count
    num_tokens = gen_len  # We know we generated gen_len tokens
    total_latency_s = (latency_per_token_ms * num_tokens) / 1000
    
    baseline_latencies.append(total_latency_s)
    baseline_tokens.append(num_tokens)
    
    if local_rank == 0:
        print(colored(f"  Sample {idx}: {total_latency_s:.4f}s, {num_tokens/total_latency_s:.2f} tok/s", "yellow"))

if local_rank == 0:
    mean_baseline_latency = np.mean(baseline_latencies)
    std_baseline_latency = np.std(baseline_latencies)
    mean_baseline_throughput = gen_len / mean_baseline_latency
    
    comparison_results['methods']['baseline'] = {
        'name': 'Autoregressive (No Speculation)',
        'mean_latency_s': float(mean_baseline_latency),
        'std_latency_s': float(std_baseline_latency),
        'mean_throughput_tok_per_s': float(mean_baseline_throughput),
        'speedup': 1.0,
        'samples': [{'latency_s': float(lat), 'throughput': float(gen_len/lat)} for lat in baseline_latencies]
    }
    
    print(colored(f"\n[Baseline] Mean Latency: {mean_baseline_latency:.4f} ± {std_baseline_latency:.4f} s", "yellow"))
    print(colored(f"[Baseline] Mean Throughput: {mean_baseline_throughput:.2f} tok/s", "yellow"))

del llm_baseline
dist.barrier()
torch.cuda.empty_cache()

# =============================================================================
# METHOD 2: Vanilla Speculative Decoding (No Retrieval Cache)
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "blue"))
    print(colored("METHOD 2: VANILLA SPECULATIVE DECODING (No Retrieval)", "blue"))
    print(colored("=" * 80, "blue"))

draft_cache_vanilla = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

llm_vanilla = DistributedLlama(
    model_name_or_path=model_name_or_path,
    local_rank=local_rank,
    world_size=world_size,
    prefill=prefill,
    gen_len=gen_len,
    temperature=temperature,
    top_p=top_p,
    flash_attn=True,
    retrieval_budget=0,  # No retrieval - vanilla spec decoding
    kv_offload=True,
    on_chip_layers=args.on_chip,
    draft=draft,
    draft_cache=draft_cache_vanilla,
    gamma=gamma
)

for rank in range(world_size):
    if local_rank == rank:
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
        llm_vanilla.init_parameters(hf_model=hf_model)
        del hf_model
    dist.barrier()

vanilla_latencies = []
vanilla_accepted_tokens = []

for idx, input_ids in enumerate(tqdm(tokenized_prompts, desc="Vanilla SpecDec", disable=(local_rank != 0))):
    input_ids = input_ids[:,:prefill].to(llm_vanilla.device)
    
    avg_tokens, latency = VanillaSpecDec_Dist(tokenizer, llm_vanilla, input_ids, gamma=gamma, max_len=gen_len, top_k=-1, top_p=top_p, temperature=temperature, verbose=False, local_rank=local_rank)
    
    # Convert to Python scalars if they're tensors
    if torch.is_tensor(avg_tokens):
        avg_tokens = avg_tokens.item()
    if torch.is_tensor(latency):
        latency = latency.item()
    
    vanilla_latencies.append(latency)
    vanilla_accepted_tokens.append(avg_tokens)
    
    if local_rank == 0:
        print(colored(f"  Sample {idx}: {latency:.4f}s, {gen_len/latency:.2f} tok/s, {avg_tokens:.2f} accepted", "blue"))

if local_rank == 0:
    mean_vanilla_latency = np.mean(vanilla_latencies)
    std_vanilla_latency = np.std(vanilla_latencies)
    mean_vanilla_throughput = gen_len / mean_vanilla_latency
    mean_vanilla_accepted = np.mean(vanilla_accepted_tokens)
    speedup_vanilla = mean_baseline_latency / mean_vanilla_latency
    
    comparison_results['methods']['vanilla_specdec'] = {
        'name': 'Vanilla Speculative Decoding',
        'mean_latency_s': float(mean_vanilla_latency),
        'std_latency_s': float(std_vanilla_latency),
        'mean_throughput_tok_per_s': float(mean_vanilla_throughput),
        'mean_accepted_tokens': float(mean_vanilla_accepted),
        'speedup': float(speedup_vanilla),
        'samples': [{'latency_s': float(lat), 'throughput': float(gen_len/lat), 'accepted_tokens': float(acc)} 
                    for lat, acc in zip(vanilla_latencies, vanilla_accepted_tokens)]
    }
    
    print(colored(f"\n[Vanilla SpecDec] Mean Latency: {mean_vanilla_latency:.4f} ± {std_vanilla_latency:.4f} s", "blue"))
    print(colored(f"[Vanilla SpecDec] Mean Throughput: {mean_vanilla_throughput:.2f} tok/s", "blue"))
    print(colored(f"[Vanilla SpecDec] Mean Accepted Tokens: {mean_vanilla_accepted:.2f}", "blue"))
    print(colored(f"[Vanilla SpecDec] Speedup vs Baseline: {speedup_vanilla:.2f}x", "blue"))

del llm_vanilla
dist.barrier()
torch.cuda.empty_cache()

# =============================================================================
# METHOD 3: MSpecKV (Speculative Decoding + Retrieval Cache)
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "green"))
    print(colored("METHOD 3: TRIFORCE (SpecDec + Retrieval Cache)", "green"))
    print(colored(f"Retrieval Budget: {retrieval_budget}", "green"))
    print(colored("=" * 80, "green"))

draft_cache_mspeckv = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

llm_mspeckv = DistributedLlama(
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
    draft_cache=draft_cache_mspeckv,
    gamma=gamma
)

for rank in range(world_size):
    if local_rank == rank:
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
        llm_mspeckv.init_parameters(hf_model=hf_model)
        del hf_model
    dist.barrier()

mspeckv_latencies = []
mspeckv_latency_per_token_s = []  # seconds/token (as returned by MSpecKV_Dist)
mspeckv_accepted_tokens = []

for idx, input_ids in enumerate(tqdm(tokenized_prompts, desc="MSpecKV", disable=(local_rank != 0))):
    input_ids = input_ids[:,:prefill].to(llm_mspeckv.device)
    
    avg_tokens, latency = MSpecKV_Dist(tokenizer, llm_mspeckv, input_ids, gamma=gamma, max_len=gen_len, top_k=-1, top_p=top_p, temperature=temperature, verbose=False, file_path=None, dataset=args.dataset)
    
    # Convert to Python scalars if they're tensors
    if torch.is_tensor(avg_tokens):
        avg_tokens = avg_tokens.item()
    if torch.is_tensor(latency):
        latency = latency.item()
    
    # NOTE: MSpecKV_Dist returns **seconds per generated token**: (time2-time1)/n.
    # Convert to total sample latency to make it comparable to Baseline/Vanilla.
    latency_per_token_s = float(latency)
    total_latency_s = latency_per_token_s * gen_len
    mspeckv_latency_per_token_s.append(latency_per_token_s)
    mspeckv_latencies.append(total_latency_s)
    mspeckv_accepted_tokens.append(avg_tokens)
    
    if local_rank == 0:
        print(colored(
            f"  Sample {idx}: {total_latency_s:.4f}s total, "
            f"{(1.0/max(latency_per_token_s, 1e-9)):.2f} tok/s, "
            f"{avg_tokens:.2f} accepted",
            "green"
        ))

if local_rank == 0:
    mean_mspeckv_latency = np.mean(mspeckv_latencies)
    std_mspeckv_latency = np.std(mspeckv_latencies)
    mean_mspeckv_throughput = gen_len / mean_mspeckv_latency
    mean_mspeckv_accepted = np.mean(mspeckv_accepted_tokens)
    speedup_mspeckv = mean_baseline_latency / mean_mspeckv_latency
    speedup_vs_vanilla = mean_vanilla_latency / mean_mspeckv_latency
    mean_mspeckv_latency_per_token_s = float(np.mean(mspeckv_latency_per_token_s)) if len(mspeckv_latency_per_token_s) > 0 else None
    
    comparison_results['methods']['mspeckv'] = {
        'name': 'MSpecKV (SpecDec + Retrieval)',
        'retrieval_budget': retrieval_budget,
        'mean_latency_s': float(mean_mspeckv_latency),
        'std_latency_s': float(std_mspeckv_latency),
        'mean_throughput_tok_per_s': float(mean_mspeckv_throughput),
        'mean_accepted_tokens': float(mean_mspeckv_accepted),
        'mean_latency_s_per_token': mean_mspeckv_latency_per_token_s,
        'speedup_vs_baseline': float(speedup_mspeckv),
        'speedup_vs_vanilla': float(speedup_vs_vanilla),
        'samples': [{'latency_s': float(lat), 'throughput': float(gen_len/lat), 'accepted_tokens': float(acc)} 
                    for lat, acc in zip(mspeckv_latencies, mspeckv_accepted_tokens)]
    }
    
    print(colored(f"\n[MSpecKV] Mean Latency: {mean_mspeckv_latency:.4f} ± {std_mspeckv_latency:.4f} s", "green"))
    print(colored(f"[MSpecKV] Mean Throughput: {mean_mspeckv_throughput:.2f} tok/s", "green"))
    print(colored(f"[MSpecKV] Mean Accepted Tokens: {mean_mspeckv_accepted:.2f}", "green"))
    print(colored(f"[MSpecKV] Speedup vs Baseline: {speedup_mspeckv:.2f}x", "green"))
    print(colored(f"[MSpecKV] Speedup vs Vanilla SpecDec: {speedup_vs_vanilla:.2f}x", "green"))

dist.barrier()
del llm_mspeckv
dist.barrier()
torch.cuda.empty_cache()


# Fill speedups now that all mean latencies are known
if local_rank == 0:
    speedup_mspeckv = mean_baseline_latency / mean_mspeckv_latency
    speedup_vs_vanilla = mean_vanilla_latency / mean_mspeckv_latency
    speedup_vs_baseline_quant = mean_baseline_latency / mean_mspeckv_quant_latency
    speedup_vs_vanilla_quant = mean_vanilla_latency / mean_mspeckv_quant_latency
    speedup_vs_mspeckv_quant = mean_mspeckv_latency / mean_mspeckv_quant_latency

    comparison_results['methods']['mspeckv']['speedup_vs_baseline'] = float(speedup_mspeckv)
    comparison_results['methods']['mspeckv']['speedup_vs_vanilla'] = float(speedup_vs_vanilla)
    comparison_results['methods']['mspeckv_kv_quant']['speedup_vs_baseline'] = float(speedup_vs_baseline_quant)
    comparison_results['methods']['mspeckv_kv_quant']['speedup_vs_vanilla'] = float(speedup_vs_vanilla_quant)
    comparison_results['methods']['mspeckv_kv_quant']['speedup_vs_mspeckv'] = float(speedup_vs_mspeckv_quant)

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("COMPARISON SUMMARY", "magenta"))
    print(colored("=" * 80, "magenta"))
    
    print(f"\n{'Method':<35} {'Latency (s)':<15} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 80)
    print(f"{'Baseline (Autoregressive)':<35} {mean_baseline_latency:>6.4f} ± {std_baseline_latency:<5.4f} {mean_baseline_throughput:>8.2f} tok/s  {'1.00x':<10}")
    print(f"{'Vanilla Speculative Decoding':<35} {mean_vanilla_latency:>6.4f} ± {std_vanilla_latency:<5.4f} {mean_vanilla_throughput:>8.2f} tok/s  {speedup_vanilla:<10.2f}x")
    print(f"{'MSpecKV (SpecDec + Retrieval)':<35} {mean_mspeckv_latency:>6.4f} ± {std_mspeckv_latency:<5.4f} {mean_mspeckv_throughput:>8.2f} tok/s  {speedup_mspeckv:<10.2f}x")
    print(f"{'MSpecKV (KV Cache Quantization)':<35} {mean_mspeckv_quant_latency:>6.4f} ± {std_mspeckv_quant_latency:<5.4f} {mean_mspeckv_quant_throughput:>8.2f} tok/s  {speedup_vs_baseline_quant:<10.2f}x {speedup_vs_mspeckv_quant:<10.2f}x")
    
    print(f"\n{'Method':<35} {'Accepted Tokens':<20}")
    print("-" * 55)
    print(f"{'Vanilla Speculative Decoding':<35} {mean_vanilla_accepted:>8.2f}")
    print(f"{'MSpecKV (SpecDec + Retrieval)':<35} {mean_mspeckv_accepted:>8.2f}")
    print(f"{'MSpecKV (KV Cache Quantization)':<35} {mean_mspeckv_quant_accepted:>8.2f}")
    
    # Save JSON
    with open(args.output_json, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(colored(f"\n✓ Detailed results saved to {args.output_json}", "cyan"))
    
    # Save CSV
    csv_file = args.output_csv
    
    # Always create new file with header if separate_files is True
    file_exists = os.path.isfile(csv_file) and not args.separate_files
    
    with open(csv_file, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'timestamp', 'model', 'prefill_length', 'gen_length', 'gamma', 'budget', 'on_chip_layers',
                'baseline_latency_s', 'baseline_throughput',
                'vanilla_latency_s', 'vanilla_throughput', 'vanilla_accepted_tokens', 'vanilla_speedup',
                'mspeckv_latency_s', 'mspeckv_throughput', 'mspeckv_accepted_tokens', 'mspeckv_speedup_vs_baseline', 'mspeckv_speedup_vs_vanilla',
                'mspeckv_quant_latency_s', 'mspeckv_quant_throughput', 'mspeckv_quant_accepted_tokens', 'mspeckv_quant_speedup_vs_baseline', 'mspeckv_quant_speedup_vs_vanilla', 'mspeckv_quant_speedup_vs_mspeckv'
            ])
        
        writer.writerow([
            comparison_results['config']['timestamp'],
            args.target,
            prefill,
            gen_len,
            gamma,
            retrieval_budget,
            args.on_chip,
            f"{mean_baseline_latency:.4f}",
            f"{mean_baseline_throughput:.2f}",
            f"{mean_vanilla_latency:.4f}",
            f"{mean_vanilla_throughput:.2f}",
            f"{mean_vanilla_accepted:.2f}",
            f"{speedup_vanilla:.2f}",
            f"{mean_mspeckv_latency:.4f}",
            f"{mean_mspeckv_throughput:.2f}",
            f"{mean_mspeckv_accepted:.2f}",
            f"{speedup_mspeckv:.2f}",
            f"{speedup_vs_vanilla:.2f}",
            f"{mean_mspeckv_quant_latency:.4f}",
            f"{mean_mspeckv_quant_throughput:.2f}",
            f"{mean_mspeckv_quant_accepted:.2f}",
            f"{speedup_vs_baseline_quant:.2f}",
            f"{speedup_vs_vanilla_quant:.2f}",
            f"{speedup_vs_mspeckv_quant:.2f}",
        ])
    
    print(colored(f"✓ Comparison appended to {csv_file}", "cyan"))
    print(colored("\n" + "=" * 80 + "\n", "magenta"))

dist.destroy_process_group()
