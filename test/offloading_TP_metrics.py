# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 test/offloading_TP_metrics.py --budget 4096 --prefill 8192 --dataset demo --target llama-7B-128K --on_chip 4 --gamma 8 --output_csv results.csv

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import argparse
from termcolor import colored
from utils.decoding import Baseline_Dist, MSpecKV_Dist
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
    parser = argparse.ArgumentParser(description='proj with Metrics Logging')

    parser.add_argument('--target', type=str, default='lwm-128K', help='target model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--prefill', type=int, default=130048, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset')
    parser.add_argument('--on_chip', type=int, default=0, help='on chip layers')
    parser.add_argument('--budget', type=int,  default=12288)
    parser.add_argument('--baseline', action='store_true', help='baseline')
    parser.add_argument('--compare', action='store_true', help='run baseline + MSpecKV + Quantized_MSpecKV')
    parser.add_argument('--resident_layers', type=int, default=16, help='max resident layers for quantized KV')
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--gamma', type=int, default=6)
    parser.add_argument('--output_csv', type=str, default='results.csv', help='output CSV file')
    parser.add_argument('--output_json', type=str, default='results.json', help='output JSON file')
    args = parser.parse_args()
    
    return args

args = parse_arguments()
torch.manual_seed(args.seed)
prefill = args.prefill
gen_len = args.gen_len
temperature = args.temp
top_p = args.top_p
retrieval_budget = args.budget

# Store experiment configuration
config = {
    'model': args.target,
    'prefill_length': prefill,
    'gen_length': gen_len,
    'temperature': temperature,
    'top_p': top_p,
    'budget': retrieval_budget,
    'on_chip_layers': args.on_chip,
    'gamma': args.gamma,
    'world_size': world_size,
    'dataset': args.dataset,
    'timestamp': datetime.now().isoformat()
}

####### model setup #######

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
input_ids = tokenized_prompts[0][:,:prefill].to(device)

# Results storage
results = {
    'config': config,
    'baseline': {},
    '15779proj': {
        'samples': [],
        'overall': {}
    },
    'quantized_mspeckv': {
        'samples': [],
        'overall': {}
    },
}

if args.baseline:
    if local_rank == 0:
        print(colored("=" * 80, "yellow"))
        print(colored("Running Baseline (Autoregressive)", "yellow"))
        print(colored("=" * 80, "yellow"))
    
    llm = DistributedLlama(model_name_or_path=model_name_or_path, local_rank=local_rank, world_size=world_size, prefill=prefill, gen_len=gen_len, temperature=temperature, top_p=top_p, flash_attn=True, retrieval_budget=0, kv_offload=True, on_chip_layers=args.on_chip)
    for rank in range(world_size):
        if local_rank == rank:
            hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
            llm.init_parameters(hf_model=hf_model)
            del hf_model
        dist.barrier()
    
    baseline_latency, gen_tokens = Baseline_Dist(tokenizer, llm, input_ids, max_len=gen_len, temperature=temperature, top_p=top_p, local_rank=local_rank)
    baseline_latency = baseline_latency/1000
    
    if local_rank == 0:
        results['baseline'] = {
            'latency_s': baseline_latency,
            'tokens_per_second': gen_tokens / baseline_latency,
            'total_tokens': gen_tokens
        }
        print(colored(f"\n[Baseline] Latency: {baseline_latency:.4f} s", "red"))
        print(colored(f"[Baseline] Tokens/s: {gen_tokens/baseline_latency:.2f}", "red"))
    dist.barrier()

else:
    if local_rank == 0:
        print(colored("=" * 80, "green"))
        print(colored("Running MSpecKV", "green"))
        print(colored("=" * 80, "green"))
    
    gamma = int(args.gamma)
    draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map=device)
    draft = draft.eval()
    draft_cache_budget = 256
    recent_size = draft_cache_budget - 16 - gamma
    draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

    llm = DistributedLlama(model_name_or_path=model_name_or_path, local_rank=local_rank, world_size=world_size, prefill=prefill, gen_len=gen_len, temperature=temperature, top_p=top_p, flash_attn=True, retrieval_budget=retrieval_budget, kv_offload=True, on_chip_layers=args.on_chip, draft=draft, draft_cache=draft_cache, gamma=gamma)
    for rank in range(world_size):
        if local_rank == rank:
            hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
            llm.init_parameters(hf_model=hf_model)
            del hf_model
        dist.barrier()

    all_avg_tokens = []
    all_latency = []

    for idx, input_ids in enumerate(tqdm(tokenized_prompts, desc="15779 Test", disable=(local_rank != 0))):
        input_ids = input_ids[:,:args.prefill].to(llm.device)

        avg_tokens, latency = MSpecKV_Dist(tokenizer, llm, input_ids, gamma=gamma, max_len=gen_len, top_k=-1, top_p=top_p, temperature=temperature, verbose=False, file_path=None, dataset=args.dataset)
        all_avg_tokens.append(avg_tokens)
        all_latency.append(latency)
        
        if local_rank == 0:
            sample_result = {
                'sample_idx': idx,
                'latency_s': latency,
                'avg_accepted_tokens': avg_tokens,
                'tokens_per_second': gen_len / latency,
                'speedup_vs_autoregressive': avg_tokens  # acceptance rate approximation
            }
            results['15779proj']['samples'].append(sample_result)
            
            print(colored(f"\n[Sample {idx}] Latency: {latency:.4f} s", "red"))
            print(colored(f"[Sample {idx}] Avg Accepted Tokens: {avg_tokens:.2f}", "red"))
            print(colored(f"[Sample {idx}] Tokens/s: {gen_len/latency:.2f}", "red"))
    
    if local_rank == 0:
        mean_latency = np.array(all_latency).mean()
        mean_tokens = np.array(all_avg_tokens).mean()
        std_latency = np.array(all_latency).std()
        std_tokens = np.array(all_avg_tokens).std()
        
        results['15779proj']['overall'] = {
            'mean_latency_s': float(mean_latency),
            'std_latency_s': float(std_latency),
            'mean_avg_accepted_tokens': float(mean_tokens),
            'std_avg_accepted_tokens': float(std_tokens),
            'mean_tokens_per_second': float(gen_len / mean_latency),
            'total_samples': len(all_latency)
        }
        
        print(colored("\n" + "=" * 80, "green"))
        print(colored("OVERALL RESULTS", "green"))
        print(colored("=" * 80, "green"))
        print(f"[Overall] Mean Latency: {mean_latency:.4f} ± {std_latency:.4f} s")
        print(f"[Overall] Mean Accepted Tokens: {mean_tokens:.2f} ± {std_tokens:.2f}")
        print(f"[Overall] Mean Tokens/s: {gen_len/mean_latency:.2f}")
        
        # Save JSON results
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(colored(f"\n✓ Results saved to {args.output_json}", "cyan"))
        
        # Save CSV results
        csv_file = args.output_csv
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow([
                    'timestamp', 'model', 'world_size', 'on_chip_layers', 'prefill_length', 
                    'gen_length', 'budget', 'gamma', 'temperature', 'top_p', 'dataset',
                    'mean_latency_s', 'std_latency_s', 'mean_accepted_tokens', 'std_accepted_tokens',
                    'mean_tokens_per_s', 'num_samples'
                ])
            
            # Write data row
            writer.writerow([
                config['timestamp'],
                config['model'],
                config['world_size'],
                config['on_chip_layers'],
                config['prefill_length'],
                config['gen_length'],
                config['budget'],
                config['gamma'],
                config['temperature'],
                config['top_p'],
                config['dataset'],
                f"{mean_latency:.4f}",
                f"{std_latency:.4f}",
                f"{mean_tokens:.2f}",
                f"{std_tokens:.2f}",
                f"{gen_len/mean_latency:.2f}",
                len(all_latency)
            ])
        
        print(colored(f"✓ Results appended to {csv_file}", "cyan"))
        print(colored("\n" + "=" * 80, "green"))

    # destory the distributed process
    dist.destroy_process_group()

if args.compare:
    dist.destroy_process_group()
