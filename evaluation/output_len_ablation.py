"""
This is output length ablation study for offloading with Quantized_MSpecKV.
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/output_len_ablation.py --budget 4096 --prefill 8192 --gen_len 256 --dataset demo --target llama-7B-128K --on_chip 16 --gamma 8 --resident_layers 16 --output_json auto
"""


import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import argparse
from termcolor import colored
from utils.decoding import MSpecKV_Dist
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
import gc

local_rank, world_size = distributed_init()
device = torch.device("cuda", local_rank)

def safe_barrier():
    """
    NCCL barrier can warn/hang if device mapping is ambiguous. Prefer passing device_ids when supported.
    Falls back for older PyTorch versions.
    """
    try:
        dist.barrier(device_ids=[local_rank])
    except TypeError:
        dist.barrier()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Output length ablation study for offloading with Quantized_MSpecKV')

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
    parser.add_argument('--resident_layers', type=int, default=16, help='resident layers')
    parser.add_argument('--gamma', type=int, default=8, help='speculation depth')
    parser.add_argument(
        '--gen_lens',
        type=str,
        default='256,512,768,1024,1536,2048',
        help='generation lengths'
    )
    parser.add_argument('--output_csv', type=str, default='comparison.csv', help='output CSV file (use "auto" for automatic naming)')
    parser.add_argument('--output_json', type=str, default='comparison.json', help='output JSON file (use "auto" for automatic naming)')
    parser.add_argument('--separate_files', action='store_true', help='create separate files per experiment (timestamp-based)')
    args = parser.parse_args()
    
    # Auto-generate filenames if requested
    # NOTE: Use getattr() for robustness in case older launchers/scripts pass a Namespace without output_csv.
    output_csv = getattr(args, 'output_csv', None)
    output_json = getattr(args, 'output_json', None)
    if output_csv == 'auto' or output_json == 'auto' or args.separate_files:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{args.target}_p{args.prefill}_g{args.gamma}_b{args.budget}"
        args.output_csv = f"results/output_len_ablation_{base_name}_{timestamp_str}.csv"
        args.output_json = f"results/output_len_ablation_{base_name}_{timestamp_str}.json"
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

# Generation length sweep values
try:
    gen_lens_values = [int(x.strip()) for x in args.gen_lens.split(',') if x.strip() != '']
except Exception as e:
    raise ValueError(
        f"Failed to parse --gen_lens='{args.gen_lens}'. Expected comma-separated ints."
    ) from e
if len(gen_lens_values) == 0:
    # If user passes empty string, fall back to single value from --gen_len (int).
    gen_lens_values = [args.gen_len]

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
        'gen_lens_sweep': gen_lens_values,
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
# On-chip layer ablation study for offloading with Quantized_MSpecKV
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "yellow"))
    print(colored("Output length ablation study for offloading with Quantized_MSpecKV", "yellow"))
    print(colored("=" * 80, "yellow"))

comparison_results['methods']['mspeckv_kv_quant'] = {
    'name': 'MSpecKV baseline vs MSpecKV with KV Cache Quantization',
    'retrieval_budget': retrieval_budget,
    'kv_cache_quant_bits': 8,
    'kv_fp16_tail': 0,
    'kv_resident_gpu': True,
    'sweep': {}
}

for sweep_idx, gen_len_sweep in enumerate(gen_lens_values):
    if local_rank == 0:
        print(colored("\n" + "-" * 80, "cyan"))
        print(colored(f"[Sweep {sweep_idx+1}/{len(gen_lens_values)}] gen_len={gen_len_sweep}", "cyan"))
        print(colored("-" * 80, "cyan"))

    # Run baseline and quant sequentially to avoid peak-memory OOM.
    def run_mspeckv_one_setting(*, quant: bool):
        draft_cache_run = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

        llm = DistributedLlama(
            model_name_or_path=model_name_or_path,
            local_rank=local_rank,
            world_size=world_size,
            prefill=prefill,
            gen_len=gen_len_sweep,
            temperature=temperature,
            top_p=top_p,
            flash_attn=True,
            retrieval_budget=retrieval_budget,  # With retrieval
            kv_offload=True,
            on_chip_layers=args.on_chip,
            draft=draft,
            draft_cache=draft_cache_run,
            gamma=gamma,
            kv_cache_quant=quant if quant else False,
            kv_cache_quant_bits=8,
            kv_fp16_tail=0,
            kv_resident_gpu=True,
            kv_resident_max_layers=args.resident_layers
        )

        for rank in range(world_size):
            if local_rank == rank:
                hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
                llm.init_parameters(hf_model=hf_model)
                del hf_model
            safe_barrier()

        latencies = []
        latency_per_token_s_list = []  # seconds/token (as returned by MSpecKV_Dist)
        accepted_tokens_list = []

        for idx, input_ids in enumerate(tqdm(
            tokenized_prompts,
            desc=f"MSpecKV_{'KVQuant' if quant else 'Baseline'}(gen_len={gen_len_sweep})",
            disable=(local_rank != 0)
        )):
            input_ids = input_ids[:, :prefill].to(llm.device)

            avg_tokens, latency = MSpecKV_Dist(
                tokenizer,
                llm,
                input_ids,
                gamma=gamma,
                max_len=gen_len_sweep,
                top_k=-1,
                top_p=top_p,
                temperature=temperature,
                verbose=False,
                file_path=None,
                dataset=args.dataset
            )

            if torch.is_tensor(avg_tokens):
                avg_tokens = avg_tokens.item()
            if torch.is_tensor(latency):
                latency = latency.item()

            latency_per_token_s = float(latency)
            total_latency_s = latency_per_token_s * gen_len_sweep
            latency_per_token_s_list.append(latency_per_token_s)
            latencies.append(total_latency_s)
            accepted_tokens_list.append(avg_tokens)

            if local_rank == 0:
                print(colored(
                    f"  Sample {idx}: {total_latency_s:.4f}s total, "
                    f"{(1.0/max(latency_per_token_s, 1e-9)):.2f} tok/s, "
                    f"{avg_tokens:.2f} accepted",
                    "green"
                ))

        safe_barrier()
        del llm
        safe_barrier()
        torch.cuda.empty_cache()
        gc.collect()

        return latencies, latency_per_token_s_list, accepted_tokens_list

    baseline_latencies, baseline_latency_per_token_s, baseline_accepted = run_mspeckv_one_setting(quant=False)
    quant_latencies, quant_latency_per_token_s, quant_accepted = run_mspeckv_one_setting(quant=True)

    if local_rank == 0:
        # Baseline stats
        mean_baseline_latency = float(np.mean(baseline_latencies)) if len(baseline_latencies) > 0 else None
        std_baseline_latency = float(np.std(baseline_latencies)) if len(baseline_latencies) > 0 else None
        mean_baseline_throughput = float(gen_len_sweep / mean_baseline_latency) if mean_baseline_latency and mean_baseline_latency > 0 else None
        mean_baseline_accepted = float(np.mean(baseline_accepted)) if len(baseline_accepted) > 0 else None
        mean_baseline_latency_per_token_s = float(np.mean(baseline_latency_per_token_s)) if len(baseline_latency_per_token_s) > 0 else None

        # Quant stats
        mean_quant_latency = float(np.mean(quant_latencies)) if len(quant_latencies) > 0 else None
        std_quant_latency = float(np.std(quant_latencies)) if len(quant_latencies) > 0 else None
        mean_quant_throughput = float(gen_len_sweep / mean_quant_latency) if mean_quant_latency and mean_quant_latency > 0 else None
        mean_quant_accepted = float(np.mean(quant_accepted)) if len(quant_accepted) > 0 else None
        mean_quant_latency_per_token_s = float(np.mean(quant_latency_per_token_s)) if len(quant_latency_per_token_s) > 0 else None

        speedup = (
            float(mean_quant_throughput) / float(mean_baseline_throughput)
            if mean_baseline_throughput and mean_quant_throughput
            else None
        )

        comparison_results['methods']['mspeckv_kv_quant']['sweep'][str(gen_len_sweep)] = {
            'gen_len': int(gen_len_sweep),
            'on_chip_layers': args.on_chip,
            'resident_layers': args.resident_layers,
            'baseline': {
                'mean_latency_s': mean_baseline_latency,
                'std_latency_s': std_baseline_latency,
                'mean_throughput_tok_per_s': mean_baseline_throughput,
                'mean_accepted_tokens': mean_baseline_accepted,
                'mean_latency_s_per_token': mean_baseline_latency_per_token_s,
            },
            'quant': {
                'mean_latency_s': mean_quant_latency,
                'std_latency_s': std_quant_latency,
                'mean_throughput_tok_per_s': mean_quant_throughput,
                'mean_accepted_tokens': mean_quant_accepted,
                'mean_latency_s_per_token': mean_quant_latency_per_token_s,
            },
            'speedup_vs_mspeckv_baseline': speedup,
        }

        print(colored(
            f"\n[MSpecKV][gen_len={gen_len_sweep}] "
            f"Mean Latency: {mean_baseline_latency:.4f} ± {std_baseline_latency:.4f} s",
            "green"
        ))
        print(colored(
            f"[MSpecKV][gen_len={gen_len_sweep}] "
            f"Mean Throughput: {mean_baseline_throughput:.2f} tok/s",
            "green"
        ))
        print(colored(
            f"[MSpecKV][gen_len={gen_len_sweep}] "
            f"Mean Accepted Tokens: {mean_baseline_accepted:.2f}",
            "green"
        ))

        print(colored(
            f"\n[MSpecKV][gen_len={gen_len_sweep}] "
            f"Mean Latency: {mean_quant_latency:.4f} ± {std_quant_latency:.4f} s",
            "green"
        ))
        print(colored(
            f"[MSpecKV][gen_len={gen_len_sweep}] "
            f"Mean Throughput: {mean_quant_throughput:.2f} tok/s",
            "green"
        ))
        print(colored(
            f"[MSpecKV][gen_len={gen_len_sweep}] "
            f"Mean Accepted Tokens: {mean_quant_accepted:.2f}",
            "green"
        ))
        print(colored(
            f"[MSpecKV][gen_len={gen_len_sweep}] "
            f"Speedup vs MSpecKV Baseline: {speedup:.2f}x" if speedup is not None else
            f"[MSpecKV][gen_len={gen_len_sweep}] Speedup vs MSpecKV Baseline: N/A",
            "green"
        ))

# Save sweep results
if local_rank == 0:
    output_csv = getattr(args, 'output_csv', None)
    output_json = getattr(args, 'output_json', None)

    # Ensure output dirs exist
    if output_csv is not None:
        out_dir = os.path.dirname(output_csv)
        if out_dir != '':
            os.makedirs(out_dir, exist_ok=True)
    if output_json is not None:
        out_dir = os.path.dirname(output_json)
        if out_dir != '':
            os.makedirs(out_dir, exist_ok=True)

    # CSV summary
    if output_csv is not None:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'gen_len',
                'on_chip_layers',
                'resident_layers',
                'baseline_mean_latency_s',
                'baseline_std_latency_s',
                'baseline_mean_throughput_tok_per_s',
                'baseline_mean_accepted_tokens',
                'baseline_mean_latency_s_per_token',
                'quant_mean_latency_s',
                'quant_std_latency_s',
                'quant_mean_throughput_tok_per_s',
                'quant_mean_accepted_tokens',
                'quant_mean_latency_s_per_token',
                'speedup_vs_mspeckv_baseline',
            ])
            sweep_dict = comparison_results['methods']['mspeckv_kv_quant']['sweep']
            for gen_len in gen_lens_values:
                row = sweep_dict.get(str(gen_len), {})
                baseline_row = row.get('baseline', {}) if isinstance(row, dict) else {}
                quant_row = row.get('quant', {}) if isinstance(row, dict) else {}
                writer.writerow([
                    gen_len,
                    args.on_chip,
                    args.resident_layers,
                    baseline_row.get('mean_latency_s'),
                    baseline_row.get('std_latency_s'),
                    baseline_row.get('mean_throughput_tok_per_s'),
                    baseline_row.get('mean_accepted_tokens'),
                    baseline_row.get('mean_latency_s_per_token'),
                    quant_row.get('mean_latency_s'),
                    quant_row.get('std_latency_s'),
                    quant_row.get('mean_throughput_tok_per_s'),
                    quant_row.get('mean_accepted_tokens'),
                    quant_row.get('mean_latency_s_per_token'),
                    row.get('speedup_vs_mspeckv_baseline') if isinstance(row, dict) else None,
                ])
        print(colored(f"\nSaved CSV sweep summary to: {output_csv}", "yellow"))

    # JSON full results (includes per-sample arrays)
    if output_json is not None:
        with open(output_json, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(colored(f"Saved JSON sweep results to: {output_json}", "yellow"))


dist.destroy_process_group()
