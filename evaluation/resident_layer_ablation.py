"""
This is resident layer ablation study for offloading with Quantized_TriForce.
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/resident_layer_ablation.py --budget 4096 --prefill 8192 --dataset demo --target llama-7B-128K --on_chip 16 --gamma 8 --output_json auto
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
from utils.decoding import Baseline_Dist, VanillaSpecDec_Dist, TriForce_Dist
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
    parser = argparse.ArgumentParser(description='Resident layer ablation study for offloading with Quantized_TriForce')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--prefill', type=int, default=8192, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset')
    parser.add_argument('--on_chip', type=int, default=4, help='on chip layers')
    parser.add_argument('--budget', type=int, default=4096, help='retrieval budget for TriForce')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--gamma', type=int, default=8, help='speculation depth')
    parser.add_argument('--resident_layers', type=int, default=8, help='resident layers (used if --resident_layers_values is empty)')
    parser.add_argument(
        '--resident_layers_values',
        type=str,
        default='4,8,10,12,16,18,20,24',
        help='comma-separated resident layers sweep values, e.g. "4,8,12,16"'
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
        args.output_csv = f"A2/resident_layer_ablation_{base_name}_{timestamp_str}.csv"
        args.output_json = f"A2/resident_layer_ablation_{base_name}_{timestamp_str}.json"
        # Create results directory if needed
        os.makedirs('A2', exist_ok=True)
    
    return args

args = parse_arguments()
torch.manual_seed(args.seed)
prefill = args.prefill
gen_len = args.gen_len
temperature = args.temp
top_p = args.top_p
retrieval_budget = args.budget
gamma = args.gamma

# Resident-layer sweep values
try:
    resident_layers_values = [int(x.strip()) for x in args.resident_layers_values.split(',') if x.strip() != '']
except Exception as e:
    raise ValueError(
        f"Failed to parse --resident_layers_values='{args.resident_layers_values}'. Expected comma-separated ints."
    ) from e
if len(resident_layers_values) == 0:
    resident_layers_values = [args.resident_layers]

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
        'resident_layers_sweep': resident_layers_values,
        'num_samples': len(tokenized_prompts),
        'timestamp': datetime.now().isoformat()
    },
    'methods': {}
}

# =============================================================================
# Draft model setup (used by Vanilla SpecDec + TriForce)
# =============================================================================
draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map=device)
draft = draft.eval()
draft_cache_budget = 256
recent_size = draft_cache_budget - 16 - gamma

# =============================================================================
# Resident layer ablation study for offloading with Quantized_TriForce
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "yellow"))
    print(colored("Resident layer ablation study for offloading with Quantized_TriForce", "yellow"))
    print(colored("=" * 80, "yellow"))

comparison_results['methods']['triforce_kv_quant'] = {
    'name': 'TriForce baseline vs TriForce with KV Cache Quantization',
    'retrieval_budget': retrieval_budget,
    'kv_cache_quant_bits': 8,
    'kv_fp16_tail': 0,
    'kv_resident_gpu': True,
    'sweep': {}
}

for sweep_idx, resident_layers in enumerate(resident_layers_values):
    if local_rank == 0:
        print(colored("\n" + "-" * 80, "cyan"))
        print(colored(f"[Sweep {sweep_idx+1}/{len(resident_layers_values)}] resident_layers={resident_layers}", "cyan"))
        print(colored("-" * 80, "cyan"))

    # Run baseline and quant sequentially to avoid peak-memory OOM.
    def run_triforce_one_setting(*, quant: bool):
        draft_cache_run = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

        llm = DistributedLlama(
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
            draft_cache=draft_cache_run,
            gamma=gamma,
            kv_cache_quant=quant if quant else False,
            kv_cache_quant_bits=8,
            kv_fp16_tail=0,
            kv_resident_gpu=True,
            kv_resident_max_layers=resident_layers
        )

        for rank in range(world_size):
            if local_rank == rank:
                hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
                llm.init_parameters(hf_model=hf_model)
                del hf_model
            safe_barrier()

        latencies = []
        latency_per_token_s_list = []  # seconds/token (as returned by TriForce_Dist)
        accepted_tokens_list = []

        for idx, input_ids in enumerate(tqdm(
            tokenized_prompts,
            desc=f"TriForce_{'KVQuant' if quant else 'Baseline'}(res_layers={resident_layers})",
            disable=(local_rank != 0)
        )):
            input_ids = input_ids[:, :prefill].to(llm.device)

            avg_tokens, latency = TriForce_Dist(
                tokenizer,
                llm,
                input_ids,
                gamma=gamma,
                max_len=gen_len,
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
            total_latency_s = latency_per_token_s * gen_len
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

    baseline_latencies, baseline_latency_per_token_s, baseline_accepted = run_triforce_one_setting(quant=False)
    quant_latencies, quant_latency_per_token_s, quant_accepted = run_triforce_one_setting(quant=True)

    if local_rank == 0:
        # Baseline stats
        mean_baseline_latency = float(np.mean(baseline_latencies)) if len(baseline_latencies) > 0 else None
        std_baseline_latency = float(np.std(baseline_latencies)) if len(baseline_latencies) > 0 else None
        mean_baseline_throughput = float(gen_len / mean_baseline_latency) if mean_baseline_latency and mean_baseline_latency > 0 else None
        mean_baseline_accepted = float(np.mean(baseline_accepted)) if len(baseline_accepted) > 0 else None
        mean_baseline_latency_per_token_s = float(np.mean(baseline_latency_per_token_s)) if len(baseline_latency_per_token_s) > 0 else None

        # Quant stats
        mean_quant_latency = float(np.mean(quant_latencies)) if len(quant_latencies) > 0 else None
        std_quant_latency = float(np.std(quant_latencies)) if len(quant_latencies) > 0 else None
        mean_quant_throughput = float(gen_len / mean_quant_latency) if mean_quant_latency and mean_quant_latency > 0 else None
        mean_quant_accepted = float(np.mean(quant_accepted)) if len(quant_accepted) > 0 else None
        mean_quant_latency_per_token_s = float(np.mean(quant_latency_per_token_s)) if len(quant_latency_per_token_s) > 0 else None

        speedup = (
            float(mean_quant_throughput) / float(mean_baseline_throughput)
            if mean_baseline_throughput and mean_quant_throughput
            else None
        )

        comparison_results['methods']['triforce_kv_quant']['sweep'][str(resident_layers)] = {
            'resident_layers': int(resident_layers),
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
            'speedup_vs_triforce_baseline': speedup,
        }

        print(colored(
            f"\n[TriForce][resident_layers={resident_layers}] "
            f"Mean Latency: {mean_baseline_latency:.4f} ± {std_baseline_latency:.4f} s",
            "green"
        ))
        print(colored(
            f"[TriForce][resident_layers={resident_layers}] "
            f"Mean Throughput: {mean_baseline_throughput:.2f} tok/s",
            "green"
        ))
        print(colored(
            f"[TriForce][resident_layers={resident_layers}] "
            f"Mean Accepted Tokens: {mean_baseline_accepted:.2f}",
            "green"
        ))

        print(colored(
            f"\n[MSpecKV][resident_layers={resident_layers}] "
            f"Mean Latency: {mean_quant_latency:.4f} ± {std_quant_latency:.4f} s",
            "green"
        ))
        print(colored(
            f"[MSpecKV][resident_layers={resident_layers}] "
            f"Mean Throughput: {mean_quant_throughput:.2f} tok/s",
            "green"
        ))
        print(colored(
            f"[MSpecKV][resident_layers={resident_layers}] "
            f"Mean Accepted Tokens: {mean_quant_accepted:.2f}",
            "green"
        ))
        print(colored(
            f"[MSpecKV][resident_layers={resident_layers}] "
            f"Speedup vs TriForce Baseline: {speedup:.2f}x" if speedup is not None else
            f"[MSpecKV][resident_layers={resident_layers}] Speedup vs TriForce Baseline: N/A",
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
                'speedup_vs_triforce_baseline',
            ])
            sweep_dict = comparison_results['methods']['triforce_kv_quant']['sweep']
            for rl in resident_layers_values:
                row = sweep_dict.get(str(rl), {})
                baseline_row = row.get('baseline', {}) if isinstance(row, dict) else {}
                quant_row = row.get('quant', {}) if isinstance(row, dict) else {}
                writer.writerow([
                    rl,
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
                    row.get('speedup_vs_triforce_baseline') if isinstance(row, dict) else None,
                ])
        print(colored(f"\nSaved CSV sweep summary to: {output_csv}", "yellow"))

    # JSON full results (includes per-sample arrays)
    if output_json is not None:
        with open(output_json, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(colored(f"Saved JSON sweep results to: {output_json}", "yellow"))


dist.destroy_process_group()
