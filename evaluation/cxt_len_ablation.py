"""
This is context length ablation study for offloading with Quantized_TriForce.
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/cxt_len_ablation.py --budget 4096 --prefill 8192 --dataset demo --target llama-7B-128K --on_chip 16 --gamma 8 --resident_layers 16 --output_json auto
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
    parser = argparse.ArgumentParser(description='Context length (prefill) ablation study for offloading with Quantized_TriForce')

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
    parser.add_argument('--resident_layers', type=int, default=16, help='resident layers')
    parser.add_argument('--gamma', type=int, default=8, help='speculation depth')
    parser.add_argument(
        '--prefill_lengths',
        type=str,
        default='8192,16384,24576,32768',
        help='prefill lengths'
    )
    parser.add_argument('--output_csv', type=str, default='comparison.csv', help='output CSV file (use "auto" for automatic naming)')
    parser.add_argument('--output_json', type=str, default='comparison.json', help='output JSON file (use "auto" for automatic naming)')
    parser.add_argument('--separate_files', action='store_true', help='create separate files per experiment (timestamp-based)')
    parser.add_argument(
        '--input_json',
        type=str,
        default=None,
        help='existing JSON results to load + update (e.g., reuse prior TriForce/MSpecKV results)'
    )
    parser.add_argument(
        '--run_methods',
        type=str,
        default='baseline,vanilla,triforce,kvquant',
        help='comma-separated subset of: baseline,vanilla,triforce,kvquant (e.g., "baseline,vanilla")'
    )
    args = parser.parse_args()
    
    # Auto-generate filenames if requested
    # NOTE: Use getattr() for robustness in case older launchers/scripts pass a Namespace without output_csv.
    output_csv = getattr(args, 'output_csv', None)
    output_json = getattr(args, 'output_json', None)
    if output_csv == 'auto' or output_json == 'auto' or args.separate_files:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{args.target}_p{args.prefill}_g{args.gamma}_b{args.budget}"
        args.output_csv = f"A1/cxt_len_ablation_{base_name}_{timestamp_str}.csv"
        args.output_json = f"A1/cxt_len_ablation_{base_name}_{timestamp_str}.json"
        # Create results directory if needed
        os.makedirs('A1', exist_ok=True)
    else:
        # If user is updating an existing JSON and didn't override output paths, overwrite the input JSON by default.
        if args.input_json is not None and args.output_json == 'comparison.json':
            args.output_json = args.input_json
        if args.input_json is not None and args.output_csv == 'comparison.csv':
            base, _ = os.path.splitext(args.input_json)
            args.output_csv = base + '.csv'
    
    return args

args = parse_arguments()
torch.manual_seed(args.seed)
prefill = args.prefill
gen_len = args.gen_len
temperature = args.temp
top_p = args.top_p
retrieval_budget = args.budget
gamma = args.gamma

# Prefill length sweep values
try:
    prefill_lengths_values = [int(x.strip()) for x in args.prefill_lengths.split(',') if x.strip() != '']
except Exception as e:
    raise ValueError(
        f"Failed to parse --prefill_lengths='{args.prefill_lengths}'. Expected comma-separated ints."
    ) from e
if len(prefill_lengths_values) == 0:
    # If user passes empty string, fall back to single value from --prefill (int).
    prefill_lengths_values = [args.prefill]

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
# Ensure dataset prompts are long enough for the largest prefill length we plan to test.
required_datalen = max(32768, max(prefill_lengths_values) if len(prefill_lengths_values) > 0 else 32768)
tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=required_datalen)

def _parse_run_methods(run_methods_str: str):
    allowed = {'baseline', 'vanilla', 'triforce', 'kvquant'}
    methods = [m.strip().lower() for m in (run_methods_str or '').split(',') if m.strip() != '']
    methods_set = set(methods)
    unknown = sorted(list(methods_set - allowed))
    if len(unknown) > 0:
        raise ValueError(f"Unknown entries in --run_methods='{run_methods_str}': {unknown}. Allowed: {sorted(list(allowed))}")
    if len(methods_set) == 0:
        raise ValueError("--run_methods resolved to empty set; provide at least one of: baseline,vanilla,triforce,kvquant")
    return methods_set

run_methods = _parse_run_methods(getattr(args, 'run_methods', 'baseline,vanilla,triforce,kvquant'))
run_baseline = 'baseline' in run_methods
run_vanilla = 'vanilla' in run_methods
run_triforce = 'triforce' in run_methods
run_kvquant = 'kvquant' in run_methods


def _upgrade_old_triforce_kv_quant_format(results: dict) -> dict:
    """
    Backward-compat: older cxt_len_ablation.json stored TriForce baseline + KV-quant under:
      methods['triforce_kv_quant']['sweep'][p] = {'baseline': {...}, 'quant': {...}, ...}
    Newer format stores:
      methods['triforce']['sweep'][p] (flat)
      methods['triforce_kv_quant']['sweep'][p] (flat)
    This upgrades in-place when possible.
    """
    if not isinstance(results, dict):
        return results
    methods = results.get('methods', {})
    if not isinstance(methods, dict):
        return results
    tkv = methods.get('triforce_kv_quant')
    if not isinstance(tkv, dict):
        return results
    sweep = tkv.get('sweep')
    if not isinstance(sweep, dict) or len(sweep) == 0:
        return results
    # Detect old nested format
    any_old = False
    for _, entry in sweep.items():
        if isinstance(entry, dict) and ('baseline' in entry or 'quant' in entry):
            any_old = True
            break
    if not any_old:
        return results

    methods.setdefault('triforce', {'name': 'TriForce (SpecDec + Retrieval)', 'sweep': {}})
    if not isinstance(methods['triforce'], dict):
        methods['triforce'] = {'name': 'TriForce (SpecDec + Retrieval)', 'sweep': {}}
    methods['triforce'].setdefault('sweep', {})
    if not isinstance(methods['triforce']['sweep'], dict):
        methods['triforce']['sweep'] = {}

    # Upgrade each sweep entry
    for p_str, entry in list(sweep.items()):
        if not isinstance(entry, dict):
            continue
        baseline_entry = entry.get('baseline', {})
        quant_entry = entry.get('quant', {})
        if isinstance(baseline_entry, dict) and len(baseline_entry) > 0:
            methods['triforce']['sweep'][p_str] = {
                'prefill_length': int(entry.get('prefill_length', int(p_str))),
                'resident_layers': int(entry.get('resident_layers', entry.get('kv_resident_max_layers', 0) or 0)) or None,
                'mean_latency_s': baseline_entry.get('mean_latency_s'),
                'std_latency_s': baseline_entry.get('std_latency_s'),
                'mean_throughput_tok_per_s': baseline_entry.get('mean_throughput_tok_per_s'),
                'mean_accepted_tokens': baseline_entry.get('mean_accepted_tokens'),
                'mean_latency_s_per_token': baseline_entry.get('mean_latency_s_per_token'),
            }
        if isinstance(quant_entry, dict) and len(quant_entry) > 0:
            # Flatten into triforce_kv_quant sweep entry
            sweep[p_str] = {
                'prefill_length': int(entry.get('prefill_length', int(p_str))),
                'resident_layers': int(entry.get('resident_layers', entry.get('kv_resident_max_layers', 0) or 0)) or None,
                'mean_latency_s': quant_entry.get('mean_latency_s'),
                'std_latency_s': quant_entry.get('std_latency_s'),
                'mean_throughput_tok_per_s': quant_entry.get('mean_throughput_tok_per_s'),
                'mean_accepted_tokens': quant_entry.get('mean_accepted_tokens'),
                'mean_latency_s_per_token': quant_entry.get('mean_latency_s_per_token'),
            }

    return results


def _init_new_results() -> dict:
    return {
        'config': {
            'model': args.target,
            'prefill_length': prefill,
            'prefill_lengths_sweep': prefill_lengths_values,
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


# Store all results (optionally load + merge)
if getattr(args, 'input_json', None) is not None and os.path.isfile(args.input_json):
    with open(args.input_json, 'r') as f:
        comparison_results = json.load(f)
    comparison_results = _upgrade_old_triforce_kv_quant_format(comparison_results)
    # Keep config up-to-date for this run (preserve any extra keys from input JSON)
    comparison_results.setdefault('config', {})
    comparison_results['config'].update({
        'model': args.target,
        'prefill_length': prefill,
        'prefill_lengths_sweep': prefill_lengths_values,
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
    })
    comparison_results.setdefault('methods', {})
else:
    comparison_results = _init_new_results()

# =============================================================================
# Draft model setup (used by Vanilla SpecDec + TriForce)
# =============================================================================
draft = None
draft_cache_budget = 256
recent_size = None
if run_vanilla or run_triforce or run_kvquant:
    draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map=device)
    draft = draft.eval()
    recent_size = draft_cache_budget - 16 - gamma

# =============================================================================
# Context length (prefill) ablation study for offloading with Quantized_TriForce
# =============================================================================
if local_rank == 0:
    print(colored("\n" + "=" * 80, "yellow"))
    print(colored("Context length (prefill) ablation study for offloading with Quantized_TriForce", "yellow"))
    print(colored("=" * 80, "yellow"))

comparison_results.setdefault('methods', {})
comparison_results['methods'].setdefault('baseline', {'name': 'Autoregressive (No Speculation)', 'sweep': {}})
comparison_results['methods'].setdefault('vanilla_specdec', {'name': 'Vanilla Speculative Decoding (No Retrieval)', 'gamma': gamma, 'sweep': {}})
comparison_results['methods'].setdefault('triforce', {'name': 'TriForce (SpecDec + Retrieval)', 'retrieval_budget': retrieval_budget, 'gamma': gamma, 'sweep': {}})
comparison_results['methods'].setdefault(
    'triforce_kv_quant',
    {
        'name': 'TriForce with KV Cache Quantization',
        'retrieval_budget': retrieval_budget,
        'kv_cache_quant_bits': 8,
        'kv_fp16_tail': 0,
        'kv_resident_gpu': True,
        'gamma': gamma,
        'sweep': {}
    }
)
# Ensure sweep dicts exist even when loading from older JSONs
for _k in ['baseline', 'vanilla_specdec', 'triforce', 'triforce_kv_quant']:
    if not isinstance(comparison_results['methods'].get(_k), dict):
        comparison_results['methods'][_k] = {'name': _k, 'sweep': {}}
    comparison_results['methods'][_k].setdefault('sweep', {})
    if not isinstance(comparison_results['methods'][_k]['sweep'], dict):
        comparison_results['methods'][_k]['sweep'] = {}

for sweep_idx, prefill_length in enumerate(prefill_lengths_values):
    if local_rank == 0:
        print(colored("\n" + "-" * 80, "cyan"))
        print(colored(f"[Sweep {sweep_idx+1}/{len(prefill_lengths_values)}] prefill_length={prefill_length}", "cyan"))
        print(colored("-" * 80, "cyan"))
    
    if prefill_length > 32768:
        resident_layers = 8
    else:
        resident_layers = args.resident_layers
    # To avoid OOM at longer contexts, run methods sequentially (freeing GPU memory in between).

    def run_baseline_one_setting():
        llm = DistributedLlama(
            model_name_or_path=model_name_or_path,
            local_rank=local_rank,
            world_size=world_size,
            prefill=prefill_length,
            gen_len=gen_len,
            temperature=temperature,
            top_p=top_p,
            flash_attn=True,
            retrieval_budget=0,  # No retrieval
            kv_offload=True,
            on_chip_layers=args.on_chip,
        )

        for rank in range(world_size):
            if local_rank == rank:
                hf_model = LlamaForCausalLM.from_pretrained(
                    model_name_or_path, torch_dtype=torch.float16, device_map='cpu'
                )
                llm.init_parameters(hf_model=hf_model)
                del hf_model
            safe_barrier()

        latencies_s = []
        latency_per_token_ms_list = []

        for idx, input_ids in enumerate(tqdm(
            tokenized_prompts,
            desc=f"Baseline(p{prefill_length})",
            disable=(local_rank != 0)
        )):
            input_ids = input_ids[:, :prefill_length].to(device)
            latency_per_token_ms, gen_tokens_tensor = Baseline_Dist(
                tokenizer,
                llm,
                input_ids,
                max_len=gen_len,
                temperature=temperature,
                top_p=top_p,
                local_rank=local_rank
            )
            if torch.is_tensor(latency_per_token_ms):
                latency_per_token_ms = latency_per_token_ms.item()
            # gen_tokens_tensor is not used for length; we know gen_len tokens are generated.
            total_latency_s = (float(latency_per_token_ms) * float(gen_len)) / 1000.0
            latencies_s.append(total_latency_s)
            latency_per_token_ms_list.append(float(latency_per_token_ms))

        safe_barrier()
        del llm
        safe_barrier()
        torch.cuda.empty_cache()
        gc.collect()
        return latencies_s, latency_per_token_ms_list

    def run_vanilla_one_setting():
        draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)
        llm = DistributedLlama(
            model_name_or_path=model_name_or_path,
            local_rank=local_rank,
            world_size=world_size,
            prefill=prefill_length,
            gen_len=gen_len,
            temperature=temperature,
            top_p=top_p,
            flash_attn=True,
            retrieval_budget=0,  # No retrieval
            kv_offload=True,
            on_chip_layers=args.on_chip,
            draft=draft,
            draft_cache=draft_cache,
            gamma=gamma,
        )

        for rank in range(world_size):
            if local_rank == rank:
                hf_model = LlamaForCausalLM.from_pretrained(
                    model_name_or_path, torch_dtype=torch.float16, device_map='cpu'
                )
                llm.init_parameters(hf_model=hf_model)
                del hf_model
            safe_barrier()

        latencies_s = []
        accepted_tokens_list = []

        for idx, input_ids in enumerate(tqdm(
            tokenized_prompts,
            desc=f"VanillaSpecDec(p{prefill_length})",
            disable=(local_rank != 0)
        )):
            input_ids = input_ids[:, :prefill_length].to(llm.device)
            avg_tokens, latency_s = VanillaSpecDec_Dist(
                tokenizer,
                llm,
                input_ids,
                gamma=gamma,
                max_len=gen_len,
                top_k=-1,
                top_p=top_p,
                temperature=temperature,
                verbose=False,
                local_rank=local_rank
            )
            if torch.is_tensor(avg_tokens):
                avg_tokens = avg_tokens.item()
            if torch.is_tensor(latency_s):
                latency_s = latency_s.item()
            latencies_s.append(float(latency_s))
            accepted_tokens_list.append(float(avg_tokens))

        safe_barrier()
        del llm
        safe_barrier()
        torch.cuda.empty_cache()
        gc.collect()
        return latencies_s, accepted_tokens_list

    def run_triforce_one_setting(*, quant: bool):
        # Fresh cache per run (avoids cross-run state)
        draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)
        llm = DistributedLlama(
            model_name_or_path=model_name_or_path,
            local_rank=local_rank,
            world_size=world_size,
            prefill=prefill_length,
            gen_len=gen_len,
            temperature=temperature,
            top_p=top_p,
            flash_attn=True,
            retrieval_budget=retrieval_budget,
            kv_offload=True,
            on_chip_layers=args.on_chip,
            draft=draft,
            draft_cache=draft_cache,
            gamma=gamma,
            kv_cache_quant=quant if quant else False,
            kv_cache_quant_bits=8,
            kv_fp16_tail=0,
            kv_resident_gpu=True,
            kv_resident_max_layers=resident_layers,
        )

        for rank in range(world_size):
            if local_rank == rank:
                hf_model = LlamaForCausalLM.from_pretrained(
                    model_name_or_path, torch_dtype=torch.float16, device_map='cpu'
                )
                llm.init_parameters(hf_model=hf_model)
                del hf_model
            safe_barrier()

        latencies = []
        latency_per_token_s_list = []
        accepted_tokens_list = []

        for idx, input_ids in enumerate(tqdm(
            tokenized_prompts,
            desc=f"TriForce_{'KVQuant' if quant else 'Baseline'}(prefill_length={prefill_length})",
            disable=(local_rank != 0)
        )):
            input_ids = input_ids[:, :prefill_length].to(llm.device)

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

        safe_barrier()
        del llm
        safe_barrier()
        torch.cuda.empty_cache()
        gc.collect()

        return latencies, latency_per_token_s_list, accepted_tokens_list

    # ----------------------------
    # Run selected methods (can skip TriForce/MSpecKV and merge with existing JSON via --input_json)
    # ----------------------------
    baseline_latencies_s, baseline_latency_per_token_ms = (None, None)
    vanilla_latencies_s, vanilla_accepted_tokens = (None, None)
    triforce_baseline_latencies, triforce_baseline_latency_per_token_s, triforce_baseline_accepted_tokens = (None, None, None)
    triforce_quant_latencies, triforce_quant_latency_per_token_s, triforce_quant_accepted_tokens = (None, None, None)

    if run_baseline:
        baseline_latencies_s, baseline_latency_per_token_ms = run_baseline_one_setting()
    if run_vanilla:
        vanilla_latencies_s, vanilla_accepted_tokens = run_vanilla_one_setting()
    if run_triforce:
        triforce_baseline_latencies, triforce_baseline_latency_per_token_s, triforce_baseline_accepted_tokens = run_triforce_one_setting(quant=False)
    if run_kvquant:
        triforce_quant_latencies, triforce_quant_latency_per_token_s, triforce_quant_accepted_tokens = run_triforce_one_setting(quant=True)

    if local_rank == 0:
        p_str = str(prefill_length)
        methods = comparison_results.get('methods', {})
        b_prev = methods.get('baseline', {}).get('sweep', {}).get(p_str, {}) if isinstance(methods, dict) else {}
        v_prev = methods.get('vanilla_specdec', {}).get('sweep', {}).get(p_str, {}) if isinstance(methods, dict) else {}
        t_prev = methods.get('triforce', {}).get('sweep', {}).get(p_str, {}) if isinstance(methods, dict) else {}
        q_prev = methods.get('triforce_kv_quant', {}).get('sweep', {}).get(p_str, {}) if isinstance(methods, dict) else {}

        # Baseline stats (use fresh run if available; else keep prior values if present)
        if run_baseline and baseline_latencies_s is not None and len(baseline_latencies_s) > 0:
            mean_baseline_latency = float(np.mean(baseline_latencies_s))
            std_baseline_latency = float(np.std(baseline_latencies_s))
            mean_baseline_throughput = float(gen_len / mean_baseline_latency) if mean_baseline_latency > 0 else None
            mean_baseline_latency_per_token_ms = float(np.mean(baseline_latency_per_token_ms)) if baseline_latency_per_token_ms is not None and len(baseline_latency_per_token_ms) > 0 else None
        else:
            mean_baseline_latency = b_prev.get('mean_latency_s')
            std_baseline_latency = b_prev.get('std_latency_s')
            mean_baseline_throughput = b_prev.get('mean_throughput_tok_per_s')
            mean_baseline_latency_per_token_ms = b_prev.get('mean_latency_ms_per_token')

        # Vanilla stats
        if run_vanilla and vanilla_latencies_s is not None and len(vanilla_latencies_s) > 0:
            mean_vanilla_latency = float(np.mean(vanilla_latencies_s))
            std_vanilla_latency = float(np.std(vanilla_latencies_s))
            mean_vanilla_throughput = float(gen_len / mean_vanilla_latency) if mean_vanilla_latency > 0 else None
            mean_vanilla_accepted = float(np.mean(vanilla_accepted_tokens)) if vanilla_accepted_tokens is not None and len(vanilla_accepted_tokens) > 0 else None
        else:
            mean_vanilla_latency = v_prev.get('mean_latency_s')
            std_vanilla_latency = v_prev.get('std_latency_s')
            mean_vanilla_throughput = v_prev.get('mean_throughput_tok_per_s')
            mean_vanilla_accepted = v_prev.get('mean_accepted_tokens')

        # TriForce baseline stats
        if run_triforce and triforce_baseline_latencies is not None and len(triforce_baseline_latencies) > 0:
            mean_triforce_baseline_latency = float(np.mean(triforce_baseline_latencies))
            std_triforce_baseline_latency = float(np.std(triforce_baseline_latencies))
            mean_triforce_baseline_throughput = float(gen_len / mean_triforce_baseline_latency) if mean_triforce_baseline_latency > 0 else None
            mean_triforce_baseline_accepted = float(np.mean(triforce_baseline_accepted_tokens)) if triforce_baseline_accepted_tokens is not None and len(triforce_baseline_accepted_tokens) > 0 else None
            mean_triforce_baseline_latency_per_token_s = float(np.mean(triforce_baseline_latency_per_token_s)) if triforce_baseline_latency_per_token_s is not None and len(triforce_baseline_latency_per_token_s) > 0 else None
        else:
            mean_triforce_baseline_latency = t_prev.get('mean_latency_s')
            std_triforce_baseline_latency = t_prev.get('std_latency_s')
            mean_triforce_baseline_throughput = t_prev.get('mean_throughput_tok_per_s')
            mean_triforce_baseline_accepted = t_prev.get('mean_accepted_tokens')
            mean_triforce_baseline_latency_per_token_s = t_prev.get('mean_latency_s_per_token')

        # KV-quant stats
        if run_kvquant and triforce_quant_latencies is not None and len(triforce_quant_latencies) > 0:
            mean_triforce_quant_latency = float(np.mean(triforce_quant_latencies))
            std_triforce_quant_latency = float(np.std(triforce_quant_latencies))
            mean_triforce_quant_throughput = float(gen_len / mean_triforce_quant_latency) if mean_triforce_quant_latency > 0 else None
            mean_triforce_quant_accepted = float(np.mean(triforce_quant_accepted_tokens)) if triforce_quant_accepted_tokens is not None and len(triforce_quant_accepted_tokens) > 0 else None
            mean_triforce_quant_latency_per_token_s = float(np.mean(triforce_quant_latency_per_token_s)) if triforce_quant_latency_per_token_s is not None and len(triforce_quant_latency_per_token_s) > 0 else None
        else:
            mean_triforce_quant_latency = q_prev.get('mean_latency_s')
            std_triforce_quant_latency = q_prev.get('std_latency_s')
            mean_triforce_quant_throughput = q_prev.get('mean_throughput_tok_per_s')
            mean_triforce_quant_accepted = q_prev.get('mean_accepted_tokens')
            mean_triforce_quant_latency_per_token_s = q_prev.get('mean_latency_s_per_token')

        # Speedups (latency-based, consistent with compare_methods.py)
        speedup_vanilla_vs_baseline = (mean_baseline_latency / mean_vanilla_latency) if (mean_baseline_latency and mean_vanilla_latency) else None
        speedup_triforce_vs_baseline = (mean_baseline_latency / mean_triforce_baseline_latency) if (mean_baseline_latency and mean_triforce_baseline_latency) else None
        speedup_quant_vs_baseline = (mean_baseline_latency / mean_triforce_quant_latency) if (mean_baseline_latency and mean_triforce_quant_latency) else None
        speedup_triforce_vs_vanilla = (mean_vanilla_latency / mean_triforce_baseline_latency) if (mean_vanilla_latency and mean_triforce_baseline_latency) else None
        speedup_quant_vs_vanilla = (mean_vanilla_latency / mean_triforce_quant_latency) if (mean_vanilla_latency and mean_triforce_quant_latency) else None
        speedup_quant_vs_triforce = (mean_triforce_baseline_latency / mean_triforce_quant_latency) if (mean_triforce_baseline_latency and mean_triforce_quant_latency) else None

        # Save / merge sweep entries (do not clobber existing TriForce/KVQuant metrics if we skipped them)
        methods = comparison_results['methods']
        methods.setdefault('baseline', {'name': 'Autoregressive (No Speculation)', 'sweep': {}})
        methods.setdefault('vanilla_specdec', {'name': 'Vanilla Speculative Decoding (No Retrieval)', 'gamma': gamma, 'sweep': {}})
        methods.setdefault('triforce', {'name': 'TriForce (SpecDec + Retrieval)', 'retrieval_budget': retrieval_budget, 'gamma': gamma, 'sweep': {}})
        methods.setdefault('triforce_kv_quant', {'name': 'TriForce with KV Cache Quantization', 'retrieval_budget': retrieval_budget, 'gamma': gamma, 'sweep': {}})
        for _k in ['baseline', 'vanilla_specdec', 'triforce', 'triforce_kv_quant']:
            methods[_k].setdefault('sweep', {})

        # Baseline (overwrite if we ran it; else keep prior)
        b_entry = methods['baseline']['sweep'].get(p_str, {'prefill_length': int(prefill_length)})
        if run_baseline:
            b_entry.update({
                'prefill_length': int(prefill_length),
                'mean_latency_s': mean_baseline_latency,
                'std_latency_s': std_baseline_latency,
                'mean_throughput_tok_per_s': mean_baseline_throughput,
                'mean_latency_ms_per_token': mean_baseline_latency_per_token_ms,
                'mean_accepted_tokens': None,
            })
        b_entry['speedup_vs_baseline'] = 1.0
        methods['baseline']['sweep'][p_str] = b_entry

        # Vanilla (overwrite if we ran it; else keep prior)
        v_entry = methods['vanilla_specdec']['sweep'].get(p_str, {'prefill_length': int(prefill_length)})
        if run_vanilla:
            v_entry.update({
                'prefill_length': int(prefill_length),
                'mean_latency_s': mean_vanilla_latency,
                'std_latency_s': std_vanilla_latency,
                'mean_throughput_tok_per_s': mean_vanilla_throughput,
                'mean_accepted_tokens': mean_vanilla_accepted,
            })
        v_entry['speedup_vs_baseline'] = speedup_vanilla_vs_baseline
        methods['vanilla_specdec']['sweep'][p_str] = v_entry

        # TriForce (only overwrite mean metrics if we ran it; always refresh speedups if possible)
        t_entry = methods['triforce']['sweep'].get(p_str, {'prefill_length': int(prefill_length), 'resident_layers': int(resident_layers)})
        if run_triforce:
            t_entry.update({
                'prefill_length': int(prefill_length),
                'resident_layers': int(resident_layers),
                'mean_latency_s': mean_triforce_baseline_latency,
                'std_latency_s': std_triforce_baseline_latency,
                'mean_throughput_tok_per_s': mean_triforce_baseline_throughput,
                'mean_accepted_tokens': mean_triforce_baseline_accepted,
                'mean_latency_s_per_token': mean_triforce_baseline_latency_per_token_s,
            })
        # Always refresh speedups based on latest baseline/vanilla (if available)
        t_entry['speedup_vs_baseline'] = speedup_triforce_vs_baseline
        t_entry['speedup_vs_vanilla'] = speedup_triforce_vs_vanilla
        methods['triforce']['sweep'][p_str] = t_entry

        # KVQuant (only overwrite mean metrics if we ran it; always refresh speedups if possible)
        q_entry = methods['triforce_kv_quant']['sweep'].get(p_str, {'prefill_length': int(prefill_length), 'resident_layers': int(resident_layers)})
        if run_kvquant:
            q_entry.update({
                'prefill_length': int(prefill_length),
                'resident_layers': int(resident_layers),
                'mean_latency_s': mean_triforce_quant_latency,
                'std_latency_s': std_triforce_quant_latency,
                'mean_throughput_tok_per_s': mean_triforce_quant_throughput,
                'mean_accepted_tokens': mean_triforce_quant_accepted,
                'mean_latency_s_per_token': mean_triforce_quant_latency_per_token_s,
            })
        q_entry['speedup_vs_baseline'] = speedup_quant_vs_baseline
        q_entry['speedup_vs_vanilla'] = speedup_quant_vs_vanilla
        q_entry['speedup_vs_triforce'] = speedup_quant_vs_triforce
        methods['triforce_kv_quant']['sweep'][p_str] = q_entry

        # Print results (compact per-sweep summary)
        def _fmt_num(x, nd=4):
            return f"{x:.{nd}f}" if isinstance(x, (float, int)) and x is not None else "NA"
        def _fmt_speed(x):
            return f"{x:.2f}x" if isinstance(x, (float, int)) and x is not None else "NA"

        print(colored(
            f"\n[Baseline][p={prefill_length}] "
            f"Latency: {_fmt_num(mean_baseline_latency, 4)} ± {_fmt_num(std_baseline_latency, 4)} s | "
            f"Throughput: {_fmt_num(mean_baseline_throughput, 2)} tok/s",
            "yellow"
        ))
        print(colored(
            f"[VanillaSpecDec][p={prefill_length}] "
            f"Latency: {_fmt_num(mean_vanilla_latency, 4)} ± {_fmt_num(std_vanilla_latency, 4)} s | "
            f"Throughput: {_fmt_num(mean_vanilla_throughput, 2)} tok/s | "
            f"Accepted: {_fmt_num(mean_vanilla_accepted, 2)} | "
            f"Speedup vs Baseline: {_fmt_speed(speedup_vanilla_vs_baseline)}",
            "blue"
        ))
        print(colored(
            f"[TriForce][p={prefill_length}] "
            f"Latency: {_fmt_num(mean_triforce_baseline_latency, 4)} ± {_fmt_num(std_triforce_baseline_latency, 4)} s | "
            f"Throughput: {_fmt_num(mean_triforce_baseline_throughput, 2)} tok/s | "
            f"Accepted: {_fmt_num(mean_triforce_baseline_accepted, 2)} | "
            f"Speedup vs Baseline: {_fmt_speed(speedup_triforce_vs_baseline)}",
            "green"
        ))
        print(colored(
            f"[MSpecKV][p={prefill_length}] "
            f"Latency: {_fmt_num(mean_triforce_quant_latency, 4)} ± {_fmt_num(std_triforce_quant_latency, 4)} s | "
            f"Throughput: {_fmt_num(mean_triforce_quant_throughput, 2)} tok/s | "
            f"Accepted: {_fmt_num(mean_triforce_quant_accepted, 2)} | "
            f"Speedup vs Baseline: {_fmt_speed(speedup_quant_vs_baseline)} | "
            f"Speedup vs TriForce: {_fmt_speed(speedup_quant_vs_triforce)}",
            "green"
        ))

    # (models freed inside run_triforce_one_setting)

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
                'prefill_length',
                'on_chip_layers',
                'resident_layers',
                'baseline_mean_latency_s',
                'baseline_std_latency_s',
                'baseline_mean_throughput_tok_per_s',
                'vanilla_mean_latency_s',
                'vanilla_std_latency_s',
                'vanilla_mean_throughput_tok_per_s',
                'vanilla_mean_accepted_tokens',
                'triforce_mean_latency_s',
                'triforce_std_latency_s',
                'triforce_mean_throughput_tok_per_s',
                'triforce_mean_accepted_tokens',
                'triforce_mean_latency_s_per_token',
                'kvquant_mean_latency_s',
                'kvquant_std_latency_s',
                'kvquant_mean_throughput_tok_per_s',
                'kvquant_mean_accepted_tokens',
                'kvquant_mean_latency_s_per_token',
                'speedup_vanilla_vs_baseline',
                'speedup_triforce_vs_baseline',
                'speedup_kvquant_vs_baseline',
                'speedup_triforce_vs_vanilla',
                'speedup_kvquant_vs_vanilla',
                'speedup_kvquant_vs_triforce',
            ])
            for prefill_length in prefill_lengths_values:
                b = comparison_results['methods']['baseline']['sweep'].get(str(prefill_length), {})
                v = comparison_results['methods']['vanilla_specdec']['sweep'].get(str(prefill_length), {})
                t = comparison_results['methods']['triforce']['sweep'].get(str(prefill_length), {})
                q = comparison_results['methods']['triforce_kv_quant']['sweep'].get(str(prefill_length), {})
                writer.writerow([
                    prefill_length,
                    args.on_chip,
                    t.get('resident_layers'),
                    b.get('mean_latency_s'),
                    b.get('std_latency_s'),
                    b.get('mean_throughput_tok_per_s'),
                    v.get('mean_latency_s'),
                    v.get('std_latency_s'),
                    v.get('mean_throughput_tok_per_s'),
                    v.get('mean_accepted_tokens'),
                    t.get('mean_latency_s'),
                    t.get('std_latency_s'),
                    t.get('mean_throughput_tok_per_s'),
                    t.get('mean_accepted_tokens'),
                    t.get('mean_latency_s_per_token'),
                    q.get('mean_latency_s'),
                    q.get('std_latency_s'),
                    q.get('mean_throughput_tok_per_s'),
                    q.get('mean_accepted_tokens'),
                    q.get('mean_latency_s_per_token'),
                    v.get('speedup_vs_baseline'),
                    t.get('speedup_vs_baseline'),
                    q.get('speedup_vs_baseline'),
                    t.get('speedup_vs_vanilla'),
                    q.get('speedup_vs_vanilla'),
                    q.get('speedup_vs_triforce'),
                ])
        print(colored(f"\nSaved CSV sweep summary to: {output_csv}", "yellow"))

    # JSON full results (includes per-sample arrays)
    if output_json is not None:
        with open(output_json, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(colored(f"Saved JSON sweep results to: {output_json}", "yellow"))

    # End-of-run speedup summary (per prefill length)
    print(colored("\n" + "=" * 80, "magenta"))
    print(colored("SPEEDUP SUMMARY (latency-based)", "magenta"))
    print(colored("=" * 80, "magenta"))
    print(f"\n{'prefill':>8} | {'vanilla_vs_base':>15} | {'triforce_vs_base':>15} | {'kvquant_vs_base':>15} | {'kvquant_vs_triforce':>18}")
    print("-" * 86)
    for p in prefill_lengths_values:
        v = comparison_results['methods']['vanilla_specdec']['sweep'].get(str(p), {})
        t = comparison_results['methods']['triforce']['sweep'].get(str(p), {})
        q = comparison_results['methods']['triforce_kv_quant']['sweep'].get(str(p), {})
        def fmt(x):
            return f"{x:.2f}x" if isinstance(x, (float, int)) and x is not None else "NA"
        print(f"{p:>8} | {fmt(v.get('speedup_vs_baseline')):>15} | {fmt(t.get('speedup_vs_baseline')):>15} | {fmt(q.get('speedup_vs_baseline')):>15} | {fmt(q.get('speedup_vs_triforce')):>18}")


dist.destroy_process_group()
