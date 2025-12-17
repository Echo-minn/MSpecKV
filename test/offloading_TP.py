# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 test/offloading_TP.py --budget 12288 --prefill 130048 --dataset demo --target llama-7B-128K --on_chip 9 --gamma 16 --seed 1

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

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
    parser.add_argument('--gamma', type=str, default=6)
    args = parser.parse_args()
    
    return args

args = parse_arguments()
torch.manual_seed(args.seed)
prefill = args.prefill
gen_len = args.gen_len
temperature = args.temp
top_p = args.top_p
retrieval_budget = args.budget

local_rank, world_size = distributed_init()
device = torch.device("cuda", local_rank)

####### tree #######

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

def _init_llm_baseline():
    llm = DistributedLlama(
        model_name_or_path=model_name_or_path,
        local_rank=local_rank,
        world_size=world_size,
        prefill=prefill,
        gen_len=gen_len,
        temperature=temperature,
        top_p=top_p,
        flash_attn=True,
        retrieval_budget=0,
        kv_offload=True,
        on_chip_layers=args.on_chip,
    )
    for rank in range(world_size):
        if local_rank == rank:
            hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
            llm.init_parameters(hf_model=hf_model)
            del hf_model
        dist.barrier()
    return llm

def _init_llm_mspeckv(kv_quant: bool):
    gamma = int(args.gamma)
    draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map=device).eval()
    recent_size = 256 - 16 - gamma
    draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)
    llm = DistributedLlama(
        model_name_or_path=model_name_or_path,
        local_rank=local_rank,
        world_size=world_size,
        prefill=prefill,
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
        kv_cache_quant=kv_quant,
        kv_cache_quant_bits=8,
        kv_fp16_tail=0,
        kv_resident_gpu=kv_quant,
        kv_resident_max_layers=(args.resident_layers if kv_quant else 0),
    )
    for rank in range(world_size):
        if local_rank == rank:
            hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='cpu')
            llm.init_parameters(hf_model=hf_model)
            del hf_model
        dist.barrier()
    return llm

def _cleanup_llm(llm):
    del llm
    dist.barrier()
    torch.cuda.empty_cache()

def _run_baseline():
    llm = _init_llm_baseline()
    lat_ms, _ = Baseline_Dist(tokenizer, llm, input_ids, max_len=gen_len, temperature=temperature, top_p=top_p, local_rank=local_rank)
    lat_s = (lat_ms / 1000.0)
    if local_rank == 0:
        print(colored(f"\n[Autoregressive] latency: {lat_s:.6f} s/token", "red"))
    _cleanup_llm(llm)
    return lat_s

def _run_mspeckv(name: str, kv_quant: bool):
    gamma = int(args.gamma)
    llm = _init_llm_mspeckv(kv_quant=kv_quant)
    lats, accs = [], []
    for prompt in tqdm(tokenized_prompts, desc=name, disable=(local_rank != 0)):
        prompt = prompt[:, :prefill].to(llm.device)
        avg_tokens, lat = MSpecKV_Dist(tokenizer, llm, prompt, gamma=gamma, max_len=gen_len, top_k=-1, top_p=top_p, temperature=temperature, verbose=False, file_path=None, dataset=args.dataset)
        lats.append(float(lat))
        accs.append(float(avg_tokens) if not torch.is_tensor(avg_tokens) else float(avg_tokens.item()))
    mean_lat = float(np.mean(lats)) if len(lats) else float('nan')
    mean_acc = float(np.mean(accs)) if len(accs) else float('nan')
    if local_rank == 0:
        print(colored(f"\n[{name}] latency: {mean_lat:.6f} s/token | accepted: {mean_acc:.2f}", "green" if not kv_quant else "cyan"))
    _cleanup_llm(llm)
    return mean_lat, mean_acc

if args.compare:
    b = _run_baseline()
    t, _ = _run_mspeckv("MSpecKV", kv_quant=False)
    q, _ = _run_mspeckv("Quantized_MSpecKV", kv_quant=True)
    if local_rank == 0:
        print(f"\nSpeedup vs AR: MSpecKV {b/t:.2f}x | Quantized_MSpecKV {b/q:.2f}x")
    dist.destroy_process_group()
    raise SystemExit(0)

if args.baseline:
    _run_baseline()
    dist.destroy_process_group()
    raise SystemExit(0)

else:
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

    ######## MSpecKV ########
    all_avg_tokens = []
    all_latency = []

    for input_ids in tqdm(tokenized_prompts, desc="MSpecKV Test"):
        input_ids = input_ids[:,:args.prefill].to(llm.device)

        avg_tokens, latency = MSpecKV_Dist(tokenizer, llm, input_ids, gamma=gamma, max_len=gen_len, top_k=-1, top_p=top_p, temperature=temperature, verbose=False, file_path=None, dataset=args.dataset)
        all_avg_tokens.append(avg_tokens)
        all_latency.append(latency)
        if local_rank == 0:
            print(colored(f"\n[MSpecKV] average latency: {latency} s", "red"))
            print(colored(f"[MSpecKV] average accepted tokens: {avg_tokens}", "red"))
    if local_rank == 0:
        print(f"[Overall Latency]: {np.array(all_latency).mean()}")
        print(f"[Overall Avg Accepted Tokens]: {np.array(all_avg_tokens).mean()}")

    # destory the distributed process
    dist.destroy_process_group()