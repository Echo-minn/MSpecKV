# CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --prefill 124928 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6 --dataset 128k

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored
from tqdm import tqdm
from data.dataset import get_dataset
from models.modeling_llama import LlamaForCausalLM
from models.modeling_llama_68m import LlamaForCausalLM as LlamaForCausalLM_68M
from models.cache import FlashSimpleCache, StreamingLLMEvictionCache, RetrievalCache
from utils.decoding import Autoregressive, TriForce
from utils.misc import print_config
from utils.graph_infer import GraphInferenceEngine

# test/on_chip.py (Add this code right after imports)
import os
import torch
import torch.distributed as dist

# --- Distributed Initialization for torchrun ---
# LOCAL_RANK will be 0 for GPU 3 and 1 for GPU 4
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Only proceed with DDP setup if torchrun environment variables are present
if dist.is_available() and 'WORLD_SIZE' in os.environ:
    
    # 1. Set the device for this specific process (0 or 1)
    # This correctly maps to the first (3) or second (4) visible device.
    torch.cuda.set_device(local_rank)
    
    # 2. Initialize the distributed process group
    dist.init_process_group(backend='nccl')
    
    # Optional: Print confirmation
    if local_rank == 0:
        print(f"DDP Initialized. Rank: {dist.get_rank()}, Local Rank: {local_rank}, World Size: {dist.get_world_size()}")
    
else:
    # Fallback for single-GPU or non-DDP run
    torch.cuda.set_device(local_rank) # Still set device 0 if it's a single run
    print("Running in non-distributed mode or DDP not available.")
# ----------------------------------------------------

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--draft', type=str, default='llama-68M', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--dataset', type=str, default='gs', help='dataset')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument('--draft_cache_budget', type=int, default=256, help='draft cache budget')
    parser.add_argument('--chunk_size', type=int, default=8, help='chunk size')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_arguments()

    ######## model initialization ########
    if args.target == 'llama-7B-128K':
        target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", torch_dtype=torch.float16, device_map="auto")
    else:
        raise NotImplementedError
    target = target.eval()

    draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", torch_dtype=torch.float16, device_map="auto")
    draft = draft.eval()

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", use_fast=True, legacy=False)
    tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=args.prefill)

    ######## sampling parameters ########

    top_k = -1
    top_p = args.top_p
    temperature = args.temp

    prefill = args.prefill
    gen_len = args.gen_len
    gamma = args.gamma
    verbose = args.verbose
    chunk_size = args.chunk_size
    max_budget = args.budget

    print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=None, method="TriForce", spec_args={'budget': args.budget, 'chunk_size': chunk_size}, dataset=args.dataset)

    ####### cache init #######

    draft_cache_budget = args.draft_cache_budget
    recent_size = draft_cache_budget - 16 - gamma
    cache = FlashSimpleCache(target, prefill+gen_len+16)
    graph_cache = RetrievalCache(target, max_budget=max_budget, prefill=prefill, gamma=gamma, chunk_size=chunk_size)
    draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)

    graph_engine = GraphInferenceEngine(target, cache, graph_cache, draft, draft_cache)
    graph_engine.initialize_cuda_graph(gamma, probs=True, temperature=temperature, top_p=top_p)

    cache.print_status()
    graph_cache.print_status()
    draft_cache.print_status()

    print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))

    ######## Warm up for baseline ########
    n_warmups = 1
    input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
    for i in tqdm(range(n_warmups), desc="Autoregressive Warmup"):
        Autoregressive(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)

    all_speed = []
    for input_ids in tqdm(tokenized_prompts[:1], desc="Autoregressive Test"):
        input_ids = input_ids.to(target.device)[:,:prefill]
        speed = Autoregressive(tokenizer, graph_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)
        all_speed.append(speed)

    baseline_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"[Autoregressive] average latency: {baseline_latency} ms", "red"))

    ######## Warm up for our method ########
    n_warmups = 3
    input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
    for i in tqdm(range(n_warmups), desc="TriForce Warmup"):
        TriForce(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset, spec_args={'budget': args.budget, 'draft': args.draft, 'chunk_size': chunk_size, 'baseline': baseline_latency/1000})

    all_acceptance_rate = []
    all_speed = []
    for input_ids in tqdm(tokenized_prompts, desc="TriForce Test"):
        input_ids = input_ids.to(target.device)[:,:prefill]

        acceptance_rate, speed = TriForce(tokenizer, graph_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset, spec_args={'budget': args.budget, 'draft': args.draft, 'chunk_size': chunk_size, 'gamma': gamma, 'temperature': temperature, 'top_p': top_p, 'baseline': baseline_latency/1000})
        all_acceptance_rate.append(acceptance_rate)
        all_speed.append(speed)

    method_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"average acceptance rate (NOT per token): {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
    print(colored(f"[TriForce] average latency: {method_latency} ms", "red"))
    print(colored(f"[E2E Speedup]: {baseline_latency / method_latency}", "red"))