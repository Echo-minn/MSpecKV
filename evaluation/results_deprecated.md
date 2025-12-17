# Evaluation Results

## Resident Layer Ablation

### results w/ and w/o KV Cache Quantization on MSpecKV
`--budget 4096 --prefill 8192 --dataset demo --target llama-7B-128K --on_chip 16`

| Resident layers | w/o latency (s) | w/o tok/s | w/o accepted | w/ Quant latency (s) | w/ Quant tok/s | w/ Quant accepted | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4  | 17.7051 | 14.46 | 7.15 | 17.3270 | 14.77 | 6.10 | 1.02× |
| 8  | 16.1352 | 15.87 | 7.48 | 14.4644 | 17.70 | 6.94 | 1.12× |
| 10 | 17.5271 | 14.61 | 7.48 | 13.8001 | 18.55 | 7.08 | 1.27× |
| 12 | 17.6354 | 14.52 | 7.18 | 14.1811 | 18.05 | 7.08 | 1.24× |
| 16 | 16.3894 | 15.62 | 6.94 | 10.4613 | 24.47 | 7.32 | 1.57× |
| 18 | 19.4326 | 13.17 | 7.12 | 10.6619 | 24.01 | 7.32 | 1.82× |
| 20 | 18.4875 | 13.85 | 6.71 | 10.8895 | 23.51 | 7.18 | 1.70× |
| 24 | 21.5845 | 11.86 | 5.67 | 10.7214 | 23.88 | 6.85 | 2.01× |

### result analysis

- **What “resident layers” means here**: in this codebase, “resident layers” is the **size of an LRU GPU-resident KV buffer** for *offloaded* layers (`kv_resident_max_layers`). It is only used when **KV quantization is enabled**, where the offloaded KV is stored on CPU as int8 + scales and can be **dequantized once and kept on GPU** for reuse across decoding steps.

- **Why more resident layers increases speedup (with quantization)**: with KV quantization enabled, every offloaded layer needs CPU→GPU transfer (int8 + scales) plus dequantization back to fp16 before attention. Increasing resident layers lets more of those layers hit the GPU-resident cache, **reducing repeated transfer + dequant work on the critical path**. That’s why throughput rises from ~14.8 tok/s (4) to ~24 tok/s (16–24) and the speedup grows up to ~2×.

## Output Length Ablation
`--budget 4096 --prefill 8192 --on_chip 16 --gamma 8 --resident_layers 16`
| Gen len | w/o latency (s) | w/o tok/s | w/o accepted | w/ Quant latency (s) | w/ Quant tok/s | w/ Quant accepted | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256  | 17.9959 | 14.23 | 7.15 | 11.6549 | 21.96 | 6.10 | 1.54× |
| 512  | 39.8386 | 12.85 | 7.40 | 20.6942 | 24.74 | 7.76 | 1.93× |
| 768  | 64.7242 | 11.87 | 7.24 | 47.8942 | 16.04 | 6.33 | 1.35× |
| 1024 | 84.8861 | 12.06 | 7.36 | 56.8657 | 18.01 | 6.91 | 1.49× |
| 1536 | 120.8360 | 12.71 | 7.37 | 75.0347 | 20.47 | 7.67 | 1.61× |
| 2048 | 170.6308 | 12.00 | 7.83 | 95.1569 | 21.52 | 7.74 | 1.79× |

### result analysis

- **Trend**: quantization improves decoding throughput at every output length (1.35–1.93×), with the speedup generally larger at longer generations (more tokens decoded).
- **Mechanism**: under KV offloading, each decoded token increases the KV length and forces more KV movement for offloaded layers. KV quantization reduces the bytes transferred (int8 + scale instead of fp16), so the *per-token* offload/memory cost grows more slowly, yielding higher tok/s.
- **Why it’s not monotonic**: speculative decoding has run-to-run variability (accepted tokens and rejection patterns change the number of verification steps and cache updates), and the bottleneck can shift between **bandwidth-bound KV transfer/dequant** vs **GPU compute** as the sequence grows. This produces local dips (e.g., at gen_len=768) even when the overall benefit remains positive.

## Context Length Ablation: with and without KV Cache Quantization on MSpecKV
`--budget 4096 --on_chip 16 --gamma 8 --resident_layers 16 `
| Prefill length | w/o latency (s) | w/o tok/s | w/o accepted | w/ Quant latency (s) | w/ Quant tok/s | w/ Quant accepted | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192  | 18.2765 | 14.01 | 7.15 | 12.2737 | 20.86 | 6.10 | 1.49× |
| 16384 | 33.7877 | 7.58  | 6.23 | 12.6869 | 20.18 | 5.39 | 2.66× |
| 24576 | 32.3367 | 7.92  | 6.56 | 12.0987 | 21.16 | 5.74 | 2.67× |
| 32768 | 47.8249 | 5.35  | 5.13 | 12.0974 | 21.16 | 5.22 | 3.95× |

### result analysis

- **Trend**: as prefill length increases, the baseline throughput collapses (14.0 → 5.35 tok/s), while the quantized throughput stays roughly flat (~20–21 tok/s). Consequently, speedup grows sharply (1.49× → 3.95×).
- **Mechanism**: during decoding, the KV cache length is approximately “prefill + decoded so far”. With KV offloading, longer prefills imply larger KV slices that must be staged for offloaded layers, pushing the baseline into a **bandwidth-dominated regime**. KV quantization compresses the offloaded KV representation, so the cost scaling with context length is dramatically reduced; decoding becomes closer to compute-bound, hence the stable tok/s.

## On-Chip Layer Ablation

`--budget 4096 --prefill 8192 --gamma 8 --resident_layers 16`
### On-chip layer ablation (baseline vs KV-quant)

| On-chip layers | w/o latency (s) | w/o tok/s | w/o accepted | w/ Quant latency (s) | w/ Quant tok/s | w/ Quant accepted | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4  | 21.6424 | 11.83 | 7.15 | 21.1880 | 12.08 | 6.37 | 1.02× |
| 6  | 23.2559 | 11.01 | 6.39 | 18.8469 | 13.58 | 6.34 | 1.23× |
| 8  | 22.2941 | 11.48 | 6.74 | 14.9452 | 17.13 | 7.25 | 1.49× |
| 10 | 20.8235 | 12.29 | 6.76 | 17.1182 | 14.95 | 6.01 | 1.22× |
| 12 | 18.7684 | 13.64 | 6.79 | 15.4251 | 16.60 | 6.29 | 1.22× |
| 14 | 17.4131 | 14.70 | 6.86 | 15.5621 | 16.45 | 6.39 | 1.12× |
| 16 | 20.9478 | 12.22 | 6.83 | 10.8042 | 23.69 | 7.00 | 1.94× |

### Result + why this speedup happens (concise)
- **Trend**: KV-quant speedup is **small at low on-chip (4)**, grows at **moderate (6–8)**, dips around **10–14**, then is **largest at 16**.
- **Why**: KV-quant primarily helps when runtime is **offload / memory-bandwidth dominated** (moving KV between CPU↔GPU). Quantization shrinks KV transfer and CPU storage bandwidth, so it matters more when more layers are offloaded.
- **Why it’s not monotonic**: changing `on_chip_layers` changes the balance between:
  - **GPU compute** (more on-chip layers → more compute stays on GPU) vs
  - **offload traffic** (fewer on-chip layers → more KV movement)
  and also impacts **cache behavior / retrieval path costs**, so the bottleneck shifts. At 16, you likely hit a “sweet spot” where quantization removes a big remaining bandwidth bottleneck without increasing other overheads.

## Baseline vs MSpecKV vs Quantized_MSpecKV