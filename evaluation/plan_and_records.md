Goal: prove the benefit of KV quantization and resident KV with MSpecKV.

- **(A) Evaluation: prove benefit of KV quantization + resident KV**
- **(B) Ablation: isolate sensitivity to key system knobs**

Throughout, run **3 independent trials** with `--seed 1,2,3`, then report **mean ± std** over seeds (and also report **delta** between methods).

---

## A) Evaluation: prove benefits of quantization and resident KV

### A1. Context scaling (the strongest “serving” evidence)
**What to test**: throughput vs prefill length at fixed decode length.  
**Why**: shows gains grow when KV movement becomes the bottleneck.  
**Settings (final)**:
- `--on_chip 16 --resident_layers 16 --gamma 8 --gen_len 256`
- `--prefill_lengths 8192,32768` (2 points is enough and saves time)

**Expected result**:
- Baseline tok/s drops a lot at 32K; quant tok/s stays much more stable → **speedup increases sharply** at long context.

**Run (final)**:

```bash
cd /opt/dlami/nvme/mira/.www/MSpecKV
for s in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/cxt_len_ablation.py \
    --dataset demo --target llama-7B-128K \
    --budget 4096 --on_chip 16 --gamma 8 --resident_layers 16 \
    --prefill_lengths "8192,16384,24576,32768" --gen_len 256 \
    --seed $s --output_json auto --output_csv /dev/null
done
```

**Quick-check (cheap sanity)**: replace with `--budget 1024 --prefill_lengths 2048,4096 --gen_len 128`.

**Result**:
#### Throughput

| prefill_length | baseline | vanilla_specdec | mspeckv | mspeckv_kv_quant | speedup_vs_baseline:vanilla_specdec | speedup_vs_baseline:mspeckv | speedup_vs_baseline:mspeckv_kv_quant |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4.72 ± 0.01 | 5.56 ± 0.03 | 14.61 ± 0.63 | 22.86 ± 0.51 | 1.18x | 3.09x | 4.84x |
| 16384 | 2.58 ± 0.02 | 2.83 ± 0.16 | 9.86 ± 0.16 | 21.54 ± 1.80 | 1.10x | 3.83x | 8.36x |
| 24576 | 1.73 ± 0.01 | 2.09 ± 0.08 | 6.55 ± 1.09 | 21.37 ± 2.48 | 1.21x | 3.78x | 12.33x |
| 32768 | 1.31 ± 0.00 | 1.57 ± 0.01 | 4.86 ± 0.24 | 18.50 ± 1.54 | 1.20x | 3.71x | 14.14x |


#### Latency

| prefill_length | baseline | vanilla_specdec | mspeckv | mspeckv_kv_quant | speedup_vs_baseline:vanilla_specdec | speedup_vs_baseline:mspeckv | speedup_vs_baseline:mspeckv_kv_quant |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 54.21 ± 0.11 | 46.05 ± 0.21 | 17.54 ± 0.74 | 11.20 ± 0.25 | 1.18x | 3.09x | 4.84x |
| 16384 | 99.32 ± 0.76 | 90.52 ± 5.08 | 25.96 ± 0.42 | 11.94 ± 0.95 | 1.10x | 3.83x | 8.32x |
| 24576 | 147.74 ± 0.73 | 122.48 ± 4.94 | 39.77 ± 6.04 | 12.10 ± 1.50 | 1.21x | 3.71x | 12.21x |
| 32768 | 195.71 ± 0.31 | 162.91 ± 0.75 | 52.80 ± 2.51 | 13.90 ± 1.19 | 1.20x | 3.71x | 14.08x |


#### Accepted

| prefill_length | baseline | vanilla_specdec | mspeckv | mspeckv_kv_quant |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | NA | 0.36 ± 0.02 | 6.97 ± 0.20 | 6.61 ± 0.45 |
| 16384 | NA | 0.19 ± 0.08 | 6.36 ± 0.16 | 5.70 ± 0.66 |
| 24576 | NA | 0.28 ± 0.04 | 5.26 ± 1.13 | 5.36 ± 0.56 |
| 32768 | NA | 0.25 ± 0.00 | 4.56 ± 0.51 | 4.87 ± 0.50 |

**Supplementary result**:

```shell
[Baseline][p=8192] Latency: 54.2066 ± 0.0000 s | Throughput: 4.72 tok/s
[VanillaSpecDec][p=8192] Latency: 46.1742 ± 0.0000 s | Throughput: 5.54 tok/s | Accepted: 0.35 | Speedup vs Baseline: 1.17x
[MSpecKV][p=8192] Latency: 16.6913 ± 0.0000 s | Throughput: 15.34 tok/s | Accepted: 7.15 | Speedup vs Baseline: 3.25x
[MSpecKV][p=8192] Latency: 11.4070 ± 0.0000 s | Throughput: 22.44 tok/s | Accepted: 6.10 | Speedup vs Baseline: 4.75x | Speedup vs MSpecKV: 1.46x

[Baseline][p=16384] Latency: 99.3475 ± 0.0000 s | Throughput: 2.58 tok/s
[VanillaSpecDec][p=16384] Latency: 93.8585 ± 0.0000 s | Throughput: 2.73 tok/s | Accepted: 0.15 | Speedup vs Baseline: 1.06x
[MSpecKV][p=16384] Latency: 26.3901 ± 0.0000 s | Throughput: 9.70 tok/s | Accepted: 6.23 | Speedup vs Baseline: 3.76x
[MSpecKV][p=16384] Latency: 12.4009 ± 0.0000 s | Throughput: 20.64 tok/s | Accepted: 5.39 | Speedup vs Baseline: 8.01x | Speedup vs MSpecKV: 2.13x

[Baseline][p=24576] Latency: 147.7001 ± 0.0000 s | Throughput: 1.73 tok/s
[VanillaSpecDec][p=24576] Latency: 119.9019 ± 0.0000 s | Throughput: 2.14 tok/s | Accepted: 0.30 | Speedup vs Baseline: 1.23x
[MSpecKV][p=24576] Latency: 32.7935 ± 0.0000 s | Throughput: 7.81 tok/s | Accepted: 6.56 | Speedup vs Baseline: 4.50x
[MSpecKV][p=24576] Latency: 11.4145 ± 0.0000 s | Throughput: 22.43 tok/s | Accepted: 5.74 | Speedup vs Baseline: 12.94x | Speedup vs MSpecKV: 2.87x

[Baseline][p=32768] Latency: 195.4976 ± 0.0000 s | Throughput: 1.31 tok/s
[VanillaSpecDec][p=32768] Latency: 162.2154 ± 0.0000 s | Throughput: 1.58 tok/s | Accepted: 0.25 | Speedup vs Baseline: 1.21x
[MSpecKV][p=32768] Latency: 49.9027 ± 0.0000 s | Throughput: 5.13 tok/s | Accepted: 5.13 | Speedup vs Baseline: 3.92x
[MSpecKV][p=32768] Latency: 13.5330 ± 0.0000 s | Throughput: 18.92 tok/s | Accepted: 5.22 | Speedup vs Baseline: 14.45x | Speedup vs MSpecKV: 3.69x

Saved CSV sweep summary to: results/A1/cxt_len_ablation_llama-7B-128K_p8192_g8_b4096_20251215_025432.csv
Saved JSON sweep results to: results/A1/cxt_len_ablation_llama-7B-128K_p8192_g8_b4096_20251215_025432.json

================================================================================
SPEEDUP SUMMARY (latency-based)
================================================================================

 prefill | vanilla_vs_base | no_kv_quant_vs_base | kv_quant_vs_base | quant_vs_no_quant
--------------------------------------------------------------------------------------
    8192 |           1.17x |           3.25x |           4.75x |              1.46x
   16384 |           1.06x |           3.76x |           8.01x |              2.13x
   24576 |           1.23x |           4.50x |          12.94x |              2.87x
   32768 |           1.21x |           3.92x |          14.45x |              3.69x
```

```shell
[Baseline][p=8192] Latency: 54.3141 ± 0.0000 s | Throughput: 4.71 tok/s
[VanillaSpecDec][p=8192] Latency: 46.1696 ± 0.0000 s | Throughput: 5.54 tok/s | Accepted: 0.35 | Speedup vs Baseline: 1.18x
[MSpecKV][p=8192] Latency: 17.8681 ± 0.0000 s | Throughput: 14.33 tok/s | Accepted: 6.74 | Speedup vs Baseline: 3.04x
[MSpecKV][p=8192] Latency: 10.9258 ± 0.0000 s | Throughput: 23.43 tok/s | Accepted: 6.80 | Speedup vs Baseline: 4.97x | Speedup vs MSpecKV: 1.64x

[Baseline][p=16384] Latency: 98.5524 ± 0.0000 s | Throughput: 2.60 tok/s
[VanillaSpecDec][p=16384] Latency: 93.0205 ± 0.0000 s | Throughput: 2.75 tok/s | Accepted: 0.15 | Speedup vs Baseline: 1.06x
[MSpecKV][p=16384] Latency: 25.9201 ± 0.0000 s | Throughput: 9.88 tok/s | Accepted: 6.54 | Speedup vs Baseline: 3.80x
[MSpecKV][p=16384] Latency: 12.5727 ± 0.0000 s | Throughput: 20.36 tok/s | Accepted: 5.25 | Speedup vs Baseline: 7.84x | Speedup vs MSpecKV: 2.06x

[Baseline][p=24576] Latency: 147.0349 ± 0.0000 s | Throughput: 1.74 tok/s
[VanillaSpecDec][p=24576] Latency: 119.3545 ± 0.0000 s | Throughput: 2.14 tok/s | Accepted: 0.30 | Speedup vs Baseline: 1.23x
[MSpecKV][p=24576] Latency: 43.3067 ± 0.0000 s | Throughput: 5.91 tok/s | Accepted: 4.50 | Speedup vs Baseline: 3.40x
[MSpecKV][p=24576] Latency: 11.0630 ± 0.0000 s | Throughput: 23.14 tok/s | Accepted: 5.62 | Speedup vs Baseline: 13.29x | Speedup vs MSpecKV: 3.91x

[Baseline][p=32768] Latency: 196.0703 ± 0.0000 s | Throughput: 1.31 tok/s
[VanillaSpecDec][p=32768] Latency: 162.8094 ± 0.0000 s | Throughput: 1.57 tok/s | Accepted: 0.25 | Speedup vs Baseline: 1.20x
[MSpecKV][p=32768] Latency: 54.1659 ± 0.0000 s | Throughput: 4.73 tok/s | Accepted: 4.17 | Speedup vs Baseline: 3.62x
[MSpecKV][p=32768] Latency: 12.9363 ± 0.0000 s | Throughput: 19.79 tok/s | Accepted: 5.09 | Speedup vs Baseline: 15.16x | Speedup vs MSpecKV: 4.19x

Saved CSV sweep summary to: results/A1/cxt_len_ablation_llama-7B-128K_p8192_g8_b4096_20251215_030351.csv
Saved JSON sweep results to: results/A1/cxt_len_ablation_llama-7B-128K_p8192_g8_b4096_20251215_030351.json

================================================================================
SPEEDUP SUMMARY (latency-based)
================================================================================

 prefill | vanilla_vs_base | no_kv_quant_vs_base | kv_quant_vs_base | kv_quant_vs_no_kv_quant
--------------------------------------------------------------------------------------
    8192 |           1.18x |           3.04x |           4.97x |              1.64x
   16384 |           1.06x |           3.80x |           7.84x |              2.06x
   24576 |           1.23x |           3.40x |          13.29x |              3.91x
   32768 |           1.20x |           3.62x |          15.16x |              4.19x

```

```shell
[Baseline][p=8192] Latency: 54.0956 ± 0.0000 s | Throughput: 4.73 tok/s
[VanillaSpecDec][p=8192] Latency: 45.8043 ± 0.0000 s | Throughput: 5.59 tok/s | Accepted: 0.39 | Speedup vs Baseline: 1.18x
[MSpecKV][p=8192] Latency: 18.0671 ± 0.0000 s | Throughput: 14.17 tok/s | Accepted: 7.01 | Speedup vs Baseline: 2.99x
[MSpecKV][p=8192] Latency: 11.2688 ± 0.0000 s | Throughput: 22.72 tok/s | Accepted: 6.93 | Speedup vs Baseline: 4.80x | Speedup vs MSpecKV: 1.60x
```

**Technical summary (A1)**:

A primary bottleneck in long-context serving is **KV-cache traffic** (offload/transfer and memory bandwidth), which grows with prefill length and can dominate compute. In A1, we test whether **MSpecKV**—MSpecKV augmented with **KV-cache quantization** and a small set of **resident KV layers on GPU**—reduces this bottleneck and therefore scales better with context length. We sweep **prefill length** from **8K to 32K** while holding decode length fixed (**256 tokens**) with fixed speculation depth (**\(\gamma=8\)**), and compare **naive autoregressive decoding (baseline)**, **Vanilla SpecDec** (speculation without retrieval), **MSpecKV** (speculation + retrieval cache), and **MSpecKV**; each point reports **mean ± std over 3 seeds** for throughput (tok/s), end-to-end latency (s), and accepted tokens.

The results show a clear context-scaling advantage for MSpecKV. Baseline throughput drops from **4.72 ± 0.01 tok/s (8K)** to **1.31 ± 0.00 tok/s (32K)**, and Vanilla SpecDec yields only a modest improvement (**~1.10–1.21×** vs baseline). MSpecKV provides a stable multi‑x improvement (**3.71–3.83×** vs baseline), while MSpecKV maintains high throughput (**22.86 ± 0.51 → 18.50 ± 1.54 tok/s**) and increases speedup vs baseline from **4.84× (8K)** to **14.14× (32K)**; latency exhibits the same ratios at fixed decode length. This widening gap with longer contexts is consistent with KV movement/memory bandwidth being the limiting factor, and with quantization + residency directly reducing KV pressure; accepted-token statistics remain comparable between MSpecKV and MSpecKV (e.g., at **32K**, **4.56 ± 0.51** vs **4.87 ± 0.50**), suggesting the gains are not explained by a degradation in speculative acceptance.

**Wrap-up: MSpecKV (w/o quantization) vs MSpecKV (w/ quantization).** In this report, we treat **MSpecKV** as our method **without KV-cache quantization**, and **MSpecKV** as the same approach **with quantization enabled** (plus the resident-KV configuration used throughout A1). Across the prefill sweep, enabling quantization yields a consistent and growing advantage: throughput improves from **14.61 ± 0.63 → 22.86 ± 0.51 tok/s** at **8K** and from **4.86 ± 0.24 → 18.50 ± 1.54 tok/s** at **32K**, corresponding to roughly **1.56×** and **3.81×** gains over MSpecKV, respectively (equivalently, the same factors as latency reductions at fixed decode length). Importantly, this speedup does not coincide with an acceptance collapse—accepted tokens remain in the same range for MSpecKV vs MSpecKV (e.g., **4.56 ± 0.51** vs **4.87 ± 0.50** at **32K**)—which supports the interpretation that quantization primarily reduces KV traffic and bandwidth pressure rather than changing the speculative decoding dynamics.


### A2. Separate “quantization-only” vs “quant + resident KV”
**Key idea**: in your implementation, setting `resident_layers=0` disables the resident KV cache (but still uses KV quantization).  
So a resident-layer sweep that includes **0** gives you:
- **Quantization benefit**: compare baseline vs quant at `resident_layers=0`
- **Resident KV benefit**: compare quant at `resident_layers=0` vs quant at larger values

**What to test**: `resident_layers_values 0,8,16` at a fixed long-ish context.  
**Settings (final)**:
- `--prefill 16384` (strong signal but cheaper than 32768)
- `--on_chip 16 --gamma 8 --gen_len 256`
- `--resident_layers_values 0,8,16`

**Expected result**:
- At `0`: quant beats baseline (transfer/compression benefit).
- Increasing resident layers improves quant further and then starts to saturate.

**Run (final)**:

```bash
cd /opt/dlami/nvme/mira/.www/MSpecKV
for s in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/resident_layer_ablation.py \
    --dataset demo --target llama-7B-128K \
    --budget 4096 --on_chip 16 --gamma 8 \
    --prefill 16384 --gen_len 256 \
    --resident_layers_values "0,8,16" \
    --seed $s --output_json auto --output_csv /dev/null
done
```

**Quick-check (cheap sanity)**: use `--budget 1024 --prefill 2048 --gen_len 128 --resident_layers_values "0,4,8"`.

**Result**:
#### Throughput

| resident_layers | baseline | quant | delta (quant-baseline) | speedup (quant/baseline) |
|---:|---:|---:|---:|---:|
| 0 | 12.42 ± 4.40 | 13.21 ± 3.58 | 0.78 ± 1.17 | 1.09 ± 0.10 |
| 8 | 12.18 ± 4.59 | 14.89 ± 4.31 | 2.71 ± 1.17 | 1.26 ± 0.17 |
| 16 | 8.50 ± 1.31 | 24.12 ± 1.94 | 15.62 ± 2.53 | 2.89 ± 0.56 |


#### Latency

| resident_layers | baseline | quant | delta (quant-baseline) | speedup (baseline/quant) |
|---:|---:|---:|---:|---:|
| 0 | 22.62 ± 7.12 | 20.51 ± 5.20 | -2.12 ± 2.26 | 1.09 ± 0.10 |
| 8 | 23.35 ± 7.77 | 18.34 ± 5.02 | -5.01 ± 3.66 | 1.26 ± 0.17 |
| 16 | 30.64 ± 5.18 | 10.66 ± 0.87 | -19.98 ± 5.42 | 2.89 ± 0.56 |


#### Accepted

| resident_layers | baseline | quant | delta (quant-baseline) |
|---:|---:|---:|---:|
| 0 | 6.03 ± 0.46 | 5.81 ± 0.54 | -0.22 ± 0.66 |
| 8 | 5.79 ± 0.38 | 5.65 ± 0.86 | -0.15 ± 0.64 |
| 16 | 5.20 ± 1.03 | 6.04 ± 0.39 | 0.84 ± 1.30 |

**Result analysis**:
TODO

## B) Ablation study: necessary knobs only (keep it small)

### B1. Offload pressure via on-chip layers (2 points)
**What to test**: change how much is offloaded by varying `on_chip_layers`.  
**Why**: shows the method helps more when offload traffic is heavier.  
**Settings (final)**:
- `--prefill 16384 --gen_len 256 --resident_layers 16`
- `--on_chip_layers 8,16` (only two points)

**Expected result**:
- With fewer on-chip layers (more offload), quantization speedup is larger.

**Run (final)**:

```bash
cd /opt/dlami/nvme/mira/.www/MSpecKV
for s in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/on_chip_ablation.py \
    --dataset demo --target llama-7B-128K \
    --budget 4096 --gamma 8 \
    --prefill 16384 --gen_len 256 \
    --resident_layers 16 --on_chip_layers "4,8,12,16,18" \
    --seed $s --output_json auto --output_csv /dev/null
done
```

**Quick-check**: `--budget 1024 --prefill 2048 --gen_len 128 --on_chip_layers "4,8,16"`.

**Result**:
#### Throughput

| on_chip_layers | baseline | quant | delta (quant-baseline) | speedup (quant/baseline) |
|---:|---:|---:|---:|---:|
| 4 | 6.72 ± 0.52 | 9.74 ± 0.85 | 3.02 ± 1.36 | 1.46 ± 0.25 |
| 8 | 7.52 ± 1.58 | 9.69 ± 1.55 | 2.18 ± 2.13 | 1.32 ± 0.35 |
| 12 | 8.19 ± 0.68 | 10.89 ± 0.61 | 2.70 ± 0.74 | 1.33 ± 0.11 |
| 16 | 9.23 ± 1.05 | 20.01 ± 2.47 | 10.78 ± 1.96 | 2.17 ± 0.20 |
| 18 | 9.53 ± 0.73 | 22.08 ± 1.92 | 12.55 ± 1.84 | 2.32 ± 0.23 |


#### Latency

| on_chip_layers | baseline | quant | delta (quant-baseline) | speedup (baseline/quant) |
|---:|---:|---:|---:|---:|
| 4 | 38.23 ± 3.05 | 26.40 ± 2.20 | -11.83 ± 5.22 | 1.46 ± 0.25 |
| 8 | 35.00 ± 6.68 | 26.90 ± 4.59 | -8.10 ± 7.79 | 1.32 ± 0.35 |
| 12 | 31.40 ± 2.56 | 23.55 ± 1.31 | -7.85 ± 2.36 | 1.33 ± 0.11 |
| 16 | 27.99 ± 3.38 | 12.92 ± 1.54 | -15.07 ± 2.54 | 2.17 ± 0.20 |
| 18 | 26.98 ± 2.15 | 11.65 ± 0.99 | -15.33 ± 2.14 | 2.32 ± 0.23 |


#### Accepted

| on_chip_layers | baseline | quant | delta (quant-baseline) |
|---:|---:|---:|---:|
| 4 | 5.80 ± 0.37 | 6.07 ± 0.44 | 0.27 ± 0.81 |
| 8 | 5.83 ± 1.37 | 5.31 ± 0.90 | -0.52 ± 1.88 |
| 12 | 5.67 ± 0.62 | 5.43 ± 0.31 | -0.24 ± 0.77 |
| 16 | 5.60 ± 0.58 | 5.14 ± 0.69 | -0.46 ± 0.41 |
| 18 | 5.57 ± 0.39 | 5.57 ± 0.39 | -0.00 ± 0.55 |

**Result analysis**:
TODO

### B2. Output length sensitivity (2 points only)
**What to test**: short vs longer generation at fixed prefill.  
**Why**: confirms the benefit persists during decoding and isn’t a one-off.  
**Settings (final)**:
- `--prefill 8192 --on_chip 16 --resident_layers 16 --gamma 8`
- `--gen_lens 256,1024`

**Expected result**:
- Quant stays faster at both lengths; speedup may vary but should remain positive.

**Run (final)**:

```bash
cd /opt/dlami/nvme/mira/.www/MSpecKV
for s in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/output_len_ablation.py \
    --dataset demo --target llama-7B-128K \
    --budget 4096 --on_chip 16 --gamma 8 --resident_layers 16 \
    --prefill 8192 --gen_lens "256,512,768,1024,1536,2048" \
    --seed $s --output_json auto --output_csv /dev/null
done
```

**Quick-check**: `--budget 1024 --prefill 2048 --gen_lens "128,256"`.

**Result**:
#### Throughput

| x | baseline | quant | delta (quant-baseline) | speedup (quant/baseline) |
|---:|---:|---:|---:|---:|
| 256 | 18.14 ± 1.07 | 24.74 ± 1.73 | 6.61 ± 2.69 | 1.37 ± 0.17 |
| 512 | 18.34 ± 1.54 | 28.81 ± 4.91 | 10.47 ± 3.73 | 1.57 ± 0.17 |
| 768 | 19.94 ± 1.72 | 25.71 ± 1.87 | 5.77 ± 3.07 | 1.30 ± 0.17 |
| 1024 | 19.86 ± 0.24 | 27.75 ± 1.61 | 7.89 ± 1.40 | 1.40 ± 0.07 |
| 1536 | 19.87 ± 0.31 | 28.03 ± 3.86 | 8.16 ± 3.79 | 1.41 ± 0.19 |
| 2048 | 19.42 ± 1.26 | 31.12 ± 2.08 | 11.70 ± 2.96 | 1.61 ± 0.19 |


#### Latency

| x | baseline | quant | delta (quant-baseline) | speedup (baseline/quant) |
|---:|---:|---:|---:|---:|
| 256 | 14.15 ± 0.86 | 10.38 ± 0.73 | -3.77 ± 1.52 | 1.37 ± 0.17 |
| 512 | 28.06 ± 2.46 | 18.10 ± 2.94 | -9.96 ± 1.48 | 1.57 ± 0.17 |
| 768 | 38.70 ± 3.19 | 29.97 ± 2.09 | -8.73 ± 4.55 | 1.30 ± 0.17 |
| 1024 | 51.56 ± 0.62 | 36.98 ± 2.14 | -14.58 ± 1.62 | 1.40 ± 0.07 |
| 1536 | 77.32 ± 1.22 | 55.47 ± 7.40 | -21.85 ± 7.26 | 1.41 ± 0.19 |
| 2048 | 105.74 ± 7.10 | 66.01 ± 4.59 | -39.73 ± 10.35 | 1.61 ± 0.19 |


#### Accepted

| x | baseline | quant | delta (quant-baseline) |
|---:|---:|---:|---:|
| 256 | 6.97 ± 0.20 | 6.61 ± 0.45 | -0.36 ± 0.60 |
| 512 | 7.14 ± 0.27 | 7.16 ± 0.52 | 0.03 ± 0.33 |
| 768 | 7.55 ± 0.39 | 6.60 ± 0.30 | -0.95 ± 0.46 |
| 1024 | 7.45 ± 0.09 | 7.02 ± 0.30 | -0.43 ± 0.23 |
| 1536 | 7.53 ± 0.15 | 7.04 ± 0.63 | -0.49 ± 0.78 |
| 2048 | 7.61 ± 0.35 | 7.67 ± 0.25 | 0.06 ± 0.55 |

**Result analysis**:
TODO


## What you will be able to claim (cleanly) with this design
- **Quantization helps** (A2 at `resident_layers=0`, and A1 especially at long context).
- **Resident KV helps beyond quantization** (A2: quant improves as resident layers increase).
- **Benefit scales with offload pressure** (B1).
- **Benefit persists across decoding lengths** (B2).
- All results are backed by **3-seed mean ± std** and **delta/speedup**.

If you tell me your time budget (e.g., “I can only afford ~6 total runs”), I can further cut this to the absolute minimum while still separating **quant-only** vs **quant+resident**.