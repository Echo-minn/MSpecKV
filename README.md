# MSpecKV

Long-context inference in large language models is increasingly bottlenecked by key–value (KV) cache traffic,
including offloading, transfer, and memory bandwidth costs, which scale linearly with context length and can
dominate computation. While speculative decoding techniques reduce redundant computation, they retain full-
precision KV caches and therefore fail to address this memory bottleneck. We present MSpecKV, a plug-and-play
multilevel speculative decoding framework that augmented with KV-cache quantization and GPU-resident KV
reuse to reduce KV movement without modifying attention kernels or retraining models. MSpecKV stores
offloaded KV in quantized form and reconstructs only the necessary subsets on demand, while maintaining high
speculative acceptance. Across prefill lengths from 8K to 32K, MSpecKV improves throughput by up to 14.1×
over naive autoregressive decoding and up to 3.8× over non-quantized variant, with comparable acceptance rates.
These results demonstrate that reducing KV-cache traffic is critical for scalable long-context serving, and that
quantization combined with structured speculative decoding provides a practical and effective solution. 

## Recorded Video
The recorded video is available at: https://youtu.be/fJIsocqpJF4

## Evaluation Results(A1)

```shell
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

## Resident Layer Abalation(A2)
**Run (final)**:

```bash
cd /opt/dlami/nvme/mira/.www/TriForce
for s in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/resident_layer_ablation.py \
    --dataset demo --target llama-7B-128K \
    --budget 4096 --on_chip 16 --gamma 8 \
    --prefill 16384 --gen_len 256 \
    --resident_layers_values "0,8,16" \
    --seed $s --output_json auto --output_csv /dev/null
done
```

**Throughput**

| resident_layers | baseline | quant | delta | speedup |
|---:|---:|---:|---:|---:|
| 0 | 12.42 ± 4.40 | 13.21 ± 3.58 | 0.78 ± 1.17 | 1.09 ± 0.10 |
| 8 | 12.18 ± 4.59 | 14.89 ± 4.31 | 2.71 ± 1.17 | 1.26 ± 0.17 |
| 16 | 8.50 ± 1.31 | 24.12 ± 1.94 | 15.62 ± 2.53 | 2.89 ± 0.56 |

## On-Chip Layer Abalation(B1)
**Run**:

```bash
cd /opt/dlami/nvme/mira/.www/TriForce
for s in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/on_chip_ablation.py \
    --dataset demo --target llama-7B-128K \
    --budget 4096 --gamma 8 \
    --prefill 16384 --gen_len 256 \
    --resident_layers 16 --on_chip_layers "4,8,12,16,18" \
    --seed $s --output_json auto --output_csv /dev/null
done
```

**Throughput**

| on_chip_layers | baseline | quant | delta | speedup |
|---:|---:|---:|---:|---:|
| 4 | 6.72 ± 0.52 | 9.74 ± 0.85 | 3.02 ± 1.36 | 1.46 ± 0.25 |
| 8 | 7.52 ± 1.58 | 9.69 ± 1.55 | 2.18 ± 2.13 | 1.32 ± 0.35 |
| 12 | 8.19 ± 0.68 | 10.89 ± 0.61 | 2.70 ± 0.74 | 1.33 ± 0.11 |
| 16 | 9.23 ± 1.05 | 20.01 ± 2.47 | 10.78 ± 1.96 | 2.17 ± 0.20 |
| 18 | 9.53 ± 0.73 | 22.08 ± 1.92 | 12.55 ± 1.84 | 2.32 ± 0.23 |

## Output Length Sensitivity(B2)
**Run**:

```bash
cd /opt/dlami/nvme/mira/.www/TriForce
for s in 1 2 3; do
  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 evaluation/output_len_ablation.py \
    --dataset demo --target llama-7B-128K \
    --budget 4096 --on_chip 16 --gamma 8 --resident_layers 16 \
    --prefill 8192 --gen_lens "256,512,768,1024,1536,2048" \
    --seed $s --output_json auto --output_csv /dev/null
done
```

**Throughput**

| x | baseline | quant | delta | speedup |
|---:|---:|---:|---:|---:|
| 256 | 18.14 ± 1.07 | 24.74 ± 1.73 | 6.61 ± 2.69 | 1.37 ± 0.17 |
| 512 | 18.34 ± 1.54 | 28.81 ± 4.91 | 10.47 ± 3.73 | 1.57 ± 0.17 |
| 768 | 19.94 ± 1.72 | 25.71 ± 1.87 | 5.77 ± 3.07 | 1.30 ± 0.17 |
| 1024 | 19.86 ± 0.24 | 27.75 ± 1.61 | 7.89 ± 1.40 | 1.40 ± 0.07 |
| 1536 | 19.87 ± 0.31 | 28.03 ± 3.86 | 8.16 ± 3.79 | 1.41 ± 0.19 |
| 2048 | 19.42 ± 1.26 | 31.12 ± 2.08 | 11.70 ± 2.96 | 1.61 ± 0.19 |
