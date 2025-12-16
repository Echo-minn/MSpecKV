import matplotlib.pyplot as plt
import numpy as np

# Data
prefill = np.array([8192, 16384, 24576, 32768])

baseline = np.array([4.72, 2.58, 1.73, 1.31])
vanilla = np.array([5.56, 2.83, 2.09, 1.57])
triforce = np.array([14.61, 9.86, 6.55, 4.86])
mspeckv = np.array([22.86, 21.54, 21.37, 18.50])

speedup_vanilla = vanilla / baseline
speedup_triforce = triforce / baseline
speedup_mspeckv = mspeckv / baseline

x = np.arange(len(prefill))
width = 0.22

plt.figure(figsize=(8, 5))
ax1 = plt.gca()

# Bars: throughput
ax1.bar(x - width, vanilla, width, label="Vanilla SpecDec", color='#F5D6B8')
ax1.bar(x, triforce, width, label="w/o KV Quant", color='#B1D195')
ax1.bar(x + width, mspeckv, width, label="w/ KV Quant", color='#C9D7EF')

ax1.set_xlabel("Prefill Length")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_xticks(x)
ax1.set_xticklabels(prefill)
ax1.set_ylim(0, 30)
ax1.legend(loc="upper left")

# Lines: speedup
ax2 = ax1.twinx()
ax2.plot(x, speedup_vanilla, marker="o", linestyle="--", label="Speedup: Vanilla/Baseline", color='orange')
ax2.plot(x, speedup_triforce, marker="o", linestyle="-.", label="Speedup: w/o KV Quant/Baseline", color='green')
ax2.plot(x, speedup_mspeckv, marker="o", linestyle="-", label="Speedup: w/ KV Quant/Baseline", color='blue')
ax2.set_ylabel("Speedup vs Baseline")
ax2.axhline(1.0, linestyle="--", linewidth=1, label="Speedup = 1", color='darkred')

ax2.set_ylim(0, 18)
ax2.legend(loc="upper right")

plt.title("Throughput (Bars) and Speedup (Lines) vs Prefill Length")
plt.tight_layout()
plt.savefig("images/A1.png")