import matplotlib.pyplot as plt
import numpy as np

# Data
gen_lens = np.array([256, 512, 768, 1024, 1536, 2048])
baseline = np.array([18.14, 18.34, 19.94, 19.86, 19.87, 19.42])
quant = np.array([24.74, 28.81, 25.71, 27.75, 28.03, 31.12])
speedup = np.array([1.37, 1.57, 1.30, 1.40, 1.41, 1.61])

x = np.arange(len(gen_lens))
width = 0.35

# Plot
plt.figure()
ax1 = plt.gca()

# Bar plots for throughput
ax1.bar(x - width/2, baseline, width, label="w/o KV Quant tok/s", color='#B1D195')
ax1.bar(x + width/2, quant, width, label="w/ KV Quant tok/s", color='#C9D7EF')
ax1.set_xlabel("Generation Length")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_xticks(x)
ax1.set_xticklabels(gen_lens)
ax1.set_ylim(0, 30)
ax1.legend(loc="upper left")

# Line plot for speedup
ax2 = ax1.twinx()
ax2.plot(x, speedup, marker="o", label="Speedup (w/ KV Quant/w/o KV Quant)", color='green')
ax2.set_ylabel("Speedup")
ax2.set_ylim(0, 4)
ax2.legend(loc="upper right")

# add speedup = 1 as a horizontal line
ax2.axhline(y=1, color='darkred', linestyle='--', label="Speedup = 1")

plt.title("Throughput and Speedup vs Generation Length")
plt.tight_layout()

plt.savefig("images/B2.png")