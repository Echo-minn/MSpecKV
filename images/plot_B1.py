import matplotlib.pyplot as plt
import numpy as np

# Data
on_chip_layers = np.array([4, 8, 12, 16, 18])
baseline = np.array([6.72, 7.52, 8.19, 9.23, 9.53])
quant = np.array([9.74, 9.69, 10.89, 20.01, 22.08])
speedup = np.array([1.46, 1.32, 1.33, 2.17, 2.32])

x = np.arange(len(on_chip_layers))
width = 0.35

# Plot
plt.figure()
ax1 = plt.gca()

# Bar plots for throughput
ax1.bar(x - width/2, baseline, width, label="w/o KV Quant tok/s", color='#B1D195')
ax1.bar(x + width/2, quant, width, label="w/ KV Quant tok/s", color='#C9D7EF')
ax1.set_xlabel("On-chip Layers")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_xticks(x)
ax1.set_xticklabels(on_chip_layers)
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

plt.title("Throughput and Speedup vs On-chip Layers")
plt.tight_layout()

plt.savefig("images/B1.png")