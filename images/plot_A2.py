import matplotlib.pyplot as plt
import numpy as np

# Data
resident_layers = np.array([0, 8, 16])
baseline = np.array([12.42, 12.18, 8.50])
quant = np.array([13.21, 14.89, 24.12])
speedup = np.array([1.09, 1.26, 2.89])

x = np.arange(len(resident_layers))
width = 0.35

# Plot
plt.figure()
ax1 = plt.gca()

# Bar plots for throughput
ax1.bar(x - width/2, baseline, width, label="w/o KV Quant tok/s", color='#B1D195')
ax1.bar(x + width/2, quant, width, label="w/ KV Quant tok/s", color='#C9D7EF')
ax1.set_xlabel("Resident Layers")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_xticks(x)
ax1.set_xticklabels(resident_layers)
ax1.set_ylim(0, 30)
ax1.legend(loc="upper left")

# Line plot for speedup
ax2 = ax1.twinx()
ax2.plot(x, speedup, marker="o", label="Speedup (w/ KV Quant/w/o KV Quant)", color='green')
ax2.set_ylabel("Speedup")
ax2.set_ylim(0, 4)
ax2.legend(loc="upper right")

# add speedup = 1 as a horizontal line
ax2.axhline(1.0, linestyle="--", linewidth=1, label="Speedup = 1", color='darkred')

plt.title("Speedup on Resident Layers")
plt.tight_layout()

plt.savefig("images/A2.png")