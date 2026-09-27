import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# Load metrics
runs = {}
for name, p in [("Full FT", "results/full_ft/metrics.json"),
                ("LoRA", "results/lora/metrics.json"),
                ("QLoRA", "results/qlora/metrics.json")]:
    with open(p, "r") as f:
        runs[name] = json.load(f)

methods = list(runs.keys())
colors = ["#4C72B0", "#55A868", "#C44E52"]

# 1. Parameter Comparison (Log scale)
plt.figure(figsize=(7, 4.5))
params = [runs[m]["trainable_params"] for m in methods]
bars = plt.bar(methods, params, color=colors, edgecolor='black', alpha=0.85)
plt.yscale("log")
plt.ylabel("Trainable Parameters (Log Scale)")
plt.title("Trainable Parameters by Fine-Tuning Strategy")
for bar, p in zip(bars, params):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f"{p:,}", ha='center', va='bottom', fontweight='bold')
plt.ylim(10**5, 10**8 * 2)
plt.tight_layout()
plt.savefig("figures/fig1_parameter_comparison.png", dpi=300)
plt.close()

# 2. Loss Curves
plt.figure(figsize=(8, 4.5))
for idx, (m, color) in enumerate(zip(methods, colors)):
    history = runs[m]["loss_history"]
    steps = [x["step"] for x in history]
    losses = [x["loss"] for x in history]
    plt.plot(steps, losses, label=m, color=color, linewidth=2.2, marker='o' if idx > 0 else None, markersize=4)
plt.xlabel("Optimization Steps")
plt.ylabel("Training Loss")
plt.title("Training Loss Trajectory Across Steps")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig2_loss_curves.png", dpi=300)
plt.close()

# 3. Peak VRAM & Training Time Dual Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
vram = [runs[m]["peak_memory_mb"] for m in methods]
ax1.bar(methods, vram, color=colors, edgecolor='black', alpha=0.85)
ax1.set_ylabel("Peak VRAM (MB)")
ax1.set_title("Peak GPU Memory Allocated")
for idx, val in enumerate(vram):
    ax1.text(idx, val + 35, f"{val:.1f} MB", ha='center', fontweight='bold')

times = [runs[m]["training_time_sec"] for m in methods]
ax2.bar(methods, times, color=colors, edgecolor='black', alpha=0.85)
ax2.set_ylabel("Wall-Clock Time (s)")
ax2.set_title("Total Training Duration")
for idx, val in enumerate(times):
    ax2.text(idx, val + 1.5, f"{val:.1f} s", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("figures/fig3_hardware_efficiency.png", dpi=300)
plt.close()

# 4. Accuracy vs. GPU Memory Trade-off Scatter Plot
plt.figure(figsize=(7, 5))
accs = [runs[m]["test_accuracy"] * 100 for m in methods]
for i, m in enumerate(methods):
    plt.scatter(vram[i], accs[i], color=colors[i], s=250, edgecolor='black', zorder=5)
    plt.annotate(
        f"{m}\n({accs[i]:.2f}%, {vram[i]:.0f}MB)",
        (vram[i], accs[i]),
        textcoords="offset points",
        xytext=(0, 12),
        ha='center',
        fontweight='bold'
    )
plt.xlabel("Peak GPU VRAM (MB)")
plt.ylabel("Test Accuracy (%)")
plt.title("Performance vs. Resource Consumption Trade-Off")
plt.xlim(min(vram) - 150, max(vram) + 200)
plt.ylim(min(accs) - 1.0, max(accs) + 1.5)
plt.tight_layout()
plt.savefig("figures/fig4_accuracy_vs_memory.png", dpi=300)
plt.close()

print("All 4 figures successfully generated and saved to 'figures/' directory.")
