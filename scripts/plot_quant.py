#!/usr/bin/env python3
"""
Generate all figures for the quantization experiments repo.
Run from the repo root: python scripts/plot_quant.py
Saves PNG files to figures/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":     True,
    "grid.alpha":    0.3,
    "grid.linestyle":"--",
})

# ── Experiment A — QAT steps vs PPL delta ────────────────────────────────────

steps  = [0, 500, 1000, 3000, 5000]
deltas = [0.175, 0.226, -0.393, 0.364, 1.004]
labels = ["PTQ\n(0 steps)", "500", "1000", "3000", "5000"]
colors = ["#B4B2A9", "#B4B2A9", "#1D9E75", "#D85A30", "#993C1D"]

fig, ax = plt.subplots(figsize=(8, 4.5))

bars = ax.bar(labels, deltas, color=colors, width=0.55, zorder=3)
ax.axhline(0, color="black", linewidth=0.8, linestyle="-", zorder=2)

for bar, delta in zip(bars, deltas):
    va = "bottom" if delta >= 0 else "top"
    offset = 0.02 if delta >= 0 else -0.02
    ax.text(bar.get_x() + bar.get_width() / 2,
            delta + offset,
            f"{delta:+.3f}%",
            ha="center", va=va, fontsize=10, fontweight="500")

ax.set_xlabel("QAT fine-tuning steps")
ax.set_ylabel("PPL delta from float32 baseline")
ax.set_title("Experiment A — QAT steps vs perplexity delta", fontweight="500")
ax.set_ylim(min(deltas) - 0.25, max(deltas) + 0.35)

legend_handles = [
    mpatches.Patch(color="#1D9E75", label="best result (1000 steps)"),
    mpatches.Patch(color="#D85A30", label="overfitting begins"),
    mpatches.Patch(color="#B4B2A9", label="PTQ / early QAT"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig("figures/exp_a_steps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/exp_a_steps.png")

# ── Experiment B — model size comparison ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models   = ["small\n(nl=6, 10.65M)", "large\n(nl=12, ~21M)"]
baseline = [4.3073, 4.3518]
ptq_ppl  = [4.3222, 4.3386]
qat_ppl  = [4.3113, 4.3886]

x = np.arange(len(models))
w = 0.25

ax = axes[0]
b1 = ax.bar(x - w, baseline, w, label="float32 baseline", color="#888780", zorder=3)
b2 = ax.bar(x,     ptq_ppl,  w, label="PTQ int8",         color="#378ADD", zorder=3)
b3 = ax.bar(x + w, qat_ppl,  w, label="QAT int8",         color="#1D9E75", zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Perplexity")
ax.set_title("Absolute perplexity by model size", fontweight="500")
ax.set_ylim(4.28, 4.42)
ax.legend(fontsize=9)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=8)

ptq_deltas = [0.348, -0.302]
qat_deltas = [0.093,  0.847]

ax2 = axes[1]
b4 = ax2.bar(x - w/2, ptq_deltas, w, label="PTQ int8",  color="#378ADD", zorder=3)
b5 = ax2.bar(x + w/2, qat_deltas, w, label="QAT int8",  color="#1D9E75", zorder=3)
ax2.axhline(0, color="black", linewidth=0.8, zorder=2)

ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel("PPL delta from float32 baseline (%)")
ax2.set_title("PPL delta by model size", fontweight="500")
ax2.legend(fontsize=9)

for bars in [b4, b5]:
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.01 if h >= 0 else -0.01
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 h + offset,
                 f"{h:+.3f}%",
                 ha="center", va=va, fontsize=9, fontweight="500")

ax2.set_ylim(min(ptq_deltas + qat_deltas) - 0.15,
             max(ptq_deltas + qat_deltas) + 0.2)

plt.suptitle("Experiment B — quantization sensitivity: small vs large model",
             fontweight="500", fontsize=12)
plt.tight_layout()
plt.savefig("figures/exp_b_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/exp_b_models.png")

print("\nAll figures saved to figures/")
print("Next: git add figures/ && git commit -m 'add result figures'")
