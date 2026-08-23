#!/usr/bin/env python3
"""Step 4b - Readable version of the cost curve."""

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "artifacts"

sweeps = {n: pd.read_csv(f"{OUT}/threshold_sweep_{n}.csv")
          for n in ["conservative", "moderate", "aggressive"]}
summary = json.load(open(f"{OUT}/cost_summary.json"))["results"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# --- LEFT: zoomed to where the decision actually happens
ax = axes[0]
colors = {"conservative": "#2563eb", "moderate": "#16a34a", "aggressive": "#9333ea"}

for name, frame in sweeps.items():
    ax.plot(frame["threshold"], frame["savings"] / 1e5,
            label=f"{name} (opt {summary[name]['money_optimal_threshold']:.2f})",
            linewidth=2.2, color=colors[name])
    best = frame.loc[frame["total_cost"].idxmin()]
    ax.plot(best["threshold"], best["savings"] / 1e5, "o",
            markersize=10, color=colors[name], markeredgecolor="white",
            markeredgewidth=1.5, zorder=5)

f1_thr = summary["moderate"]["f1_optimal_threshold"]
ax.axvline(f1_thr, linestyle="--", color="crimson", linewidth=2,
           label=f"F1-optimal ({f1_thr:.2f}) - same for all three")

ax.set_xlim(0.55, 1.0)
ax.set_ylim(0, 210)
ax.set_xlabel("Decision threshold", fontsize=11)
ax.set_ylabel("Net saved vs no detector (Rs. lakh)", fontsize=11)
ax.set_title("Money-optimal threshold moves with merchant economics.\n"
             "F1-optimal does not.", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.25)

# --- RIGHT: conservative scenario, where the gap is largest
ax = axes[1]
cons = sweeps["conservative"]

ax.plot(cons["threshold"], cons["fn_loss"] / 1e5,
        label="Fraud let through", linewidth=2.2, color="#dc2626")
ax.plot(cons["threshold"], cons["fp_loss"] / 1e5,
        label="Good customers blocked", linewidth=2.2, color="#f59e0b")
ax.plot(cons["threshold"], cons["total_cost"] / 1e5,
        label="Total cost", linewidth=3, color="black")

m_thr = summary["conservative"]["money_optimal_threshold"]
penalty = summary["conservative"]["penalty_of_using_f1"]

ax.axvline(m_thr, linestyle="--", color="#16a34a", linewidth=2,
           label=f"Money-optimal ({m_thr:.2f})")
ax.axvline(f1_thr, linestyle="--", color="crimson", linewidth=2,
           label=f"F1-optimal ({f1_thr:.2f})")
ax.axvspan(m_thr, f1_thr, alpha=0.15, color="crimson")

ax.annotate(f"Rs.{penalty/1e5:.1f} lakh\nlost in this gap",
            xy=((m_thr + f1_thr) / 2, 180), ha="center", fontsize=10,
            fontweight="bold", color="crimson")

ax.set_xlim(0.4, 1.0)
ax.set_ylim(0, 520)
ax.set_xlabel("Decision threshold", fontsize=11)
ax.set_ylabel("Cost (Rs. lakh)", fontsize=11)
ax.set_title("Conservative merchant: the two error types trade off,\n"
             "and F1 picks the wrong side", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="center right")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/cost_curve.png", dpi=160, bbox_inches="tight")
print(f"Saved: {OUT}/cost_curve.png")
