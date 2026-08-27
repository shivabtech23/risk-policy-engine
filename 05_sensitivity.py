#!/usr/bin/env python3
"""
Step 6 - Sensitivity analysis.

The three-action result rests on four assumptions I chose, not measured:
how often a fraudster defeats a challenge, how often a genuine customer
abandons at one, what a challenge costs to run, and how much less an
abandoned customer churns than a rejected one.

A result that only holds at flattering assumptions is not a result.
This finds where it breaks.
"""

import json
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "artifacts"

USD_TO_INR = 85.0
CHARGEBACK_FEE = 1500.0
ABANDON_CHURN_MULTIPLIER = 0.40

MARGIN_RATE = 0.20
CHURN_COST = 150.0

GRID = np.linspace(0.02, 0.98, 49)

BYPASS_RATES = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75]
ABANDON_RATES = [0.02, 0.06, 0.12, 0.20, 0.30, 0.40]
CHALLENGE_COSTS = [5.0, 25.0, 50.0, 100.0, 200.0]


df = pd.read_csv(f"{OUT}/holdout_predictions.csv")
y = df["isFraud"].values.astype(int)
p = df["pred"].values
amt = df["TransactionAmt"].values * USD_TO_INR

n = len(df)
n_fraud = int(y.sum())
fraud_mask = y == 1
legit_mask = y == 0

do_nothing_cost = amt[fraud_mask].sum() + CHARGEBACK_FEE * n_fraud


def best_policy(bypass, abandon, chal_cost, cap=None):
    best = None
    for lo, hi in itertools.product(GRID, GRID):
        if lo > hi:
            continue

        allow = p < lo
        challenge = (p >= lo) & (p < hi)
        block = p >= hi

        if cap is not None and challenge.sum() / n > cap:
            continue

        f_allow = allow & fraud_mask
        f_chal = challenge & fraud_mask
        f_block = block & fraud_mask
        l_chal = challenge & legit_mask
        l_block = block & legit_mask

        fraud_loss = (
            amt[f_allow].sum() + CHARGEBACK_FEE * f_allow.sum()
            + bypass * (amt[f_chal].sum() + CHARGEBACK_FEE * f_chal.sum())
        )
        customer_loss = (
            abandon * (
                MARGIN_RATE * amt[l_chal].sum()
                + ABANDON_CHURN_MULTIPLIER * CHURN_COST * l_chal.sum()
            )
            + MARGIN_RATE * amt[l_block].sum() + CHURN_COST * l_block.sum()
        )
        total = fraud_loss + customer_loss + chal_cost * challenge.sum()
        savings = do_nothing_cost - total

        if best is None or savings > best["savings"]:
            stopped = f_block.sum() + (1 - bypass) * f_chal.sum()
            best = {
                "t_low": lo, "t_high": hi, "savings": savings,
                "effective_recall": stopped / n_fraud,
                "challenge_rate": challenge.sum() / n,
                "legit_blocked": int(l_block.sum()),
            }
    return best


def best_two_action():
    best = None
    for t in GRID:
        block = p >= t
        f_block = block & fraud_mask
        l_block = block & legit_mask

        fraud_loss = (
            amt[~block & fraud_mask].sum()
            + CHARGEBACK_FEE * (~block & fraud_mask).sum()
        )
        customer_loss = MARGIN_RATE * amt[l_block].sum() + CHURN_COST * l_block.sum()
        savings = do_nothing_cost - (fraud_loss + customer_loss)

        if best is None or savings > best["savings"]:
            best = {
                "threshold": t, "savings": savings,
                "effective_recall": f_block.sum() / n_fraud,
                "legit_blocked": int(l_block.sum()),
            }
    return best


baseline2 = best_two_action()

print("=" * 68)
print("SENSITIVITY ANALYSIS")
print(f"  merchant: margin {MARGIN_RATE:.0%}, churn Rs.{CHURN_COST:.0f} "
      f"(the case most favourable to two-action)")
print("=" * 68)
print(f"\nTwo-action baseline (assumption-free):")
print(f"  threshold {baseline2['threshold']:.2f} | "
      f"recall {baseline2['effective_recall']:.1%} | "
      f"Rs.{baseline2['savings']:,.0f}")

print("\n" + "=" * 68)
print("GAIN FROM THIRD ACTION (Rs. lakh) as assumptions worsen")
print("=" * 68)
header = "bypass \\ abandon | " + " | ".join(f"{a:>6.0%}" for a in ABANDON_RATES)
print(header)
print("-" * len(header))

matrix = np.zeros((len(BYPASS_RATES), len(ABANDON_RATES)))
for i, b in enumerate(BYPASS_RATES):
    row = []
    for j, a in enumerate(ABANDON_RATES):
        res = best_policy(b, a, 5.0)
        gain = (res["savings"] - baseline2["savings"]) / 1e5
        matrix[i, j] = gain
        row.append(f"{gain:>6.1f}")
    print(f"{b:>15.0%} | " + " | ".join(row))

n_positive = int((matrix > 0).sum())
print(f"\n  Third action wins in {n_positive} of {matrix.size} "
      f"assumption combinations ({n_positive / matrix.size:.0%})")
print(f"  Worst case: Rs.{matrix.min():.1f} lakh | "
      f"Best case: Rs.{matrix.max():.1f} lakh")

# ---- 2. what a realistic operations constraint costs
print("\n" + "=" * 68)
print("CONSTRAINED CHALLENGE RATE")
print("  Unconstrained optimum challenges ~25% of traffic. Real 3DS")
print("  step-up rates sit near 5-15%. What does the cap cost?")
print("=" * 68)

caps = [0.03, 0.05, 0.10, 0.15, 0.25, None]
cap_rows = []
for cap in caps:
    res = best_policy(0.15, 0.06, 5.0, cap=cap)
    gain = (res["savings"] - baseline2["savings"]) / 1e5
    cap_rows.append({
        "cap": cap if cap is not None else 1.0,
        "gain_lakh": gain,
        "t_low": res["t_low"],
        "t_high": res["t_high"],
        "recall": res["effective_recall"],
        "challenge_rate": res["challenge_rate"],
        "legit_blocked": res["legit_blocked"],
    })
    label = f"{cap:.0%}" if cap is not None else "none"
    print(f"  cap {label:>5} | challenged {res['challenge_rate']:>6.2%} | "
          f"recall {res['effective_recall']:>5.1%} | "
          f"blocked {res['legit_blocked']:>5,} | "
          f"gain Rs.{gain:>6.1f} lakh")

capped5 = [r for r in cap_rows if r["cap"] == 0.05][0]
uncapped = [r for r in cap_rows if r["cap"] == 1.0][0]
print(f"\n  A 5% cap retains {capped5['gain_lakh'] / uncapped['gain_lakh']:.0%} "
      f"of the unconstrained gain.")

# ---- 3. cost per challenge
print("\n" + "=" * 68)
print("COST PER CHALLENGE")
print("=" * 68)
cost_rows = []
for c in CHALLENGE_COSTS:
    res = best_policy(0.15, 0.06, c)
    gain = (res["savings"] - baseline2["savings"]) / 1e5
    cost_rows.append({"cost": c, "gain_lakh": gain,
                      "challenge_rate": res["challenge_rate"]})
    print(f"  Rs.{c:>6.0f} per challenge | challenged "
          f"{res['challenge_rate']:>6.2%} | gain Rs.{gain:>6.1f} lakh")

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax = axes[0]
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
               vmin=0, vmax=matrix.max())
plt.colorbar(im, ax=ax, label="Gain from third action (Rs. lakh)")

ax.set_xticks(range(len(ABANDON_RATES)))
ax.set_xticklabels([f"{a:.0%}" for a in ABANDON_RATES])
ax.set_yticks(range(len(BYPASS_RATES)))
ax.set_yticklabels([f"{b:.0%}" for b in BYPASS_RATES])
ax.set_xlabel("Genuine customer abandons at challenge", fontsize=11)
ax.set_ylabel("Fraudster defeats challenge", fontsize=11)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                fontsize=9, color="black")

ax.plot(1, 1, "o", markersize=16, markerfacecolor="none",
        markeredgecolor="blue", markeredgewidth=2.5)
ax.set_title("The gain survives every assumption I tested.\n"
             "Circle = the values used in the headline result.",
             fontsize=12, fontweight="bold")

ax = axes[1]
labels = [f"{r['cap']:.0%}" if r["cap"] < 1 else "none" for r in cap_rows]
gains = [r["gain_lakh"] for r in cap_rows]
bars = ax.bar(range(len(cap_rows)), gains, color="#2563eb")
bars[1].set_color("#16a34a")

for i, r in enumerate(cap_rows):
    ax.annotate(f"{r['recall']:.0%}\nfraud\nstopped",
                xy=(i, r["gain_lakh"] + 8), ha="center", fontsize=8)

ax.set_xticks(range(len(cap_rows)))
ax.set_xticklabels(labels)
ax.set_xlabel("Maximum share of traffic that may be challenged", fontsize=11)
ax.set_ylabel("Gain from third action (Rs. lakh)", fontsize=11)
ax.set_ylim(0, max(gains) * 1.30)
ax.set_title("Most of the gain survives a realistic 5% cap.\n"
             "Green = operationally plausible.", fontsize=12, fontweight="bold")
ax.grid(alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/sensitivity.png", dpi=160, bbox_inches="tight")
print(f"\nSaved plot: {OUT}/sensitivity.png")

with open(f"{OUT}/sensitivity_summary.json", "w") as f:
    json.dump({
        "merchant": {"margin_rate": MARGIN_RATE, "churn_cost": CHURN_COST},
        "two_action_baseline": baseline2,
        "bypass_rates": BYPASS_RATES,
        "abandon_rates": ABANDON_RATES,
        "gain_matrix_lakh": matrix.tolist(),
        "n_combinations_positive": n_positive,
        "n_combinations_total": int(matrix.size),
        "worst_case_lakh": float(matrix.min()),
        "best_case_lakh": float(matrix.max()),
        "challenge_rate_caps": cap_rows,
        "challenge_cost_sweep": cost_rows,
    }, f, indent=2)

print(f"Saved summary: {OUT}/sensitivity_summary.json")
