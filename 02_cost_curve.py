#!/usr/bin/env python3
"""
Step 4 - The cost curve.

Takes the held-out predictions and asks a different question than F1 does:
at each possible decision threshold, how many rupees does the merchant
actually keep? Then finds where the F1-optimal and margin-optimal
thresholds diverge, and prices that gap.

No retraining. Pure arithmetic over artifacts/holdout_predictions.csv.
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "artifacts"

# IEEE-CIS amounts are USD (mean ~$135). Converting so the rupee figures
# describe a plausible Indian merchant rather than implausible Rs.135 orders.
USD_TO_INR = 85.0

# Chargeback penalty per fraud event. Razorpay's own published range is
# Rs.200-600 in direct dispute fees; their enterprise guide cites global
# benchmarks of Rs.1,500-8,000 all-in. Rs.1,500 sits at the conservative
# end of the all-in figure.
CHARGEBACK_FEE = 1500.0

# False-positive cost = lost gross margin + expected value of the customer
# who never returns. Three scenarios, because this number is genuinely
# uncertain and the whole point is that the answer moves with it.
SCENARIOS = {
    "conservative": {"margin_rate": 0.20, "churn_cost": 150.0},
    "moderate":     {"margin_rate": 0.30, "churn_cost": 400.0},
    "aggressive":   {"margin_rate": 0.40, "churn_cost": 900.0},
}

THRESHOLDS = np.linspace(0.001, 0.999, 400)


# ------------------------------------------------------------------ load
df = pd.read_csv(f"{OUT}/holdout_predictions.csv")

y = df["isFraud"].values.astype(int)
p = df["pred"].values
amt = df["TransactionAmt"].values * USD_TO_INR

n = len(df)
n_fraud = int(y.sum())

print("=" * 60)
print("HELD-OUT SET")
print("=" * 60)
print(f"  transactions   : {n:,}")
print(f"  fraud          : {n_fraud:,} ({y.mean():.3%})")
print(f"  total value    : Rs.{amt.sum():,.0f}")
print(f"  fraud value    : Rs.{amt[y == 1].sum():,.0f}")
print(f"  mean order     : Rs.{amt.mean():,.0f}")


def evaluate(threshold, margin_rate, churn_cost):
    """Cost in rupees of running the detector at this threshold.

    Only two things cost money: fraud you let through, and good
    customers you blocked. Correct decisions cost nothing.
    """
    blocked = p >= threshold

    tp = int((blocked & (y == 1)).sum())
    fp = int((blocked & (y == 0)).sum())
    fn = int((~blocked & (y == 1)).sum())
    tn = int((~blocked & (y == 0)).sum())

    fn_loss = amt[~blocked & (y == 1)].sum() + CHARGEBACK_FEE * fn
    fp_loss = margin_rate * amt[blocked & (y == 0)].sum() + churn_cost * fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fn_loss": fn_loss,
        "fp_loss": fp_loss,
        "total_cost": fn_loss + fp_loss,
    }


# Doing nothing: every fraud succeeds, no customer is ever blocked.
do_nothing_cost = amt[y == 1].sum() + CHARGEBACK_FEE * n_fraud
print(f"\n  Cost of no detector: Rs.{do_nothing_cost:,.0f}")

results = {}
summary = {}

for name, cfg in SCENARIOS.items():
    rows = [evaluate(t, cfg["margin_rate"], cfg["churn_cost"]) for t in THRESHOLDS]
    frame = pd.DataFrame(rows)
    frame["savings"] = do_nothing_cost - frame["total_cost"]
    results[name] = frame

    best_money = frame.loc[frame["total_cost"].idxmin()]
    best_f1 = frame.loc[frame["f1"].idxmax()]

    # What running the F1-optimal threshold costs versus the money-optimal one
    penalty = best_f1["total_cost"] - best_money["total_cost"]

    summary[name] = {
        "margin_rate": cfg["margin_rate"],
        "churn_cost": cfg["churn_cost"],
        "money_optimal_threshold": float(best_money["threshold"]),
        "money_optimal_savings": float(best_money["savings"]),
        "money_optimal_precision": float(best_money["precision"]),
        "money_optimal_recall": float(best_money["recall"]),
        "f1_optimal_threshold": float(best_f1["threshold"]),
        "f1_optimal_savings": float(best_f1["savings"]),
        "f1_optimal_precision": float(best_f1["precision"]),
        "f1_optimal_recall": float(best_f1["recall"]),
        "penalty_of_using_f1": float(penalty),
    }

    print("\n" + "=" * 60)
    print(f"SCENARIO: {name.upper()}   "
          f"(margin {cfg['margin_rate']:.0%}, churn Rs.{cfg['churn_cost']:.0f})")
    print("=" * 60)
    print(f"  F1-optimal threshold    : {best_f1['threshold']:.3f}")
    print(f"    precision {best_f1['precision']:.3f} | "
          f"recall {best_f1['recall']:.3f} | "
          f"blocked {int(best_f1['tp'] + best_f1['fp']):,}")
    print(f"    net saved             : Rs.{best_f1['savings']:,.0f}")
    print()
    print(f"  Money-optimal threshold : {best_money['threshold']:.3f}")
    print(f"    precision {best_money['precision']:.3f} | "
          f"recall {best_money['recall']:.3f} | "
          f"blocked {int(best_money['tp'] + best_money['fp']):,}")
    print(f"    net saved             : Rs.{best_money['savings']:,.0f}")
    print()
    print(f"  >> Cost of optimising F1 : Rs.{penalty:,.0f} "
          f"on {n:,} transactions")


# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for name, frame in results.items():
    ax.plot(frame["threshold"], frame["savings"] / 1e5, label=name, linewidth=2)
    best = frame.loc[frame["total_cost"].idxmin()]
    ax.plot(best["threshold"], best["savings"] / 1e5, "o", markersize=7)

mod = results["moderate"]
f1_pt = mod.loc[mod["f1"].idxmax()]
ax.axvline(f1_pt["threshold"], linestyle="--", color="crimson", linewidth=1.5,
           label=f"F1-optimal ({f1_pt['threshold']:.2f})")

ax.set_xlabel("Decision threshold")
ax.set_ylabel("Net saved vs no detector (Rs. lakh)")
ax.set_title("The optimal threshold moves with merchant economics")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(mod["threshold"], mod["fn_loss"] / 1e5, label="Fraud let through", linewidth=2)
ax.plot(mod["threshold"], mod["fp_loss"] / 1e5, label="Good customers blocked", linewidth=2)
ax.plot(mod["threshold"], mod["total_cost"] / 1e5, label="Total cost", linewidth=2.5, color="black")

best = mod.loc[mod["total_cost"].idxmin()]
ax.axvline(best["threshold"], linestyle="--", color="green", linewidth=1.5,
           label=f"Money-optimal ({best['threshold']:.2f})")
ax.axvline(f1_pt["threshold"], linestyle="--", color="crimson", linewidth=1.5,
           label=f"F1-optimal ({f1_pt['threshold']:.2f})")

ax.set_xlabel("Decision threshold")
ax.set_ylabel("Cost (Rs. lakh)")
ax.set_title("Where the two error types trade off (moderate scenario)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/cost_curve.png", dpi=150)
print(f"\nSaved plot: {OUT}/cost_curve.png")


# -------------------------------------------------------------- artifacts
for name, frame in results.items():
    frame.to_csv(f"{OUT}/threshold_sweep_{name}.csv", index=False)

with open(f"{OUT}/cost_summary.json", "w") as f:
    json.dump(
        {
            "assumptions": {
                "usd_to_inr": USD_TO_INR,
                "chargeback_fee_inr": CHARGEBACK_FEE,
                "scenarios": SCENARIOS,
            },
            "holdout": {
                "n_transactions": n,
                "n_fraud": n_fraud,
                "total_value_inr": float(amt.sum()),
                "do_nothing_cost_inr": float(do_nothing_cost),
            },
            "results": summary,
        },
        f,
        indent=2,
    )

print(f"Saved summary: {OUT}/cost_summary.json")
print(f"Saved sweeps: {OUT}/threshold_sweep_*.csv")
