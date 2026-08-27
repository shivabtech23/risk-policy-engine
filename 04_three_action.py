#!/usr/bin/env python3
"""
Step 5 - Three-action policy: allow / challenge / block.

The two-action framing (allow or block) is itself an assumption. Real
payment systems have a cheap middle option: step-up authentication.
A fraudster usually cannot pass an OTP they do not control; a genuine
customer usually can, with some drop-off.

That asymmetry is exploitable. This sweeps BOTH thresholds jointly and
compares the best three-action policy against the best two-action one.

Note: the two-action case is not a separate model - it is the diagonal
of this grid, where t_low == t_high and the challenge band is empty.
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

# --- step-up authentication assumptions -----------------------------
# Fraudster passes the challenge. Not zero: SIM swap, phishing, and
# social engineering all defeat OTP some of the time.
FRAUD_BYPASS_RATE = 0.15

# Genuine customer abandons at the challenge. Friction drop-off.
LEGIT_ABANDON_RATE = 0.06

# Infrastructure cost per challenge issued (3DS / SMS).
CHALLENGE_OP_COST = 5.0

# A customer who abandons at OTP is annoyed; one who is hard-rejected
# is insulted. Weight the churn accordingly.
ABANDON_CHURN_MULTIPLIER = 0.40

SCENARIOS = {
    "conservative": {"margin_rate": 0.20, "churn_cost": 150.0},
    "moderate":     {"margin_rate": 0.30, "churn_cost": 400.0},
    "aggressive":   {"margin_rate": 0.40, "churn_cost": 900.0},
}

GRID = np.linspace(0.02, 0.98, 49)


# ------------------------------------------------------------------ load
df = pd.read_csv(f"{OUT}/holdout_predictions.csv")

y = df["isFraud"].values.astype(int)
p = df["pred"].values
amt = df["TransactionAmt"].values * USD_TO_INR

n = len(df)
n_fraud = int(y.sum())
fraud_mask = y == 1
legit_mask = y == 0

do_nothing_cost = amt[fraud_mask].sum() + CHARGEBACK_FEE * n_fraud

print("=" * 64)
print("HELD-OUT SET")
print("=" * 64)
print(f"  transactions       : {n:,}")
print(f"  fraud              : {n_fraud:,} ({y.mean():.3%})")
print(f"  cost of no detector: Rs.{do_nothing_cost:,.0f}")
print()
print("STEP-UP ASSUMPTIONS")
print(f"  fraudster passes OTP   : {FRAUD_BYPASS_RATE:.0%}")
print(f"  genuine customer drops : {LEGIT_ABANDON_RATE:.0%}")
print(f"  cost per challenge     : Rs.{CHALLENGE_OP_COST:.0f}")
print(f"  abandon churn weight   : {ABANDON_CHURN_MULTIPLIER:.0%} of a hard block")


def evaluate(t_low, t_high, margin_rate, churn_cost):
    """Cost in rupees of a three-action policy.

    t_low == t_high collapses to the two-action case: the challenge
    band is empty and every transaction is either allowed or blocked.
    """
    allow = p < t_low
    challenge = (p >= t_low) & (p < t_high)
    block = p >= t_high

    # --- fraud we failed to stop
    f_allow = allow & fraud_mask
    f_chal = challenge & fraud_mask
    f_block = block & fraud_mask

    loss_allowed = amt[f_allow].sum() + CHARGEBACK_FEE * f_allow.sum()
    loss_bypassed = FRAUD_BYPASS_RATE * (
        amt[f_chal].sum() + CHARGEBACK_FEE * f_chal.sum()
    )
    fraud_loss = loss_allowed + loss_bypassed

    # --- genuine customers we cost money
    l_chal = challenge & legit_mask
    l_block = block & legit_mask

    loss_abandoned = LEGIT_ABANDON_RATE * (
        margin_rate * amt[l_chal].sum()
        + ABANDON_CHURN_MULTIPLIER * churn_cost * l_chal.sum()
    )
    loss_blocked = margin_rate * amt[l_block].sum() + churn_cost * l_block.sum()
    customer_loss = loss_abandoned + loss_blocked

    # --- running the challenges
    op_cost = CHALLENGE_OP_COST * challenge.sum()

    total = fraud_loss + customer_loss + op_cost

    # Fraud actually stopped: hard blocks, plus challenges not bypassed
    stopped = f_block.sum() + (1 - FRAUD_BYPASS_RATE) * f_chal.sum()

    return {
        "t_low": t_low,
        "t_high": t_high,
        "fraud_loss": fraud_loss,
        "customer_loss": customer_loss,
        "op_cost": op_cost,
        "total_cost": total,
        "savings": do_nothing_cost - total,
        "effective_recall": stopped / n_fraud,
        "n_challenged": int(challenge.sum()),
        "n_blocked": int(block.sum()),
        "challenge_rate": challenge.sum() / n,
        "block_rate": block.sum() / n,
        "legit_challenged": int(l_chal.sum()),
        "legit_blocked": int(l_block.sum()),
    }


summary = {}
grids = {}

for name, cfg in SCENARIOS.items():
    rows = [
        evaluate(lo, hi, cfg["margin_rate"], cfg["churn_cost"])
        for lo, hi in itertools.product(GRID, GRID)
        if lo <= hi
    ]
    frame = pd.DataFrame(rows)
    grids[name] = frame

    best3 = frame.loc[frame["savings"].idxmax()]

    # The two-action optimum is the best point on the diagonal
    diag = frame[np.isclose(frame["t_low"], frame["t_high"])]
    best2 = diag.loc[diag["savings"].idxmax()]

    gain = best3["savings"] - best2["savings"]

    summary[name] = {
        "two_action": {
            "threshold": float(best2["t_high"]),
            "savings": float(best2["savings"]),
            "effective_recall": float(best2["effective_recall"]),
            "n_blocked": int(best2["n_blocked"]),
            "legit_blocked": int(best2["legit_blocked"]),
        },
        "three_action": {
            "t_low": float(best3["t_low"]),
            "t_high": float(best3["t_high"]),
            "savings": float(best3["savings"]),
            "effective_recall": float(best3["effective_recall"]),
            "n_challenged": int(best3["n_challenged"]),
            "n_blocked": int(best3["n_blocked"]),
            "challenge_rate": float(best3["challenge_rate"]),
            "legit_challenged": int(best3["legit_challenged"]),
            "legit_blocked": int(best3["legit_blocked"]),
        },
        "gain_inr": float(gain),
        "recall_gain": float(best3["effective_recall"] - best2["effective_recall"]),
    }

    print("\n" + "=" * 64)
    print(f"SCENARIO: {name.upper()}   "
          f"(margin {cfg['margin_rate']:.0%}, churn Rs.{cfg['churn_cost']:.0f})")
    print("=" * 64)
    print(f"  TWO-ACTION (allow / block)")
    print(f"    threshold        : {best2['t_high']:.3f}")
    print(f"    fraud stopped    : {best2['effective_recall']:.1%}")
    print(f"    genuine blocked  : {int(best2['legit_blocked']):,}")
    print(f"    net saved        : Rs.{best2['savings']:,.0f}")
    print()
    print(f"  THREE-ACTION (allow / challenge / block)")
    print(f"    challenge from   : {best3['t_low']:.3f}")
    print(f"    block from       : {best3['t_high']:.3f}")
    print(f"    fraud stopped    : {best3['effective_recall']:.1%}")
    print(f"    challenged       : {int(best3['n_challenged']):,} "
          f"({best3['challenge_rate']:.2%} of traffic)")
    print(f"    genuine blocked  : {int(best3['legit_blocked']):,}")
    print(f"    net saved        : Rs.{best3['savings']:,.0f}")
    print()
    print(f"  >> Gain from the third action : Rs.{gain:,.0f}")
    print(f"  >> Extra fraud stopped        : "
          f"{(best3['effective_recall'] - best2['effective_recall']):.1%} "
          f"of all fraud")


# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# LEFT: the policy surface, with the two-action diagonal drawn on it
ax = axes[0]
cons = grids["conservative"]

pivot = cons.pivot_table(index="t_high", columns="t_low", values="savings")
vals = pivot.values / 1e5
mesh = ax.pcolormesh(pivot.columns, pivot.index, vals,
                     shading="auto", cmap="RdYlGn",
                     vmin=0, vmax=float(np.nanmax(vals)))
plt.colorbar(mesh, ax=ax, label="Net saved (Rs. lakh)")

s = summary["conservative"]
ax.plot([0.02, 0.98], [0.02, 0.98], "k--", linewidth=2,
        label="Two-action policies (all of them)")
ax.plot(s["two_action"]["threshold"], s["two_action"]["threshold"],
        "o", color="black", markersize=13, markeredgecolor="white",
        markeredgewidth=2, label=f"Best two-action ({s['two_action']['threshold']:.2f})", zorder=5)
ax.plot(s["three_action"]["t_low"], s["three_action"]["t_high"],
        "*", color="blue", markersize=24, markeredgecolor="white",
        markeredgewidth=1.5,
        label=f"Best three-action ({s['three_action']['t_low']:.2f}, {s['three_action']['t_high']:.2f})",
        zorder=5)

ax.set_xlabel("Challenge from (t_low)", fontsize=11)
ax.set_ylabel("Block from (t_high)", fontsize=11)
ax.set_title("Every two-action policy lies on one line.\n"
             "The optimum does not.", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")

# RIGHT: what the third action buys, per scenario
ax = axes[1]
names = list(SCENARIOS.keys())
x = np.arange(len(names))
w = 0.35

two_r = [summary[n]["two_action"]["effective_recall"] * 100 for n in names]
three_r = [summary[n]["three_action"]["effective_recall"] * 100 for n in names]

ax.bar(x - w/2, two_r, w, label="Allow / block", color="#94a3b8")
ax.bar(x + w/2, three_r, w, label="Allow / challenge / block", color="#2563eb")

for i, n in enumerate(names):
    gain = summary[n]["gain_inr"] / 1e5
    ax.annotate(f"+Rs.{gain:.1f}L", xy=(i + w/2, three_r[i] + 1.5),
                ha="center", fontsize=10, fontweight="bold", color="#2563eb")

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Fraud actually stopped (%)", fontsize=11)

ax.set_ylim(0, max(three_r) + 12)
ax.set_title("A cheap middle option catches more fraud\n"
             "without blocking more customers", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/three_action.png", dpi=160, bbox_inches="tight")
print(f"\nSaved plot: {OUT}/three_action.png")

for name, frame in grids.items():
    frame.to_csv(f"{OUT}/three_action_grid_{name}.csv", index=False)

with open(f"{OUT}/three_action_summary.json", "w") as f:
    json.dump(
        {
            "assumptions": {
                "usd_to_inr": USD_TO_INR,
                "chargeback_fee_inr": CHARGEBACK_FEE,
                "fraud_bypass_rate": FRAUD_BYPASS_RATE,
                "legit_abandon_rate": LEGIT_ABANDON_RATE,
                "challenge_op_cost_inr": CHALLENGE_OP_COST,
                "abandon_churn_multiplier": ABANDON_CHURN_MULTIPLIER,
                "scenarios": SCENARIOS,
            },
            "holdout": {
                "n_transactions": n,
                "n_fraud": n_fraud,
                "do_nothing_cost_inr": float(do_nothing_cost),
            },
            "results": summary,
        },
        f,
        indent=2,
    )

print(f"Saved summary: {OUT}/three_action_summary.json")
