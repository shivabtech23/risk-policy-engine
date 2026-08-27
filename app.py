"""
Razorpay Buildathon - Track 02
Threshold policy explorer.

The argument: a fraud model's decision threshold is a business decision,
not a statistical one. Move the sliders and watch the money change.
"""

import itertools

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risk Policy Explorer", layout="wide")

OUT = "artifacts"
USD_TO_INR = 85.0
CHARGEBACK_FEE = 1500.0
ABANDON_CHURN_MULTIPLIER = 0.40


@st.cache_data
def load():
    df = pd.read_csv(f"{OUT}/holdout_predictions.csv")
    return (
        df["isFraud"].values.astype(int),
        df["pred"].values,
        df["TransactionAmt"].values * USD_TO_INR,
        df,
    )


y, p, amt, raw = load()
n = len(y)
n_fraud = int(y.sum())
fraud_mask = y == 1
legit_mask = y == 0
do_nothing = amt[fraud_mask].sum() + CHARGEBACK_FEE * n_fraud


def evaluate(t_low, t_high, margin, churn, bypass, abandon, chal_cost):
    allow = p < t_low
    challenge = (p >= t_low) & (p < t_high)
    block = p >= t_high

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
        abandon * (margin * amt[l_chal].sum()
                   + ABANDON_CHURN_MULTIPLIER * churn * l_chal.sum())
        + margin * amt[l_block].sum() + churn * l_block.sum()
    )
    op = chal_cost * challenge.sum()
    stopped = f_block.sum() + (1 - bypass) * f_chal.sum()

    return {
        "savings": do_nothing - (fraud_loss + customer_loss + op),
        "recall": stopped / n_fraud,
        "legit_blocked": int(l_block.sum()),
        "legit_challenged": int(l_chal.sum()),
        "challenge_rate": challenge.sum() / n,
        "fraud_loss": fraud_loss,
        "customer_loss": customer_loss,
        "op_cost": op,
        "n_allowed": int(allow.sum()),
        "n_challenged": int(challenge.sum()),
        "n_blocked": int(block.sum()),
    }


st.title("Fraud threshold policy explorer")
st.caption(
    f"{n:,} held-out transactions | {n_fraud:,} fraudulent ({y.mean():.2%}) | "
    f"IEEE-CIS, time-ordered split | LightGBM ROC-AUC 0.911"
)

with st.sidebar:
    st.header("Merchant economics")
    margin = st.slider("Gross margin", 0.05, 0.60, 0.20, 0.05,
                       help="Profit lost when a genuine sale is blocked")
    churn = st.slider("Lost customer value (Rs.)", 0, 3000, 150, 50,
                      help="Expected future profit from a customer who never returns")

    st.header("Step-up authentication")
    bypass = st.slider("Fraudster defeats challenge", 0.0, 0.80, 0.15, 0.05)
    abandon = st.slider("Genuine customer abandons", 0.0, 0.50, 0.06, 0.02)
    chal_cost = st.slider("Cost per challenge (Rs.)", 0, 300, 5, 5)

    st.divider()
    st.caption("Every number here is an assumption, not a measurement. "
               "That is the point - the optimal policy depends on them.")

c1, c2 = st.columns(2)
with c1:
    t_low = st.slider("Challenge above risk score", 0.0, 1.0, 0.22, 0.01)
with c2:
    t_high = st.slider("Block above risk score", 0.0, 1.0, 0.92, 0.01)

if t_low > t_high:
    st.error("Challenge threshold must be below block threshold.")
    st.stop()

r = evaluate(t_low, t_high, margin, churn, bypass, abandon, chal_cost)


@st.cache_data
def best_two(margin, churn):
    best = None
    for t in np.linspace(0.02, 0.98, 49):
        block = p >= t
        l_block = block & legit_mask
        fl = (amt[~block & fraud_mask].sum()
              + CHARGEBACK_FEE * (~block & fraud_mask).sum())
        cl = margin * amt[l_block].sum() + churn * l_block.sum()
        s = do_nothing - (fl + cl)
        if best is None or s > best["savings"]:
            best = {"threshold": t, "savings": s,
                    "recall": (block & fraud_mask).sum() / n_fraud,
                    "legit_blocked": int(l_block.sum())}
    return best


b2 = best_two(margin, churn)
delta = r["savings"] - b2["savings"]

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Net saved", f"Rs.{r['savings']/1e5:,.1f}L",
          f"{delta/1e5:+,.1f}L vs best allow/block")
m2.metric("Fraud stopped", f"{r['recall']:.1%}",
          f"{(r['recall'] - b2['recall'])*100:+.1f} pts")
m3.metric("Genuine blocked", f"{r['legit_blocked']:,}",
          f"{r['legit_blocked'] - b2['legit_blocked']:+,}",
          delta_color="inverse")
m4.metric("Challenged", f"{r['challenge_rate']:.1%}",
          "of all traffic", delta_color="off")

if r["challenge_rate"] > 0.15:
    st.warning(
        f"Challenging {r['challenge_rate']:.1%} of traffic. Real 3DS step-up "
        f"rates sit near 5-15%; above that, checkout friction imposes costs "
        f"this model does not capture."
    )

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Where the money goes")
    breakdown = pd.DataFrame({
        "Component": ["Fraud not stopped", "Genuine customers lost",
                      "Challenge infrastructure"],
        "Rs. lakh": [r["fraud_loss"]/1e5, r["customer_loss"]/1e5,
                     r["op_cost"]/1e5],
    })
    breakdown["Share"] = breakdown["Rs. lakh"] / breakdown["Rs. lakh"].sum()
    st.dataframe(
        breakdown.style.format({"Rs. lakh": "{:,.1f}", "Share": "{:.1%}"}),
        hide_index=True, width="stretch",
    )
    st.caption(
        f"Doing nothing costs Rs.{do_nothing/1e5:,.1f}L. "
        f"This policy costs Rs.{(do_nothing - r['savings'])/1e5:,.1f}L."
    )

with right:
    st.subheader("What happens to traffic")
    st.dataframe(
        pd.DataFrame({
            "Action": ["Allow", "Challenge", "Block"],
            "Count": [r["n_allowed"], r["n_challenged"], r["n_blocked"]],
            "Share": [f"{r['n_allowed']/n:.1%}",
                      f"{r['n_challenged']/n:.1%}",
                      f"{r['n_blocked']/n:.1%}"],
        }),
        hide_index=True, width="stretch",
    )
    st.caption(
        f"Best two-action policy here: block above {b2['threshold']:.2f}, "
        f"stopping {b2['recall']:.1%} of fraud and blocking "
        f"{b2['legit_blocked']:,} genuine customers."
    )

st.divider()
st.subheader("Highest-risk transactions under this policy")

view = raw.copy()
view["amount_inr"] = view["TransactionAmt"] * USD_TO_INR
view["action"] = np.where(view["pred"] >= t_high, "BLOCK",
                  np.where(view["pred"] >= t_low, "CHALLENGE", "ALLOW"))
view["outcome"] = np.where(view["isFraud"] == 1, "fraud", "genuine")

tab1, tab2 = st.tabs(["Near the block threshold", "Highest risk"])

def render(frame):
    out = (frame[["TransactionID", "amount_inr", "pred", "action", "outcome"]]
           .rename(columns={"pred": "risk_score", "amount_inr": "amount_rs"}))
    st.dataframe(
        out.style.format({"risk_score": "{:.3f}", "amount_rs": "Rs.{:,.0f}"}),
        hide_index=True, width="stretch", height=360,
    )

with tab1:
    st.caption("Where the policy is making its hardest calls.")
    near = view.iloc[(view["pred"] - t_high).abs().argsort()[:40]]
    render(near.sort_values("pred", ascending=False))

with tab2:
    render(view.sort_values("pred", ascending=False).head(40))


st.caption(
    "Scores from LightGBM trained on the first 80% of transactions by time, "
    "evaluated on the last 20%. Amounts converted from USD at 85."
)
