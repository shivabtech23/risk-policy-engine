"""
Razorpay Buildathon - Track 02: Risk Policy Explorer
Advanced Fraud Threshold & Decision Engine.

The argument: A fraud model's decision threshold is a financial economics decision,
not purely a statistical one. Move the sliders, choose presets, or auto-optimize
to observe real monetary savings on held-out transaction traffic.
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Risk Policy Explorer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, glassmorphic cards, glowing badges, modern typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding & Backdrop */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Banner / Hero Header Styling */
    .hero-container {
        background: linear-gradient(135deg, #0b0f19 0%, #0f172a 40%, #1e1b4b 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.8), 0 0 30px rgba(99, 102, 241, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin: 0 0 16px 0;
        font-weight: 400;
    }
    .badge-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
    }
    .badge {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        backdrop-filter: blur(8px);
    }
    .badge-indigo {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.4);
        color: #fcd34d;
    }

    /* Metric Cards Styling */
    .metric-card-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    @media (max-width: 900px) {
        .metric-card-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .sub-positive { color: #34d399; }
    .sub-negative { color: #f87171; }
    .sub-neutral { color: #94a3b8; }

    /* Custom Container Boxes */
    .glass-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    /* Action Badges in Dataframes */
    .action-allow {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .action-challenge {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .action-block {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
    }

    /* Streamlit Widget Customization */
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & DATA LOADING
# -----------------------------------------------------------------------------
OUT = "artifacts"
USD_TO_INR = 85.0
CHARGEBACK_FEE = 1500.0
ABANDON_CHURN_MULTIPLIER = 0.40

@st.cache_data
def load_data():
    if not os.path.exists(f"{OUT}/holdout_predictions.csv"):
        st.error("Holdout predictions file not found in artifacts/. Please run training first.")
        st.stop()
    df = pd.read_csv(f"{OUT}/holdout_predictions.csv")
    
    # Precalculate fixed attributes
    y = df["isFraud"].values.astype(int)
    p = df["pred"].values
    amt = df["TransactionAmt"].values * USD_TO_INR
    n = len(y)
    n_fraud = int(y.sum())
    fraud_mask = y == 1
    legit_mask = y == 0
    do_nothing = amt[fraud_mask].sum() + CHARGEBACK_FEE * n_fraud
    
    return y, p, amt, df, n, n_fraud, fraud_mask, legit_mask, do_nothing

y, p, amt, raw_df, n, n_fraud, fraud_mask, legit_mask, do_nothing = load_data()

# Precompute cumulative stats across grid for fast evaluation & surface plotting
GRID = np.linspace(0.02, 0.98, 49)
K = len(GRID)

@st.cache_data
def precalculate_grid_stats():
    f_amt = np.array([amt[fraud_mask & (p < t)].sum() for t in GRID])
    f_cnt = np.array([(fraud_mask & (p < t)).sum() for t in GRID])
    l_amt = np.array([amt[legit_mask & (p < t)].sum() for t in GRID])
    l_cnt = np.array([(legit_mask & (p < t)).sum() for t in GRID])
    
    total_f_amt = amt[fraud_mask].sum()
    total_f_cnt = n_fraud
    total_l_amt = amt[legit_mask].sum()
    total_l_cnt = n - n_fraud
    
    return f_amt, f_cnt, l_amt, l_cnt, total_f_amt, total_f_cnt, total_l_amt, total_l_cnt

f_amt, f_cnt, l_amt, l_cnt, total_f_amt, total_f_cnt, total_l_amt, total_l_cnt = precalculate_grid_stats()

# Fast policy evaluation
def evaluate_policy(t_low, t_high, margin, churn, bypass, abandon, chal_cost):
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
        abandon * (margin * amt[l_chal].sum() + ABANDON_CHURN_MULTIPLIER * churn * l_chal.sum())
        + margin * amt[l_block].sum() + churn * l_block.sum()
    )
    op = chal_cost * challenge.sum()
    stopped = f_block.sum() + (1 - bypass) * f_chal.sum()
    total_cost = fraud_loss + customer_loss + op
    savings = do_nothing - total_cost

    return {
        "savings": savings,
        "recall": stopped / n_fraud,
        "legit_blocked": int(l_block.sum()),
        "legit_challenged": int(l_chal.sum()),
        "challenge_rate": challenge.sum() / n,
        "block_rate": block.sum() / n,
        "fraud_loss": fraud_loss,
        "customer_loss": customer_loss,
        "op_cost": op,
        "n_allowed": int(allow.sum()),
        "n_challenged": int(challenge.sum()),
        "n_blocked": int(block.sum()),
        "total_cost": total_cost,
    }

# Fast 2-Action Optimum (diagonal of grid)
@st.cache_data
def get_best_two_action(margin, churn):
    best = None
    for t in GRID:
        block = p >= t
        l_block = block & legit_mask
        f_block = block & fraud_mask
        fl = amt[~block & fraud_mask].sum() + CHARGEBACK_FEE * (~block & fraud_mask).sum()
        cl = margin * amt[l_block].sum() + churn * l_block.sum()
        s = do_nothing - (fl + cl)
        if best is None or s > best["savings"]:
            best = {
                "threshold": float(t),
                "savings": float(s),
                "recall": float(f_block.sum() / n_fraud),
                "legit_blocked": int(l_block.sum()),
            }
    return best

# Fast 3-Action Surface Precomputation (returns both global unconstrained and 10% friction-constrained optima)
@st.cache_data
def compute_surface_matrix(margin, churn, bypass, abandon, chal_cost):
    matrix_savings = np.full((K, K), np.nan)
    matrix_recall = np.full((K, K), np.nan)
    best_3 = None
    best_3_cons = None

    for i in range(K):
        for j in range(i, K):
            f_allow_amt, f_allow_cnt = f_amt[i], f_cnt[i]
            f_block_amt, f_block_cnt = total_f_amt - f_amt[j], total_f_cnt - f_cnt[j]
            l_block_amt, l_block_cnt = total_l_amt - l_amt[j], total_l_cnt - l_cnt[j]
            f_chal_amt, f_chal_cnt = f_amt[j] - f_amt[i], f_cnt[j] - f_cnt[i]
            l_chal_amt, l_chal_cnt = l_amt[j] - l_amt[i], l_cnt[j] - l_cnt[i]

            fraud_loss = (f_allow_amt + CHARGEBACK_FEE * f_allow_cnt) + bypass * (f_chal_amt + CHARGEBACK_FEE * f_chal_cnt)
            customer_loss = abandon * (margin * l_chal_amt + ABANDON_CHURN_MULTIPLIER * churn * l_chal_cnt) + margin * l_block_amt + churn * l_block_cnt
            op = chal_cost * (f_chal_cnt + l_chal_cnt)
            total_cost = fraud_loss + customer_loss + op
            s = do_nothing - total_cost
            
            stopped = f_block_cnt + (1 - bypass) * f_chal_cnt
            rec = stopped / n_fraud
            c_rate = (f_chal_cnt + l_chal_cnt) / n

            matrix_savings[j, i] = s / 1e5
            matrix_recall[j, i] = rec

            res = {
                "t_low": float(GRID[i]),
                "t_high": float(GRID[j]),
                "savings": float(s),
                "recall": float(rec),
                "legit_blocked": int(l_block_cnt),
                "n_challenged": int(f_chal_cnt + l_chal_cnt),
                "challenge_rate": float(c_rate),
            }

            if best_3 is None or s > best_3["savings"]:
                best_3 = res

            if c_rate <= 0.10:
                if best_3_cons is None or s > best_3_cons["savings"]:
                    best_3_cons = res

    if best_3_cons is None:
        best_3_cons = best_3

    return matrix_savings, matrix_recall, best_3, best_3_cons


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & PRESET PROFILES
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 Merchant Economic Profile")
    st.caption("Customize cost structure & risk appetite parameters.")

    # Preset profiles selector
    preset = st.selectbox(
        "⚡ Quick Economic Presets",
        [
            "Custom Parameters",
            "🛡️ E-Commerce Standard (20% margin, ₹150 churn)",
            "💎 Luxury Goods (40% margin, ₹800 churn)",
            "⚡ Digital Subscriptions (70% margin, ₹1,500 churn)",
            "🛒 High Volume Retail (8% margin, ₹50 churn)",
        ],
        index=1
    )

    # Preset application
    if preset == "🛡️ E-Commerce Standard (20% margin, ₹150 churn)":
        def_margin, def_churn = 0.20, 150
    elif preset == "💎 Luxury Goods (40% margin, ₹800 churn)":
        def_margin, def_churn = 0.40, 800
    elif preset == "⚡ Digital Subscriptions (70% margin, ₹1,500 churn)":
        def_margin, def_churn = 0.70, 1500
    elif preset == "🛒 High Volume Retail (8% margin, ₹50 churn)":
        def_margin, def_churn = 0.08, 50
    else:
        def_margin, def_churn = 0.20, 150

    margin = st.slider(
        "Gross Margin", 0.05, 0.80, float(def_margin), 0.05,
        help="Profit lost when a genuine customer's transaction is blocked."
    )
    churn = st.slider(
        "Lost Customer Value (Rs.)", 0, 3000, int(def_churn), 50,
        help="Expected future customer lifetime value lost when insulted by a block."
    )

    st.markdown("---")
    st.markdown("### 🔐 Step-Up 3DS Authentication")
    
    bypass = st.slider(
        "Fraudster defeats OTP", 0.0, 0.80, 0.15, 0.05,
        help="Fraction of fraudsters passing OTP via SIM swap or social engineering."
    )
    abandon = st.slider(
        "Genuine customer abandons", 0.0, 0.50, 0.06, 0.02,
        help="Friction drop-off rate when a genuine customer is challenged."
    )
    chal_cost = st.slider(
        "Cost per challenge (Rs.)", 0, 300, 5, 5,
        help="SMS / 3DS gateway infrastructure cost per challenge step."
    )

    st.markdown("---")
    st.info(
        "💡 **Key Insight**: Move beyond F1-score! The optimal decision policy depends "
        "directly on merchant margin, churn, and step-up friction."
    )


# -----------------------------------------------------------------------------
# 4. MAIN PAGE HERO BANNER & OPTIMIZATION CONTROLS
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Fraud Risk Policy Explorer</div>
    <div class="hero-subtitle">Turn statistical fraud model scores into optimal money-saving policy decisions with 3-action step-up authentication.</div>
    <div class="badge-row">
        <span class="badge badge-indigo">⚡ IEEE-CIS Dataset</span>
        <span class="badge badge-emerald">🎯 LightGBM ROC-AUC: 0.9112</span>
        <span class="badge">📊 Held-Out Test Set: 118,108 Txns</span>
        <span class="badge badge-amber">🏆 Razorpay Buildathon 2026 — Track 02 Submission</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Calculate optimal values
matrix_savings, matrix_recall, best_3, best_3_cons = compute_surface_matrix(margin, churn, bypass, abandon, chal_cost)
b2 = get_best_two_action(margin, churn)

# Threshold Sliders Header & Auto-Optimize Action
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 1.2, 1.2])

with ctrl_col2:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("🛡️ 10% Friction Cap", type="primary", use_container_width=True, help="Set policy to optimal thresholds respecting a 10% 3DS challenge budget"):
        st.session_state["t_low"] = best_3_cons["t_low"]
        st.session_state["t_high"] = best_3_cons["t_high"]
        st.toast("Policy set to 10% friction cap thresholds!", icon="🛡️")

with ctrl_col3:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("⚡ Max Savings (Unconstrained)", use_container_width=True, help="Set policy to global unconstrained mathematical maximum"):
        st.session_state["t_low"] = best_3["t_low"]
        st.session_state["t_high"] = best_3["t_high"]
        st.toast("Policy set to global unconstrained financial optimum!", icon="⚡")

with ctrl_col1:
    st.markdown("##### 🎛️ Policy Threshold Sliders")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        t_low = st.slider(
            "Challenge above risk score (t_low)", 0.0, 1.0, 
            key="t_low", value=st.session_state.get("t_low", best_3_cons["t_low"]), step=0.01
        )
    with s_col2:
        t_high = st.slider(
            "Block above risk score (t_high)", 0.0, 1.0, 
            key="t_high", value=st.session_state.get("t_high", best_3_cons["t_high"]), step=0.01
        )

if t_low > t_high:
    st.error("⚠️ Challenge threshold (t_low) must be less than or equal to Block threshold (t_high).")
    st.stop()

# Evaluate user selected policy
r = evaluate_policy(t_low, t_high, margin, churn, bypass, abandon, chal_cost)
delta_savings = r["savings"] - b2["savings"]
delta_recall = (r["recall"] - b2["recall"]) * 100
delta_blocked = r["legit_blocked"] - b2["legit_blocked"]


# -----------------------------------------------------------------------------
# 5. HERO METRIC CARDS
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="metric-card-grid">
    <div class="metric-card">
        <div class="metric-label">Net Saved vs Baseline</div>
        <div class="metric-value">Rs. {r['savings']/1e5:,.1f}L</div>
        <div class="metric-sub {'sub-positive' if delta_savings >= 0 else 'sub-negative'}">
            {'▲' if delta_savings >= 0 else '▼'} Rs. {abs(delta_savings)/1e5:,.1f}L vs 2-Action Best
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Effective Fraud Stopped</div>
        <div class="metric-value">{r['recall']:.1%}</div>
        <div class="metric-sub {'sub-positive' if delta_recall >= 0 else 'sub-negative'}">
            {'▲' if delta_recall >= 0 else '▼'} {abs(delta_recall):.1f} pts vs 2-Action Best
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Genuine Customers Blocked</div>
        <div class="metric-value">{r['legit_blocked']:,}</div>
        <div class="metric-sub {'sub-positive' if delta_blocked <= 0 else 'sub-negative'}">
            {'▼' if delta_blocked <= 0 else '▲'} {abs(delta_blocked):,} {'fewer' if delta_blocked <= 0 else 'more'} vs 2-Action Best
        </div>
    </div>
    <div class="metric-card">
        <div class="metric-label">3DS Challenge Rate</div>
        <div class="metric-value">{r['challenge_rate']:.1%}</div>
        <div class="metric-sub {'sub-positive' if r['challenge_rate'] <= 0.15 else 'sub-negative'}">
            {'🟢 Safe Friction' if r['challenge_rate'] <= 0.15 else '⚠️ High Friction (>15%)'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if r["challenge_rate"] > 0.15:
    st.warning(
        f"⚠️ **High Friction Alert**: Challenging {r['challenge_rate']:.1%} of all traffic. "
        f"Real 3DS step-up rates typically stay within 5–15%. Above this level, checkout "
        f"friction causes user drop-offs that may exceed model assumptions."
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. TABS FOR DEEP VISUAL ANALYTICS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌌 3-Action Surface Heatmap",
    "💰 Financial & Cost Breakdown",
    "🔀 Traffic Funnel & Buckets",
    "🔬 Transaction Explorer",
    "🧠 Model & Feature Importance"
])


# -----------------------------------------------------------------------------
# TAB 1: 3-ACTION DECISION SURFACE HEATMAP
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("2D Policy Surface: Net Savings Matrix (Rs. Lakh)")
    st.caption(
        "Every 2-action policy (Allow/Block) lies on the black diagonal line. "
        "The 3-action space opens up the full surface, enabling higher savings and lower friction."
    )

    surface_fig = go.Figure()

    # Heatmap trace
    surface_fig.add_trace(go.Heatmap(
        x=GRID,
        y=GRID,
        z=matrix_savings,
        colorscale="Viridis",
        colorbar=dict(title="Net Saved (Rs. Lakh)", tickformat=",.0f"),
        hovertemplate="Challenge (t_low): %{x:.2f}<br>Block (t_high): %{y:.2f}<br>Net Saved: Rs. %{z:.1f} Lakh<extra></extra>",
    ))

    # 2-Action Diagonal line
    surface_fig.add_trace(go.Scatter(
        x=[0.02, 0.98], y=[0.02, 0.98],
        mode="lines",
        line=dict(color="#cbd5e1", width=2.5, dash="dash"),
        name="2-Action Diagonal (t_low = t_high)"
    ))

    # Best 2-Action Point
    surface_fig.add_trace(go.Scatter(
        x=[b2["threshold"]], y=[b2["threshold"]],
        mode="markers+text",
        marker=dict(color="#f59e0b", size=14, symbol="circle", line=dict(color="white", width=2)),
        text=["Best 2-Action"],
        textposition="top left",
        name=f"Best 2-Action ({b2['threshold']:.2f})"
    ))

    # 10% Friction Cap Optimum Point
    surface_fig.add_trace(go.Scatter(
        x=[best_3_cons["t_low"]], y=[best_3_cons["t_high"]],
        mode="markers+text",
        marker=dict(color="#38bdf8", size=16, symbol="diamond", line=dict(color="white", width=2)),
        text=["10% Friction Cap"],
        textposition="bottom right",
        name=f"10% Friction Cap ({best_3_cons['t_low']:.2f}, {best_3_cons['t_high']:.2f})"
    ))

    # Global 3-Action Optimum Point
    surface_fig.add_trace(go.Scatter(
        x=[best_3["t_low"]], y=[best_3["t_high"]],
        mode="markers+text",
        marker=dict(color="#10b981", size=18, symbol="star", line=dict(color="white", width=2)),
        text=["Global Max Savings"],
        textposition="top right",
        name=f"Global Unconstrained ({best_3['t_low']:.2f}, {best_3['t_high']:.2f})"
    ))

    # Current Selection Point
    surface_fig.add_trace(go.Scatter(
        x=[t_low], y=[t_high],
        mode="markers",
        marker=dict(color="#ef4444", size=16, symbol="x", line=dict(color="white", width=2.5)),
        name=f"Current Selection ({t_low:.2f}, {t_high:.2f})"
    ))

    surface_fig.update_layout(
        xaxis_title="Challenge Threshold (t_low)",
        yaxis_title="Block Threshold (t_high)",
        template="plotly_dark",
        height=540,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(surface_fig, use_container_width=True)

    st.markdown("""
    <div class='glass-box'>
        <h4>💡 Policy Trade-Off: Operational Friction vs. Unconstrained Math Optimum</h4>
        <ul style='margin-bottom: 0;'>
            <li><b>Global Max Savings (Green Star)</b>: The unconstrained financial optimum challenges ~24.5% of overall traffic to capture maximum possible fraud.</li>
            <li><b>10% Friction Cap (Cyan Diamond)</b>: Under a realistic operational budget capping 3DS step-up friction at ≤10%, this policy catches <b>69.4% of all fraud</b> (vs 56% in 2-action) while blocking fewer genuine customers and preserving checkout conversion.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: FINANCIAL IMPACT & COST BREAKDOWN
# -----------------------------------------------------------------------------
with tab2:
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.subheader("Financial Loss Breakdown (Rs. Lakh)")
        st.caption("Comparison of cost components: Fraud Loss vs Customer Loss vs Operations.")

        cost_df = pd.DataFrame({
            "Category": ["Fraud Not Stopped", "Legitimate Customers Lost", "3DS Infrastructure Cost"],
            "Cost_Lakh": [r["fraud_loss"] / 1e5, r["customer_loss"] / 1e5, r["op_cost"] / 1e5]
        })

        donut_fig = px.pie(
            cost_df, values="Cost_Lakh", names="Category",
            hole=0.45,
            color="Category",
            color_discrete_map={
                "Fraud Not Stopped": "#ef4444",
                "Legitimate Customers Lost": "#f59e0b",
                "3DS Infrastructure Cost": "#3b82f6"
            }
        )
        donut_fig.update_traces(textinfo="percent+value", texttemplate="Rs.%{value:.1f}L<br>(%{percent})")
        donut_fig.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15)
        )
        st.plotly_chart(donut_fig, use_container_width=True)

    with f_col2:
        st.subheader("Net Savings Comparison")
        st.caption("Baseline cost vs. Policy outcomes.")

        comparison_df = pd.DataFrame({
            "Policy Option": ["Do Nothing (Baseline)", "Best 2-Action Policy", "Selected 3-Action Policy", "Global 3-Action Optimum"],
            "Net Saved (Rs. Lakh)": [0.0, b2["savings"] / 1e5, r["savings"] / 1e5, best_3["savings"] / 1e5],
            "Color": ["#64748b", "#f59e0b", "#6366f1", "#10b981"]
        })

        bar_fig = px.bar(
            comparison_df, x="Policy Option", y="Net Saved (Rs. Lakh)",
            color="Policy Option",
            color_discrete_sequence=["#64748b", "#f59e0b", "#6366f1", "#10b981"],
            text_auto=".1f"
        )
        bar_fig.update_layout(
            template="plotly_dark",
            height=380,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Net Saved (Rs. Lakh)"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"📌 **Financial Summary**: Doing nothing costs **Rs.{do_nothing/1e5:,.1f}L** in baseline fraud & chargebacks. "
        f"Your selected policy reduces total loss to **Rs.{(do_nothing - r['savings'])/1e5:,.1f}L**, saving **Rs.{r['savings']/1e5:,.1f}L** net."
    )


# -----------------------------------------------------------------------------
# TAB 3: TRAFFIC FUNNEL & BUCKETS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Traffic Partitioning Across Decision Buckets")
    st.caption("How transactions are routed across ALLOW, CHALLENGE, and BLOCK decisions.")

    t_col1, t_col2 = st.columns([1.2, 1])

    with t_col1:
        bucket_data = pd.DataFrame({
            "Action": ["ALLOW", "CHALLENGE", "BLOCK"],
            "Count": [r["n_allowed"], r["n_challenged"], r["n_blocked"]],
            "Percentage": [r["n_allowed"]/n * 100, r["n_challenged"]/n * 100, r["n_blocked"]/n * 100]
        })

        funnel_fig = px.bar(
            bucket_data, x="Count", y="Action", orientation="h",
            color="Action",
            color_discrete_map={"ALLOW": "#10b981", "CHALLENGE": "#f59e0b", "BLOCK": "#ef4444"},
            text_auto=",.0f"
        )
        funnel_fig.update_layout(
            template="plotly_dark",
            height=340,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Number of Transactions"
        )
        st.plotly_chart(funnel_fig, use_container_width=True)

    with t_col2:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Bucket Distribution Details")
        st.markdown(f"- **ALLOW** (`risk < {t_low:.2f}`): **{r['n_allowed']:,}** txns ({r['n_allowed']/n:.1%})")
        st.markdown(f"- **CHALLENGE** (`{t_low:.2f} ≤ risk < {t_high:.2f}`): **{r['n_challenged']:,}** txns ({r['n_challenged']/n:.1%})")
        st.markdown(f"- **BLOCK** (`risk ≥ {t_high:.2f}`): **{r['n_blocked']:,}** txns ({r['n_blocked']/n:.1%})")
        st.markdown("---")
        st.markdown(f"**Best 2-Action Baseline Benchmark**:")
        st.markdown(f"- Block Threshold: `{b2['threshold']:.2f}`")
        st.markdown(f"- Fraud Recall: `{b2['recall']:.1%}` | Blocked Genuine: `{b2['legit_blocked']:,}`")
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 4: TRANSACTION RISK EXPLORER
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Transaction Score Explorer & Hard Decision Calls")

    raw_view = raw_df.copy()
    raw_view["amount_inr"] = raw_view["TransactionAmt"] * USD_TO_INR
    raw_view["action"] = np.where(raw_view["pred"] >= t_high, "BLOCK",
                          np.where(raw_view["pred"] >= t_low, "CHALLENGE", "ALLOW"))
    raw_view["outcome"] = np.where(raw_view["isFraud"] == 1, "Fraud", "Genuine")

    # Sample for scatter plot performance
    sample_df = raw_view.sample(min(3000, len(raw_view)), random_state=42)

    scatter_fig = px.scatter(
        sample_df,
        x="pred",
        y="amount_inr",
        color="action",
        symbol="outcome",
        color_discrete_map={"ALLOW": "#10b981", "CHALLENGE": "#f59e0b", "BLOCK": "#ef4444"},
        hover_data=["TransactionID", "pred", "amount_inr", "action", "outcome"],
        opacity=0.75,
        title="Transaction Risk Score vs. Amount (Sample of 3,000 Txns)"
    )
    scatter_fig.add_vline(x=t_low, line_dash="dash", line_color="#f59e0b", annotation_text=f"t_low={t_low:.2f}")
    scatter_fig.add_vline(x=t_high, line_dash="dash", line_color="#ef4444", annotation_text=f"t_high={t_high:.2f}")
    scatter_fig.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_title="Predicted Fraud Score",
        yaxis_title="Transaction Amount (Rs.)",
        yaxis_type="log"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("### 📋 Transaction Data Table")
    view_option = st.radio("Select Table View", ["Near Block Threshold (Hard Calls)", "Highest Risk Transactions"], horizontal=True)

    if view_option == "Near Block Threshold (Hard Calls)":
        near_df = raw_view.iloc[(raw_view["pred"] - t_high).abs().argsort()[:50]]
        disp_df = near_df[["TransactionID", "amount_inr", "pred", "action", "outcome"]].sort_values("pred", ascending=False)
    else:
        disp_df = raw_view.sort_values("pred", ascending=False).head(50)[["TransactionID", "amount_inr", "pred", "action", "outcome"]]

    disp_df.columns = ["Transaction ID", "Amount (Rs.)", "Risk Score", "Policy Action", "Actual Outcome"]

    st.dataframe(
        disp_df.style.format({
            "Risk Score": "{:.4f}",
            "Amount (Rs.)": "Rs. {:,.2f}"
        }),
        use_container_width=True,
        height=350
    )


# -----------------------------------------------------------------------------
# TAB 5: MODEL & FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("LightGBM Model Architecture & Feature Importance")
    st.caption("Key predictive signals driving the transaction risk score.")

    imp_path = f"{OUT}/feature_importance.csv"
    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path).head(15)

        imp_fig = px.bar(
            imp_df,
            x="gain",
            y="feature",
            orientation="h",
            title="Top 15 Predictive Features by Gain",
            color="gain",
            color_continuous_scale="Viridis"
        )
        imp_fig.update_layout(
            template="plotly_dark",
            height=450,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Feature Importance (Gain)",
            yaxis_title="Feature Name",
            showlegend=False
        )
        st.plotly_chart(imp_fig, use_container_width=True)
    else:
        st.info("Feature importance CSV not found.")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ LightGBM Hyperparameters")
        st.markdown("- **Objective**: Binary Classification")
        st.markdown("- **Leaves**: 128 | **Learning Rate**: 0.05")
        st.markdown("- **Feature Fraction**: 0.70 | **Bagging Fraction**: 0.80")
        st.markdown("- **Unbalanced Weighting**: Enabled")
        st.markdown("</div>", unsafe_allow_html=True)

    with m_col2:
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.markdown("#### 📈 Model Performance Metrics")
        st.markdown("- **ROC-AUC**: **0.9112**")
        st.markdown("- **PR-AUC**: **0.5178** (vs 3.5% baseline fraud rate)")
        st.markdown("- **Evaluation Strategy**: Time-ordered split (Train: first 80%, Test: last 20%)")
        st.markdown("- **Test Transactions**: 118,108 transactions")
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Razorpay Buildathon 2026 — Track 02 Submission | Fraud Risk Policy Explorer | Powered by Streamlit, LightGBM & Plotly"
)
