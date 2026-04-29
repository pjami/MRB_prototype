import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="MRB Contractor — Bid Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding: 1.2rem 2rem !important; }
    h1  { font-size: 1.3rem !important; color: #1B2A4A; font-weight: 800; }
    .stMetric label { font-size: 11px !important; color: #64748B; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.2rem !important; font-weight: 700; }
    .stButton > button {
        background: #0D9488 !important; color: white !important;
        border: none !important; border-radius: 8px !important;
        font-size: 14px !important; font-weight: 600 !important;
        width: 100%; padding: 10px !important;
    }
    .card {
        border-radius: 10px; padding: 14px 16px;
        border-left: 5px solid; margin: 8px 0;
        font-size: 13px;
    }
    [data-testid="stSidebar"] { background: #1B2A4A; }
    [data-testid="stSidebar"] * { color: white !important; font-size: 13px; }
    [data-testid="stSidebar"] .stRadio label { font-size: 15px !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Simple Login ───────────────────────────────────────────────
USERNAME = "MRB"
PASSWORD = "MRB123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_screen():
    st.markdown("<h1 style='text-align:center;'>🏗️ MRB Contractor Login</h1>", unsafe_allow_html=True)
    st.caption("Please login to access the Bid Intelligence Platform")

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.container(border=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username == USERNAME and password == PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")

# Stop app here until user logs in
if not st.session_state.logged_in:
    login_screen()
    st.stop()

# ── Load Models ───────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_bundle():
    with open(os.path.join(BASE, "model_bundle.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_win_model():
    path = os.path.join(BASE, "win_loss_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

bundle         = load_bundle()
models         = bundle["models"]
risk_model     = bundle["risk_model"]
overrun_model  = bundle["overrun_model"]
margin_model   = bundle["margin_model"]
le_state       = bundle["le_state"]
le_system      = bundle["le_system"]
STATES         = bundle["state_classes"]
SYSTEMS        = bundle["system_classes"]
FEATURES       = bundle["features"]
win_loss_model = load_win_model()   # loaded (currently feature-names array only)

# ── State win rates ───────────────────────────────────────────
BASE_WR = {
    "TX": 0.48, "CA": 0.40, "FL": 0.32, "AZ": 0.28,
    "CO": 0.38, "MS": 0.35, "NC": 0.42, "LA": 0.30
}

# ── Predict ───────────────────────────────────────────────────
def predict(state, sqft, system):
    sc = le_state.transform([state])[0]
    vc = le_system.transform([system])[0]
    X  = pd.DataFrame(
        [[sqft, np.log1p(sqft), sc, vc]],
        columns=FEATURES
    )

    est  = max(models["estimated_total_cost"].predict(X)[0],   0)
    mat  = max(models["budgeted_material_cost"].predict(X)[0], 0)
    lab  = max(models["budgeted_labor_cost"].predict(X)[0],    0)
    oth  = max(models["budgeted_other_cost"].predict(X)[0],    0)

    # FIX 1: The cost sub-models are trained independently, so
    # mat + lab + oth does not equal est. Scale the three components
    # so they sum exactly to est (the most reliable single output).
    raw_total = mat + lab + oth
    if raw_total > 0:
        scale = est / raw_total
        mat   = mat * scale
        lab   = lab * scale
        oth   = oth * scale
    # After scaling:  mat + lab + oth == est  ✓

    return dict(
        est   = est,
        mat   = mat,
        lab   = lab,
        oth   = oth,
        sell  = max(models["sell_price"].predict(X)[0], 0),
        marg  = margin_model.predict(X)[0],
        risk  = int(risk_model.predict(X)[0]),
        rprob = risk_model.predict_proba(X)[0][1] * 100,
        ova   = overrun_model.predict(X)[0],
    )

# ── Win probability ───────────────────────────────────────────
# FIX 2: win_loss_model.pkl contains only feature names — the actual
# model weights are missing. Use the rule-based formula which matches
# the original business logic exactly.
def calc_win_prob(ratio, your_margin, state_wr):
    if   ratio > 1.20: base = 0.12
    elif ratio > 1.12: base = 0.28
    elif ratio > 1.05: base = 0.45
    elif ratio > 0.98: base = 0.62
    elif ratio > 0.92: base = 0.75
    else:              base = 0.85
    if your_margin > 28:   base *= 0.70
    elif your_margin > 24: base *= 0.85
    return min(max(base * (state_wr + 0.55), 0.05) * 100, 94)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ MRB Contractor")
    st.caption("Bid Intelligence Platform")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🧮  Cost Estimator",
        "🎯  Bid & Win Predictor",
    ])
    st.markdown("---")
    st.caption(f"States: {', '.join(STATES)}")
    st.caption(f"Systems: {', '.join(SYSTEMS)}")
    st.caption("Prototype Phase 1 · Azure Databricks ML")
    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE 1 — COST ESTIMATOR
# ══════════════════════════════════════════════════════════════
if page == "🧮  Cost Estimator":
    st.title("🧮 Cost Estimator")
    st.caption("Enter job details → get full cost breakdown, sell price, margin & overrun risk.")

    col_in, col_out = st.columns([1, 2.2])

    with col_in:
        with st.container(border=True):
            st.markdown("**📋 Job Details**")
            state  = st.selectbox("State",       STATES,  index=list(STATES).index("TX"))
            system = st.selectbox("System Type", SYSTEMS, index=list(SYSTEMS).index("TPO"))
            sqft   = st.number_input("Square Footage", 1000, 1000000, 100000, 5000,
                                      format="%d")
            st.markdown("")
            run = st.button("Estimate Cost →")

    with col_out:
        if run:
            r = predict(state, sqft, system)

            # ── KPI strip ──
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Material",        f"${r['mat']:,.0f}")
            k2.metric("Labor",           f"${r['lab']:,.0f}")
            k3.metric("Other",           f"${r['oth']:,.0f}")
            k4.metric("Estimated Total", f"${r['est']:,.0f}")

            st.markdown("---")
            left, right = st.columns(2)

            with left:
                st.markdown("**📋 Full Breakdown**")
                rows = [
                    ("Material Cost",        f"${r['mat']:,.0f}",          f"${r['mat']/sqft:.2f}"),
                    ("Labor Cost",           f"${r['lab']:,.0f}",          f"${r['lab']/sqft:.2f}"),
                    ("Other Cost",           f"${r['oth']:,.0f}",          f"${r['oth']/sqft:.2f}"),
                    ("Estimated Total",      f"${r['est']:,.0f}",          f"${r['est']/sqft:.2f}"),
                    ("Suggested Sell Price", f"${r['sell']:,.0f}",         f"${r['sell']/sqft:.2f}"),
                    ("Predicted Margin",     f"{r['marg']:.1f}%",          "—"),
                    ("Expected Overrun",     f"${r['ova']:,.0f}",          f"${r['ova']/sqft:.2f}"),
                    ("Expected Actual Cost", f"${r['est']+r['ova']:,.0f}", f"${(r['est']+r['ova'])/sqft:.2f}"),
                ]
                tbl = pd.DataFrame(rows, columns=["Item", "Amount", "Per Sqft"])
                st.dataframe(tbl, hide_index=True, use_container_width=True, height=318)

            with right:
                # Cost breakdown donut
                fig = go.Figure(go.Pie(
                    labels=["Material", "Labor", "Other"],
                    values=[r["mat"], r["lab"], r["oth"]],
                    hole=0.55,
                    marker_colors=["#1B2A4A", "#0D9488", "#F59E0B"],
                    textinfo="label+percent",
                    textfont_size=12,
                ))
                fig.update_layout(
                    title=dict(text="Cost Breakdown", font=dict(size=13)),
                    height=220, paper_bgcolor="white",
                    margin=dict(t=35, b=0, l=0, r=0),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Sell price scenarios bar chart
                adjs   = [-0.10, -0.05, 0, +0.05, +0.10]
                labels = [f"{a*100:+.0f}%" for a in adjs]
                prices = [r["sell"] * (1 + a) for a in adjs]
                margins= [(p - r["est"]) / p * 100 if p > 0 else 0 for p in prices]
                colors = ["#10B981" if p <= r["sell"] else "#F59E0B" for p in prices]

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=labels, y=prices,
                    marker_color=colors,
                    text=[f"${p:,.0f}<br>{m:.1f}%" for p, m in zip(prices, margins)],
                    textposition="outside", textfont_size=10,
                ))
                fig2.update_layout(
                    title=dict(text="Sell Price Scenarios", font=dict(size=13)),
                    height=210, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=35, b=0, l=0, r=0),
                    yaxis_title="Sell Price ($)", xaxis_title="Adjustment",
                    font=dict(size=11),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── Overrun risk banner ──
            if r["risk"]:
                st.markdown(f"""<div class="card" style="border-left-color:#EF4444;background:#FEF2F2">
                    🔴 <strong style="color:#EF4444">HIGH OVERRUN RISK — {r['rprob']:.0f}% probability</strong><br>
                    <span style="font-size:12px">
                    Budget an extra <b>${r['ova']:,.0f}</b> buffer &nbsp;·&nbsp;
                    Expected actual cost: <b>${r['est']+r['ova']:,.0f}</b>
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="card" style="border-left-color:#10B981;background:#F0FDF4">
                    🟢 <strong style="color:#10B981">LOW OVERRUN RISK — {r['rprob']:.0f}% probability</strong><br>
                    <span style="font-size:12px">
                    Cost likely to stay close to estimate &nbsp;·&nbsp;
                    Expected actual: <b>${r['est']+r['ova']:,.0f}</b>
                    </span>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — BID & WIN PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "🎯  Bid & Win Predictor":
    st.title("🎯 Bid & Win Predictor")
    st.caption("Enter your bid details → get win probability, risk rate, best bid & recommendations.")

    col_in, col_out = st.columns([1, 2.2])

    with col_in:
        with st.container(border=True):
            st.markdown("**📋 Bid Details**")
            state     = st.selectbox("State",       STATES,  index=list(STATES).index("TX"))
            system    = st.selectbox("System Type", SYSTEMS, index=list(SYSTEMS).index("TPO"))
            sqft      = st.number_input("Square Footage", 1000, 1000000, 100000, 5000,
                                         format="%d")
            bid_price = st.number_input("Your Bid Amount ($)", 10000, 10000000,
                                         700000, 10000, format="%d")
            st.markdown("")
            run = st.button("Predict Win →")

    with col_out:
        if run:
            r = predict(state, sqft, system)

            your_margin = (bid_price - r["est"]) / bid_price * 100 if bid_price > 0 else 0
            ratio       = bid_price / r["sell"] if r["sell"] > 0 else 1.0
            state_wr    = BASE_WR.get(state, 0.35)

            win_prob    = calc_win_prob(ratio, your_margin, state_wr)
            best_bid    = r["sell"] * (0.96 if state_wr < 0.35 else 0.99)
            best_margin = (best_bid - r["est"]) / best_bid * 100 if best_bid > 0 else 0

            if win_prob >= 65:   sig, color, em = "STRONG BID",    "#10B981", "🟢"
            elif win_prob >= 40: sig, color, em = "MODERATE",      "#F59E0B", "🟡"
            else:                sig, color, em = "HIGH RISK BID", "#EF4444", "🔴"

            # ── KPI strip ──
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Win Probability", f"{win_prob:.0f}%")
            k2.metric("Your Margin",     f"{your_margin:.1f}%")
            k3.metric("AI Best Bid",     f"${best_bid:,.0f}")
            k4.metric("State Win Rate",  f"{state_wr*100:.0f}%")

            st.markdown("---")
            left, right = st.columns(2)

            with left:
                # Win probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=win_prob,
                    number={"suffix": "%", "font": {"size": 44, "color": color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"size": 10}},
                        "bar":  {"color": color, "thickness": 0.28},
                        "steps": [
                            {"range": [0,  40],  "color": "#FEE2E2"},
                            {"range": [40, 65],  "color": "#FEF3C7"},
                            {"range": [65, 100], "color": "#D1FAE5"},
                        ],
                        "threshold": {
                            "line":      {"color": "#1B2A4A", "width": 3},
                            "thickness": 0.8,
                            "value":     win_prob
                        }
                    },
                    title={"text": f"{em}  Win Probability", "font": {"size": 14, "color": "#1B2A4A"}}
                ))
                fig.update_layout(height=230, paper_bgcolor="white",
                                  margin=dict(t=50, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

                # Signal card
                st.markdown(f"""<div class="card"
                    style="border-left-color:{color};background:{color}12">
                    <strong style="color:{color}">{em} {sig}</strong><br>
                    <span style="font-size:12px">
                    Win prob: <b>{win_prob:.0f}%</b> &nbsp;·&nbsp;
                    Overrun risk: <b>{"🔴 HIGH" if r["risk"] else "🟢 LOW"}</b>
                    ({r["rprob"]:.0f}%) &nbsp;·&nbsp;
                    State rate: <b>{state_wr*100:.0f}%</b>
                    </span>
                </div>""", unsafe_allow_html=True)

            with right:
                # Bid comparison bar chart
                fig2 = go.Figure(go.Bar(
                    x=["Your Bid", "AI Best Bid", "Est. Cost", "Suggested Sell"],
                    y=[bid_price, best_bid, r["est"], r["sell"]],
                    marker_color=["#F59E0B", "#10B981", "#1B2A4A", "#0D9488"],
                    text=[f"${v:,.0f}" for v in [bid_price, best_bid, r["est"], r["sell"]]],
                    textposition="outside", textfont_size=11,
                ))
                fig2.update_layout(
                    title=dict(text="Bid vs AI Best Bid vs Cost", font=dict(size=13)),
                    height=250, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=35, b=0, l=0, r=0),
                    yaxis_title="Amount ($)", font=dict(size=11),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")

            # ── Price sensitivity table ──
            left2, right2 = st.columns(2)

            with left2:
                st.markdown("**📐 Price Sensitivity**")
                sens = []
                for adj in [-0.10, -0.05, 0, +0.05, +0.10]:
                    sp = bid_price * (1 + adj)
                    mg = (sp - r["est"]) / sp * 100 if sp > 0 else 0
                    rt = sp / r["sell"] if r["sell"] > 0 else 1
                    wp = calc_win_prob(rt, mg, state_wr)
                    sens.append({
                        "Change":   f"{adj*100:+.0f}%",
                        "Bid":      f"${sp:,.0f}",
                        "Margin":   f"{mg:.1f}%",
                        "Win Prob": f"{wp:.0f}%",
                        "Signal":   "🟢" if wp >= 65 else ("🟡" if wp >= 40 else "🔴"),
                        "":         "◀ current" if adj == 0 else "",
                    })
                st.dataframe(pd.DataFrame(sens), hide_index=True,
                             use_container_width=True)

            with right2:
                st.markdown("**💡 AI Recommendations**")

                # FIX 3: use a flag so the fallback card only appears
                # when no other recommendation card was shown.
                any_card = False

                if bid_price > best_bid * 1.05:
                    diff = bid_price - best_bid
                    st.markdown(f"""<div class="card" style="border-left-color:#EF4444;background:#FEF2F2">
                        🔴 <strong>Bid is ${diff:,.0f} above AI best</strong><br>
                        <span style="font-size:12px">Reduce to <b>${best_bid:,.0f}</b>
                        → win prob improves significantly.
                        Margin stays at <b>{best_margin:.1f}%</b> — still profitable.</span>
                    </div>""", unsafe_allow_html=True)
                    any_card = True

                if your_margin > 25:
                    st.markdown(f"""<div class="card" style="border-left-color:#F59E0B;background:#FFFBEB">
                        🟡 <strong>Margin {your_margin:.1f}% is above 25%</strong><br>
                        <span style="font-size:12px">Win rate drops sharply above 25%.
                        Target <b>18–24%</b> for best results.</span>
                    </div>""", unsafe_allow_html=True)
                    any_card = True

                if state_wr < 0.35:
                    st.markdown(f"""<div class="card" style="border-left-color:#F59E0B;background:#FFFBEB">
                        🟡 <strong>{state} has only {state_wr*100:.0f}% win rate</strong><br>
                        <span style="font-size:12px">Competitive market.
                        Price tighter here or focus on higher win-rate states.</span>
                    </div>""", unsafe_allow_html=True)
                    any_card = True

                if r["risk"]:
                    st.markdown(f"""<div class="card" style="border-left-color:#EF4444;background:#FEF2F2">
                        ⚠️ <strong>High overrun risk ({r["rprob"]:.0f}%)</strong><br>
                        <span style="font-size:12px">Add <b>${r["ova"]:,.0f}</b> buffer to your cost plan.
                        Actual cost may reach <b>${r["est"]+r["ova"]:,.0f}</b>.</span>
                    </div>""", unsafe_allow_html=True)
                    any_card = True

                if win_prob >= 65 and your_margin <= 25 and not r["risk"]:
                    st.markdown(f"""<div class="card" style="border-left-color:#10B981;background:#F0FDF4">
                        🟢 <strong>Bid looks strong — ready to submit</strong><br>
                        <span style="font-size:12px">Win probability <b>{win_prob:.0f}%</b>,
                        margin <b>{your_margin:.1f}%</b>, low overrun risk. ✅</span>
                    </div>""", unsafe_allow_html=True)
                    any_card = True

                if not any_card:
                    st.markdown("""<div class="card" style="border-left-color:#0D9488;background:#F0FDF4">
                        ✅ <strong>Bid parameters look reasonable</strong><br>
                        <span style="font-size:12px">No major issues detected. Review the sensitivity table.</span>
                    </div>""", unsafe_allow_html=True)
