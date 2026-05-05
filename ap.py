import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MRB Contractor - Bid Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")

# =========================================================
# SKLEARN PICKLE COMPATIBILITY PATCH
# Fixes: AttributeError: Can't get attribute '_RemainderColsList'
# =========================================================
try:
    import sklearn.compose._column_transformer as _ct

    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
.block-container { padding: 1.2rem 2rem !important; }
h1 { font-size: 1.45rem !important; color: #1B2A4A; font-weight: 800; }
h2, h3 { color: #1B2A4A; }
.stMetric label { font-size: 12px !important; color: #64748B; }
.stMetric [data-testid="stMetricValue"] { font-size: 1.25rem !important; font-weight: 800; }
.stButton > button {
    background: #0D9488 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    width: 100%;
    padding: 10px !important;
}
.card {
    border-radius: 10px;
    padding: 14px 16px;
    border-left: 5px solid;
    margin: 8px 0;
    font-size: 13px;
}
[data-testid="stSidebar"] { background: #1B2A4A; }
[data-testid="stSidebar"] * { color: white !important; font-size: 13px; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SIMPLE LOGIN
# =========================================================
USERNAME = "MRB"
PASSWORD = "MRB123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login_screen():
    st.markdown("<h1 style='text-align:center;'>🏗️ MRB Contractor Login</h1>", unsafe_allow_html=True)
    st.caption("Login to access the Bid Intelligence Platform")
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


if not st.session_state.logged_in:
    login_screen()
    st.stop()

# =========================================================
# MODEL LOADING
# Keep this .py file and both .pkl files in the same folder.
# This loader accepts normal names and uploaded names with (1)/(2).
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def first_existing_file(file_names):
    for name in file_names:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    return None


@st.cache_resource(show_spinner="Loading ML models...")
def load_bundles():
    cost_path = first_existing_file([
        "cost_model_bundle.pkl",
        "cost_model_bundle(2).pkl",
        "cost_modell_bundle.pkl",
        "cost_modell_bundle(2).pkl",
    ])
    loss_path = first_existing_file([
        "modelloss_bundle.pkl",
        "modelloss_bundle(1).pkl",
        "win_loss_model_bundle.pkl",
        "win_loss_bundle.pkl",
    ])

    if cost_path is None:
        raise FileNotFoundError(
            "Cost model not found. Put cost_model_bundle.pkl in the same folder as this app."
        )

    cost_bundle = joblib.load(cost_path)
    loss_bundle = joblib.load(loss_path) if loss_path else None
    return cost_bundle, loss_bundle, os.path.basename(cost_path), os.path.basename(loss_path) if loss_path else "Not found"


try:
    cost_bundle, loss_bundle, cost_file_name, loss_file_name = load_bundles()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# =========================================================
# BUNDLE UNPACKING HELPERS
# =========================================================
def get_pipeline_categories(pipeline):
    """Return categorical values learned by OneHotEncoder inside ColumnTransformer."""
    try:
        pre = pipeline.named_steps["preprocessor"]
        cat_cols = pre.transformers_[0][2]
        cat_vals = pre.transformers_[0][1].categories_
        return {col: list(vals) for col, vals in zip(cat_cols, cat_vals)}
    except Exception:
        return {}


def clean_options(values, fallback):
    values = list(values) if values else list(fallback)
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        text = str(v).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or list(fallback)


cost_model = cost_bundle["model"]
COST_FEATURES = list(cost_bundle.get("features", []))
COST_TARGETS = list(cost_bundle.get("targets", []))
COST_METRICS = cost_bundle.get("metrics", {})

cost_cat_map = get_pipeline_categories(cost_model)

COST_STATES = clean_options(cost_cat_map.get("state"), ["AZ", "CA", "CO", "FL", "NC", "TX"])
COST_SYSTEMS = clean_options(cost_cat_map.get("system_type"), ["TPO", "COATING", "OTHER"])
COST_JOB_TYPES = clean_options(cost_cat_map.get("job_type"), ["Reroof", "Overlay / Recover", "Partial Tear Off", "Coating"])
COST_MATERIAL_TYPES = clean_options(cost_cat_map.get("material_type"), ["TPO", "COATING", "OTHER"])
COST_SLOPE_TYPES = clean_options(cost_cat_map.get("slope") or cost_cat_map.get("slope_type"), ["Low Slope", "Steep Slope"])

if loss_bundle:
    loss_model = loss_bundle["model"]
    LOSS_FEATURES = list(loss_bundle.get("features", []))
    LOSS_ACCURACY = loss_bundle.get("accuracy")
    loss_cat_map = get_pipeline_categories(loss_model)

    WIN_STATES = clean_options(loss_cat_map.get("state"), COST_STATES)
    WIN_JOB_TYPES = clean_options(loss_cat_map.get("job_type"), COST_JOB_TYPES)
    WIN_MATERIAL_TYPES = clean_options(loss_cat_map.get("material_type"), COST_MATERIAL_TYPES)
    WIN_SLOPE_TYPES = clean_options(loss_cat_map.get("slope_type") or loss_cat_map.get("slope"), ["LOW SLOPE", "STEEP SLOPE", "UNKNOWN"])

    try:
        LOSS_CLASSES = list(loss_model.classes_)
    except Exception:
        LOSS_CLASSES = list(loss_model.named_steps["model"].classes_)

    if 1 in LOSS_CLASSES:
        WIN_CLASS_INDEX = LOSS_CLASSES.index(1)
    elif "WIN" in LOSS_CLASSES:
        WIN_CLASS_INDEX = LOSS_CLASSES.index("WIN")
    else:
        WIN_CLASS_INDEX = len(LOSS_CLASSES) - 1
else:
    loss_model = None
    LOSS_FEATURES = []
    LOSS_ACCURACY = None
    LOSS_CLASSES = []
    WIN_CLASS_INDEX = 1
    WIN_STATES = COST_STATES
    WIN_JOB_TYPES = COST_JOB_TYPES
    WIN_MATERIAL_TYPES = COST_MATERIAL_TYPES
    WIN_SLOPE_TYPES = COST_SLOPE_TYPES

STATE_WIN_RATE = {
    "TX": 0.48, "CA": 0.40, "FL": 0.32, "AZ": 0.28,
    "CO": 0.38, "MS": 0.35, "NC": 0.42, "LA": 0.30,
    "KS": 0.33, "MI": 0.36, "OH": 0.37, "TN": 0.34,
}

# =========================================================
# BUSINESS LOGIC
# =========================================================
def infer_system_type(material_type):
    """Hide System Type from UI. Use Material Type to choose a valid system_type for the cost model."""
    mat = str(material_type or "").upper()
    systems_upper = {str(x).upper(): x for x in COST_SYSTEMS}

    if mat in systems_upper:
        return systems_upper[mat]
    if "TPO" in mat and "TPO" in systems_upper:
        return systems_upper["TPO"]
    if "COAT" in mat and "COATING" in systems_upper:
        return systems_upper["COATING"]
    if "OTHER" in systems_upper:
        return systems_upper["OTHER"]
    return COST_SYSTEMS[0]


def normalize_for_cost_options(value, options):
    """Use selected value if trained model knows it; otherwise use first valid trained option."""
    if value in options:
        return value
    value_upper = str(value).upper()
    for option in options:
        if str(option).upper() == value_upper:
            return option
    return options[0]


def normalize_for_loss_options(value, options):
    """Map UI value to the exact category used by the win/loss model."""
    if not options:
        return value
    if value in options:
        return value
    value_upper = str(value).strip().upper()
    for option in options:
        if str(option).strip().upper() == value_upper:
            return option
    # Common roofing synonyms/casing differences
    synonyms = {
        "OVERLAY / RECOVER": ["OVERLAY", "RECOVER", "OVERLAY / RECOVER"],
        "LOW SLOPE": ["LOW SLOPE", "LOW-SLOPE", "Low Slope"],
        "STEEP SLOPE": ["STEEP SLOPE", "STEEP-SLOPE", "Steep Slope"],
        "TPO": ["TPO"],
        "COATING": ["COATING", "COAT"],
    }
    for option in options:
        opt_upper = str(option).strip().upper()
        for canonical, aliases in synonyms.items():
            if value_upper in [str(a).upper() for a in aliases] and opt_upper in [str(a).upper() for a in aliases + [canonical]]:
                return option
    return options[0]


def cost_signature(state, sqft, job_type, material_type, slope_type):
    """One normalized key so both pages reuse the exact same cost estimate."""
    return (
        normalize_for_cost_options(state, COST_STATES),
        int(float(sqft)),
        normalize_for_cost_options(job_type, COST_JOB_TYPES),
        normalize_for_cost_options(material_type, COST_MATERIAL_TYPES),
        normalize_for_cost_options(slope_type, COST_SLOPE_TYPES),
    )


def predict_cost(state, sqft, job_type, material_type, slope_type):
    system_type = infer_system_type(material_type)

    row = {}
    for feat in COST_FEATURES:
        if feat == "state":
            row[feat] = normalize_for_cost_options(state, COST_STATES)
        elif feat == "system_type":
            row[feat] = normalize_for_cost_options(system_type, COST_SYSTEMS)
        elif feat == "job_type":
            row[feat] = normalize_for_cost_options(job_type, COST_JOB_TYPES)
        elif feat == "material_type":
            row[feat] = normalize_for_cost_options(material_type, COST_MATERIAL_TYPES)
        elif feat in ("slope", "slope_type"):
            row[feat] = normalize_for_cost_options(slope_type, COST_SLOPE_TYPES)
        elif feat == "sqft":
            row[feat] = float(sqft)
        elif feat == "log_sqft":
            row[feat] = float(np.log1p(float(sqft)))
        else:
            row[feat] = 0

    X = pd.DataFrame([row], columns=COST_FEATURES)
    pred = cost_model.predict(X)[0]
    result = dict(zip(COST_TARGETS, pred))

    material = max(float(result.get("actual_material_cost", 0)), 0)
    labor = max(float(result.get("actual_labor_cost", 0)), 0)
    other = max(float(result.get("actual_other_cost", 0)), 0)
    total = max(float(result.get("actual_total_cost", material + labor + other)), 0)

    # Keep category amounts consistent with total model output.
    subtotal = material + labor + other
    if subtotal > 0 and total > 0:
        scale = total / subtotal
        material *= scale
        labor *= scale
        other *= scale

    suggested_sell_price = total * 1.20
    margin_percent = ((suggested_sell_price - total) / suggested_sell_price * 100) if suggested_sell_price > 0 else 0

    return {
        "material": material,
        "labor": labor,
        "other": other,
        "total": total,
        "sell_price": suggested_sell_price,
        "margin_percent": margin_percent,
        "system_type_used": system_type,
    }


def predict_overrun(total_cost, sqft, bid_amount=None):
    """
    Estimate overrun buffer.

    Cost Estimator page:
        Uses cost_per_sqft only because no final bid is entered yet.

    Bid Predictor page:
        Uses bid margin as a safety buffer. A 20%+ margin should reduce
        overrun risk because the bid has room to absorb cost movement.
    """
    cost_per_sqft = total_cost / sqft if sqft else 0

    # Base cost-risk from model-estimated cost intensity.
    if cost_per_sqft > 10:
        base_rate = 0.12
    elif cost_per_sqft > 6:
        base_rate = 0.07
    else:
        base_rate = 0.05

    # If no bid is supplied, return pure cost risk.
    if bid_amount is None or bid_amount <= 0:
        rate = base_rate
        if rate >= 0.10:
            risk_label = "HIGH"
        elif rate >= 0.07:
            risk_label = "MEDIUM"
        else:
            risk_label = "LOW"
    else:
        margin = (bid_amount - total_cost) / bid_amount if bid_amount > 0 else 0

        # Margin buffer adjustment for bid page.
        if margin >= 0.20:
            rate = base_rate * 0.40
            risk_label = "LOW"
        elif margin >= 0.10:
            rate = base_rate * 0.70
            risk_label = "MEDIUM"
        else:
            rate = base_rate
            risk_label = "HIGH"

    return {
        "rate": rate,
        "probability": rate * 100,
        "amount": total_cost * rate,
        "risk_label": risk_label,
        "base_rate": base_rate,
    }


def business_win_probability_from_markup(state, estimated_cost, bid_amount):
    """
    Demo/business rule for win probability.

    The win model alone only sees raw bid_amount, so it cannot know whether the
    bid is cheap or expensive compared with the Cost Estimator. For the demo,
    we use estimated_cost as the baseline:
        - bid near estimated cost = high win chance
        - bid higher than estimated cost = lower win chance
        - bid below cost = very high win chance, but margin/overrun cards warn it is risky
    """
    if estimated_cost <= 0 or bid_amount <= 0:
        return 0.0

    markup = (float(bid_amount) - float(estimated_cost)) / float(estimated_cost)

    # Piecewise curve: close to cost should be around 80-90%; higher markup lowers win chance.
    if markup <= -0.05:
        prob = 93.0
    elif markup <= 0.00:
        prob = 90.0 - ((markup + 0.05) / 0.05) * 2.0      # 93 -> 88 approximately
    elif markup <= 0.05:
        prob = 88.0 - (markup / 0.05) * 8.0              # 88 -> 80
    elif markup <= 0.10:
        prob = 80.0 - ((markup - 0.05) / 0.05) * 15.0    # 80 -> 65
    elif markup <= 0.20:
        prob = 65.0 - ((markup - 0.10) / 0.10) * 25.0    # 65 -> 40
    elif markup <= 0.30:
        prob = 40.0 - ((markup - 0.20) / 0.10) * 20.0    # 40 -> 20
    elif markup <= 0.40:
        prob = 20.0 - ((markup - 0.30) / 0.10) * 10.0    # 20 -> 10
    else:
        prob = max(5.0, 10.0 - ((markup - 0.40) / 0.10) * 3.0)

    # Small market adjustment, but keep the close-to-cost behavior high.
    state_adj = (STATE_WIN_RATE.get(state, 0.35) - 0.35) * 20.0
    return float(np.clip(prob + state_adj, 3, 95))


def calc_win_probability(state, job_type, material_type, slope_type, bid_amount, estimated_cost=None):
    # Preferred demo/business behavior: compare bid against Cost Estimator amount.
    if estimated_cost is not None and estimated_cost > 0:
        return business_win_probability_from_markup(state, estimated_cost, bid_amount)

    # Fallback to trained model if estimated_cost is not available.
    if loss_model is not None:
        try:
            row = {}
            for feat in LOSS_FEATURES:
                if feat == "state":
                    row[feat] = state
                elif feat == "job_type":
                    row[feat] = job_type
                elif feat == "material_type":
                    row[feat] = material_type
                elif feat in ("slope", "slope_type"):
                    row[feat] = slope_type
                elif feat == "bid_amount":
                    row[feat] = float(bid_amount)
                else:
                    row[feat] = 0

            X = pd.DataFrame([row], columns=LOSS_FEATURES)
            probability = loss_model.predict_proba(X)[0][WIN_CLASS_INDEX] * 100
            return float(np.clip(probability, 0, 100))
        except Exception:
            pass

    return float(np.clip(STATE_WIN_RATE.get(state, 0.35) * 100, 0, 100))


def calc_win_probability_batch(state, job_type, material_type, slope_type, bid_amounts, estimated_cost=None):
    # Preferred demo/business behavior: vectorized markup curve.
    if estimated_cost is not None and estimated_cost > 0:
        return np.array([
            business_win_probability_from_markup(state, estimated_cost, b)
            for b in bid_amounts
        ], dtype=float)

    # Fallback to trained model if estimated_cost is not available.
    if loss_model is not None:
        try:
            n = len(bid_amounts)
            rows = {}
            for feat in LOSS_FEATURES:
                if feat == "state":
                    rows[feat] = [state] * n
                elif feat == "job_type":
                    rows[feat] = [job_type] * n
                elif feat == "material_type":
                    rows[feat] = [material_type] * n
                elif feat in ("slope", "slope_type"):
                    rows[feat] = [slope_type] * n
                elif feat == "bid_amount":
                    rows[feat] = list(bid_amounts)
                else:
                    rows[feat] = [0] * n

            X = pd.DataFrame(rows, columns=LOSS_FEATURES)
            probs = loss_model.predict_proba(X)[:, WIN_CLASS_INDEX] * 100
            return np.clip(probs.astype(float), 0, 100)
        except Exception:
            pass

    fallback = STATE_WIN_RATE.get(state, 0.35) * 100
    return np.full(len(bid_amounts), fallback)


def required_input_check(values):
    missing = [name for name, value in values.items() if value in [None, "", "— select —"] or value == 0]
    if missing:
        st.warning("Please fill in: " + ", ".join(missing))
        st.stop()

# =========================================================
# SIDEBAR - page is defined BEFORE page checks, so no NameError
# =========================================================
with st.sidebar:
    st.markdown("## 🏗️ MRB Contractor")
    st.caption("Bid Intelligence Platform")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🧮 Cost Estimator", "🎯 Bid & Win Predictor"],
        index=0,
    )
    st.markdown("---")
    st.caption(f"Cost model file: {cost_file_name}")
    st.caption(f"Win model file: {loss_file_name}")
    st.caption(f"Cost states: {', '.join(COST_STATES)}")
    st.caption(f"Win states: {', '.join(WIN_STATES)}")
    if LOSS_ACCURACY is not None:
        st.caption(f"Win model accuracy: {LOSS_ACCURACY * 100:.1f}%")
    with st.expander("Model debug"):
        st.write("COST_FEATURES", COST_FEATURES)
        st.write("COST_TARGETS", COST_TARGETS)
        st.write("LOSS_FEATURES", LOSS_FEATURES)
        st.write("LOSS_CLASSES", LOSS_CLASSES)
    st.caption("Prototype Phase 1 · Azure Databricks ML")
    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# =========================================================
# PAGE 1 - COST ESTIMATOR
# =========================================================
if page == "🧮 Cost Estimator":
    st.title("🧮 Cost Estimator")
    st.caption("Enter job details → get cost breakdown, sell price, margin, and overrun risk.")

    col_in, col_out = st.columns([1, 2.2])

    with col_in:
        with st.container(border=True):
            st.markdown("**📋 Job Details**")
            state = st.selectbox("State", ["— select —"] + COST_STATES, key="ce_state")
            job_type = st.selectbox("Job Type", ["— select —"] + COST_JOB_TYPES, key="ce_job")
            material_type = st.selectbox("Material Type", ["— select —"] + COST_MATERIAL_TYPES, key="ce_material")
            slope_type = st.selectbox("Slope Type", ["— select —"] + COST_SLOPE_TYPES, key="ce_slope")
            sqft = st.number_input(
                "Square Footage",
                min_value=0,
                max_value=5_000_000,
                value=0,
                step=1000,
                format="%d",
                key="ce_sqft",
            )
            run = st.button("Estimate Cost →", key="ce_run")

    with col_out:
        if not run:
            st.info("Enter job details and click **Estimate Cost**.")
        else:
            required_input_check({
                "State": state,
                "Job Type": job_type,
                "Material Type": material_type,
                "Slope Type": slope_type,
                "Square Footage": sqft,
            })

            cost = predict_cost(state, sqft, job_type, material_type, slope_type)
            # Save exact Cost Estimator result so Bid Predictor can reuse the same value.
            st.session_state["last_cost_signature"] = cost_signature(state, sqft, job_type, material_type, slope_type)
            st.session_state["last_cost_result"] = cost
            overrun = predict_overrun(cost["total"], sqft)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Material", f"${cost['material']:,.0f}")
            k2.metric("Labor", f"${cost['labor']:,.0f}")
            k3.metric("Other", f"${cost['other']:,.0f}")
            k4.metric("Estimated Total", f"${cost['total']:,.0f}")

            st.markdown("---")
            left, right = st.columns(2)

            with left:
                st.markdown("**📋 Full Breakdown**")
                rows = [
                    ["Material Cost", f"${cost['material']:,.0f}", f"${cost['material'] / sqft:.2f}"],
                    ["Labor Cost", f"${cost['labor']:,.0f}", f"${cost['labor'] / sqft:.2f}"],
                    ["Other Cost", f"${cost['other']:,.0f}", f"${cost['other'] / sqft:.2f}"],
                    ["Estimated Total", f"${cost['total']:,.0f}", f"${cost['total'] / sqft:.2f}"],
                    ["Suggested Sell Price", f"${cost['sell_price']:,.0f}", f"${cost['sell_price'] / sqft:.2f}"],
                    ["Predicted Margin", f"{cost['margin_percent']:.1f}%", "—"],
                    ["Expected Overrun Buffer", f"${overrun['amount']:,.0f}", f"${overrun['amount'] / sqft:.2f}"],
                    ["Expected Actual Cost", f"${cost['total'] + overrun['amount']:,.0f}", f"${(cost['total'] + overrun['amount']) / sqft:.2f}"],
                ]
                st.dataframe(
                    pd.DataFrame(rows, columns=["Item", "Amount", "Per Sqft"]),
                    hide_index=True,
                    use_container_width=True,
                    height=318,
                )
                st.caption(f"System type used internally: {cost['system_type_used']}")

            with right:
                fig = go.Figure(
                    go.Pie(
                        labels=["Material", "Labor", "Other"],
                        values=[cost["material"], cost["labor"], cost["other"]],
                        hole=0.55,
                        textinfo="label+percent",
                        textfont_size=12,
                    )
                )
                fig.update_layout(
                    title=dict(text="Cost Breakdown", font=dict(size=14)),
                    height=320,
                    paper_bgcolor="white",
                    margin=dict(t=35, b=0, l=0, r=0),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            risk = overrun["risk_label"]
            if risk == "HIGH":
                st.markdown(
                    f"""
                    <div class="card" style="border-left-color:#EF4444;background:#FEF2F2">
                    🔴 <strong style="color:#EF4444">HIGH OVERRUN RISK — {overrun['probability']:.0f}% probability</strong><br>
                    Add buffer: <b>${overrun['amount']:,.0f}</b> · Expected actual cost: <b>${cost['total'] + overrun['amount']:,.0f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif risk == "MEDIUM":
                st.markdown(
                    f"""
                    <div class="card" style="border-left-color:#F59E0B;background:#FFFBEB">
                    🟡 <strong style="color:#F59E0B">MEDIUM OVERRUN RISK — {overrun['probability']:.0f}% probability</strong><br>
                    Add buffer: <b>${overrun['amount']:,.0f}</b> · Expected actual cost: <b>${cost['total'] + overrun['amount']:,.0f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="card" style="border-left-color:#10B981;background:#F0FDF4">
                    🟢 <strong style="color:#10B981">LOW OVERRUN RISK — {overrun['probability']:.0f}% probability</strong><br>
                    Cost likely to stay close to estimate · Expected actual cost: <b>${cost['total'] + overrun['amount']:,.0f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# =========================================================
# PAGE 2 - BID & WIN PREDICTOR
# =========================================================
elif page == "🎯 Bid & Win Predictor":
    st.title("🎯 Bid & Win Predictor")
    st.caption("Enter bid details → get win probability, margin risk, best bid, and recommendations.")

    col_in, col_out = st.columns([1, 2.2])

    with col_in:
        with st.container(border=True):
            st.markdown("**📋 Bid Details**")
            # Use the SAME dropdown values as Cost Estimator so the cost amount matches exactly.
            state = st.selectbox("State", ["— select —"] + COST_STATES, key="bp_state")
            job_type = st.selectbox("Job Type", ["— select —"] + COST_JOB_TYPES, key="bp_job")
            material_type = st.selectbox("Material Type", ["— select —"] + COST_MATERIAL_TYPES, key="bp_material")
            slope_type = st.selectbox("Slope Type", ["— select —"] + COST_SLOPE_TYPES, key="bp_slope")
            sqft = st.number_input(
                "Square Footage",
                min_value=0,
                max_value=5_000_000,
                value=0,
                step=1000,
                format="%d",
                key="bp_sqft",
            )
            st.caption("Leave bid amount as 0 to use the Cost Estimator amount automatically.")
            bid_amount_input = st.number_input(
                "Your Bid Amount ($) - optional",
                min_value=0,
                max_value=50_000_000,
                value=0,
                step=10000,
                format="%d",
                key="bp_bid",
            )
            run = st.button("Predict Win →", key="bp_run")

    with col_out:
        if not run:
            st.info("Enter bid details and click **Predict Win**.")
        else:
            required_input_check({
                "State": state,
                "Job Type": job_type,
                "Material Type": material_type,
                "Slope Type": slope_type,
                "Square Footage": sqft,
            })

            # Build cost with the exact same function/inputs as Cost Estimator.
            # If user already ran Cost Estimator with the same details, reuse that exact saved number.
            sig = cost_signature(state, sqft, job_type, material_type, slope_type)
            if st.session_state.get("last_cost_signature") == sig:
                cost = st.session_state["last_cost_result"]
                cost_reuse_note = "Matched last Cost Estimator result"
            else:
                cost = predict_cost(state, sqft, job_type, material_type, slope_type)
                cost_reuse_note = "Calculated from same Cost Estimator logic"

            estimated_cost_bid = float(cost["total"])
            bid_amount = float(bid_amount_input) if bid_amount_input and bid_amount_input > 0 else estimated_cost_bid
            bid_source = "Manual bid" if bid_amount_input and bid_amount_input > 0 else "Cost Estimator amount"

            # Overrun risk on bid page must include margin buffer.
            # Example: if estimated cost is 100k and bid is 127k, margin is ~20%,
            # so overrun risk should reduce instead of staying HIGH.
            overrun = predict_overrun(cost["total"], sqft, bid_amount)

            # Map the same UI inputs to exact win/loss model categories only for win-probability scoring.
            win_state = normalize_for_loss_options(state, WIN_STATES)
            win_job_type = normalize_for_loss_options(job_type, WIN_JOB_TYPES)
            win_material_type = normalize_for_loss_options(material_type, WIN_MATERIAL_TYPES)
            win_slope_type = normalize_for_loss_options(slope_type, WIN_SLOPE_TYPES)

            win_prob = calc_win_probability(win_state, win_job_type, win_material_type, win_slope_type, bid_amount, estimated_cost_bid)
            your_margin = ((bid_amount - cost["total"]) / bid_amount * 100) if bid_amount > 0 else 0
            state_rate = STATE_WIN_RATE.get(state, 0.35)

            # Best bid scan: tied to user bid and model-estimated cost instead of fixed 50k-2M only.
            low = max(cost["total"] * 0.90, bid_amount * 0.60, 10_000)
            high = max(cost["total"] * 1.60, bid_amount * 1.40, low + 50_000)
            scan_bids = np.linspace(low, high, 350)
            scan_probs = calc_win_probability_batch(win_state, win_job_type, win_material_type, win_slope_type, scan_bids, estimated_cost_bid)

            # Choose highest win probability, but require non-negative margin where possible.
            margins = (scan_bids - cost["total"]) / scan_bids * 100
            valid = margins >= 0
            if valid.any():
                valid_indices = np.where(valid)[0]
                best_idx = valid_indices[np.argmax(scan_probs[valid])]
            else:
                best_idx = int(np.argmax(scan_probs))

            best_bid = float(scan_bids[best_idx])
            best_win_prob = float(scan_probs[best_idx])
            best_margin = ((best_bid - cost["total"]) / best_bid * 100) if best_bid > 0 else 0

            if win_prob >= 65:
                signal, color, emoji = "STRONG BID", "#10B981", "🟢"
            elif win_prob >= 40:
                signal, color, emoji = "MODERATE BID", "#F59E0B", "🟡"
            else:
                signal, color, emoji = "HIGH RISK BID", "#EF4444", "🔴"

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Estimated Cost", f"${cost['total']:,.0f}")
            k2.metric("Bid Used", f"${bid_amount:,.0f}", bid_source)
            k3.metric("Win Probability", f"{win_prob:.0f}%")
            k4.metric("Your Margin", f"{your_margin:.1f}%")
            k5.metric("State Win Rate", f"{state_rate * 100:.0f}%")

            st.metric("AI Best Bid", f"${best_bid:,.0f}", f"{best_win_prob:.0f}% win prob · {best_margin:.1f}% margin")
            st.caption(f"Cost value source: {cost_reuse_note}. If Bid Amount is 0, Bid Used = Estimated Cost exactly.")
            st.caption("Win probability uses Cost Estimator baseline: bid near estimated cost = higher win chance; higher bid = lower win chance.")
            st.caption("Overrun risk uses margin buffer: higher margin lowers financial overrun risk.")

            st.markdown("---")
            left, right = st.columns(2)

            with left:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=win_prob,
                        number={"suffix": "%", "font": {"size": 42, "color": color}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color, "thickness": 0.28},
                            "steps": [
                                {"range": [0, 40], "color": "#FEE2E2"},
                                {"range": [40, 65], "color": "#FEF3C7"},
                                {"range": [65, 100], "color": "#D1FAE5"},
                            ],
                        },
                        title={"text": f"{emoji} Win Probability", "font": {"size": 14}},
                    )
                )
                fig.update_layout(height=240, paper_bgcolor="white", margin=dict(t=50, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    f"""
                    <div class="card" style="border-left-color:{color};background:{color}12">
                    <strong style="color:{color}">{emoji} {signal}</strong><br>
                    Win probability: <b>{win_prob:.0f}%</b> · Margin: <b>{your_margin:.1f}%</b> ·
                    Overrun: <b>{overrun['risk_label']}</b> ({overrun['probability']:.0f}%)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown("**📐 Price Sensitivity**")
                adjustments = [-0.10, -0.05, 0, 0.05, 0.10]
                sensitivity_bids = np.array([bid_amount * (1 + adj) for adj in adjustments])
                sensitivity_probs = calc_win_probability_batch(win_state, win_job_type, win_material_type, win_slope_type, sensitivity_bids, estimated_cost_bid)

                table = []
                for adj, test_bid, prob in zip(adjustments, sensitivity_bids, sensitivity_probs):
                    margin = ((test_bid - cost["total"]) / test_bid * 100) if test_bid > 0 else 0
                    table.append({
                        "Change": f"{adj * 100:+.0f}%",
                        "Bid": f"${test_bid:,.0f}",
                        "Margin": f"{margin:.1f}%",
                        "Win Prob": f"{prob:.0f}%",
                        "Signal": "🟢" if prob >= 65 else ("🟡" if prob >= 40 else "🔴"),
                        "Current": "◀ bid used" if adj == 0 else "",
                    })

                st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("**💡 AI Recommendations**")
            rec1, rec2 = st.columns(2)

            with rec1:
                diff = best_bid - bid_amount
                action = "increase" if diff > 0 else "reduce"
                if abs(diff) > max(10_000, bid_amount * 0.05):
                    st.markdown(
                        f"""
                        <div class="card" style="border-left-color:#0D9488;background:#F0FDF4">
                        💡 <strong>Recommended bid: ${best_bid:,.0f}</strong><br>
                        {action.title()} by <b>${abs(diff):,.0f}</b>. Expected win probability: <b>{best_win_prob:.0f}%</b>.
                        Estimated margin at this bid: <b>{best_margin:.1f}%</b>.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="card" style="border-left-color:#10B981;background:#F0FDF4">
                        ✅ <strong>Your bid is close to the model-recommended range</strong><br>
                        Current bid and best bid are within about 5%.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if your_margin < 0:
                    st.markdown(
                        f"""
                        <div class="card" style="border-left-color:#EF4444;background:#FEF2F2">
                        🔴 <strong>Negative margin risk</strong><br>
                        Your bid is below estimated cost. Estimated cost is <b>${cost['total']:,.0f}</b>.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif your_margin > 25:
                    st.markdown(
                        f"""
                        <div class="card" style="border-left-color:#F59E0B;background:#FFFBEB">
                        🟡 <strong>High margin may reduce win chance</strong><br>
                        Margin is <b>{your_margin:.1f}%</b>. Consider checking competitiveness.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with rec2:
                if state_rate < 0.35:
                    st.markdown(
                        f"""
                        <div class="card" style="border-left-color:#F59E0B;background:#FFFBEB">
                        🟡 <strong>{state} is a lower win-rate market</strong><br>
                        Historical baseline is around <b>{state_rate * 100:.0f}%</b>. Price tighter or review similar jobs.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if overrun["risk_label"] == "HIGH":
                    st.markdown(
                        f"""
                        <div class="card" style="border-left-color:#EF4444;background:#FEF2F2">
                        ⚠️ <strong>High overrun buffer suggested</strong><br>
                        Current margin does not provide enough buffer. Add about <b>${overrun['amount']:,.0f}</b> contingency. Expected actual cost: <b>${cost['total'] + overrun['amount']:,.0f}</b>.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="card" style="border-left-color:#10B981;background:#F0FDF4">
                        🟢 <strong>Overrun risk is {overrun['risk_label']}</strong><br>
                        Suggested buffer: <b>${overrun['amount']:,.0f}</b>.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("Estimated cost used for bid and margin calculation"):
                st.write(pd.DataFrame([
                    {"Item": "Estimated Cost", "Amount": f"${cost['total']:,.0f}", "Per Sqft": f"${cost['total'] / sqft:.2f}"},
                    {"Item": "Material", "Amount": f"${cost['material']:,.0f}", "Per Sqft": f"${cost['material'] / sqft:.2f}"},
                    {"Item": "Labor", "Amount": f"${cost['labor']:,.0f}", "Per Sqft": f"${cost['labor'] / sqft:.2f}"},
                    {"Item": "Other", "Amount": f"${cost['other']:,.0f}", "Per Sqft": f"${cost['other'] / sqft:.2f}"},
                ]))
                st.caption(f"Win model categories used: state={win_state}, job_type={win_job_type}, material_type={win_material_type}, slope={win_slope_type}")
