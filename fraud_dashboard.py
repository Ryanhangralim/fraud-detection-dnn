import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from utils.robust_scaler import RobustScaler  # noqa: F401

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state: initialise widget keys on first load only ─────────────────
for _i in range(1, 29):
    if f"form_v{_i}" not in st.session_state:
        st.session_state[f"form_v{_i}"] = 0.0
if "time_val" not in st.session_state:
    st.session_state["time_val"] = 50000.0
if "amount_val" not in st.session_state:
    st.session_state["amount_val"] = 50.0


# ── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for opt in ["sgd", "adam", "adabelief"]:
        path = Path("saved") / f"best_model_{opt}.pkl"
        if path.exists():
            models[opt] = joblib.load(path)
    return models


@st.cache_resource
def load_scaler():
    path = Path("saved/robust_scaler.pkl")
    return joblib.load(path) if path.exists() else None


models = load_models()
scaler = load_scaler()

# ── Sample data ───────────────────────────────────────────
SAMPLE_LEGIT = {
    "Time": 54684.0, "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782, "V5": -0.3383,
    "V6": 0.4624, "V7": 0.2396, "V8": 0.0987, "V9": 0.3638, "V10": 0.0908,
    "V11": -0.5516, "V12": -0.6178, "V13": -0.9913, "V14": -0.3112, "V15": 1.4682,
    "V16": -0.4704, "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669, "V25": 0.1285,
    "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
}

SAMPLE_FRAUD = {'Time': 406.0, 'V1': -2.3122265423263, 'V2': 1.95199201064158, 'V3': -1.60985073229769, 'V4': 3.9979055875468, 'V5': -0.522187864667764, 'V6': -1.42654531920595, 'V7': -2.53738730624579, 'V8': 1.39165724829804, 'V9': -2.77008927719433, 'V10': -2.77227214465915, 'V11': 3.20203320709635, 'V12': -2.89990738849473, 'V13': -0.595221881324605, 'V14': -4.28925378244217, 'V15': 0.389724120274487, 'V16': -1.14074717980657, 'V17': -2.83005567450437, 'V18': -0.0168224681808257, 'V19': 0.416955705037907, 'V20': 0.126910559061474, 'V21': 0.517232370861764, 'V22': -0.0350493686052974, 'V23': -0.465211076182388, 'V24': 0.320198198514526, 'V25': 0.0445191674731724, 'V26': 0.177839798284401, 'V27': 0.261145002567677, 'V28': -0.143275874698919, 'Amount': 0.0, 'Class': 1.0}

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("Credit Card Fraud Detection")
    st.divider()

    st.subheader("Model Selection")
    available = list(models.keys())
    opt_labels = {"sgd": "SGD (Momentum)", "adam": "Adam", "adabelief": "AdaBelief"}

    if available:
        selected_opt = st.radio(
            "Optimizer",
            available,
            format_func=lambda k: opt_labels.get(k, k.upper()),
        )
    else:
        st.error("⚠️ No models found in `saved/` folder.")
        selected_opt = None

    st.divider()
    st.subheader("Threshold")
    threshold = st.slider("τ", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    st.caption(f"Classification threshold: **{threshold:.2f}**")

    st.divider()
    st.caption(
        "**Dataset:** ULB Credit Card  \n"
        "**Architecture:** [30, 128, 64, 32, 1]  \n"
        "**Scaling:** RobustScaler (Time, Amount)  \n"
        "**Imbalance:** No SMOTE"
    )

# ── Main ──────────────────────────────────────────────────
st.title("Transaction Fraud Detector")
st.caption("Input transaction features below to predict whether it is legitimate or fraudulent.")
st.divider()

# ── Quick load / randomize controls (outside the form so they react immediately) ──
st.subheader("Quick Fill")
qcol1, qcol2, qcol3 = st.columns(3)

with qcol1:
    if st.button("✅ Load Legitimate Sample", use_container_width=True):
        for k in range(1, 29):
            st.session_state[f"form_v{k}"] = SAMPLE_LEGIT[f"V{k}"]
        st.session_state["time_val"] = SAMPLE_LEGIT["Time"]
        st.session_state["amount_val"] = SAMPLE_LEGIT["Amount"]

with qcol2:
    if st.button("⚠️ Load Fraudulent Sample", use_container_width=True):
        for k in range(1, 29):
            st.session_state[f"form_v{k}"] = SAMPLE_FRAUD[f"V{k}"]
        st.session_state["time_val"] = SAMPLE_FRAUD["Time"]
        st.session_state["amount_val"] = SAMPLE_FRAUD["Amount"]

with qcol3:
    if st.button("🎲 Randomize PCA Features", use_container_width=True):
        rng = np.random.default_rng()
        for k in range(1, 29):
            st.session_state[f"form_v{k}"] = float(rng.uniform(-3.0, 3.0))

st.divider()

# ── Input form ────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("Transaction Info")
    ti_col1, ti_col2 = st.columns(2)
    with ti_col1:
        time_val = st.number_input(
            "Time (seconds from first transaction)",
            key="time_val",
            format="%.2f",
            help="Raw Time value — will be scaled automatically",
        )
    with ti_col2:
        amount_val = st.number_input(
            "Amount",
            key="amount_val",
            format="%.2f",
            help="Raw Amount value — will be scaled automatically",
        )

    st.subheader("PCA Features V1 – V28")
    pca_inputs = {}
    grid_cols = st.columns(7)
    for i in range(1, 29):
        with grid_cols[(i - 1) % 7]:
            pca_inputs[f"V{i}"] = st.number_input(
                f"V{i}",
                format="%.4f",
                key=f"form_v{i}",
            )

    st.divider()
    submitted = st.form_submit_button("🔍  Run Prediction", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────
if submitted:
    if not selected_opt or not models:
        st.error("❌ No model selected. Please load models first.")
    else:
        final_vals = {"Time": time_val, "Amount": amount_val, **pca_inputs}

        feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        X_raw = np.array([[final_vals[c] for c in feature_cols]])

        if scaler is not None:
            scaled_ta = scaler.transform(X_raw[:, [0, 29]])
            X_scaled = X_raw.copy()
            X_scaled[:, 0] = scaled_ta[:, 0]
            X_scaled[:, 29] = scaled_ta[:, 1]
        else:
            X_scaled = X_raw
            st.warning("⚠️ Scaler not found. Using raw Time & Amount values.")

        model = models[selected_opt]
        try:
            prob = float(model.predict_proba(X_scaled).ravel()[0])
        except Exception:
            prob = float(model.predict_proba(X_scaled)[0])

        is_fraud = prob >= threshold
        fraud_pct = prob * 100
        legit_pct = 100 - fraud_pct

        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            if is_fraud:
                st.error(f"## ⚠️ FRAUD DETECTED\nTransaction flagged as fraudulent")
            else:
                st.success(f"## ✅ LEGITIMATE\nTransaction appears legitimate")

        with r2:
            st.metric("Fraud Probability", f"{fraud_pct:.2f}%")
            st.progress(prob)
            st.caption(f"Legitimate: **{legit_pct:.2f}%** | Fraud: **{fraud_pct:.2f}%**")

        with r3:
            opt_display = {"sgd": "SGD (Momentum)", "adam": "Adam", "adabelief": "AdaBelief"}
            st.metric("Optimizer", opt_display.get(selected_opt, selected_opt.upper()))
            st.metric("Threshold (τ)", f"{threshold:.2f}")
            st.metric("Raw Score", f"{prob:.6f}")

        # ── Compare all models ────────────────────────────
        if len(models) > 1:
            st.divider()
            st.subheader("All Models Comparison")
            compare_cols = st.columns(len(models))
            for idx, (opt_key, mdl) in enumerate(models.items()):
                try:
                    p = float(mdl.predict_proba(X_scaled).ravel()[0])
                except Exception:
                    p = float(mdl.predict_proba(X_scaled)[0])
                verdict = p >= threshold
                label = "🔴 FRAUD" if verdict else "🟢 LEGIT"
                is_active = opt_key == selected_opt
                with compare_cols[idx]:
                    st.metric(
                        label=f"{opt_display.get(opt_key, opt_key.upper())}{' ← selected' if is_active else ''}",
                        value=f"{p * 100:.2f}%",
                        delta=label,
                        delta_color="inverse" if verdict else "normal",
                    )

st.divider()
st.caption("Deep Neural Network · ULB Credit Card Dataset · Thesis Research")
