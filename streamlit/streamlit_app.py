import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
#  LOAD MODEL & TRANSFORMER (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Try a few likely locations for the Model folder, since the deployment
    # structure (repo root vs. subfolder) can differ from local dev.
    candidate_dirs = [
        os.path.join(BASE_DIR, "Model"),            # ./Model (same folder as script)
        os.path.join(BASE_DIR, "..", "Model"),       # ../Model (script in a subfolder)
        os.path.join(BASE_DIR, "..", "..", "Model"),
        "Model",                                     # relative to current working dir
    ]


    for d in candidate_dirs:
        m = os.path.join(d, "stroke_prediction.keras")
        t = os.path.join(d, "column_trans.pkl")
        if os.path.exists(m) and os.path.exists(t):
            model_path, trans_path = m, t
            break

    if model_path is None:
        st.error(
            "❌ Could not locate `stroke_prediction.keras` / `column_trans.pkl`.\n\n"
            f"Script location: `{BASE_DIR}`\n\n"
            "Checked these folders:\n" +
            "\n".join(f"- `{os.path.normpath(d)}`" for d in candidate_dirs) +
            "\n\nMake sure a `Model/` folder containing both files is committed to your "
            "repo at one of the paths above."
        )
        st.stop()

    model     = tf.keras.models.load_model(model_path)
    col_trans = joblib.load(trans_path)
    return model, col_trans

model, col_trans = load_model()

# ─────────────────────────────────────────────────────────
#  STYLES  —  dark, high-contrast palette (no plain white)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }

/* App background — deep navy-to-indigo gradient */
.stApp {
    background: linear-gradient(150deg, #0f172a 0%, #1e1b4b 50%, #1a1033 100%);
}

/* Hero */
.hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.hero h1 {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 500;
}

/* Cards — dark slate, soft glow border */
.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(56,189,248,0.08);
}
.card-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #e2e8f0;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #334155;
}

/* Labels */
label, .stMarkdown p {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Inputs */
.stNumberInput input {
    background: #0f172a !important;
    border: 1.5px solid #475569 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    color: #e2e8f0 !important;
}
.stSelectbox > div > div {
    background: #0f172a !important;
    border: 1.5px solid #475569 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSelectbox svg { fill: #94a3b8 !important; }

/* Radio pills */
div[role="radiogroup"] label {
    background: #0f172a;
    border: 1.5px solid #6366f1;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    margin-right: 0.4rem;
    color: #c7d2fe !important;
    font-size: 0.88rem !important;
}
div[role="radiogroup"] label:hover { background: #312e81; }

/* Predict button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
    color: #f5f3ff !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.75rem;
    border-radius: 14px;
    border: none;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4);
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(124,58,237,0.55);
}

/* Result cards — dark tinted, high contrast text */
.result-low {
    background: linear-gradient(145deg, #052e1f, #06281f);
    border: 2px solid #10b981;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
}
.result-high {
    background: linear-gradient(145deg, #2f0a0a, #300a12);
    border: 2px solid #ef4444;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
}
.result-label {
    font-family: 'Poppins', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
}
.result-prob {
    font-size: 1rem;
    color: #cbd5e1;
    font-weight: 500;
    margin-top: 0.3rem;
}

/* Risk meter label */
.meter-label {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: #cbd5e1;
    margin-bottom: 0.2rem;
}

/* Expander (risk factors) */
.streamlit-expanderHeader {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
.streamlit-expanderContent {
    background: #0f172a !important;
    color: #e2e8f0 !important;
}

/* Alert boxes (warning/success) text readability */
div[data-testid="stAlert"] p { color: #e2e8f0 !important; }

.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Stroke Risk Predictor</h1>
    <p>Fill in the patient profile below to get an AI-powered stroke risk estimate</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  FORM
# ─────────────────────────────────────────────────────────
with st.form("stroke_form"):

    # ── Personal Details ──────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">👤 Personal Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gender  = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age     = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
    with c2:
        married = st.radio("Ever Married?", ["Yes", "No"], horizontal=True)
        residence = st.radio("Residence Type", ["Urban", "Rural"], horizontal=True)
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Medical History ───────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🩺 Medical History</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        hypertension  = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)
    with c4:
        glucose = st.number_input("Avg Glucose Level (mg/dL)", min_value=1.0, max_value=400.0, value=100.0, step=0.1)
        bmi     = st.number_input("BMI (kg/m²)", min_value=1.0, max_value=100.0, value=24.0, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Lifestyle ─────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">🚬 Lifestyle</div>', unsafe_allow_html=True)
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍  Predict Stroke Risk")

# ─────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────
if submitted:
    # Build dataframe with EXACT column names the transformer expects
    input_df = pd.DataFrame([{
        "gender":            gender,
        "age":               float(age),
        "hypertension":      1 if hypertension == "Yes" else 0,
        "heart disease":     1 if heart_disease == "Yes" else 0,   # space — not underscore
        "Married":           married,
        "Work Type":         work_type,                             # space — not underscore
        "Residence_type":    residence,
        "Avg_Glucose_Level": float(glucose),
        "bmi":               float(bmi),
        "Smoking Status":    smoking,                               # space — not underscore
    }])

    try:
        features     = col_trans.transform(input_df)
        raw          = model.predict(features, verbose=0)
        stroke_prob  = float(raw[0][0])
        is_stroke    = stroke_prob >= 0.5
        label        = "Stroke Risk Detected" if is_stroke else "Low Stroke Risk"
        color        = "#f87171" if is_stroke else "#34d399"
        emoji        = "⚠️" if is_stroke else "✅"
        card_class   = "result-high" if is_stroke else "result-low"

        # ── Result card ───────────────────────────────────
        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size:2.2rem">{emoji}</div>
            <div class="result-label" style="color:{color}">{label}</div>
            <div class="result-prob">Estimated probability: <b>{stroke_prob*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # ── Gauge chart ───────────────────────────────────
        st.markdown('<div class="meter-label">Risk Meter</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(stroke_prob * 100, 1),
            number={"suffix": "%", "font": {"size": 34, "color": "#e2e8f0"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#64748b",
                    "tickfont": {"color": "#94a3b8", "size": 12},
                    "tickwidth": 1,
                },
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "rgba(16,185,129,0.25)"},
                    {"range": [30, 60], "color": "rgba(251,191,36,0.25)"},
                    {"range": [60, 100],"color": "rgba(239,68,68,0.25)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.85,
                    "value": stroke_prob * 100,
                },
            },
        ))
        fig.update_layout(
            height=240,
            margin=dict(t=10, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "color": "#e2e8f0"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Risk breakdown ────────────────────────────────
        risk_factors = []
        if age > 60:
            risk_factors.append("🔴 Age above 60")
        if hypertension == "Yes":
            risk_factors.append("🔴 Hypertension present")
        if heart_disease == "Yes":
            risk_factors.append("🔴 Heart disease present")
        if glucose > 140:
            risk_factors.append("🟠 High glucose level")
        if bmi > 30:
            risk_factors.append("🟠 BMI in obese range")
        if smoking == "smokes":
            risk_factors.append("🟠 Active smoker")

        if risk_factors:
            with st.expander("📋 Key Risk Factors Detected", expanded=True):
                for r in risk_factors:
                    st.markdown(f"- {r}")

        # ── Advice ───────────────────────────────────────
        if is_stroke:
            st.warning(
                "⚠️ This result suggests an **elevated stroke risk** based on the provided profile. "
                "Please consult a qualified healthcare professional for a proper medical assessment."
            )
        else:
            st.success(
                "✅ This result suggests a **lower estimated stroke risk**. "
                "This is not a medical diagnosis — consult a doctor for any health concerns."
            )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ─────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · Powered by a TensorFlow ANN model<br>
    ⚠️ For educational/demo purposes only — not a medical device.
</div>
""", unsafe_allow_html=True)