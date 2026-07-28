import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

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
#  FIXED API SETTING (sidebar removed — edit here if needed)
# ─────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────────────────
#  STYLES — clean, bright, high-contrast theme
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit chrome + sidebar */
    #MainMenu, footer, header {visibility: hidden;}
    section[data-testid="stSidebar"] {display: none !important;}
    div[data-testid="collapsedControl"] {display: none !important;}

    .stApp {
        background: linear-gradient(160deg, #f0f5ff 0%, #eef2fb 40%, #f7f3ff 100%);
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.8rem 1rem 1.4rem 1rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #475569;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 500;
    }

    /* Glass / clean card */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1.8rem 1.8rem 1.2rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.08);
    }

    .section-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 2px solid #eef2ff;
        padding-bottom: 0.6rem;
    }

    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #f8fafc !important;
        border-radius: 10px !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #0f172a !important;
        font-weight: 500 !important;
    }
    label, .stMarkdown p {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* Radio / segmented style tweaks */
    div[role="radiogroup"] label {
        background: #eef2ff;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        border: 1.5px solid #c7d2fe;
        margin-right: 0.5rem;
        color: #1e3a8a !important;
    }
    div[role="radiogroup"] label:hover {
        background: #e0e7ff;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.8rem 0;
        border-radius: 14px;
        border: none;
        box-shadow: 0 8px 22px rgba(124, 58, 237, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(124, 58, 237, 0.45);
        color: white;
    }

    /* Result cards */
    .result-card {
        border-radius: 20px;
        padding: 1.8rem 1.8rem;
        text-align: center;
        margin-top: 1rem;
        border: 2px solid;
        box-shadow: 0 6px 24px rgba(0,0,0,0.06);
    }
    .result-low {
        background: linear-gradient(145deg, #ecfdf5, #ffffff);
        border-color: #10b981;
    }
    .result-high {
        background: linear-gradient(145deg, #fef2f2, #ffffff);
        border-color: #ef4444;
    }
    .result-label {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .result-sub {
        color: #475569;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Stroke Risk Predictor</h1>
    <p>Enter a patient profile below to get an AI-estimated stroke risk score</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  FORM
# ─────────────────────────────────────────────────────────
with st.form("prediction_form"):

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
    with c2:
        married = st.radio("Ever Married?", ["Yes", "No"], horizontal=True)
        residence_type = st.radio("Residence Type", ["Urban", "Rural"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Medical History</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)
    with c4:
        avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=1.0, max_value=400.0, value=100.0, step=0.1)
        bmi = st.number_input("BMI", min_value=1.0, max_value=80.0, value=24.0, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚬 Lifestyle</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
    with c6:
        smoking_status = st.selectbox(
            "Smoking Status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Predict Stroke Risk")

# ─────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────
if submitted:
    payload = {
        "gender": gender,
        "age": int(age),
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "Married": married,
        "Work_Type": work_type,
        "Residence_type": residence_type,
        "Avg_Glucose_Level": float(avg_glucose_level),
        "bmi": float(bmi),
        "Smoking_Status": smoking_status,
    }

    with st.spinner("Contacting model server..."):
        try:
            resp = requests.post(f"{API_URL.rstrip('/')}/predict", json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            label = data.get("label", "Unknown")
            prob = float(data.get("stroke_probability", 0.0))
            ts = data.get("timestamp", datetime.now().isoformat())

            is_high = label.strip().lower() == "stroke"
            card_class = "result-high" if is_high else "result-low"
            emoji = "⚠️" if is_high else "✅"
            color = "#dc2626" if is_high else "#059669"

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div style="font-size:2.4rem;">{emoji}</div>
                <div class="result-label" style="color:{color};">{label}</div>
                <div class="result-sub">Estimated probability: <b>{prob*100:.2f}%</b></div>
                <div class="result-sub">Predicted at {ts}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
                    "bar": {"color": color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(16,185,129,0.25)"},
                        {"range": [50, 100], "color": "rgba(239,68,68,0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f172a", "width": 3},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
            ))
            fig.update_layout(
                height=280,
                margin=dict(t=20, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#0f172a", "family": "Inter"},
            )
            st.plotly_chart(fig, use_container_width=True)

            if is_high:
                st.warning(
                    "This result suggests an elevated stroke risk based on the "
                    "provided profile. Please consult a qualified healthcare "
                    "professional for an accurate medical assessment."
                )
            else:
                st.success(
                    "This result suggests a lower estimated stroke risk. This is "
                    "not a medical diagnosis — consult a doctor for any health concerns."
                )

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Couldn't reach the API at `{API_URL}`. "
                "Make sure your FastAPI server is running and the URL is correct."
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API returned an error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

st.markdown("""
<div class="footer-note">
    Built with Streamlit · Powered by a TensorFlow model served via FastAPI<br>
    ⚠️ For educational/demo purposes only — not a medical device.
</div>
""", unsafe_allow_html=True)