import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaGuide",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* HEADER */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* PIPELINE */
    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin: 1.2rem 0 2rem;
    }
    .pipe-step {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        color: #c4b5fd;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .pipe-arrow {
        color: #60a5fa;
        font-size: 1rem;
    }

    /* CARD */
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.8rem;
    }

    /* RESULT ITEMS */
    .result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    .result-item {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 3px solid;
    }
    .result-item.disease  { border-color: #a78bfa; }
    .result-item.medicine { border-color: #60a5fa; }
    .result-item.risk     { border-color: #f59e0b; }
    .result-item.desc     { border-color: #34d399; grid-column: span 2; }
    .result-item.prec     { border-color: #f472b6; grid-column: span 2; }

    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    .result-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    .result-desc {
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* RISK BADGE */
    .risk-low    { color: #4ade80; }
    .risk-medium { color: #fbbf24; }
    .risk-high   { color: #f87171; }

    /* PRECAUTION LIST */
    .prec-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .prec-tag {
        background: rgba(244,114,182,0.12);
        border: 1px solid rgba(244,114,182,0.3);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.82rem;
        color: #f9a8d4;
    }

    /* DISCLAIMER */
    .disclaimer {
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    /* Streamlit override */
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(167,139,250,0.4) !important;
        border-radius: 10px !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: rgba(167,139,250,0.25) !important;
        border-radius: 6px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.2s !important;
        margin-top: 0.5rem;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }
    label, .stMultiSelect label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING
# ─────────────────────────────────────────────
DRUG_MAP = {
    "(vertigo) paroymsal  positional vertigo": "Meclizine",
    "alcoholic hepatitis": "Prednisolone",
    "allergy": "Loratadine",
    "bronchial asthma": "Albuterol",
    "cervical spondylosis": "Ibuprofen",
    "chicken pox": "Acyclovir",
    "chronic cholestasis": "Ursodiol",
    "common cold": "Pseudoephedrine",
    "dengue": "Acetaminophen",
    "dimorphic hemmorhoids(piles)": "Hydrocortisone",
    "drug reaction": "Diphenhydramine",
    "fungal infection": "Fluconazole",
    "gastroenteritis": "Loperamide",
    "gerd": "Omeprazole",
    "heart attack": "Aspirin",
    "hepatitis a": "Ribavirin",
    "hepatitis b": "Tenofovir",
    "hepatitis c": "Sofosbuvir",
    "hepatitis d": "Peginterferon Alfa",
    "hepatitis e": "Ribavirin",
    "hypertension": "Lisinopril",
    "hyperthyroidism": "Methimazole",
    "hypoglycemia": "Glucagon",
    "hypothyroidism": "Levothyroxine",
    "impetigo": "Mupirocin",
    "jaundice": "Ursodiol",
    "malaria": "Chloroquine",
    "migraine": "Sumatriptan",
    "osteoarthristis": "Celecoxib",
    "paralysis (brain hemorrhage)": "Alteplase",
    "peptic ulcer diseae": "Omeprazole",
    "pneumonia": "Amoxicillin",
    "psoriasis": "Methotrexate",
    "tuberculosis": "Isoniazid",
    "typhoid": "Ciprofloxacin",
    "urinary tract infection": "Trimethoprim",
    "varicose veins": "Compression Stockings",
    "aids": "Tenofovir",
    "acne": "Isotretinoin",
    "arthritis": "Diclofenac",
    "diabetes": "Metformin",
}

@st.cache_data
def load_and_train():
    df = pd.read_csv("cleaned_final_dataset.csv")

    # Parse symptoms
    all_symptoms = set()
    sym_lists = []
    for s in df["symptoms"]:
        try:
            lst = [x.strip().replace(" ", "_") for x in ast.literal_eval(s)]
        except:
            lst = [s.strip().replace(" ", "_")]
        sym_lists.append(lst)
        all_symptoms.update(lst)

    all_symptoms = sorted(list(all_symptoms))

    # Build feature matrix
    X = np.zeros((len(df), len(all_symptoms)), dtype=int)
    sym_idx = {s: i for i, s in enumerate(all_symptoms)}
    for row_i, sym_list in enumerate(sym_lists):
        for sym in sym_list:
            if sym in sym_idx:
                X[row_i, sym_idx[sym]] = 1

    le = LabelEncoder()
    y = le.fit_transform(df["disease"])

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # Build lookup: disease -> best row (most common description/precautions/risk)
    lookup = {}
    for disease in df["disease"].unique():
        subset = df[df["disease"] == disease].iloc[0]
        precs = []
        for col in ["precaution_1", "precaution_2", "precaution_3", "precaution_4"]:
            val = str(subset.get(col, "")).strip()
            if val and val.lower() not in ("nan", "none", ""):
                precs.append(val.title())
        lookup[disease] = {
            "description": str(subset.get("description", "")).strip(),
            "precautions": precs,
            "risk_level": str(subset.get("risk_level", "medium")).strip().lower(),
        }

    return clf, le, all_symptoms, sym_idx, lookup


@st.cache_data
def get_all_symptoms():
    df = pd.read_csv("cleaned_final_dataset.csv")
    all_symptoms = set()
    for s in df["symptoms"]:
        try:
            lst = [x.strip() for x in ast.literal_eval(s)]
        except:
            lst = [s.strip()]
        all_symptoms.update(lst)
    return sorted([s.replace("_", " ").title() for s in all_symptoms])


# ─────────────────────────────────────────────
# COPY DATA FILES
# ─────────────────────────────────────────────
import shutil, os
src1 = "/mnt/user-data/uploads/final_cleaned_combined_dataset__3_.csv"
src2 = "/mnt/user-data/uploads/cleaned_final_dataset.csv"
if not os.path.exists("final_cleaned_combined_dataset__3_.csv"):
    shutil.copy(src1, ".")
if not os.path.exists("cleaned_final_dataset.csv"):
    shutil.copy(src2, ".")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">💊 PharmaGuide</div>
    <div class="hero-subtitle">Select your symptoms and let AI predict the most likely disease with treatment guidance</div>
</div>
<div class="pipeline">
    <span class="pipe-step">🩺 User Symptoms</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">🤖 ML Model</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">🔍 Predict Disease</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">💊 Lookup Medicine</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step">📋 Show Results</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("🔄 Loading AI model…"):
    clf, le, all_symptoms_raw, sym_idx, lookup = load_and_train()
    symptom_display_list = get_all_symptoms()

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🩺 Select Your Symptoms</div>', unsafe_allow_html=True)

    selected_symptoms = st.multiselect(
        "Choose one or more symptoms from the list below:",
        options=symptom_display_list,
        placeholder="Type to search symptoms...",
        label_visibility="visible",
    )

    predict_clicked = st.button("🔍 Predict Disease", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">ℹ️ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#94a3b8; font-size:0.88rem; line-height:1.8;">
    1. <b style="color:#c4b5fd;">Select symptoms</b> from the dropdown<br>
    2. Click <b style="color:#60a5fa;">Predict Disease</b><br>
    3. Our <b style="color:#34d399;">Random Forest ML model</b> analyzes your symptoms<br>
    4. The system matches medicines from the <b style="color:#f9a8d4;">drug database</b><br>
    5. View your <b style="color:#fbbf24;">complete health report</b>
    <br><br>
    <span style="color:#475569; font-size:0.78rem;">⚠️ For informational purposes only. Always consult a licensed physician.</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if predict_clicked:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom before predicting.")
    else:
        # Build input vector
        input_vec = np.zeros((1, len(all_symptoms_raw)), dtype=int)
        for sym_display in selected_symptoms:
            sym_raw = sym_display.lower().replace(" ", "_")
            if sym_raw in sym_idx:
                input_vec[0, sym_idx[sym_raw]] = 1

        # Predict
        pred_encoded = clf.predict(input_vec)[0]
        pred_proba = clf.predict_proba(input_vec)[0]
        confidence = round(pred_proba.max() * 100, 1)
        disease = le.inverse_transform([pred_encoded])[0]

        # Fetch info
        info = lookup.get(disease, {})
        medicine = DRUG_MAP.get(disease.lower(), "Consult your physician")
        risk = info.get("risk_level", "medium")
        description = info.get("description", "No description available.")
        precautions = info.get("precautions", [])

        risk_class = {"low": "risk-low", "medium": "risk-medium", "high": "risk-high"}.get(risk, "risk-medium")
        risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "🟡")

        prec_tags = "".join([f'<span class="prec-tag">✔ {p}</span>' for p in precautions]) if precautions else '<span class="prec-tag">Consult a doctor</span>'

        # Get top 3 predictions with confidence
        top3_idx = pred_proba.argsort()[::-1][:3]
        top3 = [(le.inverse_transform([i])[0], round(pred_proba[i]*100, 1)) for i in top3_idx]

        st.markdown("---")
        st.markdown('<div class="card-title" style="color:#60a5fa; font-size:0.9rem; margin-bottom:0.8rem;">📊 PREDICTION RESULTS</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-grid">
            <div class="result-item disease">
                <div class="result-label">🦠 Disease</div>
                <div class="result-value">{disease.title()}</div>
                <div style="font-size:0.78rem; color:#7c3aed; margin-top:0.3rem;">Confidence: {confidence}%</div>
            </div>
            <div class="result-item medicine">
                <div class="result-label">💊 Recommended Medicine</div>
                <div class="result-value">{medicine}</div>
                <div style="font-size:0.78rem; color:#2563eb; margin-top:0.3rem;">Consult doctor before use</div>
            </div>
            <div class="result-item risk">
                <div class="result-label">⚡ Risk Level</div>
                <div class="result-value {risk_class}">{risk_icon} {risk.upper()}</div>
            </div>
            <div class="result-item" style="border-color:#818cf8;">
                <div class="result-label">🎯 Matched Symptoms</div>
                <div class="result-value" style="font-size:0.9rem;">{len(selected_symptoms)} symptom(s)</div>
            </div>
            <div class="result-item desc">
                <div class="result-label">📝 Description</div>
                <div class="result-desc">{description}</div>
            </div>
            <div class="result-item prec">
                <div class="result-label">🛡️ Precautions</div>
                <div class="prec-list">{prec_tags}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 alternatives
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="color:#94a3b8; font-size:0.78rem;">🔎 TOP 3 POSSIBLE CONDITIONS</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        colors = ["#7c3aed", "#2563eb", "#0891b2"]
        for i, (d, conf) in enumerate(top3):
            with cols[i]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:0.8rem; text-align:center;">
                    <div style="font-size:1.2rem; font-weight:700; color:{colors[i]};">{conf}%</div>
                    <div style="font-size:0.82rem; color:#cbd5e1; margin-top:0.2rem;">{d.title()}</div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ This tool is for <strong>educational and informational purposes only</strong>. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
