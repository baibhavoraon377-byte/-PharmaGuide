import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="PharmaGuide",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════
# FULL MEDICAL UI/UX CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700;800&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
  --bg-deep:       #080c1a;
  --bg-mid:        #0d1530;
  --bg-card:       rgba(13,21,56,0.85);
  --bg-card2:      rgba(10,16,42,0.9);
  --cyan:          #00d4ff;
  --cyan-dim:      rgba(0,212,255,0.12);
  --cyan-glow:     rgba(0,212,255,0.35);
  --violet:        #7b5ea7;
  --violet-bright: #a78bfa;
  --green:         #00e5a0;
  --amber:         #ffb347;
  --red:           #ff5f7e;
  --text-primary:  #e8f4fd;
  --text-secondary:#8baac8;
  --text-muted:    #4a6080;
  --border:        rgba(0,212,255,0.18);
  --border-soft:   rgba(255,255,255,0.07);
}

html, body, [class*="css"], .stApp {
  font-family: 'Exo 2', sans-serif !important;
  background: var(--bg-deep) !important;
  color: var(--text-primary) !important;
}

.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 50% -5%, rgba(0,100,200,0.2) 0%, transparent 70%),
    radial-gradient(ellipse 55% 40% at 90% 80%, rgba(0,212,255,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 50% 50% at 10% 90%, rgba(123,94,167,0.1) 0%, transparent 60%),
    linear-gradient(180deg, #080c1a 0%, #0a1020 50%, #080c1a 100%) !important;
  min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 2rem 3rem !important;
  max-width: 1300px !important;
}

/* ══ NAVBAR ══ */
.pg-navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.85rem 2rem;
  background: rgba(8,12,26,0.95);
  border-bottom: 1px solid var(--border);
  margin: 0 -2rem 2.5rem -2rem;
  backdrop-filter: blur(20px);
}
.pg-logo { display:flex; align-items:center; gap:0.75rem; }
.pg-logo-icon {
  width:38px; height:38px;
  background: linear-gradient(135deg, #00d4ff, #7b5ea7);
  border-radius:10px;
  display:flex; align-items:center; justify-content:center;
  font-size:1.15rem;
  box-shadow: 0 0 18px var(--cyan-glow);
}
.pg-logo-text {
  font-family:'Rajdhani',sans-serif;
  font-size:1.55rem; font-weight:700;
  background:linear-gradient(90deg,#00d4ff,#a78bfa);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  letter-spacing:0.06em;
}
.pg-logo-tag { font-size:0.62rem; color:var(--text-muted); letter-spacing:0.12em; text-transform:uppercase; margin-top:-4px; }
.pg-nav-pills { display:flex; gap:0.5rem; }
.pg-pill {
  padding:0.35rem 1.1rem; border-radius:20px;
  font-size:0.78rem; font-weight:500;
  border:1px solid var(--border-soft);
  color:var(--text-muted); background:rgba(255,255,255,0.02);
  letter-spacing:0.04em;
}
.pg-pill.active { background:var(--cyan-dim); border-color:var(--cyan); color:var(--cyan); }
.pg-status { display:flex; align-items:center; gap:0.5rem; font-size:0.78rem; color:var(--green); }
.pg-status-dot {
  width:8px; height:8px; border-radius:50%;
  background:var(--green); box-shadow:0 0 8px var(--green);
  animation:pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.7)} }

/* ══ HERO ══ */
.pg-hero { text-align:center; padding:0.5rem 1rem 2rem; }
.pg-hero-badge {
  display:inline-flex; align-items:center; gap:0.5rem;
  background:var(--cyan-dim); border:1px solid var(--cyan-glow);
  border-radius:20px; padding:0.3rem 1.1rem;
  font-size:0.72rem; color:var(--cyan);
  letter-spacing:0.1em; text-transform:uppercase; font-weight:600;
  margin-bottom:1rem;
}
.pg-hero-title {
  font-family:'Rajdhani',sans-serif;
  font-size:3.2rem; font-weight:700; line-height:1.1;
  letter-spacing:0.02em; color:var(--text-primary); margin-bottom:0.6rem;
}
.pg-hero-title span {
  background:linear-gradient(90deg,#00d4ff 0%,#7b5ea7 50%,#00e5a0 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.pg-hero-sub {
  color:var(--text-secondary); font-size:1rem; font-weight:300;
  max-width:560px; margin:0 auto 1.8rem; line-height:1.7;
}
.pg-pipeline {
  display:flex; align-items:center; justify-content:center;
  gap:0; flex-wrap:wrap; margin:0 auto; max-width:900px;
}
.pg-pipe-step {
  display:flex; align-items:center; gap:0.5rem;
  padding:0.55rem 1.1rem;
  background:rgba(0,212,255,0.04);
  border:1px solid var(--border);
  font-size:0.78rem; color:var(--text-secondary); font-weight:500;
}
.pg-pipe-step:first-child{border-radius:25px 0 0 25px;}
.pg-pipe-step:last-child{border-radius:0 25px 25px 0;}
.pg-pipe-step.active{background:rgba(0,212,255,0.12);border-color:var(--cyan);color:var(--cyan);}
.pg-pipe-step .sn {
  width:20px;height:20px;border-radius:50%;
  background:var(--cyan-dim);border:1px solid var(--cyan);
  color:var(--cyan);font-size:0.62rem;font-weight:700;
  display:flex;align-items:center;justify-content:center;
}
.pg-pipe-arrow{color:rgba(0,212,255,0.2);font-size:1rem;padding:0 0.1rem;}

/* ══ STATS ══ */
.pg-stats {
  display:grid; grid-template-columns:repeat(4,1fr);
  gap:1rem; margin-bottom:2rem;
}
.pg-stat {
  background:var(--bg-card2);
  border:1px solid var(--border-soft);
  border-radius:14px; padding:1.1rem 1.2rem;
  text-align:center; position:relative; overflow:hidden;
}
.pg-stat::before {
  content:''; position:absolute;
  top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  opacity:0.5;
}
.pg-stat-val { font-family:'Rajdhani',sans-serif; font-size:2rem; font-weight:700; color:var(--cyan); line-height:1; }
.pg-stat-lbl { font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin-top:0.3rem; }

/* ══ CARDS ══ */
.pg-card {
  background:var(--bg-card);
  border:1px solid var(--border-soft);
  border-radius:20px; padding:1.8rem;
  backdrop-filter:blur(20px);
  margin-bottom:1.2rem;
  position:relative; overflow:hidden;
}
.pg-card::before {
  content:''; position:absolute;
  top:0;left:20px;right:20px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,212,255,0.3),transparent);
}
.pg-card-cyan { border-color:rgba(0,212,255,0.2); box-shadow:0 0 30px rgba(0,212,255,0.04); }
.pg-card-violet { border-color:rgba(167,139,250,0.2); box-shadow:0 0 30px rgba(167,139,250,0.04); }

.pg-card-hdr { display:flex; align-items:center; gap:0.8rem; margin-bottom:1.4rem; }
.pg-card-ico {
  width:38px;height:38px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  font-size:1.1rem;flex-shrink:0;
}
.pg-card-ico.c{background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);}
.pg-card-ico.v{background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3);}
.pg-card-ico.g{background:rgba(0,229,160,0.1);border:1px solid rgba(0,229,160,0.3);}
.pg-card-ttl { font-family:'Rajdhani',sans-serif; font-size:1.15rem; font-weight:700; color:var(--text-primary); letter-spacing:0.03em; }
.pg-card-stl { font-size:0.73rem; color:var(--text-muted); margin-top:1px; }

/* ══ CHIPS ══ */
.pg-chip-wrap {
  display:flex; flex-wrap:wrap; gap:0.5rem;
  padding:0.65rem; min-height:2.8rem;
  background:rgba(0,0,0,0.2);
  border-radius:12px; border:1px solid var(--border-soft);
  margin-top:0.8rem;
}
.pg-chip {
  display:flex; align-items:center; gap:0.35rem;
  padding:0.28rem 0.85rem;
  background:rgba(0,212,255,0.09); border:1px solid rgba(0,212,255,0.3);
  border-radius:20px; font-size:0.78rem; color:var(--cyan); font-weight:500;
}
.pg-chip-empty { color:var(--text-muted); font-size:0.8rem; padding:0.3rem 0.5rem; font-style:italic; }

/* ══ BODY SVG ══ */
.pg-body-wrap { display:flex; flex-direction:column; align-items:center; padding:0.5rem; position:relative; }
.pg-body-glow { width:120px;height:245px;position:relative;margin:0 auto 1rem; }
.pg-body-svg { width:100%;height:100%;filter:drop-shadow(0 0 12px rgba(0,212,255,0.5)); }
.pg-scan { position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);animation:scan 3s ease-in-out infinite;opacity:0.6; }
@keyframes scan{0%{top:0%}100%{top:100%}}
.pg-body-lbl { font-size:0.72rem;color:var(--text-muted);text-align:center;letter-spacing:0.08em;text-transform:uppercase; }

/* ══ HOW-IT-WORKS ══ */
.hw-step { display:flex;align-items:flex-start;gap:0.9rem;padding:0.85rem 0;border-bottom:1px solid var(--border-soft); }
.hw-step:last-child{border-bottom:none;}
.hw-num {
  width:28px;height:28px;border-radius:50%;
  background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(123,94,167,0.15));
  border:1px solid var(--border);color:var(--cyan);
  font-size:0.72rem;font-weight:700;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.hw-ttl{font-size:0.85rem;font-weight:600;color:var(--text-primary);line-height:1.3;}
.hw-dsc{font-size:0.74rem;color:var(--text-muted);margin-top:0.2rem;line-height:1.5;}

/* ══ RESULTS ══ */
.res-hdr {
  display:flex;align-items:center;gap:1.2rem;
  padding:1.4rem 1.8rem;
  background:linear-gradient(135deg,rgba(0,212,255,0.07),rgba(123,94,167,0.07));
  border:1px solid var(--border);border-radius:18px;margin-bottom:1.2rem;
}
.res-disease{font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:var(--text-primary);line-height:1.1;}
.res-conf{font-size:0.8rem;color:var(--cyan);font-weight:600;margin-top:0.25rem;}
.conf-bar-bg{height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;margin-top:0.5rem;}
.conf-bar{height:100%;border-radius:3px;background:linear-gradient(90deg,#00d4ff,#7b5ea7);box-shadow:0 0 8px var(--cyan-glow);}

.res-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.2rem;}
.res-tile {
  background:var(--bg-card2);border-radius:14px;padding:1.1rem 1.3rem;
  border:1px solid var(--border-soft);position:relative;overflow:hidden;
}
.res-tile::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;}
.res-tile.tm::after{background:linear-gradient(90deg,#00d4ff,transparent);}
.res-tile.tr::after{background:linear-gradient(90deg,#ffb347,transparent);}
.res-tile.ts::after{background:linear-gradient(90deg,#a78bfa,transparent);}
.res-tile.ta::after{background:linear-gradient(90deg,#00e5a0,transparent);}
.tile-lbl{font-size:0.67rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:var(--text-muted);margin-bottom:0.5rem;}
.tile-val{font-family:'Rajdhani',sans-serif;font-size:1.25rem;font-weight:700;color:var(--text-primary);line-height:1.2;}
.tile-sub{font-size:0.7rem;color:var(--text-muted);margin-top:0.3rem;}
.risk-low{color:#00e5a0!important;} .risk-medium{color:#ffb347!important;} .risk-high{color:#ff5f7e!important;}

.desc-box {
  background:var(--bg-card2);border:1px solid var(--border-soft);
  border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;
  line-height:1.7;color:var(--text-secondary);font-size:0.87rem;
}
.desc-ttl{font-family:'Rajdhani',sans-serif;font-size:0.88rem;font-weight:700;color:var(--text-primary);margin-bottom:0.6rem;text-transform:uppercase;letter-spacing:0.06em;}

.prec-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.5rem;}
.prec-item{display:flex;align-items:flex-start;gap:0.6rem;padding:0.7rem 0.9rem;background:rgba(0,229,160,0.05);border:1px solid rgba(0,229,160,0.15);border-radius:10px;font-size:0.82rem;color:var(--text-secondary);}
.prec-dot{width:18px;height:18px;border-radius:50%;background:rgba(0,229,160,0.15);border:1px solid rgba(0,229,160,0.4);color:#00e5a0;font-size:0.62rem;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;}

.top3-item{display:flex;align-items:center;gap:1rem;padding:0.85rem 1.1rem;background:rgba(255,255,255,0.02);border:1px solid var(--border-soft);border-radius:12px;margin-bottom:0.7rem;}
.top3-rank{font-family:'Rajdhani',sans-serif;font-size:1.3rem;font-weight:700;width:28px;text-align:center;}
.top3-info{flex:1;}
.top3-name{font-size:0.88rem;font-weight:600;color:var(--text-primary);}
.top3-bar-bg{height:4px;background:rgba(255,255,255,0.05);border-radius:2px;margin-top:0.4rem;overflow:hidden;}
.top3-bar{height:100%;border-radius:2px;}
.top3-pct{font-family:'Rajdhani',sans-serif;font-size:1rem;font-weight:700;}

/* ══ DISCLAIMER ══ */
.pg-disc {
  display:flex;align-items:center;gap:0.8rem;
  padding:0.9rem 1.2rem;
  background:rgba(255,95,126,0.06);border:1px solid rgba(255,95,126,0.2);
  border-radius:12px;margin-top:1.5rem;
  font-size:0.78rem;color:#ff9eb0;line-height:1.5;
}
.pg-disc-icon{font-size:1.2rem;flex-shrink:0;}

/* ══ STREAMLIT OVERRIDES ══ */
div[data-baseweb="select"]>div {
  background:rgba(0,212,255,0.04)!important;
  border:1px solid rgba(0,212,255,0.22)!important;
  border-radius:12px!important;
  color:var(--text-primary)!important;
  font-family:'Exo 2',sans-serif!important;
}
div[data-baseweb="select"]>div:focus-within{border-color:var(--cyan)!important;box-shadow:0 0 0 2px var(--cyan-dim)!important;}
.stMultiSelect [data-baseweb="tag"]{background:rgba(0,212,255,0.1)!important;border:1px solid rgba(0,212,255,0.3)!important;border-radius:8px!important;color:var(--cyan)!important;}
.stMultiSelect [data-baseweb="tag"] span{color:var(--cyan)!important;}
div[data-baseweb="popover"]{background:#0d1530!important;border:1px solid var(--border)!important;border-radius:12px!important;}
li[role="option"]{color:var(--text-secondary)!important;}
li[role="option"]:hover{background:var(--cyan-dim)!important;color:var(--cyan)!important;}
input[type="text"],input[type="search"]{background:transparent!important;color:var(--text-primary)!important;font-family:'Exo 2',sans-serif!important;}
label,.stMultiSelect label{color:var(--text-secondary)!important;font-family:'Exo 2',sans-serif!important;font-size:0.85rem!important;font-weight:500!important;}
.stButton>button {
  background:linear-gradient(135deg,#0099cc,#00d4ff)!important;
  color:#080c1a!important;border:none!important;border-radius:12px!important;
  padding:0.75rem 2rem!important;font-size:0.92rem!important;
  font-weight:700!important;font-family:'Rajdhani',sans-serif!important;
  letter-spacing:0.08em!important;text-transform:uppercase!important;
  width:100%;transition:all 0.2s!important;
  box-shadow:0 4px 20px rgba(0,212,255,0.28)!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(0,212,255,0.42)!important;}
.stSpinner>div{border-top-color:var(--cyan)!important;}
hr{border-color:var(--border-soft)!important;margin:1.5rem 0!important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# DATA & MODEL
# ══════════════════════════════════════════════
DRUG_MAP = {
    "(vertigo) paroymsal  positional vertigo": "Meclizine",
    "alcoholic hepatitis": "Prednisolone",
    "allergy": "Loratadine",
    "bronchial asthma": "Albuterol (Salbutamol)",
    "cervical spondylosis": "Ibuprofen",
    "chicken pox": "Acyclovir",
    "chronic cholestasis": "Ursodiol",
    "common cold": "Pseudoephedrine",
    "dengue": "Acetaminophen",
    "dimorphic hemmorhoids(piles)": "Hydrocortisone Cream",
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
    "paralysis (brain hemorrhage)": "Alteplase (tPA)",
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

ICONS = {
    "fungal infection":"🍄","allergy":"🤧","gerd":"🔥","diabetes":"🩸",
    "hypertension":"💓","migraine":"🧠","pneumonia":"🫁","malaria":"🦟",
    "dengue":"🦟","typhoid":"🤒","tuberculosis":"🫁","aids":"🔴",
    "hepatitis a":"🟡","hepatitis b":"🟡","hepatitis c":"🟡","hepatitis d":"🟡","hepatitis e":"🟡",
    "heart attack":"❤️","chicken pox":"🔴","acne":"🫧","arthritis":"🦴",
    "common cold":"🤧","bronchial asthma":"💨","paralysis (brain hemorrhage)":"🧠",
    "jaundice":"🟡","impetigo":"🩹","psoriasis":"🩹","urinary tract infection":"💧",
    "varicose veins":"🦵","cervical spondylosis":"🦴","osteoarthristis":"🦴",
    "gastroenteritis":"🤢","drug reaction":"💊","default":"🦠"
}

@st.cache_data
def load_and_train():
    df = pd.read_csv("cleaned_final_dataset.csv")
    all_sym = set()
    sym_lists = []
    for s in df["symptoms"]:
        try: lst = [x.strip().replace(" ","_") for x in ast.literal_eval(s)]
        except: lst = [s.strip().replace(" ","_")]
        sym_lists.append(lst); all_sym.update(lst)
    all_sym = sorted(list(all_sym))
    X = np.zeros((len(df), len(all_sym)), dtype=int)
    si = {s:i for i,s in enumerate(all_sym)}
    for ri, sl in enumerate(sym_lists):
        for s in sl:
            if s in si: X[ri, si[s]] = 1
    le = LabelEncoder(); y = le.fit_transform(df["disease"])
    clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    lookup = {}
    for d in df["disease"].unique():
        row = df[df["disease"]==d].iloc[0]
        precs = [str(row.get(c,"")).strip().title() for c in ["precaution_1","precaution_2","precaution_3","precaution_4"]
                 if str(row.get(c,"")).strip().lower() not in ("","nan","none")]
        lookup[d] = {"description":str(row.get("description","")).strip(),
                     "precautions":precs,
                     "risk_level":str(row.get("risk_level","medium")).strip().lower()}
    return clf, le, all_sym, si, lookup

@st.cache_data
def get_symptoms():
    df = pd.read_csv("cleaned_final_dataset.csv")
    s = set()
    for v in df["symptoms"]:
        try: s.update([x.strip() for x in ast.literal_eval(v)])
        except: s.add(v.strip())
    return sorted([x.replace("_"," ").title() for x in s])

import shutil, os
for src, dst in [
    ("/mnt/user-data/uploads/final_cleaned_combined_dataset__3_.csv","final_cleaned_combined_dataset__3_.csv"),
    ("/mnt/user-data/uploads/cleaned_final_dataset.csv","cleaned_final_dataset.csv"),
]:
    if not os.path.exists(dst):
        try: shutil.copy(src,".")
        except: pass


# ══════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════
st.markdown("""
<div class="pg-navbar">
  <div class="pg-logo">
    <div class="pg-logo-icon">💊</div>
    <div>
      <div class="pg-logo-text">PharmaGuide</div>
      <div class="pg-logo-tag">AI Disease Prediction System</div>
    </div>
  </div>
  <div class="pg-nav-pills">
    <div class="pg-pill active">Symptom Checker</div>
    <div class="pg-pill">Disease Library</div>
    <div class="pg-pill">About</div>
  </div>
  <div class="pg-status"><div class="pg-status-dot"></div>AI Model Active</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════
st.markdown("""
<div class="pg-hero">
  <div class="pg-hero-badge">⚕ AI-Powered &nbsp;·&nbsp; 41 Diseases &nbsp;·&nbsp; 131 Symptoms</div>
  <div class="pg-hero-title">Intelligent <span>Symptom Analysis</span></div>
  <div class="pg-hero-sub">Describe your symptoms and our machine learning model identifies the most likely condition, recommends medicines, and provides personalised health guidance.</div>
  <div class="pg-pipeline">
    <div class="pg-pipe-step active"><span class="sn">1</span> Select Symptoms</div>
    <div class="pg-pipe-arrow">›</div>
    <div class="pg-pipe-step"><span class="sn">2</span> ML Analysis</div>
    <div class="pg-pipe-arrow">›</div>
    <div class="pg-pipe-step"><span class="sn">3</span> Predict Disease</div>
    <div class="pg-pipe-arrow">›</div>
    <div class="pg-pipe-step"><span class="sn">4</span> Match Medicine</div>
    <div class="pg-pipe-arrow">›</div>
    <div class="pg-pipe-step"><span class="sn">5</span> View Results</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════
with st.spinner("Initialising AI model…"):
    clf, le, all_sym_raw, sym_idx, lookup = load_and_train()
    sym_list = get_symptoms()

# ══════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════
st.markdown("""
<div class="pg-stats">
  <div class="pg-stat"><div class="pg-stat-val">41</div><div class="pg-stat-lbl">Diseases Covered</div></div>
  <div class="pg-stat"><div class="pg-stat-val">131</div><div class="pg-stat-lbl">Unique Symptoms</div></div>
  <div class="pg-stat"><div class="pg-stat-val">4,920</div><div class="pg-stat-lbl">Training Samples</div></div>
  <div class="pg-stat"><div class="pg-stat-val" style="color:var(--green);font-size:1.4rem;">Random Forest</div><div class="pg-stat-lbl">ML Algorithm</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════
col_main, col_side = st.columns([2, 1], gap="large")

# ── SIDEBAR ──────────────────────────────────
with col_side:
    # Body scanner visual
    st.markdown("""
    <div class="pg-card pg-card-cyan">
      <div class="pg-card-hdr">
        <div class="pg-card-ico c">🫀</div>
        <div><div class="pg-card-ttl">Body Scanner</div><div class="pg-card-stl">Scan in progress…</div></div>
      </div>
      <div class="pg-body-wrap">
        <div class="pg-body-glow">
          <svg class="pg-body-svg" viewBox="0 0 120 255" fill="none" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="60" cy="27" rx="22" ry="24" fill="rgba(0,212,255,0.07)" stroke="#00d4ff" stroke-width="1.2"/>
            <ellipse cx="60" cy="27" rx="10" ry="11" fill="rgba(0,212,255,0.1)"/>
            <rect x="51" y="49" width="18" height="17" rx="4" fill="rgba(0,212,255,0.06)" stroke="#00d4ff" stroke-width="0.9"/>
            <path d="M26 66 Q21 90 23 128 Q24 152 36 162 L84 162 Q96 152 97 128 Q99 90 94 66 Z" fill="rgba(0,212,255,0.06)" stroke="#00d4ff" stroke-width="1.2"/>
            <ellipse cx="60" cy="98" rx="17" ry="20" fill="rgba(0,212,255,0.07)" stroke="rgba(0,212,255,0.25)" stroke-width="0.8" stroke-dasharray="3 2"/>
            <circle cx="51" cy="96" r="3.5" fill="#00d4ff" opacity="0.9">
              <animate attributeName="opacity" values="0.9;0.2;0.9" dur="1.2s" repeatCount="indefinite"/>
              <animate attributeName="r" values="3.5;5;3.5" dur="1.2s" repeatCount="indefinite"/>
            </circle>
            <line x1="60" y1="66" x2="60" y2="162" stroke="rgba(0,212,255,0.18)" stroke-width="0.8" stroke-dasharray="4 3"/>
            <path d="M26 68 Q9 92 11 136 Q12 150 19 153 Q26 156 29 146 Q31 128 31 108 L31 68Z" fill="rgba(0,212,255,0.05)" stroke="#00d4ff" stroke-width="0.9"/>
            <path d="M94 68 Q111 92 109 136 Q108 150 101 153 Q94 156 91 146 Q89 128 89 108 L89 68Z" fill="rgba(0,212,255,0.05)" stroke="#00d4ff" stroke-width="0.9"/>
            <path d="M36 162 Q31 198 29 230 Q29 244 39 246 Q49 248 51 237 Q53 215 53 193 L56 162Z" fill="rgba(0,212,255,0.05)" stroke="#00d4ff" stroke-width="0.9"/>
            <path d="M84 162 Q89 198 91 230 Q91 244 81 246 Q71 248 69 237 Q67 215 67 193 L64 162Z" fill="rgba(0,212,255,0.05)" stroke="#00d4ff" stroke-width="0.9"/>
            <circle cx="60" cy="27" r="5" fill="none" stroke="#00d4ff" stroke-width="1.5">
              <animate attributeName="r" values="5;10;5" dur="2.2s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="1;0;1" dur="2.2s" repeatCount="indefinite"/>
            </circle>
            <circle cx="60" cy="98" r="5" fill="none" stroke="#00e5a0" stroke-width="1.5">
              <animate attributeName="r" values="5;10;5" dur="2.8s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="1;0;1" dur="2.8s" repeatCount="indefinite"/>
            </circle>
            <circle cx="60" cy="138" r="4" fill="none" stroke="#a78bfa" stroke-width="1.5">
              <animate attributeName="r" values="4;8;4" dur="3.2s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="1;0;1" dur="3.2s" repeatCount="indefinite"/>
            </circle>
          </svg>
          <div class="pg-scan"></div>
        </div>
        <div class="pg-body-lbl">Analysing body regions…</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="pg-card pg-card-violet">
      <div class="pg-card-hdr">
        <div class="pg-card-ico v">⚙️</div>
        <div><div class="pg-card-ttl">How It Works</div><div class="pg-card-stl">5-step AI pipeline</div></div>
      </div>
      <div class="hw-step"><div class="hw-num">1</div><div><div class="hw-ttl">Select Symptoms</div><div class="hw-dsc">Pick all symptoms you're experiencing from the searchable list</div></div></div>
      <div class="hw-step"><div class="hw-num">2</div><div><div class="hw-ttl">ML Processing</div><div class="hw-dsc">Random Forest encodes symptoms into a 131-dimensional feature vector</div></div></div>
      <div class="hw-step"><div class="hw-num">3</div><div><div class="hw-ttl">Disease Prediction</div><div class="hw-dsc">Model predicts disease with confidence across 41 possible classes</div></div></div>
      <div class="hw-step"><div class="hw-num">4</div><div><div class="hw-ttl">Medicine Lookup</div><div class="hw-dsc">Curated drug database maps disease to recommended treatment</div></div></div>
      <div class="hw-step"><div class="hw-num">5</div><div><div class="hw-ttl">Full Health Report</div><div class="hw-dsc">Disease info, risk level, description & precautions displayed</div></div></div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN PANEL ────────────────────────────────
with col_main:
    st.markdown("""
    <div class="pg-card pg-card-cyan">
      <div class="pg-card-hdr">
        <div class="pg-card-ico c">🩺</div>
        <div>
          <div class="pg-card-ttl">Choose Your Symptoms</div>
          <div class="pg-card-stl">Select all that apply — more detail improves accuracy</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    selected = st.multiselect(
        "🔍 Search and select symptoms:",
        options=sym_list,
        placeholder="Type to search — e.g. Fever, Headache, Cough...",
    )

    if selected:
        chips = "".join([f'<span class="pg-chip">● {s}</span>' for s in selected])
        st.markdown(f'<div class="pg-chip-wrap">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pg-chip-wrap"><span class="pg-chip-empty">No symptoms selected yet — search above to begin…</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    clicked = st.button("⚡ ANALYSE SYMPTOMS & PREDICT DISEASE", use_container_width=True)

    # ── PREDICTION ──
    if clicked:
        if not selected:
            st.warning("⚠️ Please select at least one symptom before running analysis.")
        else:
            with st.spinner("Running AI analysis…"):
                vec = np.zeros((1, len(all_sym_raw)), dtype=int)
                for s in selected:
                    raw = s.lower().replace(" ","_")
                    if raw in sym_idx: vec[0, sym_idx[raw]] = 1

                enc   = clf.predict(vec)[0]
                proba = clf.predict_proba(vec)[0]
                conf  = round(proba.max()*100, 1)
                dis   = le.inverse_transform([enc])[0]

                info  = lookup.get(dis, {})
                med   = DRUG_MAP.get(dis.lower(), "Consult your physician")
                risk  = info.get("risk_level","medium")
                desc  = info.get("description","No description available.")
                precs = info.get("precautions",[])
                icon  = ICONS.get(dis.lower(), ICONS["default"])

                rcls  = {"low":"risk-low","medium":"risk-medium","high":"risk-high"}.get(risk,"risk-medium")
                rlbl  = {"low":"🟢 LOW","medium":"🟡 MEDIUM","high":"🔴 HIGH"}.get(risk,"🟡 MEDIUM")

                top3_idx = proba.argsort()[::-1][:3]
                top3 = [(le.inverse_transform([i])[0], round(proba[i]*100,1)) for i in top3_idx]
                bcols = ["#00d4ff","#7b5ea7","#00e5a0"]

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Result header
            st.markdown(f"""
            <div class="res-hdr">
              <div style="font-size:2.8rem;line-height:1;">{icon}</div>
              <div style="flex:1;">
                <div style="font-size:0.68rem;color:var(--text-muted);letter-spacing:0.12em;text-transform:uppercase;font-weight:700;">Primary Diagnosis</div>
                <div class="res-disease">{dis.title()}</div>
                <div class="res-conf">AI Confidence: {conf}%</div>
                <div class="conf-bar-bg"><div class="conf-bar" style="width:{conf}%;"></div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 4 tiles
            st.markdown(f"""
            <div class="res-grid">
              <div class="res-tile tm">
                <div class="tile-lbl">💊 Recommended Medicine</div>
                <div class="tile-val">{med}</div>
                <div class="tile-sub">Always consult a physician before use</div>
              </div>
              <div class="res-tile tr">
                <div class="tile-lbl">⚡ Risk Level</div>
                <div class="tile-val {rcls}">{rlbl}</div>
                <div class="tile-sub">Based on disease severity score</div>
              </div>
              <div class="res-tile ts">
                <div class="tile-lbl">🎯 Symptoms Matched</div>
                <div class="tile-val">{len(selected)} / 131</div>
                <div class="tile-sub">Symptoms provided for analysis</div>
              </div>
              <div class="res-tile ta">
                <div class="tile-lbl">🤖 Model Confidence</div>
                <div class="tile-val" style="color:var(--green);">{conf}%</div>
                <div class="tile-sub">Random Forest prediction score</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Description
            st.markdown(f"""
            <div class="desc-box">
              <div class="desc-ttl">📋 About This Condition</div>
              {desc}
            </div>
            """, unsafe_allow_html=True)

            # ── Precautions
            if precs:
                prec_html = "".join([f'<div class="prec-item"><div class="prec-dot">✓</div>{p}</div>' for p in precs])
                st.markdown(f"""
                <div class="desc-box">
                  <div class="desc-ttl">🛡️ Recommended Precautions</div>
                  <div class="prec-grid">{prec_html}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Top 3
            top3_html = ""
            for i, (d, c) in enumerate(top3):
                d_icon = ICONS.get(d.lower(), ICONS["default"])
                top3_html += f"""
                <div class="top3-item">
                  <div class="top3-rank" style="color:{bcols[i]};">#{i+1}</div>
                  <div style="font-size:1.3rem;">{d_icon}</div>
                  <div class="top3-info">
                    <div class="top3-name">{d.title()}</div>
                    <div class="top3-bar-bg"><div class="top3-bar" style="width:{c}%;background:{bcols[i]};box-shadow:0 0 6px {bcols[i]}55;"></div></div>
                  </div>
                  <div class="top3-pct" style="color:{bcols[i]};">{c}%</div>
                </div>"""

            st.markdown(f"""
            <div class="pg-card" style="margin-top:1.2rem;">
              <div class="pg-card-hdr">
                <div class="pg-card-ico c">📊</div>
                <div><div class="pg-card-ttl">Top 3 Possible Conditions</div><div class="pg-card-stl">Ranked by model probability</div></div>
              </div>
              {top3_html}
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# DISCLAIMER
# ══════════════════════════════════════════════
st.markdown("""
<div class="pg-disc">
  <div class="pg-disc-icon">⚕️</div>
  <div><strong>Medical Disclaimer:</strong> PharmaGuide is for <strong>educational and informational purposes only</strong>.
  It is not a substitute for professional medical advice, diagnosis, or treatment.
  Always consult a qualified healthcare provider before making any medical decisions.</div>
</div>
""", unsafe_allow_html=True)
