# 💊 PharmaGuide

An intelligent symptom-based disease prediction app built with **Streamlit** and a **Random Forest ML model**, trained on real disease-symptom data.

---

## 📁 Project Structure

```
pharmaguide/
│
├── app.py                                      # Main Streamlit application
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── cleaned_final_dataset.csv                   # Disease-symptom-precaution dataset
├── final_cleaned_combined_dataset__3_.csv      # Drug-condition-safety dataset
│
└── .streamlit/
    └── config.toml                             # Streamlit theme & server config
```

---

## 🚀 Quick Start

### 1. Clone / Download the project

Place all files in a single folder as shown above.

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 🧠 How It Works

```
User Input (Symptoms)
        ↓
  ML Model (Random Forest)
        ↓
  Predict Disease
        ↓
  Lookup Table (Medicine + Info)
        ↓
  Show Results
```

- **131 unique symptoms** available for selection
- **41 diseases** the model can predict
- **Random Forest** classifier trained on 4,920 samples
- **Drug lookup table** maps each disease to a recommended medicine
- Results include: Disease, Medicine, Risk Level, Description, Precautions

---

## 📊 Dataset Info

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `cleaned_final_dataset.csv` | 4,920 | 10 | Disease, symptoms, description, precautions, risk level |
| `final_cleaned_combined_dataset__3_.csv` | 206,383 | 5 | Drug name, condition, safety rating |

---

## ⚙️ Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.8 |
| streamlit | ≥ 1.28.0 |
| scikit-learn | ≥ 1.3.0 |
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push your project to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — your app goes live instantly!

> Make sure both CSV files are included in your GitHub repo.

---

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare provider before making any medical decisions.

---

## 🛠️ Troubleshooting

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**Module not found:**
```bash
pip install -r requirements.txt --upgrade
```

**CSV file not found error:**
Make sure `cleaned_final_dataset.csv` and `final_cleaned_combined_dataset__3_.csv` are in the **same folder** as `app.py`.
