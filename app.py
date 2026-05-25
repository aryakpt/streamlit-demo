"""
app.py — Streamlit Churn Prediction App
Pastikan sudah jalankan train_model.py terlebih dahulu.
"""

import os
import pickle
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go

# ── Konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: #f1f5f9; font-size: 1.9rem; font-weight: 800; margin: 0; }
    .hero p  { color: #94a3b8; margin: 0.4rem 0 0; font-size: 0.95rem; }

    .result-churn {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #16a34a;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-label { font-size: 1rem; color: #cbd5e1; margin-bottom: 0.3rem; }
    .result-value { font-size: 2.5rem; font-weight: 800; }
    .churn-text   { color: #fca5a5; }
    .safe-text    { color: #86efac; }

    .info-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .info-box h4 { color: #e2e8f0; font-size: 0.9rem; margin: 0 0 0.5rem; }
    .info-box p  { color: #94a3b8; font-size: 0.85rem; margin: 0; }

    .section-title {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 700;
        border-left: 3px solid #6366f1;
        padding-left: 0.7rem;
        margin: 1.2rem 0 0.8rem;
    }

    div[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.45);
    }
</style>
""", unsafe_allow_html=True)


# ── Definisi Model (harus sama persis dengan train_model.py) ─────────────────
class ChurnClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Load model & preprocessor ────────────────────────────────────────────────
@st.cache_resource
def load_all():
    model_path   = "model/churn_model.pt"
    scaler_path  = "model/scaler.pkl"
    encoder_path = "model/label_encoder.pkl"

    missing = [p for p in [model_path, scaler_path, encoder_path] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing

    checkpoint = torch.load(model_path, map_location="cpu")
    model = ChurnClassifier(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)

    return model, scaler, encoders, []


model, scaler, encoders, missing_files = load_all()


# ── Cek file model ada atau belum ────────────────────────────────────────────
if missing_files:
    st.markdown("""
    <div class="hero">
        <h1>🔮 Churn Predictor</h1>
        <p>Prediksi kemungkinan pelanggan berhenti berlangganan</p>
    </div>
    """, unsafe_allow_html=True)

    st.error("⚠️ File model belum ditemukan. Jalankan perintah berikut terlebih dahulu:")
    st.code("python train_model.py", language="bash")
    st.info(f"File yang belum ada: `{'`, `'.join(missing_files)}`")
    st.stop()


# ── Sidebar — Input Pelanggan ─────────────────────────────────────────────────
st.sidebar.markdown("## 🔮 Input Data Pelanggan")
st.sidebar.markdown("---")

st.sidebar.markdown("**📋 Info Langganan**")
tenure          = st.sidebar.slider("Lama Berlangganan (bulan)", 1, 72, 12)
num_products    = st.sidebar.slider("Jumlah Produk", 1, 5, 2)
contract        = st.sidebar.selectbox("Tipe Kontrak", ["Month-to-Month", "One Year", "Two Year"])

st.sidebar.markdown("**💳 Info Pembayaran**")
monthly_charges = st.sidebar.slider("Tagihan Bulanan (Rp)", 20, 120, 65)
total_charges   = st.sidebar.slider("Total Tagihan (Rp)", 100, 8000, 1500)
payment_method  = st.sidebar.selectbox("Metode Pembayaran", ["Credit Card", "Bank Transfer", "E-Wallet"])

st.sidebar.markdown("**📞 Riwayat Support**")
support_calls   = st.sidebar.slider("Jumlah Panggilan Support", 0, 10, 2)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Prediksi Sekarang")


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔮 Customer Churn Predictor - Arya Krisna Putra</h1>
    <p>Model PyTorch (.pt) — Neural Network klasifikasi biner · Isi form di sidebar lalu klik Prediksi</p>
</div>
""", unsafe_allow_html=True)

col_main, col_info = st.columns([3, 2], gap="large")

with col_main:
    st.markdown('<p class="section-title">📊 Ringkasan Input</p>', unsafe_allow_html=True)

    # Tampilkan input sebagai tabel ringkas
    input_data = {
        "Parameter": [
            "Lama Berlangganan", "Tagihan Bulanan", "Total Tagihan",
            "Jumlah Produk", "Panggilan Support", "Tipe Kontrak", "Metode Bayar"
        ],
        "Nilai": [
            f"{tenure} bulan", f"Rp {monthly_charges}", f"Rp {total_charges:,}",
            num_products, support_calls, contract, payment_method
        ]
    }
    import pandas as pd
    st.dataframe(
        pd.DataFrame(input_data),
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    # ── Prediksi ──────────────────────────────────────────────────────────────
    if predict_btn:
        # Encode input
        contract_enc = encoders["contract"].transform([contract])[0]
        payment_enc  = encoders["payment"].transform([payment_method])[0]

        raw = np.array([[
            tenure, monthly_charges, total_charges,
            num_products, support_calls,
            contract_enc, payment_enc
        ]], dtype=np.float32)

        scaled = scaler.transform(raw)
        tensor = torch.FloatTensor(scaled)

        with torch.no_grad():
            prob_churn = model(tensor).item()

        prob_safe = 1 - prob_churn
        is_churn  = prob_churn >= 0.5

        st.markdown('<p class="section-title">🎯 Hasil Prediksi</p>', unsafe_allow_html=True)

        if is_churn:
            st.markdown(f"""
            <div class="result-churn">
                <p class="result-label">Status Pelanggan</p>
                <p class="result-value churn-text">⚠️ BERPOTENSI CHURN</p>
                <p style="color:#fca5a5; font-size:1.1rem; margin-top:0.5rem;">
                    Probabilitas Churn: <strong>{prob_churn:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <p class="result-label">Status Pelanggan</p>
                <p class="result-value safe-text">✅ PELANGGAN AMAN</p>
                <p style="color:#86efac; font-size:1.1rem; margin-top:0.5rem;">
                    Probabilitas Aman: <strong>{prob_safe:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Gauge chart
        st.markdown('<p class="section-title">📈 Gauge Probabilitas</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_churn * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#e2e8f0"}},
            delta={"reference": 50, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar":  {"color": "#dc2626" if is_churn else "#16a34a"},
                "bgcolor": "#1e293b",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0,  40], "color": "#052e16"},
                    {"range": [40, 60], "color": "#422006"},
                    {"range": [60, 100],"color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": "#facc15", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": "Risiko Churn", "font": {"color": "#94a3b8"}},
        ))
        fig.update_layout(
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(l=20, r=20, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Isi parameter di sidebar, lalu klik **Prediksi Sekarang**")


with col_info:
    st.markdown('<p class="section-title">🧠 Info Model</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h4>Arsitektur Neural Network</h4>
        <p>
        Input (7) → Linear(64) → ReLU → Dropout(0.3)<br>
        → Linear(32) → ReLU → Dropout(0.2)<br>
        → Linear(16) → ReLU → Linear(1) → Sigmoid
        </p>
    </div>
    <div class="info-box">
        <h4>📁 File yang Digunakan</h4>
        <p>
        • <code>model/churn_model.pt</code> — bobot model<br>
        • <code>model/scaler.pkl</code> — normalisasi input<br>
        • <code>model/label_encoder.pkl</code> — encode kategori
        </p>
    </div>
    <div class="info-box">
        <h4>⚡ Cara Kerja</h4>
        <p>
        1. Input user dinormalisasi dengan <strong>StandardScaler</strong><br>
        2. Kolom kategorik di-encode dengan <strong>LabelEncoder</strong><br>
        3. Tensor dimasukkan ke model PyTorch<br>
        4. Output Sigmoid → probabilitas churn (0–1)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">📌 Faktor Risiko Tinggi</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p>
        🔴 Berlangganan &lt; 12 bulan<br>
        🔴 Tagihan bulanan &gt; Rp 80<br>
        🔴 Panggilan support &gt; 5x<br>
        🔴 Kontrak Month-to-Month<br>
        🟢 Kontrak 2 tahun = risiko rendah
        </p>
    </div>
    """, unsafe_allow_html=True)
