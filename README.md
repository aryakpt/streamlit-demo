# 🔮 Churn Predictor — Streamlit + PyTorch

Aplikasi prediksi churn pelanggan menggunakan model Neural Network PyTorch (.pt).

## 📁 Struktur Project

```
streamlit-churn/
├── train_model.py          # Script training → menghasilkan file model
├── app.py                  # Streamlit app (load & inference model)
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── model/                  # Dibuat otomatis saat train_model.py dijalankan
    ├── churn_model.pt      # Bobot neural network PyTorch
    ├── scaler.pkl          # StandardScaler (normalisasi input)
    └── label_encoder.pkl  # LabelEncoder (encode kolom kategorik)
```

## 🚀 Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Training model (WAJIB dijalankan sekali)
python train_model.py

# 3. Jalankan Streamlit app
streamlit run app.py
```

## 🔄 Alur Kerja

```
train_model.py
    │
    ├── Generate data simulasi churn
    ├── Preprocessing (StandardScaler + LabelEncoder)
    ├── Training PyTorch Neural Network (60 epoch)
    └── Simpan → model/churn_model.pt
                  model/scaler.pkl
                  model/label_encoder.pkl

app.py
    │
    ├── Load model .pt dengan @st.cache_resource
    ├── User input via sidebar
    ├── Preprocess input (scaler + encoder)
    ├── Inference → probabilitas churn
    └── Tampilkan hasil + gauge chart
```

## 📌 Catatan Penting

- **Definisi class `ChurnClassifier` harus identik** di `train_model.py` dan `app.py`
- Selalu simpan `scaler` dan `encoder` bersamaan dengan model `.pt`
- Gunakan `@st.cache_resource` agar model tidak reload setiap interaksi
- `model.eval()` wajib dipanggil sebelum inference
