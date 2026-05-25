"""
train_model.py
--------------
Jalankan script ini SEKALI untuk membuat file:
  - model/churn_model.pt     (bobot neural network)
  - model/scaler.pkl         (StandardScaler untuk normalisasi input)
  - model/label_encoder.pkl  (LabelEncoder untuk kolom kategorik)

Cara jalankan:
    python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── Buat folder output ───────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

# ── 1. Generate data simulasi churn ─────────────────────────────────────────
np.random.seed(42)
N = 2000

data = pd.DataFrame({
    "tenure":           np.random.randint(1, 72, N),          # bulan berlangganan
    "monthly_charges":  np.random.uniform(20, 120, N),        # tagihan bulanan
    "total_charges":    np.random.uniform(100, 8000, N),      # total tagihan
    "num_products":     np.random.randint(1, 6, N),           # jumlah produk
    "support_calls":    np.random.randint(0, 10, N),          # panggilan support
    "contract":         np.random.choice(["Month-to-Month",
                                          "One Year",
                                          "Two Year"], N),    # tipe kontrak
    "payment_method":   np.random.choice(["Credit Card",
                                          "Bank Transfer",
                                          "E-Wallet"], N),   # metode bayar
})

# Label churn — pelanggan dengan tenure pendek & tagihan tinggi lebih rentan
churn_prob = (
    0.4 * (data["tenure"] < 12).astype(float)
    + 0.3 * (data["monthly_charges"] > 80).astype(float)
    + 0.2 * (data["support_calls"] > 5).astype(float)
    + 0.1 * (data["contract"] == "Month-to-Month").astype(float)
)
churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min())
data["churn"] = (np.random.rand(N) < churn_prob).astype(int)

print(f"Dataset: {N} baris | Churn rate: {data['churn'].mean():.1%}")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
# Encode kolom kategorik
le_contract = LabelEncoder()
le_payment  = LabelEncoder()

data["contract_enc"]       = le_contract.fit_transform(data["contract"])
data["payment_method_enc"] = le_payment.fit_transform(data["payment_method"])

# Simpan encoder
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump({"contract": le_contract, "payment": le_payment}, f)

# Pilih fitur numerik
feature_cols = [
    "tenure", "monthly_charges", "total_charges",
    "num_products", "support_calls",
    "contract_enc", "payment_method_enc",
]
X = data[feature_cols].values.astype(np.float32)
y = data["churn"].values.astype(np.float32)

# Normalisasi
scaler = StandardScaler()
X = scaler.fit_transform(X)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tensor
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=64, shuffle=True
)

# ── 3. Definisi Model ────────────────────────────────────────────────────────
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


model = ChurnClassifier(input_dim=len(feature_cols))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# ── 4. Training ──────────────────────────────────────────────────────────────
EPOCHS = 60
print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = (model(X_test_t) >= 0.5).float()
            val_acc  = accuracy_score(y_test_t.numpy(), val_pred.numpy())
        print(f"  Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

# ── 5. Evaluasi Final ────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred = (model(X_test_t) >= 0.5).float().numpy().flatten()

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Tidak Churn", "Churn"]))

# ── 6. Simpan Model ──────────────────────────────────────────────────────────
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim":        len(feature_cols),
    "feature_cols":     feature_cols,
}, "model/churn_model.pt")

print("✅ Model tersimpan di: model/churn_model.pt")
print("✅ Scaler tersimpan di: model/scaler.pkl")
print("✅ Encoder tersimpan di: model/label_encoder.pkl")
print("\nSekarang jalankan: streamlit run app.py")
