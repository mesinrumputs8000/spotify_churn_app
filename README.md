# 🎵 Spotify Churn Predictor

**Final Project · Data Science Bootcamp Batch 60 · Kelompok 1**

---

## 📁 Struktur Folder

```
spotify_churn_app/
│
├── app.py                        ← Entry point utama Streamlit
├── ml_app.py                     ← Logika prediksi & UI Machine Learning
├── pyproject.toml                ← Daftar dependensi
├── README.md                     ← Dokumentasi ini
├── model_spotify_churn.pkl       ← Model hasil training (dari notebook)
└── scaler_spotify_churn.pkl      ← StandardScaler hasil training
```

> ⚠️ File `.pkl` dihasilkan di **Section 8 (Model Deployment)** notebook Google Colab.
> Download kedua file tersebut lalu letakkan satu folder dengan `app.py`.

---

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install streamlit scikit-learn xgboost lightgbm joblib numpy pandas imbalanced-learn
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

### 3. Buka di browser
```
http://localhost:8501
```

---

## 🖥️ Fitur Aplikasi

| Halaman | Konten |
|---|---|
| 🏠 Home | Deskripsi aplikasi, sumber dataset, penjelasan fitur, dan model yang digunakan |
| 🤖 Prediction | Form input 10 fitur → prediksi churn → probabilitas → rekomendasi tindakan |

### Input yang tersedia:
- **Profil**: Usia, Jenis Kelamin, Negara, Tipe Langganan, Perangkat
- **Perilaku**: Waktu Mendengarkan, Lagu per Hari, Skip Rate, Iklan per Minggu, Mode Offline

### Output prediksi:
- Status: **Churn** atau **Aktif**
- Probabilitas Churn & Aktif (%)
- Tingkat Risiko: 🟢 Rendah / 🟡 Sedang / 🔴 Tinggi
- Progress bar probabilitas
- Rekomendasi tindakan retensi

---
