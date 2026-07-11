import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Churn Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header HTML ────────────────────────────────────────────────────────────────
html_temp = """
    <div style="background:linear-gradient(135deg,#1db954 0%,#158a3e 60%,#0d1117 100%);
                padding:2rem 2.5rem; border-radius:14px; margin-bottom:1rem;">
        <h1 style="color:#ffffff; text-align:center; margin:0; font-size:2.2rem;
                   font-family:'Segoe UI',sans-serif; letter-spacing:-0.5px;">
            🎵 Spotify Churn Predictor
        </h1>
        <h3 style="color:rgba(255,255,255,0.75); text-align:center; margin:0.4rem 0 0 0;
                   font-weight:400; font-size:1rem; font-family:'Segoe UI',sans-serif;">
            Final Project · Data Science Bootcamp Batch 60 · Kelompok 1
        </h3>
    </div>
"""

desc_temp = """
### 🎵 Spotify Churn Predictor

Aplikasi ini digunakan untuk memprediksi apakah seorang pelanggan Spotify
akan **berhenti berlangganan (churn)** atau tetap **aktif**, berdasarkan
data perilaku dan profil pengguna.

---

#### 📂 Sumber Dataset
- [Spotify Dataset for Churn Analysis — Kaggle](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis)

#### 📌 Fitur Dataset
| Fitur | Keterangan |
|---|---|
| `age` | Usia pengguna (tahun) |
| `gender` | Jenis kelamin (Male / Female / Other) |
| `country` | Negara asal pengguna |
| `subscription_type` | Tipe langganan (Free / Premium / Student / Family) |
| `listening_time` | Total waktu mendengarkan (menit) |
| `songs_played_per_day` | Rata-rata lagu diputar per hari |
| `skip_rate` | Proporsi lagu yang diskip (0.0–1.0) |
| `ads_listened_per_week` | Jumlah iklan didengar per minggu |
| `offline_listening` | Penggunaan mode offline (Ya / Tidak) |
| `device_type` | Perangkat utama (Mobile / Desktop / Web) |

#### 🤖 Model yang Digunakan
Model terbaik dipilih dari perbandingan 8 algoritma klasifikasi (Logistic Regression,
Decision Tree, Random Forest, KNN, Naive Bayes, SVM, XGBoost, LightGBM)
menggunakan metrik ROC AUC, Precision, Recall, dan F1 Score,
dengan teknik SMOTE untuk menangani ketidakseimbangan kelas.

---

#### 🗂️ Isi Aplikasi
- 🏠 **Home** — Deskripsi aplikasi dan dataset
- 🤖 **Prediction** — Prediksi churn untuk satu pengguna
"""

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    stc.html(html_temp, height=130)

    menu = ["🏠 Home", "🤖 Prediction"]
    choice = st.sidebar.selectbox("📂 Menu", menu)

    if choice == "🏠 Home":
        st.markdown(desc_temp)

    elif choice == "🤖 Prediction":
        run_ml_app()


if __name__ == "__main__":
    main()
