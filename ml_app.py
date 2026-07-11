import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Attribute info ─────────────────────────────────────────────────────────────
attribute_info = """
| Fitur | Tipe | Keterangan |
|---|---|---|
| `age` | Numerik | Usia pengguna, antara 13–60 tahun |
| `gender` | Kategorikal | Male / Female / Other |
| `country` | Kategorikal | Negara asal (AU, CA, DE, FR, IN, PK, UK, US) |
| `subscription_type` | Kategorikal | Free / Premium / Student / Family |
| `listening_time` | Numerik | Total menit mendengarkan (0–300 menit) |
| `songs_played_per_day` | Numerik | Rata-rata lagu diputar per hari (0–100) |
| `skip_rate` | Numerik | Proporsi lagu yang diskip, 0.0–0.6 |
| `ads_listened_per_week` | Numerik | Jumlah iklan didengar per minggu (0–50) |
| `offline_listening` | Biner | 1 = Ya, 0 = Tidak |
| `device_type` | Kategorikal | Mobile / Desktop / Web |
"""

# ── Kolom referensi — harus sama persis dengan X_train.columns saat training ──
# Urutan ini mengikuti hasil pd.get_dummies di notebook:
# Numerik dulu, lalu binary encoded, lalu OHE per kolom (sorted alphabetically)
FEATURE_COLUMNS = [
    # Numerik & Biner
    "age",
    "listening_time",
    "songs_played_per_day",
    "skip_rate",
    "ads_listened_per_week",
    "offline_listening",
    # OHE: gender
    "gender_Female",
    "gender_Male",
    "gender_Other",
    # OHE: country
    "country_AU",
    "country_CA",
    "country_DE",
    "country_FR",
    "country_IN",
    "country_PK",
    "country_UK",
    "country_US",
    # OHE: subscription_type
    "subscription_type_Family",
    "subscription_type_Free",
    "subscription_type_Premium",
    "subscription_type_Student",
    # OHE: device_type
    "device_type_Desktop",
    "device_type_Mobile",
    "device_type_Web",
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_model(model_file):
    """
    Load model/scaler dari path relatif terhadap lokasi file ml_app.py ini,
    supaya tetap ditemukan meskipun working directory Streamlit berbeda
    (mis. saat deploy di Streamlit Community Cloud).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_file)
    return joblib.load(model_path)


def encode_inputs(raw: dict) -> pd.DataFrame:
    """
    Encode input user menjadi DataFrame dengan kolom yang sama persis
    seperti X_train saat training (hasil pd.get_dummies).
    """
    # Buat row kosong dengan semua kolom = 0
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Isi fitur numerik & biner
    row["age"]                   = raw["age"]
    row["listening_time"]        = raw["listening_time"]
    row["songs_played_per_day"]  = raw["songs_played_per_day"]
    row["skip_rate"]             = raw["skip_rate"]
    row["ads_listened_per_week"] = raw["ads_listened_per_week"]
    row["offline_listening"]     = raw["offline_listening"]

    # OHE: country
    country_col = f"country_{raw['country']}"
    if country_col in row:
        row[country_col] = 1

    # OHE: device_type
    device_col = f"device_type_{raw['device_type']}"
    if device_col in row:
        row[device_col] = 1

    # OHE: gender
    gender_col = f"gender_{raw['gender']}"
    if gender_col in row:
        row[gender_col] = 1

    # OHE: subscription_type
    sub_col = f"subscription_type_{raw['subscription_type']}"
    if sub_col in row:
        row[sub_col] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ── Main ML app ────────────────────────────────────────────────────────────────
def run_ml_app():
    st.subheader("🤖 Prediksi Churn Pelanggan Spotify")

    with st.expander("ℹ️ Informasi Fitur Dataset"):
        st.markdown(attribute_info)

    st.markdown("---")
    st.subheader("📋 Input Data Pengguna")

    # ── Layout 2 kolom ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Profil Pengguna")
        age = st.number_input("Usia (tahun)", min_value=13, max_value=60, value=25, step=1)
        gender = st.radio("Jenis Kelamin", ["Male", "Female", "Other"], horizontal=True)
        country = st.selectbox("Negara Asal", ["AU", "CA", "DE", "FR", "IN", "PK", "UK", "US"])
        subscription_type = st.selectbox(
            "Tipe Langganan",
            ["Free", "Premium", "Student", "Family"],
            help="Free: gratis dengan iklan | Premium: berbayar tanpa iklan | Student/Family: paket khusus"
        )
        device_type = st.selectbox("Perangkat Utama", ["Desktop", "Mobile", "Web"])

    with col2:
        st.markdown("##### 🎵 Perilaku Mendengarkan")
        listening_time = st.slider(
            "Total Waktu Mendengarkan (menit)", 0, 300, 150, step=1,
            help="Total menit mendengarkan musik per bulan"
        )
        songs_played_per_day = st.slider(
            "Lagu Diputar per Hari", 0, 100, 20, step=1
        )
        skip_rate = st.slider(
            "Skip Rate", 0.0, 0.6, 0.3, step=0.01,
            help="0.0 = tidak pernah skip | 0.6 = sering skip"
        )
        ads_listened_per_week = st.number_input(
            "Iklan Didengar per Minggu", min_value=0, max_value=50, value=0, step=1,
            help="0 untuk pengguna Premium (tidak ada iklan)"
        )
        offline_listening = st.radio(
            "Menggunakan Mode Offline?", [1, 0],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
            horizontal=True
        )

    # ── Ringkasan input ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📄 Ringkasan Data yang Diinput"):
        input_display = {
            "Usia"               : f"{age} tahun",
            "Jenis Kelamin"      : gender,
            "Negara"             : country,
            "Tipe Langganan"     : subscription_type,
            "Perangkat Utama"    : device_type,
            "Waktu Mendengarkan" : f"{listening_time} menit",
            "Lagu per Hari"      : f"{songs_played_per_day} lagu",
            "Skip Rate"          : f"{skip_rate:.0%}",
            "Iklan per Minggu"   : f"{ads_listened_per_week} iklan",
            "Mode Offline"       : "Ya" if offline_listening == 1 else "Tidak",
        }
        df_disp = pd.DataFrame(input_display.items(), columns=["Fitur", "Nilai"])
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

    # ── Tombol prediksi ────────────────────────────────────────────────────────
    st.markdown("---")
    predict_btn = st.button("🔍 Prediksi Sekarang", use_container_width=True)

    if predict_btn:
        raw_input = {
            "age"                  : age,
            "listening_time"       : listening_time,
            "songs_played_per_day" : songs_played_per_day,
            "skip_rate"            : skip_rate,
            "ads_listened_per_week": ads_listened_per_week,
            "offline_listening"    : offline_listening,
            "gender"               : gender,
            "country"              : country,
            "subscription_type"    : subscription_type,
            "device_type"          : device_type,
        }

        # Encode ke DataFrame (OHE, sama persis dengan training)
        df_encoded = encode_inputs(raw_input)

        # Load model & scaler
        try:
            model  = load_model("model_spotify_churn.pkl")
            scaler = load_model("scaler_spotify_churn.pkl")
        except FileNotFoundError as e:
            st.error(f"❌ File model tidak ditemukan: {e}")
            return

        # Validasi jumlah fitur
        expected_features = scaler.n_features_in_
        actual_features   = df_encoded.shape[1]

        if actual_features != expected_features:
            st.error(
                f"❌ Jumlah fitur tidak cocok!\n\n"
                f"Model membutuhkan **{expected_features} fitur**, "
                f"tetapi input menghasilkan **{actual_features} fitur**.\n\n"
                f"Jalankan cell berikut di Colab dan kirim hasilnya:\n"
                f"```python\nprint(X_train.columns.tolist())\n```"
            )
            st.write("**Kolom saat ini di app:**")
            st.write(FEATURE_COLUMNS)
            return

        # Scale & prediksi
        df_scaled  = scaler.transform(df_encoded)
        prediction = model.predict(df_scaled)[0]
        pred_proba = model.predict_proba(df_scaled)[0]

        prob_churn  = round(pred_proba[1] * 100, 1)
        prob_active = round(pred_proba[0] * 100, 1)

        # ── Hasil prediksi ─────────────────────────────────────────────────────
        st.subheader("🎯 Hasil Prediksi")

        r1, r2, r3 = st.columns(3)
        r1.metric("Probabilitas Churn", f"{prob_churn}%")
        r2.metric("Probabilitas Aktif", f"{prob_active}%")
        risk = "🔴 Tinggi" if prob_churn >= 70 else ("🟡 Sedang" if prob_churn >= 40 else "🟢 Rendah")
        r3.metric("Tingkat Risiko", risk)

        st.markdown("")

        if prediction == 1:
            st.error("⚠️ **Pelanggan ini diprediksi akan CHURN** (berhenti berlangganan).")
        else:
            st.success("✅ **Pelanggan ini diprediksi tetap AKTIF** berlangganan.")

        st.markdown("##### Probabilitas Prediksi")
        st.markdown(f"🔴 **Churn** — {prob_churn}%")
        st.progress(prob_churn / 100)
        st.markdown(f"🟢 **Aktif** — {prob_active}%")
        st.progress(prob_active / 100)

        # ── Rekomendasi ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Rekomendasi Tindakan")

        if prediction == 1:
            st.warning("Pelanggan ini berisiko churn. Pertimbangkan langkah berikut:")
            for a in [
                "🎁 Tawarkan diskon perpanjangan langganan atau upgrade plan",
                "📩 Kirim notifikasi personal berdasarkan genre / artis favorit",
                "🔔 Rekomendasikan fitur yang belum digunakan (misal: mode offline)",
                "🎧 Sajikan playlist mingguan yang dipersonalisasi",
                "📊 Pantau aktivitas pengguna ini lebih ketat selama 30 hari ke depan",
            ]:
                st.markdown(f"- {a}")
        else:
            st.info("Pelanggan ini masih aktif. Pertahankan engagement dengan:")
            for a in [
                "⭐ Tawarkan upgrade ke paket Premium jika masih menggunakan Free",
                "🎵 Kirim rekomendasi playlist & artis baru setiap minggu",
                "🤝 Dorong penggunaan fitur sosial (berbagi playlist, kolaborasi)",
                "🎁 Berikan reward loyalitas untuk pelanggan jangka panjang",
            ]:
                st.markdown(f"- {a}")

        # ── Debug detail ───────────────────────────────────────────────────────
        with st.expander("🔢 Detail Nilai yang Diproses Model"):
            df_detail = df_encoded.T.reset_index()
            df_detail.columns = ["Fitur", "Nilai (Encoded)"]
            df_detail["Nilai (Scaled)"] = df_scaled[0]
            st.dataframe(df_detail, use_container_width=True, hide_index=True)
