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
| `country` | Kategorikal | Negara asal (AU, US, DE, IN, PK, FR, UK, CA) |
| `subscription_type` | Kategorikal | Free / Premium / Student / Family |
| `listening_time` | Numerik | Total menit mendengarkan (0–300 menit) |
| `songs_played_per_day` | Numerik | Rata-rata lagu diputar per hari (0–100) |
| `skip_rate` | Numerik | Proporsi lagu yang diskip, 0.0 (tidak pernah) – 0.6 (sering) |
| `ads_listened_per_week` | Numerik | Jumlah iklan didengar per minggu (0–50) |
| `offline_listening` | Biner | 1 = Ya (menggunakan mode offline), 0 = Tidak |
| `device_type` | Kategorikal | Perangkat utama: Mobile / Desktop / Web |
"""

# ── Encoding maps ──────────────────────────────────────────────────────────────
gender_map          = {"Male": 0, "Female": 1, "Other": 2}
country_map         = {"AU": 0, "CA": 1, "DE": 2, "FR": 3, "IN": 4, "PK": 5, "UK": 6, "US": 7}
subscription_map    = {"Family": 0, "Free": 1, "Premium": 2, "Student": 3}
device_map          = {"Desktop": 0, "Mobile": 1, "Web": 2}

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_model(model_file):
    return joblib.load(open(os.path.join(model_file), "rb"))


def encode_inputs(raw: dict) -> np.ndarray:
    """Encode raw input dict menjadi array numerik sesuai urutan fitur training."""
    return np.array([
        raw["age"],
        raw["listening_time"],
        raw["songs_played_per_day"],
        raw["skip_rate"],
        raw["ads_listened_per_week"],
        raw["offline_listening"],
        gender_map[raw["gender"]],
        country_map[raw["country"]],
        subscription_map[raw["subscription_type"]],
        device_map[raw["device_type"]],
    ]).reshape(1, -1)


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
        device_type = st.selectbox("Perangkat Utama", ["Mobile", "Desktop", "Web"])

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

    # ── Tampilkan ringkasan input ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📄 Ringkasan Data yang Diinput"):
        input_display = {
            "Usia"                      : f"{age} tahun",
            "Jenis Kelamin"             : gender,
            "Negara"                    : country,
            "Tipe Langganan"            : subscription_type,
            "Perangkat Utama"           : device_type,
            "Waktu Mendengarkan"        : f"{listening_time} menit",
            "Lagu per Hari"             : f"{songs_played_per_day} lagu",
            "Skip Rate"                 : f"{skip_rate:.0%}",
            "Iklan per Minggu"          : f"{ads_listened_per_week} iklan",
            "Mode Offline"              : "Ya" if offline_listening == 1 else "Tidak",
        }
        df_input = pd.DataFrame(input_display.items(), columns=["Fitur", "Nilai"])
        st.dataframe(df_input, use_container_width=True, hide_index=True)

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

        encoded = encode_inputs(raw_input)

        # Load model
        try:
            model  = load_model("model_spotify_churn.pkl")
            scaler = load_model("scaler_spotify_churn.pkl")
        except FileNotFoundError as e:
            st.error(f"❌ File model tidak ditemukan: {e}\n\nPastikan `model_spotify_churn.pkl` dan `scaler_spotify_churn.pkl` berada di folder yang sama.")
            return

        encoded_scaled = scaler.transform(encoded)

        prediction = model.predict(encoded_scaled)[0]
        pred_proba = model.predict_proba(encoded_scaled)[0]

        prob_churn  = round(pred_proba[1] * 100, 1)
        prob_active = round(pred_proba[0] * 100, 1)

        # ── Hasil ─────────────────────────────────────────────────────────────
        st.subheader("🎯 Hasil Prediksi")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Probabilitas Churn",  f"{prob_churn}%")
        res_col2.metric("Probabilitas Aktif",  f"{prob_active}%")
        risk = "🔴 Tinggi" if prob_churn >= 70 else ("🟡 Sedang" if prob_churn >= 40 else "🟢 Rendah")
        res_col3.metric("Tingkat Risiko", risk)

        st.markdown("")

        if prediction == 1:
            st.error(f"⚠️ **Pelanggan ini diprediksi akan CHURN** (berhenti berlangganan).")
        else:
            st.success(f"✅ **Pelanggan ini diprediksi tetap AKTIF** berlangganan.")

        # ── Progress bar probabilitas ─────────────────────────────────────────
        st.markdown("##### Probabilitas Prediksi")
        st.markdown(f"🔴 **Churn** — {prob_churn}%")
        st.progress(prob_churn / 100)
        st.markdown(f"🟢 **Aktif** — {prob_active}%")
        st.progress(prob_active / 100)

        # ── Rekomendasi ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Rekomendasi Tindakan")

        if prediction == 1:
            st.warning("Pelanggan ini berisiko churn. Pertimbangkan langkah berikut:")
            aksi = [
                "🎁 Tawarkan diskon perpanjangan langganan atau upgrade plan",
                "📩 Kirim notifikasi personal berdasarkan genre / artis favorit",
                "🔔 Rekomendasikan fitur yang belum digunakan (misal: mode offline, playlist kolaborasi)",
                "🎧 Sajikan playlist mingguan yang dipersonalisasi",
                "📊 Pantau aktivitas pengguna ini lebih ketat selama 30 hari ke depan",
            ]
            for a in aksi:
                st.markdown(f"- {a}")
        else:
            st.info("Pelanggan ini masih aktif. Pertahankan engagement dengan:")
            aksi = [
                "⭐ Tawarkan upgrade ke paket Premium jika masih menggunakan Free",
                "🎵 Kirim rekomendasi playlist & artis baru setiap minggu",
                "🤝 Dorong penggunaan fitur sosial (berbagi playlist, kolaborasi)",
                "🎁 Berikan reward loyalitas untuk pelanggan jangka panjang",
            ]
            for a in aksi:
                st.markdown(f"- {a}")

        # ── Detail encoded ─────────────────────────────────────────────────────
        with st.expander("🔢 Detail Nilai yang Diproses Model"):
            feature_names = [
                "age", "listening_time", "songs_played_per_day", "skip_rate",
                "ads_listened_per_week", "offline_listening",
                "gender (encoded)", "country (encoded)",
                "subscription_type (encoded)", "device_type (encoded)",
            ]
            df_encoded = pd.DataFrame(
                zip(feature_names, encoded[0], encoded_scaled[0]),
                columns=["Fitur", "Nilai Asli (Encoded)", "Nilai Setelah Scaling"]
            )
            st.dataframe(df_encoded, use_container_width=True, hide_index=True)
