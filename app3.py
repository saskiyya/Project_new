import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import base64
import os

# ================================================================
# PAGE CONFIG (harus paling atas sebelum st command lainnya)
# ================================================================
st.set_page_config(
    page_title="Data Science Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# LOAD DATA & MODEL
# ================================================================
@st.cache_data
def load_data():
    return pd.read_csv("dataset_gaji_data_science.csv")

@st.cache_resource
def load_model():
    return joblib.load("model_gaji.joblib")

df = load_data()

# ================================================================
# VARIABEL GLOBAL DARI DATASET
# ================================================================
all_jabatan = sorted(df["Jabatan"].unique().tolist())
all_negara  = sorted(df["Negara Tinggal"].unique().tolist())

# ================================================================
# CSS GLOBAL
# ================================================================
st.markdown("""
<style>
.stApp {
    background-color: #A2AED8;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #6D789C;
}
.card {
    background: #6D789C;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 15px;
    color: white;
}
.top-header-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2C3E7A;
}
.top-header-sub {
    font-size: 0.95rem;
    color: #4A5580;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #2C3E7A;
    margin-bottom: 6px;
}
.section-sub {
    font-size: 0.9rem;
    color: #4A5580;
    margin-bottom: 12px;
}
.footer {
    text-align: center;
    color: #6D789C;
    font-size: 0.8rem;
    margin-top: 40px;
    padding: 10px;
}
.sidebar-badge {
    background: #6D789C;
    color: white;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    display: inline-block;
    margin-bottom: 8px;
}
.sidebar-desc-box {
    background: rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #dde2f0;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("logo_tas.png"):
            st.image("logo_tas.png", width=100)

    st.markdown("""
    <div style="text-align:center; font-weight:700; font-size:24px; margin-top:8px; color:#FFFFFF;">
        SalaryPrediction
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-badge">👩‍💻 Developer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-badge">Saskia Humaira</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-desc-box">
        Aplikasi ini memprediksi estimasi gaji profesional Data Science berdasarkan pengalaman,
        jenis pekerjaan, &amp; sistem kerja.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#dde2f0; font-size:0.9rem;">
        🎯 <b>Panduan Penggunaan</b>
        <ul>
            <li>Pilih Tahun Kerja</li>
            <li>Tentukan Tingkat Pengalaman</li>
            <li>Pilih Jenis Pekerjaan</li>
            <li>Pilih Sistem Kerja</li>
            <li>Klik tombol Prediksi Gaji</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#dde2f0; font-size:0.9rem;">
        ✨ <b>Fitur</b>
        <ul>
            <li>Prediksi Gaji Real-time</li>
            <li>Input Interaktif</li>
            <li>Informasi &amp; Statistik Dataset</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
col_h1, col_h2, col_h3 = st.columns([1, 6, 1])

with col_h1:
    if os.path.exists("logo_tas.png"):
        st.image("logo_tas.png", width=70)

with col_h2:
    st.markdown("""
    <div class="top-header-title">Data Science Salary Prediction</div>
    <div class="top-header-sub">Machine Learning Regression App - Model: Linear Regression</div>
    """, unsafe_allow_html=True)

with col_h3:
    if os.path.exists("Logo_SMK_Negeri_1_Purbalingga.png"):
        st.image("Logo_SMK_Negeri_1_Purbalingga.png", width=65)

st.markdown("""<hr style="border:none; border-top:1px solid #000000; margin:0 0 0 0;">""", unsafe_allow_html=True)

# ================================================================
# NAVIGASI
# ================================================================
st.markdown("<br>", unsafe_allow_html=True)
tabs = ["🔮 Prediction", "🏠 Information", "📊 Developer", "📈 Analisis Kode"]
selected = st.radio("nav", tabs, horizontal=True, label_visibility="collapsed")
st.markdown("""<hr style="border:none; border-top:1px solid #000000; margin:0 0 20px 0;">""", unsafe_allow_html=True)

# ================================================================
# TAB: PREDICTION
# ================================================================
if selected == "🔮 Prediction":
    st.markdown("""
    Aplikasi Machine Learning regresi untuk memprediksi **Gaji Data Science (USD)** berdasarkan faktor seperti
    **Tahun Kerja**, **Tingkat Pengalaman**, **Jenis Pekerjaan**, dan **Sistem Kerja**.
    """)

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    st.subheader("📊 Input Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        tahun_kerja = st.selectbox(
            "📅 Tahun Kerja",
            options=sorted(df["Tahun Kerja"].unique().tolist(), reverse=True)
        )
        tingkat_pengalaman = st.selectbox(
            "🏅 Tingkat Pengalaman",
            options=["Pemula", "Menengah", "Senior", "Eksekutif"]
        )
        jabatan = st.selectbox(
            "🔖 Jabatan",
            options=all_jabatan
        )

    with col2:
        jenis_pekerjaan = st.selectbox(
            "💼 Jenis Pekerjaan",
            options=df["Jenis Pekerjaan"].unique().tolist()
        )
        sistem_kerja = st.selectbox(
            "🏢 Sistem Kerja",
            options=df["Sistem Kerja"].unique().tolist()
        )
        ukuran_perusahaan = st.selectbox(
            "🏗️ Ukuran Perusahaan",
            options=df["Ukuran Perusahaan"].unique().tolist()
        )

    with col3:
        negara_tinggal = st.selectbox(
            "🌍 Negara Tinggal",
            options=all_negara
        )
        lokasi_perusahaan = st.selectbox(
            "📍 Lokasi Perusahaan",
            options=sorted(df["Lokasi Perusahaan"].unique().tolist())
        )
        mata_uang = st.selectbox(
            "💱 Mata Uang",
            options=df["Mata Uang"].unique().tolist()
        )

    # Catatan: Model hanya menggunakan 4 fitur utama
    st.info("ℹ️ Prediksi menggunakan **4 fitur utama** model: Tahun Kerja, Tingkat Pengalaman, Jenis Pekerjaan, dan Sistem Kerja.", icon="ℹ️")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Prediksi Sekarang", use_container_width=True):
        # Model hanya dilatih dengan 4 fitur ini
        data_input = {
            "Tahun Kerja": tahun_kerja,
            "Tingkat Pengalaman": tingkat_pengalaman,
            "Jenis Pekerjaan": jenis_pekerjaan,
            "Sistem Kerja": sistem_kerja,
        }
        data_baru = pd.DataFrame([data_input])

        try:
            prediksi_raw = model.predict(data_baru)[0]
            prediksi = max(0, prediksi_raw)
        except Exception as ex:
            st.error(f"Error saat prediksi: {ex}")
            prediksi = 0

        st.markdown("## 🎯 Hasil Prediksi")

        col_hasil, col_detail = st.columns([2, 1])

        with col_hasil:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg,#36D1DC,#5B86E5);
                padding:30px;
                border-radius:15px;
                text-align:center;
                color:white;
            ">
                <h3>Estimasi Gaji</h3>
                <h1>${prediksi:,.0f}</h1>
                <p>Perkiraan gaji per tahun (USD)</p>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            st.markdown(f"""
            <div style="
                background-color:#F4F6F7;
                padding:20px;
                border-radius:12px;
                height: 100%;
            ">
                <h4>📊 Ringkasan</h4>
                <p>📅 Tahun: {tahun_kerja}</p>
                <p>🏅 Pengalaman: {tingkat_pengalaman}</p>
                <p>💼 Jenis: {jenis_pekerjaan}</p>
                <p>🏢 Sistem: {sistem_kerja}</p>
            </div>
            """, unsafe_allow_html=True)

        # Bar Chart
        st.markdown("### 📈 Perbandingan Rata-rata Gaji per Tingkat Pengalaman")
        avg_gaji = df.groupby("Tingkat Pengalaman")["Gaji (USD)"].mean().reset_index()
        avg_gaji.columns = ["Tingkat Pengalaman", "Rata-rata Gaji (USD)"]
        urutan = ["Pemula", "Menengah", "Senior", "Eksekutif"]
        avg_gaji["Tingkat Pengalaman"] = pd.Categorical(avg_gaji["Tingkat Pengalaman"], categories=urutan, ordered=True)
        avg_gaji = avg_gaji.sort_values("Tingkat Pengalaman")
        colors = ["#E74C3C" if t == tingkat_pengalaman else "#3498DB" for t in avg_gaji["Tingkat Pengalaman"]]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(avg_gaji["Tingkat Pengalaman"], avg_gaji["Rata-rata Gaji (USD)"], color=colors)
        ax.set_xlabel("Tingkat Pengalaman")
        ax.set_ylabel("Rata-rata Gaji (USD)")
        ax.set_title("Rata-rata Gaji per Tingkat Pengalaman")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                    f"${bar.get_height():,.0f}", ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        <div style="
            background-color:#EBF5FB;
            padding:20px;
            border-radius:12px;
            border-left:5px solid #3498DB;
            margin-top: 15px;
        ">
            💡 Berdasarkan kondisi yang dipilih, estimasi gaji adalah sekitar <b>${prediksi:,.0f} USD/tahun</b>.
            Bar merah menunjukkan tingkat pengalaman Anda saat ini (<b>{tingkat_pengalaman}</b>).
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    st.markdown('<div class="footer">©2026 - Dibuat dengan ❤ oleh Saskia Humaira menggunakan Streamlit &amp; Scikit-Learn</div>', unsafe_allow_html=True)

# ================================================================
# TAB: INFORMATION
# ================================================================
elif selected == "🏠 Information":
    st.markdown('<div class="section-title">🏠 Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Eksplorasi statistik dan tren dari dataset gaji Data Science yang digunakan untuk melatih model:</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{df.shape[0]:,}")
    col2.metric("Rata-rata Gaji (USD)", f"${df['Gaji (USD)'].mean():,.0f}")
    col3.metric("Gaji Tertinggi (USD)", f"${df['Gaji (USD)'].max():,}")
    col4.metric("Gaji Terendah (USD)", f"${df['Gaji (USD)'].min():,}")

    st.markdown("---")
    st.markdown('<div class="section-title" style="font-size:1.2rem;">🚀 Input Prediksi</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .info-card {
        background: linear-gradient(135deg, #6D789C 0%, #A2AED8 100%);
        border-radius: 14px;
        padding: 18px 20px;
        color: #ffffff;
        font-size: 0.88rem;
        line-height: 1.6;
        height: 100%;
        box-sizing: border-box;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 12px rgba(109,120,156,0.25);
        margin-bottom: 14px;
    }
    .info-card .ic-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .info-card p { color: rgba(255,255,255,0.88); margin: 0; }
    </style>
    """, unsafe_allow_html=True)

    row1 = st.columns(3)
    with row1[0]:
        st.markdown("""<div class="info-card"><div style="font-size:1.6rem">📅</div>
            <div class="ic-title">Tahun Kerja</div>
            <p>Menunjukkan tahun kapan data pekerjaan atau gaji dicatat. Penting karena gaji berubah setiap tahun mengikuti kondisi ekonomi dan tren industri.</p>
        </div>""", unsafe_allow_html=True)
    with row1[1]:
        st.markdown("""<div class="info-card"><div style="font-size:1.6rem">🎯</div>
            <div class="ic-title">Tingkat Pengalaman</div>
            <p>Menggambarkan level pengalaman seseorang dalam bekerja. Mulai dari Pemula, Menengah, Senior, hingga Eksekutif.</p>
        </div>""", unsafe_allow_html=True)
    with row1[2]:
        st.markdown("""<div class="info-card"><div style="font-size:1.6rem">🧾</div>
            <div class="ic-title">Jenis Pekerjaan</div>
            <p>Status pekerjaan yang dijalani — tetap, kontrak, freelance, atau paruh waktu. Berpengaruh pada stabilitas kerja dan sistem penggajian.</p>
        </div>""", unsafe_allow_html=True)

    row2 = st.columns(3)
    with row2[0]:
        st.markdown("""<div class="info-card"><div style="font-size:1.6rem">🏢</div>
            <div class="ic-title">Sistem Kerja</div>
            <p>Menjelaskan lokasi kerja karyawan — Remote, Hybrid, atau Kantor. Mempengaruhi tunjangan dan kompensasi yang diterima.</p>
        </div>""", unsafe_allow_html=True)
    with row2[1]:
        st.markdown("""<div class="info-card"><div style="font-size:1.6rem">🏭</div>
            <div class="ic-title">Ukuran Perusahaan</div>
            <p>Skala perusahaan tempat bekerja — Kecil, Menengah, atau Besar. Perusahaan besar umumnya menawarkan gaji lebih kompetitif.</p>
        </div>""", unsafe_allow_html=True)
    with row2[2]:
        st.markdown("""<div class="info-card"><div style="font-size:1.6rem">👔</div>
            <div class="ic-title">Jabatan</div>
            <p>Posisi spesifik dalam bidang Data Science seperti Data Analyst, ML Engineer, Data Scientist, dan lainnya yang mempengaruhi rentang gaji.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Visualisasi Dataset</div>', unsafe_allow_html=True)

    if os.path.exists("grafik_information.png"):
        st.image("grafik_information.png", use_container_width=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df["Tingkat Pengalaman"].value_counts().plot(kind="bar", ax=axes[0], color="#6D789C")
        axes[0].set_title("Distribusi Tingkat Pengalaman")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis='x', rotation=30)
        df.groupby("Tingkat Pengalaman")["Gaji (USD)"].mean().reindex(["Pemula", "Menengah", "Senior", "Eksekutif"]).plot(
            kind="bar", ax=axes[1], color="#A2AED8")
        axes[1].set_title("Rata-rata Gaji per Tingkat Pengalaman")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="footer">©2026 - Dibuat dengan ❤ oleh Saskia Humaira menggunakan Streamlit &amp; Scikit-Learn</div>', unsafe_allow_html=True)

# ================================================================
# TAB: DEVELOPER
# ================================================================
elif selected == "📊 Developer":
    st.markdown("## 👩‍💻 Developer")

    def get_base64(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img = get_base64("saskiaww.jpg") if os.path.exists("saskiaww.jpg") else ""
    img_tag = (
        f'<img src="data:image/jpeg;base64,{img}" style="width:220px;height:220px;object-fit:cover;border-radius:50%;border:6px solid #D9D8FF;box-shadow:0 8px 20px rgba(0,0,0,0.1);">'
        if img else
        '<div style="width:220px;height:220px;border-radius:50%;background:#6D789C;display:flex;align-items:center;justify-content:center;font-size:4rem;">👩‍💻</div>'
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f'<div style="text-align:center;">{img_tag}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h2 style="margin-bottom:5px;">👩‍💻 Saskia Humaira</h2>
            <p>Machine Learning Enthusiast</p>
            <p>Aplikasi ini dibuat untuk memprediksi estimasi gaji profesional Data Science berdasarkan
            pengalaman, jenis pekerjaan, dan sistem kerja menggunakan model Linear Regression.</p>
            <p>✉️ <a href="mailto:saskiahumaira6@gmail.com" style="text-decoration:none; color:#D9D8FF; font-weight:500;">
               saskiahumaira6@gmail.com</a></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.link_button("💻 Github", "https://github.com/saskiyya", use_container_width=True)
        with col_b:
            st.link_button("📷 Instagram", "https://instagram.com/saskyqarnaen_", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
            <b>🛠️ Tools</b><br><br>
            🐍 Python<br>
            📊 Streamlit<br>
            🧠 Machine Learning<br>
            📈 Data Analysis
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <b>🚀 Fitur</b><br><br>
            💰 Prediksi Gaji Data Science<br>
            📉 Grafik hasil prediksi<br>
            ⌨️ Input interaktif<br>
            📊 Analisis dataset
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">©2026 - Dibuat dengan ❤ oleh Saskia Humaira menggunakan Streamlit &amp; Scikit-Learn</div>', unsafe_allow_html=True)

# ================================================================
# TAB: ANALISIS KODE
# ================================================================
elif selected == "📈 Analisis Kode":
    st.markdown("<h2 style='text-align: center;'>📚 Analisis Kode</h2>", unsafe_allow_html=True)

    subtab1, subtab2 = st.tabs(["📊 Dataset", "📓 Notebook"])

    with subtab1:
        st.subheader("📊 Dataset Gaji Data Science")
        st.markdown("Dataset yang digunakan berisi informasi gaji profesional data science dari berbagai negara dan jabatan.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", df.shape[0])
        col2.metric("Jumlah Kolom", df.shape[1])
        col3.metric("Missing Value", int(df.isna().sum().sum()))

        st.markdown("### 📄 Tabel Dataset")
        st.dataframe(df, use_container_width=True)

    with subtab2:
        st.markdown("### 📓 Notebook")

        st.markdown("#### 📁 1. Load Dataset")
        st.code("""
import pandas as pd
df = pd.read_csv("dataset_gaji_data_science.csv")
df
        """, language='python')
        st.dataframe(df)

        st.markdown("#### 🔍 2. Data Inspection (EDA)")
        st.code("df.shape", language='python')
        st.write(df.shape)

        st.code("df.columns", language='python')
        st.write(df.columns)

        st.code("df.dtypes", language='python')
        st.write(df.dtypes)

        st.code("df.info()", language='python')
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.code("df.describe()", language='python')
        st.write(df.describe())

        st.code("df.head()", language='python')
        st.write(df.head())

        st.code("df.tail()", language='python')
        st.write(df.tail())

        st.code("df.sample()", language='python')
        st.write(df.sample())

        st.code("df.isna().sum()", language='python')
        st.write(df.isna().sum())

        st.code("df.duplicated().sum()", language='python')
        st.write(df.duplicated().sum())

        st.markdown("#### 📊 3. Visualisasi Data")

        st.code("""
import seaborn as sns
import matplotlib.pyplot as plt

df["Tingkat Pengalaman"].value_counts().plot(kind='bar', color='purple')
plt.title("Tingkat Pengalaman")
plt.show()
        """, language='python')
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        df["Tingkat Pengalaman"].value_counts().plot(kind='bar', color='purple', ax=ax1)
        ax1.set_title("Tingkat Pengalaman")
        ax1.tick_params(axis='x', rotation=30)
        st.pyplot(fig1)
        plt.close()

        st.code("""
sns.histplot(df["Gaji (USD)"], kde=True, color="green")
plt.title("Distribusi Gaji (USD)")
plt.show()
        """, language='python')
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.histplot(df["Gaji (USD)"], kde=True, color="green", ax=ax2)
        ax2.set_title("Distribusi Gaji (USD)")
        st.pyplot(fig2)
        plt.close()

        st.code("""
df["Jenis Pekerjaan"].value_counts().plot(kind='bar', color='blue')
plt.title("Jenis Pekerjaan")
plt.show()
        """, language='python')
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        df["Jenis Pekerjaan"].value_counts().plot(kind='bar', color='blue', ax=ax3)
        ax3.set_title("Jenis Pekerjaan")
        ax3.tick_params(axis='x', rotation=30)
        st.pyplot(fig3)
        plt.close()

        st.code("""
df["Sistem Kerja"].value_counts().plot(kind='bar', color='red')
plt.title("Sistem Kerja")
plt.show()
        """, language='python')
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        df["Sistem Kerja"].value_counts().plot(kind='bar', color='red', ax=ax4)
        ax4.set_title("Sistem Kerja")
        ax4.tick_params(axis='x', rotation=30)
        st.pyplot(fig4)
        plt.close()

        st.code("""
df["Ukuran Perusahaan"].value_counts().plot(kind='bar', color='brown')
plt.title("Ukuran Perusahaan")
plt.show()
        """, language='python')
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        df["Ukuran Perusahaan"].value_counts().plot(kind='bar', color='brown', ax=ax5)
        ax5.set_title("Ukuran Perusahaan")
        ax5.tick_params(axis='x', rotation=30)
        st.pyplot(fig5)
        plt.close()

        st.markdown("#### 🌡️ 4. Heatmap Korelasi")
        st.code("""
num_features = ['Tahun Kerja', 'Gaji (USD)']
sns.heatmap(df[num_features].corr(), annot=True, cmap="coolwarm")
plt.show()
        """, language='python')
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        sns.heatmap(df[['Tahun Kerja', 'Gaji (USD)']].corr(), annot=True, cmap="coolwarm", ax=ax6)
        st.pyplot(fig6)
        plt.close()

        st.markdown("#### 🤖 5. Modeling")

        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer

        X = df[["Tahun Kerja", "Tingkat Pengalaman", "Jenis Pekerjaan", "Sistem Kerja"]]
        y = df["Gaji (USD)"]
        numerical = ["Tahun Kerja"]
        categorical = ["Tingkat Pengalaman", "Jenis Pekerjaan", "Sistem Kerja"]
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression
        st.markdown("### 🔵 Linear Regression")
        st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

X = df[["Tahun Kerja", "Tingkat Pengalaman", "Jenis Pekerjaan", "Sistem Kerja"]]
y = df["Gaji (USD)"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["Tahun Kerja"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Tingkat Pengalaman", "Jenis Pekerjaan", "Sistem Kerja"])
])

model_linear = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_linear.fit(X_train, y_train)
y_pred = model_linear.predict(X_test)

print(f"R2: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
        """, language='python')
        model_lr = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        st.write(f"R2: {r2_score(y_test, y_pred_lr):.3f} | MAE: {mean_absolute_error(y_test, y_pred_lr):.3f} | MSE: {mean_squared_error(y_test, y_pred_lr):.3f}")

        # Decision Tree
        st.markdown("### 🌲 Decision Tree")
        st.code("""
from sklearn.tree import DecisionTreeRegressor

model_tree = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])
model_tree.fit(X_train, y_train)
y_pred = model_tree.predict(X_test)

print(f"R2: {r2_score(y_test, y_pred):.3f}")
        """, language='python')
        model_dt = Pipeline([("preprocessor", preprocessor), ("regressor", DecisionTreeRegressor(random_state=42))])
        model_dt.fit(X_train, y_train)
        y_pred_dt = model_dt.predict(X_test)
        st.write(f"R2: {r2_score(y_test, y_pred_dt):.3f} | MAE: {mean_absolute_error(y_test, y_pred_dt):.3f} | MSE: {mean_squared_error(y_test, y_pred_dt):.3f}")

        # Random Forest
        st.markdown("### 🌳 Random Forest")
        st.code("""
from sklearn.ensemble import RandomForestRegressor

model_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)

print(f"R2: {r2_score(y_test, y_pred):.3f}")
        """, language='python')
        model_rf = Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=42))])
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        st.write(f"R2: {r2_score(y_test, y_pred_rf):.3f} | MAE: {mean_absolute_error(y_test, y_pred_rf):.3f} | MSE: {mean_squared_error(y_test, y_pred_rf):.3f}")

        # Cross Validation
        st.markdown("### 🔁 Cross Validation")
        st.code("""
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_rf, X_train, y_train, cv=5, scoring="r2")
print("Scores:", scores)
print("Mean R2:", scores.mean())
        """, language='python')
        scores = cross_val_score(model_rf, X_train, y_train, cv=5, scoring="r2")
        st.write(f"Scores: {scores}")
        st.write(f"Mean R2: {scores.mean():.3f}")

        # Save Model
        st.markdown("### ✅ Save Model")
        st.code("""
import joblib
joblib.dump(model_linear, "model_gaji.joblib")
print("Model berhasil disimpan!")
        """, language='python')
        st.success("Model Linear Regression siap digunakan (model_gaji.joblib)")

        # Prediksi Data Baru
        st.markdown("### 🔍 Prediksi Data Baru")
        st.code("""
import joblib, pandas as pd

model = joblib.load("model_gaji.joblib")

data_baru = pd.DataFrame([{
    "Tahun Kerja": 2023,
    "Tingkat Pengalaman": "Senior",
    "Jenis Pekerjaan": "Tetap",
    "Sistem Kerja": "Remote"
}])

prediksi = model.predict(data_baru)[0]
print(f"Estimasi Gaji: ${prediksi:,.0f} USD/tahun")
        """, language='python')
        # Gunakan model yang sudah di-load di awal (bukan dari file lokal)
        model_loaded = load_model()
        data_contoh = pd.DataFrame([{
            "Tahun Kerja": 2023,
            "Tingkat Pengalaman": "Senior",
            "Jenis Pekerjaan": "Tetap",
            "Sistem Kerja": "Remote"
        }])
        hasil = model_loaded.predict(data_contoh)[0]
        st.write(f"Estimasi Gaji: ${hasil:,.0f} USD/tahun")

    st.markdown('<div class="footer">©2026 - Dibuat dengan ❤ oleh Saskia Humaira menggunakan Streamlit &amp; Scikit-Learn</div>', unsafe_allow_html=True)