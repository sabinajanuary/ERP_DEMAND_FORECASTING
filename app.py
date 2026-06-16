import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="ERP Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.kpi-card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}

.big-font {
    font-size: 22px !important;
    font-weight: bold;
}

.small-font {
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# =========================
# SIDEBAR
# =========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=100
)

st.sidebar.title("ERP Dashboard")

st.sidebar.markdown("""
### Kelompok 10

Topik 1

### Algoritma
Decision Tree Regression

### Dataset
Store Sales Time Series Forecasting

### Mata Kuliah
Kecerdasan AI
""")

# =========================
# HEADER
# =========================

st.title("📊 ERP Demand Forecasting System")

st.markdown("""
Sistem ini memanfaatkan Machine Learning
menggunakan algoritma Decision Tree Regression
untuk melakukan prediksi permintaan produk
(demand forecasting) pada modul Supply Chain ERP.
""")

st.divider()

# =========================
# TABS
# =========================

tab1, tab2, tab3 = st.tabs([
    "📈 Dashboard",
    "📊 Analytics",
    "🔮 Forecasting"
])

# ===================================================
# DASHBOARD
# ===================================================

with tab1:

    st.subheader("Ringkasan Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Jumlah Data",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Jumlah Toko",
            df["store_nbr"].nunique()
        )

    with col3:
        st.metric(
            "Kategori Produk",
            df["family"].nunique()
        )

    with col4:
        st.metric(
            "Total Promosi",
            f"{int(df['onpromotion'].sum()):,}"
        )

    st.divider()

    st.subheader("Preview Dataset")

    st.dataframe(df.head(10))

# ===================================================
# ANALYTICS
# ===================================================

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Tren Penjualan")

        sales_per_day = df.groupby(
            "date"
        )["sales"].sum()

        fig, ax = plt.subplots(figsize=(8,4))

        ax.plot(
            sales_per_day.index,
            sales_per_day.values
        )

        ax.set_title(
            "Total Penjualan Harian"
        )

        st.pyplot(fig)

    with col2:

        st.subheader(
            "Top 5 Kategori Produk"
        )

        top_family = (
            df.groupby("family")["sales"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(5)
        )

        fig2, ax2 = plt.subplots()

        ax2.pie(
            top_family,
            labels=top_family.index,
            autopct="%1.1f%%"
        )

        st.pyplot(fig2)

# ===================================================
# FORECASTING
# ===================================================

with tab3:

    st.subheader(
        "Demand Forecasting"
    )

    X = df[["onpromotion"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeRegressor(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    promo = st.number_input(
    "Masukkan Jumlah Promosi",
    min_value=0,
    value=10,
    step=1
    )

    if st.button(
        "Prediksi Penjualan"
    ):

        prediction = model.predict(
            [[promo]]
        )[0]

        colA, colB = st.columns(2)

        with colA:

            st.success(
                f"""
                Prediksi Penjualan

                {prediction:.2f} Unit
                """
            )

        with colB:

            st.info(
                f"""
                Rekomendasi Persediaan

                {prediction*1.1:.2f} Unit
                """
            )

        st.warning(
            "Gunakan hasil prediksi sebagai dasar perencanaan stok untuk mengurangi risiko overstock dan stockout."
        )

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "ERP Demand Forecasting System | Decision Tree Regression"
)