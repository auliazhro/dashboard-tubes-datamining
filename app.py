import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import ast

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# =========================================================
# KONFIGURASI PARAMETER
# =========================================================
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.50
TOP_REKOMENDASI = 5


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)


# =========================================================
# STYLE DASHBOARD
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    h1 {
        font-size: 36px !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    .hero-card {
        background: linear-gradient(90deg, #1f2937, #111827);
        padding: 24px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 16px;
        opacity: 0.9;
    }

    .info-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        padding-top: 30px;
        color: gray;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    possible_files = [
        "bread_basket_per_transaksi.xlsx",
        "hasil_apriori_bread_basket.xlsx",
        "hasil_apriori_bread_basket_kel12.xlsx"
    ]

    current_path = Path(__file__).parent

    for file_name in possible_files:
        file_path = current_path / file_name
        if file_path.exists():
            df = pd.read_excel(file_path)
            return df, file_name

    return None, None


df, file_name = load_data()

if df is None:
    st.error(
        "File dataset tidak ditemukan. Pastikan salah satu file berikut berada satu folder dengan app.py: "
        "`bread_basket_per_transaksi.xlsx`, `hasil_apriori_bread_basket.xlsx`, atau `hasil_apriori_bread_basket_kel12.xlsx`."
    )
    st.stop()


# =========================================================
# HEADER DASHBOARD
# =========================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Dashboard Market Basket Analysis</div>
        <div class="hero-subtitle">
            Analisis pola pembelian produk bakery menggunakan algoritma Apriori
            untuk rekomendasi product bundling.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA PREPARATION UNTUK DASHBOARD
# =========================================================
df_clean = df.copy()
df_clean = df_clean.drop_duplicates()

# Membuat nama kolom menjadi lebih seragam
df_clean.columns = [col.strip() for col in df_clean.columns]

# Deteksi kolom penting
kolom_lower = {col.lower(): col for col in df_clean.columns}

transaction_col = None
items_col = None
date_col = None
period_col = None
weekday_col = None

for col in df_clean.columns:
    col_lower = col.lower()

    if col_lower in ["transaction", "transaction_id", "id_transaksi"]:
        transaction_col = col

    if col_lower in ["items", "item", "produk", "product"]:
        items_col = col

    if col_lower in ["date_time", "datetime", "tanggal", "date"]:
        date_col = col

    if col_lower in ["period_day", "periode", "waktu_pembelian"]:
        period_col = col

    if col_lower in ["weekday_weekend", "jenis_hari"]:
        weekday_col = col


if transaction_col is None or items_col is None:
    st.error("Kolom transaksi atau kolom items tidak ditemukan pada dataset.")
    st.write("Kolom yang tersedia:", df_clean.columns.tolist())
    st.stop()


# Menghapus data kosong pada kolom penting
df_clean = df_clean.dropna(subset=[transaction_col, items_col])


# =========================================================
# FUNGSI MENGUBAH ITEMS MENJADI LIST
# =========================================================
def ubah_ke_list(x):
    if isinstance(x, list):
        return [str(item).strip() for item in x if str(item).strip() != ""]

    try:
        hasil = ast.literal_eval(str(x))
        if isinstance(hasil, list):
            return [str(item).strip() for item in hasil if str(item).strip() != ""]
    except:
        pass

    return [item.strip() for item in str(x).split(",") if item.strip() != ""]


df_clean["Items_List"] = df_clean[items_col].apply(ubah_ke_list)
df_clean["Jumlah_Item"] = df_clean["Items_List"].apply(len)

# Menghapus transaksi kosong
df_clean = df_clean[df_clean["Jumlah_Item"] > 0].copy()

transactions = df_clean["Items_List"].tolist()

flat_items = [item for transaksi in transactions for item in transaksi]
item_counts = pd.Series(flat_items).value_counts()

top_items_df = item_counts.reset_index()
top_items_df.columns = ["Produk", "Jumlah Pembelian"]


# =========================================================
# ONE-HOT ENCODING
# =========================================================
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket_sets = pd.DataFrame(te_array, columns=te.columns_)
basket_sets_int = basket_sets.astype(int)
basket_bool = basket_sets.astype(bool)


# =========================================================
# MODELING APRIORI
# =========================================================
frequent_itemsets = apriori(
    basket_bool,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

if len(frequent_itemsets) == 0:
    st.warning("Tidak ada frequent itemsets yang terbentuk. Coba turunkan minimum support.")
    st.stop()

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

frequent_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
)

frequent_display = frequent_itemsets.copy()
frequent_display["Itemsets"] = frequent_display["itemsets"].apply(
    lambda x: ", ".join(list(x))
)
frequent_display["Support (%)"] = frequent_display["support"] * 100

frequent_display = frequent_display[
    ["Itemsets", "Support (%)", "length"]
]


# =========================================================
# EVALUATION - ASSOCIATION RULES
# =========================================================
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

if len(rules) > 0:
    rules = rules.sort_values(
        by="lift",
        ascending=False
    )

    rules_filtered = rules[
        (rules["lift"] > 1) &
        (rules["confidence"] >= MIN_CONFIDENCE)
    ].copy()

    rules_filtered = rules_filtered.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )

    rules_filtered["antecedents_str"] = rules_filtered["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )

    rules_filtered["consequents_str"] = rules_filtered["consequents"].apply(
        lambda x: ", ".join(list(x))
    )

    rules_filtered["rule"] = (
        rules_filtered["antecedents_str"] +
        " → " +
        rules_filtered["consequents_str"]
    )

    rules_rekomendasi = rules_filtered.head(TOP_REKOMENDASI).copy()

else:
    rules_filtered = pd.DataFrame()
    rules_rekomendasi = pd.DataFrame()


# =========================================================
# MEMBUAT TABEL REKOMENDASI
# =========================================================
if len(rules_rekomendasi) > 0:
    tabel_rekomendasi = rules_rekomendasi[
        ["rule", "support", "confidence", "lift"]
    ].copy()

    tabel_rekomendasi["Support (%)"] = tabel_rekomendasi["support"] * 100
    tabel_rekomendasi["Confidence (%)"] = tabel_rekomendasi["confidence"] * 100
    tabel_rekomendasi["Lift Ratio"] = tabel_rekomendasi["lift"]

    tabel_rekomendasi = tabel_rekomendasi[
        ["rule", "Support (%)", "Confidence (%)", "Lift Ratio"]
    ]

else:
    tabel_rekomendasi = pd.DataFrame(
        columns=["rule", "Support (%)", "Confidence (%)", "Lift Ratio"]
    )


# =========================================================
# METRIC RINGKASAN
# =========================================================
total_transaksi = df_clean[transaction_col].nunique()
jumlah_item_unik = len(set(flat_items))
jumlah_frequent_itemsets = len(frequent_itemsets)
jumlah_rekomendasi = len(rules_rekomendasi)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transaksi", f"{total_transaksi:,}".replace(",", "."))
col2.metric("Item Unik", jumlah_item_unik)
col3.metric("Frequent Itemsets", jumlah_frequent_itemsets)
col4.metric("Top Rekomendasi", jumlah_rekomendasi, "rule")


st.markdown(
    f"""
    <div class="info-box">
        <b>Parameter yang digunakan:</b><br>
        Minimum Support = {MIN_SUPPORT * 100:.2f}% <br>
        Minimum Confidence = {MIN_CONFIDENCE * 100:.0f}% <br>
        Filter Evaluation = Lift > 1 dan Confidence ≥ {MIN_CONFIDENCE * 100:.0f}% <br>
        Rekomendasi akhir = Top {TOP_REKOMENDASI} association rules terbaik
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TOP 10 PRODUK
# =========================================================
st.subheader("Top 10 Produk Paling Sering Dibeli")

top_10_items = top_items_df.head(10)

fig_top_items = px.bar(
    top_10_items,
    x="Jumlah Pembelian",
    y="Produk",
    orientation="h",
    title="Top 10 Produk Paling Sering Dibeli"
)

fig_top_items.update_layout(
    yaxis=dict(autorange="reversed"),
    height=450
)

st.plotly_chart(fig_top_items, use_container_width=True)


# =========================================================
# TOP 5 REKOMENDASI BUNDLING
# =========================================================
st.subheader("Top 5 Rekomendasi Product Bundling")

st.caption(
    "Rekomendasi dipilih berdasarkan association rules dengan lift > 1, "
    "minimum confidence 50%, dan diurutkan berdasarkan nilai lift tertinggi."
)

if len(rules_rekomendasi) > 0:
    for idx, row in rules_rekomendasi.iterrows():
        st.markdown(
            f"""
            <div class="info-box">
                <b>{row['rule']}</b><br>
                Support: {row['support'] * 100:.2f}% &nbsp; | &nbsp;
                Confidence: {row['confidence'] * 100:.2f}% &nbsp; | &nbsp;
                Lift Ratio: {row['lift']:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.warning("Tidak ada rekomendasi bundling yang memenuhi kriteria filter.")


# =========================================================
# TABEL DETAIL REKOMENDASI
# =========================================================
st.subheader("Detail Nilai Rekomendasi")

st.dataframe(
    tabel_rekomendasi,
    use_container_width=True
)


# =========================================================
# GRAFIK TOP 5 REKOMENDASI
# =========================================================
if len(rules_rekomendasi) > 0:
    st.subheader("Visualisasi Top 5 Rekomendasi Berdasarkan Lift Ratio")

    fig_rules = px.bar(
        rules_rekomendasi,
        x="lift",
        y="rule",
        orientation="h",
        title="Top 5 Rekomendasi Product Bundling Berdasarkan Lift Ratio"
    )

    fig_rules.update_layout(
        yaxis=dict(autorange="reversed"),
        height=420
    )

    st.plotly_chart(fig_rules, use_container_width=True)


# =========================================================
# FREQUENT ITEMSETS
# =========================================================
st.subheader("Frequent Itemsets")

st.caption(
    "Frequent itemsets menunjukkan item atau kombinasi item yang sering muncul "
    "dalam transaksi berdasarkan minimum support."
)

st.dataframe(
    frequent_display.head(20),
    use_container_width=True
)


# =========================================================
# KOMBINASI 2 PRODUK YANG SERING MUNCUL
# =========================================================
frequent_2_item = frequent_itemsets[frequent_itemsets["length"] == 2].copy()

if len(frequent_2_item) > 0:
    st.subheader("Kombinasi 2 Produk yang Sering Dibeli Bersama")

    frequent_2_item["Kombinasi Produk"] = frequent_2_item["itemsets"].apply(
        lambda x: " + ".join(list(x))
    )

    frequent_2_item["Support (%)"] = frequent_2_item["support"] * 100

    top_pairs = frequent_2_item.sort_values(
        by="support",
        ascending=False
    ).head(10)

    fig_pairs = px.bar(
        top_pairs,
        x="Support (%)",
        y="Kombinasi Produk",
        orientation="h",
        title="Top Kombinasi 2 Produk Berdasarkan Support"
    )

    fig_pairs.update_layout(
        yaxis=dict(autorange="reversed"),
        height=450
    )

    st.plotly_chart(fig_pairs, use_container_width=True)

else:
    st.info("Tidak terdapat kombinasi 2 produk yang memenuhi minimum support.")


# =========================================================
# VISUALISASI WAKTU TRANSAKSI
# =========================================================
if period_col is not None:
    st.subheader("Distribusi Transaksi Berdasarkan Waktu Pembelian")

    period_day_count = df_clean[period_col].value_counts().reset_index()
    period_day_count.columns = ["Waktu Pembelian", "Jumlah Transaksi"]

    fig_period = px.bar(
        period_day_count,
        x="Waktu Pembelian",
        y="Jumlah Transaksi",
        title="Distribusi Transaksi Berdasarkan Waktu Pembelian"
    )

    st.plotly_chart(fig_period, use_container_width=True)


if weekday_col is not None:
    st.subheader("Perbandingan Transaksi Weekday dan Weekend")

    weekday_count = df_clean[weekday_col].value_counts().reset_index()
    weekday_count.columns = ["Jenis Hari", "Jumlah Transaksi"]

    fig_weekday = px.pie(
        weekday_count,
        names="Jenis Hari",
        values="Jumlah Transaksi",
        title="Perbandingan Transaksi Weekday dan Weekend"
    )

    st.plotly_chart(fig_weekday, use_container_width=True)


if date_col is not None:
    st.subheader("Jumlah Transaksi per Hari")

    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
    df_clean["tanggal"] = df_clean[date_col].dt.date

    transaksi_harian = df_clean.groupby("tanggal")[transaction_col].nunique().reset_index()
    transaksi_harian.columns = ["Tanggal", "Jumlah Transaksi"]

    fig_daily = px.line(
        transaksi_harian,
        x="Tanggal",
        y="Jumlah Transaksi",
        title="Jumlah Transaksi per Hari"
    )

    st.plotly_chart(fig_daily, use_container_width=True)


# =========================================================
# KESIMPULAN
# =========================================================
st.subheader("Kesimpulan")

if len(rules_rekomendasi) > 0:
    best_rule = rules_rekomendasi.iloc[0]

    st.markdown(
        f"""
        Berdasarkan hasil Market Basket Analysis menggunakan algoritma Apriori,
        diperoleh **{jumlah_frequent_itemsets} frequent itemsets** dan
        **{len(rules_filtered)} association rules relevan**. Dari rules tersebut,
        dashboard menampilkan **Top {TOP_REKOMENDASI} rekomendasi product bundling**.

        Rule terbaik adalah **{best_rule['rule']}** dengan nilai:
        - Support: **{best_rule['support'] * 100:.2f}%**
        - Confidence: **{best_rule['confidence'] * 100:.2f}%**
        - Lift Ratio: **{best_rule['lift']:.2f}**

        Nilai lift ratio lebih dari 1 menunjukkan bahwa produk pada rule tersebut
        memiliki hubungan asosiasi positif, sehingga dapat dijadikan dasar
        rekomendasi bundling atau promosi silang.
        """
    )

else:
    st.markdown(
        """
        Berdasarkan parameter yang digunakan, belum terdapat association rules
        yang memenuhi kriteria lift > 1 dan minimum confidence 50%.
        """
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        By: Kelompok 12 IS-07-03
    </div>
    """,
    unsafe_allow_html=True
)
