import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Dashboard Market Basket Analysis",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# STYLE
# ==========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1 {
        font-size: 34px !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
    }

    h2, h3 {
        font-weight: 700 !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.4rem !important;
    }

    .hero-card {
        background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 18px 22px;
        border-radius: 12px;
        margin-bottom: 14px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.12);
    }

    .hero-card h2 {
        color: white !important;
        margin: 0 !important;
        font-size: 22px !important;
    }

    .caption-box {
        padding: 10px 12px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
        font-size: 14px;
        box-shadow: 0px 1px 5px rgba(0,0,0,0.08);
    }

    .recommend-card {
        padding: 14px 16px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin-bottom: 10px;
        font-size: 14px;
        box-shadow: 0px 1px 5px rgba(0,0,0,0.08);
    }

    .footer-card {
        background-color: #1E3A8A;
        color: white;
        padding: 16px 18px;
        border-radius: 12px;
        margin-top: 12px;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        padding: 13px;
        border-radius: 10px;
        border: 1px solid rgba(148,163,184,0.45);
        box-shadow: 0px 1px 5px rgba(0,0,0,0.08);
    }

    [data-testid="stMetricValue"] {
        color: #3B82F6;
        font-size: 26px;
        font-weight: 800;
    }

    @media (prefers-color-scheme: light) {
        .caption-box, .recommend-card {
            background-color: #F8FAFC;
            color: #1F2937;
        }

        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
        }
    }

    @media (prefers-color-scheme: dark) {
        .caption-box, .recommend-card {
            background-color: #111827;
            color: #F9FAFB;
        }

        div[data-testid="stMetric"] {
            background-color: #111827;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# CHART STYLE
# ==========================================================

base_theme = st.get_option("theme.base") or "light"

if base_theme == "dark":
    CHART_TEMPLATE = "plotly_dark"
    CHART_BG = "#111827"
    CHART_TEXT = "#F9FAFB"
    CHART_GRID = "#374151"
else:
    CHART_TEMPLATE = "plotly_white"
    CHART_BG = "#FFFFFF"
    CHART_TEXT = "#111827"
    CHART_GRID = "#E5E7EB"


def apply_chart_style(fig, height):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(size=12, color=CHART_TEXT),
        title_font=dict(size=17, color=CHART_TEXT),
        margin=dict(l=5, r=20, t=45, b=5)
    )
    fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, color=CHART_TEXT)
    fig.update_yaxes(showgrid=False, zeroline=False, color=CHART_TEXT)
    return fig


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    candidate_files = [
        "bread_basket_per_transaksi.xlsx",
        "bread_basket_per_transaksi.csv",
        "bread_basket_per_transaksi.xlsx - Sheet1.csv"
    ]

    selected_file = None

    for file in candidate_files:
        if Path(file).exists():
            selected_file = file
            break

    if selected_file is None:
        st.error("File dataset tidak ditemukan. Pastikan dataset ada satu folder dengan app.py.")
        st.stop()

    if selected_file.endswith(".xlsx"):
        return pd.read_excel(selected_file)
    else:
        return pd.read_csv(selected_file)


df = load_data()

# ==========================================================
# DATA CLEANING
# ==========================================================

df_clean = df.copy()
df_clean.columns = [str(col).strip().lower() for col in df_clean.columns]
df_clean = df_clean.drop_duplicates()

if "transaction" in df_clean.columns:
    col_transaction = "transaction"
else:
    col_transaction = [c for c in df_clean.columns if "transaction" in c or "transaksi" in c or c == "id"][0]

if "items" in df_clean.columns:
    col_items = "items"
elif "item" in df_clean.columns:
    col_items = "item"
else:
    col_items = [c for c in df_clean.columns if "item" in c or "produk" in c][0]

df_clean = df_clean.dropna(subset=[col_transaction, col_items])

df_clean["Transaction"] = df_clean[col_transaction].astype(str).str.strip()
df_clean["Items"] = df_clean[col_items].astype(str).str.strip()

if "period_day" in df_clean.columns:
    df_clean["period_day"] = df_clean["period_day"].astype(str).str.strip().str.lower()

if "weekday_weekend" in df_clean.columns:
    df_clean["weekday_weekend"] = df_clean["weekday_weekend"].astype(str).str.strip().str.lower()

df_clean["Jumlah_Item"] = df_clean["Items"].apply(
    lambda x: len([item.strip() for item in str(x).split(",") if item.strip() != ""])
)

transactions = df_clean["Items"].apply(
    lambda x: [item.strip() for item in str(x).split(",") if item.strip() != ""]
).tolist()

# ==========================================================
# APRIORI
# ==========================================================

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket_sets = pd.DataFrame(te_array, columns=te.columns_).astype(bool)

frequent_itemsets = apriori(
    basket_sets,
    min_support=0.02,
    use_colnames=True
)

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3
)

rules = rules[
    (rules["lift"] > 1) &
    (rules["confidence"] >= 0.3)
].copy()

if len(rules) > 0:
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules["rule"] = rules["antecedents_str"] + " → " + rules["consequents_str"]
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)

# ==========================================================
# DATA VISUALISASI
# ==========================================================

all_items = [item for trx in transactions for item in trx]
item_counts = pd.Series(all_items).value_counts()

top_items_df = item_counts.head(10).reset_index()
top_items_df.columns = ["Produk", "Jumlah Pembelian"]

top_pairs = frequent_itemsets[frequent_itemsets["length"] == 2].copy()
top_pairs["Kombinasi Produk"] = top_pairs["itemsets"].apply(lambda x: " + ".join(list(x)))
top_pairs = top_pairs.sort_values(by="support", ascending=False).head(10)
top_pairs["Support (%)"] = top_pairs["support"] * 100

# ==========================================================
# DASHBOARD
# ==========================================================

st.title("Dashboard Market Basket Analysis")

st.markdown(
    """
    <div class="hero-card">
        <h2>Strategi Bundling Produk Bakery</h2>
        Dashboard ini membantu melihat produk yang paling sering dibeli, kombinasi produk yang cocok dijadikan bundling,
        dan waktu transaksi paling ramai.
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# RINGKASAN
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Jumlah Transaksi", df_clean["Transaction"].nunique())
col2.metric("Jumlah Item Unik", basket_sets.shape[1])
col3.metric("Frequent Itemsets", len(frequent_itemsets))
col4.metric("Association Rules", len(rules))

# ==========================================================
# PRODUK PALING BANYAK DIBELI
# ==========================================================

st.header("Top 10 Produk Paling Banyak Dibeli")

st.markdown(
    """
    <div class="caption-box">
    Grafik ini menunjukkan produk yang paling sering dibeli pelanggan. 
    Produk dengan pembelian tertinggi dapat dijadikan produk utama dalam strategi promosi.
    </div>
    """,
    unsafe_allow_html=True
)

fig_top_items = px.bar(
    top_items_df.sort_values("Jumlah Pembelian", ascending=True),
    x="Jumlah Pembelian",
    y="Produk",
    orientation="h",
    text="Jumlah Pembelian",
    title="Top 10 Produk Paling Banyak Dibeli",
    color_discrete_sequence=["#3B82F6"]
)

fig_top_items.update_traces(textposition="outside", textfont_color=CHART_TEXT)
fig_top_items = apply_chart_style(fig_top_items, 420)
st.plotly_chart(fig_top_items, use_container_width=True)

# ==========================================================
# REKOMENDASI BUNDLING
# ==========================================================

st.header("Rekomendasi Kombinasi Produk untuk Bundling")

st.markdown(
    """
    <div class="caption-box">
    Rekomendasi ini berasal dari pola transaksi pelanggan. Produk yang memiliki hubungan kuat dapat dipromosikan bersama dalam satu paket.
    </div>
    """,
    unsafe_allow_html=True
)

if len(rules) > 0:
    for _, row in rules.head(5).iterrows():
        st.markdown(
            f"""
            <div class="recommend-card">
                <b>{row['rule']}</b><br>
                Dari pelanggan yang membeli <b>{row['antecedents_str']}</b>, 
                sekitar <b>{row['confidence']*100:.2f}%</b> juga membeli <b>{row['consequents_str']}</b>.<br>
                <b>Saran:</b> Buat paket promo <b>{row['antecedents_str']} + {row['consequents_str']}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Belum ada rekomendasi bundling yang memenuhi kriteria.")

# ==========================================================
# DETAIL NILAI REKOMENDASI
# ==========================================================

st.header("Detail Nilai Rekomendasi")

st.markdown(
    """
    <div class="caption-box">
    Support menunjukkan seberapa sering kombinasi produk muncul. 
    Confidence menunjukkan peluang pelanggan membeli produk kedua setelah membeli produk pertama. 
    Lift Ratio menunjukkan kekuatan hubungan antarproduk.
    </div>
    """,
    unsafe_allow_html=True
)

if len(rules) > 0:
    rules_table = rules[["rule", "support", "confidence", "lift"]].head(5).copy()
    rules_table["Support (%)"] = rules_table["support"] * 100
    rules_table["Confidence (%)"] = rules_table["confidence"] * 100
    rules_table["Lift Ratio"] = rules_table["lift"]

    rules_table = rules_table.rename(columns={"rule": "Rule"})
    rules_table = rules_table[["Rule", "Support (%)", "Confidence (%)", "Lift Ratio"]]

    rules_table["Support (%)"] = rules_table["Support (%)"].map(lambda x: f"{x:.2f}%")
    rules_table["Confidence (%)"] = rules_table["Confidence (%)"].map(lambda x: f"{x:.2f}%")
    rules_table["Lift Ratio"] = rules_table["Lift Ratio"].map(lambda x: f"{x:.2f}x")

    st.dataframe(
        rules_table.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

# ==========================================================
# KOMBINASI PRODUK
# ==========================================================

st.header("Kombinasi Produk yang Sering Dibeli Bersama")

st.markdown(
    """
    <div class="caption-box">
    Grafik ini menunjukkan pasangan produk yang sering muncul dalam transaksi yang sama.
    Kombinasi dengan nilai support tinggi dapat dipertimbangkan untuk strategi bundling.
    </div>
    """,
    unsafe_allow_html=True
)

if not top_pairs.empty:
    fig_pairs = px.bar(
        top_pairs.sort_values("Support (%)", ascending=True),
        x="Support (%)",
        y="Kombinasi Produk",
        orientation="h",
        text="Support (%)",
        title="Kombinasi Produk yang Sering Dibeli Bersama",
        color_discrete_sequence=["#10B981"]
    )

    fig_pairs.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        textfont_color=CHART_TEXT
    )

    fig_pairs = apply_chart_style(fig_pairs, 420)
    st.plotly_chart(fig_pairs, use_container_width=True)

# ==========================================================
# WAKTU TRANSAKSI
# ==========================================================

st.header("Waktu Transaksi Paling Ramai")

col_time1, col_time2 = st.columns(2)

with col_time1:
    if "period_day" in df_clean.columns:
        st.markdown(
            """
            <div class="caption-box">
            Menunjukkan periode hari dengan jumlah transaksi terbanyak.
            </div>
            """,
            unsafe_allow_html=True
        )

        period_counts = df_clean["period_day"].value_counts().reset_index()
        period_counts.columns = ["Periode Hari", "Jumlah Transaksi"]

        fig_period = px.bar(
            period_counts,
            x="Periode Hari",
            y="Jumlah Transaksi",
            text="Jumlah Transaksi",
            title="Transaksi Berdasarkan Periode Hari",
            color_discrete_sequence=["#64748B"]
        )

        fig_period.update_traces(textposition="outside", textfont_color=CHART_TEXT)
        fig_period = apply_chart_style(fig_period, 330)
        st.plotly_chart(fig_period, use_container_width=True)

with col_time2:
    if "weekday_weekend" in df_clean.columns:
        st.markdown(
            """
            <div class="caption-box">
            Menunjukkan perbandingan transaksi pada hari kerja dan akhir pekan.
            </div>
            """,
            unsafe_allow_html=True
        )

        weekday_counts = df_clean["weekday_weekend"].value_counts().reset_index()
        weekday_counts.columns = ["Kategori Hari", "Jumlah Transaksi"]

        fig_weekday = px.bar(
            weekday_counts,
            x="Kategori Hari",
            y="Jumlah Transaksi",
            text="Jumlah Transaksi",
            title="Transaksi Weekday vs Weekend",
            color_discrete_sequence=["#F59E0B"]
        )

        fig_weekday.update_traces(textposition="outside", textfont_color=CHART_TEXT)
        fig_weekday = apply_chart_style(fig_weekday, 330)
        st.plotly_chart(fig_weekday, use_container_width=True)

# ==========================================================
# KESIMPULAN
# ==========================================================

st.markdown(
    """
    <div class="footer-card">
    <b>Kesimpulan:</b> Produk yang sering muncul bersama dapat digunakan sebagai dasar strategi promosi dan bundling.
    Produk dengan pembelian tinggi dapat dijadikan produk utama, kemudian dipasangkan dengan produk yang memiliki hubungan asosiasi kuat.
    </div>
    """,
    unsafe_allow_html=True
)
