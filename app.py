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
        padding: 9px 12px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
        font-size: 14px;
        box-shadow: 0px 1px 5px rgba(0,0,0,0.08);
    }

    .finding-card {
        padding: 15px 16px;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        margin-bottom: 10px;
        box-shadow: 0px 1px 6px rgba(0,0,0,0.10);
    }

    .finding-title {
        color: #3B82F6;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .recommend-card {
        padding: 12px 14px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin-bottom: 8px;
        font-size: 14px;
        box-shadow: 0px 1px 5px rgba(0,0,0,0.08);
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

    [data-testid="stMetricLabel"] {
        font-weight: 700;
    }

    .footer-card {
        background-color: #1E3A8A;
        color: white;
        padding: 16px 18px;
        border-radius: 12px;
        margin-top: 12px;
        font-size: 14px;
    }

    @media (prefers-color-scheme: light) {
        .caption-box {
            background-color: #F8FAFC;
            color: #1F2937;
        }

        .finding-card {
            background-color: #F8FAFC;
            color: #111827;
            border: 1px solid #E5E7EB;
            border-left: 6px solid #3B82F6;
        }

        .recommend-card {
            background-color: #F8FAFC;
            color: #111827;
        }

        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
        }
    }

    @media (prefers-color-scheme: dark) {
        .caption-box {
            background-color: #111827;
            color: #F9FAFB;
        }

        .finding-card {
            background-color: #111827;
            color: #F9FAFB;
            border: 1px solid #374151;
            border-left: 6px solid #60A5FA;
        }

        .finding-title {
            color: #60A5FA;
        }

        .recommend-card {
            background-color: #111827;
            color: #F9FAFB;
        }

        div[data-testid="stMetric"] {
            background-color: #111827;
        }

        [data-testid="stMetricValue"] {
            color: #60A5FA;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# CHART STYLE
# ==========================================================

base_theme = st.get_option("theme.base")

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

    fig.update_xaxes(
        showgrid=True,
        gridcolor=CHART_GRID,
        zeroline=False,
        color=CHART_TEXT
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        color=CHART_TEXT
    )

    return fig


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    candidate_files = [
        "bread_basket_per_transaksi.xlsx",
    ]

    selected_file = None

    for file in candidate_files:
        if Path(file).exists():
            selected_file = file
            break

    if selected_file is None:
        st.error(
            "File dataset tidak ditemukan. Pastikan file dataset berada satu folder dengan app.py."
        )
        st.stop()

    if selected_file.endswith(".xlsx"):
        data = pd.read_excel(selected_file)
    else:
        data = pd.read_csv(selected_file)

    return data


df = load_data()

# ==========================================================
# DATA CLEANING
# ==========================================================

df_clean = df.copy()

df_clean.columns = [str(col).strip().lower() for col in df_clean.columns]

df_clean = df_clean.drop_duplicates()

# Deteksi kolom transaksi
if "transaction" in df_clean.columns:
    col_transaction = "transaction"
else:
    transaction_candidates = [
        col for col in df_clean.columns
        if "transaction" in col or "transaksi" in col or col == "id"
    ]

    if len(transaction_candidates) == 0:
        st.error("Kolom transaksi tidak ditemukan.")
        st.stop()

    col_transaction = transaction_candidates[0]

# Deteksi kolom items
if "items" in df_clean.columns:
    col_items = "items"
elif "item" in df_clean.columns:
    col_items = "item"
else:
    item_candidates = [
        col for col in df_clean.columns
        if ("item" in col or "produk" in col)
        and "jumlah" not in col
        and "count" not in col
    ]

    if len(item_candidates) == 0:
        st.error("Kolom item atau items tidak ditemukan.")
        st.stop()

    col_items = item_candidates[0]

df_clean = df_clean.dropna(subset=[col_transaction, col_items])

df_clean["Transaction"] = df_clean[col_transaction].astype(str).str.strip()
df_clean["Items"] = df_clean[col_items].astype(str).str.strip()

# Kolom waktu jika tersedia
if "period_day" in df_clean.columns:
    df_clean["period_day"] = df_clean["period_day"].astype(str).str.strip().str.lower()

if "weekday_weekend" in df_clean.columns:
    df_clean["weekday_weekend"] = df_clean["weekday_weekend"].astype(str).str.strip().str.lower()

# Hitung jumlah item
df_clean["Jumlah_Item"] = df_clean["Items"].apply(
    lambda x: len([
        item.strip()
        for item in str(x).split(",")
        if item.strip() != ""
    ])
)

# Membuat list transaksi
transactions = df_clean["Items"].apply(
    lambda x: [
        item.strip()
        for item in str(x).split(",")
        if item.strip() != ""
    ]
).tolist()

# Validasi transaksi
transactions = [trx for trx in transactions if len(trx) > 0]

if len(transactions) == 0:
    st.error("Tidak ada transaksi yang valid untuk dianalisis.")
    st.stop()

# ==========================================================
# ONE-HOT ENCODING
# ==========================================================

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket_sets = pd.DataFrame(te_array, columns=te.columns_).astype(bool)

# ==========================================================
# APRIORI
# ==========================================================

frequent_itemsets = apriori(
    basket_sets,
    min_support=0.02,
    use_colnames=True
)

if frequent_itemsets.empty:
    st.error("Frequent itemsets tidak ditemukan. Coba turunkan nilai minimum support.")
    st.stop()

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

frequent_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
)

# ==========================================================
# ASSOCIATION RULES
# ==========================================================

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
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )

    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(list(x))
    )

    rules["rule"] = (
        rules["antecedents_str"] +
        " → " +
        rules["consequents_str"]
    )

    rules = rules.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )

# ==========================================================
# DATA VISUALISASI
# ==========================================================

all_items = [item for transaksi in transactions for item in transaksi]
item_counts = pd.Series(all_items).value_counts()

top_items_df = item_counts.head(10).reset_index()
top_items_df.columns = ["Produk", "Jumlah Pembelian"]

top_pairs = frequent_itemsets[frequent_itemsets["length"] == 2].copy()

if not top_pairs.empty:
    top_pairs["Kombinasi Produk"] = top_pairs["itemsets"].apply(
        lambda x: " + ".join(list(x))
    )

    top_pairs = top_pairs.sort_values(
        by="support",
        ascending=False
    ).head(10)

    top_pairs_display = top_pairs[["Kombinasi Produk", "support"]].copy()
    top_pairs_display["Support (%)"] = top_pairs_display["support"] * 100
else:
    top_pairs_display = pd.DataFrame(
        columns=["Kombinasi Produk", "Support (%)"]
    )

frequent_display = frequent_itemsets.head(10).copy()
frequent_display["Itemsets"] = frequent_display["itemsets"].apply(
    lambda x: ", ".join(list(x))
)
frequent_display["Support (%)"] = frequent_display["support"] * 100

# ==========================================================
# VIEW DASHBOARD
# ==========================================================

st.title("Dashboard Market Basket Analysis")

st.markdown(
    """
    <div class="hero-card">
        <h2>Strategi Bundling Produk Bakery</h2>
        Menampilkan pola pembelian pelanggan untuk menentukan produk promosi bersama.
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
# TEMUAN UTAMA
# ==========================================================

st.subheader("Temuan Utama")

if len(rules) > 0:
    best_rule = rules.iloc[0]

    st.markdown(
        f"""
        <div class="finding-card">
            <div class="finding-title">{best_rule['rule']}</div>
            Support: <b>{best_rule['support']*100:.2f}%</b> |
            Confidence: <b>{best_rule['confidence']*100:.2f}%</b> |
            Lift Ratio: <b>{best_rule['lift']:.2f}x</b><br>
            Sekitar <b>{best_rule['confidence']*100:.2f}%</b> pelanggan yang membeli
            <b>{best_rule['antecedents_str']}</b> juga membeli
            <b>{best_rule['consequents_str']}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Belum ditemukan rule asosiasi yang memenuhi kriteria.")

# ==========================================================
# TABEL RULES DAN REKOMENDASI
# ==========================================================

left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.subheader("Top Association Rules")

    st.markdown(
        """
        <div class="caption-box">
        Rule A → B berarti pelanggan yang membeli A cenderung membeli B.
        Support dan confidence ditampilkan dalam persen.
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(rules) > 0:
        tampil_rules = rules[["rule", "support", "confidence", "lift"]].head(8).copy()

        tampil_rules["Support (%)"] = tampil_rules["support"] * 100
        tampil_rules["Confidence (%)"] = tampil_rules["confidence"] * 100
        tampil_rules["Lift Ratio"] = tampil_rules["lift"]

        tampil_rules = tampil_rules.rename(columns={"rule": "Rule"})

        tampil_rules = tampil_rules[
            ["Rule", "Support (%)", "Confidence (%)", "Lift Ratio"]
        ]

        tampil_rules["Support (%)"] = tampil_rules["Support (%)"].map(lambda x: f"{x:.2f}%")
        tampil_rules["Confidence (%)"] = tampil_rules["Confidence (%)"].map(lambda x: f"{x:.2f}%")
        tampil_rules["Lift Ratio"] = tampil_rules["Lift Ratio"].map(lambda x: f"{x:.2f}x")

        tampil_rules = tampil_rules.reset_index(drop=True)

        st.dataframe(
            tampil_rules,
            use_container_width=True,
            height=280,
            hide_index=True
        )
    else:
        st.info("Tidak ada association rules yang memenuhi kriteria.")

with right_col:
    st.subheader("Rekomendasi Bundling")

    st.markdown(
        """
        <div class="caption-box">
        Produk berikut dapat diprioritaskan sebagai paket promosi.
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(rules) > 0:
        for _, row in rules.head(3).iterrows():
            st.markdown(
                f"""
                <div class="recommend-card">
                    <b>{row['rule']}</b><br>
                    Confidence: <b>{row['confidence']*100:.2f}%</b> |
                    Lift Ratio: <b>{row['lift']:.2f}x</b><br>
                    <b>Saran:</b> jadikan paket promo atau promosi silang.
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Belum ada rekomendasi bundling.")

# ==========================================================
# VISUALISASI 1 DAN 2
# ==========================================================

st.subheader("Visualisasi Pola Pembelian")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown(
        """
        <div class="caption-box">
        Produk paling sering dibeli. X = jumlah pembelian, Y = nama produk.
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
        title="Top 10 Produk Paling Sering Dibeli",
        color_discrete_sequence=["#3B82F6"]
    )

    fig_top_items.update_traces(
        textposition="outside",
        textfont_color=CHART_TEXT
    )

    fig_top_items = apply_chart_style(fig_top_items, 340)

    st.plotly_chart(fig_top_items, use_container_width=True)

with col_chart2:
    st.markdown(
        """
        <div class="caption-box">
        Kombinasi produk yang sering muncul bersama. X = support (%), Y = kombinasi produk.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not top_pairs_display.empty:
        fig_pairs = px.bar(
            top_pairs_display.sort_values("Support (%)", ascending=True),
            x="Support (%)",
            y="Kombinasi Produk",
            orientation="h",
            text="Support (%)",
            title="Top Kombinasi Produk Berdasarkan Support",
            color_discrete_sequence=["#10B981"]
        )

        fig_pairs.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
            textfont_color=CHART_TEXT
        )

        fig_pairs = apply_chart_style(fig_pairs, 340)

        st.plotly_chart(fig_pairs, use_container_width=True)
    else:
        st.info("Kombinasi produk belum ditemukan.")

# ==========================================================
# VISUALISASI 3 DAN 4
# ==========================================================

col_chart3, col_chart4 = st.columns(2)

with col_chart3:
    st.markdown(
        """
        <div class="caption-box">
        Frequent itemsets dengan support tertinggi. X = support (%), Y = itemset.
        </div>
        """,
        unsafe_allow_html=True
    )

    fig_freq = px.bar(
        frequent_display.sort_values("Support (%)", ascending=True),
        x="Support (%)",
        y="Itemsets",
        orientation="h",
        text="Support (%)",
        title="Top Frequent Itemsets",
        color_discrete_sequence=["#A855F7"]
    )

    fig_freq.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        textfont_color=CHART_TEXT
    )

    fig_freq = apply_chart_style(fig_freq, 340)

    st.plotly_chart(fig_freq, use_container_width=True)

with col_chart4:
    st.markdown(
        """
        <div class="caption-box">
        Perbandingan confidence dan lift ratio. X = confidence (%), Y = lift ratio.
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(rules) > 0:
        scatter_rules = rules.head(10).copy()
        scatter_rules["Confidence (%)"] = scatter_rules["confidence"] * 100
        scatter_rules["Support (%)"] = scatter_rules["support"] * 100

        fig_scatter = px.scatter(
            scatter_rules,
            x="Confidence (%)",
            y="lift",
            size="Support (%)",
            hover_name="rule",
            text="rule",
            title="Confidence vs Lift Ratio",
            color_discrete_sequence=["#EF4444"]
        )

        fig_scatter.update_traces(
            textposition="top center",
            textfont_color=CHART_TEXT,
            marker=dict(opacity=0.85, line=dict(width=1, color=CHART_TEXT))
        )

        fig_scatter = apply_chart_style(fig_scatter, 340)

        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Data rule belum tersedia untuk scatter plot.")

# ==========================================================
# VISUALISASI WAKTU
# ==========================================================

col_chart5, col_chart6 = st.columns(2)

with col_chart5:
    if "period_day" in df_clean.columns:
        st.markdown(
            """
            <div class="caption-box">
            Distribusi transaksi berdasarkan periode hari. X = periode hari, Y = jumlah transaksi.
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

        fig_period.update_traces(
            textposition="outside",
            textfont_color=CHART_TEXT
        )

        fig_period = apply_chart_style(fig_period, 300)

        st.plotly_chart(fig_period, use_container_width=True)

with col_chart6:
    if "weekday_weekend" in df_clean.columns:
        st.markdown(
            """
            <div class="caption-box">
            Distribusi transaksi weekday dan weekend. X = kategori hari, Y = jumlah transaksi.
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

        fig_weekday.update_traces(
            textposition="outside",
            textfont_color=CHART_TEXT
        )

        fig_weekday = apply_chart_style(fig_weekday, 300)

        st.plotly_chart(fig_weekday, use_container_width=True)

# ==========================================================
# TABEL TAMBAHAN
# ==========================================================

st.subheader("Ringkasan Tabel Analisis")

col_table1, col_table2 = st.columns(2)

with col_table1:
    st.markdown(
        """
        <div class="caption-box">
        Top frequent itemsets berdasarkan support.
        </div>
        """,
        unsafe_allow_html=True
    )

    frequent_table = frequent_display.copy()
    frequent_table = frequent_table[["Itemsets", "Support (%)", "length"]]
    frequent_table = frequent_table.rename(columns={"length": "Jumlah Item"})
    frequent_table["Support (%)"] = frequent_table["Support (%)"].map(lambda x: f"{x:.2f}%")
    frequent_table = frequent_table.reset_index(drop=True)

    st.dataframe(
        frequent_table,
        use_container_width=True,
        height=280,
        hide_index=True
    )

with col_table2:
    st.markdown(
        """
        <div class="caption-box">
        Kombinasi produk terbaik untuk kandidat bundling.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not top_pairs_display.empty:
        pair_table = top_pairs_display.copy()
        pair_table = pair_table[["Kombinasi Produk", "Support (%)"]]
        pair_table["Support (%)"] = pair_table["Support (%)"].map(lambda x: f"{x:.2f}%")
        pair_table = pair_table.reset_index(drop=True)

        st.dataframe(
            pair_table,
            use_container_width=True,
            height=280,
            hide_index=True
        )
    else:
        st.info("Belum ada kombinasi produk yang memenuhi nilai support.")

# ==========================================================
# KESIMPULAN
# ==========================================================

st.markdown(
    """
    <div class="footer-card">
    <b>Kesimpulan:</b> Kombinasi produk dengan nilai Lift Ratio > 1 menunjukkan hubungan asosiasi positif.
    Hasil ini dapat digunakan sebagai dasar program promosi, strategi bundling, atau penataan produk.
    </div>
    """,
    unsafe_allow_html=True
)
