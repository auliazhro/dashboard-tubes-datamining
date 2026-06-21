import streamlit as st
import pandas as pd
import plotly.express as px
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
# STYLE ADAPTIVE: BACKGROUND IKUT SISTEM, ISI LEBIH TERLIHAT
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

    .stDataFrame {
        margin-bottom: 6px !important;
    }

    /* LIGHT MODE */
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

    /* DARK MODE */
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
# HELPER STYLE UNTUK GRAFIK
# Background grafik mengikuti tema Streamlit
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
        color=CHART_TEXT,
        title_font=dict(color=CHART_TEXT),
        tickfont=dict(color=CHART_TEXT)
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        color=CHART_TEXT,
        title_font=dict(color=CHART_TEXT),
        tickfont=dict(color=CHART_TEXT)
    )

    return fig

# ==========================================================
# LOAD DATA & DATA CLEANING (PERBAIKAN PEMBACAAN CSV MULTI-KOLOM)
# ==========================================================

@st.cache_data
def load_and_clean_data():
    # Menghubungkan langsung dengan nama berkas CSV di repository kamu
    filename = "bread_basket_per_transaksi.xlsx - Sheet1.csv"
    
    # Membaca data dengan pembatasan ketat agar koma di dalam teks item tidak merusak baris
    try:
        data = pd.read_csv(filename, on_bad_lines='skip')
    except FileNotFoundError:
        data = pd.read_csv("bread_basket_per_transaksi.csv", on_bad_lines='skip')
        
    # Standardisasi nama kolom menjadi huruf kecil
    data.columns = [str(col).strip().lower() for col in data.columns]
    
    # Jika data terlanjur melebar ke samping, kita ambil 2 kolom utama secara paksa
    col_transaction = [c for c in data.columns if "transaction" in c or "id" in c][0]
    col_item = [c for c in data.columns if "item" in c or "items" in c or "produk" in c][0]
    
    df_clean = data[[col_transaction, col_item]].copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna()
    
    df_clean[col_item] = df_clean[col_item].astype(str).str.strip()
    
    # Kembalikan rujukan nama variabel standar
    df_clean["Transaction"] = df_clean[col_transaction] 
    df_clean["Items"] = df_clean[col_item]
    
    # Deteksi opsional untuk kolom periode waktu (jika ada di file csv asli)
    for ext_col in ["period_day", "weekday_weekend"]:
        if ext_col in data.columns:
            df_clean[ext_col] = data[ext_col].astype(str).str.strip().str.lower()
        else:
            # Skenario pengaman jika kolom waktu hilang akibat pergeseran CSV
            df_clean[ext_col] = "not specified"
        
    return df_clean

# ==========================================================
# ONE-HOT ENCODING (PERBAIKAN NAMEERROR TRANSACTIONS)
# ==========================================================

# PASTIKAN BARIS INI ADA UNTUK MENDEFINISIKAN VARIABEL 'transactions'
transactions = df_clean["Items"].apply(
    lambda x: [item.strip() for item in str(x).split(",") if item.strip() != ""]
).tolist()

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket_sets = pd.DataFrame(te_array, columns=te.columns_).astype(bool)

# ==========================================================
# APRIORI (Diturunkan ke 0.02 agar menangkap aturan Toast -> Coffee)
# ==========================================================

frequent_itemsets = apriori(
    basket_sets,
    min_support=0.02,
    use_colnames=True
)

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

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
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules["rule"] = rules["antecedents_str"] + " → " + rules["consequents_str"]
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)

# ==========================================================
# DATA VISUALISASI PREPARATION
# ==========================================================

all_items = [item for transaksi in transactions for item in transaksi]
item_counts = pd.Series(all_items).value_counts()

top_items_df = item_counts.head(10).reset_index()
top_items_df.columns = ["Produk", "Jumlah Pembelian"]

top_pairs = frequent_itemsets[frequent_itemsets["length"] == 2].copy()
top_pairs["Kombinasi Produk"] = top_pairs["itemsets"].apply(lambda x: " + ".join(list(x)))
top_pairs = top_pairs.sort_values(by="support", ascending=False).head(10)

top_pairs_display = top_pairs[["Kombinasi Produk", "support"]].copy()
top_pairs_display["Support (%)"] = top_pairs_display["support"] * 100
top_pairs_display = top_pairs_display[["Kombinasi Produk", "Support (%)"]]

frequent_display = frequent_itemsets.head(10).copy()
frequent_display["Itemsets"] = frequent_display["itemsets"].apply(lambda x: ", ".join(list(x)))
frequent_display["Support (%)"] = frequent_display["support"] * 100
frequent_display = frequent_display[["Itemsets", "Support (%)", "length"]]
frequent_display = frequent_display.rename(columns={"length": "Jumlah Item"})

# ==========================================================
# JENDELA UTAMA DASHBOARD
# ==========================================================

st.title("Dashboard Market Basket Analysis")

st.markdown(
    """
    <div class="hero-card">
        <h2>Strategi Bundling Produk Bakery</h2>
        Menampilkan pola pembelian pelanggan untuk menentukan produk yang cocok dipromosikan bersama.
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# RINGKASAN METRIK
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
            Support: <b>{best_rule['support']*100:.2f}%</b> &nbsp; | &nbsp;
            Confidence: <b>{best_rule['confidence']*100:.2f}%</b> &nbsp; | &nbsp;
            Lift Ratio: <b>{best_rule['lift']:.2f}x</b><br>
            Sekitar <b>{best_rule['confidence']*100:.2f}%</b> pelanggan yang membeli
            <b>{best_rule['antecedents_str']}</b> juga membeli <b>{best_rule['consequents_str']}</b>.
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
        tampil_rules = tampil_rules[["Rule", "Support (%)", "Confidence (%)", "Lift Ratio"]]

        tampil_rules_format = tampil_rules.copy()
        tampil_rules_format["Support (%)"] = tampil_rules_format["Support (%)"].map(lambda x: f"{x:.2f}%")
        tampil_rules_format["Confidence (%)"] = tampil_rules_format["Confidence (%)"].map(lambda x: f"{x:.2f}%")
        tampil_rules_format["Lift Ratio"] = tampil_rules_format["Lift Ratio"].map(lambda x: f"{x:.2f}x")
        tampil_rules_format = tampil_rules_format.reset_index(drop=True)

        st.dataframe(tampil_rules_format, use_container_width=True, height=300, hide_index=True)

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
        for i, row in rules.head(4).iterrows():
            st.markdown(
                f"""
                <div class="recommend-card">
                    <b>{row['rule']}</b><br>
                    Confidence: <b>{row['confidence']*100:.2f}%</b> |
                    Lift Ratio: <b>{row['lift']:.2f}x</b><br>
                    <b>Saran:</b> jadikan paket bundling atau promosi silang.
                </div>
                """,
                unsafe_allow_html=True
            )

# ==========================================================
# VISUALISASI BAR CHARTS & SCATTER PLOT
# ==========================================================

st.subheader("Visualisasi Pola Pembelian")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown('<div class="caption-box">Top 10 Produk Paling Sering Dibeli</div>', unsafe_allow_html=True)
    fig_top_items = px.bar(
        top_items_df.sort_values("Jumlah Pembelian", ascending=True),
        x="Jumlah Pembelian", y="Produk", orientation="h", text="Jumlah Pembelian",
        color_discrete_sequence=["#3B82F6"]
    )
    fig_top_items.update_traces(textposition="outside")
    fig_top_items = apply_chart_style(fig_top_items, 420)
    st.plotly_chart(fig_top_items, use_container_width=True)

with col_chart2:
    st.markdown('<div class="caption-box">Top Kombinasi Produk Berdasarkan Support</div>', unsafe_allow_html=True)
    fig_pairs = px.bar(
        top_pairs_display.sort_values("Support (%)", ascending=True),
        x="Support (%)", y="Kombinasi Produk", orientation="h", text="Support (%)",
        color_discrete_sequence=["#10B981"]
    )
    fig_pairs.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_pairs = apply_chart_style(fig_pairs, 420)
    st.plotly_chart(fig_pairs, use_container_width=True)

col_chart3, col_chart4 = st.columns(2)

with col_chart3:
    st.markdown('<div class="caption-box">Top Frequent Itemsets</div>', unsafe_allow_html=True)
    fig_freq = px.bar(
        frequent_display.sort_values("Support (%)", ascending=True),
        x="Support (%)", y="Itemsets", orientation="h", text="Support (%)",
        color_discrete_sequence=["#A855F7"]
    )
    fig_freq.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_freq = apply_chart_style(fig_freq, 420)
    st.plotly_chart(fig_freq, use_container_width=True)

with col_chart4:
    st.markdown('<div class="caption-box">Perbandingan Confidence dan Lift Ratio</div>', unsafe_allow_html=True)
    if len(rules) > 0:
        scatter_rules = rules.head(10).copy()
        scatter_rules["Confidence (%)"] = scatter_rules["confidence"] * 100
        scatter_rules["Support (%)"] = scatter_rules["support"] * 100

        fig_scatter = px.scatter(
            scatter_rules, x="Confidence (%)", y="lift", size="Support (%)",
            hover_name="rule", text="rule", color_discrete_sequence=["#EF4444"]
        )
        fig_scatter.update_traces(textposition="top center", textfont_color=CHART_TEXT)
        fig_scatter = apply_chart_style(fig_scatter, 420)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================================
# VISUALISASI PERIODE WAKTU TRANSAKSI
# ==========================================================

col_chart5, col_chart6 = st.columns(2)

with col_chart5:
    if "period_day" in df_clean.columns:
        st.markdown('<div class="caption-box">Transaksi Berdasarkan Periode Hari</div>', unsafe_allow_html=True)
        period_counts = df_clean["period_day"].value_counts().reset_index()
        period_counts.columns = ["Periode Hari", "Jumlah Transaksi"]

        fig_period = px.bar(period_counts, x="Periode Hari", y="Jumlah Transaksi", text="Jumlah Transaksi", color_discrete_sequence=["#64748B"])
        fig_period.update_traces(textposition="outside")
        fig_period = apply_chart_style(fig_period, 360)
        st.plotly_chart(fig_period, use_container_width=True)

with col_chart6:
    if "weekday_weekend" in df_clean.columns:
        st.markdown('<div class="caption-box">Transaksi Weekday vs Weekend</div>', unsafe_allow_html=True)
        weekday_counts = df_clean["weekday_weekend"].value_counts().reset_index()
        weekday_counts.columns = ["Kategori Hari", "Jumlah Transaksi"]

        fig_weekday = px.bar(weekday_counts, x="Kategori Hari", y="Jumlah Transaksi", text="Jumlah Transaksi", color_discrete_sequence=["#F59E0B"])
        fig_weekday.update_traces(textposition="outside")
        fig_weekday = apply_chart_style(fig_weekday, 360)
        st.plotly_chart(fig_weekday, use_container_width=True)

# ==========================================================
# KESIMPULAN FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer-card">
    <b>Kesimpulan:</b> Produk dengan pembelian tinggi dan hubungan asosiasi kuat dapat digunakan sebagai dasar strategi bundling.
    Prioritaskan kombinasi dengan confidence tinggi dan lift ratio di atas 1.
    </div>
    """,
    unsafe_allow_html=True
)
