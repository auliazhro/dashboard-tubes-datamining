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
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; }
    h1 { font-size: 34px !important; font-weight: 800 !important; margin-bottom: 0.2rem !important; }
    h2, h3 { font-weight: 700 !important; margin-top: 0.8rem !important; margin-bottom: 0.4rem !important; }
    .hero-card { background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%); color: white; padding: 18px 22px; border-radius: 12px; margin-bottom: 14px; box-shadow: 0px 2px 8px rgba(0,0,0,0.12); }
    .hero-card h2 { color: white !important; margin: 0 !important; font-size: 22px !important; }
    .caption-box { padding: 9px 12px; border-radius: 8px; border-left: 4px solid #3B82F6; margin-bottom: 10px; font-size: 14px; box-shadow: 0px 1px 5px rgba(0,0,0,0.08); }
    .finding-card { padding: 15px 16px; border-radius: 12px; border-left: 6px solid #3B82F6; margin-bottom: 10px; box-shadow: 0px 1px 6px rgba(0,0,0,0.10); }
    .finding-title { color: #3B82F6; font-size: 20px; font-weight: 800; margin-bottom: 6px; }
    .recommend-card { padding: 12px 14px; border-radius: 10px; border-left: 5px solid #10B981; margin-bottom: 8px; font-size: 14px; box-shadow: 0px 1px 5px rgba(0,0,0,0.08); }
    div[data-testid="stMetric"] { padding: 13px; border-radius: 10px; border: 1px solid rgba(148,163,184,0.45); box-shadow: 0px 1px 5px rgba(0,0,0,0.08); }
    [data-testid="stMetricValue"] { color: #3B82F6; font-size: 26px; font-weight: 800; }
    [data-testid="stMetricLabel"] { font-weight: 700; }
    .footer-card { background-color: #1E3A8A; color: white; padding: 16px 18px; border-radius: 12px; margin-top: 12px; font-size: 14px; }
    
    @media (prefers-color-scheme: light) {
        .caption-box { background-color: #F8FAFC; color: #1F2937; }
        .finding-card { background-color: #F8FAFC; color: #111827; border: 1px solid #E5E7EB; border-left: 6px solid #3B82F6; }
        .recommend-card { background-color: #F8FAFC; color: #111827; }
        div[data-testid="stMetric"] { background-color: #F8FAFC; }
    }
    @media (prefers-color-scheme: dark) {
        .caption-box { background-color: #111827; color: #F9FAFB; }
        .finding-card { background-color: #111827; color: #F9FAFB; border: 1px solid #374151; border-left: 6px solid #60A5FA; }
        .finding-title { color: #60A5FA; }
        .recommend-card { background-color: #111827; color: #F9FAFB; }
        div[data-testid="stMetric"] { background-color: #111827; }
        [data-testid="stMetricValue"] { color: #60A5FA; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

base_theme = st.get_option("theme.base")
if base_theme == "dark":
    CHART_TEMPLATE, CHART_BG, CHART_TEXT, CHART_GRID = "plotly_dark", "#111827", "#F9FAFB", "#374151"
else:
    CHART_TEMPLATE, CHART_BG, CHART_TEXT, CHART_GRID = "plotly_white", "#FFFFFF", "#111827", "#E5E7EB"

def apply_chart_style(fig, height):
    fig.update_layout(
        template=CHART_TEMPLATE, height=height, plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(size=12, color=CHART_TEXT), title_font=dict(size=17, color=CHART_TEXT), margin=dict(l=5, r=20, t=45, b=5)
    )
    fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, color=CHART_TEXT)
    fig.update_yaxes(showgrid=False, zeroline=False, color=CHART_TEXT)
    return fig

# ==========================================================
# LOAD DATA & DATA CLEANING
# ==========================================================

@st.cache_data
def load_and_clean_data():
    # Mengunci nama berkas agar membaca tepat pada file yang ada di repository GitHub-mu
    filename = "bread_basket_per_transaksi.xlsx - Sheet1.csv"
    
    # Membaca data CSV menggunakan delimiter koma standar
    data = pd.read_csv(filename, sep=',')
        
    data.columns = [str(col).strip().lower() for col in data.columns]
    
    # Membuang baris yang duplikat sejak awal data mentah
    df_clean = data.copy().drop_duplicates()
    
    # Menemukan nama kolom transaksi dan item produk secara fleksibel
    col_transaction = [c for c in df_clean.columns if "transaction" in c or "id" in c][0]
    col_item = [c for c in df_clean.columns if "item" in c or "produk" in c or "items" in c][0]
    
    # Hapus data kosong (NaN) khusus pada kolom utama agar tidak merusak encoding mlxtend
    df_clean = df_clean.dropna(subset=[col_transaction, col_item])
    
    # Memastikan tipe data berupa string tulen dan membersihkan spasi tak terlihat
    df_clean[col_transaction] = df_clean[col_transaction].astype(str).str.strip()
    df_clean[col_item] = df_clean[col_item].astype(str).str.strip()
    
    # Standarisasi penamaan variabel global untuk kebutuhan visualisasi di bawah
    df_clean["Transaction"] = df_clean[col_transaction] 
    df_clean["Items"] = df_clean[col_item]
    
    # Mempertahankan ekstraksi fitur waktu transaksional jika kolomnya tersedia
    if "period_day" in df_clean.columns:
        df_clean["period_day"] = df_clean["period_day"].astype(str).str.strip().str.lower()
    if "weekday_weekend" in df_clean.columns:
        df_clean["weekday_weekend"] = df_clean["weekday_weekend"].astype(str).str.strip().str.lower()
        
    return df_clean

# ==========================================================
# MODELING ONE-HOT ENCODING & APRIORI
# ==========================================================
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
# Ditambahkan .astype(bool) agar lolos aturan ketat mlxtend di server cloud
basket_sets = pd.DataFrame(te_array, columns=te.columns_).astype(bool)

# Nilai minimum support diset ke 0.02 (2%) agar menangkap pola rules Toast -> Coffee
frequent_itemsets = apriori(basket_sets, min_support=0.02, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules[(rules["lift"] > 1) & (rules["confidence"] >= 0.3)].copy()

if len(rules) > 0:
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules["rule"] = rules["antecedents_str"] + " → " + rules["consequents_str"]
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False)

# VISUALISASI PREPARATION
all_items = [item for transaksi in transactions for item in transaksi]
item_counts = pd.Series(all_items).value_counts()
top_items_df = item_counts.head(10).reset_index()
top_items_df.columns = ["Produk", "Jumlah Pembelian"]

top_pairs = frequent_itemsets[frequent_itemsets["length"] == 2].copy()
top_pairs["Kombinasi Produk"] = top_pairs["itemsets"].apply(lambda x: " + ".join(list(x)))
top_pairs = top_pairs.sort_values(by="support", ascending=False).head(10)

top_pairs_display = top_pairs[["Kombinasi Produk", "support"]].copy()
top_pairs_display["Support (%)"] = top_pairs_display["support"] * 100

frequent_display = frequent_itemsets.head(10).copy()
frequent_display["Itemsets"] = frequent_display["itemsets"].apply(lambda x: ", ".join(list(x)))
frequent_display["Support (%)"] = frequent_display["support"] * 100

# ==========================================================
# VIEW DASHBOARD
# ==========================================================
st.title("Dashboard Market Basket Analysis")
st.markdown('<div class="hero-card"><h2>Strategi Bundling Produk Bakery</h2>Menampilkan pola pembelian pelanggan untuk menentukan produk promosi bersama.</div>', unsafe_allow_html=True)

# RINGKASAN METRIK UTAMA
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Transaksi", df_clean["Transaction"].nunique())
col2.metric("Jumlah Item Unik", basket_sets.shape[1])
col3.metric("Frequent Itemsets", len(frequent_itemsets))
col4.metric("Association Rules", len(rules))

# KELUARAN TEMUAN UTAMA
st.subheader("Temuan Utama")
if len(rules) > 0:
    best_rule = rules.iloc[0]
    st.markdown(f'<div class="finding-card"><div class="finding-title">{best_rule["rule"]}</div>Support: <b>{best_rule["support"]*100:.2f}%</b> | Confidence: <b>{best_rule["confidence"]*100:.2f}%</b> | Lift Ratio: <b>{best_rule["lift"]:.2f}x</b><br>Sekitar <b>{best_rule["confidence"]*100:.2f}%</b> pelanggan yang membeli <b>{best_rule["antecedents_str"]}</b> juga membeli <b>{best_rule["consequents_str"]}</b>.</div>', unsafe_allow_html=True)
else:
    st.warning("Belum ditemukan rule asosiasi yang memenuhi kriteria.")

# TABEL RULES DAN REKOMENDASI BUNDLING
left_col, right_col = st.columns([1.4, 1])
with left_col:
    st.subheader("Top Association Rules")
    if len(rules) > 0:
        tampil_rules = rules[["rule", "support", "confidence", "lift"]].head(8).copy()
        tampil_rules["Support (%)"] = tampil_rules["support"] * 100
        tampil_rules["Confidence (%)"] = tampil_rules["confidence"] * 100
        tampil_rules_format = tampil_rules[["rule", "Support (%)", "Confidence (%)", "lift"]].rename(columns={"rule": "Rule", "lift": "Lift Ratio"})
        st.dataframe(tampil_rules_format.style.format({"Support (%)": "{:.2f}%", "Confidence (%)": "{:.2f}%", "Lift Ratio": "{:.2f}x"}), use_container_width=True, height=280, hide_index=True)

with right_col:
    st.subheader("Rekomendasi Bundling")
    if len(rules) > 0:
        for i, row in rules.head(3).iterrows():
            st.markdown(f'<div class="recommend-card"><b>{row["rule"]}</b><br>Confidence: <b>{row["confidence"]*100:.2f}%</b> | Lift: <b>{row["lift"]:.2f}x</b><br><b>Saran:</b> Jadikan paket promo diskon gabungan.</div>', unsafe_allow_html=True)

# PLOT GRAFIK BATANG
st.subheader("Visualisasi Pola Pembelian")
col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    fig_top_items = px.bar(top_items_df.sort_values("Jumlah Pembelian", ascending=True), x="Jumlah Pembelian", y="Produk", orientation="h", text="Jumlah Pembelian", color_discrete_sequence=["#3B82F6"])
    st.plotly_chart(apply_chart_style(fig_top_items, 340), use_container_width=True)
with col_chart2:
    fig_pairs = px.bar(top_pairs_display.sort_values("Support (%)", ascending=True), x="Support (%)", y="Kombinasi Produk", orientation="h", text="Support (%)", color_discrete_sequence=["#10B981"])
    fig_pairs.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(apply_chart_style(fig_pairs, 340), use_container_width=True)

# PLOT ANALISIS WAKTU TRANSAKSI
col_chart3, col_chart4 = st.columns(2)
with col_chart3:
    if "period_day" in df_clean.columns:
        fig_period = px.bar(df_clean["period_day"].value_counts().reset_index(), x="period_day", y="count", text="count", color_discrete_sequence=["#64748B"])
        st.plotly_chart(apply_chart_style(fig_period, 300), use_container_width=True)
with col_chart4:
    if "weekday_weekend" in df_clean.columns:
        fig_weekday = px.bar(df_clean["weekday_weekend"].value_counts().reset_index(), x="weekday_weekend", y="count", text="count", color_discrete_sequence=["#F59E0B"])
        st.plotly_chart(apply_chart_style(fig_weekday, 300), use_container_width=True)

st.markdown('<div class="footer-card"><b>Kesimpulan:</b> Kombinasi produk dengan nilai Lift Ratio > 1 menunjukkan hubungan asosiasi positif yang kuat, sangat ideal digunakan sebagai dasar program promosi atau penataan rak produk.</div>', unsafe_allow_html=True)
