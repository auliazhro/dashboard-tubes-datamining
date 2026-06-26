import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="Dashboard Market Basket Analysis",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# CSS DASHBOARD DARK MODERN
# ==========================================================

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #0F2742 0%, #07111F 40%, #050B14 100%);
    color: #F8FAFC;
}

.block-container {
    padding-top: 4.8rem !important;
    padding-bottom: 0.2rem !important;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 1250px;
}

h1 {
    font-size: 34px !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    color: #F8FAFC !important;
    font-weight: 800 !important;
}

.sub-title {
    color: #60A5FA;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 14px;
}

.hero-card {
    background: rgba(15, 45, 75, 0.85);
    border: 1px solid rgba(96, 165, 250, 0.25);
    color: #DCEBFF;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 16px;
}

.section-card {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.22);
}

.caption-box {
    background: rgba(14, 48, 82, 0.72);
    color: #BFDBFE;
    padding: 10px 12px;
    border-radius: 6px;
    border-left: 4px solid #3B82F6;
    margin-bottom: 12px;
    font-size: 14px;
}

.metric-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 10px;
    padding: 18px 18px;
    min-height: 95px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.22);
}

.metric-title {
    font-size: 14px;
    color: #E5E7EB;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #FFFFFF;
}

.metric-sub {
    font-size: 13px;
    color: #CBD5E1;
}

.recommend-card {
    background: rgba(15, 23, 42, 0.86);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 8px;
    padding: 13px 14px;
    margin-bottom: 8px;
}

.rule-title {
    font-size: 16px;
    font-weight: 800;
    color: #FFFFFF;
}

.rule-text {
    font-size: 13px;
    color: #D1D5DB;
}

.badge {
    float: right;
    border: 1px solid #22C55E;
    color: #86EFAC;
    padding: 6px 10px;
    border-radius: 12px;
    font-weight: 800;
    font-size: 13px;
}

.footer-card {
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.25);
    color: #F8FAFC;
    padding: 12px 16px;
    border-radius: 10px;
    margin-top: 8px;
    margin-bottom: 0px;
}

footer {
    visibility: hidden;
}

section.main > div {
    padding-bottom: 0rem !important;
}

[data-testid="stDataFrame"] {
    background-color: rgba(15, 23, 42, 0.85);
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# CHART STYLE
# ==========================================================

CHART_BG = "#07111F"
CHART_TEXT = "#F8FAFC"
CHART_GRID = "rgba(148,163,184,0.18)"

def apply_chart_style(fig, height):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_TEXT, size=12),
        title=None,
        title_text="",
        margin=dict(l=10, r=20, t=10, b=10)
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
    files = [
        "bread_basket_per_transaksi.xlsx"
    ]

    selected = None
    for file in files:
        if Path(file).exists():
            selected = file
            break

    if selected is None:
        st.error("Dataset tidak ditemukan. Pastikan file dataset ada satu folder dengan app.py.")
        st.stop()

    if selected.endswith(".xlsx"):
        return pd.read_excel(selected)
    return pd.read_csv(selected)

df = load_data()

# ==========================================================
# DATA CLEANING
# ==========================================================

df_clean = df.copy()
df_clean.columns = [str(c).strip().lower() for c in df_clean.columns]
df_clean = df_clean.drop_duplicates()

col_transaction = "transaction" if "transaction" in df_clean.columns else [c for c in df_clean.columns if "transaction" in c or "transaksi" in c or c == "id"][0]
col_items = "items" if "items" in df_clean.columns else "item" if "item" in df_clean.columns else [c for c in df_clean.columns if "item" in c or "produk" in c][0]

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
    min_threshold=0.50
)

rules = rules[
    (rules["lift"] > 1) &
    (rules["confidence"] >= 0.53)
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
# HEADER
# ==========================================================

st.markdown("# 📊 Dashboard Market Basket Analysis")
st.markdown('<div class="sub-title">Strategi Bundling Produk Bakery</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
    Dashboard ini membantu melihat produk yang paling sering dibeli, kombinasi produk yang cocok dijadikan bundling,
    dan waktu transaksi paling ramai untuk mendukung pengambilan keputusan strategi promosi.
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# METRIC CARD
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🛒 Jumlah Transaksi</div>
        <div class="metric-value">{df_clean["Transaction"].nunique():,}</div>
        <div class="metric-sub">transaksi unik</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📦 Jumlah Item Unik</div>
        <div class="metric-value">{basket_sets.shape[1]}</div>
        <div class="metric-sub">item unik</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🧩 Frequent Itemsets</div>
        <div class="metric-value">{len(frequent_itemsets)}</div>
        <div class="metric-sub">itemset</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🔗 Association Rules</div>
        <div class="metric-value">{len(rules)}</div>
        <div class="metric-sub">rule</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================================
# SECTION 1 & 2
# ==========================================================

left, right = st.columns([1, 1.12])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("1. Top 10 Produk Paling Banyak Dibeli")
    st.markdown(
        '<div class="caption-box">Grafik ini menunjukkan produk yang paling sering dibeli pelanggan. Produk dengan pembelian tertinggi dapat dijadikan produk utama dalam strategi promosi.</div>',
        unsafe_allow_html=True
    )

    fig_top = px.bar(
        top_items_df.sort_values("Jumlah Pembelian", ascending=True),
        x="Jumlah Pembelian",
        y="Produk",
        orientation="h",
        text="Jumlah Pembelian",
        color_discrete_sequence=["#3B82F6"]
    )

    fig_top.update_traces(
        textposition="outside",
        textfont_color="#F8FAFC",
        cliponaxis=False
    )

    fig_top.update_layout(
        xaxis_title="Jumlah Pembelian",
        yaxis_title="",
    )

    fig_top = apply_chart_style(fig_top, 315)
    st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})
    
    top5_share = top_items_df.head(5).copy()
    total_top5 = top5_share["Jumlah Pembelian"].sum()
    top5_share["Persentase"] = top5_share["Jumlah Pembelian"] / total_top5 * 100

    fig_share = px.pie(
        top5_share,
        names="Produk",
        values="Jumlah Pembelian",
        hole=0.55,
        title="Komposisi Top 5 Produk Terlaris",
        color_discrete_sequence=["#3B82F6", "#22C55E", "#A855F7", "#F59E0B", "#EF4444"]
    )

    fig_share.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_color="#F8FAFC",
        hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>"
    )

    fig_share.update_layout(
        template="plotly_dark",
        height=230,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_TEXT, size=11),
        title_font=dict(color=CHART_TEXT, size=15),
        margin=dict(l=5, r=5, t=35, b=5),
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.08,
            x=0.5,
            xanchor="center",
            font=dict(size=10)
        )
    )

    st.plotly_chart(fig_share, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("2. Rekomendasi Kombinasi Produk untuk Bundling")
    st.markdown(
        '<div class="caption-box">Rekomendasi ini berasal dari pola transaksi pelanggan. Semakin tinggi persentase, semakin besar peluang produk kedua dibeli setelah produk pertama.</div>',
        unsafe_allow_html=True
    )

    if len(rules) > 0:
        icons = ["☕", "🥐", "🥪", "🍰", "🍪"]
        for idx, (_, row) in enumerate(rules.head(4).iterrows()):
            st.markdown(
                f"""
                <div class="recommend-card">
                    <span style="font-size:30px; margin-right:14px;">{icons[idx % len(icons)]}</span>
                    <span class="badge">{row['confidence']*100:.2f}%</span>
                    <span class="rule-title">{row['rule']}</span><br>
                    <span class="rule-text">
                    Dari pelanggan yang membeli {row['antecedents_str']}, {row['confidence']*100:.2f}% juga membeli {row['consequents_str']}.<br>
                    <b>Saran:</b> Buat paket promo {row['antecedents_str']} + {row['consequents_str']}.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Belum ada rekomendasi bundling.")

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# DETAIL NILAI
# ==========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("3. Detail Nilai Rekomendasi")
st.markdown(
    '<div class="caption-box">Support menunjukkan seberapa sering kombinasi produk muncul. Confidence menunjukkan peluang pelanggan membeli produk kedua setelah produk pertama. Lift Ratio menunjukkan kekuatan hubungan antarproduk. Nilai Lift > 1 berarti hubungan produk positif.</div>',
    unsafe_allow_html=True
)

if len(rules) > 0:
    table = rules[["rule", "support", "confidence", "lift"]].head(5).copy()
    table["Support (%)"] = table["support"] * 100
    table["Confidence (%)"] = table["confidence"] * 100
    table["Lift Ratio"] = table["lift"]
    table = table.rename(columns={"rule": "Rule"})
    table = table[["Rule", "Support (%)", "Confidence (%)", "Lift Ratio"]]
    table["Support (%)"] = table["Support (%)"].map(lambda x: f"{x:.2f}%")
    table["Confidence (%)"] = table["Confidence (%)"].map(lambda x: f"{x:.2f}%")
    table["Lift Ratio"] = table["Lift Ratio"].map(lambda x: f"{x:.2f}x")

    st.dataframe(table.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# KOMBINASI PRODUK
# ==========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("4. Kombinasi Produk yang Sering Dibeli Bersama")
st.markdown(
    '<div class="caption-box">Grafik ini menunjukkan pasangan produk yang sering muncul dalam transaksi yang sama. Kombinasi dengan nilai support tinggi dapat dipertimbangkan untuk bundling.</div>',
    unsafe_allow_html=True
)

fig_pair = px.bar(
    top_pairs.sort_values("Support (%)", ascending=True),
    x="Support (%)",
    y="Kombinasi Produk",
    orientation="h",
    text="Support (%)",
    color_discrete_sequence=["#4ADE80"]
)

fig_pair.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
    textfont_color="#F8FAFC",
    cliponaxis=False
)

fig_pair.update_layout(
    xaxis_title="Support (%)",
    yaxis_title=""
)

fig_pair = apply_chart_style(fig_pair, 285)
st.plotly_chart(fig_pair, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# WAKTU TRANSAKSI
# ==========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("5. Waktu Transaksi Paling Ramai")

t1, t2 = st.columns(2)

with t1:
    if "period_day" in df_clean.columns:
        st.markdown(
            '<div class="caption-box"><b>Berdasarkan Periode Hari</b><br>Menunjukkan periode hari dengan jumlah transaksi terbanyak.</div>',
            unsafe_allow_html=True
        )

        period_counts = df_clean["period_day"].value_counts().reset_index()
        period_counts.columns = ["Periode Hari", "Jumlah Transaksi"]

        fig_period = px.bar(
            period_counts,
            x="Periode Hari",
            y="Jumlah Transaksi",
            text="Jumlah Transaksi",
            color_discrete_sequence=["#60A5FA"]
        )

        fig_period.update_traces(textposition="outside", textfont_color="#F8FAFC")
        fig_period.update_layout(xaxis_title="Periode Hari", yaxis_title="Jumlah Transaksi")
        fig_period = apply_chart_style(fig_period, 240)

        st.plotly_chart(fig_period, use_container_width=True, config={"displayModeBar": False})

with t2:
    if "weekday_weekend" in df_clean.columns:
        st.markdown(
            '<div class="caption-box"><b>Weekday vs Weekend</b><br>Menunjukkan perbandingan transaksi pada hari kerja dan akhir pekan.</div>',
            unsafe_allow_html=True
        )

        weekday_counts = df_clean["weekday_weekend"].value_counts().reset_index()
        weekday_counts.columns = ["Kategori Hari", "Jumlah Transaksi"]

        fig_weekday = px.bar(
            weekday_counts,
            x="Kategori Hari",
            y="Jumlah Transaksi",
            text="Jumlah Transaksi",
            color_discrete_sequence=["#F59E0B"]
        )

        fig_weekday.update_traces(textposition="outside", textfont_color="#F8FAFC")
        fig_weekday.update_layout(xaxis_title="Kategori Hari", yaxis_title="Jumlah Transaksi")
        fig_weekday = apply_chart_style(fig_weekday, 240)

        st.plotly_chart(fig_weekday, use_container_width=True, config={"displayModeBar": False})

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# KESIMPULAN
# ==========================================================

st.markdown(
    """
    <div class="footer-card">
        <b>⭐ Kesimpulan:</b><br>
        Produk yang sering muncul bersama dapat digunakan sebagai dasar strategi promosi dan bundling.
        Produk dengan pembelian tinggi dapat dijadikan produk utama, kemudian dipasangkan dengan produk yang memiliki hubungan asosiasi kuat.
        <br><br>
        <div style="
            text-align:left;
            font-size:13px;
            color:#94A3B8;
            font-style:italic;
        ">
            By: Kelompok 12 IS-07-03
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
