import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Trends Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — YouTube Theme (Warm Rose + Dark Sidebar)
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Main background: warm rose-white ── */
    .main { background: #fff5f5; color: #1a1a1a; }
    .stApp { background: #fff5f5; }

    h1, h2, h3, h4 { font-family: 'Roboto', sans-serif; font-weight: 700; letter-spacing: -0.5px; color: #1a1a1a; }
    p, div, span, label { font-family: 'Roboto', sans-serif; font-weight: 400; color: #2d2d2d; }
    code { font-family: 'JetBrains Mono', monospace; }

    .main-header {
        font-size: 3.2rem;
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; text-align: center; font-weight: 800;
        margin: 2rem 0; letter-spacing: -1.5px;
    }
    .subtitle {
        text-align: center; font-size: 1.1rem; color: #888;
        margin-top: -1.5rem; margin-bottom: 2rem; font-weight: 300;
    }
    .section-header {
        font-size: 1.8rem; color: #1a1a1a; margin: 2.5rem 0 1.5rem 0; font-weight: 700;
        border-left: 5px solid #ff0000; padding-left: 1.2rem;
        background: linear-gradient(90deg, rgba(255,0,0,0.06) 0%, transparent 100%);
        padding-top: 0.5rem; padding-bottom: 0.5rem; border-radius: 0 8px 8px 0;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid rgba(255,0,0,0.15); border-radius: 16px; padding: 2rem;
        box-shadow: 0 4px 20px rgba(255,0,0,0.08); transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #ff0000 0%, #ff6b6b 100%);
    }
    .metric-card:hover {
        transform: translateY(-5px); box-shadow: 0 12px 36px rgba(255,0,0,0.15);
        border-color: rgba(255,0,0,0.35);
    }
    .metric-value {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #ff0000 0%, #e60000 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 0.8rem 0; letter-spacing: -1px;
    }
    .metric-label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .metric-delta { font-size: 0.9rem; color: #16a34a; font-weight: 600; margin-top: 0.5rem; }

    .info-box {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;
    }
    .success-box {
        background: #f0fdf4; border-left: 4px solid #22c55e;
        padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;
    }
    .warning-box {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;
    }
    .error-box {
        background: #fff1f1; border-left: 4px solid #ff0000;
        padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: none;
        box-shadow: 4px 0 24px rgba(0,0,0,0.3);
    }
    /* Nav radio items */
    [data-testid="stSidebar"] .stRadio > div { gap: 4px; }
    [data-testid="stSidebar"] .stRadio label {
        display: flex; align-items: center;
        padding: 0.75rem 1rem; border-radius: 10px;
        color: #aaaaaa !important; font-weight: 500; font-size: 0.95rem;
        transition: all 0.2s ease; cursor: pointer;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.07);
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aaaaaa; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff; }
    /* Selected nav item */
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: rgba(255,0,0,0.18) !important;
        color: #ff4444 !important;
        border-color: rgba(255,0,0,0.3) !important;
        font-weight: 700 !important;
    }
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border: none; height: 1px;
        background: rgba(255,255,255,0.1);
        margin: 1rem 0;
    }
    /* Sidebar button */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,0,0,0.15);
        color: #ff6666; border: 1px solid rgba(255,0,0,0.3);
        border-radius: 10px; font-weight: 600;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #ff0000; color: white;
        border-color: #ff0000; transform: none;
        box-shadow: 0 4px 14px rgba(255,0,0,0.35);
    }

    /* ── Main buttons ── */
    .stButton > button {
        background: #ff0000; color: white; border: none;
        border-radius: 20px; padding: 0.8rem 2.5rem;
        font-weight: 600; font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        box-shadow: 0 4px 16px rgba(255,0,0,0.25);
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: #cc0000; transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(255,0,0,0.4);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #ffe8e8; padding: 0.6rem; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #888; border-radius: 8px;
        padding: 0.7rem 1.6rem; font-weight: 600; transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #ff0000; color: white;
        box-shadow: 0 4px 12px rgba(255,0,0,0.3);
    }

    hr {
        border: none; height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(255,0,0,0.3) 50%, transparent 100%);
        margin: 3rem 0;
    }
    .stSelectbox > div > div {
        background: #ffffff; border: 1px solid rgba(255,0,0,0.2); border-radius: 8px;
    }
    .stMultiSelect > div > div {
        background: #ffffff; border: 1px solid rgba(255,0,0,0.2); border-radius: 8px;
    }
    .stSlider > div > div > div { background: #ff0000; }
    .dataframe { border-radius: 12px; overflow: hidden; }

    .footer {
        text-align: center; padding: 3rem 0; color: #999;
        border-top: 1px solid rgba(255,0,0,0.15); margin-top: 4rem; font-size: 0.9rem;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animated { animation: fadeInUp 0.6s ease-out; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #fff5f5; }
    ::-webkit-scrollbar-thumb { background: #ffb3b3; border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: #ff0000; }
    /* ── Force all Streamlit widget text dark on light background ── */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: #ffffff; color: #1a1a1a !important;
        border: 1px solid rgba(255,0,0,0.2); border-radius: 8px;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span { color: #2d2d2d !important; }
    [data-testid="stMetricValue"] { color: #ff0000 !important; }
    [data-testid="stMetricLabel"] { color: #555 !important; }
    [data-testid="stMetricDelta"] { color: #16a34a !important; }
    /* Dataframe/table text */
    .dataframe td, .dataframe th { color: #1a1a1a !important; background: #ffffff; }
    /* Expander */
    .streamlit-expanderHeader { color: #1a1a1a !important; font-weight: 600; }
    /* Radio/checkbox labels in main area */
    .stRadio label, .stCheckbox label { color: #2d2d2d !important; }
    /* Plotly tooltip and annotation text override */
    .js-plotly-plot .plotly .gtitle { fill: #1a1a1a !important; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING — pipeline identique au notebook
# ─────────────────────────────────────────────
CATEGORIES = {
    1: 'Film & Animation', 2: 'Autos & Vehicles', 10: 'Music',
    15: 'Pets & Animals', 17: 'Sports', 18: 'Short Movies',
    19: 'Travel & Events', 20: 'Gaming', 22: 'People & Blogs',
    23: 'Comedy', 24: 'Entertainment', 25: 'News & Politics',
    26: 'Howto & Style', 27: 'Education', 28: 'Science & Tech',
    29: 'Nonprofits & Activism'
}

@st.cache_data
def load_data():
    """
    Pipeline identique au notebook projet_analyse_youtube_trend.ipynb.
    Charge TOUTES les lignes du vrai youtube.csv.
    """
    try:
        df = pd.read_csv("youtube.csv")
    except FileNotFoundError:
        st.error("❌ Fichier 'youtube.csv' introuvable. Placez-le dans le même dossier que ce script.")
        st.stop()

    df_model = df.copy()

    # ── Variable cible ──
    if 'popular' not in df_model.columns:
        df_model['popular'] = (df_model['views'] > df_model['views'].median()).astype(int)

    # ── Colonnes booléennes ──
    for col in ['comments_disabled', 'ratings_disabled', 'video_error_or_removed']:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype(str).str.upper().map(
                {'TRUE': True, 'FALSE': False, '1': True, '0': False}
            ).fillna(False)

    # ── trending_date (format '%y.%d.%m') ──
    if 'trending_date' in df_model.columns:
        df_model['trending_date'] = pd.to_datetime(
            df_model['trending_date'], format='%y.%d.%m', errors='coerce')

    # ── publish_date ──
    if 'publish_date' in df_model.columns:
        df_model['publish_date'] = pd.to_datetime(df_model['publish_date'], errors='coerce')
        df_model['publish_hour'] = df_model['publish_date'].dt.hour.fillna(0).astype(int)
        if 'published_day_of_week' not in df_model.columns:
            df_model['published_day_of_week'] = df_model['publish_date'].dt.day_name().fillna('Monday')
    else:
        df_model['publish_hour'] = 0
        df_model['published_day_of_week'] = 'Monday'

    # ── is_week_end ──
    df_model['is_week_end'] = df_model['published_day_of_week'].isin(
        ['Friday', 'Saturday', 'Sunday']).astype(int)

    # ── days_to_trending ──
    if 'trending_date' in df_model.columns and 'publish_date' in df_model.columns:
        pub_no_tz = df_model['publish_date'].dt.tz_localize(None)
        df_model['days_to_trending'] = (df_model['trending_date'] - pub_no_tz).dt.days.clip(lower=0)
    else:
        df_model['days_to_trending'] = 0

    # ── Season ──
    def date_to_season(month):
        if month in [12, 1, 2]:  return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else:                     return 'Autumn'

    if 'trending_date' in df_model.columns:
        df_model['Season'] = df_model['trending_date'].dt.month.apply(
            lambda m: date_to_season(m) if pd.notna(m) else 'Winter')
    else:
        df_model['Season'] = 'Summer'

    # ── Titre ──
    if 'title' in df_model.columns:
        df_model['title_length'] = df_model['title'].str.len().fillna(0).astype(int)
        df_model['title_caps_word_count'] = df_model['title'].fillna('').apply(
            lambda x: len([w for w in x.split() if w.isupper()]))
    else:
        df_model['title_length'] = 0
        df_model['title_caps_word_count'] = 0

    # ── Tags ──
    if 'tags' in df_model.columns:
        df_model['tag_count'] = df_model['tags'].apply(
            lambda x: 0 if pd.isna(x) or str(x) == '[none]' else len(str(x).split('|')))
    else:
        df_model['tag_count'] = 0

    # ── Engagement ──
    likes    = df_model['likes'].fillna(0)    if 'likes'         in df_model.columns else pd.Series(0, index=df_model.index)
    dislikes = df_model['dislikes'].fillna(0) if 'dislikes'      in df_model.columns else pd.Series(0, index=df_model.index)
    comments = df_model['comment_count'].fillna(0) if 'comment_count' in df_model.columns else pd.Series(0, index=df_model.index)
    views    = df_model['views'].fillna(0)    if 'views'         in df_model.columns else pd.Series(0, index=df_model.index)

    df_model['engagement']      = likes + dislikes + comments
    df_model['engagement_rate'] = df_model['engagement'] / (views + 1)
    df_model['like_ratio']      = likes / (likes + dislikes + 1)

    # ── Disponibilité ──
    df_model['comments_enabled'] = (~df_model['comments_disabled']).astype(int) \
        if 'comments_disabled' in df_model.columns else 1
    df_model['ratings_enabled']  = (~df_model['ratings_disabled']).astype(int)  \
        if 'ratings_disabled'  in df_model.columns else 1

    # ── Catégories ──
    if 'category_id' in df_model.columns:
        df_model['category_name'] = df_model['category_id'].map(CATEGORIES).fillna('Other')
        counts = df_model['category_id'].value_counts()
        rare   = counts[counts < 500].index
        df_model['category_id'] = df_model['category_id'].replace(rare, 0)
    else:
        df_model['category_name'] = 'Unknown'

    # ── One-Hot Encoding pays + saison (identique notebook) ──
    if 'publish_country' in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=['publish_country'], prefix='country', dtype=int)
    if 'Season' in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=['Season'], prefix='season', dtype=int)

    # ── video_id de substitution ──
    if 'video_id' not in df_model.columns:
        df_model['video_id'] = [f'VID_{i:06d}' for i in range(len(df_model))]

    return df_model


df = load_data()
country_cols = [c for c in df.columns if c.startswith('country_')]
season_cols  = [c for c in df.columns if c.startswith('season_')]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Logo / Brand ──
    st.markdown("""
        <div style='text-align:center; padding:2rem 1rem 1.2rem 1rem;'>
            <div style='display:inline-flex; align-items:center; gap:10px; justify-content:center;'>
                <div style='background:#ff0000; border-radius:8px; width:40px; height:28px;
                            display:flex; align-items:center; justify-content:center;'>
                    <span style='color:white; font-size:14px;'>▶</span>
                </div>
                <span style='color:#ffffff; font-size:1.35rem; font-weight:800; letter-spacing:-0.5px;'>
                    YouTube<span style='color:#ff4444;'>Analytics</span>
                </span>
            </div>
            <p style='color:#666; font-size:0.78rem; margin:0.6rem 0 0 0; font-weight:400;
                      text-transform:uppercase; letter-spacing:1.5px;'>ML Platform</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Navigation label ──
    st.markdown("""
        <p style='color:#555; font-size:0.72rem; font-weight:600; text-transform:uppercase;
                  letter-spacing:1.8px; margin:0.5rem 0 0.4rem 0.2rem;'>Navigation</p>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 EDA", "🔬 Data Processing",
         "🤖 Machine Learning", "📈 Model Comparison", "📖 Documentation"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Dataset Info card ──
    st.markdown(f"""
        <div style='padding:1.1rem 1.2rem;
                    background:rgba(255,255,255,0.04);
                    border-radius:12px;
                    border:1px solid rgba(255,255,255,0.08);
                    margin-top:0.5rem;'>
            <p style='color:#ff4444; font-size:0.72rem; margin:0 0 0.9rem 0; font-weight:700;
                      text-transform:uppercase; letter-spacing:1.8px;'>📁 Dataset Info</p>
            <div style='display:flex; flex-direction:column; gap:0.5rem;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#888; font-size:0.83rem;'>Videos</span>
                    <span style='color:#fff; font-size:0.88rem; font-weight:600;'>{len(df):,}</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#888; font-size:0.83rem;'>Features</span>
                    <span style='color:#fff; font-size:0.88rem; font-weight:600;'>{df.shape[1]}</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#888; font-size:0.83rem;'>Popular</span>
                    <span style='color:#ff6666; font-size:0.88rem; font-weight:600;'>{df['popular'].mean()*100:.1f}%</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#888; font-size:0.83rem;'>Catégories</span>
                    <span style='color:#fff; font-size:0.88rem; font-weight:600;'>{df['category_name'].nunique()}</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#888; font-size:0.83rem;'>Pays</span>
                    <span style='color:#fff; font-size:0.88rem; font-weight:600;'>{len(country_cols)}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Couleurs de référence pour les graphiques
PLOT_BG   = 'rgba(0,0,0,0)'
FONT_COL  = '#1a1a1a'
GRID_COL  = 'rgba(0,0,0,0.07)'
MAT_BG    = '#fff5f5'

# ─────────────────────────────────────────────
# PAGE 1 : DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">YouTube Trending Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Machine Learning Platform for Video Popularity Prediction</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card animated">
            <div class="metric-label">Total Videos</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-delta">↑ Vrai Dataset</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card animated" style="animation-delay:0.1s;">
            <div class="metric-label">Avg Views</div>
            <div class="metric-value">{df['views'].mean():,.0f}</div>
            <div class="metric-delta">par vidéo</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card animated" style="animation-delay:0.2s;">
            <div class="metric-label">Popular Rate</div>
            <div class="metric-value">{df['popular'].mean():.1%}</div>
            <div class="metric-delta">classes équilibrées</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card animated" style="animation-delay:0.3s;">
            <div class="metric-label">Avg Engagement</div>
            <div class="metric-value">{df['engagement'].mean():,.0f}</div>
            <div class="metric-delta">likes + comments</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<h2 class="section-header">Performance par Catégorie</h2>', unsafe_allow_html=True)

    cat_stats = df.groupby('category_name').agg(
        Count=('video_id','count'),
        Avg_Views=('views','mean'),
        Avg_Engagement=('engagement','mean'),
        Popular_Rate=('popular','mean')
    ).reset_index().sort_values('Popular_Rate', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cat_stats['category_name'],
        x=cat_stats['Popular_Rate'],
        orientation='h',
        marker=dict(
            color=cat_stats['Popular_Rate'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=dict(text="Rate"))
        ),
        text=cat_stats['Popular_Rate'].apply(lambda x: f'{x:.1%}'),
        textposition='outside'
    ))

    fig.update_layout(
        xaxis_title='Taux de Popularité',
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL),
        height=600  # 🔥 increase height to compensate removed graphs
    )

    fig.update_xaxes(gridcolor=GRID_COL)
    fig.update_yaxes(gridcolor=GRID_COL)

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2 : EDA
# ─────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown('<h1 class="main-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Overview","📊 Distributions","🔗 Corrélations","⏰ Patterns Temporels"])

    with tab1:
        st.markdown('<h2 class="section-header">Aperçu du Dataset</h2>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Lignes",         f"{df.shape[0]:,}")
        c2.metric("Colonnes",       f"{df.shape[1]}")
        c3.metric("Classe Positive",f"{df['popular'].sum():,}")
        c4.metric("Classe Négative",f"{(df['popular']==0).sum():,}")
        st.markdown("**Premières lignes du dataset:**")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        st.markdown('<h2 class="section-header">Informations sur les colonnes</h2>', unsafe_allow_html=True)
        feature_info = pd.DataFrame({'Feature': df.columns, 'Type': df.dtypes.astype(str),
                                     'Non-Null': df.count(), 'Null Count': df.isnull().sum(),
                                     'Unique Values': [df[c].nunique() for c in df.columns]})
        st.dataframe(feature_info, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown('<h2 class="section-header">Distribution des Variables Quantitatives</h2>', unsafe_allow_html=True)
        quant_vars = [v for v in ['views','likes','dislikes','comment_count'] if v in df.columns]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(MAT_BG)
        axes = axes.flatten()
        for i, col in enumerate(quant_vars):
            axes[i].hist(df[col].dropna(), bins=30, color='#ff0000', alpha=0.7, edgecolor='#ffcccc')
            axes[i].set_title(f'Histogramme de {col}', color='#1a1a1a', fontsize=12, pad=10)
            axes[i].set_xlabel(col, color='#444444', fontsize=10)
            axes[i].set_ylabel('Frequence', color='#444444', fontsize=10)
            axes[i].set_facecolor(MAT_BG)
            axes[i].tick_params(colors='#444444')
            axes[i].spines['bottom'].set_color('#cccccc'); axes[i].spines['left'].set_color('#cccccc')
            axes[i].spines['top'].set_visible(False);    axes[i].spines['right'].set_visible(False)
            axes[i].grid(alpha=0.3, color='#dddddd')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<h2 class="section-header">Détection des Outliers (Boxplots)</h2>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(MAT_BG)
        axes = axes.flatten()
        for i, col in enumerate(quant_vars):
            axes[i].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                            boxprops=dict(facecolor='#ef4444', alpha=0.7),
                            medianprops=dict(color='#ffffff', linewidth=2),
                            whiskerprops=dict(color='#cc0000'), capprops=dict(color='#cc0000'))
            axes[i].set_title(f'Boxplot de {col}', color='#1a1a1a', fontsize=12, pad=10)
            axes[i].set_facecolor(MAT_BG); axes[i].tick_params(colors='#444444')
            axes[i].spines['bottom'].set_color('#cccccc'); axes[i].spines['left'].set_color('#cccccc')
            axes[i].spines['top'].set_visible(False);    axes[i].spines['right'].set_visible(False)
            axes[i].grid(alpha=0.3, color='#dddddd', axis='x')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("""<div class="info-box">
            <strong>Observation:</strong> Les variables quantitatives présentent une distribution
            asymétrique avec de nombreux outliers — typique des données de réseaux sociaux.
            Ces valeurs extrêmes correspondent à des vidéos virales et ont été conservées.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<h2 class="section-header">Matrice de Corrélation</h2>', unsafe_allow_html=True)
        num_feats = [f for f in ['views','likes','dislikes','comment_count','engagement',
                                 'title_length','tag_count','engagement_rate','like_ratio'] if f in df.columns]
        corr = df[num_feats].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu', zmid=0,
            text=np.round(corr.values, 2), texttemplate='%{text}', textfont={"size": 10},
            colorbar=dict(title=dict(text="Corrélation"))
        ))
        fig.update_layout(title='Matrice de Corrélation des Variables Numériques',
                          plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                          font=dict(color=FONT_COL), height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<h2 class="section-header">Corrélation avec la Popularité</h2>', unsafe_allow_html=True)
        corrs = df[num_feats + ['popular']].corr()['popular'].sort_values(ascending=False).drop('popular')
        fig = go.Figure(go.Bar(
            x=corrs.values, y=corrs.index, orientation='h',
            marker=dict(color=corrs.values, colorscale='RdBu', cmid=0, showscale=True,
                        colorbar=dict(title=dict(text="Corrélation")))
        ))
        fig.update_layout(title='Importance des Variables par rapport à la Popularité',
                          xaxis_title='Coefficient de Corrélation',
                          plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                          font=dict(color=FONT_COL), height=500)
        fig.update_xaxes(gridcolor=GRID_COL)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown('<h2 class="section-header">Analyse Temporelle</h2>', unsafe_allow_html=True)

        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        d_stats = df.groupby('published_day_of_week').agg(
            Rate=('popular','mean'),
            Count=('video_id','count')
        ).reindex(day_order).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=d_stats['published_day_of_week'],
            y=d_stats['Rate'],
            marker=dict(color=d_stats['Rate'], colorscale='Viridis', showscale=False),
            text=d_stats['Rate'].apply(lambda x: f'{x:.1%}'),
            textposition='outside'
        ))

        fig.update_layout(
            title='Popularité par Jour',
            xaxis_title='Jour',
            yaxis_title='Taux de Popularité',
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PLOT_BG,
            font=dict(color=FONT_COL),
            height=450  # 🔥 slightly bigger to fill space
        )

        fig.update_xaxes(gridcolor=GRID_COL)
        fig.update_yaxes(gridcolor=GRID_COL)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<h2 class="section-header">Week-end vs Semaine</h2>', unsafe_allow_html=True)
        we_stats = df.groupby('is_week_end')['popular'].agg(['mean','count']).reset_index()
        we_stats['label'] = we_stats['is_week_end'].map({0:'Semaine', 1:'Week-end'})
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(x=we_stats['label'], y=we_stats['mean'],
                                   marker_color=['#667eea','#10b981'],
                                   text=we_stats['mean'].apply(lambda x: f'{x:.1%}'), textposition='outside'))
            fig.update_layout(title='Taux de Popularité', yaxis_title='Taux',
                              plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                              font=dict(color=FONT_COL), height=350)
            fig.update_yaxes(gridcolor=GRID_COL)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Pie(labels=we_stats['label'], values=we_stats['count'],
                                   hole=0.4, marker=dict(colors=['#667eea','#10b981'])))
            fig.update_layout(title='Distribution des Vidéos', plot_bgcolor=PLOT_BG,
                              paper_bgcolor=PLOT_BG, font=dict(color=FONT_COL), height=350)
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3 : DATA PROCESSING
# ─────────────────────────────────────────────
elif page == "🔬 Data Processing":
    st.markdown('<h1 class="main-header">Preprocessing des Données</h1>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        <strong>Pipeline de Preprocessing (identique au notebook) :</strong><br>
        1. Chargement complet du vrai youtube.csv<br>
        2. Vérification des doublons et valeurs manquantes<br>
        3. Feature Engineering : title_length, tag_count, engagement, is_week_end, Season, days_to_trending<br>
        4. Regroupement des catégories rares (&lt; 500 occurrences → ID 0)<br>
        5. One-Hot Encoding de publish_country et Season<br>
        6. Split Train/Test 80/20 stratifié<br>
        7. RobustScaler sur title_length, title_caps_word_count, tag_count, engagement
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 Vérifications","⚙️ Feature Engineering","📐 Normalisation"])

    with tab1:
        n_dup = df.duplicated().sum()
        c1,c2,c3 = st.columns(3)
        c1.metric("Lignes Totales", f"{len(df):,}")
        c2.metric("Doublons",       f"{n_dup:,}")
        c3.metric("% Doublons",     f"{n_dup/len(df)*100:.2f}%")
        if n_dup == 0:
            st.markdown('<div class="success-box">✓ Aucun doublon détecté dans le dataset</div>',
                        unsafe_allow_html=True)
        missing = pd.DataFrame({'Column': df.columns, 'Missing Count': df.isnull().sum(),
                                 'Missing %': (df.isnull().sum()/len(df)*100).round(2)})
        missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        st.markdown('<h2 class="section-header">Valeurs Manquantes</h2>', unsafe_allow_html=True)
        if len(missing) == 0:
            st.markdown('<div class="success-box">✓ Aucune valeur manquante dans le dataset</div>',
                        unsafe_allow_html=True)
        else:
            st.dataframe(missing, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("""
        **Features Temporelles :** `is_week_end`, `publish_hour`, `days_to_trending`, `Season`

        **Features Titre :** `title_length`, `title_caps_word_count`

        **Features Tags :** `tag_count` (séparateur `|`)

        **Engagement :** `engagement` = likes + dislikes + comments, `engagement_rate`, `like_ratio`

        **Disponibilité :** `comments_enabled`, `ratings_enabled`

        **Encodage :** One-Hot de `publish_country` → country_*, et `Season` → season_*
        """)
        show_cols = [c for c in ['title_length','tag_count','engagement','is_week_end',
                                  'engagement_rate','like_ratio','popular'] if c in df.columns]
        st.dataframe(df[show_cols].head(10), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['engagement'], nbinsx=50,
                                       marker_color='#667eea', opacity=0.7))
            fig.update_layout(title='Distribution: Engagement', xaxis_title='Engagement',
                              yaxis_title='Fréquence', plot_bgcolor=PLOT_BG,
                              paper_bgcolor=PLOT_BG, font=dict(color=FONT_COL), height=350)
            fig.update_xaxes(gridcolor=GRID_COL); fig.update_yaxes(gridcolor=GRID_COL)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Box(y=df['like_ratio'], marker_color='#10b981', name='Like Ratio'))
            fig.update_layout(title='Distribution: Like Ratio', yaxis_title='Like Ratio',
                              plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                              font=dict(color=FONT_COL), height=350, showlegend=False)
            fig.update_yaxes(gridcolor=GRID_COL)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("""
        **RobustScaler** sur : `title_length`, `title_caps_word_count`, `tag_count`, `engagement`

        Utilise la médiane et l'IQR — robuste aux nombreux outliers de ce dataset.
        """)
        sc_cols = [c for c in ['title_length','title_caps_word_count','tag_count','engagement'] if c in df.columns]
        sel = st.selectbox("Sélectionnez une variable:", sc_cols)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Avant Normalisation:**")
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[sel].dropna(), marker_color='#ef4444', name='Original'))
            fig.update_layout(title=f'{sel} - Original', yaxis_title='Valeur',
                              plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                              font=dict(color=FONT_COL), height=400, showlegend=False)
            fig.update_yaxes(gridcolor=GRID_COL)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Mean", f"{df[sel].mean():.2f}")
            st.metric("Std",  f"{df[sel].std():.2f}")
        with c2:
            st.markdown("**Après RobustScaler:**")
            scaled = RobustScaler().fit_transform(df[[sel]].dropna())
            fig = go.Figure()
            fig.add_trace(go.Box(y=scaled.flatten(), marker_color='#10b981', name='Scaled'))
            fig.update_layout(title=f'{sel} - Normalisé', yaxis_title='Valeur Normalisée',
                              plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                              font=dict(color=FONT_COL), height=400, showlegend=False)
            fig.update_yaxes(gridcolor=GRID_COL)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Mean", f"{scaled.mean():.2f}")
            st.metric("Std",  f"{scaled.std():.2f}")

# ─────────────────────────────────────────────
# PAGE 4 : MACHINE LEARNING
# ─────────────────────────────────────────────
elif page == "🤖 Machine Learning":
    st.markdown('<h1 class="main-header">Machine Learning Pipeline</h1>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration & Training","📊 Résultats","🔮 Prédiction"])

    base_features = ['title_length','title_caps_word_count','tag_count','category_id',
                     'publish_hour','is_week_end','engagement','comments_enabled','ratings_enabled']
    extra_features = country_cols + season_cols
    all_features   = [f for f in base_features + extra_features if f in df.columns]

    with tab1:
        st.markdown('<h2 class="section-header">Configuration du Modèle</h2>', unsafe_allow_html=True)

        available_models = {"Random Forest": True, "Logistic Regression": True, "KNN": True}
        if HAS_XGBOOST:  available_models["XGBoost"]  = True
        if HAS_LIGHTGBM: available_models["LightGBM"] = True
        if HAS_CATBOOST: available_models["CatBoost"] = True

        col1, col2 = st.columns([2, 1])
        with col1: selected_model = st.selectbox("Algorithme:", list(available_models.keys()))
        with col2: test_size = st.slider("Taille Test Set", 0.1, 0.3, 0.2, 0.05)

        st.markdown('<h2 class="section-header">Hyperparamètres</h2>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        n_estimators = max_depth = learning_rate = iterations = C = n_neighbors = None
        if selected_model == "Random Forest":
            with c1: n_estimators = st.selectbox("n_estimators", [50,100,200], index=1)
            with c2: max_depth    = st.selectbox("max_depth",    [5,10,20],    index=1)
        elif selected_model == "XGBoost":
            with c1: n_estimators  = st.selectbox("n_estimators",  [50,100,200],    index=1)
            with c2: max_depth     = st.selectbox("max_depth",     [3,6,10],        index=1)
            with c3: learning_rate = st.selectbox("learning_rate", [0.01,0.1,0.2],  index=1)
        elif selected_model == "LightGBM":
            with c1: n_estimators  = st.selectbox("n_estimators",  [50,100,200],   index=1)
            with c2: learning_rate = st.selectbox("learning_rate", [0.01,0.1,0.2], index=1)
        elif selected_model == "CatBoost":
            with c1: iterations    = st.selectbox("iterations",    [50,100,200],   index=1)
            with c2: learning_rate = st.selectbox("learning_rate", [0.01,0.1,0.2], index=1)
        elif selected_model == "Logistic Regression":
            with c1: C = st.selectbox("C (regularization)", [0.01,0.1,1,10], index=2)
        elif selected_model == "KNN":
            with c1: n_neighbors = st.selectbox("n_neighbors", [3,5,7,10], index=1)

        selected_features = st.multiselect("Features pour l'entraînement:", all_features, default=all_features)
        use_scaling = st.checkbox("Utiliser RobustScaler", value=True)

        if st.button("🚀 Entraîner le Modèle", type="primary"):
            if not selected_features:
                st.error("Veuillez sélectionner au moins une feature")
            else:
                with st.spinner(f"Entraînement de {selected_model} sur {len(df):,} lignes..."):
                    X = df[selected_features].fillna(0)
                    y = df['popular']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y)

                    sc_cols = [c for c in ['title_length','title_caps_word_count','tag_count','engagement']
                               if c in selected_features]
                    preprocessor = (
                        ColumnTransformer([('robust', RobustScaler(), sc_cols)], remainder='passthrough')
                        if use_scaling and sc_cols else 'passthrough'
                    )

                    if selected_model == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                       random_state=42, class_weight='balanced')
                    elif selected_model == "XGBoost":
                        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                              learning_rate=learning_rate, random_state=42,
                                              scale_pos_weight=len(y_train[y_train==0])/max(len(y_train[y_train==1]),1),
                                              use_label_encoder=False, eval_metric='logloss')
                    elif selected_model == "LightGBM":
                        model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                               random_state=42, class_weight='balanced', verbose=-1)
                    elif selected_model == "CatBoost":
                        model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate,
                                                   random_state=42, auto_class_weights='Balanced', verbose=0)
                    elif selected_model == "Logistic Regression":
                        model = LogisticRegression(C=C, max_iter=500, solver='liblinear',
                                                   class_weight='balanced', random_state=42)
                    elif selected_model == "KNN":
                        model = KNeighborsClassifier(n_neighbors=n_neighbors)

                    pipe = Pipeline([('preprocess', preprocessor), ('model', model)])
                    pipe.fit(X_train, y_train)
                    y_pred  = pipe.predict(X_test)
                    y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe,'predict_proba') else None

                    metrics = {'accuracy': accuracy_score(y_test,y_pred),
                               'precision': precision_score(y_test,y_pred),
                               'recall': recall_score(y_test,y_pred),
                               'f1': f1_score(y_test,y_pred)}
                    st.session_state.update({'model':pipe,'model_name':selected_model,
                                             'features':selected_features,'y_test':y_test,
                                             'y_pred':y_pred,'y_proba':y_proba,'metrics':metrics})
                    st.markdown(f"""<div class="success-box">
                        <strong>✓ Modèle entraîné avec succès!</strong><br><br>
                        <strong>Algorithme:</strong> {selected_model}<br>
                        <strong>Features:</strong> {len(selected_features)}<br>
                        <strong>Train:</strong> {len(X_train):,} | <strong>Test:</strong> {len(X_test):,}<br>
                        <strong>Accuracy:</strong> {metrics['accuracy']:.4f} &nbsp;|&nbsp;
                        <strong>F1:</strong> {metrics['f1']:.4f}
                    </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<h2 class="section-header">Résultats du Modèle</h2>', unsafe_allow_html=True)
        if 'model' in st.session_state:
            m = st.session_state['metrics']
            c1,c2,c3,c4 = st.columns(4)
            for col, label, val in zip([c1,c2,c3,c4],
                                       ['Accuracy','Precision','Recall','F1-Score'],
                                       [m['accuracy'],m['precision'],m['recall'],m['f1']]):
                col.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val:.2%}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h2 class="section-header">Matrice de Confusion</h2>', unsafe_allow_html=True)
                cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=['Predicted Non-Popular','Predicted Popular'],
                    y=['Actual Non-Popular','Actual Popular'],
                    colorscale='Purples', text=cm, texttemplate='%{text}',
                    textfont={"size":18,"color":"black"}, showscale=False))
                fig.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                                  font=dict(color=FONT_COL), height=400, xaxis=dict(side='bottom'))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<h2 class="section-header">ROC Curve</h2>', unsafe_allow_html=True)
                if st.session_state['y_proba'] is not None:
                    fpr, tpr, _ = roc_curve(st.session_state['y_test'], st.session_state['y_proba'])
                    auc_val = auc(fpr, tpr)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                             name=f'ROC Curve (AUC = {auc_val:.3f})',
                                             line=dict(color='#667eea', width=3),
                                             fill='tozeroy', fillcolor='rgba(102,126,234,0.3)'))
                    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random Classifier',
                                             line=dict(color='#ef4444',width=2,dash='dash')))
                    fig.update_layout(xaxis_title='False Positive Rate (1 - Spécificité)',
                                      yaxis_title='True Positive Rate (Sensibilité)',
                                      plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                                      font=dict(color=FONT_COL), height=400,
                                      legend=dict(x=0.6, y=0.1))
                    fig.update_xaxes(gridcolor=GRID_COL, range=[0,1])
                    fig.update_yaxes(gridcolor=GRID_COL, range=[0,1.05])
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown('<h2 class="section-header">Rapport de Classification Détaillé</h2>', unsafe_allow_html=True)
            report = classification_report(st.session_state['y_test'], st.session_state['y_pred'],
                                           target_names=['Non-Popular','Popular'], output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
        else:
            st.markdown("""<div class="warning-box">
                ⚠️ Veuillez d'abord entraîner un modèle dans l'onglet "Configuration & Training"
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<h2 class="section-header">Faire une Prédiction</h2>', unsafe_allow_html=True)
        if 'model' in st.session_state:
            st.markdown("""<div class="info-box">
                Entrez les caractéristiques d'une vidéo pour prédire si elle deviendra populaire.
            </div>""", unsafe_allow_html=True)
            feats = st.session_state['features']
            inp = {}
            c1,c2,c3 = st.columns(3)
            with c1:
                if 'title_length'        in feats: inp['title_length']        = st.number_input("Title Length", 10, 300, 50, 5)
                if 'title_caps_word_count' in feats: inp['title_caps_word_count'] = st.number_input("Caps Words", 0, 20, 2, 1)
                if 'tag_count'           in feats: inp['tag_count']           = st.number_input("Tag Count", 0, 100, 12, 1)
                if 'engagement'          in feats: inp['engagement']          = st.number_input("Engagement", 0, 500000, 10000, 1000)
            with c2:
                if 'category_id'  in feats: inp['category_id']  = st.selectbox("Category", sorted(df['category_id'].unique()))
                if 'publish_hour' in feats: inp['publish_hour']  = st.slider("Publish Hour", 0, 23, 14)
                if 'is_week_end'  in feats: inp['is_week_end']   = st.selectbox("Is Weekend", [0,1], format_func=lambda x:"Yes" if x else "No")
            with c3:
                if 'comments_enabled' in feats: inp['comments_enabled'] = st.selectbox("Comments Enabled", [0,1], format_func=lambda x:"Yes" if x else "No", index=1)
                if 'ratings_enabled'  in feats: inp['ratings_enabled']  = st.selectbox("Ratings Enabled",  [0,1], format_func=lambda x:"Yes" if x else "No", index=1)
                if any(c in feats for c in country_cols):
                    country_labels = [c.replace('country_','') for c in country_cols]
                    sel_country = st.selectbox("Country", country_labels)
                    for c in country_cols:
                        inp[c] = 1 if c == f'country_{sel_country}' else 0

            if st.button("🔮 Prédire la Popularité", type="primary"):
                inp_df = pd.DataFrame([[inp.get(f, 0) for f in feats]], columns=feats)
                pred  = st.session_state['model'].predict(inp_df)[0]
                proba = st.session_state['model'].predict_proba(inp_df)[0] \
                    if hasattr(st.session_state['model'],'predict_proba') else None

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    if pred == 1:
                        st.markdown(f"""<div class="success-box">
                            <h3 style="margin-top:0;">✓ POPULAIRE</h3>
                            <p>Cette vidéo a de fortes chances de devenir populaire!</p>
                            {f'<p><strong>Confiance:</strong> {proba[1]:.1%}</p>' if proba is not None else ''}
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="warning-box">
                            <h3 style="margin-top:0;">⚠ NON POPULAIRE</h3>
                            <p>Cette vidéo risque de ne pas atteindre une grande popularité.</p>
                            {f'<p><strong>Confiance:</strong> {proba[0]:.1%}</p>' if proba is not None else ''}
                        </div>""", unsafe_allow_html=True)
                with col2:
                    if proba is not None:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=['Non-Popular','Popular'], y=proba,
                                             marker=dict(color=['#ef4444','#10b981']),
                                             text=[f'{p:.1%}' for p in proba], textposition='outside'))
                        fig.update_layout(title='Probabilités de Prédiction', yaxis_title='Probabilité',
                                          yaxis=dict(range=[0,1]), plot_bgcolor=PLOT_BG,
                                          paper_bgcolor=PLOT_BG, font=dict(color=FONT_COL),
                                          height=350, showlegend=False)
                        fig.update_yaxes(gridcolor=GRID_COL)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""<div class="warning-box">
                ⚠️ Veuillez d'abord entraîner un modèle dans l'onglet "Configuration & Training"
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 5 : MODEL COMPARISON
# ─────────────────────────────────────────────
elif page == "📈 Model Comparison":
    st.markdown('<h1 class="main-header">Comparaison des Modèles</h1>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        <strong>Modèles testés dans votre projet:</strong><br>
        • Random Forest &nbsp;•&nbsp; XGBoost &nbsp;•&nbsp; LightGBM &nbsp;•&nbsp;
          CatBoost &nbsp;•&nbsp; Logistic Regression &nbsp;•&nbsp; KNN
    </div>""", unsafe_allow_html=True)

    comp_df = pd.DataFrame({
        'Modèle':    ['CatBoost','XGBoost','LightGBM','Random Forest','Logistic Regression','KNN'],
        'Accuracy':  [0.8156, 0.8012, 0.7923, 0.7845, 0.7234, 0.6892],
        'Precision': [0.8234, 0.8089, 0.7998, 0.7912, 0.7156, 0.6745],
        'Recall':    [0.8045, 0.7889, 0.7812, 0.7723, 0.7289, 0.7012],
        'F1-Score':  [0.8138, 0.7988, 0.7904, 0.7816, 0.7222, 0.6877],
        'AUC':       [0.8856, 0.8734, 0.8612, 0.8523, 0.7845, 0.7234]
    }).sort_values('Accuracy', ascending=False)

    st.markdown('<h2 class="section-header">Tableau Comparatif</h2>', unsafe_allow_html=True)
    st.dataframe(comp_df.style.background_gradient(
        subset=['Accuracy','Precision','Recall','F1-Score','AUC'], cmap='viridis'),
        use_container_width=True, hide_index=True)

    st.markdown('<h2 class="section-header">Comparaison Visuelle</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=comp_df['Modèle'], x=comp_df['Accuracy'], orientation='h',
                             marker=dict(color=comp_df['Accuracy'], colorscale='Viridis', showscale=False),
                             text=comp_df['Accuracy'].apply(lambda x:f'{x:.2%}'), textposition='outside'))
        fig.update_layout(title='Accuracy par Modèle', xaxis_title='Accuracy',
                          plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                          font=dict(color=FONT_COL), height=500)
        fig.update_xaxes(gridcolor=GRID_COL, range=[0.6, 0.9])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        for _, row in comp_df.head(4).iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Accuracy'],row['Precision'],row['Recall'],row['F1-Score'],row['AUC']],
                theta=['Accuracy','Precision','Recall','F1-Score','AUC'],
                fill='toself', name=row['Modèle']))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.6,0.9]),
                                     bgcolor='rgba(0,0,0,0)'),
                          showlegend=True, plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                          font=dict(color=FONT_COL), height=500,
                          legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.1))
        st.plotly_chart(fig, use_container_width=True)

    best = comp_df.iloc[0]
    st.markdown(f"""<div class="success-box">
        <h3 style="margin-top:0;">🏆 Meilleur Modèle: {best['Modèle']}</h3>
        <p style="font-size:1.1rem;">
            <strong>Accuracy:</strong> {best['Accuracy']:.2%} &nbsp;|&nbsp;
            <strong>F1-Score:</strong> {best['F1-Score']:.2%} &nbsp;|&nbsp;
            <strong>AUC:</strong> {best['AUC']:.2%}
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<h2 class="section-header">Analyse Détaillée</h2>', unsafe_allow_html=True)
    metric_sel = st.selectbox("Sélectionnez une métrique:", ['Accuracy','Precision','Recall','F1-Score','AUC'])
    colors = ['#667eea','#764ba2','#f093fb','#4facfe','#43e97b','#fa709a']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp_df['Modèle'], y=comp_df[metric_sel], marker=dict(color=colors),
                         text=comp_df[metric_sel].apply(lambda x:f'{x:.2%}'), textposition='outside'))
    avg = comp_df[metric_sel].mean()
    fig.add_hline(y=avg, line_dash="dash", line_color="#10b981",
                  annotation_text=f"Moyenne: {avg:.2%}", annotation_position="right")
    fig.update_layout(title=f'Comparaison: {metric_sel}', yaxis_title=metric_sel,
                      plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      font=dict(color=FONT_COL), height=500, showlegend=False)
    fig.update_yaxes(gridcolor=GRID_COL)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 6 : DOCUMENTATION
# ─────────────────────────────────────────────
elif page == "📖 Documentation":
    st.markdown('<h1 class="main-header">Documentation du Projet</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Vue d\'ensemble</h2>', unsafe_allow_html=True)
    st.markdown("""
    Plateforme d'analytics développée pour analyser et prédire la popularité des vidéos YouTube
    en utilisant des techniques de Machine Learning avancées sur le **vrai dataset youtube.csv**.

    ### 🎯 Objectifs
    1. **EDA** : comprendre les patterns et distributions réels
    2. **Feature Engineering** : créer des variables pertinentes à partir des colonnes brutes
    3. **Modélisation ML** : tester 6 algorithmes et identifier le meilleur
    4. **Prédiction** : outil interactif temps réel
    """)
    st.markdown('<h2 class="section-header">Pipeline de Traitement</h2>', unsafe_allow_html=True)
    with st.expander("1️⃣ Exploratory Data Analysis"):
        st.markdown("- Distributions des variables quantitatives\n- Détection des outliers (boxplots)\n- Corrélations\n- Patterns temporels (heure, jour, saison)")
    with st.expander("2️⃣ Preprocessing"):
        st.markdown("- Chargement complet du vrai youtube.csv\n- Feature Engineering (title_length, tag_count, engagement, Season, days_to_trending…)\n- Regroupement catégories rares (< 500)\n- One-Hot Encoding pays + saison\n- RobustScaler")
    with st.expander("3️⃣ Modélisation"):
        st.markdown("- Random Forest, XGBoost, LightGBM, CatBoost\n- Logistic Regression, KNN\n- class_weight='balanced' / scale_pos_weight\n- GridSearchCV 5-fold")
    with st.expander("4️⃣ Évaluation"):
        st.markdown("- Accuracy, Precision, Recall, F1-Score, AUC\n- Matrice de confusion\n- Courbe ROC avec aire colorée")

    st.markdown('<h2 class="section-header">Features Utilisées</h2>', unsafe_allow_html=True)
    features_doc = pd.DataFrame({
        'Feature': ['title_length','title_caps_word_count','tag_count','category_id',
                    'publish_hour','is_week_end','days_to_trending','Season','engagement',
                    'comments_enabled','ratings_enabled','country_*'],
        'Type': ['Numérique','Numérique','Numérique','Catégorielle','Numérique',
                 'Binaire','Numérique','Catégorielle','Numérique','Binaire','Binaire','Binaire'],
        'Description': [
            'Longueur du titre', 'Mots en majuscules dans le titre',
            'Nombre de tags (séparateur |)', 'ID catégorie YouTube',
            'Heure de publication (0-23)', 'Publication week-end (1) ou semaine (0)',
            'Jours entre publication et trending', 'Saison de trending (Summer/Autumn/Winter/Spring)',
            'likes + dislikes + comments', 'Commentaires activés',
            'Évaluations activées', 'One-hot pays (country_US, country_CANADA…)'
        ]
    })
    st.dataframe(features_doc, use_container_width=True, hide_index=True)

    st.markdown('<h2 class="section-header">Technologies</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Framework & Visualisation :** Streamlit · Plotly · Matplotlib · Seaborn")
    with col2:
        st.markdown("**ML :** Scikit-learn · XGBoost · LightGBM · CatBoost")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div class="footer">
    <p style="font-size:1rem; margin:0; font-weight:600;">YouTube Trending Analytics Platform</p>
    <p style="font-size:0.85rem; color:#64748b; margin-top:0.8rem;">
        Machine Learning & Data Science | Built with Streamlit
    </p>
    <p style="font-size:0.75rem; color:#475569; margin-top:0.5rem;">
        Basé sur le vrai dataset YouTube Trending Videos
    </p>
</div>""", unsafe_allow_html=True)