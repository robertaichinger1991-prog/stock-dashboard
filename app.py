import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Stock Screener Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown('''
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00ff00;
    }
    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
    }
    .tfa-badge {
        background-color: #ff9800;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .trend-badge {
        background-color: #2196f3;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .krypto-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .krypto-negative {
        color: #ff3333;
        font-weight: bold;
    }
</style>
''', unsafe_allow_html=True)


# Data loading and generation
@st.cache_data
def load_stock_data():
    # Fallback data if file not found, mirroring the generated JSON
    data = [
        {"ticker": "MSFT", "name": "Microsoft Corp.", "category": "Trendfolge", "signal": "Kauf (Strong Uptrend)", "price_eur": 415.50, "sma_50": 408.20, "sma_200": 395.10, "rsi_14": 58.4, "macd_signal": "Bullish Crossover", "quality_score": 9},
        {"ticker": "NVDA", "name": "NVIDIA Corporation", "category": "Trendfolge", "signal": "Kauf (All-Time High Momentum)", "price_eur": 128.80, "sma_50": 122.40, "sma_200": 110.15, "rsi_14": 64.2, "macd_signal": "Bullish Expansion", "quality_score": 10},
        {"ticker": "COST", "name": "Costco Wholesale", "category": "Trendfolge", "signal": "Kauf (Konsolidierung)", "price_eur": 825.10, "sma_50": 810.00, "sma_200": 775.50, "rsi_14": 56.1, "macd_signal": "Bullish", "quality_score": 8},
        {"ticker": "ENR.DE", "name": "Siemens Energy AG", "category": "Trendfolge", "signal": "Kauf (Sektor-Leader)", "price_eur": 38.40, "sma_50": 35.80, "sma_200": 29.20, "rsi_14": 62.8, "macd_signal": "Bullish Continuation", "quality_score": 8},
        {"ticker": "ASML", "name": "ASML Holding N.V.", "category": "Trendfolge", "signal": "Kauf (Bodenbestätigung)", "price_eur": 672.30, "sma_50": 655.00, "sma_200": 640.10, "rsi_14": 53.7, "macd_signal": "Bullish Turnaround", "quality_score": 9},
        {"ticker": "IFX.DE", "name": "Infineon Tech", "category": "Trendfolge", "signal": "Kauf (Erholung)", "price_eur": 31.85, "sma_50": 30.20, "sma_200": 29.80, "rsi_14": 55.4, "macd_signal": "Bullish", "quality_score": 7},
        {"ticker": "RKLB", "name": "Rocket Lab USA", "category": "Trendfolge", "signal": "Kauf (Growth)", "price_eur": 18.20, "sma_50": 15.40, "sma_200": 11.20, "rsi_14": 67.5, "macd_signal": "Strong Bullish", "quality_score": 7},
        {"ticker": "NDX1.DE", "name": "Nordex SE", "category": "Trendfolge", "signal": "Kauf (Trend)", "price_eur": 14.65, "sma_50": 13.80, "sma_200": 12.90, "rsi_14": 57.2, "macd_signal": "Bullish", "quality_score": 7},
        {"ticker": "SBUX", "name": "Starbucks Corp.", "category": "Trendfolge", "signal": "Kauf (Turnaround)", "price_eur": 88.40, "sma_50": 85.10, "sma_200": 82.30, "rsi_14": 54.9, "macd_signal": "Bullish", "quality_score": 8},
        {"ticker": "CRWV", "name": "CoreWeave Inc.", "category": "Trendfolge", "signal": "Kauf (AI Play)", "price_eur": 72.10, "sma_50": 64.50, "sma_200": 55.00, "rsi_14": 61.3, "macd_signal": "Bullish Momentum", "quality_score": 8},
        {"ticker": "NKE", "name": "NIKE Inc.", "category": "TFA Turnaround", "signal": "TFA Signal (Boden)", "price_eur": 74.30, "sma_50": 71.80, "sma_200": 78.40, "rsi_14": 52.6, "macd_signal": "Bullish Divergence", "quality_score": 8},
        {"ticker": "EVT.DE", "name": "Evotec SE", "category": "TFA Turnaround", "signal": "TFA Signal (M&A)", "price_eur": 9.45, "sma_50": 8.60, "sma_200": 10.80, "rsi_14": 58.1, "macd_signal": "Bullish Crossover", "quality_score": 7},
        {"ticker": "TKMS.DE", "name": "thyssenkrupp marine", "category": "TFA Turnaround", "signal": "TFA Signal (Spin-off)", "price_eur": 3.78, "sma_50": 3.42, "sma_200": 3.65, "rsi_14": 56.4, "macd_signal": "Bullish Expansion", "quality_score": 7},
        {"ticker": "BAYN.DE", "name": "Bayer AG", "category": "TFA Turnaround", "signal": "TFA Signal (Deep Value)", "price_eur": 21.15, "sma_50": 19.80, "sma_200": 24.50, "rsi_14": 51.9, "macd_signal": "Bullish Divergence", "quality_score": 7},
        {"ticker": "MU", "name": "Micron Tech", "category": "TFA Turnaround", "signal": "TFA Signal (Rebound)", "price_eur": 98.60, "sma_50": 93.20, "sma_200": 105.10, "rsi_14": 54.3, "macd_signal": "Bullish Crossover", "quality_score": 9}
    ]
    return pd.DataFrame(data)

@st.cache_data
def load_crypto_flows():
    # Mock data for ETF in/outflows in USD millions
    return pd.DataFrame({
        'Date': pd.date_range(end=datetime.today(), periods=7).strftime('%Y-%m-%d'),
        'BTC_Flows': [120, 85, -45, 210, 340, -10, 150],
        'ETH_Flows': [30, 15, -10, 50, 80, 5, 45],
        'SOL_Flows': [10, -5, -2, 15, 25, -8, 12]
    })


# --- Session State for Navigation ---
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'Dashboard'
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
menu_options = ['Dashboard', 'Scanner & Filter', 'Krypto ETF Flows', 'Portfolio', 'Einzelaktien-Analyse']
current_index = menu_options.index(st.session_state.current_view) if st.session_state.current_view in menu_options else 0

selected_view = st.sidebar.radio(
    "Gehe zu:",
    menu_options,
    index=current_index
)

# Update session state if sidebar changes
if selected_view != st.session_state.current_view:
    st.session_state.current_view = selected_view
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Stock Screener Pro v2.1\nMit TFA-Turnaround & Krypto Flows")


# Load data
df = load_stock_data()
crypto_df = load_crypto_flows()

# ==========================================
# VIEW: Dashboard
# ==========================================
if st.session_state.current_view == 'Dashboard':
    st.title("📈 Stock Screener Pro - Dashboard")
    st.markdown(f"**Scan Datum:** {datetime.today().strftime('%Y-%m-%d')}")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{{len(df)}}</div><div class="metric-label">Aktien gescannt</div></div>', unsafe_allow_html=True)
    with col2:
        trend_count = len(df[df['category'] == 'Trendfolge'])
        st.markdown(f'<div class="metric-card"><div class="metric-value">{{trend_count}}</div><div class="metric-label">Trendfolger (Long)</div></div>', unsafe_allow_html=True)
    with col3:
        tfa_count = len(df[df['category'] == 'TFA Turnaround'])
        st.markdown(f'<div class="metric-card"><div class="metric-value">{{tfa_count}}</div><div class="metric-label">TFA Turnarounds</div></div>', unsafe_allow_html=True)
    with col4:
        avg_score = round(df['quality_score'].mean(), 1)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{{avg_score}}/10</div><div class="metric-label">Durchschn. Quality Score</div></div>', unsafe_allow_html=True)

    # Market Overview Chart (Scatter Plot)
    st.subheader("Marktübersicht (RSI vs Quality Score)")
    fig = px.scatter(
        df, 
        x="rsi_14", 
        y="quality_score", 
        color="category",
        hover_name="ticker",
        hover_data=["price_eur", "signal"],
        color_discrete_map={"Trendfolge": "#2196f3", "TFA Turnaround": "#ff9800"},
        size="price_eur",
        size_max=30
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.add_vline(x=50, line_width=1, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# VIEW: Scanner & Filter
# ==========================================
elif st.session_state.current_view == 'Scanner & Filter':
    st.title("🔍 Scanner & Filter")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        cat_filter = st.multiselect("Kategorie", options=df['category'].unique(), default=df['category'].unique())
    with col_f2:
        min_score = st.slider("Min. Quality Score", 0, 10, 7)
    with col_f3:
        rsi_range = st.slider("RSI Bereich", 0, 100, (30, 80))
        
    # Apply filters
    filtered_df = df[
        (df['category'].isin(cat_filter)) & 
        (df['quality_score'] >= min_score) &
        (df['rsi_14'] >= rsi_range[0]) & 
        (df['rsi_14'] <= rsi_range[1])
    ]
    
    st.write(f"**Gefundene Aktien:** {len(filtered_df)}")
    
    for index, row in filtered_df.iterrows():
        with st.container():
            col_a, col_b, col_c, col_d, col_e = st.columns([1, 2, 1, 1, 1])
            
            with col_a:
                st.markdown(f"**{row['ticker']}**")
                if row['category'] == 'TFA Turnaround':
                    st.markdown('<span class="tfa-badge">TFA</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="trend-badge">TREND</span>', unsafe_allow_html=True)
                    
            with col_b:
                st.write(row['name'])
                st.caption(row['signal'])
                
            with col_c:
                st.write(f"€ {row['price_eur']:.2f}")
                
            with col_d:
                st.write(f"RSI: {row['rsi_14']}")
                
            with col_e:
                if st.button("Analyse", key=f"btn_{row['ticker']}"):
                    st.session_state.selected_ticker = row['ticker']
                    st.session_state.current_view = 'Einzelaktien-Analyse'
                    st.rerun()
            st.markdown("---")

# ==========================================
# VIEW: Krypto ETF Flows
# ==========================================
elif st.session_state.current_view == 'Krypto ETF Flows':
    st.title("₿ Krypto ETF In/Outflows")
    st.markdown("Beobachte die täglichen Zu- und Abflüsse der großen Krypto-ETFs in USD (Millionen).")
    
    btc_total = crypto_df['BTC_Flows'].sum()
    eth_total = crypto_df['ETH_Flows'].sum()
    sol_total = crypto_df['SOL_Flows'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    def format_flow(val):
        color_class = "krypto-positive" if val >= 0 else "krypto-negative"
        sign = "+" if val >= 0 else ""
        return f'<span class="{{color_class}}">{{sign}}{{val}} Mio. $</span>'

    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Bitcoin ETFs (7d)</div>
            <div class="metric-value">{{format_flow(btc_total)}}</div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Ethereum ETFs (7d)</div>
            <div class="metric-value">{{format_flow(eth_total)}}</div>
        </div>
        ''', unsafe_allow_html=True)
        
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Solana ETFs (7d)</div>
            <div class="metric-value">{{format_flow(sol_total)}}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.subheader("Verlauf der letzten 7 Tage")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=crypto_df['Date'], y=crypto_df['BTC_Flows'], name='BTC Flows', marker_color='#f7931a'))
    fig.add_trace(go.Bar(x=crypto_df['Date'], y=crypto_df['ETH_Flows'], name='ETH Flows', marker_color='#627eea'))
    fig.add_trace(go.Bar(x=crypto_df['Date'], y=crypto_df['SOL_Flows'], name='SOL Flows', marker_color='#14f195'))
    
    fig.update_layout(
        barmode='group',
        xaxis_title='Datum',
        yaxis_title='Flows (Mio. USD)',
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(crypto_df.set_index('Date'), use_container_width=True)

# ==========================================
# VIEW: Portfolio
# ==========================================
elif st.session_state.current_view == 'Portfolio':
    st.title("💼 Portfolio Manager")
    st.info("Hier kannst du deine aktuellen Bestände tracken und mit den Scanner-Signalen abgleichen.")

# ==========================================
# VIEW: Einzelaktien-Analyse
# ==========================================
elif st.session_state.current_view == 'Einzelaktien-Analyse':
    st.title("📊 Einzelaktien-Analyse")
    
    tickers = df['ticker'].tolist()
    
    default_idx = 0
    if st.session_state.selected_ticker in tickers:
        default_idx = tickers.index(st.session_state.selected_ticker)
        
    selected = st.selectbox("Aktie auswählen", options=tickers, index=default_idx)
    
    st.session_state.selected_ticker = selected
    stock = df[df['ticker'] == selected].iloc[0]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{stock['name']} ({stock['ticker']})")
        
        if stock['category'] == 'TFA Turnaround':
            st.markdown('<span class="tfa-badge">TFA Formel Signal aktiv</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="trend-badge">Trendfolge Setup</span>', unsafe_allow_html=True)
            
        st.write(f"**Aktuelles Signal:** {stock['signal']}")
        
        fig_score = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = stock['quality_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Quality / TFA Score"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "#00ff00" if stock['quality_score'] >= 8 else "#ffeb3b"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgray"},
                    {'range': [5, 7], 'color': "gray"}
                ]
            }
        ))
        fig_score.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_score, use_container_width=True)
        
    with col2:
        st.subheader("Technik & Einstieg")
        st.markdown(f"**Aktueller Preis:** € {stock['price_eur']:.2f}")
        
        st.write("---")
        st.write(f"**RSI (14):** {stock['rsi_14']}")
        st.write(f"**MACD:** {stock['macd_signal']}")
        
        st.write("---")
        st.write(f"**SMA 50:** € {stock['sma_50']:.2f}")
        st.write(f"**SMA 200:** € {stock['sma_200']:.2f}")
        
    if st.button("⬅ Zurück zum Scanner"):
        st.session_state.current_view = 'Scanner & Filter'
        st.session_state.selected_ticker = None
        st.rerun()
