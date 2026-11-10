import streamlit as st
import pandas as pd
import numpy as np

# --- 1. SEITEN-KONFIGURATION ---
st.set_page_config(page_title="Trading Dashboard", layout="wide")

# --- 2. SESSION STATE INITIALISIERUNG ---
if 'page' not in st.session_state:
    st.session_state.page = "Scanner"
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")

if st.sidebar.button("📊 Scanner", use_container_width=True):
    st.session_state.page = "Scanner"
    st.rerun()
    
if st.sidebar.button("🔎 Einzelanalyse", use_container_width=True):
    st.session_state.page = "Einzelanalyse"
    st.rerun()
    
if st.sidebar.button("₿ Krypto ETF Flows", use_container_width=True):
    st.session_state.page = "Krypto ETF Flows"
    st.rerun()

st.sidebar.markdown("---")
if st.session_state.selected_ticker:
    st.sidebar.success(f"Aktuell ausgewählt:\n**{st.session_state.selected_ticker}**")


# ==========================================
# --- ANSICHT 1: SCANNER ---
# ==========================================
if st.session_state.page == "Scanner":
    st.title("Scanner (Turnaround & Trend)")
    st.write("Wähle eine Aktie aus, um direkt zur Einzelanalyse zu springen.")
    
    st.subheader("TFA-Formula Turnaround Signale")
    turnarounds = ["NKE", "EVT", "TKMS", "BAYN", "MU"]
    
    cols = st.columns(5)
    for i, ticker in enumerate(turnarounds):
        with cols[i % 5]:
            st.info(ticker)
            if st.button(f"Analyse {ticker}", key=f"btn_{ticker}"):
                st.session_state.selected_ticker = ticker
                st.session_state.page = "Einzelanalyse"
                st.rerun()
                
    st.markdown("---")
    
    st.subheader("Classic Trend Followers")
    trends = ["MSFT", "ENR", "COST", "NDX1", "SBUX", "NVDA", "IFX", "RKLB", "ASML"]
    cols_t = st.columns(5)
    for i, ticker in enumerate(trends):
        with cols_t[i % 5]:
            st.success(ticker)
            if st.button(f"Analyse {ticker}", key=f"btn_{ticker}"):
                st.session_state.selected_ticker = ticker
                st.session_state.page = "Einzelanalyse"
                st.rerun()


# ==========================================
# --- ANSICHT 2: EINZELANALYSE ---
# ==========================================
elif st.session_state.page == "Einzelanalyse":
    st.title("Einzelanalyse")
    
    ticker = st.session_state.selected_ticker
    
    if ticker:
        st.header(f"Technische Analyse: {ticker}")
        st.write("Hier kannst du später deine echten Chart-Daten oder TradingView-Widgets einbinden.")
        
        chart_data = pd.DataFrame(np.random.randn(20, 1), columns=['Preisverlauf'])
        st.line_chart(chart_data)
        
        if st.button("⬅️ Zurück zum Scanner"):
            st.session_state.page = "Scanner"
            st.rerun()
    else:
        st.warning("Es wurde noch keine Aktie ausgewählt.")
        if st.button("Zum Scanner wechseln"):
            st.session_state.page = "Scanner"
            st.rerun()


# ==========================================
# --- ANSICHT 3: KRYPTO ETF FLOWS ---
# ==========================================
elif st.session_state.page == "Krypto ETF Flows":
    st.title("Krypto ETF In- und Outflows")
    st.write("Tägliche Nettozuflüsse der wichtigsten Bitcoin Spot ETFs (in Mio. USD)")
    
    mock_data = {
        "Datum": pd.date_range(end=pd.Timestamp.today(), periods=7, freq="B").strftime("%Y-%m-%d"),
        "IBIT (BlackRock)": [120.5, 340.2, 55.0, -10.5, 210.0, 450.3, 105.1],
        "FBTC (Fidelity)": [85.0, 110.5, 20.1, -5.0, 88.0, 150.2, 70.0],
        "ARKB (Ark/21Shares)": [20.0, 40.5, 10.0, -15.0, 30.0, 50.0, 15.0],
        "GBTC (Grayscale)": [-45.0, -20.5, -60.0, -110.2, -30.0, -15.0, -5.0]
    }
    df_flows = pd.DataFrame(mock_data).set_index("Datum")
    
    st.subheader("Heutige Flows (Vorschau)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IBIT (BlackRock)", "+105.1M", "105.1M", delta_color="normal")
    col2.metric("FBTC (Fidelity)", "+70.0M", "70.0M", delta_color="normal")
    col3.metric("ARKB (Ark)", "+15.0M", "15.0M", delta_color="normal")
    col4.metric("GBTC (Grayscale)", "-5.0M", "-5.0M", delta_color="inverse")
    
    st.subheader("Verlauf der letzten 7 Tage")
    st.bar_chart(df_flows)
    
    st.subheader("Datenübersicht")
    
    def color_negative_red(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'
        
    st.dataframe(df_flows.style.map(color_negative_red), use_container_width=True)
