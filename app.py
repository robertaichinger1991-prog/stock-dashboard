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
    
if st.sidebar.button("₿ Krypto Flows", use_container_width=True):
    st.session_state.page = "Krypto Flows"
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
    
    # --- TFA Formula Turnarounds ---
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
    
    # --- Classic Trend Followers ---
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
        st.write("Hier kannst du später deine echten Chart-Daten oder TradingView-Widgets anbinden.")
        
        # Simulierter Preis-Chart
        chart_data = pd.DataFrame(
            np.random.randn(50, 1).cumsum() + 100, 
            columns=[f'{ticker} Kursverlauf (Simulation)']
        )
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
# --- ANSICHT 3: KRYPTO FLOWS ---
# ==========================================
elif st.session_state.page == "Krypto Flows":
    st.title("Krypto ETF / ETP Flows")
    st.write("Tägliche Nettozuflüsse der institutionellen Produkte (in Mio. USD)")
    
    # Dropdown zur Auswahl der Kryptowährung
    selected_crypto = st.selectbox(
        "Wähle das gewünschte Krypto-Asset aus:",
        ["Bitcoin", "Ethereum", "Solana", "XRP", "Cardano", "Stellar"]
    )
    
    # Funktion zur Generierung von Mock-Daten basierend auf der Auswahl
    def get_crypto_flow_data(crypto_name):
        dates = pd.date_range(end=pd.Timestamp.today(), periods=7, freq="B").strftime("%Y-%m-%d")
        
        if crypto_name == "Bitcoin":
            return pd.DataFrame({
                "Datum": dates,
                "IBIT (BlackRock)": [120.5, 340.2, 55.0, -10.5, 210.0, 450.3, 105.1],
                "FBTC (Fidelity)": [85.0, 110.5, 20.1, -5.0, 88.0, 150.2, 70.0],
                "ARKB (Ark/21Shares)": [20.0, 40.5, 10.0, -15.0, 30.0, 50.0, 15.0],
                "GBTC (Grayscale)": [-45.0, -20.5, -60.0, -110.2, -30.0, -15.0, -5.0]
            }).set_index("Datum")
            
        elif crypto_name == "Ethereum":
            return pd.DataFrame({
                "Datum": dates,
                "ETHA (BlackRock)": [45.2, 80.1, -5.5, 12.0, 60.5, 110.0, 35.5],
                "FETH (Fidelity)": [20.0, 35.5, 10.2, -2.0, 25.0, 45.2, 18.0],
                "ETHW (Bitwise)": [5.5, 12.0, 2.1, 0.5, 8.0, 15.0, 4.2],
                "ETHE (Grayscale)": [-15.0, -25.5, -30.0, -10.5, -40.0, -20.0, -12.5]
            }).set_index("Datum")
            
        elif crypto_name == "Solana":
            return pd.DataFrame({
                "Datum": dates,
                "GSOL (Grayscale Trust)": [2.5, 5.0, -1.2, 3.5, 8.0, 12.5, 6.0],
                "ASOL (21Shares ETP)": [1.2, 2.5, 0.5, -0.5, 3.0, 4.5, 1.8],
                "CoinShares Physical SOL": [0.8, 1.5, 0.2, 0.0, 1.5, 2.0, 0.5]
            }).set_index("Datum")
            
        elif crypto_name == "XRP":
            return pd.DataFrame({
                "Datum": dates,
                "AXRP (21Shares ETP)": [1.5, 3.2, -0.5, 1.0, 4.5, 2.1, 1.8],
                "CoinShares Physical XRP": [0.5, 1.2, 0.0, -0.2, 1.5, 0.8, 0.4]
            }).set_index("Datum")
            
        elif crypto_name == "Cardano":
            return pd.DataFrame({
                "Datum": dates,
                "Grayscale ADA Trust": [0.8, 1.5, -0.2, 0.5, 2.0, 1.2, 0.9],
                "AADA (21Shares ETP)": [0.3, 0.8, 0.1, -0.1, 0.9, 0.5, 0.3]
            }).set_index("Datum")
            
        elif crypto_name == "Stellar":
            return pd.DataFrame({
                "Datum": dates,
                "Grayscale XLM Trust": [0.2, 0.5, -0.1, 0.2, 0.8, 0.4, 0.3],
                "AXLM (21Shares ETP)": [0.1, 0.3, 0.0, 0.0, 0.4, 0.2, 0.1]
            }).set_index("Datum")

    # Daten abrufen
    df_flows = get_crypto_flow_data(selected_crypto)
    
    # 1. METRIKEN (VORSCHAU DER LETZTEN WERTE)
    st.subheader(f"Heutige Flows: {selected_crypto}")
    
    # Erstellt automatisch so viele Spalten, wie es Produkte im DataFrame gibt
    cols = st.columns(len(df_flows.columns))
    latest_data = df_flows.iloc[-1]
    
    for i, col_name in enumerate(df_flows.columns):
        val = latest_data[col_name]
        # Formatierung: Pluszeichen für positive Werte, Farbe anpassen
        delta_color = "normal" if val >= 0 else "inverse"
        sign = "+" if val > 0 else ""
        cols[i].metric(col_name, f"{sign}{val:.1f}M", f"{val:.1f}M", delta_color=delta_color)
    
    # 2. BALKENDIAGRAMM VERLAUF
    st.subheader("Verlauf der letzten 7 Tage")
    st.bar_chart(df_flows)
    
    # 3. DATENTABELLE
    st.subheader("Datenübersicht")
    def color_negative_red(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'
        
    st.dataframe(df_flows.style.map(color_negative_red), use_container_width=True)
