import os
import json
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==========================================
# 1. KONFIGURATION & UNIVERSE PRESETS
# ==========================================
st.set_page_config(
    page_title="Stock Screener Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRESETS = {
    "DAX 40": ["SAP.DE", "SIE.DE", "ALV.DE", "AIR.DE", "MBG.DE", "BMW.DE", "BAS.DE", "DTE.DE"],
    "S&P 500 Top": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B"],
    "European Growth": ["ASML.AS", "MC.PA", "OR.PA", "NESN.SW", "NOVN.SW", "ROG.SW"],
    "Cryptos & Metals": ["BTC-USD", "ETH-USD", "SOL-USD", "GC=F", "SI=F"]
}

SECTOR_BENCHMARKS = {
    "Technology": {"KGV": 28.0, "KBV": 6.0, "EV_EBITDA": 18.0, "ROE": 0.20, "Debt_Equity": 0.8},
    "Financial Services": {"KGV": 12.0, "KBV": 1.2, "EV_EBITDA": 10.0, "ROE": 0.12, "Debt_Equity": 1.5},
    "Healthcare": {"KGV": 20.0, "KBV": 3.5, "EV_EBITDA": 14.0, "ROE": 0.15, "Debt_Equity": 0.6},
    "Consumer Cyclical": {"KGV": 22.0, "KBV": 4.0, "EV_EBITDA": 13.0, "ROE": 0.18, "Debt_Equity": 0.9},
    "Industrials": {"KGV": 18.0, "KBV": 2.8, "EV_EBITDA": 11.0, "ROE": 0.14, "Debt_Equity": 0.7},
    "Default": {"KGV": 20.0, "KBV": 3.0, "EV_EBITDA": 12.0, "ROE": 0.15, "Debt_Equity": 0.8}
}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlists.json")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

# ==========================================
# 2. HILFSFUNKTIONEN & DATENVERARBEITUNG
# ==========================================
def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

@st.cache_data(ttl=86400)
def resolve_ticker(query: str) -> str:
    query = query.strip().upper()
    if not query:
        return "AAPL"
    try:
        search = yf.Search(query, max_results=3)
        if search.quotes and len(search.quotes) > 0:
            return search.quotes[0].get("symbol", query)
    except Exception:
        pass
    return query

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, period: str = "2y"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if df.empty:
        return None, None
    
    # Flatten MultiIndex falls vorhanden
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    info = ticker.info or {}
    
    # Technische Indikatoren berechnen
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    return df, info

def compute_signal_status(df: pd.DataFrame):
    if df is None or len(df) < 200:
        return {"Status": "Unzureichende Daten", "Color": "gray"}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    golden_cross = (prev["SMA50"] <= prev["SMA200"]) and (last["SMA50"] > last["SMA200"])
    death_cross = (prev["SMA50"] >= prev["SMA200"]) and (last["SMA50"] < last["SMA200"])
    
    bullish = last["Close"] > last["SMA50"] > last["SMA200"]
    bearish = last["Close"] < last["SMA50"] < last["SMA200"]
    
    if golden_cross:
        return {"Status": "🚀 Frischer Golden Cross!", "Color": "green"}
    elif death_cross:
        return {"Status": "⚠️ Frischer Death Cross!", "Color": "red"}
    elif bullish:
        if last["RSI"] > 70:
            return {"Status": "📈 Starker Aufwärtstrend (Überkauft)", "Color": "orange"}
        return {"Status": "🟢 Intakter Aufwärtstrend", "Color": "green"}
    elif bearish:
        if last["RSI"] < 30:
            return {"Status": "📉 Starker Abwärtstrend (Überverkauft)", "Color": "orange"}
        return {"Status": "🔴 Intakter Abwärtstrend", "Color": "red"}
    else:
        return {"Status": "➡️ Seitwärts / Neutral", "Color": "gray"}

def evaluate_fundamentals(info: dict):
    sector = info.get("sector", "Default")
    bm = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["Default"])
    
    pe = info.get("trailingPE")
    pb = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    
    evals = {}
    if pe:
        evals["KGV"] = {"Wert": f"{pe:.1f}", "Status": "günstig" if pe < bm["KGV"] else "teuer"}
    else:
        evals["KGV"] = {"Wert": "N/A", "Status": "neutral"}
        
    if pb:
        evals["KBV"] = {"Wert": f"{pb:.1f}", "Status": "günstig" if pb < bm["KBV"] else "teuer"}
    else:
        evals["KBV"] = {"Wert": "N/A", "Status": "neutral"}
        
    if roe:
        evals["ROE"] = {"Wert": f"{roe*100:.1f}%", "Status": "überdurchschnittlich" if roe > bm["ROE"] else "unterdurchschnittlich"}
    else:
        evals["ROE"] = {"Wert": "N/A", "Status": "neutral"}
        
    return evals

def valuation_score(rel: dict) -> int:
    score = 0
    for info in rel.values():
        status = info.get("Status", "")
        if "günstig" in status or "überdurchschnittlich" in status:
            score += 1
        elif "teuer" in status or "unterdurchschnittlich" in status:
            score -= 1
    return score

# ==========================================
# 3. STREAMLIT BENUTZEROBERFLÄCHE
# ==========================================
st.title("📈 Stock Screener Pro")

# Sidebar - Navigation & Ticker-Auswahl
st.sidebar.header("🔍 Steuerung")

preset_choice = st.sidebar.selectbox("Universe Presets", list(PRESETS.keys()))
selected_preset_ticker = st.sidebar.selectbox("Preset Aktie wählen", PRESETS[preset_choice])

custom_ticker = st.sidebar.text_input("Oder Ticker manuell suchen", "")

if custom_ticker.strip():
    ticker_symbol = resolve_ticker(custom_ticker)
else:
    ticker_symbol = selected_preset_ticker

st.sidebar.markdown("---")
view_mode = st.sidebar.radio("Modus", ["Einzelanalyse", "Screener / Übersicht", "Portfolio Manager"])

# ------------------------------------------
# MODUS 1: EINZELANALYSE
# ------------------------------------------
if view_mode == "Einzelanalyse":
    st.subheader(f"Analyse für: **{ticker_symbol}**")
    
    with st.spinner("Lade Marktdaten..."):
        df, info = fetch_stock_data(ticker_symbol)
        
    if df is None or df.empty:
        st.error(f"Keine Daten für Ticker **{ticker_symbol}** gefunden.")
    else:
        # Kennzahlen-Karten
        col1, col2, col3, col4 = st.columns(4)
        last_price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2]
        change_pct = ((last_price - prev_price) / prev_price) * 100
        
        signal = compute_signal_status(df)
        
        col1.metric("Aktueller Kurs", f"{last_price:.2f} {info.get('currency', 'USD')}", f"{change_pct:+.2f}%")
        col2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
        col3.metric("52-Wochen Hoch", f"{df['High'].max():.2f}")
        col4.metric("Signal Status", signal["Status"])

        # Plotly Interaktiver Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Kurs"
        ), row=1, col=1)
        
        # SMAs
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], line=dict(color="orange", width=1.5), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], line=dict(color="blue", width=1.5), name="SMA 200"), row=1, col=1)
        
        # RSI Subplot
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="purple", width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Fundamentale Bewertung & Benchmarking
        st.markdown("### 📊 Fundamentalanalyse & Branchenvergleich")
        evals = evaluate_fundamentals(info)
        score = valuation_score(evals)
        
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            f_df = pd.DataFrame.from_dict(evals, orient="index")
            st.table(f_df)
            
        with col_f2:
            st.markdown(f"**Sektor:** {info.get('sector', 'N/A')}")
            st.markdown(f"**KGV (P/E):** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**Bewertungs-Score:** `{score:+d}`")
            if score > 0:
                st.success("Aktie ist im Branchenvergleich günstig/stark bewertet.")
            elif score < 0:
                st.warning("Aktie ist im Branchenvergleich teuer/schwächer bewertet.")
            else:
                st.info("Aktie ist im Branchendurchschnitt bewertet.")

# ------------------------------------------
# MODUS 2: SCREENER / ÜBERSICHT
# ------------------------------------------
elif view_mode == "Screener / Übersicht":
    st.subheader(f"Screener Übersicht: **{preset_choice}**")
    tickers = PRESETS[preset_choice]
    
    screener_data = []
    
    with st.spinner("Scanne Ticker im Universum..."):
        for t in tickers:
            df_t, info_t = fetch_stock_data(t, period="1y")
            if df_t is not None and not df_t.empty:
                last_p = df_t["Close"].iloc[-1]
                rsi = df_t["RSI"].iloc[-1]
                sig = compute_signal_status(df_t)
                evals_t = evaluate_fundamentals(info_t)
                v_score = valuation_score(evals_t)
                
                screener_data.append({
                    "Symbol": t,
                    "Name": info_t.get("shortName", t),
                    "Kurs": f"{last_p:.2f} {info_t.get('currency', '')}",
                    "RSI (14)": round(rsi, 1),
                    "Signal": sig["Status"],
                    "Fundamental Score": v_score,
                    "KGV": info_t.get("trailingPE", "N/A")
                })
                
    res_df = pd.DataFrame(screener_data)
    st.dataframe(res_df, use_container_width=True)

# ------------------------------------------
# MODUS 3: PORTFOLIO MANAGER
# ------------------------------------------
elif view_mode == "Portfolio Manager":
    st.subheader("💼 Lokales Portfolio")
    
    portfolio = load_json(PORTFOLIO_FILE, [])
    
    # Formular zum Hinzufügen von Positionen
    with st.expander("➕ Neue Position hinzufügen"):
        with st.form("add_position"):
            p_symbol = st.text_input("Ticker Symbol (z.B. AAPL, SAP.DE)").upper()
            p_shares = st.number_input("Anzahl Aktien", min_value=0.001, value=1.0)
            p_buy_price = st.number_input("Kaufkurs", min_value=0.01, value=100.0)
            submitted = st.form_submit_button("Hinzufügen")
            
            if submitted and p_symbol:
                portfolio.append({
                    "symbol": p_symbol,
                    "shares": p_shares,
                    "buy_price": p_buy_price
                })
                save_json(PORTFOLIO_FILE, portfolio)
                st.success(f"{p_symbol} zum Portfolio hinzugefügt!")
                st.rerun()

    # Portfolio Auswertung
    if portfolio:
        p_data = []
        total_val = 0.0
        total_profit = 0.0
        
        for pos in portfolio:
            df_p, info_p = fetch_stock_data(pos["symbol"], period="5d")
            if df_p is not None and not df_p.empty:
                cur_price = df_p["Close"].iloc[-1]
                market_val = cur_price * pos["shares"]
                invested = pos["buy_price"] * pos["shares"]
                pnl = market_val - invested
                pnl_pct = (pnl / invested) * 100
                
                total_val += market_val
                total_profit += pnl
                
                p_data.append({
                    "Symbol": pos["symbol"],
                    "Anteile": pos["shares"],
                    "Kaufkurs": f"{pos['buy_price']:.2f}",
                    "Aktueller Kurs": f"{cur_price:.2f}",
                    "Marktwert": f"{market_val:.2f}",
                    "Gewinn/Verlust": f"{pnl:+.2f} ({pnl_pct:+.1f}%)"
                })
        
        st.metric("Gesamtwert Portfolio", f"{total_val:.2f} USD", f"{total_profit:+.2f} USD Gesamt G/V")
        st.table(pd.DataFrame(p_data))
        
        if st.button("Portfolio leeren"):
            save_json(PORTFOLIO_FILE, [])
            st.rerun()
    else:
        st.info("Dein Portfolio ist derzeit leer.")
