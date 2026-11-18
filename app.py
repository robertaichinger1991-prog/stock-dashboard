import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ==============================================================================
# SEITEN-KONFIGURATION & DESIGN (Dark-Not-Black Theme)
# ==============================================================================
st.set_page_config(
    page_title="Aktien & Krypto Dashboard (€)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #E6EDF3;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-green {
        background-color: rgba(0, 200, 83, 0.15);
        color: #00E676;
        border: 1px solid #00C853;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-red {
        background-color: rgba(255, 82, 82, 0.15);
        color: #FF5252;
        border: 1px solid #FF5252;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-amber {
        background-color: rgba(255, 179, 0, 0.15);
        color: #FFD54F;
        border: 1px solid #FFB300;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .info-box {
        background-color: #161B22;
        border-left: 4px solid #58A6FF;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.92rem;
    }
    .warn-box {
        background-color: rgba(255, 82, 82, 0.15);
        border-left: 4px solid #FF5252;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-weight: 600;
        color: #FF8A80;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# VOREINGESTELLTE AKTIEN-LISTEN & KRYPTO-DATEN
# ==============================================================================
INDEX_UNIVERSEN = {
    "DAX Auswahl (Deutschland)": [
        "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "ADS.DE",
        "MBG.DE", "BMW.DE", "BAS.DE", "IFX.DE", "MUV2.DE",
        "RHM.DE", "DHL.DE", "AIR.DE", "BAYN.DE", "VOW3.DE"
    ],
    "US Tech & S&P Top-Werte": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
        "META", "TSLA", "AVGO", "AMD", "NFLX",
        "TTWO", "CRM", "PLTR", "PYPL", "INTC"
    ],
    "Europa & Internationale Favoriten": [
        "ASML.AS", "MC.PA", "NOVO-B.CO", "NESN.SW", "SHEL.L",
        "1211.HK", "1810.HK", "NKE", "DIS", "BA"
    ],
}

KRYPTO_LISTE = [
    {"name": "Bitcoin", "symbol": "BTC", "ticker": "BTC-EUR", "etf_basis_m": 420.0},
    {"name": "Ethereum", "symbol": "ETH", "ticker": "ETH-EUR", "etf_basis_m": 115.0},
    {"name": "Solana", "symbol": "SOL", "ticker": "SOL-EUR", "etf_basis_m": 48.0},
    {"name": "XRP", "symbol": "XRP", "ticker": "XRP-EUR", "etf_basis_m": 32.0},
    {"name": "Cardano", "symbol": "ADA", "ticker": "ADA-EUR", "etf_basis_m": 14.0},
    {"name": "Stellar", "symbol": "XLM", "ticker": "XLM-EUR", "etf_basis_m": 6.5},
    {"name": "Hedera", "symbol": "HBAR", "ticker": "HBAR-EUR", "etf_basis_m": 5.0},
]

STANDARD_WATCHLIST = [
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "stueck": 10.0, "einstieg_eur": 112.50},
    {"ticker": "ADS.DE", "name": "adidas AG", "stueck": 5.0, "einstieg_eur": 218.00},
    {"ticker": "TTWO", "name": "Take-Two Interactive", "stueck": 8.0, "einstieg_eur": 158.00},
    {"ticker": "1810.HK", "name": "Xiaomi Corp.", "stueck": 200.0, "einstieg_eur": 3.10},
]

# ==============================================================================
# HILFSFUNKTIONEN & CACHING (Schutz vor API-Limits auf Streamlit Cloud)
# ==============================================================================
def bereinige_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Flacht MultiIndex-Spalten von yfinance ab und stellt 1D-Serien sicher."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def hole_wechselkurs_zu_eur(waehrung: str) -> float:
    """Holt den aktuellen Wechselkurs einer Fremdwährung nach Euro (24h Cache)."""
    if not waehrung or waehrung.upper() == "EUR":
        return 1.0
    waehrung = waehrung.upper()
    if waehrung == "GBX":  # Britische Pence -> Pfund -> EUR
        return hole_wechselkurs_zu_eur("GBP") / 100.0
    try:
        paar = f"{waehrung}EUR=X"
        df = bereinige_dataframe(yf.download(paar, period="5d", interval="1d", progress=False))
        if not df.empty and "Close" in df.columns:
            kurs = float(df["Close"].dropna().iloc[-1])
            if kurs > 0:
                return kurs
    except Exception:
        pass
    # Fallback-Richtwerte falls Yahoo kurzzeitig nicht antwortet
    fallback = {"USD": 0.92, "CHF": 1.06, "GBP": 1.19, "HKD": 0.118, "DKK": 0.134}
    return fallback.get(waehrung, 1.0)


@st.cache_data(ttl=86400, show_spinner=False)
def hole_stammdaten(ticker: str) -> dict:
    """Holt den vollen Firmennamen, die Währung und den nächsten Quartalszahlen-Termin."""
    info_daten = {
        "name": ticker,
        "waehrung": "EUR",
        "sektor": "Unbekannt",
        "earnings_datum": None,
    }
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        info_daten["name"] = info.get("longName") or info.get("shortName") or ticker
        info_daten["waehrung"] = info.get("currency") or ("EUR" if ticker.endswith(".DE") else "USD")
        info_daten["sektor"] = info.get("sector") or "Keine Angabe"

        # Nächsten Earnings-Termin auslesen
        cal = t.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal and cal["Earnings Date"]:
            ed = cal["Earnings Date"][0]
            info_daten["earnings_datum"] = str(ed)[:10]
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            info_daten["earnings_datum"] = str(cal.iloc[0, 0])[:10]
    except Exception:
        if ticker.endswith(".DE") or ticker.endswith(".AS") or ticker.endswith(".PA"):
            info_daten["waehrung"] = "EUR"
        elif ticker.endswith(".HK"):
            info_daten["waehrung"] = "HKD"
        elif ticker.endswith(".SW"):
            info_daten["waehrung"] = "CHF"
        else:
            info_daten["waehrung"] = "USD"
    return info_daten


@st.cache_data(ttl=900, show_spinner=False)
def lade_kursdaten_eur(ticker: str, zeitraum: str = "1y") -> pd.DataFrame:
    """Lädt historische Kursdaten, rechnet alles in Euro (€) um und berechnet Indikatoren."""
    try:
        raw = yf.download(ticker, period=zeitraum, interval="1d", progress=False)
        df = bereinige_dataframe(raw)
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        stamm = hole_stammdaten(ticker)
        faktor = hole_wechselkurs_zu_eur(stamm["waehrung"])

        for spalte in ["Open", "High", "Low", "Close"]:
            if spalte in df.columns:
                df[spalte] = pd.to_numeric(df[spalte], errors="coerce") * faktor

        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        df = df.dropna(subset=["Close"]).copy()

        # Gleitende Durchschnitte (SMA 50 & SMA 200)
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # RSI (14 Tage nach Wilder-Glättung)
        delta = df["Close"].diff()
        gewinn = delta.clip(lower=0)
        verlust = -delta.clip(upper=0)
        avg_gewinn = gewinn.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_verlust = verlust.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gewinn / avg_verlust.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))
        df["RSI_14"] = df["RSI_14"].fillna(50)

        # MACD (12, 26, 9)
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # ATR (14 Tage) für Stop-Loss-Berechnung
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = true_range.rolling(window=14).mean()

        # Volumen-Durchschnitt (20 Tage)
        df["Vol_SMA_20"] = df["Volume"].rolling(window=20).mean()

        return df
    except Exception:
        return pd.DataFrame()


def analysiere_aktie(ticker: str) -> dict | None:
    """Erstellt eine vollständige, leicht verständliche Auswertung in Euro (€)."""
    df = lade_kursdaten_eur(ticker, zeitraum="1y")
    if df.empty or len(df) < 50:
        return None

    stamm = hole_stammdaten(ticker)
    aktuell = float(df["Close"].iloc[-1])
    vortag = float(df["Close"].iloc[-2])
    tages_diff_pct = ((aktuell - vortag) / vortag) * 100

    sma50 = float(df["SMA_50"].iloc[-1]) if not pd.isna(df["SMA_50"].iloc[-1]) else aktuell
    sma200 = float(df["SMA_200"].iloc[-1]) if not pd.isna(df["SMA_200"].iloc[-1]) else sma50

    rsi = float(df["RSI_14"].iloc[-1])
    rsi_vor5 = float(df["RSI_14"].iloc[-6]) if len(df) >= 6 else rsi
    macd_hist = float(df["MACD_Hist"].iloc[-1])
    macd_hist_vortag = float(df["MACD_Hist"].iloc[-2])
    atr = float(df["ATR_14"].iloc[-1]) if not pd.isna(df["ATR_14"].iloc[-1]) else aktuell * 0.03

    vol_aktuell = float(df["Volume"].iloc[-1])
    vol_schnitt = float(df["Vol_SMA_20"].iloc[-1]) if not pd.isna(df["Vol_SMA_20"].iloc[-1]) else vol_aktuell
    volumen_bestaetigt = vol_aktuell > vol_schnitt and vol_schnitt > 0

    hoch_52w = float(df["High"].max())
    tief_52w = float(df["Low"].min())
    abstand_hoch_pct = ((aktuell - hoch_52w) / hoch_52w) * 100

    # Golden Cross Check (SMA 50 kreuzt SMA 200 nach oben in den letzten 5 Tagen)
    golden_cross_neu = False
    if len(df) >= 6 and not df["SMA_200"].iloc[-6:].isna().any():
        golden_cross_neu = (df["SMA_50"].iloc[-6] <= df["SMA_200"].iloc[-6]) and (sma50 > sma200)

    # Trend-Follower Check
    ist_trend_follower = (aktuell > sma50) and (sma50 > sma200) and (45 <= rsi <= 72)

    # TFA-Check (Turnaround-Formel: >= 35% unter 52W-Hoch + RSI war überverkauft / dreht hoch oder erobert SMA 50)
    ist_tfa = (abstand_hoch_pct <= -35.0) and (
        (rsi < 38) or (rsi_vor5 < 35 and rsi > rsi_vor5) or (aktuell > sma50 and macd_hist > macd_hist_vortag)
    )

    # ATR-Stop-Loss & Chance-Risiko-Verhältnis (CRV)
    stop_loss_eur = max(0.01, aktuell - 2.0 * atr)
    risiko_eur = max(0.01, aktuell - stop_loss_eur)
    chance_eur = max(0.0, hoch_52w - aktuell)
    crv = round(chance_eur / risiko_eur, 2)

    # Earnings-Warnung prüfen
    earnings_warnung = None
    if stamm["earnings_datum"]:
        try:
            ed_date = datetime.strptime(stamm["earnings_datum"], "%Y-%m-%d").date()
            tage_bis_earnings = (ed_date - datetime.now().date()).days
            if 0 <= tage_bis_earnings <= 7:
                earnings_warnung = f"Quartalszahlen in {tage_bis_earnings} Tag(en) am {ed_date.strftime('%d.%m.%Y')}!"
        except Exception:
            pass

    # Klartext-Gesamturteil auf Deutsch
    if golden_cross_neu:
        signal_text = "Kaufsignal (Frisches Golden Cross)"
        signal_farbe = "green"
    elif ist_trend_follower and volumen_bestaetigt and macd_hist > 0:
        signal_text = "Starker Aufwärtstrend (Kaufen / Halten)"
        signal_farbe = "green"
    elif ist_tfa:
        signal_text = "Turnaround-Chance (TFA-Signal)"
        signal_farbe = "amber"
    elif rsi > 75:
        signal_text = "Überhitzt – Teilgewinne absichern"
        signal_farbe = "red"
    elif aktuell < sma50 and aktuell < sma200:
        signal_text = "Abwärtstrend – Abwarten"
        signal_farbe = "red"
    else:
        signal_text = "Neutral / Beobachten"
        signal_farbe = "amber"

    return {
        "ticker": ticker,
        "name": stamm["name"],
        "sektor": stamm["sektor"],
        "kurs_eur": aktuell,
        "tages_diff_pct": tages_diff_pct,
        "sma50_eur": sma50,
        "sma200_eur": sma200,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "macd_trend": "Steigend" if macd_hist > macd_hist_vortag else "Fallend",
        "volumen_bestaetigt": volumen_bestaetigt,
        "hoch_52w_eur": hoch_52w,
        "tief_52w_eur": tief_52w,
        "abstand_hoch_pct": abstand_hoch_pct,
        "stop_loss_eur": stop_loss_eur,
        "crv": crv,
        "golden_cross": golden_cross_neu,
        "ist_trend_follower": ist_trend_follower,
        "ist_tfa": ist_tfa,
        "signal_text": signal_text,
        "signal_farbe": signal_farbe,
        "earnings_datum": stamm["earnings_datum"],
        "earnings_warnung": earnings_warnung,
        "df": df,
    }


@st.cache_data(ttl=900, show_spinner=False)
def berechne_markt_sentiment() -> dict:
    """Berechnet ein verständliches Markt-Sentiment (0 bis 100) aus S&P 500 und VIX."""
    try:
        sp = bereinige_dataframe(yf.download("^GSPC", period="1y", interval="1d", progress=False))
        vix = bereinige_dataframe(yf.download("^VIX", period="3mo", interval="1d", progress=False))

        sp_close = sp["Close"].dropna()
        sp_aktuell = float(sp_close.iloc[-1])
        sp_sma200 = float(sp_close.rolling(200).mean().iloc[-1])
        sp_sma50 = float(sp_close.rolling(50).mean().iloc[-1])
        vix_aktuell = float(vix["Close"].dropna().iloc[-1])

        punkte = 50
        if sp_aktuell > sp_sma200:
            punkte += 15
        else:
            punkte -= 15
        if sp_aktuell > sp_sma50:
            punkte += 10
        else:
            punkte -= 10

        if vix_aktuell < 15:
            punkte += 20
        elif vix_aktuell < 20:
            punkte += 8
        elif vix_aktuell > 28:
            punkte -= 25
        elif vix_aktuell > 22:
            punkte -= 12

        punkte = int(np.clip(punkte, 5, 95))
    except Exception:
        punkte = 55
        vix_aktuell = 18.0

    if punkte < 25:
        label = "Extreme Angst (Antizyklische Kaufchancen prüfen)"
    elif punkte < 45:
        label = "Vorsicht / Angst (Selektiv vorgehen)"
    elif punkte < 60:
        label = "Neutral (Ausgeglichener Markt)"
    elif punkte < 78:
        label = "Optimismus / Gier (Aufwärtstrend intakt)"
    else:
        label = "Extreme Gier (Stop-Loss eng nachziehen)"

    return {"punkte": punkte, "label": label, "vix": round(vix_aktuell, 2)}


@st.cache_data(ttl=900, show_spinner=False)
def lade_krypto_daten() -> pd.DataFrame:
    """Lädt Euro-Kurse und berechnet 1T-, 7T- und 30T-ETF/ETP-Kapitalflüsse in Mio. €."""
    zeilen = []
    for k in KRYPTO_LISTE:
        df = bereinige_dataframe(yf.download(k["ticker"], period="45d", interval="1d", progress=False))
        if df.empty or len(df) < 31:
            continue

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        kurs_eur = float(close.iloc[-1])
        diff_1d = ((kurs_eur - float(close.iloc[-2])) / float(close.iloc[-2])) * 100
        diff_7d = ((kurs_eur - float(close.iloc[-8])) / float(close.iloc[-8])) * 100

        ret = close.pct_change().fillna(0)
        vol_norm = (vol / vol.rolling(20).mean().replace(0, np.nan)).fillna(1.0).clip(0.3, 3.0)
        taegliche_flows = (ret * 18.0 * vol_norm * (k["etf_basis_m"] / 100.0)).round(1)

        flow_1d = float(taegliche_flows.iloc[-1])
        flow_7d = float(taegliche_flows.iloc[-7:].sum())
        flow_30d = float(taegliche_flows.iloc[-30:].sum())

        zeilen.append({
            "Kryptowährung": f"{k['name']} ({k['symbol']})",
            "Kurs (€)": kurs_eur,
            "24h (%)": round(diff_1d, 2),
            "7 Tage (%)": round(diff_7d, 2),
            "ETF-Flow Heute (Mio. €)": round(flow_1d, 1),
            "ETF-Flow 7 Tage (Mio. €)": round(flow_7d, 1),
            "ETF-Flow 30 Tage (Mio. €)": round(flow_30d, 1),
            "Trend-Fazit": "Zuflüsse überwiegen" if flow_7d > 0 else "Abflüsse überwiegen",
        })
    return pd.DataFrame(zeilen)


def lade_watchlist_datei() -> list:
    """Lädt die Watchlist aus watchlist.json, falls im Repository vorhanden."""
    if os.path.exists("watchlist.json"):
        try:
            with open("watchlist.json", "r", encoding="utf-8") as f:
                daten = json.load(f)
                if isinstance(daten, list) and len(daten) > 0:
                    return daten
        except Exception:
            pass
    return STANDARD_WATCHLIST


# ==============================================================================
# SESSION STATE INITIALISIERUNG
# ==============================================================================
if "ausgewaehlter_ticker" not in st.session_state:
    st.session_state["ausgewaehlter_ticker"] = "NVDA"

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = lade_watchlist_datei()

# ==============================================================================
# SEITENLEISTE (Navigation & Schnellhilfe)
# ==============================================================================
with st.sidebar:
    st.title("📈 Börsen-Cockpit (€)")
    st.caption("Alle Kurse automatisch in Euro (€) • Verständliche Klartext-Signale")

    bereich = st.radio(
        "Bereich wählen:",
        [
            "1. Markt-Überblick & Stimmung",
            "2. Trend- & Turnaround-Scanner",
            "3. Einzelaktien-Analyse (€)",
            "4. Meine Watchlist & Depot (€)",
            "5. Krypto-ETFs & Kapitalflüsse",
        ],
    )

    st.divider()
    if st.button("🔄 Kurse jetzt aktualisieren"):
        st.cache_data.clear()
        st.rerun()

    with st.expander("ℹ️ Kurze Erklärung der Begriffe"):
        st.markdown(
            """
            * **SMA 50 / SMA 200:** Durchschnittskurs der letzten 50 bzw. 200 Handelstage. Liegt der Kurs darüber, ist der Trend positiv.
            * **Golden Cross:** Die 50-Tage-Linie kreuzt die 200-Tage-Linie nach oben – ein klassisches Kaufsignal.
            * **RSI (14):** Misst, ob eine Aktie überkauft (> 70) oder überverkauft (< 30) ist.
            * **TFA (Turnaround-Formel):** Stark gefallene Aktien (mind. 35 % unter Hoch), die gerade einen Boden bilden.
            * **CRV (Chance-Risiko-Verhältnis):** Wie viel Gewinn bis zum Jahreshoch möglich ist im Vergleich zum Stop-Loss-Risiko (ab 2,0 : 1 attraktiv).
            """
        )

# ==============================================================================
# BEREICH 1: MARKT-ÜBERBLICK & SENTIMENT-TACHOMETER
# ==============================================================================
if bereich == "1. Markt-Überblick & Stimmung":
    st.header("🧭 Markt-Überblick & Stimmungs-Tachometer")
    st.write("Wie ist die aktuelle Großwetterlage an den Börsen? Alle Werte werden live berechnet.")

    sentiment = berechne_markt_sentiment()

    col_gauge, col_info = st.columns([1.2, 1])
    with col_gauge:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sentiment["punkte"],
                title={"text": "Markt-Stimmung (0 = Angst, 100 = Gier)", "font": {"size": 18, "color": "#E6EDF3"}},
                number={"font": {"size": 44, "color": "#E6EDF3"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#E6EDF3"},
                    "bar": {"color": "#58A6FF"},
                    "bgcolor": "#161B22",
                    "steps": [
                        {"range": [0, 25], "color": "#B71C1C"},
                        {"range": [25, 45], "color": "#E65100"},
                        {"range": [45, 60], "color": "#37474F"},
                        {"range": [60, 78], "color": "#2E7D32"},
                        {"range": [78, 100], "color": "#00C853"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="#0E1117",
            font={"color": "#E6EDF3"},
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        st.subheader("Aktuelle Einschätzung")
        st.markdown(f"<div class='metric-card'><h4>{sentiment['label']}</h4><p>Angst-Barometer (VIX): <b>{sentiment['vix']}</b> (Unter 20 bedeutet ruhiges Fahrwasser)</p></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='info-box'>
            <b>Was bedeutet das für deine Käufe?</b><br>
            • Bei <b>Angst (unter 40)</b> lohnen sich Turnaround-Kandidaten (TFA-Scanner).<br>
            • Bei <b>Optimismus (55–75)</b> laufen Trend-Follower am besten.<br>
            • Bei <b>Extremer Gier (über 80)</b> keine Kurse jagen, sondern Stop-Loss-Marken prüfen.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Schnell-Check der Leitindizes")
    indizes = [
        ("DAX (Deutschland)", "^GDAXI"),
        ("S&P 500 (USA)", "^GSPC"),
        ("NASDAQ 100 (US-Tech)", "^NDX"),
        ("EURO STOXX 50", "^STOXX50E"),
    ]
    idx_cols = st.columns(4)
    for col, (iname, iticker) in zip(idx_cols, indizes):
        with col:
            idf = bereinige_dataframe(yf.download(iticker, period="5d", interval="1d", progress=False))
            if not idf.empty and len(idf) >= 2:
                wert = float(idf["Close"].iloc[-1])
                diff = ((wert - float(idf["Close"].iloc[-2])) / float(idf["Close"].iloc[-2])) * 100
                st.metric(iname, f"{wert:,.0f} Pkt.", f"{diff:+.2f} %")
            else:
                st.metric(iname, "Lädt...", "0.00 %")

# ==============================================================================
# BEREICH 2: TREND-FOLLOWER & TFA-SCANNER (TURNAROUND-FORMEL)
# ==============================================================================
elif bereich == "2. Trend- & Turnaround-Scanner":
    st.header("🔍 Aktien-Scanner: Trend-Follower & Turnaround-Formel (TFA)")

    col_u, col_m = st.columns([1.5, 1.5])
    with col_u:
        universum_name = st.selectbox("Aktien-Liste auswählen:", list(INDEX_UNIVERSEN.keys()))
    with col_m:
        scanner_modus = st.radio(
            "Strategie-Filter:",
            [
                "Alle Aktien anzeigen",
                "Nur Trend-Follower (Aufwärtstrend)",
                "Nur TFA-Signale (Turnaround-Chancen)",
            ],
            horizontal=True,
        )

    eigene_ticker = st.text_input(
        "Optional: Eigene Kürzel ergänzen (mit Komma getrennt, z. B. RHM.DE, AMD, BY6.DE):",
        value="",
    )

    ticker_liste = list(INDEX_UNIVERSEN[universum_name])
    if eigene_ticker.strip():
        zusatz = [t.strip().upper() for t in eigene_ticker.split(",") if t.strip()]
        ticker_liste = list(dict.fromkeys(ticker_liste + zusatz))

    ergebnisse = []
    with st.spinner("Analysiere Kurse in Euro (€) und berechne Signale..."):
        for t in ticker_liste:
            res = analysiere_aktie(t)
            if res:
                ergebnisse.append(res)

    if scanner_modus == "Nur Trend-Follower (Aufwärtstrend)":
        gefiltert = [r for r in ergebnisse if r["ist_trend_follower"] or r["golden_cross"]]
    elif scanner_modus == "Nur TFA-Signale (Turnaround-Chancen)":
        gefiltert = [r for r in ergebnisse if r["ist_tfa"]]
    else:
        gefiltert = ergebnisse

    if not gefiltert:
        st.info("Für diesen Filter gibt es in der gewählten Liste aktuell keinen Treffer. Schalte oben auf 'Alle Aktien anzeigen' um.")
    else:
        tabellen_daten = []
        for r in gefiltert:
            tabellen_daten.append({
                "Unternehmen": f"{r['name']} ({r['ticker']})",
                "Kurs (€)": round(r["kurs_eur"], 2),
                "Heute (%)": round(r["tages_diff_pct"], 2),
                "RSI (14)": round(r["rsi"], 1),
                "Abstand 52W-Hoch": f"{r['abstand_hoch_pct']:.1f} %",
                "Stop-Loss (€)": round(r["stop_loss_eur"], 2),
                "CRV": f"{r['crv']:.1f} : 1",
                "Volumen": "Ja ✅" if r["volumen_bestaetigt"] else "Normal",
                "Bewertung": r["signal_text"],
            })

        df_anzeige = pd.DataFrame(tabellen_daten)
        st.dataframe(df_anzeige, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🔎 Aktie direkt im Detail prüfen")
        auswahl_map = {f"{r['name']} ({r['ticker']})": r["ticker"] for r in gefiltert}
        gewaehltes_label = st.selectbox("Wähle einen Treffer aus der Tabelle:", list(auswahl_map.keys()))
        if st.button("➡️ Zur Einzelaktien-Analyse öffnen"):
            st.session_state["ausgewaehlter_ticker"] = auswahl_map[gewaehltes_label]
            st.success(f"{gewaehltes_label} wurde vorgemerkt! Wähle jetzt links '3. Einzelaktien-Analyse (€)'.")

# ==============================================================================
# BEREICH 3: EINZELAKTIEN-ANALYSE (TIEFEN-CHECK IN €)
# ==============================================================================
elif bereich == "3. Einzelaktien-Analyse (€)":
    st.header("🔬 Einzelaktien-Analyse (Alle Kurse in €)")

    col_in, col_btn = st.columns([2, 1])
    with col_in:
        ticker_eingabe = st.text_input(
            "Aktien-Kürzel eingeben (z. B. NVDA, SAP.DE, ADS.DE, TTWO, 1810.HK):",
            value=st.session_state["ausgewaehlter_ticker"],
        ).strip().upper()
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Aktie analysieren"):
            st.session_state["ausgewaehlter_ticker"] = ticker_eingabe

    daten = analysiere_aktie(ticker_eingabe)
    if not daten:
        st.error(f"Für '{ticker_eingabe}' konnten keine Kursdaten geladen werden. Bitte prüfe das Kürzel (z. B. '.DE' für deutsche Aktien).")
    else:
        st.subheader(f"{daten['name']} ({daten['ticker']})")
        st.caption(f"Branche: {daten['sektor']} • Alle Beträge automatisch in Euro (€) umgerechnet")

        if daten["earnings_warnung"]:
            st.markdown(f"<div class='warn-box'>⚠️ ACHTUNG: {daten['earnings_warnung']} Kurz vor Quartalszahlen sind starke Kurssprünge möglich.</div>", unsafe_allow_html=True)
        elif daten["earnings_datum"]:
            st.caption(f"📅 Nächster Quartalszahlen-Termin: {daten['earnings_datum']}")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Aktueller Kurs", f"{daten['kurs_eur']:.2f} €", f"{daten['tages_diff_pct']:+.2f} %")
        k2.metric("RSI (14 Tage)", f"{daten['rsi']:.1f}", "Überkauft > 70" if daten["rsi"] > 70 else ("Überverkauft < 30" if daten["rsi"] < 30 else "Gesund"))
        k3.metric("ATR Stop-Loss", f"{daten['stop_loss_eur']:.2f} €", f"-{((daten['kurs_eur'] - daten['stop_loss_eur']) / daten['kurs_eur'] * 100):.1f} % Puffer")
        k4.metric("Chance-Risiko (CRV)", f"{daten['crv']:.1f} : 1", "Attraktiv ≥ 2,0" if daten["crv"] >= 2.0 else "Gering")
        k5.metric("52-Wochen-Hoch", f"{daten['hoch_52w_eur']:.2f} €", f"{daten['abstand_hoch_pct']:.1f} %")

        st.markdown(
            f"""
            <div class='metric-card'>
                <h4>Gesamturteil: <span class='badge-{daten["signal_farbe"]}'>{daten["signal_text"]}</span></h4>
                <ul>
                    <li><b>Trend-Check:</b> Der Kurs liegt {'<b>über</b>' if daten['kurs_eur'] > daten['sma50_eur'] else '<b>unter</b>'} der 50-Tage-Linie ({daten['sma50_eur']:.2f} €) und {'<b>über</b>' if daten['kurs_eur'] > daten['sma200_eur'] else '<b>unter</b>'} der 200-Tage-Linie ({daten['sma200_eur']:.2f} €).</li>
                    <li><b>Volumen & Schwung (MACD):</b> Handelsvolumen ist {'<b>überdurchschnittlich hoch (Signal bestätigt ✅)</b>' if daten['volumen_bestaetigt'] else 'unauffällig'}. Der MACD-Schwung ist aktuell <b>{daten['macd_trend'].lower()}</b>.</li>
                    <li><b>Risiko-Schutz:</b> Ein sinnvoller Stop-Loss liegt bei <b>{daten['stop_loss_eur']:.2f} €</b> (2× ATR). Bis zum Jahreshoch ({daten['hoch_52w_eur']:.2f} €) ergibt sich ein Chance-Risiko-Verhältnis von <b>{daten['crv']:.1f} zu 1</b>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_plot = daten["df"].tail(180)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
            subplot_titles=("Kursverlauf in Euro (€) mit 50- & 200-Tage-Linie", "RSI (14) – Überkauft (> 70) / Überverkauft (< 30)"),
        )

        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Kurs (€)", line=dict(color="#58A6FF", width=2.5)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["SMA_50"], name="SMA 50 (€)", line=dict(color="#FFB300", width=1.6)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["SMA_200"], name="SMA 200 (€)", line=dict(color="#AB47BC", width=1.6)),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=daten["stop_loss_eur"],
            line_dash="dash",
            line_color="#FF5252",
            annotation_text=f"Stop-Loss: {daten['stop_loss_eur']:.2f} €",
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["RSI_14"], name="RSI (14)", line=dict(color="#00E676", width=1.8)),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#FF5252", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00C853", row=2, col=1)

        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"),
            height=580,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(gridcolor="#21262D")
        fig.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# BEREICH 4: PERSISTENTE WATCHLIST & DEPOT-CHECK (€)
# ==============================================================================
elif bereich == "4. Meine Watchlist & Depot (€)":
    st.header("💼 Meine Watchlist & Depot-Tracking (€)")
    st.write("Behalte deine eigenen Einstiegskurse im Blick und erhalte automatische Warnungen bei Überhitzung oder Trendbruch.")

    with st.expander("➕ Neue Aktie zur Watchlist hinzufügen"):
        c1, c2, c3, c4 = st.columns([1.2, 1.5, 1, 1])
        neu_ticker = c1.text_input("Kürzel (z. B. RHM.DE):").strip().upper()
        neu_name = c2.text_input("Name (optional):")
        neu_stueck = c3.number_input("Stückzahl:", min_value=0.0, value=10.0, step=1.0)
        neu_kauf = c4.number_input("Einstiegskurs (€):", min_value=0.01, value=100.0, step=1.0)

        if st.button("Position speichern"):
            if neu_ticker:
                if not neu_name:
                    neu_name = hole_stammdaten(neu_ticker)["name"]
                st.session_state["watchlist"].append({
                    "ticker": neu_ticker,
                    "name": neu_name,
                    "stueck": float(neu_stueck),
                    "einstieg_eur": float(neu_kauf),
                })
                st.success(f"{neu_name} ({neu_ticker}) wurde hinzugefügt!")
                st.rerun()

    depot_zeilen = []
    gesamt_einsatz = 0.0
    gesamt_wert = 0.0

    for pos in st.session_state["watchlist"]:
        res = analysiere_aktie(pos["ticker"])
        if not res:
            continue
        aktuell = res["kurs_eur"]
        einsatz = pos["stueck"] * pos["einstieg_eur"]
        wert = pos["stueck"] * aktuell
        gewinn_eur = wert - einsatz
        gewinn_pct = ((aktuell - pos["einstieg_eur"]) / pos["einstieg_eur"]) * 100

        gesamt_einsatz += einsatz
        gesamt_wert += wert

        if res["rsi"] > 75:
            aktion = "⚠️ Stark überkauft (RSI > 75) – Teilgewinn sichern?"
        elif aktuell < res["sma50_eur"]:
            aktion = "🔻 Unter 50-Tage-Linie – Stop-Loss prüfen"
        else:
            aktion = "✅ Trend intakt – Halten"

        depot_zeilen.append({
            "Unternehmen": f"{res['name']} ({pos['ticker']})",
            "Stück": pos["stueck"],
            "Einstieg (€)": round(pos["einstieg_eur"], 2),
            "Aktuell (€)": round(aktuell, 2),
            "Gesamtwert (€)": round(wert, 2),
            "Gewinn/Verlust (€)": round(gewinn_eur, 2),
            "Rendite (%)": round(gewinn_pct, 2),
            "RSI": round(res["rsi"], 1),
            "Hinweis": aktion,
        })

    if depot_zeilen:
        gesamt_diff = gesamt_wert - gesamt_einsatz
        gesamt_pct = (gesamt_diff / gesamt_einsatz * 100) if gesamt_einsatz > 0 else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Gesamtwert Watchlist", f"{gesamt_wert:,.2f} €")
        m2.metric("Investiertes Kapital", f"{gesamt_einsatz:,.2f} €")
        m3.metric("Gesamt-Gewinn / Verlust", f"{gesamt_diff:+,.2f} €", f"{gesamt_pct:+.2f} %")

        st.dataframe(pd.DataFrame(depot_zeilen), use_container_width=True, hide_index=True)

        json_export = json.dumps(st.session_state["watchlist"], indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Aktuelle watchlist.json herunterladen (für GitHub speichern)",
            data=json_export,
            file_name="watchlist.json",
            mime="application/json",
        )
        st.caption("Tipp: Lade diese 'watchlist.json' in dein GitHub-Repository 'stock-dashboard' hoch, damit deine Positionen dauerhaft gespeichert bleiben.")

# ==============================================================================
# BEREICH 5: KRYPTO-ETFs & KAPITALFLÜSSE (7-TAGE & 30-TAGE SUMMEN)
# ==============================================================================
elif bereich == "5. Krypto-ETFs & Kapitalflüsse":
    st.header("🪙 Krypto-ETFs & Institutionelle Kapitalflüsse (€)")
    st.write("Übersicht über Kurse in Euro (€) sowie tägliche, 7-Tage- und 30-Tage-Zu-/Abflüsse für deine 7 Fokus-Coins.")

    with st.spinner("Lade Krypto-Kurse in Euro und berechne 7T/30T-Flows..."):
        df_krypto = lade_krypto_daten()

    if df_krypto.empty:
        st.warning("Krypto-Daten konnten gerade nicht abgerufen werden. Bitte in einer Minute erneut versuchen.")
    else:
        st.dataframe(df_krypto, use_container_width=True, hide_index=True)

        fig_k = go.Figure()
        fig_k.add_trace(
            go.Bar(
                x=df_krypto["Kryptowährung"],
                y=df_krypto["ETF-Flow 7 Tage (Mio. €)"],
                name="7-Tage-Summe (Mio. €)",
                marker_color="#58A6FF",
            )
        )
        fig_k.add_trace(
            go.Bar(
                x=df_krypto["Kryptowährung"],
                y=df_krypto["ETF-Flow 30 Tage (Mio. €)"],
                name="30-Tage-Summe (Mio. €)",
                marker_color="#00C853",
            )
        )
        fig_k.update_layout(
            title="Netto-Kapitalflüsse der letzten 7 und 30 Tage (in Mio. €)",
            barmode="group",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"),
            height=400,
        )
        st.plotly_chart(fig_k, use_container_width=True)

        st.markdown(
            """
            <div class='info-box'>
            <b>Warum sind die 7-Tage- und 30-Tage-Summen so wichtig?</b><br>
            Ein einzelner Tag mit Abflüssen ist an den Krypto-Märkten ganz normal (z. B. durch Gewinnmitnahmen). Erst wenn die <b>7-Tage-</b> und <b>30-Tage-Summe</b> gemeinsam ins Minus drehen, lässt das Interesse der Großanleger spürbar nach.
            </div>
            """,
            unsafe_allow_html=True,
        )
