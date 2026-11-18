from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

# ==============================================================================
# SEITEN-KONFIGURATION & DESIGN (Dark-Not-Black Theme)
# ==============================================================================
st.set_page_config(
    page_title="Aktien & Krypto Cockpit (€)",
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
        padding: 18px;
        margin-bottom: 14px;
    }
    .badge-green {
        background-color: rgba(0, 200, 83, 0.16);
        color: #00E676;
        border: 1px solid #00C853;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .badge-red {
        background-color: rgba(255, 82, 82, 0.16);
        color: #FF5252;
        border: 1px solid #FF5252;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .badge-amber {
        background-color: rgba(255, 179, 0, 0.16);
        color: #FFD54F;
        border: 1px solid #FFB300;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .info-box {
        background-color: #161B22;
        border-left: 4px solid #58A6FF;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.94rem;
        line-height: 1.55;
    }
    .verdict-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-left: 5px solid #00E676;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        font-size: 0.96rem;
        line-height: 1.6;
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
# VOREINGESTELLTE AKTIEN-LISTEN, NAMENS-ZUORDNUNG & KRYPTO-DATEN
# ==============================================================================
INDEX_UNIVERSEN = {
    "China & Hongkong (Tech, E-Auto & Konsum)": [
        "1810.HK", "1211.HK", "0700.HK", "9988.HK", "3690.HK",
        "9618.HK", "9888.HK", "0175.HK", "2318.HK", "0981.HK",
        "NIO", "XPEV", "LI", "PDD"
    ],
    "Japan Top-Werte (Nikkei Auswahl)": [
        "7203.T", "6758.T", "7974.T", "9984.T", "8035.T",
        "6861.T", "9983.T", "6501.T", "7267.T", "8306.T",
        "6857.T", "4502.T"
    ],
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
        "NKE", "DIS", "BA"
    ],
}

# Kombiniertes Global-Universum ohne Duplikate
INDEX_UNIVERSEN["Alle Märkte kombiniert (Global-Scan)"] = list(
    dict.fromkeys(
        ticker
        for liste in INDEX_UNIVERSEN.values()
        for ticker in liste
    )
)

# Direkte Namens-Suche für ausgeschriebene Aktiennamen (inkl. Aliase)
BEKANNTE_AKTIEN_NAMEN = {
    # China & Hongkong
    "xiaomi": ("1810.HK", "Xiaomi Corporation"),
    "xiaomi corp": ("1810.HK", "Xiaomi Corporation"),
    "xiaomi corporation": ("1810.HK", "Xiaomi Corporation"),
    "byd": ("1211.HK", "BYD Company Limited"),
    "byd company": ("1211.HK", "BYD Company Limited"),
    "byd co": ("1211.HK", "BYD Company Limited"),
    "tencent": ("0700.HK", "Tencent Holdings Limited"),
    "tencent holdings": ("0700.HK", "Tencent Holdings Limited"),
    "alibaba": ("9988.HK", "Alibaba Group Holding"),
    "alibaba group": ("9988.HK", "Alibaba Group Holding"),
    "meituan": ("3690.HK", "Meituan"),
    "jd": ("9618.HK", "JD.com, Inc."),
    "jd.com": ("9618.HK", "JD.com, Inc."),
    "baidu": ("9888.HK", "Baidu, Inc."),
    "geely": ("0175.HK", "Geely Automobile Holdings"),
    "ping an": ("2318.HK", "Ping An Insurance"),
    "smic": ("0981.HK", "Semiconductor Manufacturing International"),
    "nio": ("NIO", "NIO Inc."),
    "xpeng": ("XPEV", "XPeng Inc."),
    "li auto": ("LI", "Li Auto Inc."),
    "pinduoduo": ("PDD", "PDD Holdings Inc."),
    "pdd": ("PDD", "PDD Holdings Inc."),
    # Japan
    "toyota": ("7203.T", "Toyota Motor Corporation"),
    "toyota motor": ("7203.T", "Toyota Motor Corporation"),
    "sony": ("6758.T", "Sony Group Corporation"),
    "sony group": ("6758.T", "Sony Group Corporation"),
    "nintendo": ("7974.T", "Nintendo Co., Ltd."),
    "softbank": ("9984.T", "SoftBank Group Corp."),
    "softbank group": ("9984.T", "SoftBank Group Corp."),
    "tokyo electron": ("8035.T", "Tokyo Electron Limited"),
    "keyence": ("6861.T", "Keyence Corporation"),
    "fast retailing": ("9983.T", "Fast Retailing Co., Ltd. (Uniqlo)"),
    "uniqlo": ("9983.T", "Fast Retailing Co., Ltd. (Uniqlo)"),
    "hitachi": ("6501.T", "Hitachi, Ltd."),
    "honda": ("7267.T", "Honda Motor Co., Ltd."),
    "honda motor": ("7267.T", "Honda Motor Co., Ltd."),
    "mitsubishi ufj": ("8306.T", "Mitsubishi UFJ Financial Group"),
    "advantest": ("6857.T", "Advantest Corporation"),
    "takeda": ("4502.T", "Takeda Pharmaceutical"),
    "panasonic": ("6752.T", "Panasonic Holdings"),
    "komatsu": ("6301.T", "Komatsu Ltd."),
    # Deutschland & Europa
    "sap": ("SAP.DE", "SAP SE"),
    "siemens": ("SIE.DE", "Siemens AG"),
    "siemens energy": ("ENR.DE", "Siemens Energy AG"),
    "allianz": ("ALV.DE", "Allianz SE"),
    "deutsche telekom": ("DTE.DE", "Deutsche Telekom AG"),
    "telekom": ("DTE.DE", "Deutsche Telekom AG"),
    "adidas": ("ADS.DE", "adidas AG"),
    "mercedes": ("MBG.DE", "Mercedes-Benz Group AG"),
    "mercedes-benz": ("MBG.DE", "Mercedes-Benz Group AG"),
    "bmw": ("BMW.DE", "Bayerische Motoren Werke AG"),
    "basf": ("BAS.DE", "BASF SE"),
    "infineon": ("IFX.DE", "Infineon Technologies AG"),
    "münchener rück": ("MUV2.DE", "Münchener Rückversicherungs-Gesellschaft"),
    "munich re": ("MUV2.DE", "Münchener Rückversicherungs-Gesellschaft"),
    "rheinmetall": ("RHM.DE", "Rheinmetall AG"),
    "dhl": ("DHL.DE", "Deutsche Post AG (DHL)"),
    "deutsche post": ("DHL.DE", "Deutsche Post AG (DHL)"),
    "airbus": ("AIR.DE", "Airbus SE"),
    "bayer": ("BAYN.DE", "Bayer AG"),
    "volkswagen": ("VOW3.DE", "Volkswagen AG"),
    "vw": ("VOW3.DE", "Volkswagen AG"),
    "asml": ("ASML.AS", "ASML Holding N.V."),
    "lvmh": ("MC.PA", "LVMH Moët Hennessy Louis Vuitton"),
    "novo nordisk": ("NOVO-B.CO", "Novo Nordisk A/S"),
    "nestle": ("NESN.SW", "Nestlé S.A."),
    "nestlé": ("NESN.SW", "Nestlé S.A."),
    "shell": ("SHEL.L", "Shell plc"),
    # USA
    "apple": ("AAPL", "Apple Inc."),
    "microsoft": ("MSFT", "Microsoft Corporation"),
    "nvidia": ("NVDA", "NVIDIA Corporation"),
    "amazon": ("AMZN", "Amazon.com, Inc."),
    "alphabet": ("GOOGL", "Alphabet Inc. (Google)"),
    "google": ("GOOGL", "Alphabet Inc. (Google)"),
    "meta": ("META", "Meta Platforms, Inc."),
    "facebook": ("META", "Meta Platforms, Inc."),
    "tesla": ("TSLA", "Tesla, Inc."),
    "broadcom": ("AVGO", "Broadcom Inc."),
    "amd": ("AMD", "Advanced Micro Devices, Inc."),
    "netflix": ("NFLX", "Netflix, Inc."),
    "take-two": ("TTWO", "Take-Two Interactive Software"),
    "take two": ("TTWO", "Take-Two Interactive Software"),
    "salesforce": ("CRM", "Salesforce, Inc."),
    "palantir": ("PLTR", "Palantir Technologies Inc."),
    "paypal": ("PYPL", "PayPal Holdings, Inc."),
    "intel": ("INTC", "Intel Corporation"),
    "nike": ("NKE", "NIKE, Inc."),
    "disney": ("DIS", "The Walt Disney Company"),
    "boeing": ("BA", "The Boeing Company"),
    "circle": ("CRCL", "Circle Internet Group"),
}

SCHNELLAUSWAHL_AKTIEN = {
    "Xiaomi Corporation (1810.HK) 🇨🇳": "1810.HK",
    "BYD Company Limited (1211.HK) 🇨🇳": "1211.HK",
    "Tencent Holdings (0700.HK) 🇨🇳": "0700.HK",
    "Alibaba Group (9988.HK) 🇨🇳": "9988.HK",
    "Toyota Motor Corp. (7203.T) 🇯🇵": "7203.T",
    "Sony Group Corp. (6758.T) 🇯🇵": "6758.T",
    "Nintendo Co., Ltd. (7974.T) 🇯🇵": "7974.T",
    "SoftBank Group (9984.T) 🇯🇵": "9984.T",
    "Tokyo Electron (8035.T) 🇯🇵": "8035.T",
    "NVIDIA Corporation (NVDA) 🇺🇸": "NVDA",
    "Take-Two Interactive (TTWO) 🇺🇸": "TTWO",
    "Apple Inc. (AAPL) 🇺🇸": "AAPL",
    "Microsoft Corp. (MSFT) 🇺🇸": "MSFT",
    "Amazon.com Inc. (AMZN) 🇺🇸": "AMZN",
    "Tesla Inc. (TSLA) 🇺🇸": "TSLA",
    "Palantir Technologies (PLTR) 🇺🇸": "PLTR",
    "Rheinmetall AG (RHM.DE) 🇩🇪": "RHM.DE",
    "SAP SE (SAP.DE) 🇩🇪": "SAP.DE",
    "adidas AG (ADS.DE) 🇩🇪": "ADS.DE",
    "Siemens AG (SIE.DE) 🇩🇪": "SIE.DE",
    "ASML Holding (ASML.AS) 🇪🇺": "ASML.AS",
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

# ==============================================================================
# HILFSFUNKTIONEN: NAMENSSUCHE, WÄHRUNGSUMRECHNUNG & INDIKATOREN
# ==============================================================================
def bereinige_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Flacht MultiIndex-Spalten von yfinance ab und stellt 1D-Serien sicher."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def erkenne_region(ticker: str) -> str:
    """Gibt die Börsenregion mit Flagge zurück."""
    t = ticker.upper()
    if t.endswith(".HK") or t.endswith(".SS") or t.endswith(".SZ") or t in {"NIO", "XPEV", "LI", "PDD", "BABA", "JD", "BIDU"}:
        return "🇨🇳 China/HK"
    if t.endswith(".T"):
        return "🇯🇵 Japan"
    if t.endswith(".DE"):
        return "🇩🇪 Deutschland"
    if t.endswith((".AS", ".PA", ".CO", ".SW", ".L", ".MC", ".MI")):
        return "🇪🇺 Europa"
    return "🇺🇸 USA"


@st.cache_data(ttl=86400, show_spinner=False)
def suche_ticker_aus_name(eingabe: str) -> tuple[str, str]:
    """
    Wandelt sowohl Kürzel (z. B. 1810.HK) als auch ausgeschriebene Firmennamen
    (z. B. 'Xiaomi', 'BYD', 'Toyota', 'Rheinmetall') zuverlässig in ein Börsen-Kürzel um.
    """
    sauber = eingabe.strip()
    if not sauber:
        return "1810.HK", "Xiaomi Corporation"

    klein = sauber.lower()

    # 1. Exakter Treffer im kuratierten Verzeichnis
    if klein in BEKANNTE_AKTIEN_NAMEN:
        return BEKANNTE_AKTIEN_NAMEN[klein]

    # 2. Teilstring-Treffer für ausgeschriebene Namen (z. B. "Xiaomi Corp", "Toyota Motor")
    if len(klein) >= 3:
        for schluessel, (tkr, vollname) in BEKANNTE_AKTIEN_NAMEN.items():
            if klein in schluessel or schluessel in klein:
                return tkr, vollname

    # 3. Wenn die Eingabe bereits wie ein echtes Ticker-Symbol aussieht (z. B. 1810.HK, SAP.DE, NVDA)
    if "." in sauber or "-" in sauber or (sauber.isupper() and len(sauber) <= 5 and " " not in sauber):
        return sauber.upper(), sauber.upper()

    # 4. Live-Suche über yfinance Search / Yahoo Finance API für beliebige andere Firmennamen
    try:
        search_obj = yf.Search(sauber, max_results=8)
        quotes = getattr(search_obj, "quotes", []) or []
        if not quotes:
            resp = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": sauber, "quotesCount": 8, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            if resp.status_code == 200:
                quotes = resp.json().get("quotes", [])

        bevorzugte_boersen = {"HKG", "JPX", "GER", "FRA", "NMS", "NYQ", "NGM", "AMS", "PAR", "EBS", "LSE"}
        aktien_treffer = [q for q in quotes if q.get("quoteType") == "EQUITY" and q.get("symbol")]
        if not aktien_treffer:
            aktien_treffer = [q for q in quotes if q.get("symbol")]

        # Hauptbörsen gegenüber OTC (PNK) bevorzugen
        aktien_treffer.sort(
            key=lambda q: (
                0 if q.get("exchange") in bevorzugte_boersen else (2 if q.get("exchange") == "PNK" else 1)
            )
        )
        if aktien_treffer:
            best = aktien_treffer[0]
            sym = best["symbol"].upper()
            name = best.get("longname") or best.get("shortname") or sym
            return sym, name
    except Exception:
        pass

    return sauber.upper(), sauber.upper()


@st.cache_data(ttl=86400, show_spinner=False)
def hole_wechselkurs_zu_eur(waehrung: str) -> float:
    """
    Holt den aktuellen Wechselkurs einer Fremdwährung nach Euro (€) inkl. Invers-Prüfung
    für JPY, HKD, CNY, USD, CHF, GBP, DKK.
    """
    if not waehrung or waehrung.upper() == "EUR":
        return 1.0

    w = waehrung.upper()
    if w == "GBX":  # Britische Pence -> Pfund -> EUR
        return hole_wechselkurs_zu_eur("GBP") / 100.0

    # 1. Direktes Paar versuchen (z. B. HKDEUR=X, JPYEUR=X, USDEUR=X)
    try:
        df_dir = bereinige_dataframe(yf.download(f"{w}EUR=X", period="5d", interval="1d", progress=False))
        if not df_dir.empty and "Close" in df_dir.columns:
            kurs = float(df_dir["Close"].dropna().iloc[-1])
            if kurs > 0:
                return kurs
    except Exception:
        pass

    # 2. Inverses Paar versuchen (z. B. EURJPY=X, EURHKD=X, EURCNY=X – oft liquider!)
    try:
        df_inv = bereinige_dataframe(yf.download(f"EUR{w}=X", period="5d", interval="1d", progress=False))
        if not df_inv.empty and "Close" in df_inv.columns:
            inv_kurs = float(df_inv["Close"].dropna().iloc[-1])
            if inv_kurs > 0:
                return 1.0 / inv_kurs
    except Exception:
        pass

    # 3. Realistische Fallback-Richtwerte (falls Yahoo kurzzeitig nicht antwortet)
    fallback = {
        "USD": 0.94,
        "CHF": 1.07,
        "GBP": 1.20,
        "HKD": 0.120,
        "JPY": 0.00615,
        "CNY": 0.129,
        "CNH": 0.129,
        "DKK": 0.134,
    }
    return fallback.get(w, 1.0)


def leite_waehrung_aus_ticker_ab(ticker: str) -> str:
    """Bestimmt die Heimatwährung zuverlässig anhand der Börsenendung."""
    t = ticker.upper()
    if t.endswith((".DE", ".AS", ".PA", ".MC", ".MI", ".VI", "-EUR")):
        return "EUR"
    if t.endswith(".HK"):
        return "HKD"
    if t.endswith(".T"):
        return "JPY"
    if t.endswith((".SS", ".SZ")):
        return "CNY"
    if t.endswith(".SW"):
        return "CHF"
    if t.endswith(".L"):
        return "GBX"
    if t.endswith(".CO"):
        return "DKK"
    return "USD"


@st.cache_data(ttl=86400, show_spinner=False)
def hole_stammdaten(ticker: str) -> dict:
    """Holt den vollen Firmennamen, die Heimatwährung und den nächsten Quartalszahlen-Termin."""
    standard_waehrung = leite_waehrung_aus_ticker_ab(ticker)
    info_daten = {
        "name": ticker,
        "waehrung": standard_waehrung,
        "sektor": "Keine Angabe",
        "earnings_datum": None,
    }

    # Bekannten Vollnamen vorbelegen, falls vorhanden
    for _, (tkr, vollname) in BEKANNTE_AKTIEN_NAMEN.items():
        if tkr == ticker.upper():
            info_daten["name"] = vollname
            break

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        if info.get("longName") or info.get("shortName"):
            info_daten["name"] = info.get("longName") or info.get("shortName")
        if info.get("currency"):
            info_daten["waehrung"] = info.get("currency").upper()
        if info.get("sector"):
            info_daten["sektor"] = info.get("sector")

        cal = t.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal and cal["Earnings Date"]:
            ed = cal["Earnings Date"][0]
            info_daten["earnings_datum"] = str(ed)[:10]
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            info_daten["earnings_datum"] = str(cal.iloc[0, 0])[:10]
    except Exception:
        pass

    return info_daten


@st.cache_data(ttl=900, show_spinner=False)
def lade_kursdaten_eur(ticker: str, zeitraum: str = "1y") -> pd.DataFrame:
    """Lädt historische Kursdaten, rechnet alles sauber in Euro (€) um und berechnet Indikatoren."""
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
        df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50)

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
    """Erstellt eine vollständige, ausführlich begründete Auswertung in Euro (€)."""
    df = lade_kursdaten_eur(ticker, zeitraum="1y")
    if df.empty or len(df) < 50:
        return None

    stamm = hole_stammdaten(ticker)
    region = erkenne_region(ticker)
    aktuell = float(df["Close"].iloc[-1])
    vortag = float(df["Close"].iloc[-2])
    tages_diff_pct = ((aktuell - vortag) / vortag) * 100

    sma50 = float(df["SMA_50"].iloc[-1]) if not pd.isna(df["SMA_50"].iloc[-1]) else aktuell
    sma50_vor5 = float(df["SMA_50"].iloc[-6]) if len(df) >= 6 and not pd.isna(df["SMA_50"].iloc[-6]) else sma50
    sma200 = float(df["SMA_200"].iloc[-1]) if not pd.isna(df["SMA_200"].iloc[-1]) else sma50

    abstand_sma50_pct = ((aktuell - sma50) / sma50) * 100 if sma50 > 0 else 0.0
    abstand_sma200_pct = ((aktuell - sma200) / sma200) * 100 if sma200 > 0 else 0.0
    sma50_zu_200_pct = ((sma50 - sma200) / sma200) * 100 if sma200 > 0 else 0.0

    rsi = float(df["RSI_14"].iloc[-1])
    rsi_vor5 = float(df["RSI_14"].iloc[-6]) if len(df) >= 6 else rsi
    macd_hist = float(df["MACD_Hist"].iloc[-1])
    macd_hist_vortag = float(df["MACD_Hist"].iloc[-2])
    atr = float(df["ATR_14"].iloc[-1]) if not pd.isna(df["ATR_14"].iloc[-1]) else aktuell * 0.03

    vol_aktuell = float(df["Volume"].iloc[-1])
    vol_schnitt = float(df["Vol_SMA_20"].iloc[-1]) if not pd.isna(df["Vol_SMA_20"].iloc[-1]) else vol_aktuell
    volumen_faktor = (vol_aktuell / vol_schnitt) if vol_schnitt > 0 else 1.0
    volumen_bestaetigt = volumen_faktor > 1.10

    hoch_52w = float(df["High"].max())
    tief_52w = float(df["Low"].min())
    abstand_hoch_pct = ((aktuell - hoch_52w) / hoch_52w) * 100
    abstand_tief_pct = ((aktuell - tief_52w) / tief_52w) * 100 if tief_52w > 0 else 0.0

    # Golden Cross Check (SMA 50 kreuzt SMA 200 in den letzten 7 Handelstagen nach oben)
    golden_cross_neu = False
    if len(df) >= 8 and not df["SMA_200"].iloc[-8:].isna().any():
        golden_cross_neu = (df["SMA_50"].iloc[-8] <= df["SMA_200"].iloc[-8]) and (sma50 > sma200)
    golden_cross_nah = (not golden_cross_neu) and (-2.5 <= sma50_zu_200_pct < 0.0) and (sma50 > sma50_vor5)

    # Trend-Follower Check
    ist_trend_follower = (aktuell > sma50) and (sma50 > sma200) and (45 <= rsi <= 72)

    # TFA-Check (Turnaround-Formel: >= 35% unter 52W-Hoch + erste Bodenbildung / Kaufdruck)
    ist_tfa = (abstand_hoch_pct <= -35.0) and (
        (rsi < 38) or (rsi_vor5 < 35 and rsi > rsi_vor5) or (aktuell > sma50 and macd_hist > macd_hist_vortag)
    )

    # ATR-Stop-Loss & Chance-Risiko-Verhältnis (CRV)
    stop_loss_eur = max(0.01, aktuell - 2.0 * atr)
    stop_puffer_pct = ((aktuell - stop_loss_eur) / aktuell) * 100
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

    # --------------------------------------------------------------------------
    # AUSFÜHRLICHE BEGRÜNDUNGEN & KLARTEXT-GESAMTURTEIL
    # --------------------------------------------------------------------------
    if golden_cross_neu:
        signal_text = "Starkes Kaufsignal (Frisches Golden Cross)"
        signal_farbe = "green"
        kurz_begruendung = f"SMA 50 hat SMA 200 frisch nach oben gekreuzt; Kurs {abstand_sma50_pct:+.1f} % über SMA 50 (RSI {rsi:.0f})."
        gesamtfazit = (
            f"Bei {stamm['name']} hat die 50-Tage-Linie ({sma50:.2f} €) soeben die 200-Tage-Linie ({sma200:.2f} €) "
            f"von unten nach oben durchbrochen – ein klassisches, langfristig bullisches Golden-Cross-Kaufsignal. "
            f"Mit einem RSI von {rsi:.1f} hat die Aktie noch ausreichend Luft nach oben, ohne überhitzt zu sein. "
            f"Ein Einstieg oder Teilkauf bietet sich an, abgesichert durch einen Stop-Loss bei {stop_loss_eur:.2f} € (-{stop_puffer_pct:.1f} %)."
        )
    elif ist_trend_follower and volumen_bestaetigt and macd_hist > 0:
        signal_text = "Starker Aufwärtstrend – Kaufen / Halten"
        signal_farbe = "green"
        kurz_begruendung = f"Sauberer Aufwärtstrend ({abstand_sma50_pct:+.1f} % über SMA 50), hohes Volumen ({volumen_faktor:.1f}×) & steigender MACD."
        gesamtfazit = (
            f"{stamm['name']} befindet sich in einem voll intakten Aufwärtstrend: Der Kurs notiert {abstand_sma50_pct:+.1f} % über der "
            f"50-Tage-Linie ({sma50:.2f} €) und {abstand_sma200_pct:+.1f} % über der 200-Tage-Linie ({sma200:.2f} €). "
            f"Das überdurchschnittliche Handelsvolumen ({volumen_faktor:.1f}-facher Schnitt) und der positive MACD bestätigen den Kaufdruck. "
            f"Bestehende Positionen laufen lassen; Neueinstiege lassen sich bei {stop_loss_eur:.2f} € absichern (CRV {crv:.1f} : 1)."
        )
    elif ist_trend_follower:
        signal_text = "Intakter Aufwärtstrend – Halten / Rücksetzer nutzen"
        signal_farbe = "green"
        kurz_begruendung = f"Über SMA 50 ({abstand_sma50_pct:+.1f} %) und SMA 200 ({abstand_sma200_pct:+.1f} %), RSI {rsi:.0f} im gesunden Bereich."
        gesamtfazit = (
            f"Der übergeordnete Trend bei {stamm['name']} zeigt klar nach oben, da sowohl SMA 50 ({sma50:.2f} €) als auch "
            f"SMA 200 ({sma200:.2f} €) unterhalb des aktuellen Kurses verlaufen. Der RSI liegt bei ausgeglichenen {rsi:.1f} Punkten. "
            f"Da das Volumen zuletzt unauffällig war, sind kleine Rücksetzer in Richtung der 50-Tage-Linie ({sma50:.2f} €) "
            f"besonders interessante Einstiegsgelegenheiten."
        )
    elif ist_tfa:
        signal_text = "Turnaround-Chance (TFA-Signal) – Antizyklisch beobachten"
        signal_farbe = "amber"
        kurz_begruendung = f"{abs(abstand_hoch_pct):.0f} % unter 52W-Hoch; RSI ({rsi:.0f}) dreht aus überverkaufter Zone nach oben."
        gesamtfazit = (
            f"{stamm['name']} notiert mit {abstand_hoch_pct:.1f} % deutlich unter seinem 52-Wochen-Hoch ({hoch_52w:.2f} €) "
            f"und erfüllt damit das Hauptkriterium der Turnaround-Formel (TFA). Erste Stabilisierungszeichen sind sichtbar "
            f"(RSI bei {rsi:.1f}, Trend vom 5-Tages-Tief drehend). Für risikobewusste Anleger bietet sich ein gestaffelter "
            f"Einstieg mit engem Stop-Loss bei {stop_loss_eur:.2f} € an (CRV {crv:.1f} : 1 bis zum Jahreshoch)."
        )
    elif rsi > 74:
        signal_text = "Stark überhitzt – Teilgewinne absichern"
        signal_farbe = "red"
        kurz_begruendung = f"RSI bei {rsi:.0f} (überkauft) und bereits {abstand_sma50_pct:+.1f} % über SMA 50 – Rücksetzer-Gefahr."
        gesamtfazit = (
            f"Nach einer steilen Rallye ist {stamm['name']} mit einem RSI von {rsi:.1f} technisch stark überkauft und hat sich "
            f"bereits {abstand_sma50_pct:+.1f} % weit von seiner 50-Tage-Linie ({sma50:.2f} €) entfernt. Neue Käufe auf diesem "
            f"Niveau haben ein erhöhtes Rückschlagrisiko. Wer investiert ist, sollte den Stop-Loss auf {stop_loss_eur:.2f} € "
            f"nachziehen oder Teilgewinne mitnehmen."
        )
    elif aktuell < sma50 and aktuell < sma200:
        signal_text = "Abwärtstrend aktiv – Vorerst abwarten"
        signal_farbe = "red"
        kurz_begruendung = f"Unter SMA 50 ({abstand_sma50_pct:.1f} %) und SMA 200 ({abstand_sma200_pct:.1f} %) – kein Boden bestätigt."
        gesamtfazit = (
            f"{stamm['name']} befindet sich aktuell unter beiden wichtigen Trendlinien (SMA 50 bei {sma50:.2f} € und "
            f"SMA 200 bei {sma200:.2f} €). Solange der Kurs die 50-Tage-Linie nicht zurückerobert und der MACD negativ bleibt, "
            f"dominiert der Verkaufsdruck. Ein Einstieg drängt sich noch nicht auf – erst eine Bodenbildung über {sma50:.2f} € abwarten."
        )
    else:
        richtung_text = (
            "knapp unter der 50-Tage-Linie und testet diese als Widerstand"
            if aktuell < sma50
            else "über der 50-Tage-Linie, kämpft aber noch mit der 200-Tage-Linie"
        )
        gc_zusatz = " Ein Golden Cross steht kurz bevor!" if golden_cross_nah else ""
        signal_text = "Neutral / Konsolidierung – Ausbruch abwarten"
        signal_farbe = "amber"
        kurz_begruendung = f"Kurs {abstand_sma50_pct:+.1f} % zu SMA 50, {abstand_sma200_pct:+.1f} % zu SMA 200 (RSI {rsi:.0f}) – Seitwärtsphase."
        gesamtfazit = (
            f"{stamm['name']} bewegt sich derzeit in einer Übergangs- bzw. Seitwärtsphase: Der Kurs liegt {richtung_text}.{gc_zusatz} "
            f"Mit einem RSI von {rsi:.1f} liegt weder eine Übertreibung noch ein Panikverkauf vor. "
            f"Ein neues Kaufsignal entsteht erst bei einem dynamischen Ausbruch über {max(sma50, sma200):.2f} € unter erhöhtem Volumen."
        )

    # Detaillierte Einzelbausteine für die Detailansicht
    trend_detail = (
        f"Der aktuelle Kurs von <b>{aktuell:.2f} €</b> liegt <b>{abstand_sma50_pct:+.1f} %</b> "
        f"{'über' if aktuell >= sma50 else 'unter'} der 50-Tage-Linie (<b>{sma50:.2f} €</b>) und "
        f"<b>{abstand_sma200_pct:+.1f} %</b> {'über' if aktuell >= sma200 else 'unter'} der 200-Tage-Linie (<b>{sma200:.2f} €</b>). "
        f"Die 50-Tage-Linie verläuft aktuell {'über (+' + f'{sma50_zu_200_pct:.1f} %)' if sma50 >= sma200 else 'unter (' + f'{sma50_zu_200_pct:.1f} %)'} "
        f"der 200-Tage-Linie{' – ein frisches Golden Cross wurde erkannt!' if golden_cross_neu else '.'}"
    )

    if rsi > 70:
        rsi_bewertung = "überkauft (erhöhte Rücksetzer-Gefahr)"
    elif rsi < 30:
        rsi_bewertung = "stark überverkauft (technische Gegenbewegung wahrscheinlich)"
    elif rsi >= 50:
        rsi_bewertung = "bullisch & dynamisch (Käufer haben die Oberhand)"
    else:
        rsi_bewertung = "verhalten / schwach (Käufer halten sich noch zurück)"

    momentum_detail = (
        f"Der <b>RSI (14 Tage)</b> steht bei <b>{rsi:.1f}</b> (vor 5 Tagen: {rsi_vor5:.1f}) und gilt damit als <b>{rsi_bewertung}</b>. "
        f"Das <b>MACD-Histogramm</b> ist aktuell <b>{'positiv' if macd_hist > 0 else 'negativ'}</b> und "
        f"<b>{'steigend (zunehmender Kaufdruck)' if macd_hist > macd_hist_vortag else 'fallend (nachlassender Schwung)'}</b>. "
        f"Das Handelsvolumen erreicht das <b>{volumen_faktor:.2f}-fache</b> des 20-Tage-Schnitts."
    )

    if crv >= 2.0:
        crv_text = "sehr attraktiv (mögliche Gewinnspanne bis zum Jahreshoch ist mindestens doppelt so groß wie das Stop-Loss-Risiko)"
    elif crv >= 1.2:
        crv_text = "solide / moderat (akzeptables Verhältnis, enge Absicherung sinnvoll)"
    else:
        crv_text = "gering (Kurs notiert bereits sehr nah am Jahreshoch oder die Tagesschwankung ist hoch)"

    risiko_detail = (
        f"Die 52-Wochen-Spanne reicht von <b>{tief_52w:.2f} €</b> (+{abstand_tief_pct:.1f} % vom Tief) bis "
        f"<b>{hoch_52w:.2f} €</b> ({abstand_hoch_pct:.1f} % unter Hoch). Basierend auf der aktuellen Schwankungsbreite (ATR: {atr:.2f} €) "
        f"ergibt sich ein sinnvoller Stop-Loss bei <b>{stop_loss_eur:.2f} €</b> (-{stop_puffer_pct:.1f} % Puffer) "
        f"und ein Chance-Risiko-Verhältnis von <b>{crv:.1f} : 1</b> – das ist <b>{crv_text}</b>."
    )

    return {
        "ticker": ticker,
        "name": stamm["name"],
        "region": region,
        "waehrung_orig": stamm["waehrung"],
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
        "stop_puffer_pct": stop_puffer_pct,
        "crv": crv,
        "golden_cross": golden_cross_neu,
        "ist_trend_follower": ist_trend_follower,
        "ist_tfa": ist_tfa,
        "signal_text": signal_text,
        "signal_farbe": signal_farbe,
        "kurz_begruendung": kurz_begruendung,
        "gesamtfazit": gesamtfazit,
        "trend_detail": trend_detail,
        "momentum_detail": momentum_detail,
        "risiko_detail": risiko_detail,
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
        punkte += 15 if sp_aktuell > sp_sma200 else -15
        punkte += 10 if sp_aktuell > sp_sma50 else -10

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


def berechne_backtest_signale(df: pd.DataFrame, strategie: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Erzeugt saubere, nicht-überlappende Kauf-, Verkaufs- und Golden-Cross-Signale
    ohne Kapitalentwicklung.
    """
    df_bt = df.copy()
    sma50_vor = df_bt["SMA_50"].shift(1)
    sma200_vor = df_bt["SMA_200"].shift(1)
    close_vor = df_bt["Close"].shift(1)
    rsi_vor = df_bt["RSI_14"].shift(1)
    macd_vor = df_bt["MACD_Hist"].shift(1)

    # Exakte Golden-Cross- & Death-Cross-Tage (Kreuzungspunkte von SMA 50 & SMA 200)
    df_bt["Golden_Cross"] = (sma50_vor <= sma200_vor) & (df_bt["SMA_50"] > df_bt["SMA_200"])
    df_bt["Death_Cross"] = (sma50_vor >= sma200_vor) & (df_bt["SMA_50"] < df_bt["SMA_200"])

    # Saubere Zustandsmaschine: Kauf- und Verkaufssignale wechseln sich logisch ab (kein Signal-Spam)
    kauf_flags = []
    verkauf_flags = []
    ereignisse = []
    in_position = False
    einstiegspreis = 0.0

    for idx, row in df_bt.iterrows():
        c = float(row["Close"])
        s50 = float(row["SMA_50"]) if not pd.isna(row["SMA_50"]) else c
        s200 = float(row["SMA_200"]) if not pd.isna(row["SMA_200"]) else s50
        r = float(row["RSI_14"])
        mh = float(row["MACD_Hist"])
        gc = bool(row["Golden_Cross"])
        dc = bool(row["Death_Cross"])

        c_prev = float(close_vor.loc[idx]) if not pd.isna(close_vor.loc[idx]) else c
        s50_prev = float(sma50_vor.loc[idx]) if not pd.isna(sma50_vor.loc[idx]) else s50
        r_prev = float(rsi_vor.loc[idx]) if not pd.isna(rsi_vor.loc[idx]) else r
        mh_prev = float(macd_vor.loc[idx]) if not pd.isna(macd_vor.loc[idx]) else mh

        kauf_trigger = False
        verkauf_trigger = False
        grund = ""

        if strategie == "Nur Golden Cross & Death Cross (SMA 50 / SMA 200)":
            if not in_position and gc:
                kauf_trigger = True
                grund = "Golden Cross (SMA 50 kreuzt SMA 200 nach oben)"
            elif in_position and (dc or c < s200 * 0.96):
                verkauf_trigger = True
                grund = "Death Cross" if dc else "Bruch unter SMA 200 (-4 %)"

        elif strategie == "Trend-Pullback & Ausbruch (SMA 50 + RSI)":
            if not in_position:
                if (c_prev <= s50_prev and c > s50 and 42 <= r <= 68):
                    kauf_trigger = True
                    grund = "Ausbruch über SMA 50 im gesunden RSI-Bereich"
                elif (r_prev < 30 and r >= 30):
                    kauf_trigger = True
                    grund = "RSI-Bodenbildung (Erholung aus überverkaufter Zone)"
            else:
                if (c_prev >= s50_prev and c < s50 * 0.985):
                    verkauf_trigger = True
                    grund = "Trendbruch unter SMA 50"
                elif (r > 75 and mh < mh_prev):
                    verkauf_trigger = True
                    grund = "Überkauft (RSI > 75) & fallender MACD"

        else:  # Kombi-Strategie: Golden Cross + Trend & RSI
            if not in_position:
                if gc:
                    kauf_trigger = True
                    grund = "Golden Cross (SMA 50 kreuzt SMA 200 nach oben)"
                elif (s50 > s200 and c_prev <= s50_prev and c > s50 and 40 <= r <= 68):
                    kauf_trigger = True
                    grund = "Trend-Bestätigung über SMA 50 im Aufwärtstrend"
                elif (r_prev < 30 and r >= 30 and c > s200 * 0.90):
                    kauf_trigger = True
                    grund = "Antizyklische RSI-Wende (> 30)"
            else:
                if dc:
                    verkauf_trigger = True
                    grund = "Death Cross (SMA 50 kreuzt SMA 200 nach unten)"
                elif (c < s50 * 0.97 and mh < 0):
                    verkauf_trigger = True
                    grund = "Trendbruch unter SMA 50 mit negativem MACD"
                elif (r > 76 and mh < mh_prev):
                    verkauf_trigger = True
                    grund = "Stark überhitzt (RSI > 76) – Gewinnmitnahme"

        if kauf_trigger:
            in_position = True
            einstiegspreis = c
            kauf_flags.append(True)
            verkauf_flags.append(False)
            ereignisse.append({
                "Datum": idx.strftime("%d.%m.%Y"),
                "Signal": "🟢 KAUF" + (" (★ Golden Cross)" if gc else ""),
                "Kurs (€)": round(c, 2),
                "SMA 50 (€)": round(s50, 2),
                "SMA 200 (€)": round(s200, 2),
                "RSI (14)": round(r, 1),
                "Auslöser": grund,
                "Ergebnis seit Kauf": "Offene Position",
            })
        elif verkauf_trigger:
            in_position = False
            diff_pct = ((c - einstiegspreis) / einstiegspreis * 100) if einstiegspreis > 0 else 0.0
            kauf_flags.append(False)
            verkauf_flags.append(True)
            ereignisse.append({
                "Datum": idx.strftime("%d.%m.%Y"),
                "Signal": "🔴 VERKAUF",
                "Kurs (€)": round(c, 2),
                "SMA 50 (€)": round(s50, 2),
                "SMA 200 (€)": round(s200, 2),
                "RSI (14)": round(r, 1),
                "Auslöser": grund,
                "Ergebnis seit Kauf": f"{diff_pct:+.2f} %",
            })
        else:
            kauf_flags.append(False)
            verkauf_flags.append(False)
            if gc:
                ereignisse.append({
                    "Datum": idx.strftime("%d.%m.%Y"),
                    "Signal": "⭐ GOLDEN CROSS",
                    "Kurs (€)": round(c, 2),
                    "SMA 50 (€)": round(s50, 2),
                    "SMA 200 (€)": round(s200, 2),
                    "RSI (14)": round(r, 1),
                    "Auslöser": "SMA 50 kreuzt SMA 200 nach oben (bereits investiert)",
                    "Ergebnis seit Kauf": "Trend-Bestätigung",
                })

    df_bt["Signal_Kauf"] = kauf_flags
    df_bt["Signal_Verkauf"] = verkauf_flags
    df_ereignisse = pd.DataFrame(ereignisse)
    return df_bt, df_ereignisse


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


# ==============================================================================
# SESSION STATE & NAVIGATION-CALLBACKS
# ==============================================================================
BEREICHE = [
    "1. Markt-Überblick & Stimmung",
    "2. Trend- & Turnaround-Scanner",
    "3. Einzelaktien-Analyse (€)",
    "4. Strategie-Backtesting & Signale",
    "5. Krypto-ETFs & Kapitalflüsse",
]

if "ausgewaehlter_ticker" not in st.session_state:
    st.session_state["ausgewaehlter_ticker"] = "1810.HK"

if "einzel_suchfeld" not in st.session_state:
    st.session_state["einzel_suchfeld"] = "Xiaomi"

if "nav_bereich" not in st.session_state:
    st.session_state["nav_bereich"] = BEREICHE[0]


def wechsele_zu_einzelanalyse(ticker: str, anzeige_name: str):
    st.session_state["ausgewaehlter_ticker"] = ticker
    st.session_state["einzel_suchfeld"] = anzeige_name
    st.session_state["nav_bereich"] = "3. Einzelaktien-Analyse (€)"


def wechsele_zu_backtesting(ticker: str):
    st.session_state["ausgewaehlter_ticker"] = ticker
    st.session_state["nav_bereich"] = "4. Strategie-Backtesting & Signale"


# ==============================================================================
# SEITENLEISTE
# ==============================================================================
with st.sidebar:
    st.title("📈 Börsen-Cockpit (€)")
    st.caption("Alle Kurse automatisch in Euro (€) • Inkl. China & Japan")

    bereich = st.radio("Bereich wählen:", BEREICHE, key="nav_bereich")

    st.divider()
    if st.button("🔄 Kurse jetzt aktualisieren"):
        st.cache_data.clear()
        st.rerun()

    with st.expander("ℹ️ Kurze Erklärung der Begriffe"):
        st.markdown(
            """
            * **SMA 50 / SMA 200:** Durchschnittskurs der letzten 50 bzw. 200 Handelstage. Liegt der Kurs darüber, ist der Trend positiv.
            * **Golden Cross (★):** Die 50-Tage-Linie kreuzt die 200-Tage-Linie nach oben – ein starkes mittelfristiges Kaufsignal.
            * **RSI (14):** Misst, ob eine Aktie überkauft (> 70) oder überverkauft (< 30) ist.
            * **TFA (Turnaround-Formel):** Stark gefallene Aktien (mind. 35 % unter Jahreshoch), die einen Boden bilden.
            * **CRV (Chance-Risiko-Verhältnis):** Gewinnpotenzial bis zum 52W-Hoch im Vergleich zum Stop-Loss-Risiko (ab 2,0 : 1 attraktiv).
            """
        )

# ==============================================================================
# BEREICH 1: MARKT-ÜBERBLICK & SENTIMENT-TACHOMETER
# ==============================================================================
if bereich == "1. Markt-Überblick & Stimmung":
    st.header("🧭 Markt-Überblick & Stimmungs-Tachometer")
    st.write("Aktuelle Großwetterlage an den internationalen Börsen – live berechnet.")

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
        st.markdown(
            f"<div class='metric-card'><h4>{sentiment['label']}</h4>"
            f"<p>Angst-Barometer (VIX): <b>{sentiment['vix']}</b> (Unter 20 bedeutet ruhiges Fahrwasser)</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='info-box'>
            <b>Was bedeutet das für deine Strategie?</b><br>
            • Bei <b>Angst (unter 40)</b> lohnen sich antizyklische Turnaround-Kandidaten (TFA-Scanner).<br>
            • Bei <b>Optimismus (55–75)</b> laufen Trend-Follower & Golden-Cross-Signale am zuverlässigsten.<br>
            • Bei <b>Extremer Gier (über 80)</b> keine Kurse jagen, sondern Stop-Loss-Marken eng nachziehen.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Schnell-Check der Welt-Leitindizes")
    indizes = [
        ("DAX 🇩🇪", "^GDAXI"),
        ("S&P 500 🇺🇸", "^GSPC"),
        ("Hang Seng (HK/CN) 🇨🇳", "^HSI"),
        ("Nikkei 225 (Japan) 🇯🇵", "^N225"),
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
# BEREICH 2: TREND-FOLLOWER & TFA-SCANNER (INKL. CHINA & JAPAN IN €)
# ==============================================================================
elif bereich == "2. Trend- & Turnaround-Scanner":
    st.header("🔍 Aktien-Scanner: Trend-Follower & Turnaround-Formel (TFA)")
    st.caption("Alle Kurse aus Hongkong (HKD), Japan (JPY), USA (USD) und Europa werden live in Euro (€) umgerechnet.")

    col_u, col_m = st.columns([1.4, 1.6])
    with col_u:
        universum_name = st.selectbox("Markt / Aktien-Liste auswählen:", list(INDEX_UNIVERSEN.keys()))
    with col_m:
        scanner_modus = st.radio(
            "Strategie-Filter:",
            [
                "Alle Aktien anzeigen",
                "Nur Trend-Follower & Golden Cross",
                "Nur TFA-Signale (Turnaround-Chancen)",
            ],
            horizontal=True,
        )

    eigene_eingabe = st.text_input(
        "Optional: Weitere Aktien ergänzen (Name oder Kürzel mit Komma, z. B. Xiaomi, BYD, Toyota, RHM.DE):",
        value="",
    )

    ticker_liste = list(INDEX_UNIVERSEN[universum_name])
    if eigene_eingabe.strip():
        for teil in eigene_eingabe.split(","):
            if teil.strip():
                gef_tkr, _ = suche_ticker_aus_name(teil.strip())
                if gef_tkr not in ticker_liste:
                    ticker_liste.append(gef_tkr)

    ergebnisse = []
    with st.spinner("Lade Kurse, rechne HKD / JPY / USD in Euro (€) um und berechne Signale..."):
        for t in ticker_liste:
            res = analysiere_aktie(t)
            if res:
                ergebnisse.append(res)

    if scanner_modus == "Nur Trend-Follower & Golden Cross":
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
                "Markt": r["region"],
                "Kurs (€)": round(r["kurs_eur"], 2),
                "Heute (%)": round(r["tages_diff_pct"], 2),
                "RSI (14)": round(r["rsi"], 1),
                "Abstand 52W-Hoch": f"{r['abstand_hoch_pct']:.1f} %",
                "Stop-Loss (€)": round(r["stop_loss_eur"], 2),
                "CRV": f"{r['crv']:.1f} : 1",
                "Gesamturteil": r["signal_text"],
                "Begründung im Detail": r["kurz_begruendung"],
            })

        df_anzeige = pd.DataFrame(tabellen_daten)
        st.dataframe(df_anzeige, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🔎 Treffer im Detail prüfen & direkt öffnen")
        auswahl_map = {f"{r['name']} ({r['ticker']}) – {r['kurs_eur']:.2f} €": r for r in gefiltert}
        gewaehltes_label = st.selectbox("Aktie auswählen:", list(auswahl_map.keys()))
        sel = auswahl_map[gewaehltes_label]

        st.markdown(
            f"""
            <div class='verdict-box'>
                <b>{sel['name']} ({sel['ticker']}) • {sel['region']}</b> —
                <span class='badge-{sel['signal_farbe']}'>{sel['signal_text']}</span><br><br>
                {sel['gesamtfazit']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_c1, btn_c2, _ = st.columns([1.3, 1.3, 2])
        with btn_c1:
            st.button(
                "➡️ Zur Einzelaktien-Analyse wechseln",
                on_click=wechsele_zu_einzelanalyse,
                args=(sel["ticker"], sel["name"]),
                use_container_width=True,
            )
        with btn_c2:
            st.button(
                "📊 Im Backtesting-Chart öffnen",
                on_click=wechsele_zu_backtesting,
                args=(sel["ticker"],),
                use_container_width=True,
            )

# ==============================================================================
# BEREICH 3: EINZELAKTIEN-ANALYSE (NAMENSSUCHE + AUSFÜHRLICHE EINSCHÄTZUNG IN €)
# ==============================================================================
elif bereich == "3. Einzelaktien-Analyse (€)":
    st.header("🔬 Einzelaktien-Analyse (Alle Kurse in €)")

    col_in, col_quick, col_btn = st.columns([1.8, 1.5, 0.9])
    with col_in:
        such_text = st.text_input(
            "Aktienname ausgeschrieben oder Kürzel eingeben (z. B. Xiaomi, BYD, Toyota, Nvidia, RHM.DE):",
            key="einzel_suchfeld",
        )
    with col_quick:
        schnell_wahl = st.selectbox(
            "Oder aus Schnellauswahl wählen:",
            ["— Eigene Suche oben nutzen —"] + list(SCHNELLAUSWAHL_AKTIEN.keys()),
        )
    with col_btn:
        st.write("")
        st.write("")
        suchen_geklickt = st.button("Aktie analysieren", use_container_width=True)

    if schnell_wahl != "— Eigene Suche oben nutzen —" and not suchen_geklickt:
        ziel_ticker = SCHNELLAUSWAHL_AKTIEN[schnell_wahl]
        aufgeloester_name = schnell_wahl
    else:
        ziel_ticker, aufgeloester_name = suche_ticker_aus_name(such_text)

    st.session_state["ausgewaehlter_ticker"] = ziel_ticker

    with st.spinner(f"Analysiere {aufgeloester_name} ({ziel_ticker}) in Euro (€)..."):
        daten = analysiere_aktie(ziel_ticker)

    if not daten:
        st.error(
            f"Für '{such_text}' (aufgelöst als '{ziel_ticker}') konnten keine Kursdaten geladen werden. "
            "Prüfe bitte die Schreibweise oder gib direkt das Börsenkürzel ein (z. B. 1810.HK für Xiaomi, 1211.HK für BYD)."
        )
    else:
        st.subheader(f"{daten['name']} ({daten['ticker']})")
        st.caption(
            f"{daten['region']} • Branche: {daten['sektor']} • "
            f"Originalwährung {daten['waehrung_orig']} automatisch in Euro (€) umgerechnet"
        )

        if daten["earnings_warnung"]:
            st.markdown(
                f"<div class='warn-box'>⚠️ ACHTUNG: {daten['earnings_warnung']} Kurz vor Quartalszahlen sind starke Kurssprünge möglich.</div>",
                unsafe_allow_html=True,
            )
        elif daten["earnings_datum"]:
            st.caption(f"📅 Nächster Quartalszahlen-Termin: {daten['earnings_datum']}")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Aktueller Kurs", f"{daten['kurs_eur']:.2f} €", f"{daten['tages_diff_pct']:+.2f} %")
        k2.metric(
            "RSI (14 Tage)",
            f"{daten['rsi']:.1f}",
            "Überkauft > 70" if daten["rsi"] > 70 else ("Überverkauft < 30" if daten["rsi"] < 30 else "Gesund"),
        )
        k3.metric("ATR Stop-Loss", f"{daten['stop_loss_eur']:.2f} €", f"-{daten['stop_puffer_pct']:.1f} % Puffer")
        k4.metric("Chance-Risiko (CRV)", f"{daten['crv']:.1f} : 1", "Attraktiv ≥ 2,0" if daten["crv"] >= 2.0 else "Moderat / Eng")
        k5.metric("52-Wochen-Hoch", f"{daten['hoch_52w_eur']:.2f} €", f"{daten['abstand_hoch_pct']:.1f} %")

        # Ausführliches Gesamturteil & detaillierte Begründung
        st.markdown(
            f"""
            <div class='metric-card'>
                <h4>Gesamturteil: <span class='badge-{daten["signal_farbe"]}'>{daten["signal_text"]}</span></h4>
                <p style='font-size: 1.0rem; line-height: 1.6; margin-top: 10px;'>
                    <b>Ausführliche Einschätzung & Handlungsempfehlung:</b><br>
                    {daten['gesamtfazit']}
                </p>
                <hr style='border-color: #30363D; margin: 14px 0;'>
                <ul style='line-height: 1.65;'>
                    <li><b>1. Trend- & Linien-Check (SMA 50 / SMA 200):</b> {daten['trend_detail']}</li>
                    <li><b>2. Schwung, RSI & Volumen:</b> {daten['momentum_detail']}</li>
                    <li><b>3. 52-Wochen-Spanne & Risiko-Steuerung:</b> {daten['risiko_detail']}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "📊 Kauf- & Verkaufssignale für diese Aktie im Backtesting-Chart ansehen",
            on_click=wechsele_zu_backtesting,
            args=(daten["ticker"],),
        )

        # Kurschart in Euro mit SMA 50, SMA 200 & Stop-Loss
        df_plot = daten["df"].tail(200)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
            subplot_titles=(
                f"{daten['name']} – Kursverlauf in Euro (€) mit 50- & 200-Tage-Linie",
                "RSI (14) – Überkauft (> 70) / Überverkauft (< 30)",
            ),
        )

        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Kurs (€)", line=dict(color="#58A6FF", width=2.5)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["SMA_50"], name="SMA 50 (€)", line=dict(color="#FFB300", width=1.8)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["SMA_200"], name="SMA 200 (€)", line=dict(color="#CE93D8", width=1.8)),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=daten["stop_loss_eur"],
            line_dash="dash",
            line_color="#FF5252",
            annotation_text=f"ATR Stop-Loss: {daten['stop_loss_eur']:.2f} €",
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
            height=600,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=55, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(gridcolor="#21262D")
        fig.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# BEREICH 4: BACKTESTING & SIGNAL-CHART (OHNE KAPITALENTWICKLUNG, MIT GOLDEN CROSS)
# ==============================================================================
elif bereich == "4. Strategie-Backtesting & Signale":
    st.header("📊 Signal-Backtesting: Kauf-, Verkaufs- & Golden-Cross-Signale")
    st.write("Übersichtliche Darstellung aller historischen Einstiegs-, Ausstiegs- und Golden-Cross-Punkte direkt im Euro-Chart.")

    c_s1, c_s2, c_s3 = st.columns([1.4, 1.6, 1.0])
    with c_s1:
        bt_eingabe = st.text_input(
            "Aktie (Name oder Kürzel, z. B. Xiaomi, BYD, NVDA):",
            value=st.session_state["ausgewaehlter_ticker"],
        )
        bt_ticker, bt_name_hint = suche_ticker_aus_name(bt_eingabe)
        st.session_state["ausgewaehlter_ticker"] = bt_ticker
    with c_s2:
        bt_strategie = st.selectbox(
            "Signal-Strategie wählen:",
            [
                "Kombi-Strategie: Golden Cross + Trend & RSI (Empfohlen)",
                "Nur Golden Cross & Death Cross (SMA 50 / SMA 200)",
                "Trend-Pullback & Ausbruch (SMA 50 + RSI)",
            ],
        )
    with c_s3:
        bt_zeitraum_label = st.selectbox(
            "Anzeige-Zeitraum:",
            ["1 Jahr", "2 Jahre", "3 Jahre"],
            index=1,
        )

    tage_map = {"1 Jahr": 252, "2 Jahre": 504, "3 Jahre": 756}
    anzeige_tage = tage_map[bt_zeitraum_label]

    # 5 Jahre laden, damit SMA 200 bereits ab dem ersten sichtbaren Tag vollständig berechnet ist
    df_lang = lade_kursdaten_eur(bt_ticker, zeitraum="5y")
    stamm_bt = hole_stammdaten(bt_ticker)

    if df_lang.empty or len(df_lang) < 60:
        st.error(f"Für '{bt_eingabe}' ({bt_ticker}) konnten keine ausreichenden historischen Daten geladen werden.")
    else:
        df_sichtbar = df_lang.tail(anzeige_tage).copy()
        df_bt, df_ereignisse = berechne_backtest_signale(df_sichtbar, bt_strategie)

        kauf_df = df_bt[df_bt["Signal_Kauf"]]
        verkauf_df = df_bt[df_bt["Signal_Verkauf"]]
        gc_df = df_bt[df_bt["Golden_Cross"]]

        # Kompakte Signal-Kennzahlen (ohne Kapitalentwicklung)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Aktueller Kurs (€)", f"{float(df_bt['Close'].iloc[-1]):.2f} €")
        s2.metric("🟢 Kaufsignale", f"{len(kauf_df)} Signale")
        s3.metric("🔴 Verkaufssignale", f"{len(verkauf_df)} Signale")
        s4.metric("⭐ Golden Crosses", f"{len(gc_df)} Kreuzungen")

        # Hochauflösender, klar lesbarer Signal-Chart
        fig_bt = go.Figure()

        # 1. Kurslinie mit dezenter Flächenfüllung
        fig_bt.add_trace(
            go.Scatter(
                x=df_bt.index,
                y=df_bt["Close"],
                name="Kurs (€)",
                mode="lines",
                line=dict(color="#58A6FF", width=2.6),
                fill="tozeroy",
                fillcolor="rgba(88, 166, 255, 0.06)",
                hovertemplate="<b>Kurs:</b> %{y:.2f} €<extra></extra>",
            )
        )

        # 2. Gleitende Durchschnitte SMA 50 & SMA 200
        fig_bt.add_trace(
            go.Scatter(
                x=df_bt.index,
                y=df_bt["SMA_50"],
                name="SMA 50 (50-Tage-Linie)",
                mode="lines",
                line=dict(color="#FFB300", width=1.8),
                hovertemplate="<b>SMA 50:</b> %{y:.2f} €<extra></extra>",
            )
        )
        fig_bt.add_trace(
            go.Scatter(
                x=df_bt.index,
                y=df_bt["SMA_200"],
                name="SMA 200 (200-Tage-Linie)",
                mode="lines",
                line=dict(color="#CE93D8", width=2.0),
                hovertemplate="<b>SMA 200:</b> %{y:.2f} €<extra></extra>",
            )
        )

        # 3. Kaufsignale (grüne Dreiecke klar unterhalb der Tageskerze)
        if not kauf_df.empty:
            kauf_y = kauf_df["Low"] * 0.978
            zeige_kauf_text = len(kauf_df) <= 10
            fig_bt.add_trace(
                go.Scatter(
                    x=kauf_df.index,
                    y=kauf_y,
                    name="🟢 Kaufsignal (Einstieg)",
                    mode="markers+text" if zeige_kauf_text else "markers",
                    text=["KAUF"] * len(kauf_df) if zeige_kauf_text else None,
                    textposition="bottom center",
                    textfont=dict(color="#00E676", size=11, family="Arial Black"),
                    marker=dict(
                        symbol="triangle-up",
                        size=16,
                        color="#00E676",
                        line=dict(color="#FFFFFF", width=1.5),
                    ),
                    customdata=kauf_df["Close"],
                    hovertemplate="<b>🟢 KAUFSIGNAL</b><br>Kurs: %{customdata:.2f} €<extra></extra>",
                )
            )

        # 4. Verkaufssignale (rote Dreiecke klar oberhalb der Tageskerze)
        if not verkauf_df.empty:
            verkauf_y = verkauf_df["High"] * 1.022
            zeige_verkauf_text = len(verkauf_df) <= 10
            fig_bt.add_trace(
                go.Scatter(
                    x=verkauf_df.index,
                    y=verkauf_y,
                    name="🔴 Verkaufssignal (Ausstieg)",
                    mode="markers+text" if zeige_verkauf_text else "markers",
                    text=["VERKAUF"] * len(verkauf_df) if zeige_verkauf_text else None,
                    textposition="top center",
                    textfont=dict(color="#FF5252", size=11, family="Arial Black"),
                    marker=dict(
                        symbol="triangle-down",
                        size=16,
                        color="#FF5252",
                        line=dict(color="#FFFFFF", width=1.5),
                    ),
                    customdata=verkauf_df["Close"],
                    hovertemplate="<b>🔴 VERKAUFSSIGNAL</b><br>Kurs: %{customdata:.2f} €<extra></extra>",
                )
            )

        # 5. Golden Cross Markierungen (Goldener Stern + vertikale Linie direkt am SMA-Schnittpunkt)
        if not gc_df.empty:
            fig_bt.add_trace(
                go.Scatter(
                    x=gc_df.index,
                    y=gc_df["SMA_50"],
                    name="⭐ Golden Cross (SMA 50 > SMA 200)",
                    mode="markers+text",
                    text=["★ Golden Cross"] * len(gc_df),
                    textposition="top right",
                    textfont=dict(color="#FFD700", size=12, family="Arial Black"),
                    marker=dict(
                        symbol="star",
                        size=20,
                        color="#FFD700",
                        line=dict(color="#0E1117", width=1.5),
                    ),
                    customdata=gc_df["Close"],
                    hovertemplate="<b>⭐ GOLDEN CROSS</b><br>SMA 50 kreuzt SMA 200<br>Aktienkurs: %{customdata:.2f} €<extra></extra>",
                )
            )
            for gc_datum in gc_df.index:
                fig_bt.add_vline(
                    x=gc_datum,
                    line_dash="dot",
                    line_width=1.6,
                    line_color="rgba(255, 215, 0, 0.65)",
                )

        fig_bt.update_layout(
            title=dict(
                text=f"{stamm_bt['name']} ({bt_ticker}) – Kauf-, Verkaufs- & Golden-Cross-Signale in Euro (€)",
                font=dict(size=18, color="#E6EDF3"),
            ),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3", size=13),
            height=640,
            hovermode="x unified",
            margin=dict(l=25, r=25, t=75, b=25),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(22, 27, 34, 0.85)",
                bordercolor="#30363D",
                borderwidth=1,
            ),
            xaxis=dict(
                gridcolor="#21262D",
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#8B949E",
                spikethickness=1,
                rangeselector=dict(
                    bgcolor="#161B22",
                    activecolor="#58A6FF",
                    font=dict(color="#E6EDF3"),
                    buttons=[
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1J", step="year", stepmode="backward"),
                        dict(step="all", label="Alle"),
                    ],
                ),
            ),
            yaxis=dict(
                title="Kurs in Euro (€)",
                gridcolor="#21262D",
                tickformat=".2f",
                ticksuffix=" €",
            ),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        st.subheader("📋 Chronologische Übersicht aller Signale")
        if df_ereignisse.empty:
            st.info("Im gewählten Zeitraum wurden für diese Strategie keine Signale ausgelöst. Wähle oben '2 Jahre' oder '3 Jahre'.")
        else:
            st.dataframe(df_ereignisse.iloc[::-1], use_container_width=True, hide_index=True)

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
            height=420,
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
