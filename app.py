"""
Stock Screener Pro
------------------
Lokales Streamlit-Dashboard zum Finden und Prüfen von Aktien.

Features:
- Einzelanalyse: Chart (€), LONG/FLAT/SHORT, Sentiment, Zielpfad, Scores
- Scanner: US/EU/Asien + Qualitätsfilter + Composite-Score
- Vergleichs-Matrix, Portfolio-Tracking
- Vertieftes Backtesting (mehrere Signale, Equity-Kurve, Profit Factor)
- Branchenvergleich, Piotroski, Levermann, Risiko-Kennzahlen

Start:
    python -m streamlit run app.py
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Screener Pro", layout="wide", page_icon="📈")

# Plotly: mobil-freundlich (Pinch-Zoom, Pan) + klare Darstellung
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "doubleClick": "reset",
    # Zoom/Pan explizit behalten – wichtig für Touch
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
    "toImageButtonOptions": {"format": "png", "filename": "chart", "scale": 2},
}

# Extra-Config für große/Vollbild-Ansicht (gleiche Basis)
PLOTLY_CONFIG_FULL = {
    **PLOTLY_CONFIG,
    "displayModeBar": True,
}

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="Inter, system-ui, sans-serif", size=14, color="#e8e8e8"),
    margin=dict(l=40, r=56, t=56, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(14,17,23,0.85)",
        font=dict(size=12),
        itemsizing="constant",
    ),
    hovermode="x unified",
    # pan = mit Finger schieben; Zoom über Modebar-Buttons oder Range-Slider
    dragmode="pan",
    spikedistance=50,
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(255,255,255,0.35)",
        spikethickness=1,
        fixedrange=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        side="right",
        showspikes=True,
        spikemode="across",
        spikecolor="rgba(255,255,255,0.25)",
        spikethickness=1,
        fixedrange=False,
    ),
)

# ---------------------------------------------------------------------------
# Lokale Speicherung
# ---------------------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Universen & Namen-Erkennung
# ---------------------------------------------------------------------------
BUILTIN_WATCHLISTS = {
    "Tech-Auswahl": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "CRM"],
    "DAX-Auswahl (Top 10)": ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BMW.DE", "MBG.DE", "VOW3.DE", "MUV2.DE", "AIR.DE"],
    "Europa Blue Chips": [
        "SAP.DE", "SIE.DE", "ASML.AS", "LVMH.PA", "MC.PA", "OR.PA", "TTE.PA", "SAN.PA",
        "NESN.SW", "ROG.SW", "NOVN.SW", "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "INGA.AS",
    ],
    "S&P-Auswahl (Auszug)": ["AAPL", "MSFT", "AMZN", "GOOGL", "BRK-B", "JPM", "V", "JNJ", "WMT", "XOM"],
    "Asia Tech": ["TSM", "ASML", "SONY", "NTDOY", "BABA", "TCEHY", "PDD", "BIDU", "005930.KS", "6861.T"],
    "Japan (Nikkei-Auswahl)": ["7203.T", "6758.T", "9984.T", "6861.T", "8306.T", "9432.T", "4063.T", "6501.T", "7267.T", "6902.T"],
    "Hongkong / China": ["0700.HK", "9988.HK", "3690.HK", "9618.HK", "1810.HK", "BABA", "JD", "PDD", "BIDU", "NIO"],
}

DAX40_TICKERS = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE",
    "CBK.DE", "CON.DE", "1COV.DE", "DHL.DE", "DBK.DE", "DB1.DE", "DTE.DE", "DTG.DE",
    "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE",
    "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE", "SRT3.DE",
    "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE", "G24.DE",
]

# Frankreich CAC40 (Auswahl, Yahoo-Ticker)
CAC40_TICKERS = [
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "SU.PA", "AI.PA", "BNP.PA",
    "DG.PA", "EL.PA", "CS.PA", "BN.PA", "KER.PA", "RI.PA", "STLAP.PA", "VIV.PA",
    "CAP.PA", "SGO.PA", "HO.PA", "ML.PA",
]

# Niederlande / Belgien
BENELUX_TICKERS = [
    "ASML.AS", "INGA.AS", "PHIA.AS", "AD.AS", "UNA.AS", "HEIA.AS", "PRX.AS",
    "KBC.BR", "ABI.BR", "ARGX.BR",
]

# Schweiz
SWISS_TICKERS = [
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "LONN.SW",
    "CFR.SW", "SIKA.SW", "GIVN.SW",
]

# UK (FTSE-Auswahl; Yahoo nutzt oft .L)
UK_TICKERS = [
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "DGE.L", "RIO.L",
    "BATS.L", "LLOY.L", "VOD.L", "NG.L", "BARC.L", "RR.L", "AAL.L",
]

# Europa gesamt (Scanner-Universum)
EUROPE_TICKERS = list(dict.fromkeys(
    DAX40_TICKERS + CAC40_TICKERS + BENELUX_TICKERS + SWISS_TICKERS + UK_TICKERS
))

NASDAQ_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP", "COST",
    "ADBE", "CSCO", "NFLX", "AMD", "INTC", "QCOM", "TXN", "INTU", "AMAT", "BKNG",
]

TOP10_CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
]

ASIA_TECH = ["TSM", "ASML", "SONY", "NTDOY", "BABA", "TCEHY", "PDD", "BIDU", "005930.KS", "6861.T", "2330.TW"]
JAPAN_TICKERS = ["7203.T", "6758.T", "9984.T", "6861.T", "8306.T", "9432.T", "4063.T", "6501.T", "7267.T", "6902.T"]
HK_CHINA = ["0700.HK", "9988.HK", "3690.HK", "9618.HK", "1810.HK", "BABA", "JD", "PDD", "BIDU", "NIO"]

SP500_FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "JNJ", "WMT", "XOM", "UNH", "MA", "PG", "HD", "COST", "MRK", "ABBV",
    "CVX", "PEP", "KO", "BAC", "ADBE", "CRM", "TMO", "MCD", "CSCO", "ABT",
]


@st.cache_data(ttl=86400)
def fetch_sp500_tickers():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        symbols = tables[0]["Symbol"].tolist()
        return [s.replace(".", "-") for s in symbols]
    except Exception:
        return SP500_FALLBACK


COMMON_NAMES = {
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "AMAZON": "AMZN", "NVIDIA": "NVDA", "META": "META", "FACEBOOK": "META",
    "TESLA": "TSLA", "NETFLIX": "NFLX", "SAP": "SAP.DE", "SIEMENS": "SIE.DE",
    "ALLIANZ": "ALV.DE", "DEUTSCHE TELEKOM": "DTE.DE", "TELEKOM": "DTE.DE",
    "BASF": "BAS.DE", "BMW": "BMW.DE", "MERCEDES": "MBG.DE", "MERCEDES-BENZ": "MBG.DE",
    "VOLKSWAGEN": "VOW3.DE", "VW": "VOW3.DE", "MUNICH RE": "MUV2.DE", "AIRBUS": "AIR.DE",
    "AMD": "AMD", "SALESFORCE": "CRM", "BERKSHIRE": "BRK-B", "JPMORGAN": "JPM",
    "VISA": "V", "JOHNSON & JOHNSON": "JNJ", "WALMART": "WMT", "EXXON": "XOM",
    "COCA-COLA": "KO", "COCA COLA": "KO", "DISNEY": "DIS", "NIKE": "NKE",
    "MCDONALD'S": "MCD", "MCDONALDS": "MCD", "STARBUCKS": "SBUX", "INTEL": "INTC",
    "TSMC": "TSM", "TAIWAN SEMICONDUCTOR": "TSM", "SAMSUNG": "005930.KS",
    "SONY": "SONY", "TOYOTA": "7203.T", "SOFTBANK": "9984.T", "KEYENCE": "6861.T",
    "ALIBABA": "BABA", "TENCENT": "TCEHY", "NINTENDO": "NTDOY", "ASML": "ASML.AS",
    "BYD": "BYDDY", "XIAOMI": "1810.HK", "MEITUAN": "3690.HK",
    "LVMH": "MC.PA", "LOUIS VUITTON": "MC.PA", "LOREAL": "OR.PA", "L'OREAL": "OR.PA",
    "TOTALENERGIES": "TTE.PA", "TOTAL": "TTE.PA", "SANOFI": "SAN.PA",
    "NESTLE": "NESN.SW", "NESTLÉ": "NESN.SW", "ROCHE": "ROG.SW", "NOVARTIS": "NOVN.SW",
    "ASTRAZENECA": "AZN.L", "SHELL": "SHEL.L", "HSBC": "HSBA.L", "UNILEVER": "ULVR.L",
    "BP": "BP.L", "ING": "INGA.AS", "PHILIPS": "PHIA.AS", "HERMES": "RMS.PA",
    "HERMÈS": "RMS.PA", "AIRBUS": "AIR.PA", "SCHNEIDER": "SU.PA", "BNP": "BNP.PA",
    "BNP PARIBAS": "BNP.PA", "UBS": "UBSG.SW", "ZURICH": "ZURN.SW", "ABB": "ABBN.SW",
    "INFINEON": "IFX.DE", "DEUTSCHE BANK": "DBK.DE", "COMMERZBANK": "CBK.DE",
    "ADIDAS": "ADS.DE", "BAYER": "BAYN.DE", "HENKEL": "HEN3.DE", "MERCK": "MRK.DE",
}


@st.cache_data(ttl=86400)
def resolve_ticker(query: str) -> str:
    query = query.strip()
    if not query:
        return ""
    upper = query.upper()
    if upper in COMMON_NAMES:
        return COMMON_NAMES[upper]
    looks_like_ticker = " " not in query and len(query) <= 10
    try:
        search_results = yf.Search(query, max_results=5).quotes
        if search_results:
            equities = [r for r in search_results if r.get("quoteType") == "EQUITY"]
            best = equities[0] if equities else search_results[0]
            symbol = best.get("symbol")
            if symbol:
                return symbol
    except Exception:
        pass
    return upper if looks_like_ticker else query


# ---------------------------------------------------------------------------
# Kursdaten + technische Indikatoren
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_price_history(ticker: str, period: str = "2y"):
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        return data

    data["SMA50"] = data["Close"].rolling(window=50).mean()
    data["SMA200"] = data["Close"].rolling(window=200).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]
    data["Return"] = data["Close"].pct_change()
    return data


def detect_golden_cross(data: pd.DataFrame, lookback_days: int = 10) -> bool:
    df = data.dropna(subset=["SMA50", "SMA200"])
    if len(df) < lookback_days + 1:
        return False
    recent = df.tail(lookback_days + 1)
    diff = recent["SMA50"] - recent["SMA200"]
    return bool(((diff.shift(1) < 0) & (diff > 0)).any())


def is_bullish_now(data: pd.DataFrame) -> bool:
    df = data.dropna(subset=["SMA50", "SMA200"])
    if df.empty:
        return False
    last = df.iloc[-1]
    return bool(last["SMA50"] > last["SMA200"])


def compute_ta_signal(data: pd.DataFrame) -> str:
    if data.empty or "RSI" not in data.columns:
        return "—"
    bullish = is_bullish_now(data)
    gc = detect_golden_cross(data)
    rsi_series = data["RSI"].dropna()
    last_rsi = rsi_series.iloc[-1] if not rsi_series.empty else None
    if gc or (bullish and (last_rsi is None or last_rsi < 70)):
        return "🟢 Kaufsignal"
    if (not bullish) and (last_rsi is None or last_rsi > 40):
        return "🔴 Verkaufssignal"
    return "🟡 Neutral"


def compute_fib_entry_zone(data: pd.DataFrame, lookback_days: int = 126):
    recent = data.tail(lookback_days)
    if recent.empty:
        return None
    swing_high = recent["High"].max()
    swing_low = recent["Low"].min()
    diff = swing_high - swing_low
    if diff <= 0:
        return None
    fib_50 = swing_high - 0.5 * diff
    fib_618 = swing_high - 0.618 * diff
    return min(fib_50, fib_618), max(fib_50, fib_618)


def compute_risk_metrics(data: pd.DataFrame):
    if data.empty or "Return" not in data.columns:
        return None, None
    rets = data["Return"].dropna()
    if len(rets) < 20:
        return None, None
    vol = float(rets.std() * np.sqrt(252) * 100)
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = ((cum - peak) / peak * 100).min()
    return round(vol, 1), round(dd, 1)


@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def get_fx_rates() -> dict:
    """
    FX-Kurse relativ zu EUR (wie viele Fremdwährungseinheiten für 1 EUR).
    Damit: EUR-Wert = Preis_fremd / rate[currency]
    Fallback-Werte falls Yahoo nicht erreichbar.
    """
    defaults = {
        "USD": 1.08,
        "GBP": 0.86,
        "CHF": 0.94,
        "JPY": 160.0,
        "HKD": 8.4,
        "KRW": 1450.0,
        "TWD": 34.0,
        "EUR": 1.0,
    }
    pairs = {
        "USD": "EURUSD=X",
        "GBP": "EURGBP=X",
        "CHF": "EURCHF=X",
        "JPY": "EURJPY=X",
        "HKD": "EURHKD=X",
        "KRW": "EURKRW=X",
        "TWD": "EURTWD=X",
    }
    rates = dict(defaults)
    for cur, pair in pairs.items():
        try:
            h = yf.Ticker(pair).history(period="5d")
            if not h.empty:
                rates[cur] = float(h["Close"].iloc[-1])
        except Exception:
            pass
    rates["EUR"] = 1.0
    return rates


def get_eur_usd_rate() -> float:
    """Kompatibilität: EUR/USD aus get_fx_rates."""
    return float(get_fx_rates().get("USD", 1.08))


def detect_currency(ticker: str, info: dict | None = None) -> str:
    """Best-effort Währung: EUR/JPY/HKD/KRW/TWD aus Suffix oder info, sonst USD."""
    t = ticker.upper()
    if t.endswith((".DE", ".PA", ".AS", ".BR", ".MI", ".MC", ".HE", ".ST", ".OL", ".LS", ".VI")):
        return "EUR"
    if t.endswith(".L"):
        return "GBP"
    if t.endswith(".SW"):
        return "CHF"
    if t.endswith(".T"):
        return "JPY"
    if t.endswith(".HK"):
        return "HKD"
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "KRW"
    if t.endswith(".TW"):
        return "TWD"
    if t.endswith("-USD") or t.endswith("=X"):
        return "USD"
    if info:
        cur = info.get("currency") or info.get("financialCurrency")
        if cur:
            return cur.upper()
    return "USD"


def to_eur(value, currency: str, eur_usd: float = None, rates: dict | None = None):
    """Konvertiert einen Preis in EUR. rates = get_fx_rates(); eur_usd optional Fallback."""
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    cur = (currency or "USD").upper()
    if cur == "EUR":
        return v
    if rates is None:
        rates = {}
        if eur_usd:
            rates["USD"] = eur_usd
    rate = rates.get(cur)
    if rate is None and cur == "USD" and eur_usd:
        rate = eur_usd
    if rate and rate > 0:
        return v / rate
    return v


def format_price_eur(value, currency: str, eur_usd: float = None, rates: dict | None = None) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except Exception:
        return "—"
    cur = (currency or "USD").upper()
    symbols = {
        "USD": "$", "GBP": "£", "CHF": "CHF", "JPY": "¥",
        "HKD": "HK$", "KRW": "₩", "TWD": "NT$", "EUR": "€",
    }
    if rates is None:
        rates = get_fx_rates()
        if eur_usd:
            rates = dict(rates)
            rates["USD"] = eur_usd
    eur_val = to_eur(v, cur, eur_usd=eur_usd, rates=rates)
    if eur_val is None:
        return "—"
    if cur == "EUR":
        return f"{v:.2f} €"
    sym = symbols.get(cur, cur)
    return f"{eur_val:.2f} €  ({v:.2f} {sym})"


def last_golden_cross_date(data: pd.DataFrame):
    """Datum des letzten Golden Cross (oder None)."""
    df = data.dropna(subset=["SMA50", "SMA200"]).copy()
    if len(df) < 2:
        return None
    df["diff"] = df["SMA50"] - df["SMA200"]
    df["cross"] = (df["diff"].shift(1) < 0) & (df["diff"] > 0)
    crosses = df[df["cross"]].index
    if len(crosses) == 0:
        return None
    return crosses[-1]


def last_buy_signal_date(data: pd.DataFrame):
    """
    Datum des letzten kombinierten TA-Kaufsignals.
    Definition analog compute_ta_signal: Golden Cross ODER (bullisch und RSI kam unter 70).
    """
    df = data.dropna(subset=["SMA50", "SMA200", "RSI", "Close"]).copy()
    if len(df) < 30:
        return None
    df["diff"] = df["SMA50"] - df["SMA200"]
    df["gc"] = (df["diff"].shift(1) < 0) & (df["diff"] > 0)
    df["bull"] = df["SMA50"] > df["SMA200"]
    # RSI-Übergang unter 70 während bullischem Trend
    df["rsi_entry"] = df["bull"] & (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70)
    df["signal"] = df["gc"] | df["rsi_entry"]
    hits = df[df["signal"]].index
    if len(hits) == 0:
        return None
    return hits[-1]



def compute_signal_status(data: pd.DataFrame):
    """
    Liefert LONG / FLAT / SHORT inkl. seit-Datum und Performance seit Signal.
    LONG  = bullischer Trend (SMA50 > SMA200) und kein extremes Overbought
    SHORT = bärischer Trend
    FLAT  = unsicher / seitwärts / kein klares Signal
    """
    if data.empty or "SMA50" not in data.columns:
        return {"status": "FLAT", "label": "⚪ FLAT", "since": None, "perf_pct": None, "days": None}

    df = data.dropna(subset=["SMA50", "SMA200", "Close"]).copy()
    if len(df) < 50:
        return {"status": "FLAT", "label": "⚪ FLAT", "since": None, "perf_pct": None, "days": None}

    last = df.iloc[-1]
    bull = bool(last["SMA50"] > last["SMA200"])
    rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and not pd.isna(df["RSI"].iloc[-1]) else 50.0

    # Finde Beginn des aktuellen Trends (letzter Cross)
    df["diff"] = df["SMA50"] - df["SMA200"]
    df["cross_up"] = (df["diff"].shift(1) < 0) & (df["diff"] > 0)
    df["cross_dn"] = (df["diff"].shift(1) > 0) & (df["diff"] < 0)

    if bull:
        status = "LONG"
        label = "🟢 LONG"
        crosses = df[df["cross_up"]].index
    else:
        status = "SHORT"
        label = "🔴 SHORT"
        crosses = df[df["cross_dn"]].index

    # Bei extremem RSI eher FLAT (Überkauft/Überverkauft-Warnung)
    if status == "LONG" and rsi > 75:
        status, label = "FLAT", "🟡 FLAT"
    elif status == "SHORT" and rsi < 25:
        status, label = "FLAT", "🟡 FLAT"

    since = crosses[-1] if len(crosses) else None
    perf_pct = None
    days = None
    if since is not None:
        entry = float(df.loc[since, "Close"])
        now = float(last["Close"])
        if entry > 0:
            perf_pct = (now / entry - 1) * 100
        days = (df.index[-1] - since).days

    return {
        "status": status,
        "label": label,
        "since": since,
        "perf_pct": round(perf_pct, 1) if perf_pct is not None else None,
        "days": days,
    }


def compute_sentiment(data: pd.DataFrame) -> dict:
    """
    Einfaches Sentiment 0–100 (0=extreme Angst, 100=extreme Gier).
    Basiert auf RSI + kurzfristigem Momentum.
    """
    if data.empty or "RSI" not in data.columns:
        return {"score": 50, "label": "Neutral", "color": "gray"}

    rsi = data["RSI"].dropna()
    if rsi.empty:
        return {"score": 50, "label": "Neutral", "color": "gray"}

    last_rsi = float(rsi.iloc[-1])
    # Momentum 20 Tage
    closes = data["Close"].dropna()
    mom = 0.0
    if len(closes) > 20:
        mom = float(closes.iloc[-1] / closes.iloc[-20] - 1) * 100

    # Score: RSI 50% + Momentum skaliert 50%
    # mom typisch -15..+15 → map auf 0..100
    mom_score = max(0, min(100, 50 + mom * 2.5))
    score = 0.6 * last_rsi + 0.4 * mom_score
    score = int(max(0, min(100, round(score))))

    if score >= 75:
        label, color = "Extreme Gier", "#e74c3c"
    elif score >= 60:
        label, color = "Gier", "#e67e22"
    elif score >= 40:
        label, color = "Neutral", "#f1c40f"
    elif score >= 25:
        label, color = "Angst", "#3498db"
    else:
        label, color = "Extreme Angst", "#2ecc71"

    return {"score": score, "label": label, "color": color, "rsi": round(last_rsi, 1)}


def compute_zielpfad(data: pd.DataFrame, fundamentals: dict, fx: float = 1.0) -> list:
    """
    Zielpfad analog M&T: mehrere Kursziele aus Fib + Analysten.
    Alle Werte in Chart-Währung (bereits * fx für EUR-Anzeige).
    """
    if data.empty:
        return []

    last = float(data["Close"].iloc[-1]) * fx
    fib = compute_fib_entry_zone(data)
    t_low = fundamentals.get("Analysten-Kursziel (niedrig)")
    t_mean = fundamentals.get("Analysten-Kursziel (mittel)")
    t_high = fundamentals.get("Analysten-Kursziel (hoch)")

    targets = []
    # ZIEL 1: nahes Widerstand / Fib-Oberkante oder leicht über Kurs
    if fib:
        z1 = max(fib[1] * fx, last * 1.03)
    else:
        z1 = last * 1.05
    targets.append({"name": "ZIEL 1", "kurs": z1, "typ": "nah"})

    # ZIEL 2: Analyst Mean oder +10%
    if t_mean:
        z2 = float(t_mean) * fx
    else:
        z2 = last * 1.10
    if z2 <= z1:
        z2 = z1 * 1.05
    targets.append({"name": "ZIEL 2", "kurs": z2, "typ": "mittel"})

    # ZIEL 3: zwischen Mean und High
    if t_high and t_mean:
        z3 = (float(t_mean) + float(t_high)) / 2 * fx
    elif t_high:
        z3 = float(t_high) * fx * 0.92
    else:
        z3 = last * 1.18
    if z3 <= z2:
        z3 = z2 * 1.06
    targets.append({"name": "ZIEL 3", "kurs": z3, "typ": "gestreckt"})

    # ZIEL 4 Best Case: Analyst High oder +25%
    if t_high:
        z4 = float(t_high) * fx
    else:
        z4 = last * 1.25
    if z4 <= z3:
        z4 = z3 * 1.08
    targets.append({"name": "ZIEL 4 · Best Case", "kurs": z4, "typ": "best"})

    # Risiko: unter letztem Swing / -8%
    risk = last * 0.92
    if fib:
        risk = min(risk, fib[0] * fx)

    for t in targets:
        t["weg_pct"] = (t["kurs"] / last - 1) * 100
        t["status"] = "erreicht" if last >= t["kurs"] else "aktiv"
        # grobe Chance-Schätzung (näher = höher)
        if t["weg_pct"] < 5:
            t["chance"] = 55
        elif t["weg_pct"] < 12:
            t["chance"] = 40
        elif t["weg_pct"] < 20:
            t["chance"] = 25
        else:
            t["chance"] = 12

    return {"targets": targets, "risk": risk, "risk_pct": (risk / last - 1) * 100, "last": last}


# ---------------------------------------------------------------------------
# Fundamentaldaten
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_fundamentals(ticker: str):
    info = yf.Ticker(ticker).info
    return {
        "Name": info.get("longName", ticker),
        "Sektor": info.get("sector", "—"),
        "Branche": info.get("industry", "—"),
        "Kurs": info.get("currentPrice") or info.get("regularMarketPrice"),
        "KGV (trailing)": info.get("trailingPE"),
        "KGV (forward)": info.get("forwardPE"),
        "KBV": info.get("priceToBook"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "Marktkap.": info.get("marketCap"),
        "Umsatzwachstum (YoY)": info.get("revenueGrowth"),
        "Gewinnmarge": info.get("profitMargins"),
        "Operative Marge": info.get("operatingMargins"),
        "ROE": info.get("returnOnEquity"),
        "Verschuldung (D/E)": info.get("debtToEquity"),
        "Analysten-Kursziel (mittel)": info.get("targetMeanPrice"),
        "Analysten-Kursziel (hoch)": info.get("targetHighPrice"),
        "Analysten-Kursziel (niedrig)": info.get("targetLowPrice"),
        "Empfehlung": info.get("recommendationKey"),
        "Beta": info.get("beta"),
        "Dividendenrendite": info.get("dividendYield"),
    }


# ---------------------------------------------------------------------------
# Branchenvergleich
# ---------------------------------------------------------------------------
SECTOR_BENCHMARKS = {
    "Technology":               {"KGV": 28, "KBV": 6.0, "EV/EBITDA": 18, "ROE": 0.18, "Marge": 0.18},
    "Communication Services":   {"KGV": 22, "KBV": 3.5, "EV/EBITDA": 12, "ROE": 0.12, "Marge": 0.15},
    "Consumer Cyclical":        {"KGV": 20, "KBV": 3.0, "EV/EBITDA": 12, "ROE": 0.15, "Marge": 0.08},
    "Consumer Defensive":       {"KGV": 22, "KBV": 4.0, "EV/EBITDA": 14, "ROE": 0.18, "Marge": 0.10},
    "Financial Services":       {"KGV": 12, "KBV": 1.4, "EV/EBITDA": 8,  "ROE": 0.12, "Marge": 0.25},
    "Healthcare":               {"KGV": 24, "KBV": 4.5, "EV/EBITDA": 15, "ROE": 0.15, "Marge": 0.12},
    "Industrials":              {"KGV": 18, "KBV": 3.0, "EV/EBITDA": 11, "ROE": 0.14, "Marge": 0.09},
    "Basic Materials":          {"KGV": 14, "KBV": 1.8, "EV/EBITDA": 8,  "ROE": 0.10, "Marge": 0.08},
    "Energy":                   {"KGV": 11, "KBV": 1.5, "EV/EBITDA": 6,  "ROE": 0.12, "Marge": 0.10},
    "Utilities":                {"KGV": 16, "KBV": 1.6, "EV/EBITDA": 10, "ROE": 0.09, "Marge": 0.12},
    "Real Estate":              {"KGV": 30, "KBV": 1.8, "EV/EBITDA": 16, "ROE": 0.06, "Marge": 0.20},
}


def relative_valuation(fundamentals: dict) -> dict:
    sector = fundamentals.get("Sektor", "")
    bench = SECTOR_BENCHMARKS.get(sector)
    if not bench:
        return {}
    result = {}
    mapping = {
        "KGV (trailing)": ("KGV", False),
        "KBV": ("KBV", False),
        "EV/EBITDA": ("EV/EBITDA", False),
        "ROE": ("ROE", True),
        "Operative Marge": ("Marge", True),
    }
    for key, (bench_key, higher_better) in mapping.items():
        val = fundamentals.get(key)
        bval = bench.get(bench_key)
        if val is None or bval is None:
            continue
        try:
            diff_pct = (val - bval) / bval * 100
        except Exception:
            continue
        if higher_better:
            status = "🟢 überdurchschnittlich" if diff_pct > 10 else ("🔴 unterdurchschnittlich" if diff_pct < -10 else "🟡 durchschnittlich")
        else:
            status = "🟢 günstig" if diff_pct < -10 else ("🔴 teuer" if diff_pct > 10 else "🟡 fair")
        result[key] = {"Wert": val, "Branche": bval, "Diff %": diff_pct, "Status": status}
    return result


def valuation_score(rel: dict) -> int:
    score = 0
    for info in rel.values():
        if "günstig" in info["Status"] or "überdurchschnittlich" in info["Status"]:
            score += 1
        elif "teuer" in info["Status"] or "unterdurchschnittlich" in info["Status"]:
            score -= 1
    return score


# ---------------------------------------------------------------------------
# Scoring-Modelle
# ---------------------------------------------------------------------------
def _get_row(df: pd.DataFrame, names):
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


@st.cache_data(ttl=86400)
def compute_piotroski_score(ticker: str):
    details = {}
    score = 0
    max_score = 0
    try:
        tk = yf.Ticker(ticker)
        bs = tk.balance_sheet
        fin = tk.financials
        cf = tk.cashflow
        if bs.empty or fin.empty or cf.empty:
            return None, {}, 0
        bs = bs.sort_index(axis=1, ascending=False)
        fin = fin.sort_index(axis=1, ascending=False)
        cf = cf.sort_index(axis=1, ascending=False)

        net_income = _get_row(fin, ["Net Income"])
        total_assets = _get_row(bs, ["Total Assets"])
        cfo = _get_row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        lt_debt = _get_row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        curr_assets = _get_row(bs, ["Current Assets", "Total Current Assets"])
        curr_liab = _get_row(bs, ["Current Liabilities", "Total Current Liabilities"])
        shares = _get_row(fin, ["Basic Average Shares", "Diluted Average Shares"])
        revenue = _get_row(fin, ["Total Revenue"])
        gross_profit = _get_row(fin, ["Gross Profit"])

        def try_criterion(label, fn):
            nonlocal score, max_score
            try:
                result = fn()
                if result is None:
                    return
                max_score += 1
                if result:
                    score += 1
                details[label] = bool(result)
            except Exception:
                pass

        if net_income is not None and total_assets is not None and len(net_income) >= 1:
            try_criterion("Nettogewinn positiv", lambda: net_income.iloc[0] > 0)
        if cfo is not None and len(cfo) >= 1:
            try_criterion("Operativer Cashflow positiv", lambda: cfo.iloc[0] > 0)
        if net_income is not None and total_assets is not None and len(net_income) >= 2:
            try_criterion("ROA gestiegen",
                          lambda: (net_income.iloc[0] / total_assets.iloc[0]) > (net_income.iloc[1] / total_assets.iloc[1]))
        if cfo is not None and net_income is not None:
            try_criterion("Cashflow > Gewinn", lambda: cfo.iloc[0] > net_income.iloc[0])
        if lt_debt is not None and total_assets is not None and len(lt_debt) >= 2:
            try_criterion("Verschuldung gesunken",
                          lambda: (lt_debt.iloc[0] / total_assets.iloc[0]) < (lt_debt.iloc[1] / total_assets.iloc[1]))
        if curr_assets is not None and curr_liab is not None and len(curr_assets) >= 2:
            try_criterion("Current Ratio verbessert",
                          lambda: (curr_assets.iloc[0] / curr_liab.iloc[0]) > (curr_assets.iloc[1] / curr_liab.iloc[1]))
        if shares is not None and len(shares) >= 2:
            try_criterion("Keine Verwässerung", lambda: shares.iloc[0] <= shares.iloc[1])
        if gross_profit is not None and revenue is not None and len(gross_profit) >= 2:
            try_criterion("Bruttomarge gestiegen",
                          lambda: (gross_profit.iloc[0] / revenue.iloc[0]) > (gross_profit.iloc[1] / revenue.iloc[1]))
        if revenue is not None and total_assets is not None and len(revenue) >= 2:
            try_criterion("Asset-Turnover gestiegen",
                          lambda: (revenue.iloc[0] / total_assets.iloc[0]) > (revenue.iloc[1] / total_assets.iloc[1]))
        return score, details, max_score
    except Exception:
        return None, {}, 0


@st.cache_data(ttl=3600)
def compute_levermann_score(ticker: str, six_month_return=None, twelve_month_return=None):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None, {}
    score = 0
    details = {}
    roe = info.get("returnOnEquity")
    if roe is not None:
        pts = 1 if roe > 0.20 else (-1 if roe < 0.10 else 0)
        score += pts
        details["ROE"] = f"{roe:.1%} → {pts:+d}"
    op_margin = info.get("operatingMargins")
    if op_margin is not None:
        pts = 1 if op_margin > 0.12 else (-1 if op_margin < 0.06 else 0)
        score += pts
        details["Operative Marge"] = f"{op_margin:.1%} → {pts:+d}"
    dte = info.get("debtToEquity")
    if dte is not None:
        pts = 1 if dte < 100 else (-1 if dte > 200 else 0)
        score += pts
        details["D/E"] = f"{dte:.0f} → {pts:+d}"
    rec = info.get("recommendationKey")
    if rec:
        pts = 1 if rec in ("buy", "strong_buy") else (-1 if rec in ("sell", "strong_sell") else 0)
        score += pts
        details["Analystenmeinung"] = f"{rec} → {pts:+d}"
    target = info.get("targetMeanPrice")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if target and price:
        upside = (target - price) / price
        pts = 1 if upside > 0.10 else (-1 if upside < -0.10 else 0)
        score += pts
        details["Kursziel-Abstand"] = f"{upside:+.1%} → {pts:+d}"
    if six_month_return is not None:
        pts = 1 if six_month_return > 0.05 else (-1 if six_month_return < -0.05 else 0)
        score += pts
        details["Momentum 6M"] = f"{six_month_return:+.1%} → {pts:+d}"
    if twelve_month_return is not None:
        pts = 1 if twelve_month_return > 0.05 else (-1 if twelve_month_return < -0.05 else 0)
        score += pts
        details["Momentum 12M"] = f"{twelve_month_return:+.1%} → {pts:+d}"
    return score, details


def price_return(data: pd.DataFrame, trading_days: int):
    df = data.dropna(subset=["Close"])
    if len(df) <= trading_days:
        return None
    return float(df["Close"].iloc[-1] / df["Close"].iloc[-trading_days] - 1)


def compute_composite_score(p_score, p_max, l_score, val_score, ta_signal, r6):
    score = 50.0
    if p_score is not None and p_max and p_max > 0:
        score += (p_score / p_max) * 20 - 10
    if l_score is not None:
        score += max(-15, min(15, l_score * 2.5))
    score += max(-10, min(10, val_score * 3))
    if r6 is not None:
        if r6 > 0.15:
            score += 8
        elif r6 > 0.05:
            score += 4
        elif r6 < -0.15:
            score -= 8
        elif r6 < -0.05:
            score -= 4
    if "Kaufsignal" in str(ta_signal):
        score += 7
    elif "Verkaufssignal" in str(ta_signal):
        score -= 7
    return int(max(0, min(100, round(score))))


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
def _find_signal_dates(data: pd.DataFrame, signal_type: str = "golden"):
    """Findet historische Signal-Zeitpunkte für verschiedene Strategien."""
    need = ["SMA50", "SMA200", "Close", "RSI"]
    if signal_type in ("macd", "macd_cross"):
        need += ["MACD", "MACD_signal"]
    df = data.dropna(subset=[c for c in need if c in data.columns]).copy()
    if len(df) < 200:
        return []

    if signal_type == "golden":
        df["diff"] = df["SMA50"] - df["SMA200"]
        df["signal"] = (df["diff"].shift(1) < 0) & (df["diff"] > 0)
    elif signal_type == "death":
        df["diff"] = df["SMA50"] - df["SMA200"]
        df["signal"] = (df["diff"].shift(1) > 0) & (df["diff"] < 0)
    elif signal_type == "rsi_oversold":
        # RSI kommt aus dem überverkauften Bereich (<30) zurück über 30
        df["signal"] = (df["RSI"].shift(1) < 30) & (df["RSI"] >= 30)
    elif signal_type == "rsi_overbought":
        df["signal"] = (df["RSI"].shift(1) > 70) & (df["RSI"] <= 70)
    elif signal_type == "macd":
        if "MACD" not in df.columns:
            return []
        df["signal"] = (df["MACD"].shift(1) < df["MACD_signal"].shift(1)) & (df["MACD"] >= df["MACD_signal"])
    elif signal_type == "ta":
        df["diff"] = df["SMA50"] - df["SMA200"]
        df["gc"] = (df["diff"].shift(1) < 0) & (df["diff"] > 0)
        df["bull"] = df["SMA50"] > df["SMA200"]
        df["rsi_entry"] = df["bull"] & (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70)
        df["signal"] = df["gc"] | df["rsi_entry"]
        df["signal"] = df["signal"] & ~df["signal"].shift(1).fillna(False)
    else:
        return []
    return df[df["signal"]].index.tolist()


def backtest_signals(
    data: pd.DataFrame,
    signal_type: str = "golden",
    forward_days_list=(21, 63, 126),
    hold_days: int | None = None,
):
    """
    Vertieftes Backtesting:
    - Fixed-Horizon Returns (1M/3M/6M)
    - Optional feste Haltedauer (hold_days)
    - Profit Factor, Avg Win/Loss, Max Consecutive Losses
    - Equity-Kurve (kumulierte Signal-Returns bei fester Haltedauer)
    """
    crosses = _find_signal_dates(data, signal_type)
    if not crosses:
        return {"count": 0, "summary": {}, "details": [], "equity": []}

    df = data.dropna(subset=["Close"]).copy()
    results = []
    fixed_returns = []  # für Equity bei hold_days

    for dt in crosses:
        try:
            entry_price = float(df.loc[dt, "Close"])
            idx = df.index.get_loc(dt)
            if isinstance(idx, slice):
                continue
        except Exception:
            continue

        row = {
            "Datum": dt.strftime("%Y-%m-%d"),
            "Einstieg": round(entry_price, 2),
        }

        # Fixed-horizon returns
        for days in forward_days_list:
            future_idx = idx + days
            if future_idx < len(df):
                future_price = float(df.iloc[future_idx]["Close"])
                ret = (future_price / entry_price - 1) * 100
                row[f"Rendite {days}T"] = round(ret, 2)
            else:
                row[f"Rendite {days}T"] = None

        # Max DD in den nächsten 126 Tagen
        end_idx = min(idx + 126, len(df) - 1)
        window = df.iloc[idx:end_idx + 1]["Close"]
        if len(window) > 1:
            peak = window.cummax()
            dd = float(((window - peak) / peak * 100).min())
            row["Max DD 6M"] = round(dd, 2)
        else:
            row["Max DD 6M"] = None

        # Feste Haltedauer (für Equity-Kurve)
        hd = hold_days or 63
        exit_idx = idx + hd
        if exit_idx < len(df):
            exit_price = float(df.iloc[exit_idx]["Close"])
            hold_ret = (exit_price / entry_price - 1) * 100
            row[f"Hold {hd}T"] = round(hold_ret, 2)
            fixed_returns.append(hold_ret)
        else:
            row[f"Hold {hd}T"] = None

        results.append(row)

    # --- Summary-Statistiken ---
    summary = {"count": len(results)}
    for days in forward_days_list:
        col = f"Rendite {days}T"
        vals = [r[col] for r in results if r[col] is not None]
        if vals:
            wins = [v for v in vals if v > 0]
            losses = [v for v in vals if v <= 0]
            summary[f"Ø {days}T"] = round(float(np.mean(vals)), 2)
            summary[f"Median {days}T"] = round(float(np.median(vals)), 2)
            summary[f"Trefferquote {days}T"] = round(len(wins) / len(vals) * 100, 1)
            summary[f"Best {days}T"] = round(float(np.max(vals)), 2)
            summary[f"Worst {days}T"] = round(float(np.min(vals)), 2)
            if wins:
                summary[f"Ø Win {days}T"] = round(float(np.mean(wins)), 2)
            if losses:
                summary[f"Ø Loss {days}T"] = round(float(np.mean(losses)), 2)
            # Profit Factor
            gross_win = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            summary[f"PF {days}T"] = round(gross_win / gross_loss, 2) if gross_loss > 0 else None
        else:
            for k in (f"Ø {days}T", f"Median {days}T", f"Trefferquote {days}T",
                      f"Best {days}T", f"Worst {days}T", f"PF {days}T"):
                summary[k] = None

    # Consecutive losses (auf 63T-Horizont)
    vals_63 = [r.get("Rendite 63T") for r in results if r.get("Rendite 63T") is not None]
    max_consec_loss = 0
    cur = 0
    for v in vals_63:
        if v <= 0:
            cur += 1
            max_consec_loss = max(max_consec_loss, cur)
        else:
            cur = 0
    summary["max_consec_loss"] = max_consec_loss

    # Equity-Kurve (kumuliert, feste Haltedauer)
    equity = []
    cum = 0.0
    for i, r in enumerate(fixed_returns):
        cum += r
        equity.append({"Signal #": i + 1, "Trade %": round(r, 2), "Kumuliert %": round(cum, 2)})

    summary["equity_final"] = round(cum, 2) if fixed_returns else None
    summary["hold_days"] = hold_days or 63

    return {"count": len(results), "summary": summary, "details": results, "equity": equity, "signal_dates": crosses}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Navigation")
mode = st.sidebar.radio(
    "Ansicht",
    ["📊 Einzelanalyse", "🔍 Scanner", "📋 Vergleichs-Matrix", "💼 Portfolio", "📉 Backtesting"],
)

# ===========================================================================
# 1. Einzelanalyse
# ===========================================================================
if mode == "📊 Einzelanalyse":
    st.title("📊 Aktien-Einzelanalyse")
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input("Firmenname oder Ticker", value="Apple")
    with col2:
        period = st.selectbox("Zeitraum", ["1y", "2y", "5y", "10y"], index=1)
    ticker = resolve_ticker(user_input) if user_input else ""
    if ticker and ticker.upper() != user_input.upper().strip():
        st.caption(f"→ Interpretiert als: **{ticker}**")
    if ticker:
        data = load_price_history(ticker, period)
        if data.empty:
            st.error("Keine Kursdaten gefunden. Ticker prüfen.")
        else:
            fundamentals = load_fundamentals(ticker)
            bullish = is_bullish_now(data)
            golden = detect_golden_cross(data)
            ta_signal = compute_ta_signal(data)
            vol, max_dd = compute_risk_metrics(data)
            r6 = price_return(data, 126)
            r12 = price_return(data, 252)

            # Währung & EUR-Umrechnung (USD, GBP, CHF, JPY, HKD, …)
            currency = detect_currency(ticker, fundamentals)
            rates = get_fx_rates()
            eur_usd = rates.get("USD", 1.08)
            rate = rates.get(currency, 1.0)
            # Multiplikator: Original -> EUR
            fx = (1.0 / rate) if (currency != "EUR" and rate and rate > 0) else 1.0
            last_price = float(data["Close"].iloc[-1])
            last_price_eur = last_price * fx
            symbols = {"USD": "$", "GBP": "£", "CHF": "CHF", "JPY": "¥", "HKD": "HK$", "KRW": "₩", "TWD": "NT$"}
            if currency == "EUR":
                price_eur_str = f"{last_price:.2f} €"
                price_sub = "Originalwährung EUR"
            else:
                sym = symbols.get(currency, currency)
                price_eur_str = f"{last_price_eur:.2f} €"
                price_sub = f"Original: {last_price:.2f} {sym}"

            # Chart-Daten in EUR umrechnen (nur Preisachsen, RSI/MACD bleiben unverändert)
            plot = data.copy()
            for col in ("Open", "High", "Low", "Close", "SMA50", "SMA200"):
                if col in plot.columns:
                    plot[col] = plot[col] * fx

            # Letzte Signal-Daten
            gc_date = last_golden_cross_date(data)
            buy_date = last_buy_signal_date(data)
            gc_date_str = gc_date.strftime("%d.%m.%Y") if gc_date is not None else "—"
            buy_date_str = buy_date.strftime("%d.%m.%Y") if buy_date is not None else "—"

            sig = compute_signal_status(data)
            sent = compute_sentiment(data)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Kurs", price_eur_str, price_sub)
            c2.metric("System-Status", sig["label"])
            c3.metric("Golden Cross", "✅ Ja" if golden else "—")
            c4.metric("TA-Signal", ta_signal)
            c5.metric("Sektor", str(fundamentals.get("Sektor", "—"))[:18])

            # Signal-Box (ähnlich M&T)
            st.markdown("#### 📡 System-Signal")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Status", sig["label"])
            if sig["since"] is not None:
                b2.metric("Seit", sig["since"].strftime("%d.%m.%Y"))
            else:
                b2.metric("Seit", "—")
            b3.metric("Performance seit Signal",
                      f"{sig['perf_pct']:+.1f} %" if sig["perf_pct"] is not None else "—")
            b4.metric("Haltedauer", f"{sig['days']} Tage" if sig["days"] is not None else "—")

            # Sentiment-Anzeige
            st.markdown("#### 🌡️ Markt-Sentiment (Aktie)")
            s1, s2, s3 = st.columns([2, 1, 1])
            with s1:
                st.progress(sent["score"] / 100)
            s2.metric("Score", f"{sent['score']} / 100")
            s3.metric("Lage", sent["label"])
            st.caption(f"RSI: {sent.get('rsi', '—')} · 0 = Extreme Angst · 100 = Extreme Gier")

            # Signal-Historie
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Letztes Golden Cross", gc_date_str)
            sc2.metric("Letztes Kaufsignal", buy_date_str)
            sc3.metric("Währung (Original)", currency)

            chart_title = f"{ticker} – Kurs in € + SMA50/200"
            if currency != "EUR":
                chart_title += f"  · FX-Umrechnung aktiv"

            # Chart-Steuerung
            st.markdown("#### 📈 Chart")
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                chart_big = st.toggle("Vollbild-Chart", value=False, help="Größere Ansicht zum genauen Analysieren")
            with cc2:
                show_slider = st.toggle("Zeitraum-Slider", value=True, help="Unten am Chart zoomen/verschieben – ideal am Handy")
            with cc3:
                show_buy_marks = st.toggle("Kaufsignale", value=True)
            with cc4:
                show_gc_marks = st.toggle("Golden Cross", value=True)

            chart_height = 980 if chart_big else 720

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                row_heights=[0.58, 0.18, 0.24],
                subplot_titles=(chart_title, "RSI (14)", "MACD"),
            )
            fig.add_trace(go.Candlestick(
                x=plot.index, open=plot["Open"], high=plot["High"],
                low=plot["Low"], close=plot["Close"], name="Kurs €",
                increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
                decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=plot.index, y=plot["SMA50"], name="SMA50",
                line=dict(color="#ffb74d", width=2),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=plot.index, y=plot["SMA200"], name="SMA200",
                line=dict(color="#42a5f5", width=2),
            ), row=1, col=1)

            # Kaufsignale im Chart markieren
            if show_buy_marks:
                buy_dates = _find_signal_dates(data, "ta")
                if buy_dates:
                    bx, by = [], []
                    for d in buy_dates:
                        if d in plot.index:
                            bx.append(d)
                            by.append(float(plot.loc[d, "Close"]))
                    if bx:
                        fig.add_trace(go.Scatter(
                            x=bx, y=by, mode="markers",
                            name="Kaufsignal",
                            marker=dict(
                                symbol="triangle-up",
                                size=13,
                                color="#00e676",
                                line=dict(width=1.5, color="white"),
                            ),
                            hovertemplate="Kaufsignal<br>%{x|%d.%m.%Y}<br>%{y:.2f} €<extra></extra>",
                        ), row=1, col=1)

            if show_gc_marks:
                gc_dates = _find_signal_dates(data, "golden")
                if gc_dates:
                    gx, gy = [], []
                    for d in gc_dates:
                        if d in plot.index:
                            gx.append(d)
                            gy.append(float(plot.loc[d, "Close"]))
                    if gx:
                        fig.add_trace(go.Scatter(
                            x=gx, y=gy, mode="markers",
                            name="Golden Cross",
                            marker=dict(
                                symbol="star",
                                size=14,
                                color="#ffd600",
                                line=dict(width=1, color="#333"),
                            ),
                            hovertemplate="Golden Cross<br>%{x|%d.%m.%Y}<br>%{y:.2f} €<extra></extra>",
                        ), row=1, col=1)

            for key, color, label in [
                ("Analysten-Kursziel (niedrig)", "#2ca02c", "Min"),
                ("Analysten-Kursziel (mittel)", "#1f77b4", "Normal"),
                ("Analysten-Kursziel (hoch)", "#d62728", "Max"),
            ]:
                val = fundamentals.get(key)
                if val:
                    val_eur = float(val) * fx
                    fig.add_hline(
                        y=val_eur, line_dash="dash", line_color=color,
                        annotation_text=f"{label} {val_eur:.1f} €",
                        annotation_font_size=11,
                        row=1, col=1,
                    )
            fib = compute_fib_entry_zone(data)
            if fib:
                fig.add_hrect(
                    y0=fib[0] * fx, y1=fib[1] * fx,
                    fillcolor="rgba(38,166,154,0.12)", line_width=0,
                    annotation_text="Fib-Zone", annotation_position="top left",
                    row=1, col=1,
                )

            fig.add_trace(go.Scatter(
                x=data.index, y=data["RSI"], name="RSI",
                line=dict(color="#ce93d8", width=1.8),
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.7)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.7)", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", row=2, col=1)

            fig.add_trace(go.Scatter(
                x=data.index, y=data["MACD"], name="MACD",
                line=dict(color="#64b5f6", width=1.6),
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data["MACD_signal"], name="Signal",
                line=dict(color="#ffb74d", width=1.6),
            ), row=3, col=1)
            colors = ["#26a69a" if v >= 0 else "#ef5350" for v in data["MACD_hist"].fillna(0)]
            fig.add_trace(go.Bar(
                x=data.index, y=data["MACD_hist"], name="Hist",
                marker_color=colors, opacity=0.75,
            ), row=3, col=1)

            fig.update_layout(
                **CHART_LAYOUT,
                height=chart_height,
                xaxis_rangeslider_visible=False,
                xaxis_title=None,
                yaxis_title="Kurs €",
                yaxis2=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                    fixedrange=False, title="RSI", range=[0, 100],
                ),
                yaxis3=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                    fixedrange=False, title="MACD",
                ),
            )
            # Unterster x-Achse: Range-Slider für mobiles Zoomen
            fig.update_xaxes(
                fixedrange=False,
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                rangeslider=dict(visible=show_slider, thickness=0.06),
                row=3, col=1,
            )
            fig.update_xaxes(fixedrange=False, showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=1, col=1)
            fig.update_xaxes(fixedrange=False, showgrid=True, gridcolor="rgba(255,255,255,0.08)", row=2, col=1)
            fig.update_yaxes(fixedrange=False)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key=f"chart_{ticker}_{int(chart_big)}_{int(show_slider)}",
            )
            st.caption(
                "📱 **Handy:** Zeitraum-Slider unten ziehen · Modebar (🔍 + / −) zum Zoomen · "
                "mit einem Finger verschieben · **Vollbild-Chart** für große Ansicht · "
                "🟢 Dreieck = Kaufsignal · ⭐ = Golden Cross"
            )

            # ---- Zielpfad ----
            zp = compute_zielpfad(data, fundamentals, fx=fx)
            if zp and zp.get("targets"):
                st.markdown("#### 🎯 Zielpfad")
                zcols = st.columns(len(zp["targets"]) + 1)
                for i, t in enumerate(zp["targets"]):
                    with zcols[i]:
                        status_icon = "✅" if t["status"] == "erreicht" else "▶️"
                        st.metric(
                            f"{status_icon} {t['name']}",
                            f"{t['kurs']:.2f} €",
                            f"{t['weg_pct']:+.1f}% · Chance ~{t['chance']}%",
                        )
                with zcols[-1]:
                    st.metric("⚠️ Risiko", f"{zp['risk']:.2f} €", f"{zp['risk_pct']:+.1f}%")
                st.caption("Ziele aus Fibonacci-Zone + Analysten-Kurszielen abgeleitet. Keine Garantie.")

            st.subheader("⚠️ Risiko")
            rc1, rc2, rc3, rc4 = st.columns(4)
            beta = fundamentals.get("Beta")
            rc1.metric("Beta", f"{beta:.2f}" if beta else "—")
            rc2.metric("Volatilität (ann.)", f"{vol} %" if vol is not None else "—")
            rc3.metric("Max Drawdown", f"{max_dd} %" if max_dd is not None else "—")
            mom_txt = "—"
            if r6 is not None and r12 is not None:
                mom_txt = f"{r6*100:+.1f}% / {r12*100:+.1f}%"
            rc4.metric("Momentum 6M / 12M", mom_txt)

            st.subheader("🏭 Branchenvergleich")
            st.write(f"**Sektor:** {fundamentals.get('Sektor', '—')}  |  **Branche:** {fundamentals.get('Branche', '—')}")
            rel = relative_valuation(fundamentals)
            val_sc = valuation_score(rel)
            if rel:
                cols = st.columns(len(rel))
                for i, (key, info) in enumerate(rel.items()):
                    with cols[i]:
                        val_str = f"{info['Wert']:.2f}" if isinstance(info['Wert'], (int, float)) else str(info['Wert'])
                        st.metric(key, val_str, f"{info['Diff %']:+.0f}% vs Branche")
                        st.caption(info["Status"])
            else:
                st.info("Für diesen Sektor liegen noch keine Benchmark-Werte vor.")

            st.subheader("📊 Qualitäts- & Composite-Score")
            p_score, p_details, p_max = compute_piotroski_score(ticker)
            l_score, l_details = compute_levermann_score(ticker, r6, r12)
            composite = compute_composite_score(p_score, p_max, l_score, val_sc, ta_signal, r6)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                if p_score is not None:
                    st.metric("Piotroski F-Score", f"{p_score} / {p_max}")
                    with st.expander("Details"):
                        for k, v in p_details.items():
                            st.write(f"{'✅' if v else '❌'} {k}")
                else:
                    st.write("Piotroski: keine Daten")
            with sc2:
                if l_score is not None:
                    st.metric("Levermann (vereinfacht)", l_score)
                    with st.expander("Details"):
                        for k, v in l_details.items():
                            st.write(f"{k}: {v}")
                else:
                    st.write("Levermann: keine Daten")
            with sc3:
                if composite >= 70:
                    label = f"{composite} 🟢 Stark"
                elif composite >= 55:
                    label = f"{composite} 🟡 Solide"
                else:
                    label = f"{composite} 🔴 Schwach"
                st.metric("Composite Score (0–100)", label)
                st.caption("Qualität + Bewertung + Momentum + Signal")

            st.subheader("📈 Fundamentaldaten")
            display_rows = []
            money_keys = {
                "Kurs", "Analysten-Kursziel (mittel)", "Analysten-Kursziel (hoch)",
                "Analysten-Kursziel (niedrig)", "Marktkap.",
            }
            for k, v in fundamentals.items():
                if k in ("Name", "Sektor", "Branche") or v is None:
                    continue
                if k in money_keys:
                    display_rows.append({"Kennzahl": k, "Wert": format_price_eur(v, currency, eur_usd)})
                elif k == "Dividendenrendite" and isinstance(v, (int, float)):
                    display_rows.append({"Kennzahl": k, "Wert": f"{v*100:.2f} %"})
                elif isinstance(v, float) and k in ("ROE", "Gewinnmarge", "Operative Marge", "Umsatzwachstum (YoY)"):
                    display_rows.append({"Kennzahl": k, "Wert": f"{v*100:.1f} %"})
                else:
                    display_rows.append({"Kennzahl": k, "Wert": v})
            fund_df = pd.DataFrame(display_rows)
            st.dataframe(fund_df, use_container_width=True, hide_index=True)
            if currency == "USD":
                st.caption(f"Preise in € umgerechnet mit EUR/USD ≈ {eur_usd:.4f}. Chart bleibt in Originalwährung ($).")

# ===========================================================================
# 2. Scanner
# ===========================================================================
elif mode == "🔍 Scanner":
    st.title("🔍 Long-Signal- & Qualitäts-Scanner")
    universe = st.selectbox("Universum", [
        "Tech-Auswahl",
        "🇩🇪 DAX 40",
        "🇫🇷 CAC 40 (Auswahl)",
        "🇪🇺 Europa gesamt (DAX+CAC+Benelux+CH+UK)",
        "Europa Blue Chips",
        "S&P-Auswahl (Auszug)",
        "🌏 Asia Tech",
        "🇯🇵 Japan (Nikkei-Auswahl)",
        "🇭🇰 Hongkong / China",
        "🌍 Große Auswahl (S&P + Europa + Nasdaq)",
        "🪙 Top 10 Krypto",
        "Eigene Liste",
    ])
    if universe == "🌍 Große Auswahl (S&P + Europa + Nasdaq)":
        tickers = list(set(fetch_sp500_tickers() + EUROPE_TICKERS + NASDAQ_TICKERS))
        st.warning(f"Scannt ca. {len(tickers)} Ticker – dauert mehrere Minuten.")
    elif universe == "🇪🇺 Europa gesamt (DAX+CAC+Benelux+CH+UK)":
        tickers = EUROPE_TICKERS
        st.info(f"Europa-Universum: {len(tickers)} Ticker (DE, FR, NL/BE, CH, UK).")
    elif universe == "🇩🇪 DAX 40":
        tickers = DAX40_TICKERS
    elif universe == "🇫🇷 CAC 40 (Auswahl)":
        tickers = CAC40_TICKERS
    elif universe == "🪙 Top 10 Krypto":
        tickers = TOP10_CRYPTO
    elif universe == "🌏 Asia Tech":
        tickers = ASIA_TECH
    elif universe == "🇯🇵 Japan (Nikkei-Auswahl)":
        tickers = JAPAN_TICKERS
    elif universe == "🇭🇰 Hongkong / China":
        tickers = HK_CHINA
    elif universe == "Eigene Liste":
        custom = st.text_input(
            "Ticker / Namen (kommagetrennt)",
            "SAP, ASML, LVMH, Nestle, Shell, AstraZeneca",
        )
        tickers = [resolve_ticker(t.strip()) for t in custom.split(",") if t.strip()]
    else:
        tickers = BUILTIN_WATCHLISTS.get(universe, [])

    colf1, colf2, colf3 = st.columns(3)
    only_buy = colf1.checkbox("Nur Kaufsignale", value=True)
    min_piotroski = colf2.slider("Min. Piotroski-Score", 0, 9, 0)
    min_composite = colf3.slider("Min. Composite-Score", 0, 100, 0)

    if st.button("Scanner starten", type="primary"):
        results = []
        progress = st.progress(0)
        status = st.empty()
        for i, t in enumerate(tickers):
            status.text(f"Prüfe {t} ({i+1}/{len(tickers)})")
            try:
                data = load_price_history(t, "2y")
                if data.empty:
                    continue
                signal = compute_ta_signal(data)
                if only_buy and "Kaufsignal" not in signal:
                    continue
                fund = load_fundamentals(t)
                p_score, _, p_max = compute_piotroski_score(t)
                if min_piotroski > 0 and (p_score is None or p_score < min_piotroski):
                    continue
                r6 = price_return(data, 126)
                r12 = price_return(data, 252)
                l_score, _ = compute_levermann_score(t, r6, r12)
                rel = relative_valuation(fund)
                val_sc = valuation_score(rel)
                composite = compute_composite_score(p_score, p_max, l_score, val_sc, signal, r6)
                if composite < min_composite:
                    continue
                vol, max_dd = compute_risk_metrics(data)
                cur = detect_currency(t, fund)
                rates = get_fx_rates()
                last_px = data["Close"].iloc[-1]
                px_eur = to_eur(last_px, cur, rates=rates)
                sig_st = compute_signal_status(data)
                sent = compute_sentiment(data)
                results.append({
                    "Ticker": t,
                    "Name": (fund.get("Name") or t)[:28],
                    "Sektor": (fund.get("Sektor") or "—")[:16],
                    "Kurs €": round(px_eur, 2) if px_eur is not None else "—",
                    "Status": sig_st["label"],
                    "Seit Signal %": sig_st["perf_pct"] if sig_st["perf_pct"] is not None else "—",
                    "Sentiment": sent["score"],
                    "TA-Signal": signal,
                    "Piotroski": f"{p_score}/{p_max}" if p_score is not None else "—",
                    "Composite": composite,
                    "Rel. Bew.": val_sc,
                    "RSI": round(data["RSI"].iloc[-1], 1) if not pd.isna(data["RSI"].iloc[-1]) else "—",
                    "KGV": round(fund["KGV (trailing)"], 1) if fund.get("KGV (trailing)") else "—",
                    "Vol %": vol if vol is not None else "—",
                })
            except Exception:
                pass
            progress.progress((i + 1) / len(tickers))
        status.empty()
        if results:
            df = pd.DataFrame(results).sort_values("Composite", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.success(f"{len(df)} Treffer (sortiert nach Composite-Score).")
            st.caption("Tipp: Hoher Composite + Kaufsignal + positiver Rel. Bew.-Score = interessanteste Kandidaten.")
        else:
            st.info("Keine Treffer mit den aktuellen Filtern. Filter lockern und erneut scannen.")

# ===========================================================================
# 3. Vergleichs-Matrix
# ===========================================================================
elif mode == "📋 Vergleichs-Matrix":
    st.title("📋 Vergleichs-Matrix")
    raw = st.text_input("Ticker oder Namen (kommagetrennt)", "Apple, Microsoft, SAP, NVIDIA, ASML, TSMC")
    tickers = [resolve_ticker(t.strip()) for t in raw.split(",") if t.strip()]
    if tickers and st.button("Vergleichen"):
        rows = []
        rates = get_fx_rates()
        progress = st.progress(0)
        for i, t in enumerate(tickers):
            f = load_fundamentals(t)
            data = load_price_history(t, "2y")
            p_score, _, p_max = compute_piotroski_score(t)
            r6 = price_return(data, 126) if not data.empty else None
            r12 = price_return(data, 252) if not data.empty else None
            l_score, _ = compute_levermann_score(t, r6, r12)
            rel = relative_valuation(f)
            val_sc = valuation_score(rel)
            signal = compute_ta_signal(data) if not data.empty else "—"
            composite = compute_composite_score(p_score, p_max, l_score, val_sc, signal, r6)
            vol, max_dd = compute_risk_metrics(data) if not data.empty else (None, None)
            sig_st = compute_signal_status(data) if not data.empty else {"label": "—", "perf_pct": None}
            sent = compute_sentiment(data) if not data.empty else {"score": "—"}
            cur = detect_currency(t, f)
            px = f.get("Kurs")
            if px is None and not data.empty:
                px = float(data["Close"].iloc[-1])
            px_eur = to_eur(px, cur, rates=rates) if px is not None else None
            rows.append({
                "Ticker": t,
                "Name": (f.get("Name") or t)[:22],
                "Sektor": (f.get("Sektor") or "—")[:14],
                "Kurs €": round(px_eur, 2) if px_eur is not None else "—",
                "Status": sig_st.get("label", "—"),
                "Seit Signal %": sig_st.get("perf_pct") if sig_st.get("perf_pct") is not None else "—",
                "Sentiment": sent.get("score", "—"),
                "KGV": round(f["KGV (trailing)"], 1) if f.get("KGV (trailing)") else "—",
                "ROE": f"{f['ROE']*100:.1f}%" if isinstance(f.get("ROE"), (int, float)) else "—",
                "TA-Signal": signal,
                "Piotroski": f"{p_score}/{p_max}" if p_score is not None else "—",
                "Composite": composite,
                "Vol %": vol if vol is not None else "—",
                "Max DD %": max_dd if max_dd is not None else "—",
            })
            progress.progress((i + 1) / len(tickers))
        st.dataframe(pd.DataFrame(rows).sort_values("Composite", ascending=False),
                     use_container_width=True, hide_index=True)
        st.caption("Sortiert nach Composite-Score. Status = LONG/FLAT/SHORT aus Trendlogik.")

# ===========================================================================
# 4. Portfolio
# ===========================================================================
elif mode == "💼 Portfolio":
    st.title("💼 Portfolio-Tracking")
    portfolio = load_json(PORTFOLIO_FILE, [])
    with st.form("add_position"):
        c1, c2, c3, c4 = st.columns(4)
        t = c1.text_input("Ticker / Name")
        shares = c2.number_input("Stück", min_value=0.0, value=10.0)
        buy_price = c3.number_input("Kaufkurs", min_value=0.0, value=100.0)
        buy_date = c4.date_input("Kaufdatum")
        if st.form_submit_button("Position hinzufügen"):
            resolved = resolve_ticker(t)
            if resolved:
                portfolio.append({
                    "ticker": resolved,
                    "shares": shares,
                    "buy_price": buy_price,
                    "buy_date": str(buy_date),
                })
                save_json(PORTFOLIO_FILE, portfolio)
                st.success(f"{resolved} hinzugefügt")
                st.rerun()
    if portfolio:
        rows = []
        total_invested = 0.0
        total_value = 0.0
        rates = get_fx_rates()
        for p in portfolio:
            data = load_price_history(p["ticker"], "5d")
            current_raw = data["Close"].iloc[-1] if not data.empty else p["buy_price"]
            fund_p = load_fundamentals(p["ticker"])
            cur = detect_currency(p["ticker"], fund_p)
            current = to_eur(current_raw, cur, rates=rates) or current_raw
            buy_eur = to_eur(p["buy_price"], cur, rates=rates) or p["buy_price"]
            value = current * p["shares"]
            invested = buy_eur * p["shares"]
            pnl = value - invested
            pnl_pct = (current / buy_eur - 1) * 100 if buy_eur else 0
            total_invested += invested
            total_value += value
            rows.append({
                "Ticker": p["ticker"],
                "Stück": p["shares"],
                "Kaufkurs €": round(buy_eur, 2),
                "Aktuell €": round(current, 2),
                "Wert €": round(value, 2),
                "G/V €": round(pnl, 2),
                "G/V %": round(pnl_pct, 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if total_invested > 0:
            st.metric("Gesamt-Performance", f"{total_value - total_invested:+.2f} €",
                      f"{(total_value / total_invested - 1) * 100:+.1f} %")

        del_idx = st.number_input(
            "Position löschen (Index ab 0)",
            min_value=0,
            max_value=max(0, len(portfolio) - 1),
            value=0,
            step=1,
        )
        cdel1, cdel2 = st.columns(2)
        if cdel1.button("Diese Position löschen"):
            if 0 <= int(del_idx) < len(portfolio):
                removed = portfolio.pop(int(del_idx))
                save_json(PORTFOLIO_FILE, portfolio)
                st.success(f"{removed.get('ticker', '?')} entfernt")
                st.rerun()
        if cdel2.button("Portfolio komplett leeren"):
            save_json(PORTFOLIO_FILE, [])
            st.rerun()

# ===========================================================================
# 5. Backtesting
# ===========================================================================
elif mode == "📉 Backtesting":
    st.title("📉 Backtesting – vertiefte Signal-Analyse")
    st.caption("Prüft, ob und wie stark historische Signale bei dieser Aktie mit späteren Kursgewinnen korreliert haben.")

    c_in1, c_in2 = st.columns([2, 1])
    with c_in1:
        user_input = st.text_input("Aktie (Name oder Ticker)", value="Apple")
    with c_in2:
        period = st.selectbox("Historie", ["5y", "10y", "max"], index=0)

    signal_map = {
        "Golden Cross (SMA50 > SMA200)": "golden",
        "Death Cross (SMA50 < SMA200)": "death",
        "RSI Überverkauft-Bounce (<30 → >30)": "rsi_oversold",
        "RSI Überkauft-Exit (>70 → <70)": "rsi_overbought",
        "MACD-Bullish-Cross": "macd",
        "Kombiniertes TA-Kaufsignal": "ta",
    }
    signal_choice = st.selectbox("Signal-Typ", list(signal_map.keys()))
    hold_days = st.slider("Feste Haltedauer für Equity-Kurve (Handelstage)", 10, 126, 63, 1)

    ticker = resolve_ticker(user_input) if user_input else ""
    if ticker and ticker.upper() != user_input.upper().strip():
        st.caption(f"→ Interpretiert als: **{ticker}**")

    if ticker and st.button("Backtest starten", type="primary"):
        data = load_price_history(ticker, period)
        if data.empty:
            st.error("Keine Daten.")
        else:
            sig_type = signal_map[signal_choice]
            result = backtest_signals(data, signal_type=sig_type, hold_days=hold_days)

            if result["count"] == 0:
                st.warning("In diesem Zeitraum wurden keine Signale dieses Typs gefunden. Anderen Signal-Typ oder längere Historie wählen.")
            else:
                summary = result["summary"]
                st.subheader(f"📊 {result['count']} Signale · {signal_choice}")

                # Haupt-Metriken
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ø 1 Monat", f"{summary.get('Ø 21T', '—')} %",
                          f"TQ {summary.get('Trefferquote 21T', '—')} %")
                m2.metric("Ø 3 Monate", f"{summary.get('Ø 63T', '—')} %",
                          f"TQ {summary.get('Trefferquote 63T', '—')} %")
                m3.metric("Ø 6 Monate", f"{summary.get('Ø 126T', '—')} %",
                          f"TQ {summary.get('Trefferquote 126T', '—')} %")
                m4.metric("Profit Factor (3M)", f"{summary.get('PF 63T', '—')}",
                          f"Max Verlustserie: {summary.get('max_consec_loss', '—')}")

                # Detail-Stats
                with st.expander("📈 Detail-Statistik", expanded=True):
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.write("**1 Monat (21T)**")
                        st.write(f"Median: {summary.get('Median 21T', '—')} %")
                        st.write(f"Best: {summary.get('Best 21T', '—')} %")
                        st.write(f"Worst: {summary.get('Worst 21T', '—')} %")
                        st.write(f"Ø Win: {summary.get('Ø Win 21T', '—')} %")
                        st.write(f"Ø Loss: {summary.get('Ø Loss 21T', '—')} %")
                    with d2:
                        st.write("**3 Monate (63T)**")
                        st.write(f"Median: {summary.get('Median 63T', '—')} %")
                        st.write(f"Best: {summary.get('Best 63T', '—')} %")
                        st.write(f"Worst: {summary.get('Worst 63T', '—')} %")
                        st.write(f"Ø Win: {summary.get('Ø Win 63T', '—')} %")
                        st.write(f"Ø Loss: {summary.get('Ø Loss 63T', '—')} %")
                    with d3:
                        st.write("**6 Monate (126T)**")
                        st.write(f"Median: {summary.get('Median 126T', '—')} %")
                        st.write(f"Best: {summary.get('Best 126T', '—')} %")
                        st.write(f"Worst: {summary.get('Worst 126T', '—')} %")
                        st.write(f"Ø Win: {summary.get('Ø Win 126T', '—')} %")
                        st.write(f"Ø Loss: {summary.get('Ø Loss 126T', '—')} %")

                # Equity-Kurve
                if result.get("equity"):
                    st.subheader(f"📉 Equity-Kurve (Haltedauer {hold_days} Tage)")
                    eq_df = pd.DataFrame(result["equity"])
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=eq_df["Signal #"], y=eq_df["Kumuliert %"],
                        mode="lines+markers", name="Kumulierte Rendite %",
                        line=dict(color="#26a69a", width=2.5),
                        marker=dict(size=7),
                        fill="tozeroy",
                        fillcolor="rgba(38,166,154,0.15)",
                    ))
                    fig_eq.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.35)")
                    fig_eq.update_layout(
                        **CHART_LAYOUT,
                        height=360,
                        xaxis_title="Signal-Nummer",
                        yaxis_title="Kumuliert %",
                    )
                    st.plotly_chart(fig_eq, use_container_width=True, config=PLOTLY_CONFIG)
                    final = summary.get("equity_final")
                    if final is not None:
                        if final > 0:
                            st.success(f"Kumuliert über alle Signale (je {hold_days}T halten): **{final:+.1f} %**")
                        else:
                            st.warning(f"Kumuliert über alle Signale (je {hold_days}T halten): **{final:+.1f} %**")

                # Chart mit Signal-Markern
                st.subheader("📍 Signale im Kursverlauf")
                fig_sig = go.Figure()
                fig_sig.add_trace(go.Candlestick(
                    x=data.index, open=data["Open"], high=data["High"],
                    low=data["Low"], close=data["Close"], name="Kurs",
                    increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
                    decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
                ))
                if "SMA50" in data.columns:
                    fig_sig.add_trace(go.Scatter(
                        x=data.index, y=data["SMA50"], name="SMA50",
                        line=dict(color="#ffb74d", width=1.8),
                    ))
                if "SMA200" in data.columns:
                    fig_sig.add_trace(go.Scatter(
                        x=data.index, y=data["SMA200"], name="SMA200",
                        line=dict(color="#42a5f5", width=1.8),
                    ))
                sig_dates = result.get("signal_dates") or []
                if sig_dates:
                    sig_prices = [float(data.loc[d, "Close"]) for d in sig_dates if d in data.index]
                    sig_x = [d for d in sig_dates if d in data.index]
                    is_sell = sig_type in ("death", "rsi_overbought")
                    fig_sig.add_trace(go.Scatter(
                        x=sig_x, y=sig_prices, mode="markers",
                        name="Sell-Signal" if is_sell else "Buy-Signal",
                        marker=dict(
                            symbol="triangle-down" if is_sell else "triangle-up",
                            size=14,
                            color="#ef5350" if is_sell else "#26a69a",
                            line=dict(width=1, color="white"),
                        ),
                    ))
                fig_sig.update_layout(
                    **CHART_LAYOUT,
                    height=480,
                    xaxis_rangeslider_visible=False,
                    yaxis_title="Kurs",
                )
                st.plotly_chart(fig_sig, use_container_width=True, config=PLOTLY_CONFIG)
                st.caption("📱 Zwei Finger = zoomen · Ein Finger = schieben")

                # Tabelle
                st.subheader("Einzelne Signale")
                details_df = pd.DataFrame(result["details"])
                st.dataframe(details_df, use_container_width=True, hide_index=True)

                # Bewertung
                avg_3m = summary.get("Ø 63T")
                hit_3m = summary.get("Trefferquote 63T")
                pf = summary.get("PF 63T")
                if avg_3m is not None and hit_3m is not None:
                    score = 0
                    if avg_3m > 5:
                        score += 2
                    elif avg_3m > 0:
                        score += 1
                    if hit_3m >= 55:
                        score += 2
                    elif hit_3m >= 50:
                        score += 1
                    if pf is not None and pf >= 1.5:
                        score += 2
                    elif pf is not None and pf >= 1.0:
                        score += 1

                    if score >= 5:
                        st.success(
                            f"**Starkes historisches Signal** (Score {score}/6): "
                            f"Positive Ø-Rendite, solide Trefferquote"
                            + (f", Profit Factor {pf}" if pf else "")
                            + ". Trotzdem keine Zukunftsgarantie."
                        )
                    elif score >= 3:
                        st.info(
                            f"**Brauchbares Signal** (Score {score}/6): "
                            "Historisch leicht positiver Rand – als Baustein ok, nicht allein handeln."
                        )
                    else:
                        st.warning(
                            f"**Schwaches / gemischtes Signal** (Score {score}/6): "
                            "Historisch wenig Vorteil. Vorsicht bei alleinigem Vertrauen."
                        )

                st.info(
                    "Hinweis: Backtests nutzen feste Zeithorizonte (keine Stop-Loss/Take-Profit-Simulation). "
                    "Vergangene Korrelation ≠ zukünftige Performance. Immer mit Risiko- und Qualitätsfiltern kombinieren."
                )
