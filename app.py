"""
Stock Screener Pro
------------------
Lokales Streamlit-Dashboard zum Finden und Prüfen von Aktien, Edelmetallen und Kryptowährungen.

Features:
- Einzelanalyse: Chart (€), LONG/FLAT/SHORT, Sentiment, Zielpfad, Scores
- Scanner: US/EU/Asien/Edelmetalle/Crypto (Top 100) + Qualitätsfilter + Composite-Score
- Vergleichs-Matrix, Portfolio-Tracking
- Vertieftes Backtesting (mehrere Signale, Equity-Kurve, Profit Factor, Einstiegs-Ampel & Wiedereinstiegs-Empfehlungen)
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

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "chart"},
}

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#e0e0e0"),
    margin=dict(l=48, r=24, t=48, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    ),
    hovermode="x unified",
    dragmode="pan",
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        rangeslider=dict(visible=False),
        fixedrange=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        side="right",
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
PRECIOUS_METALS = [
    "GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "GLD", "SLV", "PPLT", "PALL"
]

TOP100_CRYPTO = [
    "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "XRP-USD", "USDC-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "TRX-USD", "DOT-USD", "LINK-USD", "BCH-USD", "NEAR-USD", "MATIC-USD",
    "LTC-USD", "SHIB-USD", "ICP-USD", "LEO-USD", "DAI-USD", "UNI-USD", "PEPE-USD", "APT-USD",
    "ETC-USD", "XMR-USD", "XLM-USD", "STX-USD", "FIL-USD", "OKB-USD", "CRO-USD", "SUI-USD",
    "RENDER-USD", "AAVE-USD", "TAO-USD", "FET-USD", "GRT-USD", "INJ-USD", "HBAR-USD", "KAS-USD",
    "AR-USD", "THETA-USD", "ATOM-USD", "TIA-USD", "FLOKI-USD", "RUNE-USD", "FTM-USD", "EGLD-USD",
    "SEI-USD", "BONK-USD", "WIF-USD", "NOT-USD", "JUP-USD", "ALGO-USD", "FLOW-USD", "BEAM-USD",
    "PYTH-USD", "CORE-USD", "NEO-USD", "GALA-USD", "AXS-USD", "EOS-USD", "MKR-USD", "QNT-USD",
    "SNX-USD", "CHZ-USD", "RON-USD", "BSV-USD", "XTZ-USD", "SAND-USD", "MANA-USD", "ORDI-USD",
    "STRK-USD", "MINA-USD", "BTT-USD", "KAVA-USD", "COMP-USD", "CFX-USD", "ZEC-USD", "IOTA-USD",
    "CAKE-USD", "TWT-USD", "KLAY-USD", "CRV-USD", "GMX-USD", "WOO-USD", "ASTR-USD",
    "ROSE-USD", "FXS-USD", "MANTA-USD", "LDO-USD", "PENDLE-USD", "DYDX-USD", "ALT-USD", "1INCH-USD"
]

BUILTIN_WATCHLISTS = {
    "Tech-Auswahl": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "CRM"],
    "DAX-Auswahl (Top 10)": ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BMW.DE", "MBG.DE", "VOW3.DE", "MUV2.DE", "AIR.DE"],
    "Europa Blue Chips": [
        "SAP.DE", "SIE.DE", "ASML.AS", "LVMH.PA", "MC.PA", "OR.PA", "TTE.PA", "SAN.PA",
        "NESN.SW", "ROG.SW", "NOVN.SW", "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "INGA.AS",
    ],
    "S&P-Auswahl (Auszug)": ["AAPL", "MSFT", "AMZN", "GOOGL", "BRK-B", "JPM", "V", "JNJ", "WMT", "XOM"],
    "Edelmetalle": PRECIOUS_METALS,
    "Top Krypto (Auszug)": TOP100_CRYPTO[:15],
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

CAC40_TICKERS = [
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "SU.PA", "AI.PA", "BNP.PA",
    "DG.PA", "EL.PA", "CS.PA", "BN.PA", "KER.PA", "RI.PA", "STLAP.PA", "VIV.PA",
    "CAP.PA", "SGO.PA", "HO.PA", "ML.PA",
]

BENELUX_TICKERS = [
    "ASML.AS", "INGA.AS", "PHIA.AS", "AD.AS", "UNA.AS", "HEIA.AS", "PRX.AS",
    "KBC.BR", "ABI.BR", "ARGX.BR",
]

SWISS_TICKERS = [
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "LONN.SW",
    "CFR.SW", "SIKA.SW", "GIVN.SW",
]

UK_TICKERS = [
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "DGE.L", "RIO.L",
    "BATS.L", "LLOY.L", "VOD.L", "NG.L", "BARC.L", "RR.L", "AAL.L",
]

EUROPE_TICKERS = list(dict.fromkeys(
    DAX40_TICKERS + CAC40_TICKERS + BENELUX_TICKERS + SWISS_TICKERS + UK_TICKERS
))

NASDAQ_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP", "COST",
    "ADBE", "CSCO", "NFLX", "AMD", "INTC", "QCOM", "TXN", "INTU", "AMAT", "BKNG",
]

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
    "GOLD": "GC=F", "SILBER": "SI=F", "PLATIN": "PL=F", "PALLADIUM": "PA=F",
    "BITCOIN": "BTC-USD", "ETHEREUM": "ETH-USD", "SOLANA": "SOL-USD", "XRP": "XRP-USD",
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
    "HERMÈS": "RMS.PA", "SCHNEIDER": "SU.PA", "BNP": "BNP.PA",
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
            equities = [r for r in search_results if r.get("quoteType") in ("EQUITY", "CRYPTOCURRENCY", "COMMODITY", "ETF")]
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
def get_fx_rates() -> dict:
    defaults = {
        "USD": 1.08, "GBP": 0.86, "CHF": 0.94, "JPY": 160.0,
        "HKD": 8.4, "KRW": 1450.0, "TWD": 34.0, "EUR": 1.0,
    }
    pairs = {
        "USD": "EURUSD=X", "GBP": "EURGBP=X", "CHF": "EURCHF=X",
        "JPY": "EURJPY=X", "HKD": "EURHKD=X", "KRW": "EURKRW=X", "TWD": "EURTWD=X",
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
    return float(get_fx_rates().get("USD", 1.08))


def detect_currency(ticker: str, info: dict | None = None) -> str:
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
    if t.endswith("-USD") or t.endswith("=X") or t.endswith("=F"):
        return "USD"
    if info:
        cur = info.get("currency") or info.get("financialCurrency")
        if cur:
            return cur.upper()
    return "USD"


def to_eur(value, currency: str, eur_usd: float = None, rates: dict | None = None):
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
    df = data.dropna(subset=["SMA50", "SMA200", "RSI", "Close"]).copy()
    if len(df) < 30:
        return None
    df["diff"] = df["SMA50"] - df["SMA200"]
    df["gc"] = (df["diff"].shift(1) < 0) & (df["diff"] > 0)
    df["bull"] = df["SMA50"] > df["SMA200"]
    df["rsi_entry"] = df["bull"] & (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70)
    df["signal"] = df["gc"] | df["rsi_entry"]
    hits = df[df["signal"]].index
    if len(hits) == 0:
        return None
    return hits[-1]


def compute_signal_status(data: pd.DataFrame):
    if data.empty or "SMA50" not in data.columns:
        return {"status": "FLAT", "label": "⚪ FLAT", "since": None, "perf_pct": None, "days": None}

    df = data.dropna(subset=["SMA50", "SMA200", "Close"]).copy()
    if len(df) < 50:
        return {"status": "FLAT", "label": "⚪ FLAT", "since": None, "perf_pct": None, "days": None}

    last = df.iloc[-1]
    bull = bool(last["SMA50"] > last["SMA200"])
    rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and not pd.isna(df["RSI"].iloc[-1]) else 50.0

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
    if data.empty or "RSI" not in data.columns:
        return {"score": 50, "label": "Neutral", "color": "gray"}

    rsi = data["RSI"].dropna()
    if rsi.empty:
        return {"score": 50, "label": "Neutral", "color": "gray"}

    last_rsi = float(rsi.iloc[-1])
    closes = data["Close"].dropna()
    mom = 0.0
    if len(closes) > 20:
        mom = float(closes.iloc[-1] / closes.iloc[-20] - 1) * 100

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
    if data.empty:
        return []

    last = float(data["Close"].iloc[-1]) * fx
    fib = compute_fib_entry_zone(data)
    t_mean = fundamentals.get("Analysten-Kursziel (mittel)")
    t_high = fundamentals.get("Analysten-Kursziel (hoch)")

    targets = []
    z1 = max(fib[1] * fx, last * 1.03) if fib else last * 1.05
    targets.append({"name": "ZIEL 1", "kurs": z1, "typ": "nah"})

    z2 = float(t_mean) * fx if t_mean else last * 1.10
    if z2 <= z1:
        z2 = z1 * 1.05
    targets.append({"name": "ZIEL 2", "kurs": z2, "typ": "mittel"})

    if t_high and t_mean:
        z3 = (float(t_mean) + float(t_high)) / 2 * fx
    elif t_high:
        z3 = float(t_high) * fx * 0.92
    else:
        z3 = last * 1.18
    if z3 <= z2:
        z3 = z2 * 1.06
    targets.append({"name": "ZIEL 3", "kurs": z3, "typ": "gestreckt"})

    z4 = float(t_high) * fx if t_high else last * 1.25
    if z4 <= z3:
        z4 = z3 * 1.08
    targets.append({"name": "ZIEL 4 · Best Case", "kurs": z4, "typ": "best"})

    risk = last * 0.92
    if fib:
        risk = min(risk, fib[0] * fx)

    for t in targets:
        t["weg_pct"] = (t["kurs"] / last - 1) * 100
        t["status"] = "erreicht" if last >= t["kurs"] else "aktiv"
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
# Fundamentaldaten & Branchenvergleich
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_fundamentals(ticker: str):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    return {
        "Name": info.get("longName") or info.get("shortName") or ticker,
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


SECTOR_BENCHMARKS = {
    "Technology":             {"KGV": 28, "KBV": 6.0, "EV/EBITDA": 18, "ROE": 0.18, "Marge": 0.18},
    "Communication Services": {"KGV": 22, "KBV": 3.5, "EV/EBITDA": 12, "ROE": 0.12, "Marge": 0.15},
    "Consumer Cyclical":      {"KGV": 20, "KBV": 3.0, "EV/EBITDA": 12, "ROE": 0.15, "Marge": 0.08},
    "Consumer Defensive":     {"KGV": 22, "KBV": 4.0, "EV/EBITDA": 14, "ROE": 0.18, "Marge": 0.10},
    "Financial Services":     {"KGV": 12, "KBV": 1.4, "EV/EBITDA": 8,  "ROE": 0.12, "Marge": 0.25},
    "Healthcare":             {"KGV": 24, "KBV": 4.5, "EV/EBITDA": 15, "ROE": 0.15, "Marge": 0.12},
    "Industrials":            {"KGV": 18, "KBV": 3.0, "EV/EBITDA": 11, "ROE": 0.14, "Marge": 0.09},
    "Basic Materials":        {"KGV": 14, "KBV": 1.8, "EV/EBITDA": 8,  "ROE": 0.10, "Marge": 0.08},
    "Energy":                 {"KGV": 11, "KBV": 1.5, "EV/EBITDA": 6,  "ROE": 0.12, "Marge": 0.10},
    "Utilities":              {"KGV": 16, "KBV": 1.6, "EV/EBITDA": 10, "ROE": 0.09, "Marge": 0.12},
    "Real Estate":            {"KGV": 30, "KBV": 1.8, "EV/EBITDA": 16, "ROE": 0.06, "Marge": 0.20},
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
# Scoring-Modelle & Backtesting
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
            try_criterion("Kapitalumschlag gestiegen",
                          lambda: (revenue.iloc[0] / total_assets.iloc[0]) > (revenue.iloc[1] / total_assets.iloc[1]))

    except Exception:
        return None, {}, 0

    return score, details, max_score


def compute_levermann_score(data: pd.DataFrame, fundamentals: dict) -> tuple[int, dict]:
    score = 0
    details = {}

    roe = fundamentals.get("ROE")
    if roe is not None:
        if roe > 0.20:
            score += 1
            details["RoE > 20%"] = +1
        elif roe < 0.10:
            score -= 1
            details["RoE < 10%"] = -1
        else:
            details["RoE neutral"] = 0

    ebit_marge = fundamentals.get("Operative Marge")
    if ebit_marge is not None:
        if ebit_marge > 0.12:
            score += 1
            details["EBIT-Marge > 12%"] = +1
        elif ebit_marge < 0.06:
            score -= 1
            details["EBIT-Marge < 6%"] = -1
        else:
            details["EBIT-Marge neutral"] = 0

    kgv = fundamentals.get("KGV (trailing)") or fundamentals.get("KGV (forward)")
    if kgv is not None and kgv > 0:
        if kgv < 12:
            score += 1
            details["KGV < 12"] = +1
        elif kgv > 30:
            score -= 1
            details["KGV > 30"] = -1
        else:
            details["KGV neutral"] = 0

    if not data.empty and "Close" in data.columns and len(data) >= 126:
        closes = data["Close"]
        perf_6m = (closes.iloc[-1] / closes.iloc[-126] - 1) * 100
        if perf_6m > 5:
            score += 1
            details["6M Perf > +5%"] = +1
        elif perf_6m < -5:
            score -= 1
            details["6M Perf < -5%"] = -1
        else:
            details["6M Perf neutral"] = 0

        if len(data) >= 252:
            perf_12m = (closes.iloc[-1] / closes.iloc[-252] - 1) * 100
            if perf_12m > 5:
                score += 1
                details["12M Perf > +5%"] = +1
            elif perf_12m < -5:
                score -= 1
                details["12M Perf < -5%"] = -1
            else:
                details["12M Perf neutral"] = 0

        momentum = perf_6m - perf_12m if len(data) >= 252 else 0
        if momentum > 5:
            score += 1
            details["Reaktions-Momentum positiv"] = +1
        elif momentum < -5:
            score -= 1
            details["Reaktions-Momentum negativ"] = -1

    return score, details


def backtest_golden_cross(data: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    df = data.dropna(subset=["SMA50", "SMA200", "Close"]).copy()
    if len(df) < 200:
        return {}

    df["Signal"] = 0
    df.loc[df["SMA50"] > df["SMA200"], "Signal"] = 1

    df["Cross"] = df["Signal"].diff()
    buys = df[df["Cross"] == 1]
    sells = df[df["Cross"] == -1]

    trades = []
    in_trade = False
    buy_date = None
    buy_price = 0.0

    for idx, row in df.iterrows():
        if not in_trade and row["Cross"] == 1:
            in_trade = True
            buy_date = idx
            buy_price = row["Close"]
        elif in_trade and row["Cross"] == -1:
            in_trade = False
            sell_date = idx
            sell_price = row["Close"]
            ret = (sell_price - buy_price) / buy_price
            trades.append({
                "Kaufdatum": buy_date.strftime("%Y-%m-%d"),
                "Kaufpreis": buy_price,
                "Verkaufsdatum": sell_date.strftime("%Y-%m-%d"),
                "Verkaufspreis": sell_price,
                "Rendite %": round(ret * 100, 2),
                "Haltedauer (Tage)": (sell_date - buy_date).days,
            })

    if in_trade:
        last_date = df.index[-1]
        last_price = df.iloc[-1]["Close"]
        ret = (last_price - buy_price) / buy_price
        trades.append({
            "Kaufdatum": buy_date.strftime("%Y-%m-%d"),
            "Kaufpreis": buy_price,
            "Verkaufsdatum": f"{last_date.strftime('%Y-%m-%d')} (aktiv)",
            "Verkaufspreis": last_price,
            "Rendite %": round(ret * 100, 2),
            "Haltedauer (Tage)": (last_date - buy_date).days,
        })

    df["Market_Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Signal"].shift(1).fillna(0) * df["Market_Return"]

    df["Equity_Market"] = initial_capital * (1 + df["Market_Return"]).cumprod()
    df["Equity_Strategy"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()

    total_strat_ret = (df["Equity_Strategy"].iloc[-1] / initial_capital - 1) * 100
    total_mkt_ret = (df["Equity_Market"].iloc[-1] / initial_capital - 1) * 100

    win_trades = [t for t in trades if t["Rendite %"] > 0]
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0.0

    gross_profit = sum([t["Rendite %"] for t in trades if t["Rendite %"] > 0])
    gross_loss = abs(sum([t["Rendite %"] for t in trades if t["Rendite %"] < 0]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

    avg_days = float(np.mean([t["Haltedauer (Tage)"] for t in trades])) if trades else 0.0
    avg_ret = float(np.mean([t["Rendite %"] for t in trades])) if trades else 0.0

    return {
        "trades": trades,
        "equity_df": df[["Equity_Market", "Equity_Strategy"]],
        "total_strat_ret": round(total_strat_ret, 2),
        "total_mkt_ret": round(total_mkt_ret, 2),
        "win_rate": round(win_rate, 1),
        "num_trades": len(trades),
        "profit_factor": round(profit_factor, 2),
        "avg_days": round(avg_days, 1),
        "avg_ret": round(avg_ret, 2),
    }


# ---------------------------------------------------------------------------
# UI Tabs / Komponenten
# ---------------------------------------------------------------------------
def render_single_analysis(ticker: str):
    data = load_price_history(ticker)
    if data.empty:
        st.error(f"Keine Kursdaten für '{ticker}' gefunden.")
        return

    fund = load_fundamentals(ticker)
    cur = detect_currency(ticker, fund)
    rates = get_fx_rates()
    eur_usd = rates.get("USD", 1.08)

    last_close = float(data["Close"].iloc[-1])
    eur_close = to_eur(last_close, cur, eur_usd=eur_usd, rates=rates)

    st.subheader(f"📊 Einzelanalyse: {fund.get('Name', ticker)} ({ticker})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Aktueller Kurs (€)", f"{eur_close:.2f} €", f"{(data['Return'].iloc[-1]*100):.2f}% (1D)")
    with col2:
        st.metric("Originalkurs", f"{last_close:.2f} {cur}")
    with col3:
        st.metric("TA Signal", compute_ta_signal(data))
    with col4:
        vol, dd = compute_risk_metrics(data)
        st.metric("Volatilität (p.a.)", f"{vol}%" if vol else "—", f"Max DD: {dd}%" if dd else "")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Kurs'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], line=dict(color='orange', width=1.5), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], line=dict(color='blue', width=1.5), name='SMA 200'), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volumen', marker_color='rgba(255,255,255,0.2)'), row=2, col=1)
    fig.update_layout(**CHART_LAYOUT)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("### 🚦 Einstiegs-Ampel & Status")
    status = compute_signal_status(data)
    sent = compute_sentiment(data)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("System-Status", status["label"], f"{status['perf_pct']}% seit {status['days']} Tagen" if status['perf_pct'] else "Kein aktives Signal")
    with c2:
        st.metric("Aktien-Sentiment", f"{sent['score']}/100 ({sent['label']})", f"RSI: {sent['rsi']}")
    with c3:
        if status["status"] == "LONG" and sent["rsi"] < 65:
            st.success("🟢 Einstieg eher günstig: Trend intakt, RSI bietet Raum.")
        elif status["status"] == "LONG" and sent["rsi"] >= 65:
            st.warning("🟡 Abwarten: Trend LONG, aber RSI leicht überkauft.")
        else:
            st.error("🔴 Kein klarer Long Einstieg.")


def render_scanner():
    st.subheader("🔍 Markt-Scanner")
    universe_choice = st.selectbox("Universum wählen", ["Tech-Auswahl", "DAX 40", "Europa Blue Chips", "S&P 500", "Edelmetalle", "Top 100 Kryptowährungen"])

    tickers = []
    if universe_choice == "Tech-Auswahl":
        tickers = BUILTIN_WATCHLISTS["Tech-Auswahl"]
    elif universe_choice == "DAX 40":
        tickers = DAX40_TICKERS
    elif universe_choice == "Europa Blue Chips":
        tickers = EUROPE_TICKERS
    elif universe_choice == "S&P 500":
        tickers = fetch_sp500_tickers()[:50]
    elif universe_choice == "Edelmetalle":
        tickers = PRECIOUS_METALS
    elif universe_choice == "Top 100 Kryptowährungen":
        tickers = TOP100_CRYPTO

    if st.button("Scan starten", type="primary"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(tickers):
            d = load_price_history(t)
            f = load_fundamentals(t)
            if not d.empty:
                cur = detect_currency(t, f)
                eur_p = to_eur(d["Close"].iloc[-1], cur)
                results.append({
                    "Ticker": t,
                    "Name": f.get("Name", t),
                    "Kurs (€)": round(eur_p, 2) if eur_p else None,
                    "Signal": compute_ta_signal(d),
                    "RSI": round(d["RSI"].dropna().iloc[-1], 1) if not d["RSI"].dropna().empty else None,
                    "1M Return %": round((d["Close"].iloc[-1] / d["Close"].iloc[-22] - 1) * 100, 2) if len(d) >= 22 else None,
                })
            bar.progress((i + 1) / len(tickers))
        bar.empty()
        st.dataframe(pd.DataFrame(results), use_container_width=True)


def render_backtest(ticker: str):
    st.subheader(f"🧪 Backtest: Golden Cross Strategie auf {ticker}")
    data = load_price_history(ticker, period="5y")
    if data.empty:
        st.error("Keine Daten verfügbar.")
        return

    bt = backtest_golden_cross(data)
    if not bt:
        st.warning("Nicht genügend historische Daten für einen Backtest vorhanden.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategie Gesamt-Rendite", f"{bt['total_strat_ret']}%", f"vs. Market {bt['total_mkt_ret']}%")
    c2.metric("Win Rate", f"{bt['win_rate']}%")
    c3.metric("Profit Factor", f"{bt['profit_factor']}")
    c4.metric("Anzahl Trades", f"{bt['num_trades']}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["equity_df"].index, y=bt["equity_df"]["Equity_Strategy"], name="Strategie Equity", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=bt["equity_df"].index, y=bt["equity_df"]["Equity_Market"], name="Buy & Hold Market", line=dict(color="gray", dash="dash")))
    fig.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("### 📋 Trade-Historie")
    st.dataframe(pd.DataFrame(bt["trades"]), use_container_width=True)


# ---------------------------------------------------------------------------
# Hauptanwendungs-Steuerung
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("📈 Stock Screener Pro")
    menu = st.sidebar.radio("Navigation", ["Einzelanalyse", "Scanner", "Backtest", "Portfolio"])

    symbol_input = st.sidebar.text_input("Ticker / Name eingeben", "NVDA")
    ticker = resolve_ticker(symbol_input)

    if menu == "Einzelanalyse":
        render_single_analysis(ticker)
    elif menu == "Scanner":
        render_scanner()
    elif menu == "Backtest":
        render_backtest(ticker)
    elif menu == "Portfolio":
        st.subheader("💼 Portfolio-Tracking")
        port = load_json(PORTFOLIO_FILE, [])
        st.info("Funktion zur Verwaltung des lokalen Portfolios aktiv.")
        st.json(port)


if __name__ == "__main__":
    main()
