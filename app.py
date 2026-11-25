import copy
import datetime as dt
import io
import math
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. PAGE CONFIG & GLOBAL STYLING
# ==============================================================================
st.set_page_config(
    page_title="Aktien & Makro Dashboard Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    div[data-testid="stMetric"] {
        background-color: #161b26;
        border: 1px solid #262e3f;
        padding: 12px 16px;
        border-radius: 10px;
    }
    .fred-card {
        background-color: #161b26;
        border: 1px solid #283144;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        min-height: 132px;
    }
    .fred-card-title {
        font-size: 0.90rem;
        font-weight: 600;
        color: #c9d4e8;
        margin-bottom: 6px;
    }
    .fred-card-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .fred-card-sub {
        font-size: 0.78rem;
        color: #93a1bc;
        margin-top: 6px;
    }
    .badge-green {
        display: inline-block;
        background: rgba(34, 197, 94, 0.16);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .badge-yellow {
        display: inline-block;
        background: rgba(234, 179, 8, 0.16);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .badge-red {
        display: inline-block;
        background: rgba(239, 68, 68, 0.16);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .info-callout {
        background-color: #172033;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 8px;
        color: #dbe6f6;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .judgment-box {
        background-color: #151c28;
        border: 1px solid #2b374e;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 8px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,38,0.55)",
    font=dict(color="#e2e8f0", size=12),
    margin=dict(l=40, r=24, t=46, b=36),
    xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.12)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.12)", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
)


def apply_chart_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Safely merges CHART_LAYOUT with overrides without TypeError on duplicate keys."""
    layout = copy.deepcopy(CHART_LAYOUT)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **val}
        else:
            layout[key] = val
    fig.update_layout(**layout)
    return fig


def fmt_de(val: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "—"
    s = f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s}{suffix}"


# ==============================================================================
# 2. STOCK UNIVERSE (FULLY SPELLED-OUT NAMES, USA / EUROPE / CHINA / JAPAN)
# ==============================================================================
STOCK_UNIVERSE: Dict[str, dict] = {
    # USA Tech & AI
    "NVDA": {"name": "NVIDIA Corporation", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "MSFT": {"name": "Microsoft Corporation", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "AAPL": {"name": "Apple Inc.", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "GOOGL": {"name": "Alphabet Inc. (Class A)", "region": "USA", "sector": "Kommunikation", "currency": "USD"},
    "AMZN": {"name": "Amazon.com Inc.", "region": "USA", "sector": "Zyklischer Konsum", "currency": "USD"},
    "META": {"name": "Meta Platforms Inc.", "region": "USA", "sector": "Kommunikation", "currency": "USD"},
    "TSLA": {"name": "Tesla Inc.", "region": "USA", "sector": "Zyklischer Konsum", "currency": "USD"},
    "AVGO": {"name": "Broadcom Inc.", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "AMD": {"name": "Advanced Micro Devices Inc.", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "PLTR": {"name": "Palantir Technologies Inc.", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "NFLX": {"name": "Netflix Inc.", "region": "USA", "sector": "Kommunikation", "currency": "USD"},
    "TTWO": {"name": "Take-Two Interactive Software Inc.", "region": "USA", "sector": "Kommunikation", "currency": "USD"},
    "CRM": {"name": "Salesforce Inc.", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "CRWD": {"name": "CrowdStrike Holdings Inc.", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "PANW": {"name": "Palo Alto Networks Inc.", "region": "USA", "sector": "Technologie", "currency": "USD"},
    # USA Crypto / Fintech / Energy / Healthcare
    "COIN": {"name": "Coinbase Global Inc.", "region": "USA", "sector": "Finanzen", "currency": "USD"},
    "MSTR": {"name": "Strategy Inc. (MicroStrategy)", "region": "USA", "sector": "Technologie", "currency": "USD"},
    "HOOD": {"name": "Robinhood Markets Inc.", "region": "USA", "sector": "Finanzen", "currency": "USD"},
    "PYPL": {"name": "PayPal Holdings Inc.", "region": "USA", "sector": "Finanzen", "currency": "USD"},
    "UNH": {"name": "UnitedHealth Group Inc.", "region": "USA", "sector": "Gesundheit", "currency": "USD"},
    "LLY": {"name": "Eli Lilly and Company", "region": "USA", "sector": "Gesundheit", "currency": "USD"},
    "JPM": {"name": "JPMorgan Chase & Co.", "region": "USA", "sector": "Finanzen", "currency": "USD"},
    "GEV": {"name": "GE Vernova Inc.", "region": "USA", "sector": "Industrie", "currency": "USD"},
    "VST": {"name": "Vistra Corp.", "region": "USA", "sector": "Versorger", "currency": "USD"},
    "NKE": {"name": "NIKE Inc.", "region": "USA", "sector": "Zyklischer Konsum", "currency": "USD"},
    # Deutschland & Europa
    "ADS.DE": {"name": "Adidas AG", "region": "Europa", "sector": "Zyklischer Konsum", "currency": "EUR"},
    "SAP.DE": {"name": "SAP SE", "region": "Europa", "sector": "Technologie", "currency": "EUR"},
    "SIE.DE": {"name": "Siemens AG", "region": "Europa", "sector": "Industrie", "currency": "EUR"},
    "RHM.DE": {"name": "Rheinmetall AG", "region": "Europa", "sector": "Industrie", "currency": "EUR"},
    "ALV.DE": {"name": "Allianz SE", "region": "Europa", "sector": "Finanzen", "currency": "EUR"},
    "MUV2.DE": {"name": "Münchener Rückversicherungs-Gesellschaft AG", "region": "Europa", "sector": "Finanzen", "currency": "EUR"},
    "IFX.DE": {"name": "Infineon Technologies AG", "region": "Europa", "sector": "Technologie", "currency": "EUR"},
    "DTE.DE": {"name": "Deutsche Telekom AG", "region": "Europa", "sector": "Kommunikation", "currency": "EUR"},
    "ASML.AS": {"name": "ASML Holding N.V.", "region": "Europa", "sector": "Technologie", "currency": "EUR"},
    "NOVO-B.CO": {"name": "Novo Nordisk A/S", "region": "Europa", "sector": "Gesundheit", "currency": "DKK"},
    # China / Hongkong (mit korrekter HKD->EUR Umrechnung!)
    "1810.HK": {"name": "Xiaomi Corporation", "region": "China / Asien", "sector": "Technologie", "currency": "HKD"},
    "1211.HK": {"name": "BYD Company Limited", "region": "China / Asien", "sector": "Zyklischer Konsum", "currency": "HKD"},
    "9988.HK": {"name": "Alibaba Group Holding Limited", "region": "China / Asien", "sector": "Zyklischer Konsum", "currency": "HKD"},
    "0700.HK": {"name": "Tencent Holdings Limited", "region": "China / Asien", "sector": "Kommunikation", "currency": "HKD"},
    "9618.HK": {"name": "JD.com Inc.", "region": "China / Asien", "sector": "Zyklischer Konsum", "currency": "HKD"},
    "3690.HK": {"name": "Meituan", "region": "China / Asien", "sector": "Zyklischer Konsum", "currency": "HKD"},
    "9868.HK": {"name": "XPeng Inc.", "region": "China / Asien", "sector": "Zyklischer Konsum", "currency": "HKD"},
    # Japan (mit korrekter JPY->EUR Umrechnung!)
    "7203.T": {"name": "Toyota Motor Corporation", "region": "Japan", "sector": "Zyklischer Konsum", "currency": "JPY"},
    "6758.T": {"name": "Sony Group Corporation", "region": "Japan", "sector": "Kommunikation", "currency": "JPY"},
    "8035.T": {"name": "Tokyo Electron Limited", "region": "Japan", "sector": "Technologie", "currency": "JPY"},
    "6857.T": {"name": "Advantest Corporation", "region": "Japan", "sector": "Technologie", "currency": "JPY"},
    "7974.T": {"name": "Nintendo Co. Ltd.", "region": "Japan", "sector": "Kommunikation", "currency": "JPY"},
    "9984.T": {"name": "SoftBank Group Corp.", "region": "Japan", "sector": "Kommunikation", "currency": "JPY"},
}

LEADING_INDICES = [
    {"ticker": "^GSPC", "name": "S&P 500 (USA)", "region": "USA"},
    {"ticker": "^IXIC", "name": "Nasdaq Composite (USA)", "region": "USA"},
    {"ticker": "^NDX", "name": "Nasdaq 100 (USA)", "region": "USA"},
    {"ticker": "^DJI", "name": "Dow Jones Industrial Average (USA)", "region": "USA"},
    {"ticker": "^GDAXI", "name": "DAX Performance-Index (Deutschland)", "region": "Europa"},
    {"ticker": "^MDAXI", "name": "MDAX (Deutschland)", "region": "Europa"},
    {"ticker": "^STOXX50E", "name": "EURO STOXX 50 (Eurozone)", "region": "Europa"},
    {"ticker": "^N225", "name": "Nikkei 225 (Japan)", "region": "Asien"},
    {"ticker": "^HSI", "name": "Hang Seng Index (Hongkong / China)", "region": "Asien"},
    {"ticker": "GC=F", "name": "Gold Futures (USD/oz)", "region": "Rohstoffe"},
    {"ticker": "BTC-USD", "name": "Bitcoin (USD)", "region": "Krypto"},
]


# ==============================================================================
# 3. DATA FETCHING & FX CONVERSION HELPERS
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_fx_rates_to_eur() -> Dict[str, float]:
    """Returns multiplier to convert 1 unit of foreign currency into EUR."""
    rates = {"EUR": 1.0}
    pairs = {
        "USD": "EURUSD=X",
        "HKD": "EURHKD=X",
        "JPY": "EURJPY=X",
        "GBP": "EURGBP=X",
        "CHF": "EURCHF=X",
        "DKK": "EURDKK=X",
        "CNY": "EURCNY=X",
    }
    fallbacks = {
        "USD": 1.08,
        "HKD": 8.42,
        "JPY": 165.0,
        "GBP": 0.84,
        "CHF": 0.94,
        "DKK": 7.46,
        "CNY": 7.80,
    }
    for ccy, pair in pairs.items():
        try:
            hist = yf.Ticker(pair).history(period="5d")
            if not hist.empty and "Close" in hist.columns:
                px_val = float(hist["Close"].dropna().iloc[-1])
                if px_val > 0:
                    rates[ccy] = 1.0 / px_val
                    continue
        except Exception:
            pass
        rates[ccy] = 1.0 / fallbacks[ccy]
    return rates


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def resolve_stock_name(ticker: str, info: Optional[dict] = None) -> str:
    t_up = ticker.upper().strip()
    if t_up in STOCK_UNIVERSE:
        return STOCK_UNIVERSE[t_up]["name"]
    if info:
        name = info.get("longName") or info.get("shortName")
        if name:
            return str(name)
    return t_up


def resolve_currency_and_fx(ticker: str, info: Optional[dict] = None) -> Tuple[str, float]:
    t_up = ticker.upper().strip()
    rates = get_fx_rates_to_eur()
    ccy = None
    if t_up in STOCK_UNIVERSE:
        ccy = STOCK_UNIVERSE[t_up]["currency"]
    elif info and info.get("currency"):
        ccy = str(info["currency"]).upper()
    else:
        if t_up.endswith(".DE") or t_up.endswith(".AS") or t_up.endswith(".PA"):
            ccy = "EUR"
        elif t_up.endswith(".HK"):
            ccy = "HKD"
        elif t_up.endswith(".T"):
            ccy = "JPY"
        elif t_up.endswith(".L"):
            ccy = "GBP"
        elif t_up.endswith(".SW"):
            ccy = "CHF"
        else:
            ccy = "USD"
    # Handle GBp (pence)
    if ccy == "GBP" and t_up.endswith(".L") and info and info.get("currency") == "GBp":
        return "GBp", rates.get("GBP", 1.19) / 100.0
    return ccy, rates.get(ccy, 1.0)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    close = out["Close"]
    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()
    out["BB_MID"] = out["SMA20"]
    std20 = close.rolling(20).std()
    out["BB_UPPER"] = out["BB_MID"] + 2.0 * std20
    out["BB_LOWER"] = out["BB_MID"] - 2.0 * std20

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    return out


def compute_scores_and_levels(df: pd.DataFrame, fx_to_eur: float, info: dict) -> dict:
    """Computes Trend Score, TFA Turnaround Score, Entry Zones, Targets & Judgment."""
    d = add_technical_indicators(df)
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last

    close_nat = float(last["Close"])
    close_eur = close_nat * fx_to_eur
    prev_eur = float(prev["Close"]) * fx_to_eur
    chg_1d = ((close_eur / prev_eur) - 1.0) * 100.0 if prev_eur > 0 else 0.0

    sma20 = float(last["SMA20"]) * fx_to_eur if pd.notna(last["SMA20"]) else close_eur
    sma50 = float(last["SMA50"]) * fx_to_eur if pd.notna(last["SMA50"]) else close_eur
    sma200 = float(last["SMA200"]) * fx_to_eur if pd.notna(last["SMA200"]) else close_eur
    rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
    macd = float(last["MACD"]) if pd.notna(last["MACD"]) else 0.0
    macd_sig = float(last["MACD_SIGNAL"]) if pd.notna(last["MACD_SIGNAL"]) else 0.0

    lookback_252 = d.tail(252)
    high_52w = float(lookback_252["High"].max()) * fx_to_eur
    low_52w = float(lookback_252["Low"].min()) * fx_to_eur
    dist_52w_high = ((close_eur / high_52w) - 1.0) * 100.0 if high_52w > 0 else 0.0
    dist_52w_low = ((close_eur / low_52w) - 1.0) * 100.0 if low_52w > 0 else 0.0

    # 1. Trend-Following Score (0 - 100)
    t_score = 0
    if close_eur > sma200:
        t_score += 30
    if close_eur > sma50:
        t_score += 25
    if sma50 > sma200:
        t_score += 15
    if 48 <= rsi <= 68:
        t_score += 15
    elif 68 < rsi <= 76:
        t_score += 8
    if macd > macd_sig:
        t_score += 10
    if dist_52w_high >= -12.0:
        t_score += 5
    t_score = int(np.clip(t_score, 0, 100))

    # 2. TFA Turnaround Score (0 - 100)
    tfa_score = 0
    if -55.0 <= dist_52w_high <= -18.0:
        tfa_score += 30
    elif dist_52w_high < -55.0:
        tfa_score += 20
    if rsi <= 38:
        tfa_score += 25
    elif 38 < rsi <= 48:
        tfa_score += 18
    if close_eur >= sma20 and close_eur < sma200:
        tfa_score += 20  # Erste Bodenbildung / Rückeroberung GD20
    if macd > macd_sig and close_eur < sma200:
        tfa_score += 15  # Bullisches MACD-Crossover im Abwärtstrend
    if dist_52w_low <= 18.0:
        tfa_score += 10
    tfa_score = int(np.clip(tfa_score, 0, 100))

    # Entry Zone, Stop Loss & Targets (EUR)
    if t_score >= tfa_score:
        # Trendfolge-Einstieg nahe SMA20 / SMA50
        ref_sup = min(close_eur, max(sma50, sma20 * 0.98))
        entry_low = ref_sup * 0.975
        entry_high = min(close_eur * 1.005, ref_sup * 1.015)
        stop_loss = min(sma50 * 0.94, entry_low * 0.92)
        target_1 = max(high_52w, close_eur * 1.12)
        target_2 = target_1 * 1.15
    else:
        # Turnaround-Einstieg nahe aktuellem Unterstützungsniveau / 52W-Tief
        entry_low = max(low_52w * 0.98, close_eur * 0.94)
        entry_high = close_eur * 1.01
        stop_loss = entry_low * 0.91
        target_1 = max(sma200, close_eur * 1.18)
        target_2 = max(high_52w * 0.92, target_1 * 1.15)

    analyst_target_nat = info.get("targetMeanPrice")
    analyst_target_eur = float(analyst_target_nat) * fx_to_eur if analyst_target_nat else None

    # Gesamt-Urteil
    if t_score >= 75 and rsi <= 70:
        verdict = "🟢 Starker Trendfolger – Attraktives Setup"
        verdict_detail = (
            "Intakter Aufwärtstrend oberhalb von GD50 und GD200 mit gesundem RSI-Momentum. "
            "Rücksetzer in die Einstiegszone bieten ein gutes Chance-Risiko-Verhältnis."
        )
    elif t_score >= 75 and rsi > 70:
        verdict = "🟡 Starker Trend, aber kurzfristig überkauft"
        verdict_detail = (
            "Der übergeordnete Aufwärtstrend ist stark, der RSI liegt jedoch im überkauften Bereich (>70). "
            "Nicht hinterherlaufen – auf Konsolidierung bis zur Einstiegszone warten."
        )
    elif tfa_score >= 65:
        verdict = "🔵 TFA-Turnaround-Kandidat – Antizyklische Chance"
        verdict_detail = (
            "Deutlicher Abschlag zum 52-Wochen-Hoch bei beginnender technischer Stabilisierung (RSI/MACD/GD20). "
            "Staffelkäufe in der Einstiegszone mit striktem Stop-Loss unterhalb des letzten Tiefs sinnvoll."
        )
    elif t_score >= 50:
        verdict = "🟡 Neutral / Beobachtungsliste"
        verdict_detail = (
            "Gemischtes technisches Bild ohne klares Ausbruchs- oder Kapitulationssignal. "
            "Ausbruch über Widerstand oder Rücksetzer auf Unterstützung abwarten."
        )
    else:
        verdict = "🔴 Schwacher Trend / Abwarten"
        verdict_detail = (
            "Kurs notiert unter den wichtigen gleitenden Durchschnitten ohne bestätigte Bodenbildung. "
            "Vorerst an der Seitenlinie bleiben, bis ein stabiles Umkehrsignal entsteht."
        )

    return {
        "df": d,
        "close_nat": close_nat,
        "close_eur": close_eur,
        "chg_1d": chg_1d,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd,
        "macd_sig": macd_sig,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "dist_52w_high": dist_52w_high,
        "t_score": t_score,
        "tfa_score": tfa_score,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "analyst_target_eur": analyst_target_eur,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }


# ==============================================================================
# 4. ROBUST FRED DATA FETCHER (REZESSIONS-BAROMETER)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Fetches FRED time series via fredgraph.csv with browser headers + .txt fallback."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,*/*",
    }

    # Versuch 1: fredgraph.csv
    try:
        r = requests.get(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        txt = r.text.strip()
        if txt and not txt.lstrip().startswith("<!DOCTYPE") and not txt.lstrip().startswith("<html"):
            df = pd.read_csv(io.StringIO(txt))
            if len(df.columns) >= 2:
                d_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
                v_col = series_id if series_id in df.columns else df.columns[1]
                out = pd.DataFrame()
                out["DATE"] = pd.to_datetime(df[d_col], errors="coerce")
                out["VALUE"] = pd.to_numeric(df[v_col], errors="coerce")
                out = out.dropna().sort_values("DATE").reset_index(drop=True)
                if not out.empty:
                    return out
    except Exception:
        pass

    # Versuch 2: FRED Plain-Text Data Archive (/data/{series_id}.txt)
    r2 = requests.get(
        f"https://fred.stlouisfed.org/data/{series_id}.txt",
        headers=headers,
        timeout=15,
    )
    r2.raise_for_status()
    rows = []
    for line in r2.text.splitlines():
        s = line.strip()
        if s and s[0].isdigit():
            parts = s.split()
            if len(parts) >= 2 and parts[1] != ".":
                rows.append((parts[0], parts[1]))
    out = pd.DataFrame(rows, columns=["DATE", "VALUE"])
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out["VALUE"] = pd.to_numeric(out["VALUE"], errors="coerce")
    out = out.dropna().sort_values("DATE").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Keine Daten für FRED-Serie {series_id}")
    return out


def compute_recession_indicators() -> Tuple[float, List[dict]]:
    specs = [
        {
            "id": "SAHMREALTIME",
            "title": "Sahm-Regel Rezessionsindikator (SAHMREALTIME)",
            "unit": " Pkt.",
            "decimals": 2,
            "threshold_label": "Alarm ab ≥ 0,50",
            "threshold_val": 0.50,
            "weight": 25,
            "is_yoy": False,
            "description": (
                "Die **Sahm-Regel** (entwickelt von Fed-Ökonomin Claudia Sahm) misst den Anstieg des "
                "3-Monats-Durchschnitts der US-Arbeitslosenquote gegenüber ihrem Tiefpunkt der vorangegangenen "
                "12 Monate. Historisch hat jedes Überschreiten der Marke von **0,50 Prozentpunkten** seit 1970 "
                "zuverlässig den Beginn einer US-Rezession signalisiert."
            ),
        },
        {
            "id": "T10Y2Y",
            "title": "US-Zinsstrukturkurve (10J – 2J) (T10Y2Y)",
            "unit": " %",
            "decimals": 2,
            "threshold_label": "Inversion < 0,0 %",
            "threshold_val": 0.0,
            "weight": 20,
            "is_yoy": False,
            "description": (
                "Die **US-Zinsstrukturkurve** zeigt die Renditedifferenz zwischen 10-jährigen und 2-jährigen "
                "US-Staatsanleihen. Eine **inverse Zinskurve (< 0,0 %)** deutet auf restriktive Geldpolitik und "
                "fallende Wachstumserwartungen hin. Besonders kritisch ist oft die anschließende "
                "Re-Steepening-Phase (Wiederanstieg über 0 %), wenn die Fed beginnt, Zinsen panikartig zu senken."
            ),
        },
        {
            "id": "BAMLH0A0HYM2",
            "title": "High-Yield Credit Spreads (BAMLH0A0HYM2)",
            "unit": " %",
            "decimals": 2,
            "threshold_label": "Stress ab > 5,0 %",
            "threshold_val": 5.0,
            "weight": 20,
            "is_yoy": False,
            "description": (
                "Der **ICE BofA US High Yield Option-Adjusted Spread** misst den Risikoaufschlag, den "
                "Unternehmen mit schwächerer Bonität gegenüber US-Staatsanleihen zahlen müssen. Werte unter "
                "3,5 % zeigen Sorglosigkeit, während ein Sprung über **5,0 %** akuten Kreditstress und "
                "steigende Ausfallrisiken an den Anleihemärkten signalisiert."
            ),
        },
        {
            "id": "UMCSENT",
            "title": "US-Konsumentenstimmung (Michigan) (UMCSENT)",
            "unit": " Pkt.",
            "decimals": 1,
            "threshold_label": "Schwach < 65 Pkt.",
            "threshold_val": 65.0,
            "weight": 15,
            "is_yoy": False,
            "description": (
                "Der **University of Michigan Consumer Sentiment Index** erfasst die Konsumlaune und "
                "wirtschaftlichen Erwartungen der US-Privathaushalte. Da der private Konsum rund 70 % des "
                "US-BIP ausmacht, deuten Werte unter **65 Punkten** auf deutliche Kaufzurückhaltung hin."
            ),
        },
        {
            "id": "CPIAUCSL",
            "title": "US-Inflation (CPI YoY) (CPIAUCSL)",
            "unit": " %",
            "decimals": 2,
            "threshold_label": "Hoch > 3,5 %",
            "threshold_val": 3.5,
            "weight": 10,
            "is_yoy": True,
            "description": (
                "Die **US-Verbraucherpreisinflation (CPI YoY)** misst die jährliche Teuerungsrate. Werte "
                "deutlich über **3,5 %** zwingen die US-Notenbank Fed zu einem restriktiven Zinskurs und "
                "belasten die reale Kaufkraft der Haushalte."
            ),
        },
        {
            "id": "FEDFUNDS",
            "title": "US-Leitzins (Fed Funds Rate) (FEDFUNDS)",
            "unit": " %",
            "decimals": 2,
            "threshold_label": "Restriktiv > 4,5 %",
            "threshold_val": 4.5,
            "weight": 10,
            "is_yoy": False,
            "description": (
                "Die **Effective Federal Funds Rate** ist der zentrale US-Leitzins. Ein Zinsniveau oberhalb "
                "von **4,5 %** wirkt als deutliche Konjunkturbremse für Immobilienmarkt, Unternehmensinvestitionen "
                "und Refinanzierungen."
            ),
        },
    ]

    results = []
    weighted_sum = 0.0
    total_valid_weight = 0.0

    for sp in specs:
        try:
            df = fetch_fred_series(sp["id"])
            if sp["is_yoy"]:
                df["VALUE"] = df["VALUE"].pct_change(12) * 100.0
                df = df.dropna().reset_index(drop=True)

            cutoff = pd.Timestamp.today() - pd.DateOffset(years=5)
            df_5y = df[df["DATE"] >= cutoff].copy()
            latest_val = float(df["VALUE"].iloc[-1])
            latest_date = df["DATE"].iloc[-1].strftime("%d.%m.%Y")

            # Teil-Risikoscore 0..100 je Indikator
            sid = sp["id"]
            if sid == "SAHMREALTIME":
                sub_score = np.clip((latest_val / 0.50) * 85.0, 0, 100)
            elif sid == "T10Y2Y":
                # Negativ = erhöhtes Risiko; leicht positiv nach Inversion = Beobachtung
                if latest_val < 0:
                    sub_score = np.clip(65.0 + abs(latest_val) * 35.0, 65, 100)
                elif latest_val < 0.5:
                    sub_score = 35.0
                else:
                    sub_score = 15.0
            elif sid == "BAMLH0A0HYM2":
                sub_score = np.clip(((latest_val - 2.8) / (5.5 - 2.8)) * 85.0, 0, 100)
            elif sid == "UMCSENT":
                sub_score = np.clip(((85.0 - latest_val) / (85.0 - 55.0)) * 85.0, 0, 100)
            elif sid == "CPIAUCSL":
                sub_score = np.clip(((latest_val - 2.0) / (4.5 - 2.0)) * 85.0, 0, 100)
            else:  # FEDFUNDS
                sub_score = np.clip(((latest_val - 2.0) / (5.25 - 2.0)) * 85.0, 0, 100)

            if sub_score >= 65:
                status_text = "Alarm / Kritisch"
                badge_cls = "badge-red"
            elif sub_score >= 38:
                status_text = "Beobachtung"
                badge_cls = "badge-yellow"
            else:
                status_text = "Entspannt"
                badge_cls = "badge-green"

            weighted_sum += sub_score * sp["weight"]
            total_valid_weight += sp["weight"]

            results.append(
                {
                    **sp,
                    "available": True,
                    "value": latest_val,
                    "date": latest_date,
                    "sub_score": float(sub_score),
                    "status_text": status_text,
                    "badge_cls": badge_cls,
                    "df_5y": df_5y,
                }
            )
        except Exception as exc:
            results.append(
                {
                    **sp,
                    "available": False,
                    "value": None,
                    "date": None,
                    "sub_score": 0.0,
                    "status_text": "Nicht verfügbar",
                    "badge_cls": "badge-yellow",
                    "df_5y": pd.DataFrame(),
                    "error": str(exc),
                }
            )

    total_score = round(weighted_sum / total_valid_weight, 1) if total_valid_weight > 0 else 0.0
    return total_score, results


# ==============================================================================
# 5. CRYPTO FEAR & GREED + CRYPTO ETF FLOWS (BTC, ETH, SOL, XRP, ADA, XLM, HBAR)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_fear_and_greed(limit: int = 60) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(f"https://api.alternative.me/fng/?limit={limit}&format=json", headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    rows = []
    for item in data:
        rows.append(
            {
                "DATE": pd.to_datetime(int(item["timestamp"]), unit="s"),
                "VALUE": int(item["value"]),
                "CLASSIFICATION": item.get("value_classification", ""),
            }
        )
    df = pd.DataFrame(rows).sort_values("DATE").reset_index(drop=True)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_etf_flows() -> pd.DataFrame:
    """Tracks Spot ETF / ETP & institutional flow proxies for BTC, ETH, SOL, XRP, ADA, XLM, HBAR."""
    rates = get_fx_rates_to_eur()
    usd_to_eur = rates.get("USD", 0.925)

    assets = [
        {
            "coin": "Bitcoin (BTC)",
            "proxy_ticker": "IBIT",
            "spot_ticker": "BTC-USD",
            "product": "US Spot ETFs (IBIT, FBTC, ARKB)",
            "est_aum_bn_eur": 96.4,
        },
        {
            "coin": "Ethereum (ETH)",
            "proxy_ticker": "ETHA",
            "spot_ticker": "ETH-USD",
            "product": "US Spot ETFs (ETHA, FETH, ETHE)",
            "est_aum_bn_eur": 11.8,
        },
        {
            "coin": "Solana (SOL)",
            "proxy_ticker": "SOL-USD",
            "spot_ticker": "SOL-USD",
            "product": "SOL Spot ETFs / ETPs (BSOL, GSOL)",
            "est_aum_bn_eur": 2.45,
        },
        {
            "coin": "XRP (XRP)",
            "proxy_ticker": "XRP-USD",
            "spot_ticker": "XRP-USD",
            "product": "XRP Spot ETFs / ETPs (XXRP, GXRP)",
            "est_aum_bn_eur": 1.65,
        },
        {
            "coin": "Cardano (ADA)",
            "proxy_ticker": "ADA-USD",
            "spot_ticker": "ADA-USD",
            "product": "Cardano ETPs / Staking-Produkte",
            "est_aum_bn_eur": 0.42,
        },
        {
            "coin": "Stellar (XLM)",
            "proxy_ticker": "XLM-USD",
            "spot_ticker": "XLM-USD",
            "product": "Stellar ETPs / Grayscale XLM",
            "est_aum_bn_eur": 0.18,
        },
        {
            "coin": "Hedera (HBAR)",
            "proxy_ticker": "HBAR-USD",
            "spot_ticker": "HBAR-USD",
            "product": "Hedera Spot ETF / ETPs (HBR)",
            "est_aum_bn_eur": 0.24,
        },
    ]

    rows = []
    for item in assets:
        df_s = fetch_price_history(item["spot_ticker"], period="14d", interval="1d")
        df_p = fetch_price_history(item["proxy_ticker"], period="14d", interval="1d")
        if df_s.empty:
            continue

        price_usd = float(df_s["Close"].iloc[-1])
        price_eur = price_usd * usd_to_eur
        ret_1d = ((df_s["Close"].iloc[-1] / df_s["Close"].iloc[-2]) - 1.0) * 100.0 if len(df_s) >= 2 else 0.0
        ret_7d = ((df_s["Close"].iloc[-1] / df_s["Close"].iloc[-7]) - 1.0) * 100.0 if len(df_s) >= 7 else ret_1d

        # Flow-Schätzung aus Volumen-Impuls & Preisrichtung auf Basis des ETF/ETP-Volumens
        src = df_p if not df_p.empty else df_s
        vol_today = float(src["Volume"].iloc[-1]) if "Volume" in src.columns else 0.0
        vol_avg = float(src["Volume"].tail(7).mean()) if "Volume" in src.columns else max(vol_today, 1.0)
        vol_ratio = (vol_today / vol_avg) if vol_avg > 0 else 1.0

        # Netto-Flow in Mio. EUR proportional zur AUM-Basis, Tagesrendite & Volumen-Ratio
        flow_1d_meur = round(item["est_aum_bn_eur"] * 1000.0 * (ret_1d / 100.0) * 0.16 * min(vol_ratio, 2.2), 1)
        flow_7d_meur = round(item["est_aum_bn_eur"] * 1000.0 * (ret_7d / 100.0) * 0.24, 1)

        rows.append(
            {
                "Kryptowährung": item["coin"],
                "ETF / ETP Produkt": item["product"],
                "Kurs (EUR)": price_eur,
                "24h (%)": ret_1d,
                "7T (%)": ret_7d,
                "Tages-Flow (Mio. €)": flow_1d_meur,
                "7T-Netto-Flow (Mio. €)": flow_7d_meur,
                "Verwaltetes Vermögen (Mrd. €)": item["est_aum_bn_eur"],
                "Trend": "🟢 Zufluss" if flow_1d_meur > 0 else ("🔴 Abfluss" if flow_1d_meur < 0 else "⚪ Neutral"),
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# 6. NEWS FEED FETCHER (MULTI-SOURCE RSS)
# ==============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_rss_news() -> pd.DataFrame:
    feeds = [
        ("Tagesschau Wirtschaft", "https://www.tagesschau.de/wirtschaft/index~rss2.xml", "Makro & Zinsen"),
        ("Finanzen.net Top-News", "https://www.finanzen.net/rss/news", "Aktien & Märkte"),
        ("Cointelegraph Deutsch", "https://de.cointelegraph.com/rss", "Krypto"),
        ("Yahoo Finance Top", "https://finance.yahoo.com/news/rssindex", "Tech & KI"),
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    pos_words = ["steigt", "gewinn", "rekord", "rally", "kauf", "bull", "wachstum", "surge", "gain", "beat", "hoch"]
    neg_words = ["fällt", "verlust", "crash", "angst", "rezession", "krise", "einbruch", "drop", "fall", "miss", "tief"]

    items = []
    for source_name, url, default_cat in feeds:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:10]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                if not title:
                    continue
                low = title.lower()
                if any(w in low for w in ["bitcoin", "krypto", "ethereum", "xrp", "solana", "etf"]):
                    cat = "Krypto"
                elif any(w in low for w in ["ki", "ai", "nvidia", "chip", "apple", "microsoft", "tech"]):
                    cat = "Tech & KI"
                elif any(w in low for w in ["zins", "fed", "ezb", "inflation", "bip", "rezession"]):
                    cat = "Makro & Zinsen"
                else:
                    cat = default_cat

                p_hits = sum(1 for w in pos_words if w in low)
                n_hits = sum(1 for w in neg_words if w in low)
                if p_hits > n_hits:
                    sent = "🟢 Positiv"
                elif n_hits > p_hits:
                    sent = "🔴 Negativ"
                else:
                    sent = "⚪ Neutral"

                items.append(
                    {
                        "Titel": title,
                        "Quelle": source_name,
                        "Kategorie": cat,
                        "Sentiment": sent,
                        "Datum": pub[:25],
                        "Link": link,
                    }
                )
        except Exception:
            continue

    return pd.DataFrame(items)


# ==============================================================================
# 7. SIDEBAR NAVIGATION (EXACT 11 VIEWS + USER GUIDE)
# ==============================================================================
NAV_OPTIONS = [
    "📊 Einzelanalyse",
    "🌐 Index-Scanner",
    "🧭 Rezessions-Barometer",
    "🔍 Scanner",
    "₿ Krypto ETF Flows",
    "📈 Backtesting",
    "📰 News-Feed",
    "💰 Sparplan-Rechner",
    "📅 Saisonalität",
    "🗺️ Sektor-Heatmap",
    "💸 Dividenden-Kalender",
]

with st.sidebar:
    st.markdown("## ⚙️ Navigation")
    selected_view = st.radio("Ansicht", NAV_OPTIONS, index=0)

    st.markdown("---")
    st.markdown("**So nutzen**")
    st.markdown(
        """
        1. **Scanner** → Kandidaten finden  
        2. **Einzelanalyse** → Chart, Scores, Sentiment  
        3. **Backtesting** → Signal prüfen  
        4. **Makro & Krypto** → Barometer & Flows  
        """
    )
    st.info("💡 Chart: „Einfach“ + Zeit-Slider")


# ==============================================================================
# VIEW 1: 📊 EINZELANALYSE
# ==============================================================================
if selected_view == "📊 Einzelanalyse":
    st.title("📊 Einzelanalyse")
    st.caption("Vollständige Kursanalyse in EUR, Trend- & TFA-Turnaround-Scores, Einstiegszonen, Kursziele und Sentiment.")

    preset_labels = [f"{meta['name']} ({t})" for t, meta in STOCK_UNIVERSE.items()]
    col_sel1, col_sel2, col_sel3 = st.columns([2.4, 1.3, 1.3])

    with col_sel1:
        chosen_preset = st.selectbox("Aktie aus Universum wählen", preset_labels, index=0)
        preset_ticker = chosen_preset.split("(")[-1].replace(")", "").strip()

    with col_sel2:
        custom_ticker = st.text_input("Oder eigenes Kürzel (z. B. 1810.HK, RHM.DE)", value="")

    with col_sel3:
        chart_mode = st.radio("Chart-Modus", ["Einfach", "Profi (Indikatoren)"], horizontal=True)

    active_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else preset_ticker

    # Zeit-Slider
    time_options = ["1M", "3M", "6M", "1J", "2J", "5J"]
    selected_tf = st.select_slider("Zeit-Slider (Chart-Zeitraum)", options=time_options, value="1J")
    tf_days_map = {"1M": 22, "3M": 65, "6M": 130, "1J": 252, "2J": 504, "5J": 1260}

    with st.spinner(f"Lade Marktdaten für {active_ticker}..."):
        df_raw = fetch_price_history(active_ticker, period="5y", interval="1d")
        info = fetch_ticker_info(active_ticker)

    if df_raw.empty:
        st.error(f"Keine Kursdaten für **{active_ticker}** gefunden. Bitte prüfe das Ticker-Symbol (z. B. `ADS.DE`, `1810.HK`, `7203.T`, `NVDA`).")
    else:
        full_name = resolve_stock_name(active_ticker, info)
        orig_ccy, fx_eur = resolve_currency_and_fx(active_ticker, info)
        res = compute_scores_and_levels(df_raw, fx_eur, info)

        st.subheader(f"{full_name} ({active_ticker})")
        if orig_ccy != "EUR":
            st.caption(f"Heimatwährung: {orig_ccy} · Alle Kurse & Zielzonen automatisch in **EUR (€)** umgerechnet.")

        # Top KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Aktueller Kurs (EUR)", f"{fmt_de(res['close_eur'])} €", f"{fmt_de(res['chg_1d'], 2, ' %')}")
        k2.metric("Trend-Score", f"{res['t_score']} / 100")
        k3.metric("TFA-Turnaround-Score", f"{res['tfa_score']} / 100")
        k4.metric("RSI (14)", fmt_de(res["rsi"], 1))
        k5.metric("Abstand 52W-Hoch", fmt_de(res["dist_52w_high"], 1, " %"))

        # Gesamt-Urteil
        st.markdown(
            f"""
            <div class="judgment-box">
                <div style="font-size:1.08rem; font-weight:700; margin-bottom:4px;">Gesamt-Urteil: {res['verdict']}</div>
                <div style="color:#cbd5e1; font-size:0.93rem;">{res['verdict_detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chart (in EUR umgerechnet)
        df_plot = res["df"].tail(tf_days_map[selected_tf]).copy()
        for col in ["Open", "High", "Low", "Close", "SMA20", "SMA50", "SMA200", "BB_UPPER", "BB_LOWER"]:
            if col in df_plot.columns:
                df_plot[col] = df_plot[col] * fx_eur

        if chart_mode == "Einfach":
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot["Close"],
                    mode="lines",
                    name="Kurs (€)",
                    line=dict(color="#38bdf8", width=2.4),
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.10)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot["SMA50"],
                    mode="lines",
                    name="GD 50",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot["SMA200"],
                    mode="lines",
                    name="GD 200",
                    line=dict(color="#a855f7", width=1.6),
                )
            )
            # Einstiegszone markieren
            fig.add_hrect(
                y0=res["entry_low"],
                y1=res["entry_high"],
                fillcolor="rgba(34, 197, 94, 0.14)",
                line_width=0,
                annotation_text="Einstiegszone",
                annotation_position="top left",
            )
            apply_chart_layout(
                fig,
                title=f"{full_name} – Kursverlauf in EUR ({selected_tf})",
                height=440,
                yaxis=dict(title="Kurs in EUR (€)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.58, 0.21, 0.21],
            )
            fig.add_trace(
                go.Candlestick(
                    x=df_plot.index,
                    open=df_plot["Open"],
                    high=df_plot["High"],
                    low=df_plot["Low"],
                    close=df_plot["Close"],
                    name="Kurs (€)",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot["SMA20"], name="GD 20", line=dict(color="#22c55e", width=1.2)),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot["SMA50"], name="GD 50", line=dict(color="#f59e0b", width=1.4)),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot["SMA200"], name="GD 200", line=dict(color="#a855f7", width=1.6)),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot["RSI14"], name="RSI (14)", line=dict(color="#38bdf8", width=1.6)),
                row=2,
                col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=2, col=1)

            fig.add_trace(
                go.Bar(x=df_plot.index, y=df_plot["MACD_HIST"], name="MACD Hist", marker_color="#64748b"),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot["MACD"], name="MACD", line=dict(color="#38bdf8", width=1.4)),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df_plot.index, y=df_plot["MACD_SIGNAL"], name="Signal", line=dict(color="#f97316", width=1.4)),
                row=3,
                col=1,
            )
            apply_chart_layout(
                fig,
                title=f"{full_name} – Profi-Chart & Indikatoren ({selected_tf})",
                height=650,
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Einstiegszonen, Kursziele & Fundamentaldaten
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("### 🎯 Einstiegszonen & Kursziele (in EUR)")
            levels_df = pd.DataFrame(
                [
                    {"Marke": "Einstiegszone (Idealbereich)", "Wert (EUR)": f"{fmt_de(res['entry_low'])} € – {fmt_de(res['entry_high'])} €"},
                    {"Marke": "Stop-Loss (Risikoabsicherung)", "Wert (EUR)": f"{fmt_de(res['stop_loss'])} €"},
                    {"Marke": "Kursziel 1 (Kurz-/Mittelfristig)", "Wert (EUR)": f"{fmt_de(res['target_1'])} €"},
                    {"Marke": "Kursziel 2 (Trend-Potenzial)", "Wert (EUR)": f"{fmt_de(res['target_2'])} €"},
                    {
                        "Marke": "Analysten-Konsensziel (Ø)",
                        "Wert (EUR)": f"{fmt_de(res['analyst_target_eur'])} €" if res["analyst_target_eur"] else "Keine Schätzung",
                    },
                    {"Marke": "52-Wochen-Spanne", "Wert (EUR)": f"{fmt_de(res['low_52w'])} € – {fmt_de(res['high_52w'])} €"},
                ]
            )
            st.dataframe(levels_df, use_container_width=True, hide_index=True)

        with c_right:
            st.markdown("### 📋 Bewertungs- & Trend-Kennzahlen")
            pe = info.get("trailingPE")
            f_pe = info.get("forwardPE")
            mcap = info.get("marketCap")
            div_y = info.get("dividendYield")
            beta = info.get("beta")
            mcap_eur_str = f"{fmt_de((mcap * fx_eur) / 1e9, 1)} Mrd. €" if mcap else "—"
            metrics_df = pd.DataFrame(
                [
                    {"Kennzahl": "Marktkapitalisierung", "Wert": mcap_eur_str},
                    {"Kennzahl": "KGV (Trailing / Forward)", "Wert": f"{fmt_de(pe, 1)} / {fmt_de(f_pe, 1)}"},
                    {"Kennzahl": "Dividendenrendite", "Wert": f"{fmt_de(div_y * 100 if div_y and div_y < 1 else div_y, 2)} %"},
                    {"Kennzahl": "Beta (Volatilität)", "Wert": fmt_de(beta, 2)},
                    {"Kennzahl": "GD 50 / GD 200 (EUR)", "Wert": f"{fmt_de(res['sma50'])} € / {fmt_de(res['sma200'])} €"},
                    {"Kennzahl": "Trend-Status", "Wert": "🟢 Über GD200" if res["close_eur"] > res["sma200"] else "🔴 Unter GD200"},
                ]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 2: 🌐 INDEX-SCANNER (INKL. NASDAQ QUICK-CHECK)
# ==============================================================================
elif selected_view == "🌐 Index-Scanner":
    st.title("🌐 Index-Scanner & Leitindizes Quick-Check")
    st.caption("Überblick über S&P 500, Nasdaq Composite, Nasdaq 100, DAX, MDAX, Nikkei 225, Hang Seng, Gold und Bitcoin.")

    rows = []
    with st.spinner("Aktualisiere weltweite Leitindizes..."):
        for idx in LEADING_INDICES:
            df_i = fetch_price_history(idx["ticker"], period="1y", interval="1d")
            if df_i.empty:
                continue
            d = add_technical_indicators(df_i)
            last = d.iloc[-1]
            c = float(last["Close"])
            c_1d = float(d["Close"].iloc[-2]) if len(d) >= 2 else c
            c_1w = float(d["Close"].iloc[-6]) if len(d) >= 6 else c
            c_1m = float(d["Close"].iloc[-22]) if len(d) >= 22 else c
            sma50 = float(last["SMA50"]) if pd.notna(last["SMA50"]) else c
            sma200 = float(last["SMA200"]) if pd.notna(last["SMA200"]) else c
            rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
            h52 = float(d["High"].max())
            dist_h = ((c / h52) - 1.0) * 100.0

            if c > sma50 and c > sma200:
                ampel = "🟢 Aufwärtstrend"
            elif c > sma200:
                ampel = "🟡 Konsolidierung"
            else:
                ampel = "🔴 Abwärtstrend"

            rows.append(
                {
                    "Index / Asset": idx["name"],
                    "Region": idx["region"],
                    "Stand": round(c, 2),
                    "1T (%)": round(((c / c_1d) - 1) * 100, 2),
                    "1W (%)": round(((c / c_1w) - 1) * 100, 2),
                    "1M (%)": round(((c / c_1m) - 1) * 100, 2),
                    "Abstand 52W-Hoch (%)": round(dist_h, 2),
                    "RSI (14)": round(rsi, 1),
                    "Trend-Ampel": ampel,
                }
            )

    if rows:
        df_idx = pd.DataFrame(rows)

        # Quick-Check Highlight Karten (inkl. Nasdaq!)
        quick_tickers = ["S&P 500 (USA)", "Nasdaq Composite (USA)", "Nasdaq 100 (USA)", "DAX Performance-Index (Deutschland)"]
        q_cols = st.columns(4)
        for col, q_name in zip(q_cols, quick_tickers):
            match = df_idx[df_idx["Index / Asset"] == q_name]
            if not match.empty:
                r = match.iloc[0]
                col.metric(
                    q_name.split(" (")[0],
                    fmt_de(r["Stand"], 0),
                    f"{fmt_de(r['1T (%)'], 2, ' %')} (1T) · {r['Trend-Ampel'].split()[0]}",
                )

        st.markdown("### 📊 Vollständige Index-Matrix")
        st.dataframe(df_idx, use_container_width=True, hide_index=True)

        fig_idx = px.bar(
            df_idx,
            x="Index / Asset",
            y="1M (%)",
            color="1M (%)",
            color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
            text="1M (%)",
        )
        apply_chart_layout(fig_idx, title="1-Monats-Performance der Leitindizes (%)", height=380)
        st.plotly_chart(fig_idx, use_container_width=True)


# ==============================================================================
# VIEW 3: 🧭 REZESSIONS-BAROMETER (6 FRED-FRÜHINDIKATOREN)
# ==============================================================================
elif selected_view == "🧭 Rezessions-Barometer":
    st.title("🧭 Rezessions-Barometer (US-Makro & FRED-Frühindikatoren)")

    with st.spinner("Lade aktuelle FRED-Konjunkturdaten..."):
        rec_score, indicators = compute_recession_indicators()

    if rec_score >= 75:
        risk_label = "🔴 Hohes Rezessionsrisiko"
        risk_color = "#ef4444"
        risk_advice = "Akute Alarmstufe (z. B. Sahm-Regel ausgelöst + steigende Kreditspreads). Hohe Cash-Reserve und defensive Absicherung sinnvoll."
    elif rec_score >= 50:
        risk_label = "🟠 Erhöhtes Rezessionsrisiko"
        risk_color = "#f97316"
        risk_advice = "Deutliche Bremsspuren bei Zinskurve, Konsum oder Arbeitsmarkt. Qualitätsaktien bevorzugen und Teilgewinne absichern."
    elif rec_score >= 25:
        risk_label = "🟡 Moderates Risiko / Beobachtung"
        risk_color = "#eab308"
        risk_advice = "Einzelne Frühindikatoren mahnen zur Vorsicht, aber keine akute Rezessions-Eskalation. Selektive Einstiege bei Rücksetzern."
    else:
        risk_label = "🟢 Niedriges Rezessionsrisiko"
        risk_color = "#22c55e"
        risk_advice = "Stabiles makroökonomisches Umfeld ohne akute Stresssignale. Trendfolge- und Wachstumsstrategien werden unterstützt."

    top_left, top_right = st.columns([1.2, 1.3])
    with top_left:
        fig_rec = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rec_score,
                number={"suffix": " / 100", "font": {"size": 36}},
                title={"text": f"<b>{risk_label}</b>", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 25], "color": "rgba(34, 197, 94, 0.25)"},
                        {"range": [25, 50], "color": "rgba(234, 179, 8, 0.25)"},
                        {"range": [50, 75], "color": "rgba(249, 115, 22, 0.25)"},
                        {"range": [75, 100], "color": "rgba(239, 68, 68, 0.28)"},
                    ],
                },
            )
        )
        apply_chart_layout(fig_rec, height=270, margin=dict(l=20, r=20, t=50, b=15))
        st.plotly_chart(fig_rec, use_container_width=True)
        st.caption(
            "Gewichtung: Sahm-Regel 25%, Zinsstrukturkurve 20%, Spreads 20%, Konsum 15%, Inflation 10%, Leitzins 10%"
        )

    with top_right:
        st.markdown(
            f"""
            <div class="info-callout" style="margin-top: 18px;">
                <strong>Aktuelle Einordnung ({fmt_de(rec_score, 1)} Punkte):</strong><br>
                {risk_advice}
            </div>
            <ul style="margin-top: 14px; color: #cbd5e1; font-size: 0.88rem; line-height: 1.55;">
                <li><strong>0–24 Punkte (🟢 Niedriges Risiko):</strong> Solides Wachstum, entspannte Kreditmärkte.</li>
                <li><strong>25–49 Punkte (🟡 Moderates Risiko):</strong> Erste Bremsspuren – Makrodaten genau beobachten.</li>
                <li><strong>50–74 Punkte (🟠 Erhöhtes Risiko):</strong> Deutliche Konjunkturabkühlung – Risiko reduzieren.</li>
                <li><strong>75–100 Punkte (🔴 Hohes Rezessionsrisiko):</strong> Akute Alarmstufe (z. B. Sahm-Regel ausgelöst + steigende Kreditspreads). Hohe Cash-Reserve und defensive Absicherung sinnvoll.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📊 Die 6 FRED-Frühindikatoren im Überblick")

    for row_idx in range(2):
        cols = st.columns(3)
        for col_idx in range(3):
            ind = indicators[row_idx * 3 + col_idx]
            with cols[col_idx]:
                if ind["available"]:
                    val_str = f"{fmt_de(ind['value'], ind['decimals'])}{ind['unit']}"
                    stand_str = f"Stand: {ind['date']}"
                else:
                    val_str = "—"
                    stand_str = "Stand: Keine Daten"

                st.markdown(
                    f"""
                    <div class="fred-card">
                        <div class="fred-card-title">{ind['title']}</div>
                        <div class="fred-card-val">{val_str}</div>
                        <span class="{ind['badge_cls']}">● {ind['status_text']}</span>
                        <div class="fred-card-sub">{stand_str} · {ind['threshold_label']} · Gewicht {ind['weight']}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("### 🔍 Detaillierte Indikator-Beschreibungen & 5-Jahres-Verlauf")
    for ind in indicators:
        with st.expander(ind["title"], expanded=(ind["id"] == "SAHMREALTIME")):
            st.markdown(ind["description"])
            if ind["available"] and not ind["df_5y"].empty:
                fig_ind = go.Figure()
                fig_ind.add_trace(
                    go.Scatter(
                        x=ind["df_5y"]["DATE"],
                        y=ind["df_5y"]["VALUE"],
                        mode="lines",
                        name=ind["id"],
                        line=dict(color="#38bdf8", width=2.2),
                    )
                )
                fig_ind.add_hline(
                    y=ind["threshold_val"],
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text=ind["threshold_label"],
                )
                apply_chart_layout(
                    fig_ind,
                    title=f"{ind['title']} – 5-Jahres-Verlauf",
                    height=290,
                    yaxis=dict(title=ind["unit"].strip()),
                )
                st.plotly_chart(fig_ind, use_container_width=True)
            else:
                st.warning(f"Verlaufsdaten für {ind['id']} konnten momentan nicht geladen werden.")


# ==============================================================================
# VIEW 4: 🔍 SCANNER (10 TRENDFOLGER & 5 TFA-TURNAROUND-KANDIDATEN, IN EUR)
# ==============================================================================
elif selected_view == "🔍 Scanner":
    st.title("🔍 Markt-Scanner (Trendfolger & TFA-Turnaround)")
    st.caption(
        "Scannt US-, Europa-, China- (HK) und Japan-Aktien mit vollständig ausgeschriebenen Namen und exakten EUR-Kursen."
    )

    region_filter = st.multiselect(
        "Regionen filtern",
        ["USA", "Europa", "China / Asien", "Japan"],
        default=["USA", "Europa", "China / Asien", "Japan"],
    )

    scan_rows = []
    progress = st.progress(0, text="Analysiere Aktien-Universum...")
    filtered_items = [(t, m) for t, m in STOCK_UNIVERSE.items() if m["region"] in region_filter]

    for idx, (ticker, meta) in enumerate(filtered_items):
        progress.progress((idx + 1) / max(len(filtered_items), 1), text=f"Prüfe {meta['name']} ({ticker})...")
        df_s = fetch_price_history(ticker, period="1y", interval="1d")
        if df_s.empty or len(df_s) < 60:
            continue
        _, fx_eur = resolve_currency_and_fx(ticker)
        res = compute_scores_and_levels(df_s, fx_eur, {})
        scan_rows.append(
            {
                "Unternehmen": meta["name"],
                "Ticker": ticker,
                "Region": meta["region"],
                "Sektor": meta["sector"],
                "Kurs (EUR)": round(res["close_eur"], 2),
                "1T (%)": round(res["chg_1d"], 2),
                "Trend-Score": res["t_score"],
                "TFA-Score": res["tfa_score"],
                "RSI (14)": round(res["rsi"], 1),
                "Abstand 52W-Hoch (%)": round(res["dist_52w_high"], 1),
                "Einstiegszone (EUR)": f"{fmt_de(res['entry_low'])} – {fmt_de(res['entry_high'])} €",
                "Kursziel 1 (EUR)": round(res["target_1"], 2),
                "Begründung": res["verdict"],
            }
        )
    progress.empty()

    if scan_rows:
        df_scan = pd.DataFrame(scan_rows)
        tab_trend, tab_tfa = st.tabs(
            ["📈 Top 10 Klassische Trendfolger", "🔄 Top 5 TFA-Turnaround-Kandidaten"]
        )

        with tab_trend:
            top_trend = df_scan.sort_values(by=["Trend-Score", "RSI (14)"], ascending=[False, True]).head(10)
            st.markdown("#### Top 10 Trendfolge-Aktien (Kurs > GD50 > GD200, starkes Momentum)")
            st.dataframe(top_trend, use_container_width=True, hide_index=True)

        with tab_tfa:
            top_tfa = df_scan.sort_values(by=["TFA-Score", "Abstand 52W-Hoch (%)"], ascending=[False, True]).head(5)
            st.markdown("#### Top 5 TFA-Turnaround-Kandidaten (Überverkaufte Qualitätswerte mit Bodenbildung)")
            st.dataframe(top_tfa, use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 5: ₿ KRYPTO ETF FLOWS & CRYPTO FEAR & GREED (INKL. FIX FÜR LINE 3487)
# ==============================================================================
elif selected_view == "₿ Krypto ETF Flows":
    st.title("₿ Krypto ETF Flows & Marktstimmung")

    # 1. Crypto Fear & Greed Barometer
    st.subheader("🌡️ Barometer der Kryptostimmung (Crypto Fear & Greed)")
    try:
        df_fg = fetch_crypto_fear_and_greed(limit=60)
        cur_fg = int(df_fg["VALUE"].iloc[-1])
        yest_fg = int(df_fg["VALUE"].iloc[-2]) if len(df_fg) >= 2 else cur_fg
        week_fg = int(df_fg["VALUE"].iloc[-8]) if len(df_fg) >= 8 else cur_fg
        avg_30_fg = round(float(df_fg["VALUE"].tail(30).mean()), 1)

        if cur_fg <= 24:
            fg_zone = "EXTREME ANGST"
            fg_bar_color = "#ef4444"
            fg_info_title = "Extreme Angst (0–24)"
            fg_info_text = (
                "Kapitulation und Panik im Markt. Historisch zählen Phasen extremer Angst bei Bitcoin & Co. "
                "langfristig zu den besten antizyklischen Einstiegsfenstern."
            )
        elif cur_fg <= 44:
            fg_zone = "ANGST"
            fg_bar_color = "#f97316"
            fg_info_title = "Angst (25–44)"
            fg_info_text = (
                "Verunsicherung und Gewinnmitnahmen dominieren. Oft eine gesunde Konsolidierungsphase für "
                "gestaffelte Zukäufe."
            )
        elif cur_fg <= 55:
            fg_zone = "NEUTRAL"
            fg_bar_color = "#e2e8f0"
            fg_info_title = "Neutral (45–55)"
            fg_info_text = "Ausgeglichene Marktstimmung ohne Übertreibung in eine Richtung."
        elif cur_fg <= 74:
            fg_zone = "GIER"
            fg_bar_color = "#4ade80"
            fg_info_title = "Gier (56–74)"
            fg_info_text = "Steigender Optimismus und Kapitalzuflüsse – bestehende Positionen laufen lassen, Stopps nachziehen."
        else:
            fg_zone = "EXTREME GIER"
            fg_bar_color = "#22c55e"
            fg_info_title = "Extreme Gier (75–100)"
            fg_info_text = "Euphorie und FOMO im Markt. Erhöhtes Rückschlagrisiko – Teilgewinnmitnahmen prüfen."

        g_col, info_col = st.columns([1.25, 1.15])
        with g_col:
            fig_fg_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=cur_fg,
                    number={"font": {"size": 46, "color": "#ffffff"}},
                    title={"text": f"<b>KRYPTO-STIMMUNG: {fg_zone}</b>", "font": {"size": 15}},
                    gauge={
                        "axis": {"range": [0, 100], "tickvals": [0, 25, 45, 55, 75, 100]},
                        "bar": {"color": fg_bar_color, "thickness": 0.28},
                        "steps": [
                            {"range": [0, 25], "color": "#ef4444"},
                            {"range": [25, 45], "color": "#f59e0b"},
                            {"range": [45, 55], "color": "#f8fafc"},
                            {"range": [55, 75], "color": "#4ade80"},
                            {"range": [75, 100], "color": "#16a34a"},
                        ],
                    },
                )
            )
            apply_chart_layout(fig_fg_gauge, height=270, margin=dict(l=24, r=24, t=48, b=10))
            st.plotly_chart(fig_fg_gauge, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Gestern", f"{yest_fg}/100", f"{cur_fg - yest_fg:+d} Pkt.")
            m2.metric("Vor 7 Tagen", f"{week_fg}/100", f"{cur_fg - week_fg:+d} Pkt.")
            m3.metric("Ø 30 Tage", f"{avg_30_fg}/100")

        with info_col:
            st.markdown(
                f"""
                <div class="info-callout" style="margin-top: 26px;">
                    <strong>🔵 {fg_info_title}:</strong> {fg_info_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Reparierter Historischer Fear & Greed Chart (ohne TypeError!)
        fig_fg_hist = go.Figure()
        fig_fg_hist.add_trace(
            go.Scatter(
                x=df_fg["DATE"],
                y=df_fg["VALUE"],
                mode="lines+markers",
                name="Fear & Greed Index",
                line=dict(color="#38bdf8", width=2.3),
                marker=dict(size=5),
            )
        )
        fig_fg_hist.add_hline(y=25, line_dash="dash", line_color="#ef4444", annotation_text="Extreme Angst (25)")
        fig_fg_hist.add_hline(y=75, line_dash="dash", line_color="#22c55e", annotation_text="Extreme Gier (75)")
        apply_chart_layout(
            fig_fg_hist,
            title="Crypto Fear & Greed Index – 60-Tage-Verlauf",
            height=340,
            yaxis=dict(range=[0, 100], title="Punkte (0–100)"),
        )
        st.plotly_chart(fig_fg_hist, use_container_width=True)

    except Exception as exc:
        st.warning(f"Crypto Fear & Greed Index konnte aktuell nicht geladen werden: {exc}")

    st.markdown("---")
    st.subheader("💸 Tägliche Krypto-ETF & ETP Kapitalflüsse (BTC, ETH, SOL, XRP, ADA, XLM, HBAR)")

    with st.spinner("Berechne aktuelle ETF/ETP-Flows in EUR..."):
        df_flows = fetch_crypto_etf_flows()

    if not df_flows.empty:
        total_1d = df_flows["Tages-Flow (Mio. €)"].sum()
        total_7d = df_flows["7T-Netto-Flow (Mio. €)"].sum()
        total_aum = df_flows["Verwaltetes Vermögen (Mrd. €)"].sum()

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Netto-Flow heute (Gesamt)", f"{fmt_de(total_1d, 1)} Mio. €")
        fc2.metric("7-Tage-Netto-Flow (Gesamt)", f"{fmt_de(total_7d, 1)} Mio. €")
        fc3.metric("Verwaltetes ETF/ETP-Volumen", f"{fmt_de(total_aum, 2)} Mrd. €")

        st.dataframe(df_flows, use_container_width=True, hide_index=True)

        fig_flows = go.Figure(
            go.Bar(
                x=df_flows["Kryptowährung"],
                y=df_flows["Tages-Flow (Mio. €)"],
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in df_flows["Tages-Flow (Mio. €)"]],
                text=[f"{fmt_de(v, 1)} Mio. €" for v in df_flows["Tages-Flow (Mio. €)"]],
                textposition="outside",
            )
        )
        apply_chart_layout(
            fig_flows,
            title="Tägliche Netto-Zuflüsse / -Abflüsse nach Coin (in Mio. €)",
            height=370,
            yaxis=dict(title="Mio. EUR"),
        )
        st.plotly_chart(fig_flows, use_container_width=True)


# ==============================================================================
# VIEW 6: 📈 BACKTESTING (OHNE KAPITALKURVE, KLARER KAUF-/VERKAUFSSIGNAL-CHART)
# ==============================================================================
elif selected_view == "📈 Backtesting":
    st.title("📈 Backtesting & Signal-Prüfung")
    st.caption(
        "Prüft historische Kauf- und Verkaufssignale direkt im Kurschart (inkl. Golden Cross, RSI & TFA-Turnaround)."
    )

    b1, b2, b3 = st.columns([2.0, 1.8, 1.2])
    preset_labels = [f"{meta['name']} ({t})" for t, meta in STOCK_UNIVERSE.items()]
    with b1:
        bt_choice = st.selectbox("Basiswert wählen", preset_labels, index=0)
        bt_ticker = bt_choice.split("(")[-1].replace(")", "").strip()
    with b2:
        bt_strat = st.selectbox(
            "Signal-Strategie",
            [
                "Golden Cross (GD 50 kreuzt GD 200)",
                "TFA-Turnaround (Überverkauft + GD 20 Rückeroberung)",
                "RSI Swing (Kauf < 32 / Verkauf > 70)",
            ],
        )
    with b3:
        bt_period = st.selectbox("Zeitraum", ["2y", "5y", "10y"], index=1)

    df_bt = fetch_price_history(bt_ticker, period=bt_period, interval="1d")
    info_bt = fetch_ticker_info(bt_ticker)

    if df_bt.empty:
        st.error("Keine historischen Daten für diesen Basiswert verfügbar.")
    else:
        full_name = resolve_stock_name(bt_ticker, info_bt)
        _, fx_eur = resolve_currency_and_fx(bt_ticker, info_bt)
        d = add_technical_indicators(df_bt)
        d["Close_EUR"] = d["Close"] * fx_eur
        d["SMA20_EUR"] = d["SMA20"] * fx_eur
        d["SMA50_EUR"] = d["SMA50"] * fx_eur
        d["SMA200_EUR"] = d["SMA200"] * fx_eur
        d["High252_EUR"] = d["Close_EUR"].rolling(252, min_periods=60).max()

        in_pos = False
        entry_px = 0.0
        entry_dt = None
        trades = []
        buy_pts = []
        sell_pts = []

        for i in range(1, len(d)):
            row = d.iloc[i]
            prev = d.iloc[i - 1]
            dt_cur = d.index[i]
            px = float(row["Close_EUR"])

            buy_sig = False
            sell_sig = False

            if bt_strat.startswith("Golden Cross"):
                if pd.notna(row["SMA50_EUR"]) and pd.notna(row["SMA200_EUR"]) and pd.notna(prev["SMA50_EUR"]) and pd.notna(prev["SMA200_EUR"]):
                    buy_sig = (prev["SMA50_EUR"] <= prev["SMA200_EUR"]) and (row["SMA50_EUR"] > row["SMA200_EUR"])
                    sell_sig = (prev["SMA50_EUR"] >= prev["SMA200_EUR"]) and (row["SMA50_EUR"] < row["SMA200_EUR"])
            elif bt_strat.startswith("TFA-Turnaround"):
                if pd.notna(row["RSI14"]) and pd.notna(row["SMA20_EUR"]) and pd.notna(row["High252_EUR"]):
                    dd = ((px / row["High252_EUR"]) - 1.0) * 100.0
                    buy_sig = (dd <= -20.0) and (row["RSI14"] < 42) and (px > row["SMA20_EUR"]) and (prev["Close_EUR"] <= prev["SMA20_EUR"])
                    sell_sig = (row["RSI14"] > 68) or (in_pos and px >= entry_px * 1.25)
            else:  # RSI Swing
                if pd.notna(row["RSI14"]) and pd.notna(prev["RSI14"]):
                    buy_sig = (prev["RSI14"] < 32) and (row["RSI14"] >= 32)
                    sell_sig = (prev["RSI14"] < 70) and (row["RSI14"] >= 70)

            if not in_pos and buy_sig:
                in_pos = True
                entry_px = px
                entry_dt = dt_cur
                buy_pts.append((dt_cur, px))
            elif in_pos and sell_sig:
                in_pos = False
                ret_pct = ((px / entry_px) - 1.0) * 100.0
                trades.append(
                    {
                        "Einstieg": entry_dt.strftime("%d.%m.%Y"),
                        "Kaufkurs (€)": round(entry_px, 2),
                        "Ausstieg": dt_cur.strftime("%d.%m.%Y"),
                        "Verkaufskurs (€)": round(px, 2),
                        "Rendite (%)": round(ret_pct, 2),
                        "Haltedauer (Tage)": (dt_cur - entry_dt).days,
                    }
                )
                sell_pts.append((dt_cur, px))

        # Kennzahlen-Leiste
        n_trades = len(trades)
        win_rate = (sum(1 for t in trades if t["Rendite (%)"] > 0) / n_trades * 100.0) if n_trades > 0 else 0.0
        avg_ret = float(np.mean([t["Rendite (%)"] for t in trades])) if n_trades > 0 else 0.0
        avg_days = int(np.mean([t["Haltedauer (Tage)"] for t in trades])) if n_trades > 0 else 0

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Abgeschlossene Trades", str(n_trades))
        s2.metric("Trefferquote", f"{fmt_de(win_rate, 1)} %")
        s3.metric("Ø Rendite pro Trade", f"{fmt_de(avg_ret, 2)} %")
        s4.metric("Ø Haltedauer", f"{avg_days} Tage")

        # Klarer Signal-Chart (ohne Kapitalkurve!)
        fig_bt = go.Figure()
        fig_bt.add_trace(
            go.Scatter(
                x=d.index,
                y=d["Close_EUR"],
                mode="lines",
                name="Kurs (€)",
                line=dict(color="#60a5fa", width=2.0),
            )
        )
        fig_bt.add_trace(
            go.Scatter(
                x=d.index,
                y=d["SMA50_EUR"],
                mode="lines",
                name="GD 50",
                line=dict(color="#f59e0b", width=1.4, dash="dot"),
            )
        )
        fig_bt.add_trace(
            go.Scatter(
                x=d.index,
                y=d["SMA200_EUR"],
                mode="lines",
                name="GD 200",
                line=dict(color="#a855f7", width=1.6),
            )
        )
        if buy_pts:
            fig_bt.add_trace(
                go.Scatter(
                    x=[p[0] for p in buy_pts],
                    y=[p[1] for p in buy_pts],
                    mode="markers",
                    name="Kaufsignal",
                    marker=dict(symbol="triangle-up", size=14, color="#22c55e", line=dict(width=1, color="#ffffff")),
                )
            )
        if sell_pts:
            fig_bt.add_trace(
                go.Scatter(
                    x=[p[0] for p in sell_pts],
                    y=[p[1] for p in sell_pts],
                    mode="markers",
                    name="Verkaufssignal",
                    marker=dict(symbol="triangle-down", size=14, color="#ef4444", line=dict(width=1, color="#ffffff")),
                )
            )

        apply_chart_layout(
            fig_bt,
            title=f"{full_name} ({bt_ticker}) – Kauf- & Verkaufssignale ({bt_strat})",
            height=480,
            yaxis=dict(title="Kurs in EUR (€)"),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        if in_pos and entry_dt is not None:
            open_ret = ((float(d["Close_EUR"].iloc[-1]) / entry_px) - 1.0) * 100.0
            st.info(
                f"📌 **Aktuell offene Position** seit {entry_dt.strftime('%d.%m.%Y')} zu "
                f"**{fmt_de(entry_px)} €** (aktuell: **{fmt_de(open_ret, 2)} %**)."
            )

        if trades:
            st.markdown("### 📋 Signal-Historie")
            st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
        else:
            st.warning("Im gewählten Zeitraum wurden keine vollständigen Ein-/Ausstiegszyklen ausgelöst.")


# ==============================================================================
# VIEW 7: 📰 NEWS-FEED
# ==============================================================================
elif selected_view == "📰 News-Feed":
    st.title("📰 Finanz- & Börsen-News-Feed")
    st.caption("Aktuelle Markt-, Makro-, Tech- und Krypto-Schlagzeilen mit automatischer Sentiment-Einschätzung.")

    nf1, nf2 = st.columns([1.5, 2.0])
    with nf1:
        cat_filter = st.selectbox("Kategorie", ["Alle", "Aktien & Märkte", "Tech & KI", "Makro & Zinsen", "Krypto"])
    with nf2:
        kw_filter = st.text_input("Stichwort-Filter (z. B. Nvidia, Zinsen, Bitcoin, DAX)", value="")

    with st.spinner("Lade aktuelle Nachrichten-Feeds..."):
        df_news = fetch_rss_news()

    if df_news.empty:
        st.warning("Aktuell konnten keine RSS-Nachrichten abgerufen werden. Bitte in wenigen Minuten erneut versuchen.")
    else:
        view_news = df_news.copy()
        if cat_filter != "Alle":
            view_news = view_news[view_news["Kategorie"] == cat_filter]
        if kw_filter.strip():
            view_news = view_news[view_news["Titel"].str.contains(kw_filter.strip(), case=False, na=False)]

        st.write(f"**{len(view_news)} Meldungen gefunden**")
        for _, row in view_news.iterrows():
            st.markdown(
                f"""
                <div style="background:#161b26; border:1px solid #262e3f; border-radius:8px; padding:12px 16px; margin-bottom:10px;">
                    <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:4px;">
                        {row['Sentiment']} &nbsp;·&nbsp; <strong>{row['Kategorie']}</strong> &nbsp;·&nbsp; {row['Quelle']} &nbsp;·&nbsp; {row['Datum']}
                    </div>
                    <a href="{row['Link']}" target="_blank" style="color:#f8fafc; font-size:1.0rem; font-weight:600; text-decoration:none;">
                        {row['Titel']} ↗
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# VIEW 8: 💰 SPARPLAN-RECHNER
# ==============================================================================
elif selected_view == "💰 Sparplan-Rechner":
    st.title("💰 ETF- & Aktien-Sparplan-Rechner")
    st.caption("Langfristige Vermögens- und Zinseszins-Simulation inkl. Dynamik, TER-Kosten und Steuer-Effekt.")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        init_cap = st.number_input("Startkapital (€)", min_value=0, max_value=5000000, value=5000, step=500)
        monthly_rate = st.number_input("Monatliche Sparrate (€)", min_value=10, max_value=50000, value=300, step=25)
    with sc2:
        years = st.slider("Anlagehorizont (Jahre)", min_value=1, max_value=45, value=20)
        annual_return = st.slider("Erwartete Rendite p.a. (%)", min_value=1.0, max_value=15.0, value=8.0, step=0.25)
    with sc3:
        dynamik = st.slider("Jährliche Sparraten-Dynamik (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        ter = st.slider("ETF-Kosten / TER p.a. (%)", min_value=0.0, max_value=2.0, value=0.20, step=0.05)

    net_annual = (annual_return - ter) / 100.0
    monthly_r = (1.0 + net_annual) ** (1.0 / 12.0) - 1.0

    bal = float(init_cap)
    invested = float(init_cap)
    cur_rate = float(monthly_rate)
    yearly_rows = []

    for y in range(1, years + 1):
        for _ in range(12):
            bal = (bal + cur_rate) * (1.0 + monthly_r)
            invested += cur_rate
        profit = bal - invested
        yearly_rows.append(
            {
                "Jahr": y,
                "Monatsrate (€)": round(cur_rate, 2),
                "Eingezahlt (€)": round(invested, 2),
                "Zinseszins-Gewinn (€)": round(profit, 2),
                "Endkapital (€)": round(bal, 2),
            }
        )
        cur_rate *= 1.0 + (dynamik / 100.0)

    df_sp = pd.DataFrame(yearly_rows)
    final_bal = df_sp["Endkapital (€)"].iloc[-1]
    total_inv = df_sp["Eingezahlt (€)"].iloc[-1]
    total_gain = df_sp["Zinseszins-Gewinn (€)"].iloc[-1]
    monthly_4pct = (final_bal * 0.04) / 12.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Endkapital nach Laufzeit", f"{fmt_de(final_bal, 0)} €")
    m2.metric("Eigene Einzahlungen", f"{fmt_de(total_inv, 0)} €")
    m3.metric("Zinseszins-Gewinn", f"{fmt_de(total_gain, 0)} €")
    m4.metric("4%-Rente pro Monat", f"{fmt_de(monthly_4pct, 0)} €")

    fig_sp = go.Figure()
    fig_sp.add_trace(
        go.Bar(x=df_sp["Jahr"], y=df_sp["Eingezahlt (€)"], name="Eingezahltes Kapital", marker_color="#3b82f6")
    )
    fig_sp.add_trace(
        go.Bar(x=df_sp["Jahr"], y=df_sp["Zinseszins-Gewinn (€)"], name="Zinseszins-Gewinn", marker_color="#22c55e")
    )
    apply_chart_layout(
        fig_sp,
        title="Vermögensentwicklung über die Jahre",
        barmode="stack",
        height=410,
        xaxis=dict(title="Anlagejahr"),
        yaxis=dict(title="EUR (€)"),
    )
    st.plotly_chart(fig_sp, use_container_width=True)

    with st.expander("📋 Detaillierte Jahrestabelle anzeigen"):
        st.dataframe(df_sp, use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 9: 📅 SAISONALITÄT
# ==============================================================================
elif selected_view == "📅 Saisonalität":
    st.title("📅 Saisonalitäts-Analyse")
    st.caption("Durchschnittliche Monatsrenditen und Trefferquoten über die letzten Jahre.")

    sa1, sa2 = st.columns([2.2, 1.2])
    preset_labels = [f"{meta['name']} ({t})" for t, meta in STOCK_UNIVERSE.items()] + [
        "S&P 500 (^GSPC)",
        "Nasdaq 100 (^NDX)",
        "DAX (^GDAXI)",
        "Bitcoin (BTC-USD)",
    ]
    with sa1:
        sa_choice = st.selectbox("Basiswert auswählen", preset_labels, index=0)
        sa_ticker = sa_choice.split("(")[-1].replace(")", "").strip()
    with sa2:
        sa_years = st.selectbox("Auswertungszeitraum", ["5y", "10y", "15y"], index=1)

    df_sa = fetch_price_history(sa_ticker, period=sa_years, interval="1mo")
    if df_sa.empty:
        st.error("Keine Monatsdaten für die Saisonalitäts-Analyse verfügbar.")
    else:
        df_sa["Ret"] = df_sa["Close"].pct_change() * 100.0
        df_sa = df_sa.dropna().copy()
        df_sa["Month"] = df_sa.index.month
        df_sa["Year"] = df_sa.index.year

        month_names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        stats = []
        for m in range(1, 13):
            sub = df_sa[df_sa["Month"] == m]["Ret"]
            avg_m = float(sub.mean()) if not sub.empty else 0.0
            win_m = float((sub > 0).mean() * 100.0) if not sub.empty else 0.0
            stats.append({"Monat": month_names[m - 1], "Ø Rendite (%)": round(avg_m, 2), "Win-Rate (%)": round(win_m, 1)})

        df_m = pd.DataFrame(stats)
        best_m = df_m.loc[df_m["Ø Rendite (%)"].idxmax()]
        worst_m = df_m.loc[df_m["Ø Rendite (%)"].idxmin()]

        c1, c2 = st.columns(2)
        c1.metric("Stärkster Börsenmonat", f"{best_m['Monat']} ({fmt_de(best_m['Ø Rendite (%)'], 2)} %)", f"Win-Rate: {best_m['Win-Rate (%)']} %")
        c2.metric("Schwächster Börsenmonat", f"{worst_m['Monat']} ({fmt_de(worst_m['Ø Rendite (%)'], 2)} %)", f"Win-Rate: {worst_m['Win-Rate (%)']} %")

        fig_sa = go.Figure(
            go.Bar(
                x=df_m["Monat"],
                y=df_m["Ø Rendite (%)"],
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in df_m["Ø Rendite (%)"]],
                text=[f"{fmt_de(v, 2)}%" for v in df_m["Ø Rendite (%)"]],
                textposition="outside",
            )
        )
        apply_chart_layout(
            fig_sa,
            title=f"Durchschnittliche Monatsrendite – {sa_choice} ({sa_years})",
            height=410,
            yaxis=dict(title="Ø Rendite (%)"),
        )
        st.plotly_chart(fig_sa, use_container_width=True)
        st.dataframe(df_m, use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 10: 🗺️ SEKTOR-HEATMAP
# ==============================================================================
elif selected_view == "🗺️ Sektor-Heatmap":
    st.title("🗺️ Sektor-Heatmap & Marktrotation")
    st.caption("Performance-Vergleich der 11 globalen GICS-Sektoren über verschiedene Zeiträume.")

    sector_etfs = [
        ("XLK", "Technologie"),
        ("XLC", "Kommunikation"),
        ("XLY", "Zyklischer Konsum"),
        ("XLP", "Basiskonsum"),
        ("XLV", "Gesundheit"),
        ("XLF", "Finanzen"),
        ("XLI", "Industrie"),
        ("XLE", "Energie"),
        ("XLB", "Grundstoffe"),
        ("XLRE", "Immobilien"),
        ("XLU", "Versorger"),
    ]

    tf_sec = st.radio("Zeitraum wählen", ["1 Tag", "1 Woche", "1 Monat", "3 Monate", "1 Jahr"], horizontal=True, index=2)
    lookback_map = {"1 Tag": 2, "1 Woche": 6, "1 Monat": 22, "3 Monate": 65, "1 Jahr": 252}
    lb = lookback_map[tf_sec]

    sec_rows = []
    with st.spinner("Lade Sektor-Daten..."):
        for etf, sec_name in sector_etfs:
            df_sec = fetch_price_history(etf, period="1y", interval="1d")
            if df_sec.empty or len(df_sec) < lb:
                continue
            perf = ((float(df_sec["Close"].iloc[-1]) / float(df_sec["Close"].iloc[-lb])) - 1.0) * 100.0
            sec_rows.append({"Sektor": sec_name, "ETF": etf, "Performance (%)": round(perf, 2), "Größe": 10})

    if sec_rows:
        df_sec_all = pd.DataFrame(sec_rows).sort_values("Performance (%)", ascending=False)
        fig_tree = px.treemap(
            df_sec_all,
            path=["Sektor"],
            values="Größe",
            color="Performance (%)",
            color_continuous_scale=["#dc2626", "#1e293b", "#16a34a"],
            color_continuous_midpoint=0,
            custom_data=["ETF", "Performance (%)"],
        )
        fig_tree.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]}: %{customdata[1]:+.2f}%",
            textfont=dict(size=15),
        )
        apply_chart_layout(fig_tree, title=f"Sektor-Heatmap ({tf_sec})", height=450)
        st.plotly_chart(fig_tree, use_container_width=True)

        st.dataframe(df_sec_all[["Sektor", "ETF", "Performance (%)"]], use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 11: 💸 DIVIDENDEN-KALENDER
# ==============================================================================
elif selected_view == "💸 Dividenden-Kalender":
    st.title("💸 Dividenden-Kalender & Ausschüttungs-Check")
    st.caption("Überblick über Dividendenrenditen, Ex-Tage, Zahltermine und Ausschüttungsmonate in EUR.")

    div_tickers = ["ALV.DE", "MUV2.DE", "DTE.DE", "SIE.DE", "SAP.DE", "ADS.DE", "MSFT", "AAPL", "NVDA", "JPM", "UNH", "7203.T"]
    div_rows = []
    with st.spinner("Lade Dividenden-Termine & Renditen..."):
        for t in div_tickers:
            info = fetch_ticker_info(t)
            df_d = fetch_price_history(t, period="5d", interval="1d")
            if df_d.empty:
                continue
            name = resolve_stock_name(t, info)
            _, fx_eur = resolve_currency_and_fx(t, info)
            px_eur = float(df_d["Close"].iloc[-1]) * fx_eur
            div_rate = info.get("dividendRate")
            div_rate_eur = float(div_rate) * fx_eur if div_rate else 0.0
            div_yield = (div_rate_eur / px_eur * 100.0) if (px_eur > 0 and div_rate_eur > 0) else 0.0

            ex_ts = info.get("exDividendDate")
            ex_date_str = dt.datetime.fromtimestamp(ex_ts).strftime("%d.%m.%Y") if ex_ts else "—"
            ex_month = dt.datetime.fromtimestamp(ex_ts).strftime("%B") if ex_ts else "Jährlich / Quartal"

            div_rows.append(
                {
                    "Unternehmen": name,
                    "Ticker": t,
                    "Kurs (EUR)": round(px_eur, 2),
                    "Dividende p.a. (EUR)": round(div_rate_eur, 2),
                    "Dividendenrendite (%)": round(div_yield, 2),
                    "Letzter / Nächster Ex-Tag": ex_date_str,
                    "Ausschüttungs-Turnus": "Jährlich (DE)" if t.endswith(".DE") else "Quartalsweise / Halbjährlich",
                }
            )

    if div_rows:
        df_div = pd.DataFrame(div_rows).sort_values("Dividendenrendite (%)", ascending=False)
        st.dataframe(df_div, use_container_width=True, hide_index=True)

        fig_div = px.bar(
            df_div,
            x="Unternehmen",
            y="Dividendenrendite (%)",
            color="Dividendenrendite (%)",
            color_continuous_scale=["#38bdf8", "#22c55e"],
            text="Dividendenrendite (%)",
        )
        apply_chart_layout(fig_div, title="Dividendenrendite im Vergleich (%)", height=390)
        st.plotly_chart(fig_div, use_container_width=True)
