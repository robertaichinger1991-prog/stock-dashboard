"""
Stock Screener & Analysis Dashboard – verbesserte Version
---------------------------------------------------------
Ziel: Gute Aktien finden, deren Signale historisch auch wirklich funktioniert haben.

Features:
- Technische Analyse: Golden Cross, RSI, MACD, kombiniertes TA-Signal, Fib-Zone
- Fundamentaldaten + Branchenvergleich (relativ zum Sektor)
- Piotroski F-Score + Levermann-Score
- Qualitäts- + Momentum-Score (Composite)
- Risiko: Beta, Volatilität, Max-Drawdown
- Erweitertes Backtesting (Golden Cross + TA-Signal)
- Aufgewerteter Scanner mit Qualitätsfiltern
- Vergleichs-Matrix, Portfolio-Tracking

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
    "S&P-Auswahl (Auszug)": ["AAPL", "MSFT", "AMZN", "GOOGL", "BRK-B", "JPM", "V", "JNJ", "WMT", "XOM"],
}

DAX40_TICKERS = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE",
    "CBK.DE", "CON.DE", "1COV.DE", "DHL.DE", "DBK.DE", "DB1.DE", "DTE.DE", "DTG.DE",
    "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE",
    "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE", "SRT3.DE",
    "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE",
]

NASDAQ_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP", "COST",
    "ADBE", "CSCO", "NFLX", "AMD", "INTC", "QCOM", "TXN", "INTU", "AMAT", "BKNG",
]

TOP10_CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
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
    "AMD": "AMD", "SALESFORCE": "CRM", "BERKSHIRE": "BRK-B", "JPMORGAN": "JPM",
    "VISA": "V", "JOHNSON & JOHNSON": "JNJ", "WALMART": "WMT", "EXXON": "XOM",
    "COCA-COLA": "KO", "COCA COLA": "KO", "DISNEY": "DIS", "NIKE": "NKE",
    "MCDONALD'S": "MCD", "MCDONALDS": "MCD", "STARBUCKS": "SBUX", "INTEL": "INTC",
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
def get_eur_usd_rate() -> float:
    """Aktueller EUR/USD-Kurs (wie viele USD für 1 EUR). Fallback 1.08."""
    try:
        t = yf.Ticker("EURUSD=X")
        h = t.history(period="5d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return 1.08


def detect_currency(ticker: str, info: dict | None = None) -> str:
    """Best-effort Währung: EUR für .DE/.PA etc., sonst aus info oder USD."""
    t = ticker.upper()
    if t.endswith((".DE", ".PA", ".AS", ".BR", ".MI", ".MC", ".HE", ".ST", ".OL", ".LS", ".VI")):
        return "EUR"
    if t.endswith("-USD") or t.endswith("=X"):
        return "USD"
    if info:
        cur = info.get("currency") or info.get("financialCurrency")
        if cur:
            return cur.upper()
    return "USD"


def to_eur(value, currency: str, eur_usd: float):
    """Konvertiert einen Zahlenwert nach EUR (falls USD)."""
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if currency == "EUR":
        return v
    if currency == "USD" and eur_usd and eur_usd > 0:
        return v / eur_usd
    return v  # andere Währungen unverändert lassen


def format_price_eur(value, currency: str, eur_usd: float) -> str:
    if value is None:
        return "—"
    eur_val = to_eur(value, currency, eur_usd)
    if eur_val is None:
        return "—"
    if currency == "EUR":
        return f"{eur_val:.2f} €"
    if currency == "USD":
        return f"{eur_val:.2f} €  ({float(value):.2f} $)"
    return f"{eur_val:.2f} €"


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
    df = data.dropna(subset=["SMA50", "SMA200", "Close", "RSI"]).copy()
    if len(df) < 250:
        return []
    if signal_type == "golden":
        df["diff"] = df["SMA50"] - df["SMA200"]
        df["signal"] = (df["diff"].shift(1) < 0) & (df["diff"] > 0)
    else:
        df["diff"] = df["SMA50"] - df["SMA200"]
        df["gc"] = (df["diff"].shift(1) < 0) & (df["diff"] > 0)
        df["bull"] = df["SMA50"] > df["SMA200"]
        df["signal"] = df["gc"] | (df["bull"] & (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70))
        df["signal"] = df["signal"] & ~df["signal"].shift(1).fillna(False)
    return df[df["signal"]].index.tolist()


def backtest_signals(data: pd.DataFrame, signal_type: str = "golden", forward_days_list=(21, 63, 126)):
    crosses = _find_signal_dates(data, signal_type)
    if not crosses:
        return {"count": 0, "summary": {}, "details": []}
    df = data.dropna(subset=["Close"])
    results = []
    for dt in crosses:
        try:
            entry_price = df.loc[dt, "Close"]
            idx = df.index.get_loc(dt)
        except Exception:
            continue
        row = {"Datum": dt.strftime("%Y-%m-%d")}
        for days in forward_days_list:
            future_idx = idx + days
            if future_idx < len(df):
                future_price = df.iloc[future_idx]["Close"]
                ret = (future_price / entry_price - 1) * 100
                row[f"Rendite {days}T"] = round(ret, 2)
            else:
                row[f"Rendite {days}T"] = None
        end_idx = min(idx + 126, len(df) - 1)
        window = df.iloc[idx:end_idx + 1]["Close"]
        if len(window) > 1:
            peak = window.cummax()
            dd = ((window - peak) / peak * 100).min()
            row["Max DD 6M"] = round(dd, 2)
        else:
            row["Max DD 6M"] = None
        results.append(row)
    summary = {"count": len(results)}
    for days in forward_days_list:
        col = f"Rendite {days}T"
        vals = [r[col] for r in results if r[col] is not None]
        if vals:
            summary[f"Ø {days}T"] = round(float(np.mean(vals)), 2)
            summary[f"Trefferquote {days}T"] = round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1)
            summary[f"Median {days}T"] = round(float(np.median(vals)), 2)
        else:
            summary[f"Ø {days}T"] = None
            summary[f"Trefferquote {days}T"] = None
            summary[f"Median {days}T"] = None
    return {"count": len(results), "summary": summary, "details": results}


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

            # Währung & EUR-Umrechnung
            currency = detect_currency(ticker, fundamentals)
            eur_usd = get_eur_usd_rate()
            # Faktor: USD -> EUR (dividieren durch EURUSD)
            fx = (1.0 / eur_usd) if (currency == "USD" and eur_usd and eur_usd > 0) else 1.0
            last_price = float(data["Close"].iloc[-1])
            last_price_eur = last_price * fx
            if currency == "USD":
                price_eur_str = f"{last_price_eur:.2f} €"
                price_sub = f"Original: {last_price:.2f} $"
            else:
                price_eur_str = f"{last_price:.2f} €"
                price_sub = "Originalwährung EUR"

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

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Kurs", price_eur_str, price_sub)
            c2.metric("Trend", "🟢 Bullisch" if bullish else "🔴 Bärisch")
            c3.metric("Golden Cross", "✅ Ja" if golden else "—")
            c4.metric("TA-Signal", ta_signal)
            c5.metric("Sektor", str(fundamentals.get("Sektor", "—"))[:18])

            # Signal-Historie-Zeile
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Letztes Golden Cross", gc_date_str)
            sc2.metric("Letztes Kaufsignal", buy_date_str)
            sc3.metric("Währung (Original)", currency)

            chart_title = f"{ticker} – Kurs in € + SMA50/200"
            if currency == "USD":
                chart_title += f"  (EUR/USD {eur_usd:.4f})"

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                row_heights=[0.55, 0.2, 0.25],
                subplot_titles=(chart_title, "RSI (14)", "MACD"),
            )
            fig.add_trace(go.Candlestick(
                x=plot.index, open=plot["Open"], high=plot["High"],
                low=plot["Low"], close=plot["Close"], name="Kurs €"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot.index, y=plot["SMA50"], name="SMA50", line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot.index, y=plot["SMA200"], name="SMA200", line=dict(color="blue")), row=1, col=1)
            for key, color, label in [
                ("Analysten-Kursziel (niedrig)", "#2ca02c", "Min"),
                ("Analysten-Kursziel (mittel)", "#1f77b4", "Normal"),
                ("Analysten-Kursziel (hoch)", "#d62728", "Max"),
            ]:
                val = fundamentals.get(key)
                if val:
                    val_eur = float(val) * fx
                    fig.add_hline(y=val_eur, line_dash="dash", line_color=color,
                                  annotation_text=f"{label} {val_eur:.1f} €", row=1, col=1)
            fib = compute_fib_entry_zone(data)
            if fib:
                fig.add_hrect(y0=fib[0] * fx, y1=fib[1] * fx, fillcolor="green", opacity=0.15,
                              annotation_text="Fib-Einstiegszone", row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="blue")), row=3, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal", line=dict(color="orange")), row=3, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data["MACD_hist"], name="Hist"), row=3, col=1)
            fig.update_layout(height=780, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

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
        "Tech-Auswahl", "DAX-Auswahl (Top 10)", "S&P-Auswahl (Auszug)",
        "🌍 Große Auswahl (S&P500 + DAX + Nasdaq)", "🪙 Top 10 Krypto", "Eigene Liste"
    ])
    if universe == "🌍 Große Auswahl (S&P500 + DAX + Nasdaq)":
        tickers = list(set(fetch_sp500_tickers() + DAX40_TICKERS + NASDAQ_TICKERS))
        st.warning(f"Scannt ca. {len(tickers)} Ticker – dauert 2–5 Minuten.")
    elif universe == "🪙 Top 10 Krypto":
        tickers = TOP10_CRYPTO
    elif universe == "Eigene Liste":
        custom = st.text_input("Ticker / Namen (kommagetrennt)", "Apple, Microsoft, SAP, NVIDIA, ASML")
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
                rate = get_eur_usd_rate()
                last_px = data["Close"].iloc[-1]
                px_eur = to_eur(last_px, cur, rate)
                results.append({
                    "Ticker": t,
                    "Name": (fund.get("Name") or t)[:28],
                    "Sektor": (fund.get("Sektor") or "—")[:16],
                    "Kurs €": round(px_eur, 2) if px_eur is not None else "—",
                    "TA-Signal": signal,
                    "Golden Cross": "✅" if detect_golden_cross(data) else "—",
                    "Piotroski": f"{p_score}/{p_max}" if p_score is not None else "—",
                    "Composite": composite,
                    "Rel. Bew.": val_sc,
                    "RSI": round(data["RSI"].iloc[-1], 1) if not pd.isna(data["RSI"].iloc[-1]) else "—",
                    "KGV": round(fund["KGV (trailing)"], 1) if fund.get("KGV (trailing)") else "—",
                    "Vol %": vol if vol is not None else "—",
                    "Max DD %": max_dd if max_dd is not None else "—",
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
    raw = st.text_input("Ticker oder Namen (kommagetrennt)", "Apple, Microsoft, SAP, NVIDIA, ASML")
    tickers = [resolve_ticker(t.strip()) for t in raw.split(",") if t.strip()]
    if tickers and st.button("Vergleichen"):
        rows = []
        for t in tickers:
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
            rows.append({
                "Ticker": t,
                "Name": (f.get("Name") or t)[:22],
                "Sektor": (f.get("Sektor") or "—")[:14],
                "Kurs": f.get("Kurs"),
                "KGV": f.get("KGV (trailing)"),
                "KBV": f.get("KBV"),
                "ROE": f.get("ROE"),
                "TA-Signal": signal,
                "Piotroski": f"{p_score}/{p_max}" if p_score is not None else "—",
                "Composite": composite,
                "Vol %": vol,
                "Max DD %": max_dd,
            })
        st.dataframe(pd.DataFrame(rows).sort_values("Composite", ascending=False),
                     use_container_width=True, hide_index=True)

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
        for p in portfolio:
            data = load_price_history(p["ticker"], "5d")
            current_raw = data["Close"].iloc[-1] if not data.empty else p["buy_price"]
            fund_p = load_fundamentals(p["ticker"])
            cur = detect_currency(p["ticker"], fund_p)
            rate = get_eur_usd_rate()
            current = to_eur(current_raw, cur, rate) or current_raw
            buy_eur = to_eur(p["buy_price"], cur, rate) or p["buy_price"]
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
        if st.button("Portfolio leeren"):
            save_json(PORTFOLIO_FILE, [])
            st.rerun()

# ===========================================================================
# 5. Backtesting
# ===========================================================================
elif mode == "📉 Backtesting":
    st.title("📉 Backtesting – Signale historisch prüfen")
    user_input = st.text_input("Aktie (Name oder Ticker)", value="Apple")
    ticker = resolve_ticker(user_input) if user_input else ""
    period = st.selectbox("Historie", ["5y", "10y", "max"], index=0)
    signal_choice = st.radio("Welches Signal testen?", ["Golden Cross", "Kombiniertes TA-Kaufsignal"], horizontal=True)
    if ticker and st.button("Backtest starten", type="primary"):
        data = load_price_history(ticker, period)
        if data.empty:
            st.error("Keine Daten.")
        else:
            sig_type = "golden" if signal_choice == "Golden Cross" else "ta"
            result = backtest_signals(data, signal_type=sig_type)
            if result["count"] == 0:
                st.warning("In diesem Zeitraum wurden keine Signale gefunden.")
            else:
                summary = result["summary"]
                st.subheader(f"Zusammenfassung – {result['count']} Signale gefunden ({signal_choice})")
                c1, c2, c3 = st.columns(3)
                c1.metric("Ø Rendite 1 Monat", f"{summary.get('Ø 21T', '—')} %",
                          f"Trefferquote {summary.get('Trefferquote 21T', '—')} %")
                c2.metric("Ø Rendite 3 Monate", f"{summary.get('Ø 63T', '—')} %",
                          f"Trefferquote {summary.get('Trefferquote 63T', '—')} %")
                c3.metric("Ø Rendite 6 Monate", f"{summary.get('Ø 126T', '—')} %",
                          f"Trefferquote {summary.get('Trefferquote 126T', '—')} %")
                st.caption(f"Median 3M: {summary.get('Median 63T', '—')} %  |  Median 6M: {summary.get('Median 126T', '—')} %")
                st.subheader("Einzelne Signale")
                details_df = pd.DataFrame(result["details"])
                st.dataframe(details_df, use_container_width=True, hide_index=True)
                avg_6m = summary.get("Ø 126T")
                hit_6m = summary.get("Trefferquote 126T")
                if avg_6m is not None and hit_6m is not None:
                    if avg_6m > 8 and hit_6m > 55:
                        st.success("Historisch hat dieses Signal bei dieser Aktie relativ gut funktioniert (positive Ø-Rendite + anständige Trefferquote).")
                    elif avg_6m > 0 and hit_6m > 50:
                        st.info("Leicht positiver historischer Rand. Signal war brauchbar, aber nicht überragend.")
                    else:
                        st.warning("Historisch eher schwach oder gemischt. Vorsicht bei alleinigem Vertrauen auf dieses Signal.")
                st.info(
                    "Wichtiger Hinweis: Vergangene Performance ist keine Garantie für die Zukunft. "
                    "Das Backtesting zeigt nur, wie stark das Signal bei dieser Aktie historisch mit späteren Kursbewegungen korreliert hat."
                )
