import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

# ==============================================================================
# SEITENKONFIGURATION & BASIS-STYLING
# ==============================================================================
st.set_page_config(
    page_title="Stock Screener Pro – Globale Märkte, Makro & Krypto-Flows",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 10px 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# HILFSFUNKTIONEN: WÄHRUNGSUMRECHNUNG (EUR) & MARKTDATEN
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_fx_rates_to_eur() -> dict[str, float]:
    """Lädt aktuelle Wechselkurse zur Umrechnung von Fremdwährungen in Euro (EUR)."""
    rates = {
        "EUR": 1.0,
        "USD": 0.92,
        "HKD": 0.118,
        "JPY": 0.0061,
        "CNY": 0.127,
        "GBP": 1.18,
        "CHF": 1.06,
    }
    pairs = {
        "USD": "EURUSD=X",
        "HKD": "EURHKD=X",
        "JPY": "EURJPY=X",
        "CNY": "EURCNY=X",
        "GBP": "EURGBP=X",
        "CHF": "EURCHF=X",
    }
    try:
        raw = yf.download(list(pairs.values()), period="5d", interval="1d", progress=False, auto_adjust=True)
        close_df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        for ccy, ticker in pairs.items():
            if ticker in close_df.columns:
                s = close_df[ticker].dropna()
                if not s.empty and float(s.iloc[-1]) > 0:
                    rates[ccy] = 1.0 / float(s.iloc[-1])
    except Exception:
        pass
    return rates


def convert_to_eur(value: float, currency: str, fx_rates: dict[str, float]) -> float:
    """Rechnet einen Betrag aus einer Quellwährung sauber in EUR um."""
    if value is None or np.isnan(value):
        return 0.0
    ccy = (currency or "EUR").upper()
    if ccy == "GBX" or ccy == "GBP":
        if ccy == "GBX":
            return float(value) / 100.0 * fx_rates.get("GBP", 1.18)
        return float(value) * fx_rates.get("GBP", 1.18)
    return float(value) * fx_rates.get(ccy, 1.0)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Lädt bereinigte OHLCV-Kursdaten für ein beliebiges Symbol."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df = df.dropna(subset=["Close"]).copy()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_metadata(ticker: str) -> dict:
    """Lädt Stammdaten und Fundamentalkennzahlen eines Tickers."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info
    except Exception:
        return {}


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Berechnet den Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Berechnet die Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


# ==============================================================================
# UNIVERSUM FÜR SCANNER (VOLL AUSGESCHRIEBENE NAMEN, DE/US/CN/JP, KRYPTO, METALLE)
# ==============================================================================
SCANNER_UNIVERSE = [
    # Deutschland & Europa
    {"ticker": "SAP.DE", "name": "SAP SE", "region": "Deutschland", "category": "Aktien", "currency": "EUR", "catalyst": "Cloud-ERP-Wachstum & KI-Integration"},
    {"ticker": "SIE.DE", "name": "Siemens AG", "region": "Deutschland", "category": "Aktien", "currency": "EUR", "catalyst": "Industrieautomatisierung & Digital Industries"},
    {"ticker": "ALV.DE", "name": "Allianz SE", "region": "Deutschland", "category": "Aktien", "currency": "EUR", "catalyst": "Stabile Kapitalrendite & Aktienrückkäufe"},
    {"ticker": "ADS.DE", "name": "Adidas AG", "region": "Deutschland", "category": "Aktien", "currency": "EUR", "catalyst": "Margenerholung & globale Markendynamik"},
    {"ticker": "RHM.DE", "name": "Rheinmetall AG", "region": "Deutschland", "category": "Aktien", "currency": "EUR", "catalyst": "Rekord-Auftragsbestand & Kapazitätsausbau"},
    {"ticker": "IFX.DE", "name": "Infineon Technologies AG", "region": "Deutschland", "category": "Aktien", "currency": "EUR", "catalyst": "Leistungshalbleiter für KI-Rechenzentren & E-Mobilität"},
    # USA
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Blackwell/Rubin KI-Rechenzentrums-Nachfrage"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Azure Cloud & Copilot-Monetarisierung"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "AWS-Wachstum & Logistik-Effizienz"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "KI-gestützte Werbeeffizienz & Cashflow-Stärke"},
    {"ticker": "NFLX", "name": "Netflix Inc.", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Werbe-Abo-Skalierung & Live-Sport-Rechte"},
    {"ticker": "TTWO", "name": "Take-Two Interactive Software Inc.", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Grand Theft Auto VI Produktzyklus & Recurring Revenue"},
    {"ticker": "CRWD", "name": "CrowdStrike Holdings Inc.", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Konsolidierung im Cybersecurity-Sektor"},
    {"ticker": "COIN", "name": "Coinbase Global Inc.", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Institutionelle Krypto-Verwahrung & ETF-Volumina"},
    {"ticker": "MSTR", "name": "MicroStrategy Incorporated", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Bitcoin-Treasury-Hebel & Kapitalmarkt-Zugang"},
    {"ticker": "ALB", "name": "Albemarle Corporation", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Bodenbildung im Lithium-Zyklus & Kostensenkungen"},
    {"ticker": "ENPH", "name": "Enphase Energy Inc.", "region": "USA", "category": "Aktien", "currency": "USD", "catalyst": "Lagerabbau bei Wechselrichtern & Zinssenkungsfantasie"},
    # China (HKEX mit sauberer EUR-Umrechnung)
    {"ticker": "1810.HK", "name": "Xiaomi Corporation", "region": "China", "category": "Aktien", "currency": "HKD", "catalyst": "SU7/YU7 Elektroauto-Hochlauf & KI-Ökosystem"},
    {"ticker": "1211.HK", "name": "BYD Company Limited", "region": "China", "category": "Aktien", "currency": "HKD", "catalyst": "Globale E-Auto-Expansion & Batterie-Technologieführerschaft"},
    {"ticker": "0700.HK", "name": "Tencent Holdings Limited", "region": "China", "category": "Aktien", "currency": "HKD", "catalyst": "Gaming-Pipeline, WeChat-Monetarisierung & Rückkäufe"},
    {"ticker": "9988.HK", "name": "Alibaba Group Holding Limited", "region": "China", "category": "Aktien", "currency": "HKD", "catalyst": "Cloud-KI-Wachstum & Bewertungserholung"},
    {"ticker": "9618.HK", "name": "JD.com Inc.", "region": "China", "category": "Aktien", "currency": "HKD", "catalyst": "Konsumstimulus in China & Margenstabilisierung"},
    # Japan (TSE mit sauberer JPY -> EUR Umrechnung)
    {"ticker": "7203.T", "name": "Toyota Motor Corporation", "region": "Japan", "category": "Aktien", "currency": "JPY", "catalyst": "Hybrid-Nachfrage, schwacher Yen & Aktionärsrendite"},
    {"ticker": "6758.T", "name": "Sony Group Corporation", "region": "Japan", "category": "Aktien", "currency": "JPY", "catalyst": "PlayStation-Ökosystem, Bildsensoren & Musikrechte"},
    {"ticker": "8035.T", "name": "Tokyo Electron Limited", "region": "Japan", "category": "Aktien", "currency": "JPY", "catalyst": "Wafer-Fab-Equipment für KI-Chips"},
    {"ticker": "6861.T", "name": "Keyence Corporation", "region": "Japan", "category": "Aktien", "currency": "JPY", "catalyst": "Industrie-Sensorik & globale Automatisierung"},
    {"ticker": "7974.T", "name": "Nintendo Co. Ltd.", "region": "Japan", "category": "Aktien", "currency": "JPY", "catalyst": "Nachfolgekonsole & starke IP-Monetarisierung"},
    # Kryptowährungen
    {"ticker": "BTC-USD", "name": "Bitcoin (BTC)", "region": "Global", "category": "Kryptowährungen", "currency": "USD", "catalyst": "Spot-ETF-Zuflüsse & institutionelle Allokation"},
    {"ticker": "ETH-USD", "name": "Ethereum (ETH)", "region": "Global", "category": "Kryptowährungen", "currency": "USD", "catalyst": "Spot-ETF-Nachfrage, Staking & Layer-2-Skalierung"},
    {"ticker": "SOL-USD", "name": "Solana (SOL)", "region": "Global", "category": "Kryptowährungen", "currency": "USD", "catalyst": "Hohe On-Chain-Aktivität & ETF-Zulassungsdynamik"},
    {"ticker": "XRP-USD", "name": "XRP (XRP)", "region": "Global", "category": "Kryptowährungen", "currency": "USD", "catalyst": "Regulatorische Klarheit & Zahlungsnetzwerk-Adoption"},
    {"ticker": "ADA-USD", "name": "Cardano (ADA)", "region": "Global", "category": "Kryptowährungen", "currency": "USD", "catalyst": "Governance-Meilensteine & DeFi-Ökosystem"},
    {"ticker": "XLM-USD", "name": "Stellar (XLM)", "region": "Global", "category": "Kryptowährungen", "currency": "USD", "catalyst": "Grenzüberschreitende Zahlungen & Tokenisierung"},
    {"ticker": "HBAR-USD", "name": "Hedera (HBAR)", "region": "Global", "category": "Kryptowährungen", "currency": "USD", "catalyst": "Enterprise-Partnerschaften & RWA-Tokenisierung"},
    # Edelmetalle
    {"ticker": "GC=F", "name": "Gold (Feinunze)", "region": "Rohstoffe", "category": "Edelmetalle", "currency": "USD", "catalyst": "Zentralbankkäufe & Absicherung gegen Staatsverschuldung"},
    {"ticker": "SI=F", "name": "Silber (Feinunze)", "region": "Rohstoffe", "category": "Edelmetalle", "currency": "USD", "catalyst": "Industrienachfrage (Solar/Elektronik) & Angebotsdefizit"},
    {"ticker": "PL=F", "name": "Platin (Feinunze)", "region": "Rohstoffe", "category": "Edelmetalle", "currency": "USD", "catalyst": "Strukturelles Angebotsdefizit & Wasserstoff-Katalyse"},
    {"ticker": "PA=F", "name": "Palladium (Feinunze)", "region": "Rohstoffe", "category": "Edelmetalle", "currency": "USD", "catalyst": "Antizyklische Bodenbildung nach langem Abwärtstrend"},
]


# ==============================================================================
# 1. LEITINDIZES QUICK CHECK (INKL. NASDAQ)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_leading_indices() -> pd.DataFrame:
    """Lädt die wichtigsten Leitindizes inklusive Nasdaq für den Quick-Check."""
    indices = [
        {"ticker": "^GDAXI", "name": "DAX 40 (Deutschland)", "ccy": "EUR"},
        {"ticker": "^GSPC", "name": "S&P 500 (USA)", "ccy": "USD"},
        {"ticker": "^IXIC", "name": "Nasdaq Composite (USA)", "ccy": "USD"},
        {"ticker": "^NDX", "name": "Nasdaq 100 (USA Tech)", "ccy": "USD"},
        {"ticker": "^DJI", "name": "Dow Jones Industrial", "ccy": "USD"},
        {"ticker": "^N225", "name": "Nikkei 225 (Japan)", "ccy": "JPY"},
        {"ticker": "^HSI", "name": "Hang Seng Index (Hongkong)", "ccy": "HKD"},
        {"ticker": "^STOXX50E", "name": "Euro Stoxx 50 (Europa)", "ccy": "EUR"},
    ]
    tickers = [x["ticker"] for x in indices]
    rows = []
    try:
        raw = yf.download(tickers, period="220d", interval="1d", progress=False, auto_adjust=True)
        close_df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        for item in indices:
            tk = item["ticker"]
            if tk not in close_df.columns:
                continue
            s = close_df[tk].dropna()
            if len(s) < 2:
                continue
            last_val = float(s.iloc[-1])
            prev_val = float(s.iloc[-2])
            chg_1d = (last_val / prev_val - 1.0) * 100.0
            chg_1m = (last_val / float(s.iloc[-22]) - 1.0) * 100.0 if len(s) >= 22 else 0.0
            sma50 = float(s.tail(50).mean())
            sma200 = float(s.tail(200).mean()) if len(s) >= 200 else sma50

            if last_val > sma50 and last_val > sma200:
                trend = "🟢 Aufwärtstrend"
            elif last_val < sma50 and last_val < sma200:
                trend = "🔴 Abwärtstrend"
            else:
                trend = "🟡 Übergangsphase"

            rows.append({
                "Leitindex": item["name"],
                "Punktestand": round(last_val, 2),
                "1T (%)": round(chg_1d, 2),
                "1M (%)": round(chg_1m, 2),
                "Abstand SMA 50 (%)": round((last_val / sma50 - 1.0) * 100.0, 2),
                "Abstand SMA 200 (%)": round((last_val / sma200 - 1.0) * 100.0, 2),
                "Trendstatus": trend,
            })
    except Exception:
        pass
    return pd.DataFrame(rows)


def render_market_overview() -> None:
    """Rendert die Marktübersicht mit Leitindizes-Quick-Check inklusive Nasdaq."""
    st.header("🌍 Globale Marktübersicht & Leitindizes-Quick-Check")
    df_idx = fetch_leading_indices()

    if df_idx.empty:
        st.warning("Leitindizes konnten aktuell nicht geladen werden. Bitte später erneut versuchen.")
        return

    highlight_names = ["DAX 40 (Deutschland)", "S&P 500 (USA)", "Nasdaq Composite (USA)", "Nasdaq 100 (USA Tech)"]
    cols = st.columns(4)
    for i, h_name in enumerate(highlight_names):
        sub = df_idx[df_idx["Leitindex"] == h_name]
        if not sub.empty:
            r = sub.iloc[0]
            cols[i].metric(
                label=r["Leitindex"],
                value=f"{r['Punktestand']:,.2f} Pkt.",
                delta=f"{r['1T (%)']:+.2f} % (1T)",
            )

    st.subheader("Vollständige Leitindizes-Matrix")
    st.dataframe(
        df_idx,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Punktestand": st.column_config.NumberColumn(format="%.2f"),
            "1T (%)": st.column_config.NumberColumn(format="%+.2f %%"),
            "1M (%)": st.column_config.NumberColumn(format="%+.2f %%"),
            "Abstand SMA 50 (%)": st.column_config.NumberColumn(format="%+.2f %%"),
            "Abstand SMA 200 (%)": st.column_config.NumberColumn(format="%+.2f %%"),
        },
    )

    bull_count = int(df_idx["Trendstatus"].str.contains("Aufwärtstrend").sum())
    total_idx = len(df_idx)
    if bull_count >= total_idx * 0.7:
        st.success(
            f"**Gesamteinschätzung Leitindizes:** Starkes globales Umfeld – {bull_count} von {total_idx} "
            "Leitindizes (inkl. US-Tech/Nasdaq) notieren stabil über SMA 50 und SMA 200."
        )
    elif bull_count >= total_idx * 0.4:
        st.info(
            f"**Gesamteinschätzung Leitindizes:** Selektives Marktumfeld – {bull_count} von {total_idx} "
            "Leitindizes befinden sich im intakten Aufwärtstrend. Fokus auf relative Stärke."
        )
    else:
        st.warning(
            f"**Gesamteinschätzung Leitindizes:** Defensives Umfeld – nur {bull_count} von {total_idx} "
            "Leitindizes über ihren gleitenden Durchschnitten. Risikomanagement priorisieren."
        )


# ==============================================================================
# 2. TRENDFOLGE- & TFA-TURNAROUND-SCANNER (AKTIEN, KRYPTO & EDELMETALLE)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def run_market_scanner() -> pd.DataFrame:
    """Scannt das komplette Universum in EUR und bewertet Trendfolge- und TFA-Turnaround-Setups."""
    fx_rates = get_fx_rates_to_eur()
    tickers = [item["ticker"] for item in SCANNER_UNIVERSE]
    results = []

    try:
        raw = yf.download(tickers, period="1y", interval="1d", progress=False, auto_adjust=True)
        close_df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        high_df = raw["High"] if isinstance(raw.columns, pd.MultiIndex) else raw
        low_df = raw["Low"] if isinstance(raw.columns, pd.MultiIndex) else raw
        vol_df = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else None

        for item in SCANNER_UNIVERSE:
            tk = item["ticker"]
            if tk not in close_df.columns:
                continue
            s_close = close_df[tk].dropna()
            if len(s_close) < 60:
                continue

            s_high = high_df[tk].dropna()
            s_low = low_df[tk].dropna()

            last_orig = float(s_close.iloc[-1])
            prev_orig = float(s_close.iloc[-2])
            ccy = item["currency"]

            price_eur = convert_to_eur(last_orig, ccy, fx_rates)
            chg_1d = (last_orig / prev_orig - 1.0) * 100.0
            chg_1m = (last_orig / float(s_close.iloc[-22]) - 1.0) * 100.0 if len(s_close) >= 22 else 0.0

            sma50_orig = float(s_close.tail(50).mean())
            sma200_orig = float(s_close.tail(200).mean()) if len(s_close) >= 200 else sma50_orig
            high_52w_orig = float(s_high.max())
            low_52w_orig = float(s_low.min())

            sma50_eur = convert_to_eur(sma50_orig, ccy, fx_rates)
            sma200_eur = convert_to_eur(sma200_orig, ccy, fx_rates)

            dist_sma50 = (last_orig / sma50_orig - 1.0) * 100.0
            dist_sma200 = (last_orig / sma200_orig - 1.0) * 100.0
            dist_52w_high = (last_orig / high_52w_orig - 1.0) * 100.0
            dist_52w_low = (last_orig / low_52w_orig - 1.0) * 100.0

            rsi_series = compute_rsi(s_close, 14)
            rsi_val = float(rsi_series.iloc[-1])

            tmp_df = pd.DataFrame({"High": s_high, "Low": s_low, "Close": s_close}).dropna()
            atr_orig = float(compute_atr(tmp_df, 14).iloc[-1]) if len(tmp_df) >= 15 else last_orig * 0.03
            atr_eur = convert_to_eur(atr_orig, ccy, fx_rates)

            vol_ratio = 1.0
            if vol_df is not None and tk in vol_df.columns:
                s_vol = vol_df[tk].dropna()
                if len(s_vol) >= 20 and float(s_vol.tail(20).mean()) > 0:
                    vol_ratio = float(s_vol.tail(5).mean() / s_vol.tail(20).mean())

            trend_score = 0
            if last_orig > sma50_orig:
                trend_score += 25
            if last_orig > sma200_orig:
                trend_score += 25
            if sma50_orig > sma200_orig:
                trend_score += 15
            if dist_52w_high > -12.0:
                trend_score += 15
            if 50.0 <= rsi_val <= 72.0:
                trend_score += 10
            if chg_1m > 0:
                trend_score += 10

            tfa_score = 0
            if dist_52w_high <= -22.0:
                tfa_score += 30
            elif dist_52w_high <= -15.0:
                tfa_score += 15
            if 5.0 <= dist_52w_low <= 45.0:
                tfa_score += 20
            if last_orig > sma50_orig and dist_sma50 <= 12.0:
                tfa_score += 25
            if vol_ratio >= 1.1:
                tfa_score += 15
            if 45.0 <= rsi_val <= 68.0:
                tfa_score += 10

            entry_low = max(price_eur - 0.5 * atr_eur, price_eur * 0.94)
            entry_high = price_eur + 0.25 * atr_eur
            stop_loss = max(price_eur - 2.0 * atr_eur, price_eur * 0.85)
            target_price = price_eur + 4.2 * atr_eur
            risk = max(price_eur - stop_loss, 0.01)
            reward = max(target_price - price_eur, 0.01)
            crv = round(reward / risk, 2)

            if trend_score >= 80:
                verdict = "🟢 Starker Trendkauf"
                assessment = (
                    f"Intakter Aufwärtstrend über SMA 50 ({sma50_eur:.2f} €) und SMA 200 ({sma200_eur:.2f} €). "
                    f"Relative Stärke nahe 52W-Hoch mit CRV {crv:.1f}."
                )
            elif tfa_score >= 70:
                verdict = "🚀 TFA-Turnaround-Signal"
                assessment = (
                    f"Bodenbildung {dist_52w_high:.1f} % unter 52W-Hoch mit frischem Ausbruch über SMA 50. "
                    f"Katalysator: {item['catalyst']}."
                )
            elif trend_score >= 60 or tfa_score >= 55:
                verdict = "🟡 Beobachten / Rücksetzer nutzen"
                assessment = (
                    f"Konstruktive Ausgangslage (RSI {rsi_val:.0f}), Einstieg idealerweise in der Zone "
                    f"{entry_low:.2f} € – {entry_high:.2f} €."
                )
            else:
                verdict = "🔴 Abwarten / Kein Setup"
                assessment = (
                    f"Aktuell unter wichtigen Durchschnitten (Abstand SMA 200: {dist_sma200:+.1f} %). "
                    "Bodenbildung abwarten."
                )

            results.append({
                "Name": item["name"],
                "Ticker": tk,
                "Kategorie": item["category"],
                "Region": item["region"],
                "Kurs (EUR)": round(price_eur, 4 if price_eur < 2 else 2),
                "1T (%)": round(chg_1d, 2),
                "1M (%)": round(chg_1m, 2),
                "RSI (14)": round(rsi_val, 1),
                "Trend-Score": trend_score,
                "TFA-Score": tfa_score,
                "Einstiegszone (EUR)": f"{entry_low:.2f} – {entry_high:.2f} €",
                "Stop-Loss (EUR)": round(stop_loss, 2),
                "Kursziel (EUR)": round(target_price, 2),
                "CRV": crv,
                "Gesamturteil": verdict,
                "Einschätzung & Katalysator": assessment,
            })
    except Exception:
        pass

    return pd.DataFrame(results)


def render_scanner_view() -> None:
    """Rendert den Trendfolge- & TFA-Turnaround-Scanner für Aktien, Krypto & Edelmetalle."""
    st.header("🔎 Trendfolge- & TFA-Turnaround-Scanner (EUR)")
    st.caption(
        "Alle internationalen Titel (inkl. China & Japan) werden automatisch in Euro (EUR) umgerechnet "
        "und mit voll ausgeschriebenen Namen dargestellt."
    )

    df_scan = run_market_scanner()
    if df_scan.empty:
        st.warning("Scanner-Daten konnten derzeit nicht abgerufen werden.")
        return

    col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 1.4])
    with col_f1:
        cat_filter = st.multiselect(
            "Anlageklasse",
            options=["Aktien", "Kryptowährungen", "Edelmetalle"],
            default=["Aktien", "Kryptowährungen", "Edelmetalle"],
        )
    with col_f2:
        reg_filter = st.multiselect(
            "Region / Markt",
            options=sorted(df_scan["Region"].unique().tolist()),
            default=sorted(df_scan["Region"].unique().tolist()),
        )
    with col_f3:
        mode_filter = st.radio(
            "Scanner-Fokus",
            options=["Alle Setups", "Top Trendfolger (Trend-Score ≥ 70)", "TFA-Turnaround-Kandidaten (TFA-Score ≥ 60)"],
            horizontal=True,
        )

    filtered = df_scan[df_scan["Kategorie"].isin(cat_filter) & df_scan["Region"].isin(reg_filter)].copy()
    if mode_filter == "Top Trendfolger (Trend-Score ≥ 70)":
        filtered = filtered[filtered["Trend-Score"] >= 70].sort_values(by="Trend-Score", ascending=False)
    elif mode_filter == "TFA-Turnaround-Kandidaten (TFA-Score ≥ 60)":
        filtered = filtered[filtered["TFA-Score"] >= 60].sort_values(by="TFA-Score", ascending=False)
    else:
        filtered = filtered.sort_values(by=["Trend-Score", "TFA-Score"], ascending=False)

    st.dataframe(
        filtered[
            [
                "Name",
                "Kategorie",
                "Region",
                "Kurs (EUR)",
                "1T (%)",
                "1M (%)",
                "RSI (14)",
                "Trend-Score",
                "TFA-Score",
                "Einstiegszone (EUR)",
                "Stop-Loss (EUR)",
                "Kursziel (EUR)",
                "CRV",
                "Gesamturteil",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "1T (%)": st.column_config.NumberColumn(format="%+.2f %%"),
            "1M (%)": st.column_config.NumberColumn(format="%+.2f %%"),
            "Stop-Loss (EUR)": st.column_config.NumberColumn(format="%.2f €"),
            "Kursziel (EUR)": st.column_config.NumberColumn(format="%.2f €"),
        },
    )

    st.subheader("📋 Detaillierte Einschätzungen & Katalysatoren")
    for _, row in filtered.head(10).iterrows():
        with st.expander(f"{row['Gesamturteil']} | {row['Name']} ({row['Kurs (EUR)']} € | CRV {row['CRV']})"):
            st.markdown(
                f"- **Anlageklasse / Region:** {row['Kategorie']} ({row['Region']})\n"
                f"- **Scores:** Trend-Score **{row['Trend-Score']}/100** | TFA-Turnaround-Score **{row['TFA-Score']}/100** | RSI (14): **{row['RSI (14)']}**\n"
                f"- **Trade-Parameter:** Einstiegszone **{row['Einstiegszone (EUR)']}** | ATR-Stop-Loss **{row['Stop-Loss (EUR)']} €** | Kursziel **{row['Kursziel (EUR)']} €** (CRV **{row['CRV']}**)\n"
                f"- **Begründung:** {row['Einschätzung & Katalysator']}"
            )


# ==============================================================================
# 3. EINZELAKTIEN-ANALYSE (ATR-STOP, CRV, PIOTROSKI F-SCORE & LEVERMANN-SCORE)
# ==============================================================================
def calculate_piotroski_score(info: dict) -> tuple[int, list[str]]:
    """Berechnet den Piotroski F-Score (0–9) aus verfügbaren Fundamentaldaten."""
    score = 0
    notes = []

    roa = info.get("returnOnAssets")
    if roa is not None and roa > 0:
        score += 2
        notes.append(f"Positiver ROA ({roa * 100:.1f} %) (+2)")
    else:
        notes.append("ROA nicht positiv / k.A. (0)")

    ocf = info.get("operatingCashflow")
    net_inc = info.get("netIncomeToCommon")
    if ocf is not None and ocf > 0:
        score += 2
        notes.append("Positiver operativer Cashflow (+2)")
    if ocf is not None and net_inc is not None and ocf > net_inc:
        score += 1
        notes.append("Operativer Cashflow > Nettogewinn (hohe Gewinnqualität) (+1)")

    debt_eq = info.get("debtToEquity")
    if debt_eq is not None and debt_eq < 100:
        score += 1
        notes.append(f"Solide Verschuldungsquote (Debt/Equity {debt_eq:.0f} %) (+1)")

    curr_ratio = info.get("currentRatio")
    if curr_ratio is not None and curr_ratio >= 1.2:
        score += 1
        notes.append(f"Gute Liquidität 3. Grades (Current Ratio {curr_ratio:.2f}) (+1)")

    gross_margin = info.get("grossMargins")
    if gross_margin is not None and gross_margin > 0.30:
        score += 1
        notes.append(f"Starke Bruttomarge ({gross_margin * 100:.1f} %) (+1)")

    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None and rev_growth > 0.05:
        score += 1
        notes.append(f"Umsatzwachstum ({rev_growth * 100:.1f} %) (+1)")

    return min(score, 9), notes


def calculate_levermann_score(info: dict, chg_6m: float, chg_1y: float, rsi_val: float) -> tuple[int, list[str]]:
    """Berechnet eine praxisnahe Approximation des Levermann-Scores (-6 bis +10)."""
    score = 0
    notes = []

    roe = info.get("returnOnEquity")
    if roe is not None:
        if roe > 0.20:
            score += 1
            notes.append(f"Eigenkapitalrendite > 20 % ({roe * 100:.1f} %) (+1)")
        elif roe < 0.10:
            score -= 1
            notes.append(f"Eigenkapitalrendite < 10 % ({roe * 100:.1f} %) (-1)")

    ebit_margin = info.get("operatingMargins")
    if ebit_margin is not None:
        if ebit_margin > 0.12:
            score += 1
            notes.append(f"EBIT-Marge > 12 % ({ebit_margin * 100:.1f} %) (+1)")
        elif ebit_margin < 0.06:
            score -= 1
            notes.append(f"EBIT-Marge < 6 % ({ebit_margin * 100:.1f} %) (-1)")

    eq_ratio_proxy = info.get("debtToEquity")
    if eq_ratio_proxy is not None:
        if eq_ratio_proxy < 80:
            score += 1
            notes.append("Niedrige Verschuldung / hohe Eigenkapitalquote (+1)")
        elif eq_ratio_proxy > 200:
            score -= 1
            notes.append("Hohe Verschuldungsquote (-1)")

    pe = info.get("trailingPE")
    fwd_pe = info.get("forwardPE")
    if pe is not None and pe > 0:
        if pe < 16:
            score += 1
            notes.append(f"Attraktives KGV ({pe:.1f}) (+1)")
        elif pe > 30:
            score -= 1
            notes.append(f"Hohes KGV ({pe:.1f}) (-1)")
    if fwd_pe is not None and fwd_pe > 0:
        if fwd_pe < 16:
            score += 1
            notes.append(f"Attraktives Forward-KGV ({fwd_pe:.1f}) (+1)")
        elif fwd_pe > 28:
            score -= 1
            notes.append(f"Hohes Forward-KGV ({fwd_pe:.1f}) (-1)")

    if chg_6m > 5.0:
        score += 1
        notes.append(f"Positives 6-Monats-Momentum ({chg_6m:+.1f} %) (+1)")
    elif chg_6m < -5.0:
        score -= 1
        notes.append(f"Negatives 6-Monats-Momentum ({chg_6m:+.1f} %) (-1)")

    if chg_1y > 5.0:
        score += 1
        notes.append(f"Positives 12-Monats-Momentum ({chg_1y:+.1f} %) (+1)")
    elif chg_1y < -5.0:
        score -= 1
        notes.append(f"Negatives 12-Monats-Momentum ({chg_1y:+.1f} %) (-1)")

    if rsi_val < 32:
        score += 1
        notes.append("Überverkaufte Reversal-Chance (RSI < 32) (+1)")
    elif rsi_val > 75:
        score -= 1
        notes.append("Kurzfristig überkauft (RSI > 75) (-1)")

    return score, notes


def render_single_stock_analysis() -> None:
    """Rendert die Einzelaktien-Analyse mit Chart, ATR-Stop-Loss, CRV, Piotroski- & Levermann-Score."""
    st.header("🔬 Einzelaktien- & Asset-Tiefenanalyse")

    preset_map = {f"{x['name']} ({x['ticker']})": x["ticker"] for x in SCANNER_UNIVERSE}
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_label = st.selectbox("Titel aus Universum wählen", options=list(preset_map.keys()), index=0)
    with col_sel2:
        custom_ticker = st.text_input("Oder eigenes Symbol eingeben (z. B. AAPL, SAP.DE, 1810.HK)", value="")

    ticker = custom_ticker.strip().upper() if custom_ticker.strip() else preset_map[selected_label]
    df = fetch_price_history(ticker, period="2y", interval="1d")

    if df.empty or len(df) < 50:
        st.error(f"Keine ausreichenden Kursdaten für '{ticker}' gefunden.")
        return

    info = fetch_ticker_metadata(ticker)
    fx_rates = get_fx_rates_to_eur()
    default_ccy = (
        "EUR" if ticker.endswith(".DE")
        else "HKD" if ticker.endswith(".HK")
        else "JPY" if ticker.endswith(".T")
        else "USD"
    )
    ccy = info.get("currency") or default_ccy
    full_name = info.get("longName") or info.get("shortName") or selected_label

    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["ATR"] = compute_atr(df, 14)

    last_orig = float(df["Close"].iloc[-1])
    atr_orig = float(df["ATR"].iloc[-1]) if not np.isnan(df["ATR"].iloc[-1]) else last_orig * 0.03

    price_eur = convert_to_eur(last_orig, ccy, fx_rates)
    atr_eur = convert_to_eur(atr_orig, ccy, fx_rates)
    stop_loss_eur = max(price_eur - 2.0 * atr_eur, price_eur * 0.85)
    target_eur = price_eur + 4.5 * atr_eur
    entry_low_eur = max(price_eur - 0.5 * atr_eur, price_eur * 0.94)
    entry_high_eur = price_eur + 0.25 * atr_eur
    crv = round((target_eur - price_eur) / max(price_eur - stop_loss_eur, 0.01), 2)

    chg_6m = (last_orig / float(df["Close"].iloc[-126]) - 1.0) * 100.0 if len(df) >= 126 else 0.0
    chg_1y = (last_orig / float(df["Close"].iloc[-252]) - 1.0) * 100.0 if len(df) >= 252 else 0.0
    rsi_now = float(df["RSI"].iloc[-1])

    piotroski_val, piotroski_notes = calculate_piotroski_score(info)
    levermann_val, levermann_notes = calculate_levermann_score(info, chg_6m, chg_1y, rsi_now)

    st.subheader(f"{full_name} ({ticker})")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Aktueller Kurs", f"{price_eur:,.2f} €")
    k2.metric("ATR (14T) Stop-Loss", f"{stop_loss_eur:,.2f} €", delta=f"-{((price_eur - stop_loss_eur) / price_eur * 100):.1f} %", delta_color="inverse")
    k3.metric("Kursziel (4,5x ATR)", f"{target_eur:,.2f} €", delta=f"+{((target_eur - price_eur) / price_eur * 100):.1f} %")
    k4.metric("CRV (Chance/Risiko)", f"{crv:.2f}")
    k5.metric("Piotroski F-Score", f"{piotroski_val} / 9")
    k6.metric("Levermann-Score", f"{levermann_val:+d} Pkt.")

    plot_df = df.tail(260).copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{full_name} – Kursverlauf & Gleitende Durchschnitte ({ccy})", "RSI (14)"),
    )
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Kurs",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["SMA50"], mode="lines", name="SMA 50", line=dict(color="#2563eb", width=1.8)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["SMA200"], mode="lines", name="SMA 200", line=dict(color="#f59e0b", width=2.0)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["RSI"], mode="lines", name="RSI (14)", line=dict(color="#8b5cf6", width=1.6)),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_layout(height=540, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(fig, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("### 📊 Fundamentale Scoring-Details")
        st.markdown(f"**Piotroski F-Score ({piotroski_val}/9):**")
        for n in piotroski_notes:
            st.markdown(f"- {n}")
        st.markdown(f"**Levermann-Score ({levermann_val:+d} Punkte):**")
        for n in levermann_notes:
            st.markdown(f"- {n}")

    with c_right:
        st.markdown("### 🎯 Trade-Planung & Gesamturteil")
        st.info(
            f"**Optimale Einstiegszone:** {entry_low_eur:,.2f} € – {entry_high_eur:,.2f} €  \n"
            f"**ATR-Stop-Loss (2x ATR):** {stop_loss_eur:,.2f} € (Volatilität ATR: {atr_eur:,.2f} €)  \n"
            f"**Technisches Kursziel:** {target_eur:,.2f} € | **Chance-Risiko-Verhältnis:** {crv:.2f}"
        )
        summary_text = info.get("longBusinessSummary", "")
        if summary_text:
            st.markdown("**Unternehmensprofil:**")
            st.write(summary_text[:650] + ("..." if len(summary_text) > 650 else ""))


# ==============================================================================
# 4. REZESSIONS-BAROMETER (6 FRED-INDIKATOREN, VERLAUFSCHARTS & STRATEGIE)
# ==============================================================================
@st.cache_data(ttl=7200, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    """Lädt eine makroökonomische Zeitreihe direkt von FRED als CSV (ohne API-Key)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        date_col = df.columns[0]
        val_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna().set_index(date_col)
        return df[val_col]
    except Exception:
        return pd.Series(dtype=float)


def render_recession_barometer() -> None:
    """Rendert das eigenständige Rezessions-Barometer auf Basis der 6 FRED-Indikatoren."""
    st.header("🧭 US- & Globales Rezessions-Barometer (FRED-Makrodaten)")
    st.caption(
        "Echtzeit-Auswertung von 6 führenden Makro-Indikatoren der Federal Reserve Bank of St. Louis (FRED) "
        "inklusive Risiko-Score, Schwellenwerten und Portfolio-Strategie."
    )

    indicators_cfg = [
        {
            "id": "SAHMREALTIME",
            "name": "Sahm-Regel Rezessionsindikator",
            "unit": "Prozentpunkte",
            "weight": 0.25,
            "threshold_warn": 0.30,
            "threshold_crit": 0.50,
            "desc": "Misst den Anstieg der US-Arbeitslosenquote gegenüber dem 12-Monats-Tief. Ab ≥ 0,50 Pkt. wurde historisch jede US-Rezession ausgelöst.",
        },
        {
            "id": "T10Y2Y",
            "name": "US-Zinsstrukturkurve (10J – 2J)",
            "unit": "%",
            "weight": 0.20,
            "threshold_warn": 0.15,
            "threshold_crit": 0.00,
            "desc": "Differenz zwischen 10-jährigen und 2-jährigen US-Staatsanleihen. Eine Inversion (< 0 %) sowie die anschließende Re-Steepening-Phase sind klassische Rezessionsvorboten.",
        },
        {
            "id": "BAMLH0A0HYM2",
            "name": "US High-Yield Credit Spreads",
            "unit": "%",
            "weight": 0.20,
            "threshold_warn": 4.50,
            "threshold_crit": 6.00,
            "desc": "Risikoaufschlag für hochverzinsliche Unternehmensanleihen. Steigende Spreads signalisieren Kreditstress und Liquiditätsengpässe.",
        },
        {
            "id": "UMCSENT",
            "name": "US-Konsumentenstimmung (Univ. of Michigan)",
            "unit": "Punkte",
            "weight": 0.15,
            "threshold_warn": 68.0,
            "threshold_crit": 58.0,
            "desc": "Frühindikator für den privaten Konsum, der rund 70 % des US-BIP ausmacht.",
        },
        {
            "id": "CPIAUCSL",
            "name": "US-Inflation (CPI YoY)",
            "unit": "%",
            "weight": 0.10,
            "threshold_warn": 3.20,
            "threshold_crit": 4.50,
            "desc": "Jährliche Verbraucherpreisinflation. Hohe Teuerung schränkt den Zinssenkungsspielraum der Notenbanken ein.",
        },
        {
            "id": "FEDFUNDS",
            "name": "US-Leitzins (Fed Funds Rate)",
            "unit": "%",
            "weight": 0.10,
            "threshold_warn": 4.25,
            "threshold_crit": 5.25,
            "desc": "Geldpolitischer Leitzins der Federal Reserve. Ein restriktives Zinsniveau bremst Investitionen und Kreditvergabe mit Zeitverzögerung.",
        },
    ]

    series_map = {}
    risk_scores = {}
    summary_rows = []

    for cfg in indicators_cfg:
        s = fetch_fred_series(cfg["id"])
        if cfg["id"] == "CPIAUCSL" and len(s) >= 13:
            s = s.pct_change(12).dropna() * 100.0
        series_map[cfg["id"]] = s

        if s.empty:
            val = 0.0
            r_score = 40.0
            status = "🟡 Keine Live-Daten"
        else:
            val = float(s.iloc[-1])
            sid = cfg["id"]
            if sid == "SAHMREALTIME":
                r_score = float(np.clip((val / 0.60) * 100.0, 0.0, 100.0))
            elif sid == "T10Y2Y":
                if val < 0:
                    r_score = float(np.clip(65.0 + abs(val) * 35.0, 65.0, 100.0))
                elif val < 0.35:
                    r_score = 50.0
                else:
                    r_score = float(np.clip(35.0 - val * 15.0, 5.0, 35.0))
            elif sid == "BAMLH0A0HYM2":
                r_score = float(np.clip(((val - 2.8) / 4.2) * 100.0, 0.0, 100.0))
            elif sid == "UMCSENT":
                r_score = float(np.clip(((90.0 - val) / 40.0) * 100.0, 0.0, 100.0))
            elif sid == "CPIAUCSL":
                r_score = float(np.clip(((val - 2.0) / 3.5) * 100.0, 0.0, 100.0))
            else:
                r_score = float(np.clip(((val - 2.0) / 3.5) * 100.0, 0.0, 100.0))

            if r_score >= 65:
                status = "🔴 Erhöhtes Risiko"
            elif r_score >= 40:
                status = "🟠 Beobachten"
            else:
                status = "🟢 Entspannt"

        risk_scores[cfg["id"]] = round(r_score, 1)
        summary_rows.append({
            "Indikator": cfg["name"],
            "FRED-Code": cfg["id"],
            "Aktueller Wert": f"{val:.2f} {cfg['unit']}" if not s.empty else "k.A.",
            "Teil-Risiko (0–100)": round(r_score, 1),
            "Gewichtung": f"{int(cfg['weight'] * 100)} %",
            "Status": status,
        })

    total_risk = round(
        sum(risk_scores[cfg["id"]] * cfg["weight"] for cfg in indicators_cfg),
        1,
    )

    if total_risk <= 30:
        regime = "Niedriges Rezessionsrisiko (Expansion)"
        gauge_color = "#2ca02c"
        strat_title = "🟢 Expansives Umfeld – Prozyklische Aktien- & Wachstumsallokation"
        strat_text = (
            "Arbeitsmarkt, Kreditspreads und Konsum zeigen keine akuten Rezessionssignale. "
            "Qualitätsaktien, Halbleiter/KI-Trends sowie zyklische Wachstumswerte und Krypto-Assets "
            "finden ein konstruktives Makro-Umfeld vor."
        )
    elif total_risk <= 55:
        regime = "Moderates Risiko (Spätzyklus / Soft Landing)"
        gauge_color = "#eab308"
        strat_title = "🟡 Spätzyklische Übergangsphase – Qualitätsfokus & gestaffelte Einstiege"
        strat_text = (
            "Einzelne Frühindikatoren (z. B. Zinsniveau oder Konsumentenstimmung) mahnen zur Vorsicht, "
            "während der Arbeitsmarkt noch stabil bleibt. Fokus auf bilanzstarke Qualitätsunternehmen "
            "(hoher Piotroski-Score), selektive TFA-Turnarounds und ausreichend Cash-Quote für Rücksetzer."
        )
    elif total_risk <= 75:
        regime = "Erhöhtes Rezessionsrisiko (Warnstufe)"
        gauge_color = "#ff7f0e"
        strat_title = "🟠 Erhöhte Makro-Risiken – Defensives Risikomanagement aktivieren"
        strat_text = (
            "Mehrere Frühindikatoren signalisieren eine deutliche konjunkturelle Abkühlung. "
            "Engere ATR-Stop-Loss-Marken setzen, Zykliker reduzieren und defensive Cashflow-Werte "
            "sowie Edelmetalle (Gold) stärker gewichten."
        )
    else:
        regime = "Akute Rezessionsgefahr (Stress-Regime)"
        gauge_color = "#d62728"
        strat_title = "🔴 Hohes Rezessionsrisiko – Kapitalerhalt & Antizyklische Vorbereitung"
        strat_text = (
            "Sahm-Regel und Kreditspreads deuten auf eine wirtschaftliche Kontraktion hin. "
            "Liquidität halten, Hebel vermeiden und erst bei einer Trendwende der Frühindikatoren "
            "antizyklisch in stark überverkaufte Qualitätsaktien umschichten."
        )

    col_g, col_s = st.columns([1.05, 1.45], gap="large")
    with col_g:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=total_risk,
                number={"suffix": " / 100", "font": {"size": 34}},
                title={
                    "text": f"<b>Rezessions-Risiko-Score</b><br><span style='font-size:13px;color:gray'>{regime}</span>",
                    "font": {"size": 17},
                },
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(44, 160, 44, 0.22)"},
                        {"range": [30, 55], "color": "rgba(234, 179, 8, 0.22)"},
                        {"range": [55, 75], "color": "rgba(255, 127, 14, 0.22)"},
                        {"range": [75, 100], "color": "rgba(214, 39, 40, 0.25)"},
                    ],
                },
            )
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=55, b=15), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_s:
        st.subheader(strat_title)
        st.write(strat_text)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.subheader("📈 Historische Verläufe & Detaillierte Indikator-Beschreibungen")
    cols_hist = st.columns(2)
    for idx, cfg in enumerate(indicators_cfg):
        s = series_map[cfg["id"]]
        with cols_hist[idx % 2]:
            st.markdown(f"**{cfg['name']} (`{cfg['id']}`)**")
            st.caption(cfg["desc"])
            if not s.empty:
                s_recent = s[s.index >= (datetime.now() - timedelta(days=365 * 5))]
                fig_h = go.Figure()
                fig_h.add_trace(
                    go.Scatter(
                        x=s_recent.index,
                        y=s_recent.values,
                        mode="lines",
                        name=cfg["id"],
                        line=dict(color="#2563eb", width=2),
                    )
                )
                fig_h.add_hline(y=cfg["threshold_crit"], line_dash="dash", line_color="red", annotation_text="Kritisch")
                fig_h.update_layout(height=230, margin=dict(l=15, r=15, t=20, b=20))
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.info("Historische Reihe aktuell nicht verfügbar.")


# ==============================================================================
# 5. BACKTESTING-ENGINE (GOLDEN CROSS & RSI – KLARER SIGNALCHART OHNE KAPITALKURVE)
# ==============================================================================
def render_backtesting_engine() -> None:
    """Rendert die Backtesting-Engine für Golden Cross & RSI mit klarem Kauf-/Verkaufssignal-Chart."""
    st.header("⏪ Backtesting-Engine (Golden Cross & RSI)")
    st.caption(
        "Visuelle Auswertung historischer Kauf- und Verkaufssignale direkt im Kurschart "
        "inklusive Kennzahlen und detaillierter Trade-Historie."
    )

    preset_map = {f"{x['name']} ({x['ticker']})": x["ticker"] for x in SCANNER_UNIVERSE}
    c1, c2, c3 = st.columns([1.6, 1.4, 1.0])
    with c1:
        sel_name = st.selectbox("Basiswert auswählen", options=list(preset_map.keys()), index=6, key="bt_asset")
    with c2:
        strategy = st.selectbox(
            "Handelsstrategie",
            options=["Golden Cross (SMA 50 / SMA 200)", "RSI-Reversal (Kauf < 30 / Verkauf > 70)"],
        )
    with c3:
        years = st.slider("Zeitraum (Jahre)", min_value=2, max_value=10, value=5)

    ticker = preset_map[sel_name]
    df = fetch_price_history(ticker, period=f"{years}y", interval="1d")
    if df.empty or len(df) < 210:
        st.warning("Für den gewählten Zeitraum liegen nicht genügend historische Daten vor.")
        return

    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df = df.dropna().copy()

    trades = []
    in_pos = False
    entry_date = None
    entry_price = 0.0

    for i in range(1, len(df)):
        date_now = df.index[i]
        price_now = float(df["Close"].iloc[i])

        if strategy.startswith("Golden Cross"):
            buy_cond = (df["SMA50"].iloc[i - 1] <= df["SMA200"].iloc[i - 1]) and (df["SMA50"].iloc[i] > df["SMA200"].iloc[i])
            sell_cond = (df["SMA50"].iloc[i - 1] >= df["SMA200"].iloc[i - 1]) and (df["SMA50"].iloc[i] < df["SMA200"].iloc[i])
        else:
            buy_cond = (df["RSI"].iloc[i - 1] >= 30.0) and (df["RSI"].iloc[i] < 30.0)
            sell_cond = (df["RSI"].iloc[i - 1] <= 70.0) and (df["RSI"].iloc[i] > 70.0)

        if not in_pos and buy_cond:
            in_pos = True
            entry_date = date_now
            entry_price = price_now
        elif in_pos and sell_cond:
            ret_pct = (price_now / entry_price - 1.0) * 100.0
            trades.append({
                "Einstieg": entry_date.strftime("%Y-%m-%d"),
                "Einstiegskurs": round(entry_price, 2),
                "Ausstieg": date_now.strftime("%Y-%m-%d"),
                "Ausstiegskurs": round(price_now, 2),
                "Haltedauer (Tage)": (date_now - entry_date).days,
                "Rendite (%)": round(ret_pct, 2),
                "Ergebnis": "✅ Gewinn" if ret_pct > 0 else "❌ Verlust",
            })
            in_pos = False

    if in_pos:
        last_date = df.index[-1]
        last_price = float(df["Close"].iloc[-1])
        ret_pct = (last_price / entry_price - 1.0) * 100.0
        trades.append({
            "Einstieg": entry_date.strftime("%Y-%m-%d"),
            "Einstiegskurs": round(entry_price, 2),
            "Ausstieg": f"{last_date.strftime('%Y-%m-%d')} (Offen)",
            "Ausstiegskurs": round(last_price, 2),
            "Haltedauer (Tage)": (last_date - entry_date).days,
            "Rendite (%)": round(ret_pct, 2),
            "Ergebnis": "⏳ Laufende Position",
        })

    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)
    win_rate = float((df_trades["Rendite (%)"] > 0).mean() * 100.0) if total_trades > 0 else 0.0
    avg_ret = float(df_trades["Rendite (%)"].mean()) if total_trades > 0 else 0.0
    cum_ret = float(((1.0 + df_trades["Rendite (%)"] / 100.0).prod() - 1.0) * 100.0) if total_trades > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Abgeschlossene / Aktive Signale", f"{total_trades}")
    m2.metric("Trefferquote", f"{win_rate:.1f} %")
    m3.metric("Ø Rendite pro Trade", f"{avg_ret:+.2f} %")
    m4.metric("Kumulierte Strategie-Rendite", f"{cum_ret:+.2f} %")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Schlusskurs", line=dict(color="#1f2937", width=2))
    )
    if strategy.startswith("Golden Cross"):
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA 50", line=dict(color="#2563eb", width=1.5))
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA200"], mode="lines", name="SMA 200", line=dict(color="#f59e0b", width=1.8))
        )

    if not df_trades.empty:
        buy_dates = pd.to_datetime(df_trades["Einstieg"])
        buy_prices = df_trades["Einstiegskurs"]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="Kaufsignal (Buy)",
                marker=dict(symbol="triangle-up", size=14, color="#16a34a", line=dict(width=1, color="black")),
            )
        )

        closed_trades = df_trades[~df_trades["Ausstieg"].str.contains("Offen")]
        if not closed_trades.empty:
            sell_dates = pd.to_datetime(closed_trades["Ausstieg"])
            sell_prices = closed_trades["Ausstiegskurs"]
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    name="Verkaufssignal (Sell)",
                    marker=dict(symbol="triangle-down", size=14, color="#dc2626", line=dict(width=1, color="black")),
                )
            )

    fig.update_layout(
        title=f"Kauf- & Verkaufssignale im Kurschart – {sel_name} ({strategy})",
        height=480,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📜 Detaillierte Trade-Historie")
    if df_trades.empty:
        st.info("Im gewählten Zeitraum wurden keine Handelssignale ausgelöst.")
    else:
        st.dataframe(df_trades, use_container_width=True, hide_index=True)


# ==============================================================================
# 6. KRYPTO & ETF-FLOWS INKL. BAROMETER & GESAMTMARKT-EINSCHÄTZUNG (IN EUR)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_market_indicators() -> dict:
    """Lädt Sentiment- (Fear & Greed), Trend- (SMA 50/200) und ETF-Proxy-Daten für den Krypto-Gesamtmarkt."""
    fng_current = 50.0
    fng_prev_week = 50.0
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=7", timeout=6)
        if resp.status_code == 200:
            fng_data = resp.json().get("data", [])
            if fng_data:
                fng_current = float(fng_data[0]["value"])
                fng_prev_week = float(fng_data[-1]["value"])
    except Exception:
        pass

    crypto_tickers = {
        "BTC-USD": "Bitcoin (BTC)",
        "ETH-USD": "Ethereum (ETH)",
        "SOL-USD": "Solana (SOL)",
        "XRP-USD": "XRP (XRP)",
        "ADA-USD": "Cardano (ADA)",
        "XLM-USD": "Stellar (XLM)",
        "HBAR-USD": "Hedera (HBAR)",
    }
    etf_proxies = ["IBIT", "ETHA"]
    all_tickers = list(crypto_tickers.keys()) + etf_proxies
    fx_rates = get_fx_rates_to_eur()

    above_sma50_count = 0
    above_sma200_count = 0
    positive_7d_altcoins = 0
    btc_ret_7d = 0.0
    alt_returns_7d = []
    etf_flow_proxy_score = 50.0
    coin_details = []

    try:
        raw = yf.download(
            all_tickers,
            period="210d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        close_df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        vol_df = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else None

        for ticker, full_name in crypto_tickers.items():
            if ticker not in close_df.columns:
                continue
            series = close_df[ticker].dropna()
            if len(series) < 50:
                continue

            last_price = float(series.iloc[-1])
            price_eur = convert_to_eur(last_price, "USD", fx_rates)
            sma50 = float(series.tail(50).mean())
            sma200 = float(series.tail(200).mean()) if len(series) >= 200 else sma50
            ret_7d = float((last_price / series.iloc[-8] - 1.0) * 100) if len(series) >= 8 else 0.0

            is_above_50 = last_price > sma50
            is_above_200 = last_price > sma200
            if is_above_50:
                above_sma50_count += 1
            if is_above_200:
                above_sma200_count += 1

            if ticker == "BTC-USD":
                btc_ret_7d = ret_7d
            else:
                alt_returns_7d.append(ret_7d)
                if ret_7d > 0:
                    positive_7d_altcoins += 1

            coin_details.append({
                "Asset": full_name,
                "Kurs (EUR)": round(price_eur, 4 if price_eur < 2 else 2),
                "7T-Trend (%)": round(ret_7d, 2),
                "Über SMA 50": "Ja" if is_above_50 else "Nein",
                "Über SMA 200": "Ja" if is_above_200 else "Nein",
            })

        etf_signals = []
        for etf in etf_proxies:
            if etf in close_df.columns and len(close_df[etf].dropna()) >= 6:
                s_c = close_df[etf].dropna()
                ret_5d = float((s_c.iloc[-1] / s_c.iloc[-6] - 1.0) * 100)
                vol_ratio = 1.0
                if vol_df is not None and etf in vol_df.columns:
                    s_v = vol_df[etf].dropna()
                    if len(s_v) >= 20 and s_v.tail(20).mean() > 0:
                        vol_ratio = float(s_v.tail(5).mean() / s_v.tail(20).mean())
                raw_sig = 50.0 + (ret_5d * 4.5 * np.clip(vol_ratio, 0.7, 1.8))
                etf_signals.append(float(np.clip(raw_sig, 0.0, 100.0)))
        if etf_signals:
            etf_flow_proxy_score = float(np.mean(etf_signals))

    except Exception:
        pass

    total_coins = max(len(coin_details), 1)
    total_alts = max(len(alt_returns_7d), 1)
    avg_alt_ret_7d = float(np.mean(alt_returns_7d)) if alt_returns_7d else 0.0

    score_fng = float(np.clip(fng_current + 0.25 * (fng_current - fng_prev_week), 0.0, 100.0))
    score_trend = float(
        np.clip(
            ((above_sma50_count / total_coins) * 60.0) + ((above_sma200_count / total_coins) * 40.0),
            0.0,
            100.0,
        )
    )
    breadth_ratio = (positive_7d_altcoins / total_alts) * 70.0
    rotation_bonus = np.clip(15.0 + (avg_alt_ret_7d - btc_ret_7d) * 1.5, 0.0, 30.0)
    score_breadth = float(np.clip(breadth_ratio + rotation_bonus, 0.0, 100.0))

    return {
        "fng_current": fng_current,
        "fng_prev_week": fng_prev_week,
        "score_fng": round(score_fng, 1),
        "score_trend": round(score_trend, 1),
        "score_breadth": round(score_breadth, 1),
        "etf_flow_proxy_score": round(etf_flow_proxy_score, 1),
        "btc_ret_7d": round(btc_ret_7d, 2),
        "avg_alt_ret_7d": round(avg_alt_ret_7d, 2),
        "above_sma50_count": above_sma50_count,
        "above_sma200_count": above_sma200_count,
        "total_coins": total_coins,
        "coin_details": coin_details,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def generate_daily_crypto_etf_flows_eur(days: int = 30) -> pd.DataFrame:
    """
    Berechnet geschätzte tägliche Netto-Kapitalflüsse (in Mio. EUR) für Bitcoin, Ethereum,
    Solana, XRP, Cardano, Stellar und Hedera auf Basis realer Spot-ETF-/ETP-Kurs- & Volumendaten.
    """
    fx_rates = get_fx_rates_to_eur()
    usd_to_eur = fx_rates.get("USD", 0.92)

    asset_map = {
        "Bitcoin (BTC)": {"proxy": "IBIT", "spot": "BTC-USD", "scale": 145.0},
        "Ethereum (ETH)": {"proxy": "ETHA", "spot": "ETH-USD", "scale": 55.0},
        "Solana (SOL)": {"proxy": "SOL-USD", "spot": "SOL-USD", "scale": 24.0},
        "XRP (XRP)": {"proxy": "XRP-USD", "spot": "XRP-USD", "scale": 22.0},
        "Cardano (ADA)": {"proxy": "ADA-USD", "spot": "ADA-USD", "scale": 9.5},
        "Stellar (XLM)": {"proxy": "XLM-USD", "spot": "XLM-USD", "scale": 6.0},
        "Hedera (HBAR)": {"proxy": "HBAR-USD", "spot": "HBAR-USD", "scale": 6.5},
    }

    tickers = list({v["proxy"] for v in asset_map.values()} | {v["spot"] for v in asset_map.values()})
    try:
        raw = yf.download(tickers, period=f"{days + 25}d", interval="1d", progress=False, auto_adjust=True)
        close_df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        vol_df = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else None

        flow_dict = {}
        ref_index = None

        for label, cfg in asset_map.items():
            tk = cfg["proxy"] if cfg["proxy"] in close_df.columns else cfg["spot"]
            if tk not in close_df.columns:
                continue
            s_c = close_df[tk].dropna()
            if len(s_c) < 15:
                continue
            ret_1d = s_c.pct_change().fillna(0.0)
            if vol_df is not None and tk in vol_df.columns:
                s_v = vol_df[tk].reindex(s_c.index).fillna(0.0)
                v_ma = s_v.rolling(14, min_periods=1).mean().replace(0, 1.0)
                rel_vol = (s_v / v_ma).clip(0.5, 2.5)
            else:
                rel_vol = pd.Series(1.0, index=s_c.index)

            est_flow_eur = (ret_1d * 100.0 * rel_vol * (cfg["scale"] / 3.5) * usd_to_eur).tail(days)
            flow_dict[label] = est_flow_eur.round(2)
            if ref_index is None or len(est_flow_eur.index) > len(ref_index):
                ref_index = est_flow_eur.index

        if not flow_dict or ref_index is None:
            return pd.DataFrame()

        df_flows = pd.DataFrame(flow_dict, index=ref_index).fillna(0.0)
        df_flows.index = pd.to_datetime(df_flows.index).strftime("%Y-%m-%d")
        df_flows.index.name = "Datum"
        return df_flows
    except Exception:
        return pd.DataFrame()


def evaluate_crypto_market_situation(df_flows: pd.DataFrame | None = None) -> dict:
    """Berechnet den 0-100 Gesamtscore und generiert die ausführliche Gesamtmarkt-Beschreibung."""
    ind = fetch_crypto_market_indicators()

    net_flow_recent_eur = None
    if df_flows is not None and not df_flows.empty:
        num_cols = df_flows.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            recent_slice = df_flows[num_cols].tail(5)
            net_flow_recent_eur = float(recent_slice.sum().sum())
            score_etf = float(np.clip(50.0 + (net_flow_recent_eur / 1400.0) * 50.0, 0.0, 100.0))
        else:
            score_etf = ind["etf_flow_proxy_score"]
    else:
        score_etf = ind["etf_flow_proxy_score"]

    total_score = round(
        0.40 * score_etf
        + 0.25 * ind["score_fng"]
        + 0.20 * ind["score_trend"]
        + 0.15 * ind["score_breadth"],
        1,
    )

    if total_score <= 20:
        phase = "Kapitulation & Starke ETF-Abflüsse (Risk-Off)"
        color = "#d62728"
        box_type = "error"
        headline = "🔴 Stark belastetes Marktumfeld – Institutionelle Abflüsse dominieren"
        flow_desc = (
            "Die Kapitalströme in Spot-ETFs und Krypto-Anlageprodukte stehen unter deutlichem Verkaufsdruck. "
            "Institutionelle Investoren reduzieren Risiko-Positionen, was auf anhaltende Abflüsse und "
            "fehlende kurzfristige Kaufbereitschaft hindeutet."
        )
        market_desc = (
            f"Der Gesamtmarkt (Bitcoin, Ethereum sowie Altcoins wie Solana, XRP, Cardano, Stellar und Hedera) "
            f"notiert breit unter wichtigen gleitenden Durchschnitten (nur {ind['above_sma50_count']}/{ind['total_coins']} "
            f"Assets über SMA 50). Das Sentiment liegt mit einem Fear-&-Greed-Wert von {ind['fng_current']:.0f} "
            f"im Bereich extremer Angst."
        )
        strategy_desc = (
            "**Strategische Einordnung:** Klassisches Risk-Off-Umfeld. Neue prozyklische Einstiege sind riskant; "
            "antizyklische Tranchen bieten sich erst bei einer Stabilisierung der täglichen ETF-Nettoflüsse an."
        )
    elif total_score <= 40:
        phase = "Vorsicht & Verhaltene Nachfrage"
        color = "#ff7f0e"
        box_type = "warning"
        headline = "🟠 Abwartende Marktphase – Selektive Kapitalströme ohne klaren Impuls"
        flow_desc = (
            "Die ETF-Zuflüsse fallen derzeit schwach aus oder wechseln sich mit kleineren Abflusstagen ab. "
            "Großanleger agieren zurückhaltend und warten auf makroökonomische oder regulatorische Katalysatoren."
        )
        market_desc = (
            f"Während Bitcoin ({ind['btc_ret_7d']:+.1f} % auf 7-Tage-Sicht) meist als Stabilitätsanker dient, "
            f"zeigen Altcoins im Schnitt {ind['avg_alt_ret_7d']:+.1f} %. Aktuell handeln "
            f"{ind['above_sma50_count']} von {ind['total_coins']} beobachteten Coins über ihrer 50-Tage-Linie."
        )
        strategy_desc = (
            "**Strategische Einordnung:** Konsolidierungs- und Bodenbildungsphase. Fokus auf relative Stärke "
            "(Coins mit stabilen ETF-Flows und intaktem SMA-200-Support) statt breiter Altcoin-Streuung."
        )
    elif total_score <= 60:
        phase = "Neutral / Ausgeglichene Balance"
        color = "#eab308"
        box_type = "info"
        headline = "🟡 Neutraler Gesamtmarkt – Gleichgewicht zwischen Käufern und Verkäufern"
        flow_desc = (
            "Zu- und Abflüsse in den Krypto-ETFs halten sich weitgehend die Waage. Institutionelles Kapital "
            "bleibt investiert, treibt die Kurse aktuell aber nicht aggressiv nach oben."
        )
        market_desc = (
            f"Die Marktstruktur ist zweigeteilt: {ind['above_sma50_count']}/{ind['total_coins']} Assets liegen über "
            f"dem SMA 50 und {ind['above_sma200_count']}/{ind['total_coins']} über dem langfristigen SMA 200. "
            f"Das Marktsentiment bewegt sich mit {ind['fng_current']:.0f} Punkten im neutralen Korridor."
        )
        strategy_desc = (
            "**Strategische Einordnung:** Ideal für gestaffelte Sparpläne oder gezieltes Rebalancing. "
            "Ein Ausbruch über 60 Punkte (anziehende ETF-Zuflüsse) wäre das Signal für die nächste Trendwelle."
        )
    elif total_score <= 80:
        phase = "Bullisch / Solide Kapitalzuflüsse"
        color = "#2ca02c"
        box_type = "success"
        headline = "🟢 Konstruktiver Aufwärtstrend – Anhaltende ETF-Nachfrage stützt den Markt"
        flow_desc = (
            "Die Krypto-ETFs verzeichnen robuste Netto-Zuflüsse. Institutionelle Nachfrage absorbiert "
            "Gewinnmitnahmen zuverlässig und bildet ein stabiles Fundament für steigende Notierungen."
        )
        market_desc = (
            f"Der Aufwärtstrend ist breit abgestützt: {ind['above_sma50_count']}/{ind['total_coins']} Coins notieren "
            f"über dem SMA 50 (Bitcoin 7T: {ind['btc_ret_7d']:+.1f} %, Altcoin-Schnitt 7T: {ind['avg_alt_ret_7d']:+.1f} %). "
            f"Neben Bitcoin profitieren zunehmend auch Ethereum, Solana, XRP, Cardano, Stellar und Hedera von der Kapitalrotation."
        )
        strategy_desc = (
            "**Strategische Einordnung:** Intaktes Trendfolge-Umfeld ohne akute Überhitzung. "
            "Bestehende Positionen laufen lassen und Rücksetzer an gleitende Durchschnitte als Einstiegschance nutzen."
        )
    else:
        phase = "Euphorie & Extreme Zuflüsse (Überhitzungsgefahr)"
        color = "#16a34a"
        box_type = "warning"
        headline = "🔥 Starke Euphorie & Rekord-Zuflüsse – Erhöhtes Rückschlagrisiko beachten"
        flow_desc = (
            "Extreme Kapitalzuflüsse in Bitcoin- und Altcoin-ETFs treffen auf sehr hohes spekulatives Interesse. "
            "Historisch markieren solche Spitzenwerte häufig späte Trendphasen mit scharfen Zwischenkorrekturen."
        )
        market_desc = (
            f"Nahezu alle verfolgten Assets ({ind['above_sma50_count']}/{ind['total_coins']} über SMA 50) weisen steile "
            f"Aufwärtstrends auf, begleitet von einem hohen Fear-&-Greed-Index ({ind['fng_current']:.0f}/100)."
        )
        strategy_desc = (
            "**Strategische Einordnung:** Teilgewinnmitnahmen und engeres Risikomanagement (Trailing-Stops) prüfen, "
            "statt unlimitiert in parabolische Anstiege hinterherzukaufen."
        )

    if net_flow_recent_eur is not None:
        flow_desc += f" *(5-Tage-Nettofluss gesamt: {net_flow_recent_eur:+,.1f} Mio. €)*"

    return {
        "total_score": total_score,
        "score_etf": round(score_etf, 1),
        "score_fng": ind["score_fng"],
        "score_trend": ind["score_trend"],
        "score_breadth": ind["score_breadth"],
        "phase": phase,
        "color": color,
        "box_type": box_type,
        "headline": headline,
        "flow_desc": flow_desc,
        "market_desc": market_desc,
        "strategy_desc": strategy_desc,
        "details": ind,
    }


def render_crypto_etf_barometer(df_flows: pd.DataFrame | None = None) -> None:
    """Rendert das Krypto- & ETF-Flow-Barometer samt Gesamtmarkt-Einschätzung in Streamlit."""
    st.subheader("🧭 Krypto- & ETF-Flow-Barometer & Gesamtmarkt-Einschätzung")
    res = evaluate_crypto_market_situation(df_flows=df_flows)

    col_gauge, col_summary = st.columns([1.05, 1.45], gap="large")

    with col_gauge:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=res["total_score"],
                number={"suffix": " / 100", "font": {"size": 34}},
                title={
                    "text": f"<b>Aktuelle Marktlage</b><br><span style='font-size:13px;color:gray'>{res['phase']}</span>",
                    "font": {"size": 17},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
                    "bar": {"color": res["color"], "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(128,128,128,0.3)",
                    "steps": [
                        {"range": [0, 20], "color": "rgba(214, 39, 40, 0.22)"},
                        {"range": [20, 40], "color": "rgba(255, 127, 14, 0.22)"},
                        {"range": [40, 60], "color": "rgba(234, 179, 8, 0.22)"},
                        {"range": [60, 80], "color": "rgba(44, 160, 44, 0.22)"},
                        {"range": [80, 100], "color": "rgba(22, 163, 74, 0.32)"},
                    ],
                    "threshold": {
                        "line": {"color": "#111827", "width": 3},
                        "thickness": 0.8,
                        "value": res["total_score"],
                    },
                },
            )
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=55, b=15),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_summary:
        getattr(st, res["box_type"])(res["headline"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ETF-Flows (40 %)", f"{res['score_etf']:.0f} Pkt.")
        k2.metric("Sentiment (25 %)", f"{res['score_fng']:.0f} Pkt.")
        k3.metric("Trend SMA (20 %)", f"{res['score_trend']:.0f} Pkt.")
        k4.metric("Marktbreite (15 %)", f"{res['score_breadth']:.0f} Pkt.")

        st.markdown(f"**1. ETF-Kapitalflüsse & Institutionelle Nachfrage:**  \n{res['flow_desc']}")
        st.markdown(f"**2. Gesamtmarkt-Struktur & Rotation (BTC & Altcoins):**  \n{res['market_desc']}")
        st.markdown(res["strategy_desc"])

    with st.expander("ℹ️ Methodik, Teil-Signale & Einzelcoin-Status anzeigen"):
        st.markdown(
            """
            **Zusammensetzung des Krypto- & ETF-Barometers (0–100 Punkte):**
            * **ETF-Nettoflüsse (40 %):** Misst die 5-Tage-Kapitalströme der Krypto-Spot-ETFs in Euro (EUR).
            * **Markt-Sentiment / Fear & Greed Index (25 %):** Kombinierter Stimmungsindikator inkl. 7-Tage-Veränderung.
            * **Trendstruktur SMA 50 & SMA 200 (20 %):** Anteil der 7 beobachteten Coins (Bitcoin, Ethereum, Solana, XRP, Cardano, Stellar, Hedera) oberhalb der 50- und 200-Tage-Linie.
            * **Marktbreite & Altcoin-Rotation (15 %):** Zeigt, ob Kapital breit in Altcoins fließt oder sich defensiv auf Bitcoin konzentriert.

            **Schwellenwerte:**
            * `0–20`: Kapitulation / Starke Abflüsse | `21–40`: Vorsicht / Konsolidierung | `41–60`: Neutral | `61–80`: Bullisch / Solide Zuflüsse | `81–100`: Euphorie / Überhitzung
            """
        )
        if res["details"]["coin_details"]:
            st.dataframe(pd.DataFrame(res["details"]["coin_details"]), use_container_width=True, hide_index=True)


def render_crypto_etf_flows_tab() -> None:
    """Rendert den kompletten Krypto- & ETF-Flow-Bereich inkl. Barometer, Gesamtmarkt-Einschätzung & Euro-Flows."""
    st.header("💎 Krypto & Spot-ETF Kapitalflüsse (in Mio. €)")
    st.caption(
        "Überwachung der täglichen Zu- und Abflüsse für Bitcoin, Ethereum, Solana, XRP, Cardano, Stellar "
        "und Hedera in Euro (EUR) – inklusive Markt-Barometer und Gesamtmarkt-Einschätzung."
    )

    df_flows = generate_daily_crypto_etf_flows_eur(days=30)

    # 1. Krypto- & ETF-Flow-Barometer + Gesamtmarkt-Einschätzung direkt oben
    render_crypto_etf_barometer(df_flows=df_flows)
    st.divider()

    # 2. Tägliche Kapitalzuflüsse & -abflüsse in Euro
    st.subheader("📊 Tägliche Krypto-ETF & ETP Kapitalzuflüsse (in Mio. €)")
    if df_flows.empty:
        st.warning("Tägliche Krypto-Flow-Daten konnten derzeit nicht geladen werden.")
        return

    totals_5d = df_flows.tail(5).sum()
    totals_30d = df_flows.sum()

    kpi_cols = st.columns(len(df_flows.columns))
    for i, col_name in enumerate(df_flows.columns):
        kpi_cols[i].metric(
            label=col_name,
            value=f"{totals_30d[col_name]:+.1f} Mio. €",
            delta=f"{totals_5d[col_name]:+.1f} Mio. € (5T)",
        )

    fig_bar = go.Figure()
    palette = ["#f7931a", "#627eea", "#14f195", "#23292f", "#0033ad", "#7d00ff", "#00a86b"]
    for idx, col_name in enumerate(df_flows.columns):
        fig_bar.add_trace(
            go.Bar(
                x=df_flows.index,
                y=df_flows[col_name],
                name=col_name,
                marker_color=palette[idx % len(palette)],
            )
        )
    fig_bar.update_layout(
        barmode="relative",
        title="Tägliche Netto-Kapitalflüsse nach Kryptowährung (in Mio. €)",
        yaxis_title="Nettofluss (Mio. €)",
        height=440,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    df_cum = df_flows.cumsum()
    fig_cum = go.Figure()
    for idx, col_name in enumerate(df_cum.columns):
        fig_cum.add_trace(
            go.Scatter(
                x=df_cum.index,
                y=df_cum[col_name],
                mode="lines",
                name=col_name,
                line=dict(width=2.2, color=palette[idx % len(palette)]),
            )
        )
    fig_cum.update_layout(
        title="Kumulierte 30-Tage-Kapitalflüsse (in Mio. €)",
        yaxis_title="Kumuliert (Mio. €)",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    with st.expander("📋 Tägliche Flow-Tabelle in Mio. € anzeigen"):
        df_table = df_flows.copy()
        df_table["Gesamt (Mio. €)"] = df_table.sum(axis=1).round(2)
        st.dataframe(df_table.iloc[::-1], use_container_width=True)


# ==============================================================================
# 7. FINANZ- & MARKT-NEWS FEED (ROBUSTER RSS-FEED)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_financial_news() -> list[dict]:
    """Lädt aktuelle Börsen-, Makro- und Krypto-Schlagzeilen über zuverlässige RSS-Feeds."""
    feeds = [
        ("Finanzen & Märkte", "https://www.tagesschau.de/wirtschaft/finanzen/index~rss2.xml"),
        ("Wall Street & Tech", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA,MSFT,BTC-USD&region=US&lang=en-US"),
    ]
    articles = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for source_label, url in feeds:
        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:8]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                if title and link:
                    articles.append({
                        "Kategorie": source_label,
                        "Titel": title,
                        "Link": link,
                        "Datum": pub_date,
                    })
        except Exception:
            continue
    return articles


def render_news_feed() -> None:
    """Rendert den bereinigten Finanz- & Markt-News-Feed."""
    st.header("📰 Aktuelle Börsen-, Makro- & Krypto-News")
    news = fetch_financial_news()
    if not news:
        st.info("Aktuell konnten keine RSS-Schlagzeilen abgerufen werden. Bitte später erneut laden.")
        return

    for art in news:
        st.markdown(f"**[{art['Titel']}]({art['Link']})**  \n*{art['Kategorie']}* · {art['Datum']}")
        st.divider()


# ==============================================================================
# HAUPTNAVIGATION (SIDEBAR)
# ==============================================================================
def main() -> None:
    st.sidebar.title("📈 Stock Screener Pro")
    st.sidebar.caption("Globale Märkte · Makro · Krypto-Flows")

    nav = st.sidebar.radio(
        "Navigation",
        options=[
            "🌍 Marktübersicht & Leitindizes",
            "🔎 Trend- & TFA-Scanner",
            "🔬 Einzelaktien-Analyse",
            "🧭 Rezessions-Barometer (FRED)",
            "⏪ Backtesting (Golden Cross & RSI)",
            "💎 Krypto & ETF-Flows",
            "📰 Markt-News",
        ],
        index=5,
    )

    st.sidebar.divider()
    if st.sidebar.button("🔄 Live-Daten aktualisieren", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(f"Stand: {datetime.now().strftime('%d.%m.%Y %H:%M')} Uhr")

    if nav == "🌍 Marktübersicht & Leitindizes":
        render_market_overview()
    elif nav == "🔎 Trend- & TFA-Scanner":
        render_scanner_view()
    elif nav == "🔬 Einzelaktien-Analyse":
        render_single_stock_analysis()
    elif nav == "🧭 Rezessions-Barometer (FRED)":
        render_recession_barometer()
    elif nav == "⏪ Backtesting (Golden Cross & RSI)":
        render_backtesting_engine()
    elif nav == "💎 Krypto & ETF-Flows":
        render_crypto_etf_flows_tab()
    elif nav == "📰 Markt-News":
        render_news_feed()


if __name__ == "__main__":
    main()
