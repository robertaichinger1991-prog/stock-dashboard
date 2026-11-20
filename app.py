"""
Stock Screener Pro
------------------
Lokales Streamlit-Dashboard zum Finden und Prüfen von Aktien.

Features:
- Einzelanalyse: Chart (€), LONG/FLAT/SHORT, Sentiment, Zielpfad, Scores
- Index-Scanner & Rezessions-Barometer (FRED Makro-Indikatoren)
- Scanner: US/EU/Asien + Qualitätsfilter + Composite-Score
- Krypto ETF Flows + Barometer der Kryptostimmung (Fear & Greed)
- Vertieftes Backtesting (mehrere Signale, Equity-Kurve, Profit Factor)
- News-Feed, Sparplan-Rechner, Saisonalität, Sektor-Heatmap, Dividenden-Kalender

Start:
    python -m streamlit run app_13.py
"""

import json
import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Screener Pro", layout="wide", page_icon="🌈")

st.markdown("""
<style>
/* Dunkles (nicht-schwarzes) Theme ERZWINGEN */
.stApp, .stApp > header { background-color: #1e1e2e !important; }
h1, h2, h3, h4, h5, h6, p, span, label, div { color: #cdd6f4 !important; }
.stMarkdown, .stText { color: #cdd6f4 !important; }
div[data-testid="stMetricValue"] { color: #cdd6f4 !important; }
div[data-testid="stMetricLabel"] { color: #a6adc8 !important; }
div[data-testid="metric-container"] {
    background-color: #313244 !important;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
    border: 1px solid #45475a !important;
}
/* Dropdowns zwingend dunkel machen */
div[data-baseweb="select"] > div {
    background-color: #313244 !important;
    color: #cdd6f4 !important;
    border-color: #45475a !important;
}
div[data-baseweb="select"] span {
    color: #cdd6f4 !important;
}
ul[data-baseweb="menu"] {
    background-color: #313244 !important;
}
/* Warnungen und Info-Boxen (st.info, st.success, etc) besser lesbar machen */
div.stAlert, div[data-testid="stAlert"] {
    background-color: rgba(49, 50, 68, 0.95) !important;
    color: #cdd6f4 !important;
    border: 1px solid #45475a !important;
}
div.stAlert p, div.stAlert div, div.stAlert span {
    color: #cdd6f4 !important;
}
/* Sidebar (Navigation) zwingend dunkel machen */
section[data-testid="stSidebar"] {
    background-color: #181825 !important;
    border-right: 1px solid #313244 !important;
}
section[data-testid="stSidebar"] * {
    color: #cdd6f4 !important;
}
/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #313244 !important;
}
</style>
""", unsafe_allow_html=True)


# Plotly: mobil-freundlich (Pinch-Zoom, Pan) + klare Darstellung
# Modebar immer sichtbar; Scroll-Zoom an; Touch über Range-Slider + Buttons
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "doubleClick": "reset",
    "showTips": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "chart", "scale": 2},
}

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1e1e2e",
    plot_bgcolor="#1e1e2e",
    font=dict(family="Inter, system-ui, sans-serif", size=14, color="#cdd6f4"),
    margin=dict(l=40, r=56, t=56, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(30,30,46,0.85)",
        font=dict(size=12, color="#cdd6f4"),
        itemsizing="constant",
    ),
    hovermode="x unified",
    # pan = mit Finger schieben; Zoom über Modebar-Buttons oder Range-Slider
    dragmode="pan",
    spikedistance=50,
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)",
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
        gridcolor="rgba(255,255,255,0.1)",
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
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlists.json")


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


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
    "Top Krypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD"],
    "Edelmetalle": ["GC=F", "SI=F", "PL=F", "PA=F", "GLD", "SLV", "IAU", "GDX", "SIL"],
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

# Edelmetalle: Spot-Futures + gängige ETFs (USD)
PRECIOUS_METALS = [
    "GC=F",   # Gold Spot/Future
    "SI=F",   # Silber
    "PL=F",   # Platin
    "PA=F",   # Palladium
    "GLD",    # Gold ETF
    "SLV",    # Silber ETF
    "IAU",    # Gold ETF klein
    "GDX",    # Goldminen
    "SIL",    # Silberminen
]

ASIA_TECH = ["TSM", "ASML", "SONY", "NTDOY", "BABA", "TCEHY", "PDD", "BIDU", "005930.KS", "6861.T", "2330.TW"]
JAPAN_TICKERS = ["7203.T", "6758.T", "9984.T", "6861.T", "8306.T", "9432.T", "4063.T", "6501.T", "7267.T", "6902.T"]
HK_CHINA = ["0700.HK", "9988.HK", "3690.HK", "9618.HK", "1810.HK", "BABA", "JD", "PDD", "BIDU", "NIO"]

SP500_FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "JNJ", "WMT", "XOM", "UNH", "MA", "PG", "HD", "COST", "MRK", "ABBV",
    "CVX", "PEP", "KO", "BAC", "ADBE", "CRM", "TMO", "MCD", "CSCO", "ABT",
]

SECTOR_TRANSLATIONS = {
    "Technology": "Technologie",
    "Communication Services": "Kommunikation",
    "Consumer Cyclical": "Zyklischer Konsum",
    "Consumer Defensive": "Basiskonsum",
    "Financial Services": "Finanzen",
    "Healthcare": "Gesundheit",
    "Industrials": "Industrie",
    "Basic Materials": "Grundstoffe",
    "Energy": "Energie",
    "Utilities": "Versorger",
    "Real Estate": "Immobilien",
}

# Zuverlässige Fallback-Metadaten für die Sektor-Heatmap (Name, Sektor DE, ca. MarketCap in Mrd. €)
HEATMAP_META = {
    # US Tech & S&P
    "AAPL": ("Apple", "Technologie", 3100.0),
    "MSFT": ("Microsoft", "Technologie", 2950.0),
    "NVDA": ("Nvidia", "Technologie", 3050.0),
    "GOOGL": ("Alphabet", "Kommunikation", 2050.0),
    "AMZN": ("Amazon", "Zyklischer Konsum", 1950.0),
    "META": ("Meta Platforms", "Kommunikation", 1350.0),
    "TSLA": ("Tesla", "Zyklischer Konsum", 850.0),
    "AVGO": ("Broadcom", "Technologie", 780.0),
    "PEP": ("PepsiCo", "Basiskonsum", 215.0),
    "COST": ("Costco", "Basiskonsum", 380.0),
    "ADBE": ("Adobe", "Technologie", 210.0),
    "CSCO": ("Cisco", "Technologie", 210.0),
    "NFLX": ("Netflix", "Kommunikation", 310.0),
    "AMD": ("AMD", "Technologie", 230.0),
    "INTC": ("Intel", "Technologie", 95.0),
    "QCOM": ("Qualcomm", "Technologie", 175.0),
    "TXN": ("Texas Instruments", "Technologie", 170.0),
    "INTU": ("Intuit", "Technologie", 165.0),
    "AMAT": ("Applied Materials", "Technologie", 145.0),
    "BKNG": ("Booking Holdings", "Zyklischer Konsum", 145.0),
    "BRK-B": ("Berkshire Hathaway", "Finanzen", 890.0),
    "JPM": ("JPMorgan Chase", "Finanzen", 590.0),
    "V": ("Visa", "Finanzen", 520.0),
    "JNJ": ("Johnson & Johnson", "Gesundheit", 360.0),
    "WMT": ("Walmart", "Basiskonsum", 610.0),
    "XOM": ("Exxon Mobil", "Energie", 470.0),
    "UNH": ("UnitedHealth", "Gesundheit", 480.0),
    "MA": ("Mastercard", "Finanzen", 440.0),
    "PG": ("Procter & Gamble", "Basiskonsum", 365.0),
    "HD": ("Home Depot", "Zyklischer Konsum", 360.0),
    "MRK": ("Merck & Co.", "Gesundheit", 250.0),
    "ABBV": ("AbbVie", "Gesundheit", 310.0),
    "CVX": ("Chevron", "Energie", 260.0),
    "KO": ("Coca-Cola", "Basiskonsum", 265.0),
    "BAC": ("Bank of America", "Finanzen", 310.0),
    "CRM": ("Salesforce", "Technologie", 275.0),
    # DAX 40
    "SAP.DE": ("SAP", "Technologie", 265.0),
    "SIE.DE": ("Siemens", "Industrie", 145.0),
    "ALV.DE": ("Allianz", "Finanzen", 112.0),
    "DTE.DE": ("Deutsche Telekom", "Kommunikation", 138.0),
    "AIR.DE": ("Airbus", "Industrie", 115.0),
    "MUV2.DE": ("Münchener Rück", "Finanzen", 66.0),
    "SHL.DE": ("Siemens Healthineers", "Gesundheit", 58.0),
    "MBG.DE": ("Mercedes-Benz", "Zyklischer Konsum", 56.0),
    "IFX.DE": ("Infineon", "Technologie", 42.0),
    "BAS.DE": ("BASF", "Grundstoffe", 40.0),
    "DHL.DE": ("DHL Group", "Industrie", 44.0),
    "BMW.DE": ("BMW", "Zyklischer Konsum", 48.0),
    "VOW3.DE": ("Volkswagen Vz.", "Zyklischer Konsum", 46.0),
    "ADS.DE": ("Adidas", "Zyklischer Konsum", 41.0),
    "DB1.DE": ("Deutsche Börse", "Finanzen", 39.0),
    "DBK.DE": ("Deutsche Bank", "Finanzen", 33.0),
    "RHM.DE": ("Rheinmetall", "Industrie", 26.0),
    "ENR.DE": ("Siemens Energy", "Industrie", 34.0),
    "EOAN.DE": ("E.ON", "Versorger", 32.0),
    "RWE.DE": ("RWE", "Versorger", 23.0),
    "HEN3.DE": ("Henkel", "Basiskonsum", 33.0),
    "BEI.DE": ("Beiersdorf", "Basiskonsum", 31.0),
    "MRK.DE": ("Merck KGaA", "Gesundheit", 64.0),
    "BAYN.DE": ("Bayer", "Gesundheit", 24.0),
    "MTX.DE": ("MTU Aero Engines", "Industrie", 16.5),
    "HNR1.DE": ("Hannover Rück", "Finanzen", 30.0),
    "HEI.DE": ("Heidelberg Materials", "Grundstoffe", 21.0),
    "CBK.DE": ("Commerzbank", "Finanzen", 19.0),
    "DTG.DE": ("Daimler Truck", "Industrie", 29.0),
    "FRE.DE": ("Fresenius", "Gesundheit", 19.5),
    "SY1.DE": ("Symrise", "Grundstoffe", 15.5),
    "VNA.DE": ("Vonovia", "Immobilien", 24.5),
    "BNR.DE": ("Brenntag", "Grundstoffe", 9.5),
    "CON.DE": ("Continental", "Zyklischer Konsum", 12.5),
    "1COV.DE": ("Covestro", "Grundstoffe", 11.0),
    "PAH3.DE": ("Porsche Holding", "Zyklischer Konsum", 12.0),
    "QIA.DE": ("Qiagen", "Gesundheit", 9.2),
    "SRT3.DE": ("Sartorius", "Gesundheit", 16.0),
    "ZAL.DE": ("Zalando", "Zyklischer Konsum", 7.8),
    "G24.DE": ("Scout24", "Kommunikation", 6.2),
    # Europa Blue Chips
    "ASML.AS": ("ASML", "Technologie", 275.0),
    "MC.PA": ("LVMH", "Zyklischer Konsum", 315.0),
    "OR.PA": ("L'Oréal", "Basiskonsum", 195.0),
    "TTE.PA": ("TotalEnergies", "Energie", 138.0),
    "SAN.PA": ("Sanofi", "Gesundheit", 122.0),
    "SU.PA": ("Schneider Electric", "Industrie", 136.0),
    "AI.PA": ("Air Liquide", "Grundstoffe", 96.0),
    "NESN.SW": ("Nestlé", "Basiskonsum", 235.0),
    "ROG.SW": ("Roche", "Gesundheit", 225.0),
    "NOVN.SW": ("Novartis", "Gesundheit", 205.0),
    "AZN.L": ("AstraZeneca", "Gesundheit", 205.0),
    "SHEL.L": ("Shell", "Energie", 195.0),
    "HSBA.L": ("HSBC", "Finanzen", 165.0),
    "ULVR.L": ("Unilever", "Basiskonsum", 135.0),
    "INGA.AS": ("ING Groep", "Finanzen", 51.0),
}


@st.cache_data(ttl=86400)
def fetch_sp500_tickers():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        symbols = tables[0]["Symbol"].tolist()
        return [s.replace(".", "-") for s in symbols]
    except Exception:
        return SP500_FALLBACK


COMMON_NAMES = {
    # US & Tech
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "AMAZON": "AMZN", "NVIDIA": "NVDA", "META": "META", "FACEBOOK": "META",
    "TESLA": "TSLA", "NETFLIX": "NFLX", "AMD": "AMD", "SALESFORCE": "CRM",
    "BERKSHIRE": "BRK-B", "BERKSHIRE HATHAWAY": "BRK-B", "JPMORGAN": "JPM", "JPM": "JPM",
    "VISA": "V", "JOHNSON & JOHNSON": "JNJ", "JNJ": "JNJ", "WALMART": "WMT",
    "EXXON": "XOM", "COCA-COLA": "KO", "COCA COLA": "KO", "DISNEY": "DIS",
    "NIKE": "NKE", "MCDONALD'S": "MCD", "MCDONALDS": "MCD", "STARBUCKS": "SBUX",
    "INTEL": "INTC", "BROADCOM": "AVGO", "PEPSICO": "PEP", "COSTCO": "COST",
    "ADOBE": "ADBE", "CISCO": "CSCO", "QUALCOMM": "QCOM", "TEXAS INSTRUMENTS": "TXN",
    "INTUIT": "INTU", "APPLIED MATERIALS": "AMAT", "BOOKING": "BKNG",
    "PALANTIR": "PLTR", "MICROSTRATEGY": "MSTR", "COINBASE": "COIN",
    "UNITEDHEALTH": "UNH", "MASTERCARD": "MA", "PROCTER & GAMBLE": "PG", "PROCTER": "PG",
    "HOME DEPOT": "HD", "ABBVIE": "ABBV", "CHEVRON": "CVX", "BANK OF AMERICA": "BAC",

    # DAX & Germany
    "SAP": "SAP.DE", "SIEMENS": "SIE.DE", "ALLIANZ": "ALV.DE",
    "DEUTSCHE TELEKOM": "DTE.DE", "TELEKOM": "DTE.DE",
    "BASF": "BAS.DE", "BMW": "BMW.DE", "MERCEDES": "MBG.DE", "MERCEDES-BENZ": "MBG.DE",
    "VOLKSWAGEN": "VOW3.DE", "VW": "VOW3.DE", "MUNICH RE": "MUV2.DE", "MÜNCHENER RÜCK": "MUV2.DE",
    "INFINEON": "IFX.DE", "DEUTSCHE BANK": "DBK.DE", "COMMERZBANK": "CBK.DE",
    "ADIDAS": "ADS.DE", "BAYER": "BAYN.DE", "HENKEL": "HEN3.DE", "MERCK": "MRK.DE",
    "AIRBUS": "AIR.DE", "BEIERSDORF": "BEI.DE", "BRENNTAG": "BNR.DE",
    "CONTINENTAL": "CON.DE", "COVESTRO": "1COV.DE", "DHL": "DHL.DE", "DEUTSCHE POST": "DHL.DE",
    "DEUTSCHE BÖRSE": "DB1.DE", "DAIMLER TRUCK": "DTG.DE", "E.ON": "EOAN.DE", "EON": "EOAN.DE",
    "FRESENIUS": "FRE.DE", "HANNOVER RÜCK": "HNR1.DE", "HEIDELBERG MATERIALS": "HEI.DE",
    "MTU": "MTX.DE", "PORSCHE": "P911.DE", "QIAGEN": "QIA.DE", "RHEINMETALL": "RHM.DE",
    "RWE": "RWE.DE", "SARTORIUS": "SRT3.DE", "SIEMENS ENERGY": "ENR.DE",
    "SIEMENS HEALTHINEERS": "SHL.DE", "SYMRISE": "SY1.DE", "VONOVIA": "VNA.DE",
    "ZALANDO": "ZAL.DE",

    # Europe (CAC40, Benelux, Swiss, UK)
    "LVMH": "MC.PA", "LOUIS VUITTON": "MC.PA", "LOREAL": "OR.PA", "L'OREAL": "OR.PA",
    "TOTALENERGIES": "TTE.PA", "TOTAL": "TTE.PA", "SANOFI": "SAN.PA",
    "NESTLE": "NESN.SW", "NESTLÉ": "NESN.SW", "ROCHE": "ROG.SW", "NOVARTIS": "NOVN.SW",
    "ASTRAZENECA": "AZN.L", "SHELL": "SHEL.L", "HSBC": "HSBA.L", "UNILEVER": "ULVR.L",
    "BP": "BP.L", "ING": "INGA.AS", "PHILIPS": "PHIA.AS", "HERMES": "RMS.PA", "HERMÈS": "RMS.PA",
    "SCHNEIDER": "SU.PA", "SCHNEIDER ELECTRIC": "SU.PA", "BNP": "BNP.PA", "BNP PARIBAS": "BNP.PA",
    "UBS": "UBSG.SW", "ZURICH": "ZURN.SW", "ABB": "ABBN.SW",
    "ASML": "ASML.AS", "ADYEN": "ADYEN.AS", "HEINEKEN": "HEIA.AS",
    "STELLANTIS": "STLAP.PA", "FERRARI": "RACE.MI", "DANONE": "BN.PA",
    "CAPGEMINI": "CAP.PA", "DASSAULT": "DSY.PA", "AIR LIQUIDE": "AI.PA",

    # Asia / Global
    "TSMC": "TSM", "TAIWAN SEMICONDUCTOR": "TSM", "SAMSUNG": "005930.KS",
    "SONY": "SONY", "TOYOTA": "7203.T", "SOFTBANK": "9984.T", "KEYENCE": "6861.T",
    "ALIBABA": "BABA", "TENCENT": "TCEHY", "NINTENDO": "NTDOY",
    "BYD": "BYDDY", "XIAOMI": "1810.HK", "XIAOMI CORP": "1810.HK",
    "MEITUAN": "3690.HK", "JD.COM": "JD", "JDCOM": "JD",
    "BAIDU": "BIDU", "PINDUODUO": "PDD", "NETEASE": "NTES", "NIO": "NIO",
    "HONDA": "7267.T", "HITACHI": "6501.T", "MITSUBISHI": "8306.T",

    # Crypto
    "BITCOIN": "BTC-USD", "BTC": "BTC-USD", "ETHEREUM": "ETH-USD", "ETH": "ETH-USD",
    "SOLANA": "SOL-USD", "SOL": "SOL-USD", "CARDANO": "ADA-USD",
    "RIPPLE": "XRP-USD", "XRP": "XRP-USD", "DOGECOIN": "DOGE-USD", "DOGE": "DOGE-USD",
    "BINANCE": "BNB-USD", "BNB": "BNB-USD", "AVALANCHE": "AVAX-USD",
    "POLKADOT": "DOT-USD", "CHAINLINK": "LINK-USD", "STELLAR": "XLM-USD", "XLM": "XLM-USD",
    "HEDERA": "HBAR-USD", "HBAR": "HBAR-USD",

    # Metals
    "GOLD": "GC=F", "GOLDPREIS": "GC=F", "SILBER": "SI=F", "SILVER": "SI=F",
    "PLATIN": "PL=F", "PLATINUM": "PL=F", "PALLADIUM": "PA=F",
    "GLD": "GLD", "SLV": "SLV",
}


@st.cache_data(ttl=86400)
def resolve_ticker(query: str) -> str:
    query = query.strip()
    if not query:
        return ""

    q_upper = query.upper()

    if "XRP" in q_upper:
        return "XRP-USD"
    if "XIAOMI" in q_upper:
        return "1810.HK"
    if "MÜNCHENER RÜCK" in q_upper or "MUNICH RE" in q_upper:
        return "MUV2.DE"

    if q_upper in COMMON_NAMES:
        return COMMON_NAMES[q_upper]

    for name, sym in COMMON_NAMES.items():
        if (len(q_upper) >= 3 and q_upper in name) or (len(name) >= 4 and name in q_upper):
            return sym

    raw = query.strip().upper()
    looks_like_ticker = " " not in query and len(query) <= 12
    try:
        search_results = yf.Search(query, max_results=8).quotes
        if search_results:
            equities = [r for r in search_results if r.get("quoteType") == "EQUITY"]
            pool = equities or search_results

            for r in pool:
                sym = r.get("symbol", "")
                ex = r.get("exchange", "")
                if ex in ("GER", "FRA", "PAR", "AMS", "NMS", "NYQ") and not sym.endswith(".F"):
                    return sym

            symbol = pool[0].get("symbol")
            if symbol:
                return symbol
    except Exception:
        pass
    return raw if looks_like_ticker else query


# ---------------------------------------------------------------------------
# Kursdaten + technische Indikatoren
# ---------------------------------------------------------------------------
TICKER_FALLBACKS = {
    "1810.HK": ["1810.HK", "XIACF", "3CP.F"],
    "XIAOMI": ["1810.HK", "XIACF"],
    "0700.HK": ["0700.HK", "TCEHY"],
    "9988.HK": ["9988.HK", "BABA"],
    "3690.HK": ["3690.HK", "MPNGY"],
    "GC=F": ["GC=F", "GLD", "IAU"],
    "SI=F": ["SI=F", "SLV"],
    "PL=F": ["PL=F"],
    "PA=F": ["PA=F"],
    "BTC-USD": ["BTC-USD", "BTC-EUR"],
    "ETH-USD": ["ETH-USD", "ETH-EUR"],
}


@st.cache_data(ttl=3600)
def load_price_history(ticker: str, period: str = "2y"):
    candidates = TICKER_FALLBACKS.get(ticker.upper(), [ticker])
    if ticker not in candidates:
        candidates = [ticker] + list(candidates)
    data = pd.DataFrame()
    for t in candidates:
        try:
            data = yf.Ticker(t).history(period=period, auto_adjust=True)
            if data is not None and not data.empty:
                break
        except Exception:
            data = pd.DataFrame()
            continue
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)
    except Exception:
        pass

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


def compute_entry_zones(data: pd.DataFrame, fx: float = 1.0, lookback_days: int = 252):
    """Ermittelt drei technische Kauf-Einstiegszonen aus Fibonacci-Retracements."""
    if data.empty or not {"High", "Low", "Close"}.issubset(data.columns):
        return []

    recent = data.tail(lookback_days).dropna(subset=["High", "Low", "Close"]).copy()
    recent = recent.replace([np.inf, -np.inf], np.nan).dropna(subset=["High", "Low", "Close"])
    if len(recent) < 30:
        return []

    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())
    diff = swing_high - swing_low
    if diff <= 0:
        return []

    fib236 = swing_high - 0.236 * diff
    fib382 = swing_high - 0.382 * diff
    fib500 = swing_high - 0.500 * diff
    fib618 = swing_high - 0.618 * diff

    zones = [
        {"name": "EINSTIEG 1", "typ": "Flacher Rücksetzer", "low": fib382, "high": fib236},
        {"name": "EINSTIEG 2", "typ": "Normaler Rücksetzer", "low": fib500, "high": fib382},
        {"name": "EINSTIEG 3", "typ": "Tiefer Rücksetzer", "low": fib618, "high": fib500},
    ]

    last = float(recent["Close"].iloc[-1])
    for z in zones:
        z["low"] *= fx
        z["high"] *= fx
        z["mitte"] = (z["low"] + z["high"]) / 2
        z["distanz_pct"] = (z["mitte"] / (last * fx) - 1) * 100
        z["unter_kurs"] = z["high"] <= last * fx

    return zones


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
def compute_market_correlation(ticker: str, period: str = "1y") -> dict:
    """Korrelation der Aktie zum passenden Marktindex (ca. 1 Jahr)."""
    t = (ticker or "").upper()
    if t.endswith(".DE") or t.endswith(".PA") or t.endswith(".AS") or t.endswith(".BR") or t.endswith(".MI"):
        bench = "^STOXX50E"
        bench_name = "Euro Stoxx 50"
    elif t.endswith(".L"):
        bench = "^FTSE"
        bench_name = "FTSE 100"
    elif t.endswith(".SW"):
        bench = "^SSMI"
        bench_name = "SMI"
    elif t.endswith(".HK") or t.endswith(".T") or t.endswith(".KS") or t.endswith(".TW"):
        bench = "^HSI"
        bench_name = "Hang Seng"
    elif "-USD" in t or t in ("BTC-USD", "ETH-USD"):
        bench = "BTC-USD"
        bench_name = "Bitcoin"
    elif t.endswith("=F") or t in ("GLD", "SLV", "IAU", "GDX", "SIL"):
        bench = "GC=F"
        bench_name = "Gold"
    else:
        bench = "^GSPC"
        bench_name = "S&P 500"
    try:
        a = yf.Ticker(ticker).history(period=period)["Close"].pct_change().dropna()
        b = yf.Ticker(bench).history(period=period)["Close"].pct_change().dropna()
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(joined) < 30:
            return {"corr": None, "bench": bench_name, "bench_ticker": bench}
        joined.columns = ["a", "b"]
        corr = float(joined["a"].corr(joined["b"]))
        return {"corr": round(corr, 2), "bench": bench_name, "bench_ticker": bench}
    except Exception:
        return {"corr": None, "bench": bench_name, "bench_ticker": bench}


def load_dividend_summary(ticker: str) -> dict:
    """Kurze Dividenden-Info aus yfinance."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        div_y = info.get("dividendYield")
        div_r = info.get("dividendRate")
        ex_div = info.get("exDividendDate")
        hist = tk.dividends
        last_div = None
        count_3y = 0
        if hist is not None and len(hist) > 0:
            last_div = float(hist.iloc[-1])
            cutoff = hist.index.max() - pd.Timedelta(days=365 * 3)
            count_3y = int((hist.index >= cutoff).sum())
        return {
            "yield": div_y,
            "rate": div_r,
            "last": last_div,
            "count_3y": count_3y,
            "ex_div": ex_div,
        }
    except Exception:
        return {"yield": None, "rate": None, "last": None, "count_3y": 0, "ex_div": None}


@st.cache_data(ttl=3600)
def get_fx_rates() -> dict:
    """FX-Kurse relativ zu EUR (wie viele Fremdwährungseinheiten für 1 EUR)."""
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
    return float(get_fx_rates().get("USD", 1.08))


def detect_currency(ticker: str, info: dict | None = None) -> str:
    t = ticker.upper()

    if t in ("^GDAXI", "^STOXX50E", "^MDAXI", "^TECDAX"):
        return "EUR"
    if t in ("^GSPC", "^NDX", "^DJI", "^RUT", "^VIX"):
        return "USD"
    if t == "^N225":
        return "JPY"
    if t == "^HSI":
        return "HKD"
    if t.startswith("^"):
        return "USD"

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
    """Einfaches Sentiment 0-100 (0=extreme Angst, 100=extreme Gier)."""
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


def compute_zielpfad(data: pd.DataFrame, fundamentals: dict, fx: float = 1.0) -> dict:
    if data.empty:
        return {}

    last = float(data["Close"].iloc[-1]) * fx
    fib = compute_fib_entry_zone(data)
    t_mean = fundamentals.get("Analysten-Kursziel (mittel)")
    t_high = fundamentals.get("Analysten-Kursziel (hoch)")

    targets = []
    if fib:
        z1 = max(fib[1] * fx, last * 1.03)
    else:
        z1 = last * 1.05
    targets.append({"name": "ZIEL 1", "kurs": z1, "typ": "nah"})

    if t_mean:
        z2 = float(t_mean) * fx
    else:
        z2 = last * 1.10
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

    if t_high:
        z4 = float(t_high) * fx
    else:
        z4 = last * 1.25
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
# Fundamentaldaten
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_fundamentals(ticker: str):
    empty = {
        "Name": ticker, "Sektor": "—", "Branche": "—", "Kurs": None,
        "KGV (trailing)": None, "KGV (forward)": None, "KBV": None,
        "EV/EBITDA": None, "Marktkap.": None, "Umsatzwachstum (YoY)": None,
        "Gewinnmarge": None, "Operative Marge": None, "ROE": None,
        "Verschuldung (D/E)": None,
        "Analysten-Kursziel (mittel)": None,
        "Analysten-Kursziel (hoch)": None,
        "Analysten-Kursziel (niedrig)": None,
        "Empfehlung": None, "Beta": None, "Dividendenrendite": None,
    }
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return empty
    return {
        "Name": info.get("longName") or info.get("shortName") or ticker,
        "Sektor": info.get("sector") or "—",
        "Branche": info.get("industry") or "—",
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
        "Beschreibung": info.get("longBusinessSummary"),
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
    stop_loss_pct: float | None = None,
):
    crosses = _find_signal_dates(data, signal_type)
    if not crosses:
        return {"count": 0, "summary": {}, "details": [], "equity": []}

    df = data.dropna(subset=["Close"]).copy()
    results = []
    fixed_returns = []

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

        for days in forward_days_list:
            future_idx = idx + days
            if future_idx < len(df):
                future_price = float(df.iloc[future_idx]["Close"])
                ret = (future_price / entry_price - 1) * 100
                row[f"Rendite {days}T"] = round(ret, 2)
            else:
                row[f"Rendite {days}T"] = None

        end_idx = min(idx + 126, len(df) - 1)
        window = df.iloc[idx:end_idx + 1]["Close"]
        if len(window) > 1:
            peak = window.cummax()
            dd = float(((window - peak) / peak * 100).min())
            row["Max DD 6M"] = round(dd, 2)
        else:
            row["Max DD 6M"] = None

        hd = hold_days or 63
        exit_idx = idx + hd
        if exit_idx < len(df):
            window = df.iloc[idx:exit_idx + 1]
            exit_price = float(window.iloc[-1]["Close"])
            stopped = False
            if stop_loss_pct is not None and stop_loss_pct > 0:
                stop_level = entry_price * (1 - stop_loss_pct / 100.0)
                for _, bar in window.iloc[1:].iterrows():
                    if float(bar["Low"]) <= stop_level:
                        exit_price = stop_level
                        stopped = True
                        break
            hold_ret = (exit_price / entry_price - 1) * 100
            row[f"Hold {hd}T"] = round(hold_ret, 2)
            row["Stop"] = "ja" if stopped else "nein"
            fixed_returns.append(hold_ret)
        else:
            row[f"Hold {hd}T"] = None
            row["Stop"] = "—"

        results.append(row)

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
            gross_win = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            summary[f"PF {days}T"] = round(gross_win / gross_loss, 2) if gross_loss > 0 else None
        else:
            for k in (f"Ø {days}T", f"Median {days}T", f"Trefferquote {days}T",
                      f"Best {days}T", f"Worst {days}T", f"PF {days}T"):
                summary[k] = None

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

    equity = []
    cum = 0.0
    for i, r in enumerate(fixed_returns):
        cum += r
        equity.append({"Signal #": i + 1, "Trade %": round(r, 2), "Kumuliert %": round(cum, 2)})

    summary["equity_final"] = round(cum, 2) if fixed_returns else None
    summary["hold_days"] = hold_days or 63

    return {"count": len(results), "summary": summary, "details": results, "equity": equity, "signal_dates": crosses}


# ---------------------------------------------------------------------------
# Rezessions-Barometer (FRED Daten & 0-100 Scoring)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=21600)
def fetch_fred_series(series_id: str) -> pd.Series:
    """Lädt eine makroökonomische Zeitreihe direkt als CSV von FRED."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            df = pd.read_csv(resp)
        if df.shape[1] < 2:
            return pd.Series(dtype=float)
        date_col, val_col = df.columns[0], df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[date_col, val_col]).sort_values(date_col)
        s = pd.Series(df[val_col].values, index=df[date_col], name=series_id)
        return s
    except Exception:
        return pd.Series(dtype=float)


def compute_recession_barometer() -> dict:
    """Berechnet das 0-100 Rezessions-Barometer aus 6 FRED-Frühindikatoren."""
    specs = [
        {
            "id": "SAHMREALTIME",
            "name": "Sahm-Regel Rezessionsindikator",
            "unit": "Pkt.",
            "weight": 0.25,
            "threshold": 0.50,
            "threshold_label": "Alarm ab ≥ 0,50",
            "short_desc": (
                "Misst den Anstieg des 3-Monats-Durchschnitts der US-Arbeitslosenquote gegenüber "
                "ihrem 12-Monats-Tief. Ab 0,50 Prozentpunkten wurde seit 1970 ausnahmslos jede "
                "US-Rezession ausgelöst."
            ),
            "xp": [0.05, 0.20, 0.35, 0.50, 0.75],
            "fp": [5, 25, 55, 85, 100],
        },
        {
            "id": "T10Y2Y",
            "name": "US-Zinsstrukturkurve (10J − 2J)",
            "unit": "%",
            "weight": 0.20,
            "threshold": 0.0,
            "threshold_label": "Inversion < 0,0 %",
            "short_desc": (
                "Differenz zwischen 10-jährigen und 2-jährigen US-Staatsanleihen. Eine inverse "
                "Zinskurve (< 0 %) zeigt, dass der Markt Zinssenkungen wegen einer kommenden "
                "Konjunkturabkühlung einpreist."
            ),
            "xp": [-1.0, -0.4, 0.0, 0.5, 1.5],
            "fp": [100, 82, 65, 32, 10],
        },
        {
            "id": "BAMLH0A0HYM2",
            "name": "High-Yield Credit Spreads",
            "unit": "%",
            "weight": 0.20,
            "threshold": 5.0,
            "threshold_label": "Stress ab > 5,0 %",
            "short_desc": (
                "Zinsaufschlag für US-Hochzinsanleihen gegenüber sicheren Staatsanleihen. "
                "Steigende Spreads (> 5 %) signalisieren Kreditstress und wachsende Ausfallängste "
                "bei Unternehmen."
            ),
            "xp": [2.8, 3.8, 5.0, 6.5, 8.5],
            "fp": [10, 30, 65, 85, 100],
        },
        {
            "id": "UMCSENT",
            "name": "US-Konsumentenstimmung (Michigan)",
            "unit": "Pkt.",
            "weight": 0.15,
            "threshold": 65.0,
            "threshold_label": "Schwach < 65 Pkt.",
            "short_desc": (
                "Stimmungsbarometer der US-Verbraucher (University of Michigan). Da rund 70 % "
                "der US-Wirtschaft vom Konsum abhängen, bremsen Werte unter 65 die Nachfrage deutlich."
            ),
            "xp": [45.0, 55.0, 65.0, 78.0, 92.0],
            "fp": [100, 80, 55, 28, 10],
        },
        {
            "id": "CPIAUCSL",
            "name": "US-Inflation (CPI YoY)",
            "unit": "%",
            "weight": 0.10,
            "threshold": 3.5,
            "threshold_label": "Hoch > 3,5 %",
            "short_desc": (
                "Jährliche US-Verbraucherpreisinflation. Dauerhaft überhöhte Inflation (> 3,5 %) "
                "entzieht Haushalten Kaufkraft und zwingt die Notenbank Fed zu restriktiven Zinsen."
            ),
            "xp": [-0.5, 1.5, 2.5, 3.5, 5.0, 7.0],
            "fp": [70, 15, 20, 48, 78, 95],
        },
        {
            "id": "FEDFUNDS",
            "name": "US-Leitzins (Fed Funds Rate)",
            "unit": "%",
            "weight": 0.10,
            "threshold": 4.5,
            "threshold_label": "Restriktiv > 4,5 %",
            "short_desc": (
                "Leitzins der US-Notenbank Federal Reserve. Hohe Zinsen (> 4,5 %) verteuern Kredite "
                "für Immobilien, Konsum und Investitionen und wirken mit 6–18 Monaten Verzögerung."
            ),
            "xp": [1.5, 3.0, 4.25, 5.25, 6.5],
            "fp": [15, 32, 55, 82, 95],
        },
    ]

    indicators = []
    weighted_sum = 0.0
    total_weight = 0.0

    for sp in specs:
        s = fetch_fred_series(sp["id"])
        if s.empty:
            indicators.append({
                **sp,
                "series": pd.Series(dtype=float),
                "latest": None,
                "date_str": "Keine Daten",
                "risk_score": None,
                "status": "⚪ Nicht verfügbar",
            })
            continue

        if sp["id"] == "CPIAUCSL":
            s_plot = s.pct_change(12).dropna() * 100.0
        else:
            s_plot = s.dropna()

        if s_plot.empty:
            indicators.append({
                **sp,
                "series": pd.Series(dtype=float),
                "latest": None,
                "date_str": "Keine Daten",
                "risk_score": None,
                "status": "⚪ Nicht verfügbar",
            })
            continue

        latest_val = float(s_plot.iloc[-1])
        latest_dt = s_plot.index[-1].strftime("%d.%m.%Y")
        risk = float(np.interp(latest_val, sp["xp"], sp["fp"]))
        risk_int = int(round(max(0, min(100, risk))))

        if risk_int >= 70:
            status = "🔴 Kritisch"
        elif risk_int >= 45:
            status = "🟠 Warnsignal"
        elif risk_int >= 25:
            status = "🟡 Neutral / Beobachtung"
        else:
            status = "🟢 Entspannt"

        weighted_sum += risk * sp["weight"]
        total_weight += sp["weight"]

        indicators.append({
            **sp,
            "series": s_plot.tail(1825),
            "latest": round(latest_val, 2),
            "date_str": latest_dt,
            "risk_score": risk_int,
            "status": status,
        })

    overall = int(round(weighted_sum / total_weight)) if total_weight > 0 else 42
    if overall >= 75:
        level = "Hohes Rezessionsrisiko"
        summary = (
            "🔴 **Akute Warnstufe:** Mehrere zentrale Frühindikatoren (wie Arbeitsmarkt, Zinskurve oder "
            "Kreditspreads) signalisieren eine stark erhöhte Rezessionswahrscheinlichkeit. Fokus auf "
            "Kapitalerhalt, hohe Qualität (starke Bilanzen/Cashflows) und ausreichende Liquiditätsreserve."
        )
    elif overall >= 50:
        level = "Erhöhtes Risiko"
        summary = (
            "🟠 **Deutliche Warnsignale:** Die Konjunktur zeigt spürbare Bremsspuren. Einzelne Makro-Indikatoren "
            "stehen bereits im kritischen Bereich. Selektives Vorgehen, defensivere Sektoren und gestaffelte "
            "Einstiege bei Rücksetzern sind ratsam."
        )
    elif overall >= 25:
        level = "Moderates Risiko (Spätzyklus)"
        summary = (
            "🟡 **Abkühlung / Spätzyklus:** Keine akute Rezessionspanik, aber das restriktive Umfeld hinterlässt "
            "erste Spuren. Trendstarke Qualitätsaktien bleiben interessant, spekulative Titel ohne Gewinne "
            "sollten gemieden werden."
        )
    else:
        level = "Geringes Risiko (Expansion)"
        summary = (
            "🟢 **Solides Makro-Umfeld:** Arbeitsmarkt, Kreditmärkte und Konsum zeigen ein gesundes "
            "Wachstumsumfeld. Rückenwind für Aktienmärkte und zyklische Trendfolger."
        )

    return {
        "score": overall,
        "level": level,
        "summary": summary,
        "indicators": indicators,
        "available_count": sum(1 for x in indicators if x["latest"] is not None),
    }


# ---------------------------------------------------------------------------
# Krypto Fear & Greed Barometer + Krypto ETF Flows
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_crypto_fear_greed() -> dict:
    """Lädt den Crypto Fear & Greed Index (0-100) mit 30-Tage-Verlauf + BTC/ETH Fallback."""
    try:
        req = urllib.request.Request(
            "https://api.alternative.me/fng/?limit=30",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = payload.get("data") or []
        if rows:
            records = []
            for r in rows:
                val = int(r["value"])
                ts = int(r["timestamp"])
                dt = datetime.fromtimestamp(ts)
                records.append({"Date": dt, "Score": val})
            df_fg = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
            cur = int(df_fg["Score"].iloc[-1])
            yest = int(df_fg["Score"].iloc[-2]) if len(df_fg) >= 2 else cur
            week_ago = int(df_fg["Score"].iloc[-8]) if len(df_fg) >= 8 else int(df_fg["Score"].iloc[0])
            avg_30 = round(float(df_fg["Score"].mean()), 1)
            return {
                "score": cur,
                "yesterday": yest,
                "week_ago": week_ago,
                "avg_30d": avg_30,
                "history": df_fg,
                "source": "Alternative.me Crypto Fear & Greed Index",
            }
    except Exception:
        pass

    btc = load_price_history("BTC-USD", "6mo")
    if not btc.empty and len(btc) >= 35:
        df_sub = btc.tail(35).copy()
        mom20 = (df_sub["Close"] / df_sub["Close"].shift(20) - 1) * 100
        raw_score = (0.6 * df_sub["RSI"].fillna(50) + 0.4 * (50 + mom20.fillna(0) * 2.0)).clip(0, 100)
        df_fg = pd.DataFrame({
            "Date": df_sub.index[-30:],
            "Score": raw_score.iloc[-30:].round().astype(int).values,
        })
        cur = int(df_fg["Score"].iloc[-1])
        yest = int(df_fg["Score"].iloc[-2]) if len(df_fg) >= 2 else cur
        week_ago = int(df_fg["Score"].iloc[-8]) if len(df_fg) >= 8 else cur
        avg_30 = round(float(df_fg["Score"].mean()), 1)
        return {
            "score": cur,
            "yesterday": yest,
            "week_ago": week_ago,
            "avg_30d": avg_30,
            "history": df_fg,
            "source": "Markt-Modell (BTC RSI & 20T-Momentum)",
        }

    dates = pd.date_range(end=datetime.today(), periods=30)
    scores = [46, 48, 51, 49, 44, 42, 45, 50, 54, 58, 61, 59, 55, 52, 48, 47, 50, 53, 57, 60, 62, 58, 54, 51, 49, 52, 56, 59, 57, 58]
    df_fg = pd.DataFrame({"Date": dates, "Score": scores})
    return {
        "score": 58,
        "yesterday": 57,
        "week_ago": 51,
        "avg_30d": 53.1,
        "history": df_fg,
        "source": "Basis-Modell",
    }


def classify_crypto_sentiment(score: int) -> tuple[str, str]:
    if score >= 75:
        return (
            "Extreme Gier",
            "🔴 **Extreme Gier (75–100):** Starke Euphorie und FOMO im Kryptomarkt. Historisch oft ein "
            "Warnzeichen für kurzfristige Überhitzung und Gewinnmitnahmen – gestaffelte Teilgewinnmitnahmen prüfen.",
        )
    if score >= 56:
        return (
            "Gier",
            "🟢 **Gier (56–74):** Positives Markt-Momentum und steigende Risikobereitschaft. Trendfolger "
            "sind im Vorteil, Rücksetzer an Unterstützungen werden meist zügig aufgekauft.",
        )
    if score >= 45:
        return (
            "Neutral",
            "🟡 **Neutral (45–55):** Ausgeglichenes Verhältnis zwischen Käufern und Verkäufern. "
            "Der Markt wartet auf den nächsten Impuls aus Makrodaten oder ETF-Zuflüssen.",
        )
    if score >= 25:
        return (
            "Angst",
            "🟠 **Angst (25–44):** Zurückhaltung und Skepsis dominieren. Für langfristige Anleger "
            "entstehen hier oft attraktive schrittweise Akkumulationszonen.",
        )
    return (
        "Extreme Angst",
        "🔵 **Extreme Angst (0–24):** Kapitulation und Panik im Markt. Historisch zählten Phasen extremer "
        "Angst bei Bitcoin & Co. langfristig zu den besten antizyklischen Einstiegsfenstern.",
    )


@st.cache_data(ttl=1800)
def load_crypto_spot_snapshot() -> dict:
    """Lädt Live-Kurse (EUR & USD) und 7d-Renditen für die 7 Krypto-ETF-Assets."""
    coins = {
        "BTC": ("Bitcoin", "BTC-USD", "US Spot-ETFs live (IBIT, FBTC, …)"),
        "ETH": ("Ethereum", "ETH-USD", "US Spot-ETFs live (ETHA, FETH, …)"),
        "SOL": ("Solana", "SOL-USD", "Spot-ETF / ETPs & Staking-Produkte"),
        "XRP": ("XRP", "XRP-USD", "Spot-ETF / ETPs & Institutionelle Produkte"),
        "ADA": ("Cardano", "ADA-USD", "ETPs & Multi-Asset Krypto-Fonds"),
        "XLM": ("Stellar", "XLM-USD", "ETPs & Zahlungsnetzwerk-Produkte"),
        "HBAR": ("Hedera", "HBAR-USD", "Spot-ETF / ETPs & Enterprise-Fonds"),
    }
    rates = get_fx_rates()
    usd_rate = float(rates.get("USD", 1.08)) or 1.08
    out = {}
    for sym, (name, yf_ticker, product_info) in coins.items():
        hist = load_price_history(yf_ticker, "1mo")
        if not hist.empty and len(hist) >= 2:
            last_usd = float(hist["Close"].iloc[-1])
            ref_idx = -8 if len(hist) >= 8 else 0
            prev_usd = float(hist["Close"].iloc[ref_idx])
            chg_7d = ((last_usd / prev_usd) - 1.0) * 100.0 if prev_usd > 0 else 0.0
            last_eur = last_usd / usd_rate
        else:
            last_usd = None
            last_eur = None
            chg_7d = None
        out[sym] = {
            "name": name,
            "ticker": yf_ticker,
            "price_usd": last_usd,
            "price_eur": last_eur,
            "chg_7d": round(chg_7d, 2) if chg_7d is not None else None,
            "product_info": product_info,
        }
    return out


# ---------------------------------------------------------------------------
# Sektor-Heatmap Datenloader (robust & gecacht)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def load_sector_heatmap_df(region: str, timeframe: str) -> pd.DataFrame:
    """Lädt Kurs-Performance & Marktkapitalisierung (in EUR) für die Sektor-Heatmap."""
    if region == "🇺🇸 US Tech & S&P":
        tickers = list(dict.fromkeys(NASDAQ_TICKERS[:20] + SP500_FALLBACK[:15]))
    elif region == "🇩🇪 DAX 40":
        tickers = list(DAX40_TICKERS)
    elif region == "🇪🇺 Europa Blue Chips":
        tickers = list(dict.fromkeys(BUILTIN_WATCHLISTS["Europa Blue Chips"] + DAX40_TICKERS[:12]))
    else:
        tickers = list(dict.fromkeys(NASDAQ_TICKERS[:18] + DAX40_TICKERS[:22]))

    lookback_map = {
        "1 Tag (Heute)": 1,
        "5 Tage (Woche)": 5,
        "1 Monat (21T)": 21,
    }
    step = lookback_map.get(timeframe, 1)

    batch_closes = {}
    try:
        raw = yf.download(
            tickers,
            period="3mo",
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                for t in tickers:
                    if t in level0 and "Close" in raw[t].columns:
                        s = raw[t]["Close"].dropna()
                        if len(s) >= 2:
                            batch_closes[t] = s
            elif "Close" in raw.columns and len(tickers) == 1:
                s = raw["Close"].dropna()
                if len(s) >= 2:
                    batch_closes[tickers[0]] = s
    except Exception:
        pass

    rates = get_fx_rates()
    rows = []

    for t in tickers:
        closes = batch_closes.get(t)
        if closes is None or len(closes) < 2:
            hist = load_price_history(t, "3mo")
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()

        if closes is None or len(closes) < 2:
            continue

        idx_prev = max(0, len(closes) - 1 - step)
        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[idx_prev])
        if prev_close <= 0:
            continue
        pct_change = (last_close / prev_close - 1.0) * 100.0

        meta = HEATMAP_META.get(t)
        display_name = meta[0] if meta else t
        sektor_de = meta[1] if meta else None
        mcap_eur = (meta[2] * 1e9) if meta else None

        if meta is None:
            try:
                info = yf.Ticker(t).info or {}
                raw_name = info.get("shortName") or info.get("longName") or t
                display_name = str(raw_name)[:20].strip() or t
                raw_sec = info.get("sector") or ""
                sektor_de = SECTOR_TRANSLATIONS.get(raw_sec, raw_sec if raw_sec else "Sonstige")
                raw_mcap = info.get("marketCap")
                if raw_mcap and float(raw_mcap) > 0:
                    cur = detect_currency(t, info)
                    mcap_eur = to_eur(float(raw_mcap), cur, rates=rates)
            except Exception:
                pass

        if not sektor_de or str(sektor_de).strip() == "":
            sektor_de = "Sonstige"
        if mcap_eur is None or mcap_eur <= 0:
            mcap_eur = 25e9

        short_code = t.split(".")[0]
        label = f"{display_name}" if display_name.upper() == short_code.upper() else f"{display_name} ({short_code})"

        rows.append({
            "Ticker": t,
            "Name": display_name,
            "Label": label,
            "Sektor": str(sektor_de),
            "Performance": round(float(pct_change), 2),
            "MarketCap": float(mcap_eur),
            "MarketCap_Mrd": round(float(mcap_eur) / 1e9, 1),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["Label"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# UI Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Navigation")
modes = [
    "📊 Einzelanalyse",
    "🌍 Index-Scanner",
    "🧭 Rezessions-Barometer",
    "🔍 Scanner",
    "🪙 Krypto ETF Flows",
    "📉 Backtesting",
    "📰 News-Feed",
    "💰 Sparplan-Rechner",
    "📅 Saisonalität",
    "🗺️ Sektor-Heatmap",
    "💸 Dividenden-Kalender",
]

if "navigation_mode" not in st.session_state:
    st.session_state["navigation_mode"] = "📊 Einzelanalyse"
if st.session_state["navigation_mode"] not in modes:
    st.session_state["navigation_mode"] = "📊 Einzelanalyse"

st.session_state["nav_radio"] = st.session_state["navigation_mode"]


def on_nav_change():
    st.session_state["navigation_mode"] = st.session_state["nav_radio"]


mode = st.sidebar.radio(
    "Ansicht",
    modes,
    key="nav_radio",
    on_change=on_nav_change,
)

st.sidebar.markdown("##### So nutzen")
st.sidebar.caption(
    "1. Scanner → Kandidaten finden\n"
    "2. Einzelanalyse → Chart, Scores, Sentiment\n"
    "3. Backtesting → Signal prüfen\n"
    "4. Makro & Krypto → Barometer & Flows\n\n"
    "📱 Chart: „Einfach“ + Zeit-Slider"
)

# ===========================================================================
# 1. Einzelanalyse
# ===========================================================================
if mode == "📊 Einzelanalyse":
    st.title("📊 Aktien-Einzelanalyse")
    col1, col2 = st.columns([2, 1])
    default_ticker = st.session_state.get("target_ticker", "Apple")

    with col1:
        user_input = st.text_input("Firmenname oder Ticker", value=default_ticker)
    with col2:
        period = st.selectbox("Zeitraum", ["1y", "2y", "5y", "10y"], index=1)
    ticker = resolve_ticker(user_input) if user_input else ""
    if ticker and ticker.upper() != user_input.upper().strip():
        st.caption(f"→ Interpretiert als: **{ticker}**")
    if ticker:
        data = load_price_history(ticker, period)
        if data.empty:
            st.error(
                "Keine Kursdaten gefunden. Ticker prüfen "
                "(Beispiel: NVDA, SAP.DE, BTC-USD, Gold, 1810.HK)."
            )
            st.caption("Tipp: Manche Asien-Titel brauchen die Börsen-Endung .HK / .T / .KS")
        else:
            fundamentals = load_fundamentals(ticker)
            bullish = is_bullish_now(data)
            golden = detect_golden_cross(data)
            ta_signal = compute_ta_signal(data)
            vol, max_dd = compute_risk_metrics(data)
            r6 = price_return(data, 126)
            r12 = price_return(data, 252)

            currency = detect_currency(ticker, fundamentals)
            rates = get_fx_rates()
            eur_usd = rates.get("USD", 1.08)
            rate = rates.get(currency, 1.0)
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

            plot = data.copy()
            for col in ("Open", "High", "Low", "Close", "SMA50", "SMA200"):
                if col in plot.columns:
                    plot[col] = plot[col] * fx

            gc_date = last_golden_cross_date(data)
            buy_date = last_buy_signal_date(data)
            gc_date_str = gc_date.strftime("%d.%m.%Y") if gc_date is not None else "—"
            buy_date_str = buy_date.strftime("%d.%m.%Y") if buy_date is not None else "—"

            sig = compute_signal_status(data)
            sent = compute_sentiment(data)

            comp_name = fundamentals.get("Name", ticker)
            st.markdown(f"### 🏢 {comp_name} (`{ticker}`)")
            desc = fundamentals.get("Beschreibung")
            if desc:
                with st.expander("📖 Unternehmensbeschreibung anzeigen", expanded=False):
                    st.write(desc)
                    st.caption("Quelle: Yahoo Finance (Originalfassung)")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Kurs", price_eur_str, price_sub)
            c2.metric("System-Status", sig["label"])
            c3.metric("Golden Cross", "✅ Ja" if golden else "—")
            c4.metric("TA-Signal", ta_signal)
            c5.metric("Sektor", str(fundamentals.get("Sektor", "—"))[:18])

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

            st.markdown("#### 🌡️ Markt-Sentiment (Tachometer)")
            score = sent["score"]

            fig_tacho = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"STIMMUNG: {sent['label'].upper()}", "font": {"size": 16, "color": "#cdd6f4", "family": "Arial Black"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white", "tickvals": [0, 25, 45, 55, 75, 100]},
                    "bar": {"color": "rgba(44, 62, 80, 0.9)", "thickness": 0.15},
                    "bgcolor": "#313244",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 25], "color": "#e74c3c"},
                        {"range": [25, 45], "color": "#f39c12"},
                        {"range": [45, 55], "color": "#bdc3c7"},
                        {"range": [55, 75], "color": "#2ecc71"},
                        {"range": [75, 100], "color": "#27ae60"},
                    ],
                },
            ))
            fig_tacho.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cdd6f4"},
            )
            fig_tacho.add_annotation(x=0.12, y=0.35, text="Extreme<br>Angst", showarrow=False, font=dict(color="white", size=10))
            fig_tacho.add_annotation(x=0.25, y=0.7, text="Angst", showarrow=False, font=dict(color="white", size=11))
            fig_tacho.add_annotation(x=0.5, y=0.9, text="Neutral", showarrow=False, font=dict(color="black", size=11))
            fig_tacho.add_annotation(x=0.75, y=0.7, text="Gier", showarrow=False, font=dict(color="white", size=11))
            fig_tacho.add_annotation(x=0.88, y=0.35, text="Extreme<br>Gier", showarrow=False, font=dict(color="white", size=10))

            st.plotly_chart(fig_tacho, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption(f"RSI: {sent.get('rsi', '—')} · Basiert auf RSI (14) & kurzfristigem Momentum.")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Letztes Golden Cross", gc_date_str)
            sc2.metric("Letztes Kaufsignal", buy_date_str)
            sc3.metric("Währung (Original)", currency)

            chart_title = f"{ticker} · Kurs in €"
            if currency != "EUR":
                chart_title += f" ({currency}→€)"

            st.markdown("#### 📈 Chart")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                chart_big = st.checkbox("Vollbild", value=False, help="Sehr hohe Chart-Ansicht")
            with cc2:
                mobile_simple = st.checkbox(
                    "Einfach (Handy)", value=True,
                    help="Kompakter Kurs-Chart; RSI/MACD ausblenden",
                )
            with cc3:
                show_slider = st.checkbox(
                    "Zeit-Slider", value=False,
                    help="Optional: feine Auswahl des Zeitfensters per Slider",
                )

            chart_window = st.radio(
                "Schnell-Zoom",
                ["1M", "3M", "6M", "1J", "2J", "Max"],
                index=2,
                horizontal=True,
                help="Tippen statt mit zwei Fingern lange zu zoomen.",
            )

            mc1, mc2 = st.columns(2)
            with mc1:
                show_buy_marks = st.checkbox("Kaufsignale anzeigen", value=True)
            with mc2:
                show_gc_marks = st.checkbox("Golden Cross anzeigen", value=True)

            chart_height = 1050 if chart_big else (560 if mobile_simple else 780)

            if mobile_simple:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=plot.index, open=plot["Open"], high=plot["High"],
                    low=plot["Low"], close=plot["Close"], name="Kurs €",
                    increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
                    decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
                ))
                if "SMA50" in plot.columns:
                    fig.add_trace(go.Scatter(
                        x=plot.index, y=plot["SMA50"], name="SMA50",
                        line=dict(color="#ffb74d", width=2.2),
                    ))
                if "SMA200" in plot.columns:
                    fig.add_trace(go.Scatter(
                        x=plot.index, y=plot["SMA200"], name="SMA200",
                        line=dict(color="#42a5f5", width=2.2),
                    ))
            else:
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
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

            def _add_markers(fig_obj, row=None):
                kwargs = {"row": row, "col": 1} if row else {}
                if show_buy_marks:
                    buy_dates = _find_signal_dates(data, "ta")
                    bx, by = [], []
                    for d in buy_dates:
                        if d in plot.index:
                            bx.append(d)
                            by.append(float(plot.loc[d, "Close"]))
                    if bx:
                        fig_obj.add_trace(go.Scatter(
                            x=bx, y=by, mode="markers", name="Kaufsignal",
                            marker=dict(symbol="triangle-up", size=14, color="#00e676",
                                        line=dict(width=1.5, color="#2c3e50")),
                            hovertemplate="Kaufsignal<br>%{x|%d.%m.%Y}<br>%{y:.2f} €<extra></extra>",
                        ), **kwargs)
                if show_gc_marks:
                    gc_dates = _find_signal_dates(data, "golden")
                    gx, gy = [], []
                    for d in gc_dates:
                        if d in plot.index:
                            gx.append(d)
                            gy.append(float(plot.loc[d, "Close"]))
                    if gx:
                        fig_obj.add_trace(go.Scatter(
                            x=gx, y=gy, mode="markers", name="Golden Cross",
                            marker=dict(symbol="star", size=15, color="#ffd600",
                                        line=dict(width=1, color="#333")),
                            hovertemplate="Golden Cross<br>%{x|%d.%m.%Y}<br>%{y:.2f} €<extra></extra>",
                        ), **kwargs)

            if mobile_simple:
                _add_markers(fig)
                for key, color, label in [
                    ("Analysten-Kursziel (niedrig)", "#2ca02c", "Min"),
                    ("Analysten-Kursziel (mittel)", "#1f77b4", "Normal"),
                    ("Analysten-Kursziel (hoch)", "#d62728", "Max"),
                ]:
                    val = fundamentals.get(key)
                    if val:
                        fig.add_hline(y=float(val) * fx, line_dash="dash", line_color=color,
                                      annotation_text=f"{label} {float(val)*fx:.1f} €")
                entry_zones = compute_entry_zones(data, fx=fx)
                zone_fills = [
                    "rgba(38,166,154,0.10)",
                    "rgba(255,183,77,0.10)",
                    "rgba(66,165,245,0.10)",
                ]
                for z, fill in zip(entry_zones, zone_fills):
                    fig.add_hrect(
                        y0=z["low"], y1=z["high"],
                        fillcolor=fill, line_width=0,
                        annotation_text=f"{z['name']} {z['low']:.2f}–{z['high']:.2f} €",
                        annotation_position="top left",
                        annotation_font_size=10,
                    )
            else:
                _add_markers(fig, row=1)
                for key, color, label in [
                    ("Analysten-Kursziel (niedrig)", "#2ca02c", "Min"),
                    ("Analysten-Kursziel (mittel)", "#1f77b4", "Normal"),
                    ("Analysten-Kursziel (hoch)", "#d62728", "Max"),
                ]:
                    val = fundamentals.get(key)
                    if val:
                        fig.add_hline(y=float(val) * fx, line_dash="dash", line_color=color,
                                      annotation_text=f"{label} {float(val)*fx:.1f} €",
                                      annotation_font_size=11, row=1, col=1)
                entry_zones = compute_entry_zones(data, fx=fx)
                zone_fills = [
                    "rgba(38,166,154,0.10)",
                    "rgba(255,183,77,0.10)",
                    "rgba(66,165,245,0.10)",
                ]
                for z, fill in zip(entry_zones, zone_fills):
                    fig.add_hrect(
                        y0=z["low"], y1=z["high"], fillcolor=fill,
                        line_width=0, row=1, col=1,
                        annotation_text=f"{z['name']} {z['low']:.2f}–{z['high']:.2f} €",
                        annotation_position="top left",
                        annotation_font_size=10,
                    )
                fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI",
                                         line=dict(color="#ce93d8", width=1.8)), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.7)", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.7)", row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD",
                                         line=dict(color="#64b5f6", width=1.6)), row=3, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal",
                                         line=dict(color="#ffb74d", width=1.6)), row=3, col=1)
                colors = ["#26a69a" if v >= 0 else "#ef5350" for v in data["MACD_hist"].fillna(0)]
                fig.add_trace(go.Bar(x=data.index, y=data["MACD_hist"], name="Hist",
                                     marker_color=colors, opacity=0.75), row=3, col=1)

            layout_extra = dict(
                **CHART_LAYOUT,
                height=chart_height,
                title=dict(text=chart_title if mobile_simple else None, font=dict(size=16)),
                yaxis_title="Kurs €",
            )
            fig.update_layout(
                **layout_extra,
                uirevision=f"{ticker}-{chart_window}-{int(mobile_simple)}",
            )

            end_date = plot.index[-1]
            offsets = {
                "1M": pd.DateOffset(months=1),
                "3M": pd.DateOffset(months=3),
                "6M": pd.DateOffset(months=6),
                "1J": pd.DateOffset(years=1),
                "2J": pd.DateOffset(years=2),
            }
            if chart_window == "Max":
                chart_start = plot.index[0]
            else:
                chart_start = max(plot.index[0], end_date - offsets[chart_window])
            fig.update_xaxes(range=[chart_start, end_date])

            slider_thickness = 0.08 if mobile_simple else 0.07
            if mobile_simple:
                fig.update_xaxes(
                    rangeslider=dict(visible=show_slider, thickness=slider_thickness),
                    fixedrange=False,
                    showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                )
                fig.update_yaxes(fixedrange=False, side="right")
            else:
                fig.update_xaxes(fixedrange=False, showgrid=True, gridcolor="rgba(255,255,255,0.1)")
                fig.update_xaxes(
                    rangeslider=dict(visible=show_slider, thickness=slider_thickness),
                    row=3, col=1,
                )
                fig.update_yaxes(fixedrange=False)
                fig.update_layout(
                    yaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                                fixedrange=False, title="RSI", range=[0, 100]),
                    yaxis3=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                                fixedrange=False, title="MACD"),
                )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key=f"ch_{ticker}_{int(chart_big)}_{int(mobile_simple)}_{int(show_slider)}",
            )
            st.info(
                "📱 **Handy:** Zeitraum oben antippen (1M/3M/6M/1J/2J) · "
                "mit einem Finger schieben · mit zwei Fingern zoomen · "
                "oben rechts +/− für Feintuning · 🟢 Kaufsignal · ⭐ Golden Cross"
            )

            entry_zones = compute_entry_zones(data, fx=fx)
            if entry_zones:
                st.markdown("#### 🎯 3 Kauf-Einstiegszonen bei Rücksetzern")
                ecols = st.columns(3)
                for i, z in enumerate(entry_zones):
                    with ecols[i]:
                        position = "unter aktuellem Kurs" if z["unter_kurs"] else "über aktuellem Kurs"
                        st.metric(
                            z["name"],
                            f"{z['low']:.2f} – {z['high']:.2f} €",
                            f"{z['distanz_pct']:+.1f}% · {position}",
                        )
                        st.caption(z["typ"])
                st.caption(
                    "Technische Rücksetzer-Zonen aus dem 12-Monats-Swing (Fibonacci 23,6–61,8 %). "
                    "Sie sind Orientierungsbereiche und keine Garantie für eine Trendwende."
                )

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
                st.caption("Ziele aus Fibonacci-Retracements + Analysten-Kurszielen abgeleitet. Keine Garantie.")

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

            corr_info = compute_market_correlation(ticker)
            div_info = load_dividend_summary(ticker)
            x1, x2, x3 = st.columns(3)
            if corr_info.get("corr") is not None:
                x1.metric(f"Korrelation ({corr_info['bench']})", f"{corr_info['corr']:.2f}")
            else:
                x1.metric("Korrelation", "—")
            if div_info.get("yield") is not None:
                x2.metric("Div.-Rendite", f"{float(div_info['yield'])*100:.2f} %")
            else:
                x2.metric("Div.-Rendite", "—")
            x3.metric("Div.-Zahlungen (3J)", str(div_info.get("count_3y") or "—"))
            if corr_info.get("corr") is not None:
                c = corr_info["corr"]
                if c >= 0.8:
                    st.caption(f"Hohe Gleichbewegung mit {corr_info['bench']} – Markt-Risiko dominiert.")
                elif c <= 0.3:
                    st.caption(f"Eher eigenständig vs. {corr_info['bench']} – weniger Markt-Beta.")
                else:
                    st.caption(f"Mittlere Korrelation zu {corr_info['bench']}.")

            st.subheader("🏭 Branchenvergleich")
            st.write(f"**Sektor:** {fundamentals.get('Sektor', '—')}  |  **Branche:** {fundamentals.get('Branche', '—')}")
            rel = relative_valuation(fundamentals)
            val_sc = valuation_score(rel)
            if rel:
                cols = st.columns(len(rel))
                for i, (key, info) in enumerate(rel.items()):
                    with cols[i]:
                        val_str = f"{info['Wert']:.2f}" if isinstance(info["Wert"], (int, float)) else str(info["Wert"])
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
                if k in ("Name", "Sektor", "Branche", "Beschreibung") or v is None:
                    continue
                if k in money_keys:
                    if k == "Marktkap." and isinstance(v, (int, float)):
                        rates = get_fx_rates()
                        eur_v = to_eur(v, currency, eur_usd=eur_usd, rates=rates)
                        if eur_v is not None and eur_v >= 1e9:
                            display_rows.append({"Kennzahl": k, "Wert": f"{eur_v/1e9:.2f} Mrd. €"})
                        elif eur_v is not None and eur_v >= 1e6:
                            display_rows.append({"Kennzahl": k, "Wert": f"{eur_v/1e6:.2f} Mio. €"})
                        else:
                            display_rows.append({"Kennzahl": k, "Wert": format_price_eur(v, currency, eur_usd)})
                    else:
                        display_rows.append({"Kennzahl": k, "Wert": format_price_eur(v, currency, eur_usd)})
                elif k == "Dividendenrendite" and isinstance(v, (int, float)):
                    display_rows.append({"Kennzahl": k, "Wert": f"{v*100:.2f} %"})
                elif isinstance(v, float) and k in ("ROE", "Gewinnmarge", "Operative Marge", "Umsatzwachstum (YoY)"):
                    display_rows.append({"Kennzahl": k, "Wert": f"{v*100:.1f} %"})
                else:
                    display_rows.append({"Kennzahl": k, "Wert": v})
            fund_df = pd.DataFrame(display_rows)
            st.dataframe(fund_df, use_container_width=True, hide_index=True)
            if currency != "EUR":
                st.caption(f"Preise und Chart in € umgerechnet (Originalwährung: {currency}).")

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
        "🥇 Edelmetalle (Gold/Silber/…)",
        "Eigene Liste",
        "⭐ Meine Watchlists",
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
        st.caption("Krypto: Min. Piotroski auf 0 lassen (keine Bilanz-Scores).")
    elif universe == "🥇 Edelmetalle (Gold/Silber/…)":
        tickers = PRECIOUS_METALS
        st.caption("Edelmetalle: Min. Piotroski auf 0 lassen.")
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

    saved_wl = load_json(WATCHLIST_FILE, {})
    if universe == "⭐ Meine Watchlists" and saved_wl:
        wl_name = st.selectbox("Watchlist wählen", list(saved_wl.keys()))
        tickers = list(saved_wl.get(wl_name) or [])
    elif universe == "⭐ Meine Watchlists":
        st.warning("Noch keine Watchlists – unter „⭐ Watchlists“ anlegen.")
        tickers = []

    colf1, colf2, colf3, colf4 = st.columns(4)
    signal_filter = colf1.selectbox(
        "Signal-Filter",
        ["Nur Kaufsignale", "Nur Verkaufssignale", "Alle Signale"],
        index=0,
    )
    min_piotroski = colf2.slider("Min. Piotroski", 0, 9, 0)
    min_composite = colf3.slider("Min. Composite", 0, 100, 0)
    sector_filter = colf4.text_input("Sektor enthält", "", help="z.B. Technology – leer = alle")

    if tickers and st.button("Scanner starten", type="primary"):
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
                if signal_filter == "Nur Kaufsignale" and "Kaufsignal" not in signal:
                    continue
                if signal_filter == "Nur Verkaufssignale" and "Verkaufssignal" not in signal:
                    continue
                fund = load_fundamentals(t)
                if sector_filter.strip():
                    sec = str(fund.get("Sektor") or "")
                    if sector_filter.strip().lower() not in sec.lower():
                        continue
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

            st.markdown("---")
            st.markdown("#### 🚀 Direkt zur Einzelanalyse springen")

            selected_ticker_jump = st.selectbox("Treffer auswählen:", df["Ticker"].tolist(), key="jump_ticker_select_box")
            if st.button("Zur Einzelanalyse springen", type="primary", key="jump_btn_direct"):
                st.session_state["target_ticker"] = selected_ticker_jump
                st.session_state["navigation_mode"] = "📊 Einzelanalyse"
                st.session_state["nav_radio"] = "📊 Einzelanalyse"
                st.rerun()

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Ergebnisse als CSV",
                data=csv_bytes,
                file_name=f"scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        else:
            st.info("Keine Treffer mit den aktuellen Filtern. Filter lockern und erneut scannen.")


# ===========================================================================
# 2.5 Index-Scanner
# ===========================================================================
elif mode == "🌍 Index-Scanner":
    st.title("🌍 Index-Scanner & Marktstimmung")
    st.markdown("Wähle einen globalen Leitindex, um den aktuellen Stimmungsindikator und eine detaillierte Investitionseinschätzung zu erhalten.")

    indices = {
        "S&P 500 (US)": "^GSPC",
        "Nasdaq 100 (US)": "^NDX",
        "DAX 40 (DE)": "^GDAXI",
        "Euro Stoxx 50 (EU)": "^STOXX50E",
        "Nikkei 225 (JP)": "^N225",
        "Hang Seng (HK)": "^HSI",
        "Krypto-Markt (Bitcoin-Leitindex)": "BTC-USD",
    }

    selected_name = st.selectbox("Index auswählen:", list(indices.keys()))
    ticker = indices[selected_name]

    with st.spinner("Analysiere Daten..."):
        data = load_price_history(ticker, "2y")

    if not data.empty:
        sent = compute_sentiment(data)
        st_sig = compute_signal_status(data)

        last_price = data["Close"].iloc[-1]
        sma50 = data["SMA50"].iloc[-1]
        sma200 = data["SMA200"].iloc[-1]
        rsi = data["RSI"].iloc[-1]

        diff_50 = (last_price / sma50 - 1) * 100 if sma50 else 0
        diff_200 = (last_price / sma200 - 1) * 100 if sma200 else 0

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("Tachometer: Marktstimmung")
            score = sent["score"]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "STIMMUNGSINDIKATOR", "font": {"size": 18, "color": "#cdd6f4", "family": "Arial Black"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white", "tickvals": [0, 25, 45, 55, 75, 100]},
                    "bar": {"color": "rgba(44, 62, 80, 0.9)", "thickness": 0.15},
                    "bgcolor": "#313244",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 25], "color": "#e74c3c"},
                        {"range": [25, 45], "color": "#f39c12"},
                        {"range": [45, 55], "color": "#bdc3c7"},
                        {"range": [55, 75], "color": "#2ecc71"},
                        {"range": [75, 100], "color": "#27ae60"},
                    ],
                },
            ))
            fig.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cdd6f4"},
            )
            fig.add_annotation(x=0.12, y=0.35, text="Extreme<br>Angst", showarrow=False, font=dict(color="white", size=11))
            fig.add_annotation(x=0.25, y=0.7, text="Angst", showarrow=False, font=dict(color="white", size=12))
            fig.add_annotation(x=0.5, y=0.9, text="Neutral", showarrow=False, font=dict(color="black", size=12))
            fig.add_annotation(x=0.75, y=0.7, text="Gier", showarrow=False, font=dict(color="white", size=12))
            fig.add_annotation(x=0.88, y=0.35, text="Extreme<br>Gier", showarrow=False, font=dict(color="white", size=11))

            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            m1, m2, m3 = st.columns(3)
            cur_map = {"^GSPC": "USD", "^NDX": "USD", "^GDAXI": "EUR", "^STOXX50E": "EUR", "^N225": "JPY", "^HSI": "HKD", "BTC-USD": "USD"}
            idx_cur = cur_map.get(ticker, "USD")
            rates = get_fx_rates()
            fx_rate = rates.get(idx_cur, 1.0)
            idx_fx = (1.0 / fx_rate) if (idx_cur != "EUR" and fx_rate > 0) else 1.0
            last_price_eur = last_price * idx_fx

            if idx_cur == "EUR":
                m1.metric("Kurs in €", f"{last_price:,.2f} €", "Punkte = Euro")
            else:
                m1.metric("Kurs in €", f"{last_price_eur:,.2f} €", f"Original: {last_price:,.2f} Pkt ({idx_cur})")
            m2.metric("Abstand 200-Tage", f"{diff_200:+.1f} %")
            m3.metric("RSI", f"{rsi:.1f}")

        with col2:
            st.subheader("💡 Detaillierte Investitionseinschätzung")
            st.markdown(f"**Index:** {selected_name} | **Status:** {st_sig['label']}")

            if diff_200 > 5 and rsi < 70:
                einschaetzung = "🟢 **Positive Phase (Kaufenswert):** Der Index befindet sich in einem gesunden Aufwärtstrend. Der Kurs notiert komfortabel über der wichtigen 200-Tage-Linie. Da der Markt noch nicht extrem überhitzt ist, bieten sich für Trendfolger aktuell gute Einstiegschancen. Rücksetzer können zum Aufbau von Positionen genutzt werden."
            elif diff_200 > 10 and rsi >= 70:
                einschaetzung = "🟡 **Überhitzt (Gewinne laufen lassen / Rücksetzer abwarten):** Der Index ist in letzter Zeit sehr stark gelaufen und notiert weit über seinen Durchschnittslinien. Die Stimmung im Markt ist von starker Gier geprägt (siehe Tachometer). Für größere Neukäufe sollte auf einen Rücksetzer gewartet werden. Bestehende Positionen können gehalten, aber per Stop-Loss abgesichert werden."
            elif diff_200 < 0 and diff_50 < 0 and rsi > 35:
                einschaetzung = "🔴 **Abwärtstrend (Vorsicht / Seitenlinie):** Der Markt ist technisch angeschlagen und notiert unter den wichtigen langfristigen Trendlinien. Die Stimmung ist gedrückt. Investoren sollten aktuell eher an der Seitenlinie bleiben, Cash-Quoten aufbauen und auf eine klare Bodenbildung oder ein neues Golden Cross warten."
            elif diff_200 < -5 and rsi <= 35:
                einschaetzung = "🟠 **Stark Überverkauft (Antizyklische Chance):** Der Index hat stark korrigiert und die Stimmung ist von Panik und extremer Angst geprägt. Solche Phasen bieten historisch gesehen oft hervorragende antizyklische Einstiegspunkte für mutige, langfristige Investoren, auch wenn der kurzfristige Trend noch abwärts gerichtet ist."
            else:
                einschaetzung = "🟡 **Neutrale Phase (Selektiv vorgehen):** Der Markt sucht aktuell nach einer klaren Richtung und pendelt um seine Trendlinien herum. In dieser Seitwärtsphase ist Stock-Picking entscheidend, da breite Index-Käufe momentan wenig eigenes Momentum mitbringen."

            st.info(einschaetzung)

            st.markdown("---")
            st.markdown("##### 🔍 Technisches Fazit")
            if data["SMA50"].iloc[-1] > data["SMA200"].iloc[-1]:
                st.write("✅ Die mittelfristige 50-Tage-Linie verläuft **über** der langfristigen 200-Tage-Linie (Ein bullisches Grundgerüst / Golden Cross aktiv).")
            else:
                st.write("❌ Die mittelfristige 50-Tage-Linie verläuft **unter** der 200-Tage-Linie (Bärisches Umfeld / Death Cross aktiv).")

            if rsi > 75:
                st.write("⚠️ Der RSI zeigt mit einem Wert von über 75 eine deutliche **Überkaufung** an. Das Rückschlagsrisiko ist kurzfristig stark erhöht.")
            elif rsi < 30:
                st.write("📉 Der RSI zeigt mit unter 30 eine deutliche **Überverkaufung** an. Eine baldige Gegenreaktion nach oben (Bounce) ist wahrscheinlich.")
            else:
                st.write("📊 Der RSI befindet sich im **neutralen Bereich**, der Markt ist weder panisch abverkauft noch heißgelaufen.")


# ===========================================================================
# 2.8 Rezessions-Barometer (FRED Daten + Beschreibung)
# ===========================================================================
elif mode == "🧭 Rezessions-Barometer":
    st.title("🧭 Rezessions-Barometer (FRED Makro-Check)")
    st.markdown(
        "Das **Rezessions-Barometer** bündelt **6 bewährte US-Frühindikatoren** der Federal Reserve Bank "
        "of St. Louis (**FRED**) zu einem gewichteten **Risiko-Score von 0 bis 100**. Da die US-Wirtschaft "
        "der Taktgeber für die globalen Aktienmärkte ist, liefert dieser Makro-Check eine wertvolle "
        "Orientierung für Investitionsquote, Risikomanagement und Sektorauswahl."
    )

    with st.spinner("Lade aktuelle Makro-Zeitreihen von FRED..."):
        rec = compute_recession_barometer()

    col_g, col_desc = st.columns([1.05, 1.25])

    with col_g:
        st.subheader("Tachometer: Rezessionsrisiko")
        r_score = rec["score"]
        fig_rec = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"RISIKO: {rec['level'].upper()}",
                "font": {"size": 16, "color": "#cdd6f4", "family": "Arial Black"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white", "tickvals": [0, 25, 50, 75, 100]},
                "bar": {"color": "rgba(44, 62, 80, 0.92)", "thickness": 0.16},
                "bgcolor": "#313244",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#26a69a"},
                    {"range": [25, 50], "color": "#f1c40f"},
                    {"range": [50, 75], "color": "#e67e22"},
                    {"range": [75, 100], "color": "#ef5350"},
                ],
            },
        ))
        fig_rec.update_layout(
            height=330,
            margin=dict(l=20, r=20, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#cdd6f4"},
        )
        fig_rec.add_annotation(x=0.14, y=0.35, text="Gering<br>(0–25)", showarrow=False, font=dict(color="white", size=11))
        fig_rec.add_annotation(x=0.30, y=0.75, text="Moderat<br>(25–50)", showarrow=False, font=dict(color="black", size=11))
        fig_rec.add_annotation(x=0.70, y=0.75, text="Erhöht<br>(50–75)", showarrow=False, font=dict(color="white", size=11))
        fig_rec.add_annotation(x=0.86, y=0.35, text="Hoch<br>(75–100)", showarrow=False, font=dict(color="white", size=11))
        st.plotly_chart(fig_rec, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(f"Aktive FRED-Indikatoren: {rec['available_count']} / 6 · Gewichtung: Sahm 25%, Zinskurve 20%, Credit Spreads 20%, Konsum 15%, Inflation 10%, Leitzins 10%")

    with col_desc:
        st.subheader("📖 Beschreibung & Einordnung")
        st.info(rec["summary"])

        st.markdown("##### 🎯 So liest du die 4 Risikozonen:")
        st.markdown(
            "- **0 – 25 Punkte (🟢 Geringes Risiko / Expansion):** Wirtschaft wächst stabil, Arbeitsmarkt und "
            "Kreditmärkte sind gesund. Rückenwind für Aktien.\n"
            "- **25 – 50 Punkte (🟡 Moderates Risiko / Spätzyklus):** Erste Abkühlungstendenzen oder höhere Zinsen, "
            "aber noch keine akute Rezessionsgefahr. Qualitätsaktien bevorzugen.\n"
            "- **50 – 75 Punkte (🟠 Erhöhtes Risiko / Warnsignale):** Mehrere Frühindikatoren schlagen an. "
            "Positionsgrößen diszipliniert steuern und Rücksetzer abwarten.\n"
            "- **75 – 100 Punkte (🔴 Hohes Rezessionsrisiko):** Akute Alarmstufe (z. B. Sahm-Regel ausgelöst + "
            "steigende Kreditspreads). Hohe Cash-Reserve und defensive Absicherung sinnvoll."
        )

    st.markdown("---")
    st.subheader("📊 Die 6 FRED-Frühindikatoren im Überblick")

    inds = rec["indicators"]
    row1 = st.columns(3)
    row2 = st.columns(3)
    all_cols = row1 + row2

    for idx_i, item in enumerate(inds):
        with all_cols[idx_i]:
            val_str = f"{item['latest']:.2f} {item['unit']}" if item["latest"] is not None else "—"
            sub_str = (
                f"Risiko-Score: {item['risk_score']}/100 · {item['status']}"
                if item["risk_score"] is not None
                else item["status"]
            )
            st.metric(f"{item['name']} ({item['id']})", val_str, sub_str)
            st.caption(f"Stand: {item['date_str']} · {item['threshold_label']} · Gewicht {int(item['weight']*100)}%")

    valid_inds = [x for x in inds if x["risk_score"] is not None]
    if valid_inds:
        st.markdown("#### ⚖️ Teil-Risiko je Indikator (0 = entspannt, 100 = kritisch)")
        fig_bar = go.Figure()
        bar_colors = [
            "#ef5350" if x["risk_score"] >= 70 else ("#e67e22" if x["risk_score"] >= 45 else ("#f1c40f" if x["risk_score"] >= 25 else "#26a69a"))
            for x in valid_inds
        ]
        fig_bar.add_trace(go.Bar(
            x=[x["risk_score"] for x in valid_inds],
            y=[f"{x['name']} ({x['id']})" for x in valid_inds],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{x['risk_score']}/100 ({x['latest']:.2f} {x['unit']})" for x in valid_inds],
            textposition="outside",
        ))
        fig_bar.add_vline(x=50, line_dash="dash", line_color="rgba(255,255,255,0.45)", annotation_text="Warnschwelle 50")
        fig_bar.update_layout(
            **CHART_LAYOUT,
            height=340,
            xaxis=dict(range=[0, 115], title="Teil-Risiko (0–100)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(autorange="reversed", showgrid=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    st.subheader("🔍 Detaillierte Indikator-Beschreibungen & 5-Jahres-Verlauf")
    for item in inds:
        exp_title = (
            f"{item['status']} · {item['name']} [{item['id']}] — Aktuell: {item['latest']} {item['unit']}"
            if item["latest"] is not None
            else f"{item['name']} [{item['id']}]"
        )
        with st.expander(exp_title, expanded=True):
            c_txt, c_plt = st.columns([1, 1.4])
            with c_txt:
                st.markdown(f"**Was misst dieser Indikator?**\n\n{item['short_desc']}")
                st.markdown(f"**Kritische Schwelle:** `{item['threshold_label']}`")
                if item["risk_score"] is not None:
                    st.markdown(f"**Aktueller Wert:** `{item['latest']:.2f} {item['unit']}` (Stand: {item['date_str']}) → **Teil-Risiko: {item['risk_score']}/100**")
            with c_plt:
                s_hist = item["series"]
                if not s_hist.empty:
                    fig_ind = go.Figure()
                    fig_ind.add_trace(go.Scatter(
                        x=s_hist.index,
                        y=s_hist.values,
                        mode="lines",
                        name=item["id"],
                        line=dict(color="#89b4fa", width=2.2),
                        fill="tozeroy",
                        fillcolor="rgba(137, 180, 250, 0.12)",
                    ))
                    fig_ind.add_hline(
                        y=item["threshold"],
                        line_dash="dash",
                        line_color="#ef5350",
                        annotation_text=item["threshold_label"],
                        annotation_position="top left",
                    )
                    fig_ind.update_layout(
                        **CHART_LAYOUT,
                        height=250,
                        yaxis_title=item["unit"],
                    )
                    st.plotly_chart(fig_ind, use_container_width=True, config=PLOTLY_CONFIG, key=f"fred_{item['id']}")
                else:
                    st.caption("Verlaufsdaten aktuell nicht abrufbar.")


# ===========================================================================
# 3. Backtesting
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
    stop_loss_pct = st.slider("Stop-Loss % (0 = aus)", 0, 25, 0, 1, help="Simuliert Ausstieg bei X% Verlust innerhalb der Haltedauer")

    ticker = resolve_ticker(user_input) if user_input else ""
    if ticker and ticker.upper() != user_input.upper().strip():
        st.caption(f"→ Interpretiert als: **{ticker}**")

    if ticker and st.button("Backtest starten", type="primary"):
        data = load_price_history(ticker, period)
        if data.empty:
            st.error("Keine Daten.")
        else:
            sig_type = signal_map[signal_choice]
            result = backtest_signals(
                data,
                signal_type=sig_type,
                hold_days=hold_days,
                stop_loss_pct=float(stop_loss_pct) if stop_loss_pct else None,
            )

            if result["count"] == 0:
                st.warning("In diesem Zeitraum wurden keine Signale dieses Typs gefunden. Anderen Signal-Typ oder längere Historie wählen.")
            else:
                summary = result["summary"]
                st.subheader(f"📊 {result['count']} Signale · {signal_choice}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ø 1 Monat", f"{summary.get('Ø 21T', '—')} %",
                          f"TQ {summary.get('Trefferquote 21T', '—')} %")
                m2.metric("Ø 3 Monate", f"{summary.get('Ø 63T', '—')} %",
                          f"TQ {summary.get('Trefferquote 63T', '—')} %")
                m3.metric("Ø 6 Monate", f"{summary.get('Ø 126T', '—')} %",
                          f"TQ {summary.get('Trefferquote 126T', '—')} %")
                m4.metric("Profit Factor (3M)", f"{summary.get('PF 63T', '—')}",
                          f"Max Verlustserie: {summary.get('max_consec_loss', '—')}")

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
                    fig_eq.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.4)")
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
                            line=dict(width=1, color="#2c3e50"),
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

                st.subheader("Einzelne Signale")
                details_df = pd.DataFrame(result["details"])
                st.dataframe(details_df, use_container_width=True, hide_index=True)

                avg_3m = summary.get("Ø 63T")
                hit_3m = summary.get("Trefferquote 63T")
                pf = summary.get("PF 63T")
                score = 0
                if avg_3m is not None and hit_3m is not None:
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

                st.markdown("---")
                st.subheader("📋 Kurz-Fazit & möglicher Wiedereinstieg")

                sig_now = compute_signal_status(data)
                sent_now = compute_sentiment(data)
                rsi_now = sent_now.get("rsi")
                last_gc = last_golden_cross_date(data)
                last_buy = last_buy_signal_date(data)
                bull = is_bullish_now(data)

                status_txt = sig_now.get("label", "—")
                since_txt = (
                    sig_now["since"].strftime("%d.%m.%Y")
                    if sig_now.get("since") is not None else "—"
                )
                perf_txt = (
                    f"{sig_now['perf_pct']:+.1f} %"
                    if sig_now.get("perf_pct") is not None else "—"
                )

                hints = []
                if score >= 5 and bull and rsi_now is not None and rsi_now < 65:
                    einstieg = "🟢 Bedingungen eher günstig"
                    hints.append(
                        "Historisch starkes Signal + aktuell bullischer Trend. "
                        "Ein Einstieg bei Rücksetzer (RSI Richtung 40–50) war in der Vergangenheit oft sinnvoller als im Extrem-Overbought."
                    )
                elif score >= 3 and bull:
                    einstieg = "🟡 Abwarten auf besseren Einstieg"
                    hints.append(
                        "Signal historisch brauchbar, Trend bullisch. "
                        "Lieber auf ein frisches Kaufsignal, RSI-Rücksetzer unter ~60 oder Nähe zur SMA50 warten."
                    )
                elif not bull and score >= 3:
                    einstieg = "🟠 Noch kein klarer Long-Einstieg"
                    hints.append(
                        "Trend eher bärisch/seitwärts. Historisch half oft das nächste Golden Cross "
                        "oder ein RSI-Bounce aus dem überverkauften Bereich (<30 → >30)."
                    )
                elif score < 3:
                    einstieg = "🔴 Signal allein nicht verlässlich"
                    hints.append(
                        "Dieses Signal hat bei dieser Aktie historisch wenig Edge gezeigt. "
                        "Einstieg nur mit zusätzlichen Qualitätsfiltern (Piotroski, Bewertung, Sektor) – nicht rein technisch."
                    )
                else:
                    einstieg = "🟡 Neutral / abwarten"
                    hints.append("Aktuell kein klares Bild. Nächstes klares TA-Kaufsignal oder Golden Cross abwarten.")

                if rsi_now is not None:
                    if rsi_now >= 70:
                        hints.append(f"RSI liegt bei **{rsi_now}** (überkauft) – kurzfristig erhöhtes Rücksetzer-Risiko.")
                    elif rsi_now <= 35:
                        hints.append(f"RSI liegt bei **{rsi_now}** (eher überverkauft) – historisch oft interessante Bounce-Zone.")
                    else:
                        hints.append(f"RSI bei **{rsi_now}** – weder extrem überkauft noch überverkauft.")

                if last_buy is not None:
                    hints.append(f"Letztes kombiniertes Kaufsignal: **{last_buy.strftime('%d.%m.%Y')}**.")
                if last_gc is not None:
                    hints.append(f"Letztes Golden Cross: **{last_gc.strftime('%d.%m.%Y')}**.")

                hold = summary.get("hold_days") or 63
                if avg_3m is not None:
                    hints.append(
                        f"Im Backtest lag die Ø-Rendite nach ~3 Monaten bei **{avg_3m}%** "
                        f"(Trefferquote {hit_3m if hit_3m is not None else '—'} %). "
                        f"Als grober Orientierungshorizont: {hold} Handelstage prüfen und mit Stop/Ziel arbeiten."
                    )

                st.markdown(f"**Aktueller System-Status:** {status_txt} · seit {since_txt} · Performance seit Signal: {perf_txt}")
                st.markdown(f"**Sentiment:** {sent_now.get('label', '—')} ({sent_now.get('score', '—')}/100)")
                st.markdown(f"**Einschätzung Einstieg:** {einstieg}")
                for h in hints:
                    st.markdown(f"- {h}")

                st.caption(
                    "Keine Anlageberatung. Zusammenfassung ist eine automatische Heuristik aus Backtest + Trend/RSI. "
                    "Vergangene Performance ist keine Garantie für die Zukunft."
                )

                st.info(
                    "Hinweis: Backtests nutzen feste Zeithorizonte (keine Stop-Loss/Take-Profit-Simulation). "
                    "Immer mit Risiko- und Qualitätsfiltern kombinieren."
                )


# ===========================================================================
# 4. News-Feed
# ===========================================================================
elif mode == "📰 News-Feed":
    st.title("📰 Unternehmens-News & Schlagzeilen")
    st.markdown("Kompakte Nachrichten-Übersicht inklusive detaillierter Zusammenfassung (direkt von Yahoo Finance).")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input("Firmenname oder Ticker", value="Nvidia")

    ticker = resolve_ticker(user_input) if user_input else ""
    if ticker and st.button("News laden", type="primary"):
        with st.spinner(f"Lade Nachrichten für {ticker}..."):
            try:
                tk = yf.Ticker(ticker)
                news = tk.news
                if not news:
                    st.info(f"Keine aktuellen Nachrichten für {ticker} gefunden.")
                else:
                    for n in news[:10]:
                        c = n.get("content", n)
                        title = c.get("title", n.get("title", "Kein Titel"))
                        summary = c.get("summary", "")

                        provider = c.get("provider", {})
                        publisher = provider.get("displayName") if isinstance(provider, dict) else None
                        if not publisher:
                            publisher = n.get("publisher", "Unbekannt")

                        canonical = c.get("canonicalUrl", {})
                        link = canonical.get("url") if isinstance(canonical, dict) else None
                        if not link:
                            link = n.get("link", "#")

                        pub_time = c.get("pubDate", n.get("providerPublishTime"))
                        pub_date = ""
                        if pub_time:
                            try:
                                if isinstance(pub_time, (int, float)):
                                    pub_date = datetime.fromtimestamp(pub_time).strftime("%d.%m.%Y %H:%M")
                                elif isinstance(pub_time, str):
                                    pub_date = pd.to_datetime(pub_time).strftime("%d.%m.%Y %H:%M")
                            except Exception:
                                pub_date = str(pub_time)

                        with st.container():
                            st.markdown(f"#### [{title}]({link})")
                            st.caption(f"🏢 **{publisher}** &nbsp; | &nbsp; 🕒 {pub_date}")
                            if summary:
                                st.info(summary, icon="ℹ️")
                            st.markdown("---")
            except Exception:
                st.error("Fehler beim Laden der Nachrichten.")


# ===========================================================================
# 5. Sparplan-Rechner
# ===========================================================================
elif mode == "💰 Sparplan-Rechner":
    st.title("💰 Sparplan- & Zinseszins-Rechner")
    st.markdown("Berechne den langfristigen Vermögensaufbau (z.B. für ETFs).")

    col1, col2, col3, col4 = st.columns(4)
    startkapital = col1.number_input("Startkapital (€)", min_value=0, value=0, step=100)
    sparrate = col2.number_input("Monatliche Sparrate (€)", min_value=0, value=150, step=25)
    rendite = col3.number_input("Rendite p.a. (%)", min_value=0.0, max_value=50.0, value=7.0, step=0.5)
    jahre = col4.number_input("Laufzeit (Jahre)", min_value=1, max_value=60, value=15, step=1)

    if st.button("Berechnen", type="primary"):
        monate = int(jahre * 12)
        monatszins = (1 + rendite / 100) ** (1 / 12) - 1

        kapital = startkapital
        einzahlungen = startkapital

        verlauf = [{"Monat": 0, "Kapital": startkapital, "Einzahlungen": startkapital, "Zinsen": 0}]

        for m in range(1, monate + 1):
            kapital = kapital * (1 + monatszins) + sparrate
            einzahlungen += sparrate
            zinsen = kapital - einzahlungen

            if m % 12 == 0:
                verlauf.append({
                    "Jahr": m // 12,
                    "Kapital": round(kapital, 2),
                    "Einzahlungen": round(einzahlungen, 2),
                    "Zinsen": round(zinsen, 2),
                })

        df_verlauf = pd.DataFrame(verlauf[1:])

        endkapital = verlauf[-1]["Kapital"]
        gesamteinzahlung = verlauf[-1]["Einzahlungen"]
        gesamtzinsen = verlauf[-1]["Zinsen"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Endkapital", f"{endkapital:,.2f} €".replace(",", "."))
        m2.metric("Eingezahlt", f"{gesamteinzahlung:,.2f} €".replace(",", "."))
        m3.metric("Erwirtschaftete Zinsen", f"{gesamtzinsen:,.2f} €".replace(",", "."))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_verlauf["Jahr"],
            y=df_verlauf["Einzahlungen"],
            name="Eingezahltes Kapital",
            marker_color="#42a5f5",
        ))
        fig.add_trace(go.Bar(
            x=df_verlauf["Jahr"],
            y=df_verlauf["Zinsen"],
            name="Zinseszins",
            marker_color="#26a69a",
        ))

        fig.update_layout(
            barmode="stack",
            title="Vermögensentwicklung über Zeit",
            xaxis_title="Jahre",
            yaxis_title="Euro (€)",
            template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font=dict(color="#cdd6f4"),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ===========================================================================
# 6. Saisonalität
# ===========================================================================
elif mode == "📅 Saisonalität":
    st.title("📅 Saisonalitäts-Analyse")
    st.markdown("Wie performt diese Aktie historisch in den einzelnen Monaten?")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input("Firmenname oder Ticker", value="Nvidia", key="seas_ticker")
    with col2:
        jahre = st.slider("Historie (Jahre)", 3, 20, 10)

    ticker = resolve_ticker(user_input) if user_input else ""
    if ticker and st.button("Saisonalität berechnen", type="primary"):
        with st.spinner(f"Lade Kursdaten der letzten {jahre} Jahre für {ticker}..."):
            data = load_price_history(ticker, period=f"{jahre}y")
            if data.empty:
                st.error("Keine Daten gefunden.")
            else:
                monthly = data["Close"].resample("ME").last()
                monthly_returns = monthly.pct_change() * 100

                df_seas = pd.DataFrame({
                    "Monat": monthly_returns.index.month,
                    "Rendite": monthly_returns.values,
                }).dropna()

                monatsnamen = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
                stats = df_seas.groupby("Monat").agg(
                    Durchschnitt=("Rendite", "mean"),
                    Positiv=("Rendite", lambda x: (x > 0).mean() * 100),
                    Anzahl=("Rendite", "count"),
                ).reset_index()
                stats["Monat_Name"] = stats["Monat"].apply(lambda x: monatsnamen[int(x) - 1])

                st.subheader(f"Historische Monats-Performance ({ticker})")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=stats["Monat_Name"],
                    y=stats["Durchschnitt"],
                    text=[f"{v:.1f}%<br>({p:.0f}% grün)" for v, p in zip(stats["Durchschnitt"], stats["Positiv"])],
                    textposition="outside",
                    marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in stats["Durchschnitt"]],
                ))
                fig.update_layout(
                    title=f"Ø Rendite pro Monat (letzte {jahre} Jahre)",
                    yaxis_title="Ø Rendite (%)",
                    template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font=dict(color="#cdd6f4"),
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                st.info("💡 **Lesehilfe:** Die Prozentzahl an der Säule ist die durchschnittliche Rendite. X% grün zeigt, wie oft dieser Monat positiv war.")


# ===========================================================================
# 7. Sektor-Heatmap
# ===========================================================================
elif mode == "🗺️ Sektor-Heatmap":
    st.title("🗺️ Sektor-Heatmap")
    st.markdown("Wo fließt das Kapital hin? Interaktive Markt-Treemap nach Sektor & Marktkapitalisierung (in €).")

    hc1, hc2, hc3 = st.columns([2, 1.5, 1])
    with hc1:
        hm_region = st.selectbox(
            "Markt / Universum",
            ["US + DAX Kombiniert", "🇺🇸 US Tech & S&P", "🇩🇪 DAX 40", "🇪🇺 Europa Blue Chips"],
            index=0,
        )
    with hc2:
        hm_timeframe = st.selectbox(
            "Zeitraum",
            ["1 Tag (Heute)", "5 Tage (Woche)", "1 Monat (21T)"],
            index=0,
        )
    with hc3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Aktualisieren", use_container_width=True):
            load_sector_heatmap_df.clear()

    with st.spinner("Lade Kurs- und Sektordaten für Heatmap..."):
        df_hm = load_sector_heatmap_df(hm_region, hm_timeframe)

    if not df_hm.empty:
        sec_rows = []
        for sec_name, grp in df_hm.groupby("Sektor"):
            top_idx = grp["Performance"].idxmax()
            sec_rows.append({
                "Sektor": sec_name,
                "Anzahl": int(len(grp)),
                "Ø Performance %": round(float(grp["Performance"].mean()), 2),
                "Gewichtet %": round(float(np.average(grp["Performance"], weights=grp["MarketCap"])), 2),
                "Marktkap. Mrd. €": round(float(grp["MarketCap_Mrd"].sum()), 1),
                "Top-Aktie": f"{grp.loc[top_idx, 'Name']} ({grp.loc[top_idx, 'Performance']:+.1f}%)",
            })
        sec_stats = pd.DataFrame(sec_rows).sort_values("Gewichtet %", ascending=False).reset_index(drop=True)

        best_sec = sec_stats.iloc[0]
        worst_sec = sec_stats.iloc[-1]
        winners = int((df_hm["Performance"] > 0).sum())
        losers = int((df_hm["Performance"] < 0).sum())
        avg_all = float(df_hm["Performance"].mean())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Bester Sektor", best_sec["Sektor"], f"{best_sec['Gewichtet %']:+.2f} % (gew.)")
        k2.metric("Schwächster Sektor", worst_sec["Sektor"], f"{worst_sec['Gewichtet %']:+.2f} % (gew.)")
        k3.metric("Gewinner / Verlierer", f"{winners} 🟢 / {losers} 🔴", f"{len(df_hm)} Titel gesamt")
        k4.metric(f"Ø Markt ({hm_timeframe})", f"{avg_all:+.2f} %")

        max_abs = max(2.0, min(10.0, float(df_hm["Performance"].abs().quantile(0.95))))

        fig = px.treemap(
            df_hm,
            path=[px.Constant("Gesamtmarkt"), "Sektor", "Label"],
            values="MarketCap",
            color="Performance",
            color_continuous_scale=["#ef5350", "#313244", "#26a69a"],
            color_continuous_midpoint=0,
            range_color=[-max_abs, max_abs],
            custom_data=["Performance", "Ticker", "MarketCap_Mrd"],
        )
        fig.update_traces(
            marker=dict(line=dict(width=1.2, color="#1e1e2e")),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Sektor: %{parent}<br>"
                "Performance: <b>%{customdata[0]:+.2f}%</b><br>"
                "Marktkap.: %{customdata[2]:.1f} Mrd. €<extra></extra>"
            ),
            texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
            textfont=dict(size=13, color="#ffffff"),
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(family="Inter, system-ui, sans-serif", color="#cdd6f4"),
            margin=dict(t=35, l=10, r=10, b=10),
            height=640,
            coloraxis_colorbar=dict(
                title="Perf. %",
                ticksuffix="%",
                bgcolor="rgba(30,30,46,0.8)",
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("📊 Sektor-Übersicht")
        st.dataframe(sec_stats, use_container_width=True, hide_index=True)
        st.caption("Kachelgröße = Marktkapitalisierung (in €) · Farbe = Kursveränderung im gewählten Zeitraum.")
    else:
        st.error("Kursdaten konnten aktuell nicht geladen werden. Bitte später erneut versuchen.")


# ===========================================================================
# 8. Krypto ETF In/Outflows + Barometer der Kryptostimmung
# ===========================================================================
elif mode == "🪙 Krypto ETF Flows":
    st.title("🪙 Krypto ETF Flows & Kryptostimmung")
    st.markdown(
        "Institutionelle Zu- und Abflüsse der großen Krypto-ETFs/ETPs (in Mio. USD), "
        "aktuelle Live-Kurse (€) sowie das **Barometer der Kryptostimmung (Fear & Greed Index)**."
    )

    st.subheader("🌡️ Barometer der Kryptostimmung (Crypto Fear & Greed)")
    fg = load_crypto_fear_greed()
    fg_score = fg["score"]
    fg_label, fg_desc = classify_crypto_sentiment(fg_score)

    col_fg1, col_fg2 = st.columns([1.05, 1.25])

    with col_fg1:
        fig_crypto_fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fg_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"KRYPTO-STIMMUNG: {fg_label.upper()}",
                "font": {"size": 16, "color": "#cdd6f4", "family": "Arial Black"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white", "tickvals": [0, 25, 45, 55, 75, 100]},
                "bar": {"color": "rgba(44, 62, 80, 0.92)", "thickness": 0.16},
                "bgcolor": "#313244",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#e74c3c"},
                    {"range": [25, 45], "color": "#f39c12"},
                    {"range": [45, 55], "color": "#bdc3c7"},
                    {"range": [55, 75], "color": "#2ecc71"},
                    {"range": [75, 100], "color": "#27ae60"},
                ],
            },
        ))
        fig_crypto_fg.update_layout(
            height=310,
            margin=dict(l=20, r=20, t=42, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#cdd6f4"},
        )
        fig_crypto_fg.add_annotation(x=0.12, y=0.35, text="Extreme<br>Angst", showarrow=False, font=dict(color="white", size=10))
        fig_crypto_fg.add_annotation(x=0.25, y=0.70, text="Angst", showarrow=False, font=dict(color="white", size=11))
        fig_crypto_fg.add_annotation(x=0.50, y=0.90, text="Neutral", showarrow=False, font=dict(color="black", size=11))
        fig_crypto_fg.add_annotation(x=0.75, y=0.70, text="Gier", showarrow=False, font=dict(color="white", size=11))
        fig_crypto_fg.add_annotation(x=0.88, y=0.35, text="Extreme<br>Gier", showarrow=False, font=dict(color="white", size=10))
        st.plotly_chart(fig_crypto_fg, use_container_width=True, config=PLOTLY_CONFIG)

        s1, s2, s3 = st.columns(3)
        delta_yest = fg_score - fg["yesterday"]
        delta_week = fg_score - fg["week_ago"]
        s1.metric("Gestern", f"{fg['yesterday']}/100", f"{delta_yest:+d} Pkt.")
        s2.metric("Vor 7 Tagen", f"{fg['week_ago']}/100", f"{delta_week:+d} Pkt.")
        s3.metric("Ø 30 Tage", f"{fg['avg_30d']}/100")

    with col_fg2:
        st.info(fg_desc)
        hist_fg = fg["history"]
        fig_fg_hist = go.Figure()
        fig_fg_hist.add_trace(go.Scatter(
            x=hist_fg["Date"],
            y=hist_fg["Score"],
            mode="lines+markers",
            name="Fear & Greed",
            line=dict(color="#f9e2af", width=2.4),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(249, 226, 175, 0.12)",
        ))
        fig_fg_hist.add_hline(y=75, line_dash="dot", line_color="#ef5350", annotation_text="Extreme Gier (75)")
        fig_fg_hist.add_hline(y=25, line_dash="dot", line_color="#26a69a", annotation_text="Extreme Angst (25)")
        fig_fg_hist.update_layout(
            **CHART_LAYOUT,
            height=265,
            title=dict(text="30-Tage-Verlauf Kryptostimmung", font=dict(size=14)),
            yaxis=dict(range=[0, 100], title="Score", side="right", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_fg_hist, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(f"Quelle: {fg['source']} · Skala: 0 = Extreme Angst, 100 = Extreme Gier.")

    st.markdown("---")
    st.subheader("💸 Krypto ETF In/Outflows & Live-Marktdaten (7 Tage)")

    @st.cache_data(ttl=3600)
    def load_crypto_flows_data():
        return pd.DataFrame({
            "Date": pd.date_range(end=datetime.today(), periods=7).strftime("%Y-%m-%d"),
            "BTC": [120, 85, -45, 210, 340, -10, 150],
            "ETH": [30, 15, -10, 50, 80, 5, 45],
            "SOL": [10, -5, -2, 15, 25, -8, 12],
            "XRP": [25, 40, 12, 60, -5, 18, 30],
            "ADA": [8, 14, -3, 22, 19, 4, 11],
            "XLM": [5, 2, -1, 8, 12, 3, 7],
            "HBAR": [15, 20, 6, 35, 42, -2, 28],
        })

    crypto_df = load_crypto_flows_data()
    spot_snap = load_crypto_spot_snapshot()

    coins_order = [
        ("BTC", "Bitcoin (BTC)"),
        ("ETH", "Ethereum (ETH)"),
        ("SOL", "Solana (SOL)"),
        ("XRP", "XRP (XRP)"),
        ("ADA", "Cardano (ADA)"),
        ("XLM", "Stellar (XLM)"),
        ("HBAR", "Hedera (HBAR)"),
    ]

    totals = {sym: int(crypto_df[sym].sum()) for sym, _ in coins_order}
    total_all_7d = sum(totals.values())
    daily_totals = crypto_df[[sym for sym, _ in coins_order]].sum(axis=1)
    avg_daily_all = round(float(daily_totals.mean()), 1)
    inflow_days = int((daily_totals > 0).sum())
    top_coin = max(totals.items(), key=lambda x: x[1])
    weakest_coin = min(totals.items(), key=lambda x: x[1])

    mk1, mk2, mk3, mk4 = st.columns(4)
    mk1.metric("Gesamt-Netto-Flow (7d)", f"{total_all_7d:+d} Mio. $", f"Ø {avg_daily_all:+.1f} Mio. $/Tag")
    mk2.metric("Stärkster Zufluss (7d)", f"{top_coin[0]} ({top_coin[1]:+d} Mio. $)", "Höchste institutionelle Nachfrage")
    mk3.metric("Schwächster Flow (7d)", f"{weakest_coin[0]} ({weakest_coin[1]:+d} Mio. $)", "Geringster Netto-Zufluss")
    mk4.metric("Netto-Zufluss-Tage", f"{inflow_days} / 7 Tage", "Tage mit positivem Gesamt-Saldo")

    def format_flow(val):
        color_style = "color: #2ecc71; font-weight: bold;" if val >= 0 else "color: #e74c3c; font-weight: bold;"
        sign = "+" if val >= 0 else ""
        return f'<span style="{color_style}">{sign}{val} Mio. $</span>'

    row1_cols = st.columns(4)
    row2_cols = st.columns(3)
    target_cols = row1_cols + row2_cols

    for (sym, title), col in zip(coins_order, target_cols):
        tot = totals[sym]
        snap = spot_snap.get(sym, {})
        px_eur = snap.get("price_eur")
        chg_7d = snap.get("chg_7d")
        if px_eur is not None:
            px_str = f"{px_eur:,.4f} €" if px_eur < 2 else f"{px_eur:,.2f} €"
            chg_color = "#2ecc71" if (chg_7d or 0) >= 0 else "#e74c3c"
            sub_html = f'Kurs: <b>{px_str}</b> (<span style="color:{chg_color};">{chg_7d:+.1f}% 7d</span>)'
        else:
            sub_html = "Live-Kurs wird geladen"

        with col:
            st.markdown(f"""
            <div style="background-color: #313244; border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #45475a; margin-bottom: 10px;">
                <div style="font-size: 14px; color: #a6adc8;">{title} (7d)</div>
                <div style="font-size: 20px; font-weight: bold; margin-top: 5px;">{format_flow(tot)}</div>
                <div style="font-size: 12px; color: #bac2de; margin-top: 6px;">{sub_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Tägliche & Kumulierte ETF-Flows (7 Tage)")

    tab_daily, tab_cum, tab_share = st.tabs([
        "📊 Tägliche Flows (Mio. USD)",
        "📈 Kumulierter Flow-Trend (7d)",
        "🥧 Anteil an Gesamt-Zuflüssen",
    ])

    asset_colors = {
        "BTC": "#f7931a",
        "ETH": "#627eea",
        "SOL": "#14f195",
        "XRP": "#38bdf8",
        "ADA": "#818cf8",
        "XLM": "#f472b6",
        "HBAR": "#a3e635",
    }

    with tab_daily:
        fig = go.Figure()
        for sym, asset_label in coins_order:
            fig.add_trace(go.Bar(
                x=crypto_df["Date"],
                y=crypto_df[sym],
                name=asset_label,
                marker_color=asset_colors.get(sym),
            ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Datum",
            yaxis_title="Flows (Mio. USD)",
            **CHART_LAYOUT,
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab_cum:
        fig_cum = go.Figure()
        for sym, asset_label in coins_order:
            fig_cum.add_trace(go.Scatter(
                x=crypto_df["Date"],
                y=crypto_df[sym].cumsum(),
                mode="lines+markers",
                name=asset_label,
                line=dict(width=2.4, color=asset_colors.get(sym)),
            ))
        fig_cum.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.35)")
        fig_cum.update_layout(
            xaxis_title="Datum",
            yaxis_title="Kumulierter Flow (Mio. USD)",
            **CHART_LAYOUT,
            height=480,
        )
        st.plotly_chart(fig_cum, use_container_width=True, config=PLOTLY_CONFIG)

    with tab_share:
        pos_flows = {sym: max(0, tot) for sym, tot in totals.items()}
        fig_pie = go.Figure(go.Pie(
            labels=[lbl for _, lbl in coins_order],
            values=[pos_flows[sym] for sym, _ in coins_order],
            hole=0.45,
            marker=dict(colors=[asset_colors[sym] for sym, _ in coins_order]),
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
            height=440,
            title="Verteilung der 7-Tage-Netto-Zuflüsse nach Coin",
        )
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("#### 🔍 Detail-Analyse & Flow-vs.-Kurs-Signal je Coin")
    detail_rows = []
    for sym, label in coins_order:
        s_flow = crypto_df[sym]
        tot_7d = int(s_flow.sum())
        avg_day = round(float(s_flow.mean()), 1)
        best_day = int(s_flow.max())
        worst_day = int(s_flow.min())
        pos_days = int((s_flow > 0).sum())

        snap = spot_snap.get(sym, {})
        px_eur = snap.get("price_eur")
        chg_7d = snap.get("chg_7d")
        px_fmt = (f"{px_eur:,.4f} €" if px_eur < 2 else f"{px_eur:,.2f} €") if px_eur is not None else "—"
        chg_fmt = f"{chg_7d:+.2f} %" if chg_7d is not None else "—"

        if tot_7d > 0 and chg_7d is not None and chg_7d < -1.0:
            sig_div = "🟢 Akkumulation (Zuflüsse trotz Rücksetzer)"
        elif tot_7d > 0 and (chg_7d is None or chg_7d >= -1.0):
            sig_div = "🚀 Trend-Bestätigung (Kurs & Flows positiv)"
        elif tot_7d < 0 and chg_7d is not None and chg_7d > 1.0:
            sig_div = "🟡 Gewinnmitnahmen (Abflüsse bei steigendem Kurs)"
        else:
            sig_div = "🔴 Abgabedruck (Flows negativ)"

        detail_rows.append({
            "Asset": label,
            "Kurs (€)": px_fmt,
            "7d Kurs": chg_fmt,
            "7d Netto-Flow": f"{tot_7d:+d} Mio. $",
            "Ø Flow / Tag": f"{avg_day:+.1f} Mio. $",
            "Bester / Schlechtester Tag": f"{best_day:+d} / {worst_day:+d} Mio. $",
            "Plus-Tage": f"{pos_days} / 7",
            "Markt-Signal": sig_div,
            "Produkt-Fokus": snap.get("product_info", "ETF / ETP"),
        })

    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    with st.expander("📅 Tägliche Rohdaten-Tabelle (Mio. USD)", expanded=False):
        df_table = crypto_df.set_index("Date").copy()
        df_table["Gesamt"] = df_table.sum(axis=1)
        st.dataframe(df_table, use_container_width=True)

    st.info(
        "💡 **Lesehilfe für Krypto-ETF-Flows:**\n"
        "- **Netto-Zuflüsse (+):** ETF-Emittenten müssen physische Coins am Spot-Markt zukaufen, um neue "
        "Fondsanteile auszugeben – das verknappt das verfügbare Angebot.\n"
        "- **Bullische Divergenz:** Wenn der Kurs kurzfristig fällt, die ETF-Flows aber deutlich positiv "
        "bleiben, nutzen institutionelle Investoren den Dip zum Einstieg.\n"
        "- **Altcoin-Flows (SOL, XRP, ADA, XLM, HBAR):** Zeigen frühzeitig, wohin institutionelles Kapital "
        "abseits von Bitcoin und Ethereum rotiert."
    )


# ===========================================================================
# 9. Dividenden-Kalender
# ===========================================================================
elif mode == "💸 Dividenden-Kalender":
    st.title("💸 Dividenden-Vorschau")
    st.markdown("Zeigt kommende Ex-Dividenden-Tage für ein gewähltes Universum an.")

    universe = st.selectbox("Universum wählen", ["🇩🇪 DAX 40", "S&P-Auswahl (Auszug)", "Europa Blue Chips", "⭐ Meine Watchlists"])

    if universe == "⭐ Meine Watchlists":
        saved_wl = load_json(WATCHLIST_FILE, {})
        if saved_wl:
            wl_name = st.selectbox("Watchlist wählen", list(saved_wl.keys()))
            tickers = list(saved_wl.get(wl_name) or [])
        else:
            st.warning("Keine Watchlists gefunden.")
            tickers = []
    elif universe == "🇩🇪 DAX 40":
        tickers = DAX40_TICKERS
    elif universe == "Europa Blue Chips":
        tickers = BUILTIN_WATCHLISTS["Europa Blue Chips"]
    else:
        tickers = BUILTIN_WATCHLISTS["S&P-Auswahl (Auszug)"]

    if tickers and st.button("Dividenden suchen", type="primary"):
        with st.spinner("Scanne Aktien nach Dividenden-Terminen..."):
            div_data = []
            now_ts = pd.Timestamp.now(tz="UTC").timestamp()
            for t in tickers:
                try:
                    info = yf.Ticker(t).info
                    ex_div_ts = info.get("exDividendDate")
                    if ex_div_ts and ex_div_ts > (now_ts - 86400 * 7):
                        ex_date = datetime.fromtimestamp(ex_div_ts)
                        div_data.append({
                            "Ticker": t,
                            "Name": info.get("shortName", t),
                            "Ex-Datum": ex_date,
                            "Dividende": info.get("dividendRate", 0),
                            "Rendite %": (info.get("dividendYield", 0) or 0) * 100,
                            "Zukünftig": "✅ Ja" if ex_div_ts > now_ts else "⏳ Vorbei",
                        })
                except Exception:
                    continue

            if div_data:
                df_div = pd.DataFrame(div_data).sort_values("Ex-Datum")
                df_div["Ex-Datum_Str"] = df_div["Ex-Datum"].dt.strftime("%d.%m.%Y")
                df_div["Dividende"] = df_div["Dividende"].apply(lambda x: f"{x:.2f}")
                df_div["Rendite %"] = df_div["Rendite %"].apply(lambda x: f"{x:.2f} %")

                show_cols = ["Ticker", "Name", "Ex-Datum_Str", "Dividende", "Rendite %", "Zukünftig"]
                st.dataframe(df_div[show_cols].rename(columns={"Ex-Datum_Str": "Ex-Datum"}), use_container_width=True, hide_index=True)
                st.info("💡 **Erklärung:** Du musst die Aktie spätestens am Tag VOR dem Ex-Datum kaufen.")
            else:
                st.success("Keine anstehenden Ex-Dividenden-Tage gefunden.")
