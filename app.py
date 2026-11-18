from datetime import datetime
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

# ==============================================================================
# SEITEN-KONFIGURATION & DESIGN (Stock Screener Pro - Dark-Not-Black Theme)
# ==============================================================================
st.set_page_config(
    page_title="Stock Screener Pro (€)",
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
    .fred-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        min-height: 225px;
        line-height: 1.55;
    }
    .badge-green {
        background-color: rgba(0, 200, 83, 0.16);
        color: #00E676;
        border: 1px solid #00C853;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.86rem;
    }
    .badge-red {
        background-color: rgba(255, 82, 82, 0.16);
        color: #FF5252;
        border: 1px solid #FF5252;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.86rem;
    }
    .badge-amber {
        background-color: rgba(255, 179, 0, 0.16);
        color: #FFD54F;
        border: 1px solid #FFB300;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.86rem;
    }
    .info-box {
        background-color: #161B22;
        border-left: 4px solid #58A6FF;
        padding: 15px 18px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .entry-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-left: 4px solid #00E676;
        padding: 15px 18px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .target-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-left: 4px solid #FFB300;
        padding: 15px 18px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .verdict-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-left: 5px solid #58A6FF;
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
# VOREINGESTELLTE AKTIEN-LISTEN, NAMENS-ZUORDNUNG, SEKTOREN & KRYPTO-DATEN
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

INDEX_UNIVERSEN["Alle Märkte kombiniert (Global-Scan)"] = list(
    dict.fromkeys(
        ticker
        for liste in INDEX_UNIVERSEN.values()
        for ticker in liste
    )
)

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

SEKTOR_ETFS = [
    ("Technologie & Software", "XLK"),
    ("Halbleiter & Chips", "SMH"),
    ("Kommunikation & Medien", "XLC"),
    ("Finanzen & Banken", "XLF"),
    ("Gesundheit & Pharma", "XLV"),
    ("Industrie & Rüstung", "XLI"),
    ("Zyklischer Konsum & E-Auto", "XLY"),
    ("Basiskonsumgüter", "XLP"),
    ("Energie, Öl & Gas", "XLE"),
    ("Rohstoffe & Chemie", "XLB"),
    ("Versorger & Infrastruktur", "XLU"),
    ("Immobilien", "XLRE"),
]

DIVIDENDEN_STANDARD = [
    "ALV.DE", "MUV2.DE", "DTE.DE", "MBG.DE", "BAS.DE",
    "7203.T", "8306.T", "0700.HK", "1211.HK",
    "AAPL", "MSFT", "KO", "JNJ", "PG", "MCD"
]

KRYPTO_LISTE = [
    {"name": "Bitcoin", "symbol": "BTC", "ticker": "BTC-EUR", "etf_basis_m": 420.0},
    {"name": "Ethereum", "symbol": "ETH", "ticker": "ETH-EUR", "etf_basis_m": 115.0},
    {"name": "Solana", "symbol": "SOL", "ticker": "SOL-EUR", "etf_basis_m": 48.0},
    {"name": "XRP", "symbol": "XRP", "ticker": "XRP-EUR", "etf_basis_m": 32.0},
    {"name": "Cardano", "symbol": "ADA", "ticker": "ADA-EUR", "etf_basis_m": 14.0},
    {"name": "Stellar", "symbol": "XLM", "ticker": "XLM-EUR", "etf_basis_m": 6.5},
    {"name": "Hedera", "symbol": "HBAR", "ticker": "HBAR-EUR", "etf_basis_m": 5.0},
]

MONATSNAMEN_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

# ==============================================================================
# HILFSFUNKTIONEN: NAMENSSUCHE, WÄHRUNGSUMRECHNUNG, FRED & INDIKATOREN
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

    if klein in BEKANNTE_AKTIEN_NAMEN:
        return BEKANNTE_AKTIEN_NAMEN[klein]

    if len(klein) >= 3:
        for schluessel, (tkr, vollname) in BEKANNTE_AKTIEN_NAMEN.items():
            if klein in schluessel or schluessel in klein:
                return tkr, vollname

    if "." in sauber or "-" in sauber or (sauber.isupper() and len(sauber) <= 5 and " " not in sauber):
        return sauber.upper(), sauber.upper()

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
    if w == "GBX":
        return hole_wechselkurs_zu_eur("GBP") / 100.0

    try:
        df_dir = bereinige_dataframe(yf.download(f"{w}EUR=X", period="5d", interval="1d", progress=False))
        if not df_dir.empty and "Close" in df_dir.columns:
            kurs = float(df_dir["Close"].dropna().iloc[-1])
            if kurs > 0:
                return kurs
    except Exception:
        pass

    try:
        df_inv = bereinige_dataframe(yf.download(f"EUR{w}=X", period="5d", interval="1d", progress=False))
        if not df_inv.empty and "Close" in df_inv.columns:
            inv_kurs = float(df_inv["Close"].dropna().iloc[-1])
            if inv_kurs > 0:
                return 1.0 / inv_kurs
    except Exception:
        pass

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
    """Holt Firmennamen, Währung, Sektor, Analysten-Kursziele, Dividende und Earnings-Datum."""
    standard_waehrung = leite_waehrung_aus_ticker_ab(ticker)
    info_daten = {
        "name": ticker,
        "waehrung": standard_waehrung,
        "sektor": "Keine Angabe",
        "earnings_datum": None,
        "div_rendite_pct": 0.0,
        "div_rate_orig": 0.0,
        "ex_div_datum": None,
        "payout_ratio_pct": None,
        "target_mean_orig": None,
        "target_high_orig": None,
        "target_low_orig": None,
        "analysten_anzahl": 0,
        "analysten_urteil": None,
    }

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

        # Analysten-Kursziele
        if info.get("targetMeanPrice"):
            info_daten["target_mean_orig"] = float(info["targetMeanPrice"])
        if info.get("targetHighPrice"):
            info_daten["target_high_orig"] = float(info["targetHighPrice"])
        if info.get("targetLowPrice"):
            info_daten["target_low_orig"] = float(info["targetLowPrice"])
        if info.get("numberOfAnalystOpinions"):
            info_daten["analysten_anzahl"] = int(info["numberOfAnalystOpinions"])
        if info.get("recommendationKey"):
            info_daten["analysten_urteil"] = str(info["recommendationKey"]).upper()

        div_yield = info.get("dividendYield")
        if div_yield is not None:
            info_daten["div_rendite_pct"] = round(div_yield * 100, 2) if div_yield < 0.35 else round(div_yield, 2)

        if info.get("dividendRate"):
            info_daten["div_rate_orig"] = float(info["dividendRate"])

        if info.get("payoutRatio") is not None:
            info_daten["payout_ratio_pct"] = round(float(info["payoutRatio"]) * 100, 1)

        if info.get("exDividendDate"):
            try:
                ts = int(info["exDividendDate"])
                info_daten["ex_div_datum"] = datetime.fromtimestamp(ts).strftime("%d.%m.%Y")
            except Exception:
                pass

        cal = t.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal and cal["Earnings Date"]:
            ed = cal["Earnings Date"][0]
            info_daten["earnings_datum"] = str(ed)[:10]
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            info_daten["earnings_datum"] = str(cal.iloc[0, 0])[:10]
    except Exception:
        pass

    return info_daten


# ==============================================================================
# FRED REZESSIONS-BAROMETER (OHNE API-KEY – ALLE 6 MAKRO-INDIKATOREN IM DETAIL)
# ==============================================================================
@st.cache_data(ttl=21600, show_spinner=False)
def hole_fred_serie(series_id: str) -> pd.Series:
    """Lädt eine historische Wirtschaftsreihe direkt von der US-Notenbank FRED (ohne API-Key)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/csv,text/plain,*/*",
            },
            timeout=10,
        )
        if resp.status_code != 200 or not resp.text or "<html" in resp.text.lower():
            return pd.Series(dtype=float)
        df = pd.read_csv(io.StringIO(resp.text))
        if df.shape[1] < 2:
            return pd.Series(dtype=float)
        datum_spalte = df.columns[0]
        wert_spalte = df.columns[1]
        df[datum_spalte] = pd.to_datetime(df[datum_spalte], errors="coerce")
        df[wert_spalte] = pd.to_numeric(df[wert_spalte].replace(".", np.nan), errors="coerce")
        df = df.dropna(subset=[datum_spalte, wert_spalte]).set_index(datum_spalte)
        return df[wert_spalte].sort_index()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=21600, show_spinner=False)
def berechne_rezessions_barometer() -> dict:
    """
    Analysiert alle 6 zentralen US-Notenbank-Indikatoren (FRED) mit ausführlichen
    Erklärungen, Schwellenwerten, Ampel-Logik und Portfolio-Empfehlungen.
    """
    detail_karten = []
    tabellen_zeilen = []
    risiko_punkte = 0
    verfuegbare_gewichte = 0

    # 1. Sahm-Regel Rezessionsindikator (SAHMREALTIME) – Gewicht: 25 Punkte
    sahm = hole_fred_serie("SAHMREALTIME")
    if not sahm.empty:
        verfuegbare_gewichte += 25
        v = float(sahm.iloc[-1])
        stand = sahm.index[-1].strftime("%m/%Y")
        if v >= 0.50:
            risiko_punkte += 25
            ampel_farbe, ampel_text = "red", "🔴 Warnsignal (Rezessions-Trigger ausgelöst)"
            status_detail = (
                f"Mit aktuell <b>{v:.2f} %-Punkten</b> liegt die Sahm-Regel über der kritischen Marke von 0,50 %-Punkten. "
                "In jedem US-Konjunkturzyklus seit 1970 ging das Überschreiten dieser Schwelle mit einer Rezession einher."
            )
        elif v >= 0.30:
            risiko_punkte += 14
            ampel_farbe, ampel_text = "amber", "🟡 Beobachtung (Leicht erhöht)"
            status_detail = (
                f"Der Wert steht bei <b>{v:.2f} %-Punkten</b> und nähert sich langsam der Warnschwelle (0,50). "
                "Der US-Arbeitsmarkt kühlt spürbar ab, befindet sich aber noch nicht im akuten Rezessionsbereich."
            )
        else:
            risiko_punkte += 2
            ampel_farbe, ampel_text = "green", "🟢 Entspannt (Unter Schwelle)"
            status_detail = (
                f"Mit <b>{v:.2f} %-Punkten</b> liegt der Indikator deutlich unter der Warnschwelle von 0,50 %-Punkten. "
                "Der Arbeitsmarkt zeigt aktuell keinen plötzlichen Einbruch."
            )
    else:
        v, stand = 0.23, "Fallback"
        ampel_farbe, ampel_text = "green", "🟢 Entspannt (Richtwert)"
        status_detail = "FRED-Liveabruf kurzzeitig nicht erreichbar – Richtwert unterhalb der 0,50 %-Schwelle."

    detail_karten.append({
        "titel": "1. Sahm-Regel (Arbeitsmarkt-Trigger)",
        "code": "FRED: SAHMREALTIME",
        "wert": f"{v:.2f} %-Punkte",
        "schwelle": "Kritisch ab ≥ 0,50 %-Punkte",
        "stand": stand,
        "ampel_farbe": ampel_farbe,
        "ampel_text": ampel_text,
        "was_ist_das": "Misst den Anstieg des 3-Monats-Durchschnitts der US-Arbeitslosenquote gegenüber dem tiefsten Wert der letzten 12 Monate.",
        "bewertung": status_detail,
    })
    tabellen_zeilen.append({
        "Indikator": "1. Sahm-Regel (SAHMREALTIME)",
        "Aktueller Wert": f"{v:.2f} %-Pkt.",
        "Kritische Schwelle": "≥ 0,50 %-Pkt.",
        "Ampel": ampel_text,
        "Stand": stand,
    })

    # 2. 10J–2J US-Zinskurve (T10Y2Y) – Gewicht: 20 Punkte
    t10y2y = hole_fred_serie("T10Y2Y")
    if not t10y2y.empty:
        verfuegbare_gewichte += 20
        v2 = float(t10y2y.iloc[-1])
        min_18m = float(t10y2y.tail(380).min())
        stand2 = t10y2y.index[-1].strftime("%d.%m.%Y")
        if v2 < 0.0:
            risiko_punkte += 18
            ampel_farbe2, ampel_text2 = "red", "🔴 Warnsignal (Inverse Zinskurve)"
            status_detail2 = (
                f"Der Zinsabstand beträgt <b>{v2:+.2f} %</b> (negativ). Kurzfristige Anleihen werfen mehr ab als 10-jährige – "
                "ein historisch sehr zuverlässiges Frühwarnsignal für eine bevorstehende Konjunkturabkühlung."
            )
        elif min_18m < -0.10 and v2 >= 0.0:
            risiko_punkte += 12
            ampel_farbe2, ampel_text2 = "amber", "🟡 Beobachtung (Re-Steepening nach Inversion)"
            status_detail2 = (
                f"Der Spread liegt bei <b>{v2:+.2f} %</b>, nachdem die Kurve zuvor invertiert war. Historisch beginnt die "
                "eigentliche Wirtschaftsflaute oft erst dann, wenn die Zinskurve nach einer Inversion wieder ins Plus dreht."
            )
        else:
            risiko_punkte += 3
            ampel_farbe2, ampel_text2 = "green", "🟢 Entspannt (Normale Zinskurve)"
            status_detail2 = (
                f"Mit <b>{v2:+.2f} %</b> ist die Zinskurve normal und positiv geneigt. Langfristige Investitionen werden "
                "wieder höher verzinst als kurzfristige Gelder."
            )
    else:
        v2, stand2 = 0.25, "Fallback"
        ampel_farbe2, ampel_text2 = "amber", "🟡 Beobachtung (Richtwert)"
        status_detail2 = "FRED-Liveabruf kurzzeitig nicht erreichbar – Richtwert im leicht positiven Bereich."

    detail_karten.append({
        "titel": "2. US-Zinskurve (10 Jahre minus 2 Jahre)",
        "code": "FRED: T10Y2Y",
        "wert": f"{v2:+.2f} %",
        "schwelle": "Kritisch bei < 0,00 % (Inversion)",
        "stand": stand2,
        "ampel_farbe": ampel_farbe2,
        "ampel_text": ampel_text2,
        "was_ist_das": "Vergleicht die Rendite 10-jähriger US-Staatsanleihen mit 2-jährigen Anleihen. Eine Inversion (< 0 %) ging fast jeder US-Rezession 12–18 Monate voraus.",
        "bewertung": status_detail2,
    })
    tabellen_zeilen.append({
        "Indikator": "2. Zinskurve 10J–2J (T10Y2Y)",
        "Aktueller Wert": f"{v2:+.2f} %",
        "Kritische Schwelle": "< 0,00 %",
        "Ampel": ampel_text2,
        "Stand": stand2,
    })

    # 3. 10J–3M US-Zinskurve (T10Y3M, Fallback über Yahoo ^TNX & ^IRX) – Gewicht: 15 Punkte
    t10y3m = hole_fred_serie("T10Y3M")
    if t10y3m.empty:
        try:
            tnx = bereinige_dataframe(yf.download("^TNX", period="1mo", interval="1d", progress=False))
            irx = bereinige_dataframe(yf.download("^IRX", period="1mo", interval="1d", progress=False))
            if not tnx.empty and not irx.empty:
                diff_s = (tnx["Close"] - irx["Close"]).dropna()
                t10y3m = diff_s
        except Exception:
            pass

    if not t10y3m.empty:
        verfuegbare_gewichte += 15
        v3 = float(t10y3m.iloc[-1])
        stand3 = t10y3m.index[-1].strftime("%d.%m.%Y")
        if v3 < 0.0:
            risiko_punkte += 14
            ampel_farbe3, ampel_text3 = "red", "🔴 Warnsignal (Restriktive Fed-Zinsen)"
            status_detail3 = (
                f"Der Abstand zwischen 10-jährigen und 3-monatigen US-Zinsen liegt bei <b>{v3:+.2f} %</b>. "
                "Das von der US-Notenbank Fed bevorzugte Rezessionsmodell signalisiert damit spürbaren Bremsdruck."
            )
        elif v3 < 0.45:
            risiko_punkte += 7
            ampel_farbe3, ampel_text3 = "amber", "🟡 Beobachtung (Flache Zinsstruktur)"
            status_detail3 = (
                f"Mit <b>{v3:+.2f} %</b> ist der Zinsabstand noch recht flach. Die Geldpolitik wirkt neutral bis leicht bremsend."
            )
        else:
            risiko_punkte += 2
            ampel_farbe3, ampel_text3 = "green", "🟢 Entspannt (Gesunder Zinsabstand)"
            status_detail3 = (
                f"Mit <b>{v3:+.2f} %</b> notieren langfristige Renditen klar über dem Geldmarktzins – Banken und Kreditmärkte arbeiten normal."
            )
    else:
        v3, stand3 = 0.15, "Fallback"
        ampel_farbe3, ampel_text3 = "amber", "🟡 Beobachtung (Richtwert)"
        status_detail3 = "Liveabruf kurzzeitig nicht erreichbar."

    detail_karten.append({
        "titel": "3. Fed-Zinsstruktur (10 Jahre minus 3 Monate)",
        "code": "FRED: T10Y3M",
        "wert": f"{v3:+.2f} %",
        "schwelle": "Kritisch bei < 0,00 %",
        "stand": stand3,
        "ampel_farbe": ampel_farbe3,
        "ampel_text": ampel_text3,
        "was_ist_das": "Der bevorzugte Zinsindikator der Federal Reserve: Misst die Spannung zwischen dem aktuellen Leitzinsniveau (3 Monate) und den langfristigen Wachstumserwartungen (10 Jahre).",
        "bewertung": status_detail3,
    })
    tabellen_zeilen.append({
        "Indikator": "3. Fed-Zinsstruktur 10J–3M (T10Y3M)",
        "Aktueller Wert": f"{v3:+.2f} %",
        "Kritische Schwelle": "< 0,00 %",
        "Ampel": ampel_text3,
        "Stand": stand3,
    })

    # 4. US-Arbeitslosenquote (UNRATE) – Gewicht: 15 Punkte
    unrate = hole_fred_serie("UNRATE")
    if not unrate.empty and len(unrate) >= 13:
        verfuegbare_gewichte += 15
        v4 = float(unrate.iloc[-1])
        vor_12m = float(unrate.iloc[-13])
        diff_12m = v4 - vor_12m
        stand4 = unrate.index[-1].strftime("%m/%Y")
        if diff_12m >= 0.5 or v4 >= 5.0:
            risiko_punkte += 14
            ampel_farbe4, ampel_text4 = "red", "🔴 Warnsignal (Arbeitslosigkeit steigt deutlich)"
            status_detail4 = (
                f"Die Quote liegt bei <b>{v4:.1f} %</b> und ist im Jahresvergleich um <b>{diff_12m:+.1f} %-Punkte</b> gestiegen. "
                "Steigende Arbeitslosigkeit dämpft direkt den privaten Konsum (rund 70 % der US-Wirtschaft)."
            )
        elif diff_12m >= 0.2:
            risiko_punkte += 8
            ampel_farbe4, ampel_text4 = "amber", "🟡 Beobachtung (Leichter Aufwärtstrend)"
            status_detail4 = (
                f"Mit <b>{v4:.1f} %</b> ({diff_12m:+.1f} %-Punkte gegenüber Vorjahr) bleibt das Niveau zwar historisch niedrig, "
                "aber die Dynamik am Stellenmarkt hat sich merklich verlangsamt."
            )
        else:
            risiko_punkte += 2
            ampel_farbe4, ampel_text4 = "green", "🟢 Entspannt (Stabiler Arbeitsmarkt)"
            status_detail4 = (
                f"Die Arbeitslosenquote liegt bei soliden <b>{v4:.1f} %</b> ({diff_12m:+.1f} %-Punkte zum Vorjahr) – "
                "Verbraucher verfügen über stabile Einkommen."
            )
    else:
        v4, diff_12m, stand4 = 4.1, 0.1, "Fallback"
        ampel_farbe4, ampel_text4 = "green", "🟢 Entspannt (Richtwert)"
        status_detail4 = "Stabile US-Beschäftigungslage."

    detail_karten.append({
        "titel": "4. US-Arbeitslosenquote (UNRATE)",
        "code": "FRED: UNRATE",
        "wert": f"{v4:.1f} % ({diff_12m:+.1f} % YoY)",
        "schwelle": "Kritisch ab +0,5 %-Pkt. YoY / ≥ 5,0 %",
        "stand": stand4,
        "ampel_farbe": ampel_farbe4,
        "ampel_text": ampel_text4,
        "was_ist_das": "Offizielle US-Arbeitslosenquote und ihre Veränderung gegenüber dem Vorjahr. Ein schneller Anstieg löst eine Abwärtsspirale aus Konsumrückgang und Entlassungen aus.",
        "bewertung": status_detail4,
    })
    tabellen_zeilen.append({
        "Indikator": "4. US-Arbeitslosenquote (UNRATE)",
        "Aktueller Wert": f"{v4:.1f} % ({diff_12m:+.1f} % YoY)",
        "Kritische Schwelle": "+0,5 %-Pkt. YoY",
        "Ampel": ampel_text4,
        "Stand": stand4,
    })

    # 5. High-Yield Credit Spread (BAMLH0A0HYM2) – Gewicht: 15 Punkte
    hy = hole_fred_serie("BAMLH0A0HYM2")
    if not hy.empty:
        verfuegbare_gewichte += 15
        v5 = float(hy.iloc[-1])
        stand5 = hy.index[-1].strftime("%d.%m.%Y")
        if v5 >= 5.0:
            risiko_punkte += 15
            ampel_farbe5, ampel_text5 = "red", "🔴 Warnsignal (Hoher Kreditstress)"
            status_detail5 = (
                f"Der Risikoaufschlag liegt bei <b>{v5:.2f} %</b> (über 5,00 %). Investoren verlangen massive Prämien für "
                "Unternehmensanleihen – ein Zeichen für akute Ausfallängste und teure Refinanzierung."
            )
        elif v5 >= 3.8:
            risiko_punkte += 8
            ampel_farbe5, ampel_text5 = "amber", "🟡 Beobachtung (Leicht erhöht)"
            status_detail5 = (
                f"Mit <b>{v5:.2f} %</b> ziehen die Risikoaufschläge leicht an. Noch keine Panik am Kreditmarkt, "
                "aber hochverschuldete Unternehmen geraten zunehmend unter Druck."
            )
        else:
            risiko_punkte += 2
            ampel_farbe5, ampel_text5 = "green", "🟢 Entspannt (Niedrige Risikoaufschläge)"
            status_detail5 = (
                f"Mit nur <b>{v5:.2f} %</b> signalisiert der Anleihemarkt hohe Ruhe. Unternehmen können sich problemlos "
                "und günstig refinanzieren."
            )
    else:
        v5, stand5 = 3.10, "Fallback"
        ampel_farbe5, ampel_text5 = "green", "🟢 Entspannt (Richtwert)"
        status_detail5 = "Anleihemärkte ohne akuten Kreditstress."

    detail_karten.append({
        "titel": "5. High-Yield Credit Spread (Kreditrisiko)",
        "code": "FRED: BAMLH0A0HYM2",
        "wert": f"{v5:.2f} %",
        "schwelle": "Kritisch ab ≥ 5,00 %",
        "stand": stand5,
        "ampel_farbe": ampel_farbe5,
        "ampel_text": ampel_text5,
        "was_ist_das": "Zinsaufschlag, den schwächer bewertete US-Unternehmen gegenüber sicheren Staatsanleihen zahlen müssen. Reagiert oft früher als der Aktienmarkt auf Finanzkrisen.",
        "bewertung": status_detail5,
    })
    tabellen_zeilen.append({
        "Indikator": "5. High-Yield Credit Spread",
        "Aktueller Wert": f"{v5:.2f} %",
        "Kritische Schwelle": "≥ 5,00 %",
        "Ampel": ampel_text5,
        "Stand": stand5,
    })

    # 6. US-Industrieproduktion YoY (INDPRO) – Gewicht: 10 Punkte
    indpro = hole_fred_serie("INDPRO")
    if not indpro.empty and len(indpro) >= 13:
        verfuegbare_gewichte += 10
        v6 = float(indpro.iloc[-1])
        vor_12m6 = float(indpro.iloc[-13])
        yoy_pct = ((v6 - vor_12m6) / vor_12m6) * 100 if vor_12m6 > 0 else 0.0
        stand6 = indpro.index[-1].strftime("%m/%Y")
        if yoy_pct <= -1.5:
            risiko_punkte += 10
            ampel_farbe6, ampel_text6 = "red", "🔴 Warnsignal (Industrie schrumpft)"
            status_detail6 = (
                f"Die US-Industrieproduktion liegt <b>{yoy_pct:+.2f} %</b> unter dem Vorjahresniveau. "
                "Ein Rückgang von mehr als 1,5 % deutet auf sinkende Auftragseingänge und fallende Unternehmensgewinne hin."
            )
        elif yoy_pct < 0.5:
            risiko_punkte += 5
            ampel_farbe6, ampel_text6 = "amber", "🟡 Beobachtung (Stagnation)"
            status_detail6 = (
                f"Mit <b>{yoy_pct:+.2f} %</b> gegenüber dem Vorjahr tritt die Industrie auf der Stelle – "
                "weder Boom noch akuter Einbruch."
            )
        else:
            risiko_punkte += 1
            ampel_farbe6, ampel_text6 = "green", "🟢 Entspannt (Solides Wachstum)"
            status_detail6 = (
                f"Die Industrieproduktion wächst um <b>{yoy_pct:+.2f} %</b> im Jahresvergleich und stützt damit "
                "die Gewinne zyklischer Unternehmen."
            )
    else:
        yoy_pct, stand6 = 0.8, "Fallback"
        ampel_farbe6, ampel_text6 = "green", "🟢 Entspannt (Richtwert)"
        status_detail6 = "Industrieproduktion im leicht positiven Bereich."

    detail_karten.append({
        "titel": "6. US-Industrieproduktion (Jahresrate)",
        "code": "FRED: INDPRO",
        "wert": f"{yoy_pct:+.2f} % YoY",
        "schwelle": "Kritisch ab ≤ -1,50 % YoY",
        "stand": stand6,
        "ampel_farbe": ampel_farbe6,
        "ampel_text": ampel_text6,
        "was_ist_das": "Misst den realen Output von Fabriken, Bergbau und Energieversorgern im Vergleich zum Vorjahresmonat – ein harter Konjunktur-Gradmesser ohne Preiseffekte.",
        "bewertung": status_detail6,
    })
    tabellen_zeilen.append({
        "Indikator": "6. US-Industrieproduktion (INDPRO)",
        "Aktueller Wert": f"{yoy_pct:+.2f} % YoY",
        "Kritische Schwelle": "≤ -1,50 % YoY",
        "Ampel": ampel_text6,
        "Stand": stand6,
    })

    if verfuegbare_gewichte > 0:
        score = int(np.clip(round((risiko_punkte / verfuegbare_gewichte) * 100), 5, 95))
    else:
        score = 28

    if score < 35:
        ampel_gesamt = "🟢 NIEDRIGES REZESSIONSRISIKO (Wachstums- & Expansionsphase)"
        strategie = (
            "<b>1. Aktienquote & Ausrichtung:</b> Volle Investitionsquote (85–100 %) vertretbar. Fokus auf Trend-Follower, "
            "Golden-Cross-Ausbrüche, Technologie, Halbleiter und zyklische Marktführer.<br>"
            "<b>2. Einstiege & Risikomanagement:</b> Rücksetzer an die 50-Tage-Linie (SMA 50) aktiv als Kaufchance nutzen; "
            "regulärer ATR-Stop-Loss (2× ATR) reicht zur Absicherung aus.<br>"
            "<b>3. Cash-Reserve:</b> 5–15 % taktische Reserve für spontane Chancen."
        )
    elif score < 62:
        ampel_gesamt = "🟡 ERHÖHTE AUFMERKSAMKEIT (Spätzyklische Übergangsphase)"
        strategie = (
            "<b>1. Aktienquote & Ausrichtung:</b> Selektiv vorgehen (70–85 % investiert). Qualitätsaktien mit hoher "
            "Preissetzungsmacht, starken Bilanzen und nachhaltigen Dividenden bevorzugen.<br>"
            "<b>2. Einstiege & Risikomanagement:</b> Keine überkauften Kurse (RSI > 70) jagen; Einstiege auf zwei Tranchen "
            "aufteilen und auf Turnaround-Signale (TFA) mit engem Stop-Loss achten.<br>"
            "<b>3. Cash-Reserve:</b> 15–30 % Cash bereithalten, um bei Marktkorrekturen günstig zuzukaufen."
        )
    else:
        ampel_gesamt = "🔴 ERHÖHTES REZESSIONSRISIKO (Kapitalschutz & Defensiv-Modus)"
        strategie = (
            "<b>1. Aktienquote & Ausrichtung:</b> Risiko reduzieren (50–65 % Aktienquote). Hochverschuldete Zykliker abbauen "
            "und auf defensive Branchen (Gesundheit, Basiskonsum, Versorger) umschichten.<br>"
            "<b>2. Einstiege & Risikomanagement:</b> Gewinne konsequent mit engen Stop-Loss-Marken absichern. Neue Käufe erst "
            "nach klarer Bodenbildung und positiver Sahm-/Kreditmarkt-Entspannung.<br>"
            "<b>3. Cash-Reserve:</b> 35–50 % Liquidität zum Schutz vor stärkeren Drawdowns."
        )

    return {
        "score": score,
        "ampel_gesamt": ampel_gesamt,
        "strategie": strategie,
        "detail_karten": detail_karten,
        "tabelle": pd.DataFrame(tabellen_zeilen),
        "sahm_serie": sahm.tail(60) if not sahm.empty else pd.Series(dtype=float),
        "zinskurve_serie": t10y2y.tail(750) if not t10y2y.empty else pd.Series(dtype=float),
        "hy_serie": hy.tail(750) if not hy.empty else pd.Series(dtype=float),
    }


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

        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        delta = df["Close"].diff()
        gewinn = delta.clip(lower=0)
        verlust = -delta.clip(upper=0)
        avg_gewinn = gewinn.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_verlust = verlust.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gewinn / avg_verlust.replace(0, np.nan)
        df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50)

        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = true_range.rolling(window=14).mean()

        df["Vol_SMA_20"] = df["Volume"].rolling(window=20).mean()
        return df
    except Exception:
        return pd.DataFrame()


def analysiere_aktie(ticker: str) -> dict | None:
    """
    Erstellt eine ausführliche Analyse in Euro (€) inklusive technischer & fundamentaler
    Kursziele, konkreter Einstiegszonen (Pullback & Ausbruch) und detaillierter Begründung.
    """
    df = lade_kursdaten_eur(ticker, zeitraum="1y")
    if df.empty or len(df) < 50:
        return None

    stamm = hole_stammdaten(ticker)
    faktor = hole_wechselkurs_zu_eur(stamm["waehrung"])
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
    hoch_20t = float(df["High"].tail(20).max())
    tief_20t = float(df["Low"].tail(20).min())
    abstand_hoch_pct = ((aktuell - hoch_52w) / hoch_52w) * 100
    abstand_tief_pct = ((aktuell - tief_52w) / tief_52w) * 100 if tief_52w > 0 else 0.0

    # --------------------------------------------------------------------------
    # KURSIZIELE (€) – TECHNISCH + ANALYSTEN-KONSENS
    # --------------------------------------------------------------------------
    # Kursziel 1: Kurz-/Mittelfristiges Widerstandsziel (52W-Hoch bzw. Ausbruchsprojektion)
    if aktuell >= hoch_52w * 0.96:
        kursziel_1_eur = round(hoch_52w + 2.5 * atr, 2)
        kursziel_1_label = "Technisches Ausbruchsziel (+2,5× ATR über 52W-Hoch)"
    else:
        kursziel_1_eur = round(hoch_52w, 2)
        kursziel_1_label = "52-Wochen-Hoch (Widerstandsziel)"
    potenzial_1_pct = ((kursziel_1_eur - aktuell) / aktuell) * 100

    # Kursziel 2: Analysten-Konsens in € (falls vorhanden), sonst Trendkanal-Projektion
    analysten_mean_eur = (
        round(stamm["target_mean_orig"] * faktor, 2)
        if stamm["target_mean_orig"] and stamm["target_mean_orig"] > 0
        else None
    )
    analysten_high_eur = (
        round(stamm["target_high_orig"] * faktor, 2)
        if stamm["target_high_orig"] and stamm["target_high_orig"] > 0
        else None
    )
    analysten_low_eur = (
        round(stamm["target_low_orig"] * faktor, 2)
        if stamm["target_low_orig"] and stamm["target_low_orig"] > 0
        else None
    )

    if analysten_mean_eur and analysten_mean_eur > aktuell * 0.60:
        kursziel_2_eur = analysten_mean_eur
        anz = stamm["analysten_anzahl"]
        kursziel_2_label = f"Analysten-Konsens ({anz} Analysten)" if anz > 0 else "Analysten-Konsens"
    else:
        kursziel_2_eur = round(max(kursziel_1_eur * 1.12, aktuell + 4.0 * atr), 2)
        kursziel_2_label = "Mittelfristiges Trendkanal-Ziel"
    potenzial_2_pct = ((kursziel_2_eur - aktuell) / aktuell) * 100

    # --------------------------------------------------------------------------
    # KONKRETE EINSTIEGSCHANCEN / KAUFZONEN (€)
    # --------------------------------------------------------------------------
    stop_loss_eur = max(0.01, round(aktuell - 2.0 * atr, 2))
    stop_puffer_pct = ((aktuell - stop_loss_eur) / aktuell) * 100

    if aktuell > sma50:
        # Aufwärtstrend: Idealer Pullback-Einstieg nahe SMA 50 oder 20-Tage-Tief
        basis_support = max(sma50, tief_20t)
        kaufzone_von = round(max(stop_loss_eur * 1.01, basis_support - 0.4 * atr), 2)
        kaufzone_bis = round(min(aktuell * 0.995, basis_support + 0.5 * atr), 2)
        if kaufzone_von >= kaufzone_bis:
            kaufzone_von = round(aktuell - 1.2 * atr, 2)
            kaufzone_bis = round(aktuell - 0.4 * atr, 2)
        ausbruch_trigger = round(hoch_20t + 0.2 * atr, 2)
        einstieg_kurz = f"{kaufzone_von:.2f} – {kaufzone_bis:.2f} €"
        einstieg_strategie_text = (
            f"<b>1. Idealer Rücksetzer (Pullback-Kaufzone):</b> <b>{kaufzone_von:.2f} € bis {kaufzone_bis:.2f} €</b> "
            f"(Unterstützungsbereich nahe SMA 50 / 20-Tage-Tief). Hier stimmt das Chance-Risiko-Verhältnis am besten.<br>"
            f"<b>2. Ausbruchs-Einstieg (Momentum-Trigger):</b> Bei Tagesschlusskurs über <b>{ausbruch_trigger:.2f} €</b> "
            f"unter erhöhtem Volumen bietet sich ein prozyklischer Einstieg an."
        )
    elif aktuell >= sma200:
        # Zwischen SMA 200 und SMA 50
        kaufzone_von = round(max(stop_loss_eur * 1.01, sma200 - 0.3 * atr), 2)
        kaufzone_bis = round(sma200 + 0.6 * atr, 2)
        if kaufzone_von >= kaufzone_bis:
            kaufzone_von = round(aktuell * 0.96, 2)
            kaufzone_bis = round(aktuell * 0.99, 2)
        ausbruch_trigger = round(sma50 + 0.25 * atr, 2)
        einstieg_kurz = f"{kaufzone_von:.2f} – {kaufzone_bis:.2f} € (oder > {ausbruch_trigger:.2f} €)"
        einstieg_strategie_text = (
            f"<b>1. Unterstützungs-Zone (nahe SMA 200):</b> <b>{kaufzone_von:.2f} € bis {kaufzone_bis:.2f} €</b> – "
            f"solange die 200-Tage-Linie ({sma200:.2f} €) hält, ist hier ein gestaffelter Ersteinstieg möglich.<br>"
            f"<b>2. Bestätigungs-Kaufsignal:</b> Erst wenn der Kurs die 50-Tage-Linie bei <b>{ausbruch_trigger:.2f} €</b> "
            f"dynamisch zurückerobert, ist der Aufwärtstrend wieder aktiv."
        )
    else:
        # Unter SMA 50 und SMA 200 (Abwärtstrend / Turnaround-Suche)
        kaufzone_von = round(max(tief_52w * 0.99, aktuell - 1.3 * atr), 2)
        kaufzone_bis = round(aktuell - 0.3 * atr, 2)
        ausbruch_trigger = round(sma50 + 0.25 * atr, 2)
        einstieg_kurz = f"Erst über {ausbruch_trigger:.2f} € (TFA: {kaufzone_von:.2f}–{kaufzone_bis:.2f} €)"
        einstieg_strategie_text = (
            f"<b>1. Konservativer Einstieg (Trendwende abwarten):</b> Erst bei Rückeroberung der 50-Tage-Linie über "
            f"<b>{ausbruch_trigger:.2f} €</b> einsteigen, um nicht in ein fallendes Messer zu greifen.<br>"
            f"<b>2. Antizyklische TFA-Abstauberzone:</b> Nur für mutige Turnaround-Anleger in kleinen Tranchen zwischen "
            f"<b>{kaufzone_von:.2f} € und {kaufzone_bis:.2f} €</b> mit striktem Stop-Loss bei <b>{stop_loss_eur:.2f} €</b>."
        )

    risiko_eur = max(0.01, aktuell - stop_loss_eur)
    bestes_ziel = max(kursziel_1_eur, kursziel_2_eur)
    chance_eur = max(0.0, bestes_ziel - aktuell)
    crv = round(chance_eur / risiko_eur, 2)

    # Golden Cross, Trend-Follower & TFA Check
    golden_cross_neu = False
    if len(df) >= 8 and not df["SMA_200"].iloc[-8:].isna().any():
        golden_cross_neu = (df["SMA_50"].iloc[-8] <= df["SMA_200"].iloc[-8]) and (sma50 > sma200)
    golden_cross_nah = (not golden_cross_neu) and (-2.5 <= sma50_zu_200_pct < 0.0) and (sma50 > sma50_vor5)

    ist_trend_follower = (aktuell > sma50) and (sma50 > sma200) and (45 <= rsi <= 72)
    ist_tfa = (abstand_hoch_pct <= -35.0) and (
        (rsi < 38) or (rsi_vor5 < 35 and rsi > rsi_vor5) or (aktuell > sma50 and macd_hist > macd_hist_vortag)
    )

    earnings_warnung = None
    if stamm["earnings_datum"]:
        try:
            ed_date = datetime.strptime(stamm["earnings_datum"], "%Y-%m-%d").date()
            tage_bis_earnings = (ed_date - datetime.now().date()).days
            if 0 <= tage_bis_earnings <= 7:
                earnings_warnung = f"Quartalszahlen in {tage_bis_earnings} Tag(en) am {ed_date.strftime('%d.%m.%Y')}!"
        except Exception:
            pass

    # Ausführliche Begründungen & Klartext-Gesamturteil
    if golden_cross_neu:
        signal_text = "Starkes Kaufsignal (Frisches Golden Cross)"
        signal_farbe = "green"
        kurz_begruendung = f"Frisches Golden Cross; Kurs {abstand_sma50_pct:+.1f} % über SMA 50, Ziel {bestes_ziel:.2f} € ({max(potenzial_1_pct, potenzial_2_pct):+.1f} %)."
        gesamtfazit = (
            f"Bei {stamm['name']} hat die 50-Tage-Linie ({sma50:.2f} €) soeben die 200-Tage-Linie ({sma200:.2f} €) "
            f"von unten nach oben gekreuzt – ein starkes, langfristig bullisches Golden-Cross-Kaufsignal. "
            f"Mit einem RSI von {rsi:.1f} hat die Aktie noch ausreichend Raum nach oben bis zum Kursziel von {bestes_ziel:.2f} € "
            f"({max(potenzial_1_pct, potenzial_2_pct):+.1f} %). Einstiege in der Zone {einstieg_kurz} lassen sich bei {stop_loss_eur:.2f} € absichern."
        )
    elif ist_trend_follower and volumen_bestaetigt and macd_hist > 0:
        signal_text = "Starker Aufwärtstrend – Kaufen / Halten"
        signal_farbe = "green"
        kurz_begruendung = f"Intakter Aufwärtstrend ({abstand_sma50_pct:+.1f} % über SMA 50), hohes Volumen ({volumen_faktor:.1f}×) & Ziel {bestes_ziel:.2f} €."
        gesamtfazit = (
            f"{stamm['name']} befindet sich in einem dynamischen Aufwärtstrend: Der Kurs liegt {abstand_sma50_pct:+.1f} % über der "
            f"50-Tage-Linie ({sma50:.2f} €) und {abstand_sma200_pct:+.1f} % über der 200-Tage-Linie ({sma200:.2f} €). "
            f"Überdurchschnittliches Volumen ({volumen_faktor:.1f}-facher Schnitt) und ein steigender MACD untermauern das "
            f"Aufwärtspotenzial bis {bestes_ziel:.2f} € ({max(potenzial_1_pct, potenzial_2_pct):+.1f} %). Bestehende Stücke halten; "
            f"Neukäufe bevorzugt bei leichten Rücksetzern um {kaufzone_von:.2f}–{kaufzone_bis:.2f} €."
        )
    elif ist_trend_follower:
        signal_text = "Intakter Aufwärtstrend – Rücksetzer nutzen"
        signal_farbe = "green"
        kurz_begruendung = f"Über SMA 50 ({abstand_sma50_pct:+.1f} %) & SMA 200 ({abstand_sma200_pct:+.1f} %), RSI {rsi:.0f}; Ziel {bestes_ziel:.2f} €."
        gesamtfazit = (
            f"Der übergeordnete Trend bei {stamm['name']} zeigt klar nach oben, da sowohl SMA 50 ({sma50:.2f} €) als auch "
            f"SMA 200 ({sma200:.2f} €) unterhalb des Kurses verlaufen. Der RSI liegt bei gesunden {rsi:.1f} Punkten. "
            f"Rücksetzer in die Kaufzone {kaufzone_von:.2f}–{kaufzone_bis:.2f} € bieten ein attraktives Einstiegsfenster "
            f"mit Kursziel {bestes_ziel:.2f} € ({max(potenzial_1_pct, potenzial_2_pct):+.1f} %)."
        )
    elif ist_tfa:
        signal_text = "Turnaround-Chance (TFA-Signal) – Antizyklisch beobachten"
        signal_farbe = "amber"
        kurz_begruendung = f"{abs(abstand_hoch_pct):.0f} % unter 52W-Hoch; RSI ({rsi:.0f}) dreht nach oben; Erholungsziel {bestes_ziel:.2f} €."
        gesamtfazit = (
            f"{stamm['name']} notiert mit {abstand_hoch_pct:.1f} % deutlich unter seinem 52-Wochen-Hoch ({hoch_52w:.2f} €) "
            f"und erfüllt damit die Turnaround-Formel (TFA). Erste Stabilisierungstendenzen sind erkennbar "
            f"(RSI {rsi:.1f}). Für risikobewusste Anleger bietet sich ein gestaffelter Aufbau mit engem Stop-Loss "
            f"bei {stop_loss_eur:.2f} € an (Potenzial bis {bestes_ziel:.2f} €, CRV {crv:.1f} : 1)."
        )
    elif rsi > 74:
        signal_text = "Stark überhitzt – Rücksetzer abwarten"
        signal_farbe = "red"
        kurz_begruendung = f"RSI bei {rsi:.0f} (überkauft), bereits {abstand_sma50_pct:+.1f} % über SMA 50 – erst bei {kaufzone_von:.2f}–{kaufzone_bis:.2f} € kaufen."
        gesamtfazit = (
            f"Nach einer steilen Rallye ist {stamm['name']} mit einem RSI von {rsi:.1f} technisch stark überkauft und hat sich "
            f"{abstand_sma50_pct:+.1f} % weit von der 50-Tage-Linie ({sma50:.2f} €) entfernt. Jetzt nicht hinterherlaufen: "
            f"Neueinstiege erst nach einer Abkühlung in die Zone {kaufzone_von:.2f}–{kaufzone_bis:.2f} € planen; "
            f"bestehende Gewinne bei {stop_loss_eur:.2f} € absichern."
        )
    elif aktuell < sma50 and aktuell < sma200:
        signal_text = "Abwärtstrend aktiv – Ausbruch über SMA 50 abwarten"
        signal_farbe = "red"
        kurz_begruendung = f"Unter SMA 50 ({abstand_sma50_pct:.1f} %) & SMA 200 ({abstand_sma200_pct:.1f} %) – Einstieg erst über {ausbruch_trigger:.2f} €."
        gesamtfazit = (
            f"{stamm['name']} notiert aktuell unter beiden wichtigen Trendlinien (SMA 50 bei {sma50:.2f} € und "
            f"SMA 200 bei {sma200:.2f} €). Solange der Kurs unter {sma50:.2f} € bleibt, dominiert der Abwärtsdruck. "
            f"Vorerst abwarten und erst bei einer bestätigten Bodenbildung über {ausbruch_trigger:.2f} € einsteigen."
        )
    else:
        richtung_text = (
            "knapp unter der 50-Tage-Linie und testet diese als Widerstand"
            if aktuell < sma50
            else "über der 50-Tage-Linie, arbeitet aber noch an der 200-Tage-Linie"
        )
        gc_zusatz = " Ein Golden Cross steht kurz bevor!" if golden_cross_nah else ""
        signal_text = "Neutral / Konsolidierung – Ausbruch abwarten"
        signal_farbe = "amber"
        kurz_begruendung = f"Seitwärtsphase (RSI {rsi:.0f}); Kaufzone {kaufzone_von:.2f}–{kaufzone_bis:.2f} € oder Ausbruch > {ausbruch_trigger:.2f} €."
        gesamtfazit = (
            f"{stamm['name']} befindet sich in einer Konsolidierungsphase: Der Kurs liegt {richtung_text}.{gc_zusatz} "
            f"Mit einem RSI von {rsi:.1f} ist die Aktie neutral bewertet. Ein klares Kaufsignal entsteht beim Ausbruch "
            f"über {ausbruch_trigger:.2f} € mit Ziel {bestes_ziel:.2f} € ({max(potenzial_1_pct, potenzial_2_pct):+.1f} %)."
        )

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

    analysten_info_zeile = ""
    if analysten_mean_eur:
        spanne_txt = (
            f" (Spanne: {analysten_low_eur:.2f} € bis {analysten_high_eur:.2f} €)"
            if analysten_low_eur and analysten_high_eur
            else ""
        )
        urteil_txt = f" • Konsens-Rating: <b>{stamm['analysten_urteil']}</b>" if stamm["analysten_urteil"] else ""
        analysten_info_zeile = (
            f"<br><b>Analysten-Schätzung ({stamm['analysten_anzahl']} Studien):</b> Ø Kursziel <b>{analysten_mean_eur:.2f} €</b>"
            f"{spanne_txt}{urteil_txt}"
        )

    kursziele_detail = (
        f"<b>Kursziel 1 ({kursziel_1_label}):</b> <b>{kursziel_1_eur:.2f} €</b> ({potenzial_1_pct:+.1f} % Potenzial)<br>"
        f"<b>Kursziel 2 ({kursziel_2_label}):</b> <b>{kursziel_2_eur:.2f} €</b> ({potenzial_2_pct:+.1f} % Potenzial)"
        f"{analysten_info_zeile}"
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
        "kursziel_1_eur": kursziel_1_eur,
        "potenzial_1_pct": potenzial_1_pct,
        "kursziel_2_eur": kursziel_2_eur,
        "potenzial_2_pct": potenzial_2_pct,
        "bestes_ziel_eur": bestes_ziel,
        "bestes_potenzial_pct": round(((bestes_ziel - aktuell) / aktuell) * 100, 1),
        "kaufzone_von": kaufzone_von,
        "kaufzone_bis": kaufzone_bis,
        "ausbruch_trigger": ausbruch_trigger,
        "einstieg_kurz": einstieg_kurz,
        "einstieg_strategie_text": einstieg_strategie_text,
        "kursziele_detail": kursziele_detail,
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

    df_bt["Golden_Cross"] = (sma50_vor <= sma200_vor) & (df_bt["SMA_50"] > df_bt["SMA_200"])
    df_bt["Death_Cross"] = (sma50_vor >= sma200_vor) & (df_bt["SMA_50"] < df_bt["SMA_200"])

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

        else:
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


@st.cache_data(ttl=3600, show_spinner=False)
def lade_sektor_heatmap() -> pd.DataFrame:
    """Lädt Sektor-ETFs, rechnet in Euro (€) um und berechnet 1T, 1W, 1M und 3M Performance."""
    faktor_usd_eur = hole_wechselkurs_zu_eur("USD")
    zeilen = []
    for sektor_name, etf_ticker in SEKTOR_ETFS:
        df = bereinige_dataframe(yf.download(etf_ticker, period="4mo", interval="1d", progress=False))
        if df.empty or len(df) < 65:
            continue
        close_eur = pd.to_numeric(df["Close"], errors="coerce").dropna() * faktor_usd_eur
        aktuell = float(close_eur.iloc[-1])
        p_1t = ((aktuell - float(close_eur.iloc[-2])) / float(close_eur.iloc[-2])) * 100
        p_1w = ((aktuell - float(close_eur.iloc[-6])) / float(close_eur.iloc[-6])) * 100
        p_1m = ((aktuell - float(close_eur.iloc[-22])) / float(close_eur.iloc[-22])) * 100
        p_3m = ((aktuell - float(close_eur.iloc[-64])) / float(close_eur.iloc[-64])) * 100
        zeilen.append({
            "Sektor": sektor_name,
            "ETF": etf_ticker,
            "Kurs (€)": round(aktuell, 2),
            "1 Tag (%)": round(p_1t, 2),
            "1 Woche (%)": round(p_1w, 2),
            "1 Monat (%)": round(p_1m, 2),
            "3 Monate (%)": round(p_3m, 2),
        })
    return pd.DataFrame(zeilen)


# ==============================================================================
# SESSION STATE & NAVIGATION-CALLBACKS
# ==============================================================================
BEREICHE = [
    "1. Markt-Überblick & Leitindizes",
    "2. Rezessions-Barometer (FRED Detail)",
    "3. Trend- & Turnaround-Scanner",
    "4. Einzelaktien-Analyse (€)",
    "5. Strategie-Backtesting & Signal-Chart",
    "6. Saisonalitäts-Analyse (€)",
    "7. Sektor-Heatmap & Rotation",
    "8. Dividenden-Kalender (€)",
    "9. Krypto-ETFs & Kapitalflüsse",
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
    st.session_state["nav_bereich"] = "4. Einzelaktien-Analyse (€)"


def wechsele_zu_backtesting(ticker: str):
    st.session_state["ausgewaehlter_ticker"] = ticker
    st.session_state["nav_bereich"] = "5. Strategie-Backtesting & Signal-Chart"


# ==============================================================================
# SEITENLEISTE
# ==============================================================================
with st.sidebar:
    st.title("📈 Stock Screener Pro (€)")
    st.caption("Alle Kurse in Euro (€) • Inkl. Rezessions-Barometer, China & Japan")

    bereich = st.radio("Navigation:", BEREICHE, key="nav_bereich")

    st.divider()
    if st.button("🔄 Alle Live-Daten aktualisieren"):
        st.cache_data.clear()
        st.rerun()

    with st.expander("ℹ️ Indikator- & Makro-Lexikon"):
        st.markdown(
            """
            * **Sahm-Regel:** Steigt der 3-Monats-Schnitt der US-Arbeitslosenquote um mind. 0,50 %-Punkte über das 12-Monats-Tief, begann historisch fast immer eine Rezession.
            * **Zinskurve (10J–2J / 10J–3M):** Liegen Kurzfristzinsen über Langfristzinsen (< 0 %), gilt das als klassisches Frühwarnsignal.
            * **Golden Cross (★):** Die 50-Tage-Linie kreuzt die 200-Tage-Linie nach oben – ein starkes mittelfristiges Kaufsignal.
            * **TFA (Turnaround-Formel):** Stark gefallene Aktien (mind. 35 % unter 52W-Hoch), die einen Boden bilden.
            """
        )

# ==============================================================================
# BEREICH 1: MARKT-ÜBERBLICK, STIMMUNG & LEITINDIZES (INKL. NASDAQ)
# ==============================================================================
if bereich == "1. Markt-Überblick & Leitindizes":
    st.header("🧭 Markt-Überblick & Börsenstimmung")
    st.write("Aktuelle Großwetterlage an den Weltbörsen – inklusive Angst-Barometer (VIX) und aller 5 wichtigen Leitindizes.")

    sentiment = berechne_markt_sentiment()

    col_gauge, col_info = st.columns([1.15, 1.05])
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
        fig_gauge.update_layout(paper_bgcolor="#0E1117", font={"color": "#E6EDF3"}, height=300, margin=dict(l=20, r=20, t=50, b=15))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        st.subheader("Aktuelle Stimmungslage")
        st.markdown(
            f"<div class='metric-card'><h4>{sentiment['label']}</h4>"
            f"<p>Angst-Barometer (VIX): <b>{sentiment['vix']}</b> (Unter 20 bedeutet ruhiges Fahrwasser)</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='info-box'>
            <b>Was bedeutet das für deine Einstiege?</b><br>
            • Bei <b>Angst (unter 40)</b> lohnen sich antizyklische Turnaround-Kandidaten (TFA-Scanner).<br>
            • Bei <b>Optimismus (55–75)</b> laufen Trend-Follower & Golden-Cross-Signale am zuverlässigsten.<br>
            • Den langfristigen Konjunktur-Check findest du im separaten Menüpunkt <b>„2. Rezessions-Barometer (FRED Detail)“</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("🌍 Welt-Leitindizes im Schnell-Check (inkl. NASDAQ)")
    indizes = [
        ("DAX 🇩🇪", "^GDAXI"),
        ("S&P 500 🇺🇸", "^GSPC"),
        ("NASDAQ 100 🇺🇸", "^NDX"),
        ("Hang Seng (HK/CN) 🇨🇳", "^HSI"),
        ("Nikkei 225 (Japan) 🇯🇵", "^N225"),
    ]
    idx_cols = st.columns(5)
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
# BEREICH 2: REZESSIONS-BAROMETER (EIGENE SEITE MIT ALLEN 6 FRED-INDIKATOREN IM DETAIL)
# ==============================================================================
elif bereich == "2. Rezessions-Barometer (FRED Detail)":
    st.header("🚦 US-Notenbank (FRED) Rezessions-Barometer")
    st.write(
        "Echtzeit-Konjunkturanalyse auf Basis der 6 wichtigsten Frühindikatoren der Federal Reserve (FRED) – "
        "ohne API-Key, mit Ampelsystem, Detail-Beschreibungen und konkreter Portfolio-Steuerung."
    )

    with st.spinner("Lade Echtzeit-Wirtschaftsdaten der US-Notenbank (FRED)..."):
        rezession = berechne_rezessions_barometer()

    col_rez1, col_rez2 = st.columns([1.15, 1.05])
    with col_rez1:
        fig_rez = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rezession["score"],
                title={"text": "Rezessions-Risiko-Score (0 = Sicher, 100 = Rezession)", "font": {"size": 17, "color": "#E6EDF3"}},
                number={"font": {"size": 46, "color": "#E6EDF3"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#E6EDF3"},
                    "bar": {"color": "#FFD54F"},
                    "bgcolor": "#161B22",
                    "steps": [
                        {"range": [0, 35], "color": "#2E7D32"},
                        {"range": [35, 62], "color": "#E65100"},
                        {"range": [62, 100], "color": "#B71C1C"},
                    ],
                },
            )
        )
        fig_rez.update_layout(paper_bgcolor="#0E1117", font={"color": "#E6EDF3"}, height=310, margin=dict(l=20, r=20, t=55, b=15))
        st.plotly_chart(fig_rez, use_container_width=True)

    with col_rez2:
        st.subheader("Gesamt-Ampel & Konjunktur-Fazit")
        st.markdown(
            f"<div class='metric-card'><h4>{rezession['ampel_gesamt']}</h4>"
            f"<p>Risiko-Score: <b>{rezession['score']} von 100 Punkten</b> (gewichtet über alle 6 FRED-Frühindikatoren)</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='info-box'><b>📋 Konkrete Portfolio-Strategie-Empfehlung:</b><br>{rezession['strategie']}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🚦 Die 6 FRED Rezessions-Indikatoren im Detail")
    st.caption("Jeder Indikator wird einzeln mit seinem aktuellen Wert, der kritischen Warnschwelle und seiner wirtschaftlichen Bedeutung erklärt.")

    karten = rezession["detail_karten"]
    for reihe_start in range(0, len(karten), 2):
        c_left, c_right = st.columns(2)
        for col_widget, idx in zip([c_left, c_right], [reihe_start, reihe_start + 1]):
            if idx < len(karten):
                k = karten[idx]
                with col_widget:
                    st.markdown(
                        f"""
                        <div class='fred-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                                <b style='font-size:1.05rem;'>{k['titel']}</b>
                                <span class='badge-{k['ampel_farbe']}'>{k['ampel_text']}</span>
                            </div>
                            <div style='font-size:0.9rem; color:#8B949E; margin-bottom:10px;'>
                                <b>Aktuell:</b> <span style='color:#E6EDF3; font-size:1.02rem;'>{k['wert']}</span> &nbsp;|&nbsp;
                                <b>Schwelle:</b> {k['schwelle']} &nbsp;|&nbsp; <b>Stand:</b> {k['stand']}
                            </div>
                            <p style='font-size:0.9rem; color:#C9D1D9; margin-bottom:8px;'>
                                <b>Was wird gemessen?</b> {k['was_ist_das']}
                            </p>
                            <p style='font-size:0.92rem; color:#E6EDF3; margin-bottom:0;'>
                                <b>Einordnung:</b> {k['bewertung']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.subheader("📊 Kompakte Übersichtstabelle der 6 Makro-Indikatoren")
    st.dataframe(rezession["tabelle"], use_container_width=True, hide_index=True)

    st.subheader("📈 Historische Verläufe der wichtigsten Frühwarnsysteme")
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        if not rezession["sahm_serie"].empty:
            fig_sahm = go.Figure()
            fig_sahm.add_trace(go.Scatter(x=rezession["sahm_serie"].index, y=rezession["sahm_serie"].values, name="Sahm-Regel", line=dict(color="#58A6FF", width=2.2)))
            fig_sahm.add_hline(y=0.50, line_dash="dash", line_color="#FF5252", annotation_text="Schwelle 0,50")
            fig_sahm.update_layout(title="Sahm-Regel (5 Jahre)", paper_bgcolor="#0E1117", plot_bgcolor="#161B22", font=dict(color="#E6EDF3"), height=290, margin=dict(l=15, r=15, t=45, b=15))
            st.plotly_chart(fig_sahm, use_container_width=True)
    with c_f2:
        if not rezession["zinskurve_serie"].empty:
            fig_zk = go.Figure()
            fig_zk.add_trace(go.Scatter(x=rezession["zinskurve_serie"].index, y=rezession["zinskurve_serie"].values, name="10J–2J", line=dict(color="#00E676", width=2.2)))
            fig_zk.add_hline(y=0.0, line_dash="dash", line_color="#FFB300", annotation_text="0,00 %")
            fig_zk.update_layout(title="Zinskurve 10J–2J (3 Jahre)", paper_bgcolor="#0E1117", plot_bgcolor="#161B22", font=dict(color="#E6EDF3"), height=290, margin=dict(l=15, r=15, t=45, b=15))
            st.plotly_chart(fig_zk, use_container_width=True)
    with c_f3:
        if not rezession["hy_serie"].empty:
            fig_hy = go.Figure()
            fig_hy.add_trace(go.Scatter(x=rezession["hy_serie"].index, y=rezession["hy_serie"].values, name="HY Spread", line=dict(color="#CE93D8", width=2.2)))
            fig_hy.add_hline(y=5.0, line_dash="dash", line_color="#FF5252", annotation_text="Stress 5,0 %")
            fig_hy.update_layout(title="High-Yield Credit Spread (3 Jahre)", paper_bgcolor="#0E1117", plot_bgcolor="#161B22", font=dict(color="#E6EDF3"), height=290, margin=dict(l=15, r=15, t=45, b=15))
            st.plotly_chart(fig_hy, use_container_width=True)

# ==============================================================================
# BEREICH 3: TREND-FOLLOWER & TFA-SCANNER (INKL. CHINA, JAPAN, KURSIZIELE & EINSTIEGE)
# ==============================================================================
elif bereich == "3. Trend- & Turnaround-Scanner":
    st.header("🔍 Aktien-Scanner: Trend-Follower & Turnaround-Formel (TFA)")
    st.caption("Alle Kurse, Kursziele und Einstiegszonen aus Hongkong (HKD), Japan (JPY), USA (USD) und Europa werden live in Euro (€) umgerechnet.")

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
    with st.spinner("Lade Kurse, berechne Euro-Kursziele, Einstiegszonen & Signale..."):
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
                "Einstiegschance (€)": r["einstieg_kurz"],
                "Kursziel (€)": round(r["bestes_ziel_eur"], 2),
                "Potenzial (%)": f"{r['bestes_potenzial_pct']:+.1f} %",
                "Stop-Loss (€)": round(r["stop_loss_eur"], 2),
                "CRV": f"{r['crv']:.1f} : 1",
                "Gesamturteil": r["signal_text"],
                "Begründung": r["kurz_begruendung"],
            })

        df_anzeige = pd.DataFrame(tabellen_daten)
        st.dataframe(df_anzeige, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🔎 Treffer im Detail prüfen (Kursziele, Einstiegschancen & Begründung)")
        auswahl_map = {f"{r['name']} ({r['ticker']}) – {r['kurs_eur']:.2f} €": r for r in gefiltert}
        gewaehltes_label = st.selectbox("Aktie auswählen:", list(auswahl_map.keys()))
        sel = auswahl_map[gewaehltes_label]

        st.markdown(
            f"""
            <div class='verdict-box'>
                <b>{sel['name']} ({sel['ticker']}) • {sel['region']}</b> —
                <span class='badge-{sel['signal_farbe']}'>{sel['signal_text']}</span><br><br>
                <b>Gesamteinschätzung:</b> {sel['gesamtfazit']}<br><br>
                <b>🎯 Kursziele:</b> {sel['kursziele_detail']}<br><br>
                <b>🛒 Konkrete Einstiegschancen:</b><br>{sel['einstieg_strategie_text']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_c1, btn_c2, _ = st.columns([1.3, 1.3, 2])
        with btn_c1:
            st.button(
                "➡️ Zur vollständigen Einzelaktien-Analyse",
                on_click=wechsele_zu_einzelanalyse,
                args=(sel["ticker"], sel["name"]),
                use_container_width=True,
            )
        with btn_c2:
            st.button(
                "📊 Im Backtesting-Signal-Chart öffnen",
                on_click=wechsele_zu_backtesting,
                args=(sel["ticker"],),
                use_container_width=True,
            )

# ==============================================================================
# BEREICH 4: EINZELAKTIEN-ANALYSE (NAMENSSUCHE + KURSIZIELE + EINSTIEGSCHANCEN)
# ==============================================================================
elif bereich == "4. Einzelaktien-Analyse (€)":
    st.header("🔬 Einzelaktien-Analyse (Kurse, Kursziele & Einstiegschancen in €)")

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

        # Reihe 1: Aktueller Kurs, RSI, Stop-Loss, CRV, 52W-Hoch
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

        # Reihe 2: Kursziele & Einstiegschancen auf einen Blick
        z1, z2, z3, z4 = st.columns(4)
        z1.metric("🎯 Kursziel 1 (Technik)", f"{daten['kursziel_1_eur']:.2f} €", f"{daten['potenzial_1_pct']:+.1f} %")
        z2.metric("🎯 Kursziel 2 (Konsens/Ziel)", f"{daten['kursziel_2_eur']:.2f} €", f"{daten['potenzial_2_pct']:+.1f} %")
        z3.metric("🛒 Pullback-Kaufzone", f"{daten['kaufzone_von']:.2f} – {daten['kaufzone_bis']:.2f} €", "Idealer Rücksetzer")
        z4.metric("🚀 Ausbruchs-Marke", f"> {daten['ausbruch_trigger']:.2f} €", "Prozyklischer Trigger")

        # Ausführliches Gesamturteil & Begründung
        st.markdown(
            f"""
            <div class='metric-card'>
                <h4>Gesamturteil: <span class='badge-{daten["signal_farbe"]}'>{daten["signal_text"]}</span></h4>
                <p style='font-size: 1.0rem; line-height: 1.6; margin-top: 10px;'>
                    <b>Ausführliche Gesamteinschätzung:</b><br>
                    {daten['gesamtfazit']}
                </p>
                <hr style='border-color: #30363D; margin: 14px 0;'>
                <ul style='line-height: 1.65;'>
                    <li><b>Trend- & Linien-Check (SMA 50 / SMA 200):</b> {daten['trend_detail']}</li>
                    <li><b>Schwung, RSI & Volumen:</b> {daten['momentum_detail']}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Zwei separate Boxen: Konkrete Einstiegschancen & Kursziele
        c_plan1, c_plan2 = st.columns(2)
        with c_plan1:
            st.markdown(
                f"""
                <div class='entry-box'>
                    <h4 style='margin-top:0; color:#00E676;'>🛒 Konkrete Einstiegschancen & Kaufzonen</h4>
                    {daten['einstieg_strategie_text']}<br><br>
                    <b>Absicherung (Stop-Loss):</b> <b>{daten['stop_loss_eur']:.2f} €</b> (-{daten['stop_puffer_pct']:.1f} % unter aktuellem Kurs, berechnet über 2× ATR).
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_plan2:
            st.markdown(
                f"""
                <div class='target-box'>
                    <h4 style='margin-top:0; color:#FFD54F;'>🎯 Kursziele & Gewinnpotenzial</h4>
                    {daten['kursziele_detail']}<br><br>
                    <b>Chance-Risiko-Verhältnis (CRV):</b> <b>{daten['crv']:.1f} : 1</b> bezogen auf das Hauptziel von <b>{daten['bestes_ziel_eur']:.2f} €</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.button(
            "📊 Kauf- & Verkaufssignale für diese Aktie im Backtesting-Chart ansehen",
            on_click=wechsele_zu_backtesting,
            args=(daten["ticker"],),
        )

        # Kurschart in Euro mit Kursziel, Kaufzone, SMA 50, SMA 200 & Stop-Loss
        df_plot = daten["df"].tail(200)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
            subplot_titles=(
                f"{daten['name']} – Kursverlauf, Kursziel, Kaufzone & Stop-Loss in Euro (€)",
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
            y=daten["bestes_ziel_eur"],
            line_dash="dash",
            line_color="#00E676",
            annotation_text=f"Kursziel: {daten['bestes_ziel_eur']:.2f} €",
            row=1,
            col=1,
        )
        fig.add_hrect(
            y0=daten["kaufzone_von"],
            y1=daten["kaufzone_bis"],
            fillcolor="rgba(88, 166, 255, 0.14)",
            line_width=0,
            annotation_text="Kaufzone",
            annotation_position="bottom left",
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
            height=620,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=55, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(gridcolor="#21262D")
        fig.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# BEREICH 5: BACKTESTING & SIGNAL-CHART (OHNE KAPITALENTWICKLUNG, MIT GOLDEN CROSS)
# ==============================================================================
elif bereich == "5. Strategie-Backtesting & Signal-Chart":
    st.header("📊 Signal-Backtesting: Kauf-, Verkaufs- & Golden-Cross-Signale")
    st.write("Übersichtliche Darstellung aller historischen Einstiegs-, Ausstiegs- und Golden-Cross-Punkte direkt im Euro-Chart.")

    c_s1, c_s2, c_s3 = st.columns([1.4, 1.6, 1.0])
    with c_s1:
        bt_eingabe = st.text_input(
            "Aktie (Name oder Kürzel, z. B. Xiaomi, BYD, NVDA):",
            value=st.session_state["ausgewaehlter_ticker"],
        )
        bt_ticker, _ = suche_ticker_aus_name(bt_eingabe)
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

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Aktueller Kurs (€)", f"{float(df_bt['Close'].iloc[-1]):.2f} €")
        s2.metric("🟢 Kaufsignale", f"{len(kauf_df)} Signale")
        s3.metric("🔴 Verkaufssignale", f"{len(verkauf_df)} Signale")
        s4.metric("⭐ Golden Crosses", f"{len(gc_df)} Kreuzungen")

        fig_bt = go.Figure()

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
# BEREICH 6: SAISONALITÄTS-ANALYSE (€)
# ==============================================================================
elif bereich == "6. Saisonalitäts-Analyse (€)":
    st.header("📅 Saisonalitäts-Analyse: Beste & schwächste Börsenmonate")
    st.write("Analysiert, in welchen Kalendermonaten eine Aktie historisch am stärksten steigt und wie hoch die Trefferquote ist.")

    col_sais1, col_sais2 = st.columns([1.8, 1.0])
    with col_sais1:
        sais_eingabe = st.text_input(
            "Aktienname oder Kürzel eingeben (z. B. Xiaomi, BYD, Toyota, Nvidia, SAP):",
            value=st.session_state["ausgewaehlter_ticker"],
        )
        sais_ticker, _ = suche_ticker_aus_name(sais_eingabe)
    with col_sais2:
        sais_jahre = st.selectbox("Rückblick-Zeitraum:", ["5 Jahre", "10 Jahre", "3 Jahre"], index=0)

    df_sais = lade_kursdaten_eur(sais_ticker, zeitraum="10y" if sais_jahre == "10 Jahre" else ("5y" if sais_jahre == "5 Jahre" else "3y"))
    stamm_sais = hole_stammdaten(sais_ticker)

    if df_sais.empty or len(df_sais) < 120:
        st.warning(f"Für {stamm_sais['name']} ({sais_ticker}) liegen noch nicht genügend historische Monatsdaten vor.")
    else:
        monats_close = df_sais["Close"].resample("ME").last().dropna()
        monats_ret = monats_close.pct_change().dropna() * 100

        sais_rows = []
        for m in range(1, 13):
            werte = monats_ret[monats_ret.index.month == m]
            if len(werte) > 0:
                avg_r = float(werte.mean())
                win_rate = float((werte > 0).mean() * 100)
                sais_rows.append({
                    "Monat_Nr": m,
                    "Monat": MONATSNAMEN_DE[m - 1],
                    "Ø Rendite (%)": round(avg_r, 2),
                    "Trefferquote (%)": round(win_rate, 0),
                    "Anzahl Jahre": len(werte),
                })

        df_monate = pd.DataFrame(sais_rows)
        bester = df_monate.loc[df_monate["Ø Rendite (%)"].idxmax()]
        schlechtester = df_monate.loc[df_monate["Ø Rendite (%)"].idxmin()]

        m1, m2, m3 = st.columns(3)
        m1.metric("Unternehmen", f"{stamm_sais['name']}")
        m2.metric("Stärkster Monat", f"{bester['Monat']} ({bester['Ø Rendite (%)']:+.2f} %)", f"{int(bester['Trefferquote (%)'])} % Trefferquote")
        m3.metric("Schwächster Monat", f"{schlechtester['Monat']} ({schlechtester['Ø Rendite (%)']:+.2f} %)", f"{int(schlechtester['Trefferquote (%)'])} % Trefferquote")

        farben = ["#00E676" if v >= 0 else "#FF5252" for v in df_monate["Ø Rendite (%)"]]
        fig_sais = go.Figure(
            go.Bar(
                x=df_monate["Monat"],
                y=df_monate["Ø Rendite (%)"],
                marker_color=farben,
                text=[f"{v:+.1f}%" for v in df_monate["Ø Rendite (%)"]],
                textposition="outside",
            )
        )
        fig_sais.update_layout(
            title=f"{stamm_sais['name']} ({sais_ticker}) – Durchschnittliche Monatsrendite ({sais_jahre})",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"),
            height=430,
            yaxis=dict(title="Ø Monatsperformance (%)", gridcolor="#21262D"),
        )
        st.plotly_chart(fig_sais, use_container_width=True)
        st.dataframe(df_monate[["Monat", "Ø Rendite (%)", "Trefferquote (%)", "Anzahl Jahre"]], use_container_width=True, hide_index=True)

# ==============================================================================
# BEREICH 7: SEKTOR-HEATMAP & BRANCHEN-ROTATION
# ==============================================================================
elif bereich == "7. Sektor-Heatmap & Rotation":
    st.header("🗺️ Sektor-Heatmap & Branchen-Rotation (€)")
    st.write("Zeigt auf einen Blick, in welche Wirtschaftssektoren aktuell Kapital fließt und welche Branchen zurückbleiben.")

    with st.spinner("Lade Sektor-Performance in Euro (€)..."):
        df_sektor = lade_sektor_heatmap()

    if df_sektor.empty:
        st.warning("Sektor-Daten konnten gerade nicht geladen werden.")
    else:
        df_sektor_sort = df_sektor.sort_values(by="1 Monat (%)", ascending=False)

        z_werte = df_sektor_sort[["1 Tag (%)", "1 Woche (%)", "1 Monat (%)", "3 Monate (%)"]].values
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=z_werte,
                x=["1 Tag (%)", "1 Woche (%)", "1 Monat (%)", "3 Monate (%)"],
                y=df_sektor_sort["Sektor"],
                colorscale=[[0.0, "#B71C1C"], [0.5, "#161B22"], [1.0, "#00C853"]],
                zmid=0,
                text=np.round(z_werte, 2),
                texttemplate="%{text:+.2f} %",
                textfont={"size": 13, "color": "#FFFFFF"},
            )
        )
        fig_hm.update_layout(
            title="Sektor-Heatmap: Performance-Vergleich über 4 Zeiträume",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"),
            height=520,
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.dataframe(df_sektor_sort, use_container_width=True, hide_index=True)

# ==============================================================================
# BEREICH 8: DIVIDENDEN-KALENDER & AUSSCHÜTTUNGS-CHECK (€)
# ==============================================================================
elif bereich == "8. Dividenden-Kalender (€)":
    st.header("💰 Dividenden-Kalender & Ausschüttungs-Check (€)")
    st.write("Übersicht über Dividendenrenditen, jährliche Ausschüttung in Euro (€), Ex-Dividenden-Tage und Ausschüttungsquoten.")

    div_zusatz = st.text_input(
        "Weitere Aktien prüfen (Name oder Kürzel mit Komma, z. B. Allianz, Toyota, BYD, Siemens):",
        value="",
    )

    div_liste = list(DIVIDENDEN_STANDARD)
    if div_zusatz.strip():
        for teil in div_zusatz.split(","):
            if teil.strip():
                tkr, _ = suche_ticker_aus_name(teil.strip())
                if tkr not in div_liste:
                    div_liste.insert(0, tkr)

    div_zeilen = []
    with st.spinner("Lade Dividenden-Daten in Euro (€)..."):
        for t in div_liste:
            stamm = hole_stammdaten(t)
            faktor = hole_wechselkurs_zu_eur(stamm["waehrung"])
            div_eur = round(stamm["div_rate_orig"] * faktor, 2)
            payout = stamm["payout_ratio_pct"]

            if stamm["div_rendite_pct"] == 0 and div_eur == 0:
                bewertung = "Wachstumsaktie (keine Dividende)"
            elif payout is not None and payout > 85:
                bewertung = "⚠️ Hohe Ausschüttungsquote (> 85 %)"
            elif stamm["div_rendite_pct"] >= 3.0:
                bewertung = "🟢 Attraktiver Dividendenzahler"
            else:
                bewertung = "Solide Basis-Dividende"

            div_zeilen.append({
                "Unternehmen": f"{stamm['name']} ({t})",
                "Region": erkenne_region(t),
                "Dividendenrendite (%)": stamm["div_rendite_pct"],
                "Jahresdividende (€)": div_eur,
                "Ex-Dividenden-Tag": stamm["ex_div_datum"] or "Noch nicht terminiert",
                "Ausschüttungsquote": f"{payout:.1f} %" if payout is not None else "k. A.",
                "Einschätzung": bewertung,
            })

    df_div = pd.DataFrame(div_zeilen).sort_values(by="Dividendenrendite (%)", ascending=False)
    st.dataframe(df_div, use_container_width=True, hide_index=True)

# ==============================================================================
# BEREICH 9: KRYPTO-ETFs & KAPITALFLÜSSE (7-TAGE & 30-TAGE SUMMEN)
# ==============================================================================
elif bereich == "9. Krypto-ETFs & Kapitalflüsse":
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
