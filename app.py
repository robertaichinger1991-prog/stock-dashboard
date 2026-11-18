import io
import json
import os
import urllib.request
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
    .badge-orange {
        background-color: rgba(255, 112, 67, 0.18);
        color: #FF8A65;
        border: 1px solid #FF7043;
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

        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        delta = df["Close"].diff()
        gewinn = delta.clip(lower=0)
        verlust = -delta.clip(upper=0)
        avg_gewinn = gewinn.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_verlust = verlust.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gewinn / avg_verlust.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))
        df["RSI_14"] = df["RSI_14"].fillna(50)

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

    golden_cross_neu = False
    if len(df) >= 6 and not df["SMA_200"].iloc[-6:].isna().any():
        golden_cross_neu = (df["SMA_50"].iloc[-6] <= df["SMA_200"].iloc[-6]) and (sma50 > sma200)

    ist_trend_follower = (aktuell > sma50) and (sma50 > sma200) and (45 <= rsi <= 72)

    ist_tfa = (abstand_hoch_pct <= -35.0) and (
        (rsi < 38) or (rsi_vor5 < 35 and rsi > rsi_vor5) or (aktuell > sma50 and macd_hist > macd_hist_vortag)
    )

    stop_loss_eur = max(0.01, aktuell - 2.0 * atr)
    risiko_eur = max(0.01, aktuell - stop_loss_eur)
    chance_eur = max(0.0, hoch_52w - aktuell)
    crv = round(chance_eur / risiko_eur, 2)

    earnings_warnung = None
    if stamm["earnings_datum"]:
        try:
            ed_date = datetime.strptime(stamm["earnings_datum"], "%Y-%m-%d").date()
            tage_bis_earnings = (ed_date - datetime.now().date()).days
            if 0 <= tage_bis_earnings <= 7:
                earnings_warnung = f"Quartalszahlen in {tage_bis_earnings} Tag(en) am {ed_date.strftime('%d.%m.%Y')}!"
        except Exception:
            pass

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


# ==============================================================================
# FRED LIVE-DATEN & REZESSIONS-BAROMETER (OHNE API-KEY)
# ==============================================================================
@st.cache_data(ttl=21600, show_spinner=False)
def lade_fred_serie(series_id: str) -> pd.Series:
    """Lädt eine Zeitreihe direkt als CSV von der US-Notenbank (FRED) ohne API-Key."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        csv_text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(csv_text))
    datum_spalte = df.columns[0]
    wert_spalte = df.columns[1]
    df[datum_spalte] = pd.to_datetime(df[datum_spalte], errors="coerce")
    df[wert_spalte] = pd.to_numeric(df[wert_spalte], errors="coerce")
    df = df.dropna(subset=[datum_spalte, wert_spalte]).sort_values(datum_spalte)
    return pd.Series(df[wert_spalte].values, index=df[datum_spalte], name=series_id)


@st.cache_data(ttl=21600, show_spinner=False)
def berechne_rezessions_barometer() -> dict | None:
    """Berechnet den Gesamt-Score (0-100) und alle 6 US-Makro- & Zinsindikatoren."""
    try:
        s_sahm = lade_fred_serie("SAHMREALTIME")
        s_curve = lade_fred_serie("T10Y2Y")
        s_hy = lade_fred_serie("BAMLH0A0HYM2")
        s_sent = lade_fred_serie("UMCSENT")
        s_cpi_raw = lade_fred_serie("CPIAUCSL")
        s_fed = lade_fred_serie("FEDFUNDS")

        # CPIAUCSL ist ein Indexstand -> Jahresinflationsrate (%) berechnen
        s_cpi_yoy = ((s_cpi_raw / s_cpi_raw.shift(12)) - 1.0) * 100.0
        s_cpi_yoy = s_cpi_yoy.dropna()

        # 5-Jahres-Ausschnitt für übersichtliche Verlaufscharts
        start_5y = pd.Timestamp.now() - pd.DateOffset(years=5)

        indikatoren = []

        # 1. Sahm-Regel (SAHMREALTIME)
        v_sahm = float(s_sahm.iloc[-1])
        v_sahm_prev = float(s_sahm.iloc[-2])
        if v_sahm >= 0.50:
            r_sahm = int(np.clip(85 + (v_sahm - 0.50) * 30, 85, 100))
            st_sahm, col_sahm = "Alarmsignal (≥ 0,50)", "red"
            fazit_sahm = "Die Sahm-Regel hat ausgelöst – historisch ein starkes Rezessionszeichen am US-Arbeitsmarkt."
        elif v_sahm >= 0.30:
            r_sahm = int(45 + ((v_sahm - 0.30) / 0.20) * 30)
            st_sahm, col_sahm = "Vorwarnstufe", "amber"
            fazit_sahm = "Die Arbeitslosigkeit zieht spürbar an – der Arbeitsmarkt nähert sich der kritischen 0,50-Schwelle."
        else:
            r_sahm = int(np.clip((v_sahm / 0.30) * 25, 5, 25))
            st_sahm, col_sahm = "Entspannt", "green"
            fazit_sahm = "Der US-Arbeitsmarkt zeigt sich stabil; kein akutes Rezessionssignal."

        indikatoren.append({
            "id": "SAHMREALTIME",
            "titel": "1. Sahm-Regel (SAHMREALTIME)",
            "untertitel": "Echtzeit-Warnsignal für den US-Arbeitsmarkt",
            "wert": v_sahm,
            "delta": v_sahm - v_sahm_prev,
            "einheit": "Pkt.",
            "stand": s_sahm.index[-1].strftime("%d.%m.%Y"),
            "risiko": r_sahm,
            "gewicht": 0.25,
            "status": st_sahm,
            "farbe": col_sahm,
            "fazit": fazit_sahm,
            "erklaerung": (
                "Die **Sahm-Regel** vergleicht den 3-Monats-Durchschnitt der US-Arbeitslosenquote mit ihrem "
                "Tiefstand der letzten 12 Monate. Steigt die Arbeitslosigkeit schnell an, brechen Konsum und "
                "Wirtschaftswachstum oft kurz darauf ein."
            ),
            "schwellenwerte": (
                "- **Unter 0,30 Pkt. (Grün):** Gesunder, stabiler Arbeitsmarkt.\n"
                "- **0,30 bis 0,49 Pkt. (Gelb):** Spürbare Abkühlung – erhöhte Aufmerksamkeit.\n"
                "- **Ab 0,50 Pkt. (Rot):** Offizielles Sahm-Rezessionssignal (traf in fast allen US-Rezessionen seit 1970 zu)."
            ),
            "verlauf": s_sahm[s_sahm.index >= start_5y],
            "linien": [
                {"y": 0.50, "color": "#FF5252", "text": "Alarmschwelle (0,50)"},
                {"y": 0.30, "color": "#FFB300", "text": "Vorwarnung (0,30)"},
            ],
        })

        # 2. US-Zinskurve (T10Y2Y)
        v_curve = float(s_curve.iloc[-1])
        v_curve_prev = float(s_curve.iloc[-22]) if len(s_curve) >= 22 else float(s_curve.iloc[-2])
        if v_curve < 0.0:
            r_curve = int(np.clip(70 + abs(v_curve) * 20, 70, 95))
            st_curve, col_curve = "Invertiert (< 0 %)", "red"
            fazit_curve = "Die Zinskurve ist invers – kurzfristige Zinsen liegen über den langfristigen (klassisches Frühwarnsignal)."
        elif v_curve <= 0.50:
            r_curve = 45
            st_curve, col_curve = "Flach / Übergang", "amber"
            fazit_curve = "Flache bzw. frisch wieder positive Zinskurve – historisch folgt eine Abkühlung oft nach der Rückkehr über 0 %."
        else:
            r_curve = 15
            st_curve, col_curve = "Normal (> +0,50 %)", "green"
            fazit_curve = "Normale Zinsstruktur: Langfristige Anleihen bringen wieder deutlich mehr Zinsen als kurzfristige."

        indikatoren.append({
            "id": "T10Y2Y",
            "titel": "2. US-Zinskurve (T10Y2Y)",
            "untertitel": "Differenz zwischen 10-jährigen und 2-jährigen US-Staatsanleihen",
            "wert": v_curve,
            "delta": v_curve - v_curve_prev,
            "einheit": "%",
            "stand": s_curve.index[-1].strftime("%d.%m.%Y"),
            "risiko": r_curve,
            "gewicht": 0.20,
            "status": st_curve,
            "farbe": col_curve,
            "fazit": fazit_curve,
            "erklaerung": (
                "Normalerweise erhalten Anleger für 10-jährige Staatsanleihen höhere Zinsen als für 2-jährige. "
                "Rutscht die Differenz **unter 0 % (inverse Zinskurve)**, erwarten die Anleihemärkte sinkende "
                "Zinsen wegen einer kommenden Wirtschaftsschwäche. Auch das schnelle Wiederansteigen über 0 % "
                "markiert oft den Beginn der eigentlichen Rezession."
            ),
            "schwellenwerte": (
                "- **Über +0,50 % (Grün):** Gesunde, positiv geneigte Zinskurve.\n"
                "- **0,00 % bis +0,50 % (Gelb):** Flache Zinskurve / sensible Übergangsphase.\n"
                "- **Unter 0,00 % (Rot):** Inverse Zinskurve – Kredite werden für Banken unattraktiver."
            ),
            "verlauf": s_curve[s_curve.index >= start_5y],
            "linien": [
                {"y": 0.0, "color": "#FF5252", "text": "Nulllinie / Inversion (0,0 %)"},
                {"y": 0.5, "color": "#00C853", "text": "Normalbereich (+0,5 %)"},
            ],
        })

        # 3. Kredit-Stress (BAMLH0A0HYM2)
        v_hy = float(s_hy.iloc[-1])
        v_hy_prev = float(s_hy.iloc[-22]) if len(s_hy) >= 22 else float(s_hy.iloc[-2])
        if v_hy > 5.0:
            r_hy = int(np.clip(75 + (v_hy - 5.0) * 10, 75, 100))
            st_hy, col_hy = "Hoher Kredit-Stress", "red"
            fazit_hy = "Deutliche Risikoaufschläge bei Unternehmensanleihen – Investoren fürchten Zahlungsausfälle."
        elif v_hy >= 3.5:
            r_hy = int(40 + ((v_hy - 3.5) / 1.5) * 30)
            st_hy, col_hy = "Leicht erhöht", "amber"
            fazit_hy = "Die Risikoaufschläge ziehen an, bleiben aber noch unterhalb einer echten Kreditklemme."
        else:
            r_hy = int(np.clip((v_hy / 3.5) * 25, 10, 25))
            st_hy, col_hy = "Entspannt", "green"
            fazit_hy = "Niedrige Risikoaufschläge: Unternehmen können sich problemlos am Kapitalmarkt refinanzieren."

        indikatoren.append({
            "id": "BAMLH0A0HYM2",
            "titel": "3. Kredit-Stress (BAMLH0A0HYM2)",
            "untertitel": "Risikoaufschlag für Unternehmensanleihen (High Yield Spread)",
            "wert": v_hy,
            "delta": v_hy - v_hy_prev,
            "einheit": "%",
            "stand": s_hy.index[-1].strftime("%d.%m.%Y"),
            "risiko": r_hy,
            "gewicht": 0.20,
            "status": st_hy,
            "farbe": col_hy,
            "fazit": fazit_hy,
            "erklaerung": (
                "Dieser Wert zeigt, wie viel **mehr Zinsen** schwächere US-Unternehmen (High Yield / Hochzinsanleihen) "
                "im Vergleich zu sicheren US-Staatsanleihen zahlen müssen. Steigt dieser Aufschlag sprunghaft an, "
                "trocknet die Kreditversorgung der Wirtschaft aus."
            ),
            "schwellenwerte": (
                "- **Unter 3,50 % (Grün):** Sehr ruhiges Kreditumfeld, hohe Risikobereitschaft.\n"
                "- **3,50 % bis 5,00 % (Gelb):** Wachsende Vorsicht bei schwächeren Bilanzen.\n"
                "- **Über 5,00 % (Rot):** Akuter Stress an den Kreditmärkten und erhöhte Ausfallgefahr."
            ),
            "verlauf": s_hy[s_hy.index >= start_5y],
            "linien": [
                {"y": 5.0, "color": "#FF5252", "text": "Stress-Schwelle (5,0 %)"},
                {"y": 3.5, "color": "#FFB300", "text": "Vorsicht (3,5 %)"},
            ],
        })

        # 4. US-Verbraucherstimmung (UMCSENT)
        v_sent = float(s_sent.iloc[-1])
        v_sent_prev = float(s_sent.iloc[-2])
        if v_sent < 60.0:
            r_sent = int(np.clip(75 + (60.0 - v_sent) * 1.5, 75, 95))
            st_sent, col_sent = "Pessimistisch (< 60)", "red"
            fazit_sent = "Die US-Verbraucher sind stark verunsichert – ein Warnsignal, da Konsum ~70 % der US-Wirtschaft ausmacht."
        elif v_sent <= 75.0:
            r_sent = int(40 + ((75.0 - v_sent) / 15.0) * 25)
            st_sent, col_sent = "Verhalten", "amber"
            fazit_sent = "Das Konsumklima liegt unter dem langjährigen Durchschnitt; Haushalte achten stärker auf Preise."
        else:
            r_sent = 15
            st_sent, col_sent = "Zuversichtlich", "green"
            fazit_sent = "Gute Konsumlaune in den USA stützt Unternehmensgewinne und Konjunktur."

        indikatoren.append({
            "id": "UMCSENT",
            "titel": "4. US-Verbraucherstimmung (UMCSENT)",
            "untertitel": "Konsumklima der Universität Michigan",
            "wert": v_sent,
            "delta": v_sent - v_sent_prev,
            "einheit": "Pkt.",
            "stand": s_sent.index[-1].strftime("%d.%m.%Y"),
            "risiko": r_sent,
            "gewicht": 0.15,
            "status": st_sent,
            "farbe": col_sent,
            "fazit": fazit_sent,
            "erklaerung": (
                "Die monatliche Umfrage der **Universität Michigan** misst, wie US-Haushalte ihre eigene Finanzlage "
                "und die Wirtschaftsaussichten einschätzen. Da rund 70 % der US-Wirtschaftsleistung am privaten Konsum "
                "hängen, schlägt schwache Stimmung schnell auf die Unternehmensgewinne durch."
            ),
            "schwellenwerte": (
                "- **Über 75 Pkt. (Grün):** Optimistische Verbraucher, solide Konsumnachfrage.\n"
                "- **60 bis 75 Pkt. (Gelb):** Gedämpfte Stimmung, Zurückhaltung bei großen Anschaffungen.\n"
                "- **Unter 60 Pkt. (Rot):** Historisches Rezessionsniveau beim Konsumklima."
            ),
            "verlauf": s_sent[s_sent.index >= start_5y],
            "linien": [
                {"y": 75.0, "color": "#00C853", "text": "Solide (75 Pkt.)"},
                {"y": 60.0, "color": "#FF5252", "text": "Rezessionszone (60 Pkt.)"},
            ],
        })

        # 5. US-Inflation (CPIAUCSL – Jahresrate)
        v_cpi = float(s_cpi_yoy.iloc[-1])
        v_cpi_prev = float(s_cpi_yoy.iloc[-2])
        if v_cpi > 4.0 or v_cpi < 0.5:
            r_cpi = 80
            st_cpi, col_cpi = "Kritisch", "red"
            fazit_cpi = (
                "Hohe Inflation frisst Kaufkraft und zwingt die Notenbank zu hohen Zinsen."
                if v_cpi > 4.0
                else "Extrem niedrige Inflation deutet auf gefährliche Nachfrageschwäche hin."
            )
        elif v_cpi > 2.8 or v_cpi < 1.5:
            r_cpi = 48
            st_cpi, col_cpi = "Über Zielwert", "amber"
            fazit_cpi = "Die Inflation liegt noch spürbar über dem 2-%-Ziel der US-Notenbank."
        else:
            r_cpi = 15
            st_cpi, col_cpi = "Im Zielkorridor", "green"
            fazit_cpi = "Preisstabilität nahe dem 2-%-Ziel gibt der US-Notenbank vollen Handlungsspielraum."

        indikatoren.append({
            "id": "CPIAUCSL",
            "titel": "5. US-Inflation (CPIAUCSL)",
            "untertitel": "Jahresrate der Verbraucherpreise (VPI)",
            "wert": v_cpi,
            "delta": v_cpi - v_cpi_prev,
            "einheit": "%",
            "stand": s_cpi_yoy.index[-1].strftime("%d.%m.%Y"),
            "risiko": r_cpi,
            "gewicht": 0.10,
            "status": st_cpi,
            "farbe": col_cpi,
            "fazit": fazit_cpi,
            "erklaerung": (
                "Die **Jahresrate der US-Verbraucherpreise (CPI)** zeigt, um wie viel Prozent das Leben für "
                "US-Konsumenten gegenüber dem Vorjahr teurer geworden ist. Zu hohe Inflation zwingt die Notenbank "
                "auf die Zinsbremse; ein Absturz unter 0,5 % signalisiert dagegen Nachfrageschwäche."
            ),
            "schwellenwerte": (
                "- **1,5 % bis 2,8 % (Grün):** Idealer Zielbereich der US-Notenbank.\n"
                "- **2,8 % bis 4,0 % (Gelb):** Erhöhter Preisdruck, Zinssenkungen werden gebremst.\n"
                "- **Über 4,0 % oder unter 0,5 % (Rot):** Kaufkraftverlust / Zinsstress bzw. Deflationsrisiko."
            ),
            "verlauf": s_cpi_yoy[s_cpi_yoy.index >= start_5y],
            "linien": [
                {"y": 4.0, "color": "#FF5252", "text": "Hoch-Inflation (4,0 %)"},
                {"y": 2.0, "color": "#00C853", "text": "Fed-Zielwert (2,0 %)"},
            ],
        })

        # 6. US-Leitzins (FEDFUNDS)
        v_fed = float(s_fed.iloc[-1])
        v_fed_prev = float(s_fed.iloc[-2])
        if v_fed > 4.25:
            r_fed = int(np.clip(70 + (v_fed - 4.25) * 12, 70, 90))
            st_fed, col_fed = "Stark restriktiv", "red"
            fazit_fed = "Hohes Zinsniveau verteuert Kredite, Hypotheken und Unternehmensfinanzierungen erheblich."
        elif v_fed >= 2.50:
            r_fed = 45
            st_fed, col_fed = "Moderat bremsend", "amber"
            fazit_fed = "Das Zinsniveau wirkt leicht bremsend auf Kreditvergabe und Immobilienmarkt."
        else:
            r_fed = 15
            st_fed, col_fed = "Locker / Neutral", "green"
            fazit_fed = "Niedrige Leitzinsen unterstützen Investitionen und Aktienbewertungen."

        indikatoren.append({
            "id": "FEDFUNDS",
            "titel": "6. US-Leitzins (FEDFUNDS)",
            "untertitel": "Aktueller Leitzins der US-Notenbank (Fed)",
            "wert": v_fed,
            "delta": v_fed - v_fed_prev,
            "einheit": "%",
            "stand": s_fed.index[-1].strftime("%d.%m.%Y"),
            "risiko": r_fed,
            "gewicht": 0.10,
            "status": st_fed,
            "farbe": col_fed,
            "fazit": fazit_fed,
            "erklaerung": (
                "Die **Federal Funds Rate** ist der wichtigste Leitzins der Welt. Hohe Leitzinsen wirken wie ein "
                "Bremsklotz für Kredite, Immobilien und Unternehmensinvestitionen – allerdings oft mit einer "
                "Zeitverzögerung von 6 bis 18 Monaten."
            ),
            "schwellenwerte": (
                "- **Unter 2,50 % (Grün):** Günstiges Finanzierungsumfeld für Wirtschaft und Börse.\n"
                "- **2,50 % bis 4,25 % (Gelb):** Spürbare Zinsbelastung für verschuldete Unternehmen.\n"
                "- **Über 4,25 % (Rot):** Stark restriktive Geldpolitik mit erhöhtem Abkühlungsrisiko."
            ),
            "verlauf": s_fed[s_fed.index >= start_5y],
            "linien": [
                {"y": 4.25, "color": "#FF5252", "text": "Restriktiv (4,25 %)"},
                {"y": 2.50, "color": "#FFB300", "text": "Neutralbereich (2,50 %)"},
            ],
        })

        gesamt_score = int(round(sum(ind["risiko"] * ind["gewicht"] for ind in indikatoren)))
        gesamt_score = int(np.clip(gesamt_score, 0, 100))

        if gesamt_score < 30:
            gesamtlage = "Expansion"
            lage_farbe = "green"
            strategie = (
                "<b>Portfolio-Strategie (Offensive Trend-Phase):</b><br>"
                "• <b>Aktien & ETFs:</b> Das Makro-Umfeld gibt grünes Licht. Trend-Follower-Aktien über SMA 50/200 und reguläre ETF-Sparpläne können voll laufen.<br>"
                "• <b>Risikomanagement:</b> Gewinne nicht zu früh abschneiden, sondern über nachgezogene ATR-Stop-Loss-Marken absichern."
            )
        elif gesamt_score < 55:
            gesamtlage = "Spätzyklus / Schleichende Abkühlung"
            lage_farbe = "amber"
            strategie = (
                "<b>Portfolio-Strategie (Selektive Qualitäts-Phase):</b><br>"
                "• <b>Aktien & ETFs:</b> Fokus auf bilanzstarke Qualitätswerte und breite Welt-ETFs legen; hochverschuldete Zykliker reduzieren.<br>"
                "• <b>Risikomanagement:</b> Neue Käufe nur noch bei bestätigtem Volumen und CRV ≥ 2,0 tätigen und eine erste <b>Cash-Reserve (10–15 %)</b> aufbauen."
            )
        elif gesamt_score < 75:
            gesamtlage = "Erhöhte Rezessionsgefahr"
            lage_farbe = "orange"
            strategie = (
                "<b>Portfolio-Strategie (Defensive Schutz-Phase):</b><br>"
                "• <b>Aktien & ETFs:</b> Überhitzte Positionen (RSI > 70) teilweise sichern und Stop-Loss-Marken konsequent beachten.<br>"
                "• <b>Cash-Strategie:</b> <b>20–30 % Cash-Quote</b> vorhalten, um bei stärkeren Marktrücksetzern gezielt antizyklische TFA-Turnaround-Chancen einzusammeln."
            )
        else:
            gesamtlage = "Akuter Rezessionsalarm"
            lage_farbe = "red"
            strategie = (
                "<b>Portfolio-Strategie (Kapitalerhalt & Antizyklische Vorbereitung):</b><br>"
                "• <b>Kapitalerhalt:</b> Stop-Loss-Regeln strikt durchziehen und keine spekulativen Ausbrüche mehr jagen.<br>"
                "• <b>Chancen nutzen:</b> Langfristige ETF-Sparpläne diszipliniert weiterlaufen lassen (günstige Einstiegskurse) und Cash-Reserven erst bei extremer Marktpanik (Sentiment < 25) gestaffelt investieren."
            )

        return {
            "score": gesamt_score,
            "gesamtlage": gesamtlage,
            "lage_farbe": lage_farbe,
            "strategie": strategie,
            "indikatoren": indikatoren,
        }
    except Exception:
        return None


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
            "🧭 Rezessions-Barometer",
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
            * **Rezessions-Barometer:** Bündelt 6 Live-Indikatoren der US-Notenbank (FRED) zu einem Risiko-Score von 0 bis 100.
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
# NEU: REZESSIONS-BAROMETER (LIVE US-NOTENBANK FRED-DATEN OHNE API-KEY)
# ==============================================================================
elif bereich == "🧭 Rezessions-Barometer":
    st.header("🧭 US-Makro & Rezessions-Barometer (FRED Live-Daten)")
    st.write("Automatische Berechnung des aktuellen Rezessionsrisikos aus den Originaldaten der US-Notenbank (St. Louis Fed / FRED – ohne zusätzlichen API-Key).")

    with st.spinner("Lade aktuelle Wirtschafts- & Zinsdaten der US-Notenbank (FRED)..."):
        baro = berechne_rezessions_barometer()

    if not baro:
        st.warning("Die FRED-Server der US-Notenbank antworten gerade verzögert. Bitte klicke links auf '🔄 Kurse jetzt aktualisieren' oder versuche es gleich noch einmal.")
    else:
        col_g, col_s = st.columns([1.15, 1.15])
        with col_g:
            fig_rez = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=baro["score"],
                    title={
                        "text": "Rezessions-Risiko-Score (0 = Entspannt, 100 = Alarm)",
                        "font": {"size": 17, "color": "#E6EDF3"},
                    },
                    number={"suffix": " Pkt.", "font": {"size": 40, "color": "#E6EDF3"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#E6EDF3"},
                        "bar": {"color": "#58A6FF"},
                        "bgcolor": "#161B22",
                        "steps": [
                            {"range": [0, 30], "color": "#2E7D32"},
                            {"range": [30, 55], "color": "#F9A825"},
                            {"range": [55, 75], "color": "#EF6C00"},
                            {"range": [75, 100], "color": "#C62828"},
                        ],
                    },
                )
            )
            fig_rez.update_layout(
                paper_bgcolor="#0E1117",
                font={"color": "#E6EDF3"},
                height=310,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_rez, use_container_width=True)

        with col_s:
            st.subheader("Aktuelle Gesamtlage & Portfolio-Strategie")
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h4>Gesamtlage: <span class='badge-{baro["lage_farbe"]}'>{baro["gesamtlage"]}</span> ({baro["score"]} / 100 Punkte)</h4>
                    <p style='margin-top: 8px; color: #8B949E; font-size: 0.9rem;'>
                        Gewichteter Gesamtindex aus Arbeitsmarkt (Sahm-Regel), Zinskurve, Kredit-Spreads, Verbraucherstimmung, Inflation und Fed-Leitzins.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='info-box'>{baro['strategie']}</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 6 Wirtschafts- & Zinsindikatoren mit Klartext-Erklärung")

        indikatoren = baro["indikatoren"]
        for i in range(0, len(indikatoren), 2):
            cols = st.columns(2)
            for idx_col, ind in enumerate(indikatoren[i : i + 2]):
                with cols[idx_col]:
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <h4 style='margin:0;'>{ind['titel']}</h4>
                                <span class='badge-{ind["farbe"]}'>{ind['status']}</span>
                            </div>
                            <p style='color:#8B949E; font-size:0.88rem; margin:4px 0 10px 0;'>{ind['untertitel']} • Stand: {ind['stand']}</p>
                            <div style='font-size:1.45rem; font-weight:700; margin-bottom:6px;'>
                                {ind['wert']:.2f} {ind['einheit']}
                                <span style='font-size:0.9rem; font-weight:400; color:#8B949E;'>({ind['delta']:+.2f} {ind['einheit']} zur Vorperiode)</span>
                            </div>
                            <p style='margin:0; font-size:0.93rem;'>{ind['fazit']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("📖 Erklärung, Schwellenwerte & Verlauf"):
                        st.markdown(ind["erklaerung"])
                        st.markdown("**Wichtige Schwellenwerte:**")
                        st.markdown(ind["schwellenwerte"])

                        s_hist = ind["verlauf"]
                        if not s_hist.empty:
                            fig_ind = go.Figure()
                            fig_ind.add_trace(
                                go.Scatter(
                                    x=s_hist.index,
                                    y=s_hist.values,
                                    mode="lines",
                                    name=ind["id"],
                                    line=dict(color="#58A6FF", width=2.2),
                                    fill="tozeroy",
                                    fillcolor="rgba(88, 166, 255, 0.10)",
                                )
                            )
                            for linie in ind["linien"]:
                                fig_ind.add_hline(
                                    y=linie["y"],
                                    line_dash="dash",
                                    line_color=linie["color"],
                                    annotation_text=linie["text"],
                                    annotation_font_color=linie["color"],
                                )
                            fig_ind.update_layout(
                                title=f"Historischer Verlauf (5 Jahre) – {ind['id']}",
                                paper_bgcolor="#0E1117",
                                plot_bgcolor="#161B22",
                                font=dict(color="#E6EDF3", size=12),
                                height=260,
                                margin=dict(l=15, r=15, t=40, b=15),
                                showlegend=False,
                            )
                            fig_ind.update_xaxes(gridcolor="#21262D")
                            fig_ind.update_yaxes(gridcolor="#21262D")
                            st.plotly_chart(fig_ind, use_container_width=True)

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
