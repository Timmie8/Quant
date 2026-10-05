from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ==========================================
# STREAMLIT PAGINA CONFIGURATIE
# ==========================================
st.set_page_config(
    page_title="USA Stocks - Stoxline & Quant Ensemble AI Scanner",
    page_icon="🇺🇸",
    layout="wide",
)

# ==========================================
# USA TICKERS LIJSTEN
# ==========================================
USA_PRESETS = {
    "Magnificent 7 & Big Tech": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
    ],
    "Semiconductors / AI Hardware": [
        "NVDA",
        "AMD",
        "AVGO",
        "INTC",
        "QCOM",
        "MU",
        "ARM",
        "SMCI",
        "TSM",
        "AMAT",
        "LRCX",
    ],
    "NASDAQ 100 Leaders": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "AVGO",
        "COST",
        "PEP",
        "CSCO",
        "TMUS",
        "NFLX",
        "AMD",
        "INTC",
        "QCOM",
        "AMAT",
    ],
    "Dow Jones Industrial Top 10": [
        "UNH",
        "GS",
        "HD",
        "MSFT",
        "CAT",
        "CRM",
        "AMGN",
        "V",
        "MCD",
        "BA",
    ],
    "Growth & Momentum USA": [
        "PLTR",
        "COIN",
        "AMD",
        "SHOP",
        "SQ",
        "SNOW",
        "U",
        "NET",
        "PANW",
        "CRWD",
    ],
}

# ==========================================
# REKENFUNCTIE (PURE PANDAS - GEEN PANDAS-TA)
# ==========================================
def calculate_stoxline_and_ensemble(df: pd.DataFrame) -> dict:
    if len(df) < 50:
        return None

    # Zorg dat de kolommen eenduidig als Series worden ingelezen
    close = (
        df["Close"].squeeze()
        if isinstance(df["Close"], pd.DataFrame)
        else df["Close"]
    )
    low = (
        df["Low"].squeeze() if isinstance(df["Low"], pd.DataFrame) else df["Low"]
    )
    high = (
        df["High"].squeeze()
        if isinstance(df["High"], pd.DataFrame)
        else df["High"]
    )

    # 1. Moving Averages
    sma5 = close.rolling(window=5).mean()
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()

    # 2. RSI Berekening (14 periodes)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))

    # 3. MACD Berekening (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # 4. Support & Resistance (10-daags Lookback)
    support10 = low.rolling(window=10).min()
    resistance10 = high.rolling(window=10).max()

    # Laatste geldige waarden ophalen
    c_last = float(close.iloc[-1])
    sma5_last = float(sma5.iloc[-1])
    sma5_prev = float(sma5.iloc[-2])
    sma20_last = float(sma20.iloc[-1])
    sma50_last = float(sma50.iloc[-1])
    rsi_last = float(rsi_series.iloc[-1])
    macd_last = float(macd_line.iloc[-1])
    sig_last = float(signal_line.iloc[-1])
    sup10_last = float(support10.iloc[-1])
    res10_last = float(resistance10.iloc[-1])

    # Percentage afwijking berekenen
    sup_pct = ((sup10_last - c_last) / c_last) * 100
    res_pct = ((res10_last - c_last) / c_last) * 100

    # Short-Term Score (0 tot 5 punten)
    st_points = 0.0
    if c_last > sma5_last:
        st_points += 1.25
    if sma5_last > sma5_prev:
        st_points += 1.25
    if 45 <= rsi_last <= 70:
        st_points += 1.25
    elif rsi_last > 70:
        st_points += 0.5
    if macd_last > sig_last:
        st_points += 1.25

    st_stars = int(min(max(round(st_points), 0), 5))

    # Mid-Term Score (0 tot 5 punten)
    mt_points = 0.0
    if c_last > sma20_last:
        mt_points += 1.25
    if c_last > sma50_last:
        mt_points += 1.25
    if sma20_last > sma50_last:
        mt_points += 1.25
    if macd_last > 0:
        mt_points += 1.25

    mt_stars = int(min(max(round(mt_points), 0), 5))

    # Quant Ensemble AI Score (0-100)
    st_score_100 = (st_points / 5.0) * 100
    mt_score_100 = (mt_points / 5.0) * 100
    ensemble_score = round((st_score_100 + mt_score_100) / 2, 1)

    if ensemble_score >= 60:
        sentiment = "Bullish 🟢"
    elif ensemble_score <= 40:
        sentiment = "Bearish 🔴"
    else:
        sentiment = "Neutral 🟡"

    return {
        "Koers": round(c_last, 2),
        "AI Score": ensemble_score,
        "Sentiment": sentiment,
        "Short-Term Rating": f"{'★' * st_stars}{'☆' * (5 - st_stars)} ({st_stars}/5)",
        "Mid-Term Rating": f"{'★' * mt_stars}{'☆' * (5 - mt_stars)} ({mt_stars}/5)",
        "RSI (14)": round(rsi_last, 1),
        "Support (10d)": round(sup10_last, 2),
        "Support %": round(sup_pct, 1),
        "Resistance (10d)": round(res10_last, 2),
        "Resistance %": round(res_pct, 1),
        "df": df,
        "sma5": sma5,
        "sma20": sma20,
        "sma50": sma50,
        "support10": support10,
        "resistance10": resistance10,
    }


@st.cache_data(ttl=300)
def fetch_and_scan(tickers: list):
    """Data ophalen via yfinance met caching (5 minuten)."""
    results = {}
    if not tickers:
        return results

    data = yf.download(
        tickers, period="6m", interval="1d", group_by="ticker", progress=False
    )

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                df = data[ticker].dropna()

            res = calculate_stoxline_and_ensemble(df)
            if res:
                results[ticker] = res
        except Exception:
            continue

    return results


# ==========================================
# SIDEBAR / USA TICKER SELECTIE
# ==========================================
st.sidebar.title("🇺🇸 USA Stocks Scanner")

# Preset selectie
selected_preset = st.sidebar.selectbox(
    "Kies een USA Categorie / Index:",
    ["Aangepast / Handmatig"] + list(USA_PRESETS.keys()),
)

# Knoppen voor snel laden
col_btn1, col_btn2 = st.sidebar.columns(2)
load_all_usa = col_btn1.button("🌐 Scan Alle USA")
reset_btn = col_btn2.button("🔄 Herlaad")

if reset_btn:
    st.cache_data.clear()

# Tickerlijst samenstellen
if load_all_usa:
    all_usa_tickers = sorted(
        list(set([t for sub in USA_PRESETS.values() for t in sub]))
    )
    default_text = ", ".join(all_usa_tickers)
elif selected_preset != "Aangepast / Handmatig":
    default_text = ", ".join(USA_PRESETS[selected_preset])
else:
    default_text = "NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, PLTR"

user_input = st.sidebar.text_area(
    "USA Tickers (gescheiden door komma's):", default_text, height=120
)

tickers = [
    t.strip().upper() for t in user_input.split(",") if t.strip() != ""
]

st.sidebar.markdown("---")
st.sidebar.info(
    "**Quant Ensemble Logica:**\n"
    "- **Short-Term (50%):** SMA5, RSI (14), MACD Cross\n"
    "- **Mid-Term (50%):** SMA20, SMA50, MACD Trend\n"
    "- **AI Score:** 0 tot 100 punten"
)

# ==========================================
# HOOFDSCHERM
# ==========================================
st.title("📊 USA Stocks - Stoxline & Quant Ensemble AI Scanner")
st.caption(f"Laatst bijgewerkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if tickers:
    with st.spinner(f"Live data ophalen voor {len(tickers)} USA aandelen..."):
        scan_results = fetch_and_scan(tickers)

    if scan_results:
        # Overzichtstabel opbouwen
        table_data = []
        for ticker, res in scan_results.items():
            table_data.append({
                "Ticker": ticker,
                "Koers ($)": f"${res['Koers']:.2f}",
                "AI Score": res["AI Score"],
                "Sentiment": res["Sentiment"],
                "Short-Term Rating": res["Short-Term Rating"],
                "Mid-Term Rating": res["Mid-Term Rating"],
                "RSI (14)": res["RSI (14)"],
                "Support (10d)": f
