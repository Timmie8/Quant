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
    page_title="USA Stocks - Custom AI Scanner",
    page_icon="🇺🇸",
    layout="wide",
)

# ==========================================
# PRESET USA TICKERS EN CATEGORIEËN
# ==========================================
USA_PRESETS = {
    "Zelf invoeren / Aangepast": "",
    "Big Tech / Mag 7": "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA",
    "Semiconductors & AI": "NVDA, AMD, AVGO, INTC, QCOM, MU, ARM, SMCI, TSM",
    "NASDAQ 100 Leaders": "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, NFLX, AMD",
    "Growth & Momentum": "PLTR, COIN, AMD, SHOP, SQ, SNOW, U, NET, PANW, CRWD",
}

# ==========================================
# REKENFUNCTIE (QUANT ENSEMBLE AI SCORE)
# ==========================================
def calculate_stoxline_and_ensemble(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 15:
        return None

    # Zorg voor zuivere Series
    close = df["Close"].dropna()
    low = df["Low"].dropna() if "Low" in df.columns else close
    high = df["High"].dropna() if "High" in df.columns else close

    if len(close) < 15:
        return None

    # 1. Moving Averages
    sma5 = close.rolling(window=min(5, len(close))).mean()
    sma20 = close.rolling(window=min(20, len(close))).mean()
    sma50 = close.rolling(window=min(50, len(close))).mean()

    # 2. RSI (14 periodes)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))

    # 3. MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # 4. Support & Resistance (10 dagen)
    support10 = low.rolling(window=min(10, len(low))).min()
    resistance10 = high.rolling(window=min(10, len(high))).max()

    c_last = float(close.iloc[-1])
    sma5_last = float(sma5.iloc[-1]) if not pd.isna(sma5.iloc[-1]) else c_last
    sma5_prev = float(sma5.iloc[-2]) if len(sma5) > 1 and not pd.isna(sma5.iloc[-2]) else sma5_last
    sma20_last = float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else c_last
    sma50_last = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else c_last
    rsi_last = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    macd_last = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
    sig_last = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
    sup10_last = float(support10.iloc[-1]) if not pd.isna(support10.iloc[-1]) else c_last
    res10_last = float(resistance10.iloc[-1]) if not pd.isna(resistance10.iloc[-1]) else c_last

    sup_pct = ((sup10_last - c_last) / c_last) * 100
    res_pct = ((res10_last - c_last) / c_last) * 100

    # Short-Term Score (0-5)
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

    # Mid-Term Score (0-5)
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

    # Ensemble Score (0-100)
    st_score_100 = (st_points / 5.0) * 100
    mt_score_100 = (mt_points / 5.0) * 100
    ensemble_score = round((st_score_100 + mt_score_100) / 2, 1)

    if ensemble_score >=
