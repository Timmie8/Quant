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
    if df is None or len(df) < 20:
        return None

    # Eenduidige Series extractor (omzeilt MultiIndex / DataFrame kolom-issues)
    def extract_series(col_name):
        if col_name in df.columns:
            val = df[col_name]
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            return val.dropna()
        return None

    close = extract_series("Close")
    low = extract_series("Low")
    high = extract_series("High")

    if close is None or low is None or high is None or len(close) < 20:
        return None

    # 1. Moving Averages
    sma5 = close.rolling(window=5).mean()
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=min(50, len(close))).mean()

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
    sma5_last = float(sma5.iloc[-1]) if not pd.isna(sma5.iloc[-1]) else c_last
    sma5_prev = float(sma5.iloc[-2]) if len(sma5) > 1 and not pd.isna(sma5.iloc[-2]) else sma5_last
    sma20_last = float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else c_last
    sma50_last = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else c_last
    rsi_last = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    macd_last = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
    sig_last = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
    sup10_last = float(support10.iloc[-1]) if not pd.isna(support10.iloc[-1]) else c_last
    res10_last = float(resistance10.iloc[-1]) if not pd.isna(resistance10.iloc[-1]) else c_last

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

    # Bouw opgeschoond DF voor Plotly grafiek
    chart_df = pd.DataFrame({
        "Open": extract_series("Open"),
        "High": high,
        "Low": low,
        "Close": close
    }).dropna()

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
        "df": chart_df,
        "sma5": sma5,
        "sma20": sma20,
        "sma50": sma50,
        "support10": support10,
        "resistance10": resistance10,
    }


@st.cache_data(ttl=300)
def fetch_and_scan(tickers: list):
    """Data ophalen via yfinance met caching en MultiIndex afhandeling."""
    results = {}
    if not tickers:
        return results

    try:
        data = yf.download(
            tickers,
            period="6m",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        data = None

    for ticker in tickers:
        df_ticker = None
        try:
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.levels[0]:
                        df_ticker = data[ticker].dropna(how="all")
                    elif ticker in data.columns.levels[1]:
                        df_ticker = data.xs(ticker, level=1, axis=1).dropna(how="all")
                else:
                    df_ticker = data.dropna(how="all")

            # Fallback direct via Ticker.history als bulk download geen resultaat gaf
            if df_ticker is None or df_ticker.empty or len(df_ticker) < 10:
                t_obj = yf.Ticker(ticker)
                df_ticker = t_obj.history(period="6m")

            if df_ticker is not None and not df_ticker.empty:
                res = calculate_stoxline_and_ensemble(df_ticker)
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
    with st.spinner(f"Live data ophalen via yfinance voor {len(tickers)} USA aandelen..."):
        scan_results = fetch_and_scan(tickers)

    if scan_results:
        # Overzichtstabel opbouwen
        table_data = []
        for ticker, res in scan_results.items():
            table_data.append(
                {
                    "Ticker": ticker,
                    "Koers ($)": f"${res['Koers']:.2f}",
                    "AI Score": res["AI Score"],
                    "Sentiment": res["Sentiment"],
                    "Short-Term Rating": res["Short-Term Rating"],
                    "Mid-Term Rating": res["Mid-Term Rating"],
                    "RSI (14)": res["RSI (14)"],
                    "Support (10d)": f"${res['Support (10d)']:.2f} ({res['Support %']}%)",
                    "Resistance (10d)": f"${res['Resistance (10d)']:.2f} (+{res['Resistance %']}%)",
                }
            )

        df_summary = pd.DataFrame(table_data).sort_values(
            by="AI Score", ascending=False
        )

        # Highlight Metrics
        col1, col2, col3 = st.columns(3)
        top_stock = df_summary.iloc[0]
        col1.metric(
            "⭐ Hoogste AI Score",
            top_stock["Ticker"],
            f"{top_stock['AI Score']}/100",
        )
        col2.metric("📊 Totaal Gescand", len(df_summary))
        col3.metric(
            "🟢 Bullish Aandelen",
            len(
                df_summary[df_summary["Sentiment"].str.contains("Bullish")]
            ),
        )

        st.markdown("### 📋 Quant Ensemble Overzicht")
        st.dataframe(
            df_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "AI Score": st.column_config.ProgressColumn(
                    "AI Score",
                    help="Quant Ensemble AI Score (0-100)",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

        st.markdown("---")
        st.markdown("### 📈 Detail Analyse & Grafiek per USA Aandeel")

        selected_ticker = st.selectbox(
            "Kies een aandeel om de grafiek te bekijken:",
            list(scan_results.keys()),
        )

        if selected_ticker:
            stock = scan_results[selected_ticker]
            df_chart = stock["df"]

            # Plotly Interactieve Grafiek
            fig = go.Figure()

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df_chart.index,
                    open=df_chart["Open"],
                    high=df_chart["High"],
                    low=df_chart["Low"],
                    close=df_chart["Close"],
                    name="Koers",
                )
            )

            # Moving Averages
            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=stock["sma5"],
                    mode="lines",
                    name="SMA 5",
                    line=dict(color="orange", width=1),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=stock["sma20"],
                    mode="lines",
                    name="SMA 20",
                    line=dict(color="blue", width=1.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=stock["sma50"],
                    mode="lines",
                    name="SMA 50",
                    line=dict(color="purple", width=1.5),
                )
            )

            # Support & Resistance
            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=stock["support10"],
                    mode="lines",
                    name="Support 10d",
                    line=dict(color="red", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=stock["resistance10"],
                    mode="lines",
                    name="Resistance 10d",
                    line=dict(color="green", dash="dash"),
                )
            )

            fig.update_layout(
                title=f"{selected_ticker} (US) - Technische Grafiek & Support/Resistance",
                xaxis_title="Datum",
                yaxis_title="Prijs ($)",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Geen data gevonden voor de opgegeven USA tickers. Controleer je internetverbinding of voer geldige USA tickers in.")
else:
    st.info("Voer minimaal één ticker in in de sidebar om te scannen.")
