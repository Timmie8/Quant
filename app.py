from datetime import datetime
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ==========================================
# STREAMLIT PAGINA CONFIGURATIE
# ==========================================
st.set_page_config(
    page_title="Stoxline & Quant Ensemble AI Scanner",
    page_icon="📈",
    layout="wide",
)


# ==========================================
# REKENFUNCTIE (STOXLINE & QUANT ENSEMBLE)
# ==========================================
def calculate_stoxline_and_ensemble(df: pd.DataFrame) -> dict:
    if len(df) < 50:
        return None

    close = df["Close"]
    low = df["Low"]
    high = df["High"]

    # Indicatoren
    sma5 = ta.sma(close, length=5)
    sma20 = ta.sma(close, length=20)
    sma50 = ta.sma(close, length=50)
    rsi_series = ta.rsi(close, length=14)

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    macd_line = macd_df["MACD_12_26_9"]
    signal_line = macd_df["MACDs_12_26_9"]

    support10 = low.rolling(window=10).min()
    resistance10 = high.rolling(window=10).max()

    # Laatste geldige waarden
    c_last = close.iloc[-1]
    sma5_last = sma5.iloc[-1]
    sma5_prev = sma5.iloc[-2]
    sma20_last = sma20.iloc[-1]
    sma50_last = sma50.iloc[-1]
    rsi_last = rsi_series.iloc[-1]
    macd_last = macd_line.iloc[-1]
    sig_last = signal_line.iloc[-1]
    sup10_last = support10.iloc[-1]
    res10_last = resistance10.iloc[-1]

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
    """Data ophalen met Streamlit caching (5 minuten)."""
    results = {}
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
        except Exception as e:
            st.error(f"Fout bij ophalen {ticker}: {e}")

    return results


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🔍 Scanner Instellingen")
default_tickers = "ASML.AS, NVDA, TSLA, AAPL, MSFT, AMZN, META"
user_input = st.sidebar.text_area(
    "Vul tickers in (gescheiden door komma's):", default_tickers, height=100
)

tickers = [
    t.strip().upper() for t in user_input.split(",") if t.strip() != ""
]

refresh_btn = st.sidebar.button("🔄 Herlaad Data", use_container_width=True)
if refresh_btn:
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.info(
    "**Logica:**\n"
    "- Short-Term (SMA5, RSI, MACD Cross)\n"
    "- Mid-Term (SMA20, SMA50, MACD > 0)\n"
    "- AI Ensemble Score = Gemiddelde gewogen %"
)

# ==========================================
# HOOFDSCHERM
# ==========================================
st.title("📊 Stoxline & Quant Ensemble AI Dashboard")
st.caption(f"Laatst bijgewerkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if tickers:
    with st.spinner("Live koersen & indicatoren ophalen..."):
        scan_results = fetch_and_scan(tickers)

    if scan_results:
        # Tabeloverzicht maken
        table_data = []
        for ticker, res in scan_results.items():
            table_data.append({
                "Ticker": ticker,
                "Koers ($/€)": res["Koers"],
                "AI Score": res["AI Score"],
                "Sentiment": res["Sentiment"],
                "Short-Term Rating": res["Short-Term Rating"],
                "Mid-Term Rating": res["Mid-Term Rating"],
                "RSI (14)": res["RSI (14)"],
                "Support (10d)": f"{res['Support (10d)']} ({res['Support %']}%)",
                "Resistance (10d)": f"{res['Resistance (10d)']} (+{res['Resistance %']}%)",
            })

        df_summary = pd.DataFrame(table_data).sort_values(
            by="AI Score", ascending=False
        )

        # Top Statistieken Metric Cards
        col1, col2, col3 = st.columns(3)
        top_stock = df_summary.iloc[0]
        col1.metric("⭐ Highest AI Score", top_stock["Ticker"], f"{top_stock['AI Score']}/100")
        col2.metric("📊 Totaal Gescand", len(df_summary))
        col3.metric("🟢 Bullish Aandelen", len(df_summary[df_summary["Sentiment"].str.contains("Bullish")]))

        st.markdown("### 📋 Quant Ensemble Overzicht")
        st.dataframe(
            df_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "AI Score": st.column_config.ProgressColumn(
                    "AI Score",
                    help="Quant Ensemble AI Score (0-100)",
                    format="%f",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

        st.markdown("---")
        st.markdown("### 📈 Detail Analyse & Grafiek per Aandeel")

        selected_ticker = st.selectbox("Kies een aandeel om de grafiek te bekijken:", list(scan_results.keys()))

        if selected_ticker:
            stock = scan_results[selected_ticker]
            df_chart = stock["df"]

            # Plotly Kandelaar + Moving Averages Grafiek
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_chart.index,
                open=df_chart["Open"],
                high=df_chart["High"],
                low=df_chart["Low"],
                close=df_chart["Close"],
                name="Koers"
            ))

            # Moving Averages
            fig.add_trace(go.Scatter(x=df_chart.index, y=stock["sma5"], mode="lines", name="SMA 5", line=dict(color="orange", width=1)))
            fig.add_trace(go.Scatter(x=df_chart.index, y=stock["sma20"], mode="lines", name="SMA 20", line=dict(color="blue", width=1.5)))
            fig.add_trace(go.Scatter(x=df_chart.index, y=stock["sma50"], mode="lines", name="SMA 50", line=dict(color="purple", width=1.5)))

            # Support & Resistance
            fig.add_trace(go.Scatter(x=df_chart.index, y=stock["support10"], mode="lines", name="Support 10d", line=dict(color="red", dash="dash")))
            fig.add_trace(go.Scatter(x=df_chart.index, y=stock["resistance10"], mode="lines", name="Resistance 10d", line=dict(color="green", dash="dash")))

            fig.update_layout(
                title=f"{selected_ticker} - Technische Grafiek & Support/Resistance",
                xaxis_title="Datum",
                yaxis_title="Prijs",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Geen data gevonden voor de opgegeven tickers.")
else:
    st.info("Voer minimaal één ticker in in de sidebar om te scannen.")
