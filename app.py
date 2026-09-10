"""
Real-Time IT Stock Analysis Dashboard
Run locally with: streamlit run app.py

Requires: it_stock_multivariate_lstm.h5, multi_scaler.pkl, feature_list.pkl
(generated in the Colab notebook — download them from Colab and place them
in the same folder as this app.py before running)
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import mplfinance as mpf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(page_title="IT Stock Predictor", layout="wide")
st.title("Real-Time IT Stock Analysis (RNN/LSTM)")
st.caption("Predicts next-day closing price for Indian IT sector stocks using a trained LSTM model.")

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_assets():
    model = load_model("it_stock_multivariate_lstm.h5")
    scaler = joblib.load("multi_scaler.pkl")
    features = joblib.load("feature_list.pkl")
    return model, scaler, features

try:
    model, scaler, FEATURES = load_assets()
    assets_loaded = True
except Exception:
    assets_loaded = False
    st.error(
        "Model files not found. Please run the Colab notebook through Step 19, "
        "download it_stock_multivariate_lstm.h5, multi_scaler.pkl, and feature_list.pkl, "
        "and place them in this app's folder."
    )

WINDOW_SIZE = 60

STOCKS = {
    "Infosys (Large Cap)": {"ticker": "INFY.NS", "search_name": "Infosys"},
    "TCS (Large Cap)": {"ticker": "TCS.NS", "search_name": "TCS"},
    "Wipro (Large Cap)": {"ticker": "WIPRO.NS", "search_name": "Wipro"},
    "HCLTech (Large Cap)": {"ticker": "HCLTECH.NS", "search_name": "HCLTech"},
    "TechMahindra (Large Cap)": {"ticker": "TECHM.NS", "search_name": "Tech Mahindra"},
    "Persistent Systems (Mid Cap)": {"ticker": "PERSISTENT.NS", "search_name": "Persistent Systems"},
    "Zensar Technologies (Small Cap)": {"ticker": "ZENSARTECH.NS", "search_name": "Zensar Technologies"},
}

# ---------------- Sidebar ----------------
st.sidebar.header("Select Stock")
choice = st.sidebar.selectbox("IT Company", list(STOCKS.keys()))
ticker = STOCKS[choice]["ticker"]
search_name = STOCKS[choice]["search_name"]

run_button = st.sidebar.button("Fetch & Predict")

TIMEFRAMES = {
    "1D": {"period": "1d", "interval": "5m"},
    "1W": {"period": "7d", "interval": "15m"},
    "1M": {"period": "1mo", "interval": "1d"},
    "3M": {"period": "3mo", "interval": "1d"},
    "6M": {"period": "6mo", "interval": "1d"},
    "1Y": {"period": "1y", "interval": "1d"},
    "3Y": {"period": "3y", "interval": "1wk"},
    "5Y": {"period": "5y", "interval": "1wk"},
    "All": {"period": "max", "interval": "1mo"},
}


# ---------------- Helpers ----------------
def flatten_columns(df):
    """Newer yfinance versions can return MultiIndex columns even for a
    single ticker — flatten them so downstream code always sees plain
    column names like 'Close' instead of ('Close', 'INFY.NS')."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def to_scalar(value):
    """Safely convert a pandas/numpy value (which may be a 0-d array,
    1-element array, or plain scalar) into a plain Python float."""
    return float(np.ravel(value)[0])


def fetch_multivariate_data(ticker_symbol, period="3y"):
    df = yf.download(ticker_symbol, period=period, interval="1d", progress=False)
    df = flatten_columns(df)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df


@st.cache_data(ttl=900)  # cache for 15 minutes so the overview table loads fast
def fetch_all_stocks_ohlc(stock_dict):
    """Latest Open/High/Low/Close (+ % change) for every stock in the list."""
    rows = []
    for name, info in stock_dict.items():
        sym = info["ticker"]
        try:
            hist = yf.download(sym, period="5d", interval="1d", progress=False)
            hist = flatten_columns(hist)
            hist = hist.dropna()
            if len(hist) < 2:
                continue
            latest = hist.iloc[-1]
            prev_close = hist.iloc[-2]["Close"]
            pct_change = ((latest["Close"] - prev_close) / prev_close) * 100
            rows.append({
                "Company": name,
                "Ticker": sym,
                "Open": round(to_scalar(latest["Open"]), 2),
                "High": round(to_scalar(latest["High"]), 2),
                "Low": round(to_scalar(latest["Low"]), 2),
                "Close": round(to_scalar(latest["Close"]), 2),
                "Change %": round(to_scalar(pct_change), 2),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_candlestick_data(ticker_symbol, period, interval):
    """OHLC data for the candlestick chart at the chosen timeframe."""
    df = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
    df = flatten_columns(df)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


@st.cache_data(ttl=1800)  # news doesn't need to refresh every click — cache 30 min
def fetch_news_sentiment(company_search_name, max_articles=15):
    """Fetch recent news headlines (Google News RSS, free, no API key) and
    score each one with VADER sentiment analysis."""
    query = company_search_name.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"

    feed = feedparser.parse(url)
    analyzer = SentimentIntensityAnalyzer()

    rows = []
    for entry in feed.entries[:max_articles]:
        title = entry.title
        score = analyzer.polarity_scores(title)["compound"]
        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        rows.append({
            "Headline": title,
            "Published": entry.get("published", "Unknown"),
            "Sentiment Score": round(score, 3),
            "Sentiment": label,
        })

    return pd.DataFrame(rows)


def generate_fusion_recommendation(df, pred_price, latest_close, news_sentiment_avg, has_news):
    """Rule-based signal — NOT financial advice.
    Combines trend (MA50 vs MA200), momentum (RSI), the model's own
    next-day prediction, AND recent news sentiment into one score.
    """
    score = 0
    reasons = []

    ma50 = to_scalar(df["MA_50"].values[-1])
    ma200 = to_scalar(df["MA_200"].values[-1])
    rsi = to_scalar(df["RSI"].values[-1])

    if ma50 > ma200:
        score += 1
        reasons.append("Technical trend: 50-day average is above the 200-day average (uptrend).")
    else:
        score -= 1
        reasons.append("Technical trend: 50-day average is below the 200-day average (downtrend).")

    if rsi > 70:
        score -= 1
        reasons.append(f"Momentum: RSI is {rsi:.1f} — stock looks overbought.")
    elif rsi < 30:
        score += 1
        reasons.append(f"Momentum: RSI is {rsi:.1f} — stock looks oversold, possible rebound zone.")
    else:
        reasons.append(f"Momentum: RSI is {rsi:.1f} — neutral range.")

    predicted_change_pct = ((pred_price - latest_close) / latest_close) * 100
    if predicted_change_pct > 0.5:
        score += 1
        reasons.append(f"Model prediction: LSTM expects a rise of {predicted_change_pct:.2f}% next session.")
    elif predicted_change_pct < -0.5:
        score -= 1
        reasons.append(f"Model prediction: LSTM expects a fall of {predicted_change_pct:.2f}% next session.")
    else:
        reasons.append("Model prediction: LSTM expects little movement next session.")

    if has_news:
        if news_sentiment_avg >= 0.15:
            score += 1
            reasons.append(f"News sentiment: Positive coverage (avg score {news_sentiment_avg:.2f}).")
        elif news_sentiment_avg <= -0.15:
            score -= 1
            reasons.append(f"News sentiment: Negative coverage (avg score {news_sentiment_avg:.2f}).")
        else:
            reasons.append(f"News sentiment: Neutral coverage (avg score {news_sentiment_avg:.2f}).")
    else:
        reasons.append("News sentiment: No recent headlines found — factor skipped.")

    if score >= 3:
        label, color = "Strong Bullish Signal", "green"
    elif score >= 1:
        label, color = "Bullish Signal", "green"
    elif score <= -3:
        label, color = "Strong Bearish Signal", "red"
    elif score <= -1:
        label, color = "Bearish Signal", "red"
    else:
        label, color = "Neutral / Mixed Signal", "orange"

    return label, color, reasons


# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["Selected Stock", "Market Overview (All Stocks)"])

# ================= TAB 2: Market Overview Table (all stocks) =================
with tab2:
    st.subheader("Open / High / Low / Close — All Tracked Stocks")
    st.caption("Latest available trading day for each stock.")
    with st.spinner("Fetching latest data for all stocks..."):
        overview_df = fetch_all_stocks_ohlc(STOCKS)

    if overview_df.empty:
        st.warning("Could not fetch data right now. Try again in a moment.")
    else:
        st.dataframe(
            overview_df.style.applymap(
                lambda v: "color: green;" if isinstance(v, (int, float)) and v > 0
                else ("color: red;" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["Change %"]
            ),
            use_container_width=True,
            hide_index=True,
        )

# ================= TAB 1: Selected Stock — Chart, Summary, Prediction, News, Recommendation =================
with tab1:
    # Remember which stock was fetched, so switching the timeframe below the
    # chart or news below doesn't require clicking "Fetch & Predict" again.
    if run_button:
        st.session_state["fetched"] = True
        st.session_state["choice"] = choice
        st.session_state["ticker"] = ticker
        st.session_state["search_name"] = search_name
        st.session_state["selected_timeframe"] = None  # reset so old chart doesn't carry over

    if st.session_state.get("fetched") and assets_loaded:
        active_choice = st.session_state["choice"]
        active_ticker = st.session_state["ticker"]
        active_search_name = st.session_state["search_name"]

        with st.spinner(f"Fetching latest data for {active_choice}..."):
            df = fetch_multivariate_data(active_ticker, period="3y")

        if len(df) < WINDOW_SIZE:
            st.error("Not enough recent data to make a prediction.")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                # 1) Original closing-price line chart — always shown
                st.subheader(f"{active_choice} — Last 6 Months Closing Price")
                fig_line, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index[-130:], df["Close"].values[-130:], color="black")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (INR)")
                ax.grid(alpha=0.3)
                st.pyplot(fig_line)

                # 2) Timeframe buttons — candlestick chart appears ONLY
                #    after a timeframe button is clicked, right below.
                st.write("**View detailed chart for a specific period:**")
                tf_cols = st.columns(len(TIMEFRAMES))
                for i, tf_name in enumerate(TIMEFRAMES.keys()):
                    if tf_cols[i].button(tf_name, key=f"tf_btn_{tf_name}"):
                        st.session_state["selected_timeframe"] = tf_name

                if st.session_state.get("selected_timeframe"):
                    tf_name = st.session_state["selected_timeframe"]
                    tf = TIMEFRAMES[tf_name]
                    st.subheader(f"{active_choice} — {tf_name} Chart (OHLC)")
                    with st.spinner("Loading chart..."):
                        candle_df = fetch_candlestick_data(active_ticker, tf["period"], tf["interval"])

                    if candle_df.empty or len(candle_df) < 2:
                        st.warning("Not enough data available for this timeframe.")
                    else:
                        fig_candle, _ = mpf.plot(
                            candle_df,
                            type="candle",
                            style="charles",
                            volume=True,
                            returnfig=True,
                            figsize=(10, 6),
                        )
                        st.pyplot(fig_candle)

            # ---------- Prepare prediction ----------
            last_window_raw = df[FEATURES].values[-WINDOW_SIZE:]
            last_window_scaled = scaler.transform(last_window_raw)
            last_window_scaled = last_window_scaled.reshape(1, WINDOW_SIZE, len(FEATURES))

            pred_scaled = model.predict(last_window_scaled, verbose=0)

            target_idx = FEATURES.index("Close")
            dummy = np.zeros((1, len(FEATURES)))
            dummy[0, target_idx] = pred_scaled.flatten()[0]
            pred_price = to_scalar(scaler.inverse_transform(dummy)[0, target_idx])

            latest_close = to_scalar(df["Close"].values[-1])
            change = pred_price - latest_close
            change_pct = (change / latest_close) * 100
            rsi_latest = to_scalar(df["RSI"].values[-1])

            with col2:
                st.subheader("Prediction")
                st.metric(
                    label="Predicted Next Close",
                    value=f"₹{pred_price:,.2f}",
                    delta=f"{change_pct:+.2f}%"
                )
                st.write(f"**Latest actual close:** ₹{latest_close:,.2f}")
                st.write(f"**As of:** {df.index[-1].date()}")
                st.write(f"**Current RSI:** {rsi_latest:.1f}")

            st.divider()

            # ---------- Summary Section ----------
            st.subheader("Summary")
            period_high = to_scalar(df["Close"].max())
            period_low = to_scalar(df["Close"].min())
            avg_close = to_scalar(df["Close"].mean())
            daily_returns = df["Close"].pct_change().dropna()
            volatility_pct = to_scalar(daily_returns.std()) * 100
            ma50 = to_scalar(df["MA_50"].values[-1])
            ma200 = to_scalar(df["MA_200"].values[-1])
            trend_text = "Uptrend (50-day MA above 200-day MA)" if ma50 > ma200 else "Downtrend (50-day MA below 200-day MA)"

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("3Y High", f"₹{period_high:,.2f}")
            s2.metric("3Y Low", f"₹{period_low:,.2f}")
            s3.metric("3Y Average", f"₹{avg_close:,.2f}")
            s4.metric("Daily Volatility", f"{volatility_pct:.2f}%")

            st.write(
                f"**{active_choice}** is currently in a **{trend_text.lower()}** based on moving averages. "
                f"Over the last 3 years, the stock has traded between ₹{period_low:,.2f} and ₹{period_high:,.2f}, "
                f"averaging around ₹{avg_close:,.2f}, with average daily volatility of {volatility_pct:.2f}%."
            )

            st.divider()

            # ---------- News Sentiment Section ----------
            st.subheader("Recent News Sentiment")
            st.caption("Live headlines from Google News, scored with VADER sentiment analysis.")

            with st.spinner("Fetching recent news..."):
                news_df = fetch_news_sentiment(active_search_name)

            has_news = not news_df.empty

            if not has_news:
                avg_sentiment = 0.0
                st.info("No recent news headlines found for this stock.")
            else:
                avg_sentiment = news_df["Sentiment Score"].mean()

                n1, n2, n3 = st.columns(3)
                n1.metric("Avg. Sentiment Score", f"{avg_sentiment:+.2f}")
                n2.metric("Positive Headlines", int((news_df["Sentiment"] == "Positive").sum()))
                n3.metric("Negative Headlines", int((news_df["Sentiment"] == "Negative").sum()))

                def highlight_sentiment(val):
                    if val == "Positive":
                        return "color: green;"
                    elif val == "Negative":
                        return "color: red;"
                    return "color: gray;"

                st.dataframe(
                    news_df.style.applymap(highlight_sentiment, subset=["Sentiment"]),
                    use_container_width=True,
                    hide_index=True,
                )

            st.divider()

            # ---------- Fusion Recommendation Section ----------
            st.subheader("Technical + News Signal (Educational Only)")
            label, color, reasons = generate_fusion_recommendation(
                df, pred_price, latest_close, avg_sentiment, has_news
            )

            st.markdown(f"### :{color}[{label}]")
            for r in reasons:
                st.write(f"- {r}")

            st.warning(
                "**Disclaimer:** This signal is generated purely from historical technical "
                "indicators (moving averages, RSI), this model's own prediction, and recent "
                "news headline sentiment. It is an academic/portfolio demonstration only — "
                "it is **not financial advice** and should not be used to make real investment "
                "or trading decisions. Please consult a licensed financial advisor before investing."
            )

    elif not assets_loaded:
        st.info("Waiting for model files to run predictions.")
    else:
        st.info("Select a stock and click **Fetch & Predict** in the sidebar to get started.")
