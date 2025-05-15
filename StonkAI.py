import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datasets import load_dataset
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from gpt4all import GPT4All
from transformers import pipeline
import concurrent.futures

# Load Mistral model using GPT4All and cache it
@st.cache_resource
def load_mistral_model():
    return GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", model_path="models")

mistral_model = load_mistral_model()

# Load FinGPT dataset and convert to pandas DataFrame
@st.cache_resource
def load_fingpt_dataset():
    dataset = load_dataset("FinGPT/fingpt-forecaster-dow30-202305-202405")
    return dataset["train"].to_pandas()

df_fingpt = load_fingpt_dataset()

# Load sentiment analysis model (FinBERT)
@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="yiyanghkust/finbert-tone", return_all_scores=True)

sentiment_model = load_sentiment_model()

# Fetch historical stock data using yfinance
@st.cache_data
def get_stock_data(ticker, period="1y", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data if not data.empty else None
    except Exception:
        return None

# Fetch stock metadata
@st.cache_data
def get_stock_metadata(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Name": info.get("shortName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Current Price": info.get("currentPrice", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": info.get("fiftyTwoWeekLow", "N/A")
        }
    except Exception:
        return None

# Add moving averages
def add_moving_averages(df):
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    return df

# Train Decision Tree
@st.cache_data
def train_decision_tree(df):
    df = add_moving_averages(df)
    df[["MA50", "MA200", "Close"]] = df[["MA50", "MA200", "Close"]].fillna(df[["MA50", "MA200", "Close"]].mean())
    df = df.dropna(subset=["Close"])
    if df.shape[0] < 50:
        return None, None, None

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=["Target"])

    features = df[["Close", "MA50", "MA200"]]
    target = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy, features.columns.tolist()

# Async Mistral prediction
def predict_stock_trend(ticker, model, features, df):
    if model is None:
        return "Not enough data to generate prediction."

    decision = "increase" if model.predict(df[features].iloc[-1].values.reshape(1, -1))[0] == 1 else "decrease"
    prompt = f"""
    You are a financial analyst writing a full stock forecast report for {ticker}.

    Include the following:
    - A summary of the stock's recent trends.
    - [Positive Developments] - at least 3 bullish indicators from data or news.
    - [Potential Concerns] - at least 3 bearish indicators or risks.
    - [Decision Tree Insight] - explain how the model predicted a price {decision}, and technical reasons why.
    """
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(mistral_model.generate, prompt, max_tokens=600, temp=0.7)
            return future.result()
    except Exception as e:
        return f"Error generating prediction: {e}"

# Cached sentiment analysis
@st.cache_data
def get_stock_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={YOUR_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    articles = response.json().get("articles", [])[:10]
    sentiments = {"Positive": [], "Neutral": [], "Negative": []}

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        url = article.get("url", "")
        text = f"{title}. {description}"

        sentiment_result = sentiment_model(text)
        if isinstance(sentiment_result, list) and len(sentiment_result) > 0 and isinstance(sentiment_result[0], list):
            sentiment_result = sentiment_result[0]
        dominant_sentiment = max(sentiment_result, key=lambda x: x["score"])["label"]
        sentiments[dominant_sentiment].append((title, url))

    return sentiments

# Streamlit UI
st.title("StonkAI 📈")
st.write("Enter a stonk ticker to get AI-powered predictions!")
st.write("Disclaimer: This AI is NOT 100% accurate...")

ticker = st.text_input("Enter Stonk Ticker (e.g., AAPL, TSLA)").upper()

if ticker:
    metadata = get_stock_metadata(ticker)

    if metadata:
        st.subheader(f"📊 Stonk Info: {metadata['Name']}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Sector:** {metadata['Sector']}")
            st.write(f"**Market Cap:** {metadata['Market Cap']:,}")
        with col2:
            st.write(f"**Current Price:** ${metadata['Current Price']}")
            st.write(f"**52-Week High:** ${metadata['52-Week High']}")
            st.write(f"**52-Week Low:** ${metadata['52-Week Low']}")

        df = get_stock_data(ticker)
        if df is not None and not df.empty:
            st.subheader(f"📈 Stonk Data for {ticker}")
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name=ticker)])
            fig.update_layout(title=f"{ticker} Stonk Price", xaxis_title="Date", yaxis_title="Price (USD)", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)

            st.subheader("📈 Moving Averages")
            df = add_moving_averages(df)
            st.line_chart(df[["Close", "MA50", "MA200"]])

            st.subheader("📰 News Headlines")
            news = get_stock_news_sentiment(ticker)
            if news:
                for sentiment, articles in news.items():
                    st.write(f"**{sentiment} News:**")
                    for title, link in articles:
                        st.markdown(f"- [{title}]({link})")
            else:
                st.warning("No news data available.")

            model, accuracy, features = train_decision_tree(df)

            st.subheader("🤖 Stonk Prediction:")
            prediction = predict_stock_trend(ticker, model, features, df)
            st.text(prediction)

            st.subheader("🌳 Decision Tree Model Prediction")
            if model:
                latest = df[features].iloc[-1].values.reshape(1, -1)
                result = model.predict(latest)
                st.write("📌 Decision Tree says:", "⬆️ Price Increase Expected" if result[0] == 1 else "🔽️ Price Decrease Expected")
                st.caption(f"(Model Accuracy: {accuracy:.2%})")
        else:
            st.warning(f"No historical data available for {ticker}.")
    else:
        st.error(f"Invalid ticker: {ticker}. Please enter a valid stock symbol.")
