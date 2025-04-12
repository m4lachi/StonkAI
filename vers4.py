import streamlit as st
import yfinance as yf
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import requests
import asyncio
from datasets import load_dataset
import plotly.graph_objects as go

# Load dataset from Hugging Face
dataset = load_dataset("FinGPT/fingpt-forecaster-dow30-202305-202405")

# Convert to Pandas DataFrame
df_fingpt = dataset["train"].to_pandas()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Load sentiment analysis model (FinBERT)
sentiment_model = pipeline("text-classification", model="yiyanghkust/finbert-tone", return_all_scores=True)

# Define model paths
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
lora_model_path = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.float16,  # Use float16 to save memory
    device_map="auto"           # Automatically selects CPU/GPU
)

# Check if the model loaded successfully
#if not fin_model or "model" not in fin_model or "tokenizer" not in fin_model:
#    st.error("AI Model failed to load.")
#else:
#   st.success("AI Model successfully loaded!")

# Function to fetch stock data
def get_stock_data(ticker, period="1y", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)

        return data if not data.empty else None
    except Exception:
        return None

# Function to fetch stock metadata
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

# Function to look into past trends from the dataset
def get_stock_trend_from_dataset(ticker):
    """
    Fetch past trends from dataset for the given stock.
    """
    for entry in dataset["train"]:
        if entry["ticker"] == ticker:
            return entry["trend"]  # Assuming 'trend' contains stock movement insights
    return "No trend data available."

# Function to predict stock trend
def predict_stock_trend(ticker):
    df = get_stock_data(ticker, period="1y", interval="1d")
    if df is None:
        return "No data available for prediction."

    latest_price = df["Close"].iloc[-1]
    ma50 = df["Close"].rolling(window=50).mean().iloc[-1]
    ma200 = df["Close"].rolling(window=200).mean().iloc[-1]
    rsi_series = df["Close"].pct_change().rolling(14).mean()
    rsi = "N/A" if rsi_series.iloc[-1] == 0 else 100 - (100 / (1 + rsi_series.iloc[-1]))

    prompt = f"""
    Stock: {ticker}
    Current Price: {latest_price}
    50-Day MA: {ma50}
    200-Day MA: {ma200}
    RSI: {rsi}
    """
    sysPrompt = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"

    try:
        input_ids = tokenizer(sysPrompt, return_tensors="pt").input_ids
        output = base_model.generate(input_ids, max_new_tokens=500, do_sample=True, temperature=0.7)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        return prediction
    except Exception as e:
        return f"Error during generation: {e}"

def get_stock_news_sentiment(ticker):
    """
    Fetches the latest news articles about the stock and analyzes sentiment.
    :param ticker: Stock ticker (e.g., AAPL)
    :return: Sentiment breakdown (positive/negative/neutral)
    """
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={"API_KEY"}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None
    
    news_articles = response.json().get("articles", [])[:15]  # Limit to 10 articles
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for article in news_articles:
        # Ensure no None values
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        text = title + ". " + description  # Now safe to concatenate

        sentiment_result = sentiment_model(text)

        # Debugging: Print to check structure
        print("Sentiment Model Output:", sentiment_result)

        if isinstance(sentiment_result, list) and len(sentiment_result) > 0 and isinstance(sentiment_result[0], list):
            sentiment_result = sentiment_result[0]  # Unwrap nested list if needed

        dominant_sentiment = max(sentiment_result, key=lambda x: x["score"])["label"]
        sentiments[dominant_sentiment] += 1

    return sentiments

def add_moving_averages(df):
    """
    Adds 50-day and 200-day moving averages to the stock price DataFrame.
    """
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    return df

# Streamlit UI
st.title("StonkAI 📈")
st.write("Enter a stock ticker to get AI-powered predictions!")
st.write("Disclaimer: This AI is NOT 100 percent accurate, this prediction is based solely on technical analysis and does not take into account any external factors that may affect the stock's price, such as economic indicators, industry trends, or geopolitical events. As such, the prediction should be treated as a general guide rather than a definitive forecast.")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)").upper()

if ticker:
    # Fetch stock metadata
    metadata = get_stock_metadata(ticker)
    
    if metadata:
        # Display metadata
        st.subheader(f"📊 Stock Info: {metadata['Name']}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Sector:** {metadata['Sector']}")
            st.write(f"**Market Cap:** {metadata['Market Cap']:,}")
        with col2:
            st.write(f"**Current Price:** ${metadata['Current Price']}")
            st.write(f"**52-Week High:** ${metadata['52-Week High']}")
            st.write(f"**52-Week Low:** ${metadata['52-Week Low']}")

        # Fetch and display stock data with candlestick chart
        st.subheader(f"📈 Stock Data for {ticker}")
        df = get_stock_data(ticker)

        if df is not None and not df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker
            )])

            fig.update_layout(
                title=f"{ticker} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig)
        else:
            st.warning(f"No historical data available for {ticker}.")

        
        # Fetch and display stock sentiment
        st.subheader(f"📰 Sentiment Analysis for {ticker}")
        sentiments = get_stock_news_sentiment(ticker)
        if sentiments:                
            st.write(f"**Positive News:** {sentiments['Positive']} articles")
            st.write(f"**Neutral News:** {sentiments['Neutral']} articles")
            st.write(f"**Negative News:** {sentiments['Negative']} articles")
        else:
            st.warning(f"No news found for {ticker}.")

        # Fetch and display stock data with MA
        df = get_stock_data(ticker)
        if df is not None:
            df = add_moving_averages(df)
            st.subheader(f"📈 Stock Price & Moving Averages for {ticker}")
            st.line_chart(df[["Close", "MA50", "MA200"]])  # Display Close price + moving averages

        # AI Prediction
        st.subheader("🤖 AI Prediction:")
        prediction = predict_stock_trend(ticker)
        st.text(prediction)
    else:
        st.error(f"Invalid ticker: {ticker}. Please enter a valid stock symbol.")
