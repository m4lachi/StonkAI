# 📈 StonkAI – AI-Powered Stock Analysis & Prediction Web App

StonkAI is a GenAI-powered web application that provides stock analysis, sentiment insights, and price movement predictions using a blend of financial data, deep learning models (like LLaMA 2), and NLP models like FinBERT. Built with Streamlit, this app lets users explore stock performance, evaluate recent news sentiment, and receive AI-generated forecasts for U.S. stocks.

---

## 🚀 Features

- 🔍 **Real-time Stock Metadata**  
  View stock info including current price, market cap, sector, 52-week high/low, and more.

- 📊 **Interactive Stock Charts**  
  Visualize historical data using candlestick and line charts with moving averages (MA50 and MA200).

- 🧠 **AI-Based Trend Prediction**  
  Generate natural language predictions using Meta's LLaMA-2 and fine-tuned FinGPT forecasting models.

- 📰 **News Sentiment Analysis**  
  Analyze recent news headlines and descriptions using FinBERT to provide a sentiment breakdown.

- 📚 **Preloaded Historical Trends**  
  Explore trends and model training data from the `FinGPT/fingpt-forecaster-dow30-202305-202405` dataset.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data & Charting**: `yfinance`, `pandas`, `plotly`
- **NLP & Forecasting**:
  - LLaMA-2 via `transformers`
  - FinBERT (`yiyanghkust/finbert-tone`)
  - FinGPT's `fingpt-forecaster-dow30-202305-202405`
- **APIs**:
  - [Yahoo Finance](https://pypi.org/project/yfinance/)
  - [NewsAPI](https://newsapi.org/)

---

## 🧠 How It Works

1. **Enter a stock ticker (e.g., AAPL, TSLA)**
2. The app fetches:
   - Historical stock data & financials
   - Technical indicators like moving averages & RSI
   - News headlines for sentiment analysis
3. A prompt is dynamically built with financial stats
4. The LLaMA-2 model generates a structured prediction:
   - Positive Developments  
   - Potential Concerns  
   - Prediction & Analysis

---

## 🖼️ Screenshots

> _(Optional: Add images of the Streamlit UI, stock chart visualizations, and AI-generated prediction outputs.)_

---

## 🧪 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/stonk-ai.git
cd stonk-ai

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Run the App

```bash
streamlit run app.py
```

---

## 🔑 API Keys

You’ll need a **NewsAPI key**. Replace the placeholder in the code:

```python
url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={'YOUR_API_KEY'}"
```

Get yours from: [https://newsapi.org/](https://newsapi.org/)

---

## 📝 Disclaimer

This AI tool provides **technical analysis-based predictions** and **should not** be used as financial advice. It does **not** consider fundamental data, geopolitical events, or macroeconomic trends. Always consult a financial advisor before making investment decisions.

---

## 🤝 Credits

- [Meta's LLaMA-2](https://ai.meta.com/llama/)
- [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone)
- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
- [NewsAPI](https://newsapi.org/)
- [Yahoo Finance](https://finance.yahoo.com/)

