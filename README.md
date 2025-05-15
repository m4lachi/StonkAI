Here’s a clean, professional, and informative `README.md` file you can use for your `StonkAI` project:

---

# 📈 StonkAI: AI-Powered Stock Forecasting App

**StonkAI** is a powerful Streamlit-based web application that delivers real-time stock analysis and forecasts using a combination of traditional machine learning and modern AI language models—all running **locally on CPU**. It's built for investors, students, and data enthusiasts who want deeper, explainable stock insights.

---

## 🚀 Features

* 🔍 **Live Stock Data**: Fetches historical stock prices, metadata, and visual charts using `yfinance`.
* 🌿 **Decision Tree Predictions**: Trains a custom Decision Tree Classifier to forecast short-term stock direction.
* 🧠 **LLM-Powered Reports**: Uses the **Mistral-7B** model via GPT4All to generate detailed, analyst-style forecast reports.
* 📰 **Sentiment Analysis**: Scrapes and scores recent news headlines using FinBERT to gauge market sentiment.
* 📊 **Visual Insights**: Displays candlestick charts and moving averages (50 & 200-day) to track trends.
* ✅ **All Local**: No OpenAI API required—everything runs on your machine.

---

## 🛠️ Tech Stack

| Component                  | Usage                                       |
| -------------------------- | ------------------------------------------- |
| **Streamlit**              | UI framework                                |
| **yfinance**               | Historical stock data                       |
| **Plotly**                 | Candlestick and line charts                 |
| **scikit-learn**           | Decision Tree model                         |
| **GPT4All**                | Mistral-7B LLM for forecasts                |
| **transformers (FinBERT)** | Financial sentiment analysis                |
| **datasets (FinGPT)**      | Training data for contextual AI predictions |
| **NewsAPI**                | Recent news headlines                       |

---

## 🧪 How It Works

1. **User Inputs a Ticker**

   * Example: `AAPL`, `TSLA`, `MSFT`

2. **App Displays Metadata & Charts**

   * Market cap, current price, 52-week high/low
   * Candlestick chart + moving averages

3. **Decision Tree Prediction**

   * Predicts if the next day’s price will go **up** or **down**
   * Accuracy score is displayed

4. **AI Forecast Report (Mistral-7B)**

   * Generates a 250+ word forecast using a prompt
   * Includes bullish/bearish indicators and technical justification

5. **News Sentiment (FinBERT)**

   * Classifies top 10 headlines into **Positive**, **Neutral**, or **Negative**

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stonk-ai.git
cd stonk-ai
```

### 2. Install Dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Download Mistral Model (if not already present)

Place the model file here:

```
models/mistral-7b-instruct-v0.1.Q4_0.gguf
```

You can download it from GPT4All's [official model releases](https://gpt4all.io/index.html).

---

## 🧠 Example Prompt (LLM Forecast)

```
You are a financial analyst writing a full stock forecast report for AAPL.

Include the following:
- A summary of the stock's recent trends.
- [Positive Developments] - at least 3 bullish indicators from data or news.
- [Potential Concerns] - at least 3 bearish indicators or risks.
- [Decision Tree Insight] - explain how the model predicted a price increase.
```

---

## 🛡️ Disclaimer

> This tool is for **educational and informational** purposes only. It does not constitute financial advice. Always do your own research before making investment decisions.

---

## 🧑‍💻 Author

Built with ❤️ by Michelle — bridging AI, finance, and user-friendly apps.
