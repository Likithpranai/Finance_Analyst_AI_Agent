# Finance Analyst ReAct Agent

A powerful AI agent for financial analysis that follows the ReAct pattern (Reason → Act → Observe → Loop) to provide comprehensive stock analysis.

## Features

- **Stock Price Analysis**: Get current and historical stock prices
- **Technical Indicators**: Calculate and interpret RSI, MACD, and other indicators
- **Company Information**: Access fundamental data about companies
- **Market News Tracking**: Get the latest news about stocks
- **Visualization**: Generate stock charts with technical indicators
- **Natural Language Analysis**: Powered by Google's Gemini AI

## How It Works

The Finance Analyst ReAct Agent follows the ReAct pattern:

1. **Reason**: Analyze the user's query to determine what information is needed
2. **Act**: Select and execute the appropriate financial analysis tools
3. **Observe**: Analyze the results from the tools
4. **Loop**: If needed, use additional tools based on initial observations

## Requirements

- Python 3.8+
- Google Gemini API key (set in `.env` file as `GEMINI_API_KEY`)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the agent:

```
python finance_analyst_agent.py
```

Example queries:
- "What's the current price of AAPL?"
- "Tell me about MSFT company information"
- "Is TSLA overbought or oversold based on RSI?"
- "Show me the MACD analysis for NVDA"
- "What are the latest news about AMZN?"
- "Create a visualization of GOOG with bollinger bands"
- "Give me a complete technical analysis of META"

## Tools

The agent has access to the following tools:

1. `get_stock_price(symbol)`: Get the current price of a stock
2. `get_stock_history(symbol, period)`: Get historical data for a stock
3. `calculate_rsi(symbol, period)`: Calculate the RSI technical indicator
4. `calculate_macd(symbol)`: Calculate the MACD technical indicator
5. `get_company_info(symbol)`: Get company information
6. `get_stock_news(symbol, max_items)`: Get the latest news articles about a stock
7. `visualize_stock(symbol, period, indicators)`: Create a visualization of stock data with technical indicators

## Error Handling

The agent includes robust error handling with fallback mechanisms:
- If the Gemini AI is unavailable, it falls back to a simple structured analysis
- All tool functions include try-except blocks to handle API failures gracefully
