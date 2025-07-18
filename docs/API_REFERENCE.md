# Finance Analyst AI Agent API Reference

This document provides a comprehensive reference for the Finance Analyst AI Agent API.

## Main Agent

### `FinanceAnalystReActAgent`

The main agent class that implements the ReAct pattern for financial analysis.

```python
from finance_analyst_agent import FinanceAnalystReActAgent

# Initialize the agent
agent = FinanceAnalystReActAgent()

# Process a query
response = agent.process_query("What is the current RSI for AAPL?")
print(response)
```

#### Methods

- `process_query(query: str) -> dict`: Process a natural language query and return analysis results
- `determine_tools_needed(query: str) -> list`: Determine which tools are needed for a given query
- `extract_stock_symbol(query: str) -> str`: Extract stock symbol from a query
- `detect_asset_type(query: str) -> str`: Detect the type of asset (stock, crypto, forex)

## Tools

### Technical Analysis

```python
from tools.technical_analysis import TechnicalAnalysisTools

# Initialize the tools
ta_tools = TechnicalAnalysisTools()

# Calculate RSI
import yfinance as yf
data = yf.download("AAPL", period="1y")
rsi_result = ta_tools.calculate_rsi(data)
```

#### Available Technical Indicators

- `calculate_rsi(data, period=14)`: Calculate Relative Strength Index
- `calculate_macd(data)`: Calculate Moving Average Convergence Divergence
- `calculate_bollinger_bands(data, window=20, num_std=2)`: Calculate Bollinger Bands
- `calculate_sma(data, window=50)`: Calculate Simple Moving Average
- `calculate_ema(data, window=50)`: Calculate Exponential Moving Average
- `calculate_obv(data)`: Calculate On-Balance Volume
- `calculate_adline(data)`: Calculate Accumulation/Distribution Line
- `calculate_adx(data, window=14)`: Calculate Average Directional Index

### Fundamental Analysis

```python
from tools.fundamental_analysis import FundamentalAnalysisTools

# Initialize the tools
fa_tools = FundamentalAnalysisTools()

# Get financial ratios
ratios = fa_tools.get_financial_ratios("AAPL")
```

#### Available Fundamental Analysis Methods

- `get_financial_ratios(symbol)`: Get key financial ratios
- `get_income_statement(symbol)`: Get income statement data
- `get_balance_sheet(symbol)`: Get balance sheet data
- `get_cash_flow(symbol)`: Get cash flow statement data
- `get_company_info(symbol)`: Get company information and profile
- `compare_with_industry(symbol, metrics)`: Compare company metrics with industry averages

### Portfolio Management

```python
from tools.portfolio_management import PortfolioTools

# Initialize the tools
portfolio_tools = PortfolioTools()

# Optimize portfolio
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
weights = portfolio_tools.optimize_portfolio(tickers)
```

#### Available Portfolio Methods

- `optimize_portfolio(tickers, method="max_sharpe")`: Optimize portfolio weights
- `calculate_portfolio_performance(tickers, weights)`: Calculate historical portfolio performance
- `calculate_portfolio_risk(tickers, weights)`: Calculate portfolio risk metrics
- `rebalance_portfolio(current_portfolio, target_weights)`: Generate rebalancing recommendations

### Visualization

```python
from tools.visualization import VisualizationTools

# Initialize the tools
viz_tools = VisualizationTools()

# Create stock chart
import yfinance as yf
data = yf.download("AAPL", period="1y")
chart_path = viz_tools.create_stock_chart(data, "AAPL", indicators=["sma50", "rsi"])
```

#### Available Visualization Methods

- `create_stock_chart(data, symbol, indicators=None)`: Create a stock price chart with indicators
- `create_comparison_chart(symbols, period="1y")`: Create a comparison chart for multiple stocks
- `create_correlation_heatmap(symbols, period="1y")`: Create a correlation heatmap
- `create_financial_ratio_chart(symbol)`: Create a chart of financial ratios

### Data Sources

#### YFinance

```python
from tools.stock_data import StockTools

# Initialize the tools
stock_tools = StockTools()

# Get stock data
data = stock_tools.get_stock_data("AAPL", period="1y")
```

#### Alpha Vantage

```python
from tools.alpha_vantage import AlphaVantageTools

# Initialize the tools
av_tools = AlphaVantageTools()

# Get real-time quote
quote = av_tools.get_quote("AAPL")
```

#### Polygon.io

```python
from tools.polygon import PolygonTools

# Initialize the tools
polygon_tools = PolygonTools()

# Get real-time quote
quote = polygon_tools.get_real_time_quote("AAPL")
```

## Unified Entry Point

### `finance_agent.py`

```python
# Command-line usage
python finance_agent.py --query "What is the RSI for AAPL?"

# Interactive mode
python finance_agent.py

# Dashboard mode
python finance_agent.py --dashboard
```

## Configuration

Configuration settings can be modified in `config/settings.py` or by setting environment variables in a `.env` file.

### Required Environment Variables

- `GEMINI_API_KEY`: Google Gemini API key (required)

### Optional Environment Variables

- `ALPHA_VANTAGE_API_KEY`: Alpha Vantage API key for real-time data
- `POLYGON_API_KEY`: Polygon.io API key for real-time data
- `USE_REDIS_CACHE`: Whether to use Redis for caching (true/false)
- `REDIS_HOST`: Redis host (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_TO_FILE`: Whether to log to file (true/false)

## Error Handling

The agent includes comprehensive error handling:

- API errors are caught and logged
- Data validation ensures proper input formats
- Fallback data sources are used when primary sources fail
- Graceful degradation when optional dependencies are missing
