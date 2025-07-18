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

## Architecture

### Agent Framework
The enhanced system uses a modular multi-agent architecture:

1. **Base Agent**: Core functionality for all specialized agents
   - Gemini AI model initialization
   - Tool registration and execution
   - Response generation

2. **Specialized Agents**:
   - **Technical Analysis Agent**: Chart patterns, indicators, price data
   - **Fundamental Analysis Agent**: Financial statements, ratios, company profiles
   - **Risk Analysis Agent**: Volatility, VaR, portfolio risk
   - **Trading Agent**: Trading signals, backtesting, execution
   - **Sentiment Analysis Agent**: News sentiment, social media analysis, market perception

3. **Agent Orchestrator**: Coordinates the specialized agents
   - Query classification
   - Agent dispatching
   - Response synthesis

4. **Agent Memory**: Persistent memory system
   - Conversation history
   - Analysis results
   - User preferences
   - Watched symbols
   - Alerts

### Integration Modules
- **LangChain Integration**: Advanced agent capabilities
- **CrewAI Integration**: Multi-agent collaboration workflows

## Tools

### Technical Analysis Tools
- Historical price data retrieval
- Technical indicator calculation (RSI, MACD, Bollinger Bands, etc.)
- Chart pattern recognition
- Support/resistance identification
- Trend analysis
- Technical visualization

### Fundamental Analysis Tools
- Financial statement retrieval and analysis
- Key financial ratio calculation
- Company profile information
- Industry comparison
- Valuation metrics
- Growth analysis

### Risk Analysis Tools
- Volatility calculation
- Value at Risk (VaR)

### Sentiment Analysis Tools
- Financial news sentiment analysis
- Social media sentiment tracking
- Sentiment trend visualization
- News article aggregation
- Multiple NLP models (FinBERT, RoBERTa, VADER)
- Sentiment impact analysis
- Risk-adjusted returns (Sharpe, Sortino)
- Beta and correlation analysis
- Portfolio risk assessment
- Diversification metrics

### Trading Tools
- Trading signal generation
- Strategy backtesting
- Order execution simulation
- Position management
- Risk management rules

## Error Handling

The system includes robust error handling with fallback mechanisms:
- Gemini AI model fallback options
- Redis unavailability handled with in-memory storage
- API failures handled gracefully with informative messages
- Comprehensive logging for debugging
