# Finance Analyst ReAct Agent

A professional-grade AI agent for comprehensive financial analysis that follows the ReAct pattern (Reason → Act → Observe → Loop). This intelligent system processes natural language queries to provide in-depth analysis of stocks, cryptocurrencies, and forex markets with explanations, visualizations, and actionable insights.

## Features

### Multi-Asset Analysis
- **Stocks**: Technical analysis, fundamental data, price trends, and market positioning
- **Cryptocurrencies**: Price data, market analysis, and trend identification
- **Forex**: Exchange rate data, trend analysis, and market insights

### Technical Analysis
- **Core Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Volume Indicators**: On-Balance Volume (OBV), Accumulation/Distribution Line
- **Trend Strength**: Average Directional Index (ADX)
- **Pattern Recognition**: Support/resistance levels, chart patterns

### Fundamental Analysis
- **Financial Statements**: Income statements, balance sheets, cash flow statements
- **Key Ratios**: P/E, P/B, P/S, EPS, ROE, ROA, debt-to-equity
- **Industry Comparisons**: Benchmark against sector peers
- **Valuation Metrics**: Intrinsic value estimation, growth projections

### Advanced Capabilities
- **Predictive Analytics**: Time series forecasting, anomaly detection
- **Portfolio Management**: Risk metrics, optimization, efficient frontier
- **Backtesting**: Strategy testing, performance evaluation
- **Visualization**: Interactive charts, financial trends, correlation matrices
- **Real-Time Data**: Low-latency market data (when available)

### AI-Powered Analysis
- **Natural Language Processing**: Powered by Google's Gemini AI
- **ReAct Pattern**: Reason → Act → Observe → Loop methodology
- **Contextual Understanding**: Interprets complex financial queries
- **Professional Formatting**: Clear summaries, detailed analysis, and actionable recommendations

## How It Works

The Finance Analyst ReAct Agent implements the ReAct pattern (Reason → Act → Observe → Loop), a powerful AI reasoning framework that combines reasoning and acting in an iterative process:

1. **Reason**: The agent analyzes the user's query to understand intent, identify the financial asset (stock/crypto/forex), and determine which tools and data sources are needed.

2. **Act**: Based on its reasoning, the agent executes the appropriate financial analysis tools, fetching real-time or historical data, calculating indicators, or retrieving fundamental information.

3. **Observe**: The agent analyzes the results from the executed tools, interpreting technical indicators, identifying patterns, and extracting meaningful insights.

4. **Loop**: If the initial analysis is insufficient, the agent selects additional tools to gather more information, creating a comprehensive analysis through multiple iterations.

This process is powered by Google's Gemini AI, which provides the reasoning capabilities to interpret financial data and generate professional-grade analysis with actionable insights.

## Requirements

- **Python**: 3.8 or higher
- **API Keys**:
  - Google Gemini API key (required)
  - Alpha Vantage API key (optional, for enhanced data)
  - Polygon.io API key (optional, for real-time data)
- **System Requirements**:
  - 4GB+ RAM recommended
  - Internet connection for real-time data
- **Required Python packages**: See `requirements.txt`

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Finance_Analyst_AI_Agent.git
   cd Finance_Analyst_AI_Agent
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here  # Optional
   POLYGON_API_KEY=your_polygon_key_here  # Optional
   ```

## Usage

### Command Line Interface

The Finance Analyst AI Agent supports multiple usage modes:

1. **Interactive Mode** (default):
   ```bash
   python finance_agent.py
   ```
   This starts an interactive session where you can ask multiple questions.

2. **Single Query Mode**:
   ```bash
   python finance_agent.py --query "Calculate the RSI for AAPL and tell me if it's overbought"
   ```
   Process a single query and exit.

### Example Queries

#### Basic Analysis
- "What's the current price of AAPL?"
- "Show me TSLA's performance over the last 6 months"
- "Compare MSFT and GOOGL stock prices"

#### Technical Analysis
- "Calculate the RSI for NVDA and tell me if it's overbought or oversold"
- "What's the MACD saying about AMD's trend?"
- "Calculate and explain the On-Balance Volume for AMZN"
- "Show me the ADX for META to understand trend strength"
- "Calculate the Accumulation/Distribution Line for TSLA"

#### Fundamental Analysis
- "What are AAPL's key financial ratios?"
- "Show me MSFT's income statement"
- "Compare NVDA's P/E ratio to the semiconductor industry average"

#### Comprehensive Analysis
- "Give me a complete analysis of AAPL including technical indicators and fundamentals"
- "Should I invest in TSLA based on current technical and fundamental factors?"
- "Analyze Bitcoin's recent price action and predict future movement"

#### Portfolio Analysis
- "Optimize a portfolio of AAPL, MSFT, and GOOGL for maximum Sharpe ratio"
- "Calculate the risk metrics for my portfolio of tech stocks"
- "Backtest an RSI strategy on AAPL for the past year"

## Architecture

### Core Components

#### ReAct Agent Framework
The Finance Analyst AI Agent is built on the ReAct (Reason-Act-Observe-Loop) pattern, which enables sophisticated reasoning and tool use:

1. **FinanceAnalystReActAgent**: The main agent class that implements the ReAct pattern
   - Manages tool selection and execution
   - Processes natural language queries
   - Generates structured financial analysis
   - Handles error recovery and fallback mechanisms

2. **Tool Integration System**:
   - Dynamic tool registration and discovery
   - Tool execution with proper error handling
   - Result formatting and interpretation
   - Tool chaining for complex analyses

3. **Reasoning Engine**:
   - Powered by Google's Gemini AI
   - Context-aware query analysis
   - Multi-step reasoning for complex financial questions
   - Structured output generation

### Data Processing Pipeline

1. **Data Acquisition Layer**:
   - YFinance integration for historical data
   - Alpha Vantage integration for fundamental data
   - Polygon.io for real-time professional-grade data
   - WebSocket support for streaming data

2. **Data Processing Layer**:
   - Technical indicator calculation
   - Statistical analysis
   - Time series processing
   - Data cleaning and normalization

3. **Analysis Layer**:
   - Pattern recognition
   - Trend identification
   - Signal generation
   - Risk assessment

4. **Visualization Layer**:
   - Interactive charts
   - Technical overlays
   - Comparative visualizations
   - Portfolio dashboards

### Optimization Features

1. **Caching System**:
   - Redis-based caching (when available)
   - In-memory fallback cache
   - Intelligent cache invalidation
   - Request deduplication

2. **Parallel Processing**:
   - Concurrent tool execution
   - Asynchronous data fetching
   - Batch processing for multiple symbols

## Project Structure

```
Finance_Analyst_AI_Agent/
├── finance_agent.py           # Main entry point
├── finance_analyst_agent.py   # Core ReAct agent implementation
├── tools/                     # Tool implementations
│   ├── technical_analysis.py  # Technical indicators and analysis
│   ├── fundamental_analysis.py # Fundamental data and ratios
│   ├── predictive_analytics.py # Forecasting and predictions
│   ├── enhanced_visualization.py # Chart generation
│   ├── combined_analysis.py   # Integrated analysis tools
│   ├── portfolio_management.py # Portfolio optimization
│   ├── backtesting.py         # Strategy testing
│   ├── alpha_vantage_tools.py # Alpha Vantage API integration
│   ├── polygon_integration.py # Polygon.io API integration
│   ├── real_time_data_integration.py # Real-time data handling
│   └── cache_manager.py       # Data caching system
├── agent_framework/           # Agent architecture components
│   ├── base_agent.py          # Base agent class
│   ├── chains/                # Reasoning chains
│   └── memory/                # Agent memory system
├── models/                    # ML model definitions
├── data/                      # Data storage
├── tests/                     # Unit and integration tests
└── utils/                     # Utility functions
```

## Contributing

Contributions to the Finance Analyst AI Agent are welcome! Here's how you can contribute:

1. **Adding New Indicators**: Implement new technical or fundamental indicators in the appropriate tools module.

2. **Enhancing Visualization**: Improve chart generation and visualization capabilities.

3. **Optimizing Performance**: Enhance data fetching, caching, or processing efficiency.

4. **Adding Data Sources**: Integrate additional financial data providers.

5. **Improving Documentation**: Enhance code comments, docstrings, and user guides.

6. **Testing**: Add unit tests, integration tests, or test new features.

Please follow these steps for contributions:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## Future Development

The Finance Analyst AI Agent is continuously evolving. Planned enhancements include:

1. **Advanced ML Models**:
   - Sentiment analysis for news and social media
   - Deep learning for price prediction
   - Reinforcement learning for trading strategies

2. **Enhanced User Interfaces**:
   - Web dashboard for interactive analysis
   - Mobile app for on-the-go insights
   - Voice interface for hands-free operation

3. **Expanded Asset Coverage**:
   - Options and derivatives analysis
   - Commodities and futures
   - Fixed income securities
   - Alternative investments

4. **Enterprise Features**:
   - Multi-user support
   - Team collaboration tools
   - Custom report generation
   - Compliance and audit trails

5. **Integration Capabilities**:
   - Trading platform connections
   - CRM system integration
   - Data export to spreadsheets and databases
   - API for third-party applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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
