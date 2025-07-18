"""
Settings configuration for the Finance Analyst AI Agent.
"""

import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Model Configuration
GEMINI_MODEL_NAMES = ['gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro']
DEFAULT_GEMINI_MODEL = GEMINI_MODEL_NAMES[0]

# Data Sources
DEFAULT_DATA_SOURCE = "yfinance"  # Options: "yfinance", "alpha_vantage", "polygon"
FALLBACK_DATA_SOURCES = ["yfinance", "alpha_vantage"]

# Cache Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
CACHE_EXPIRATION = {
    "stock_price": 300,  # 5 minutes
    "stock_history": 3600,  # 1 hour
    "company_info": 86400,  # 1 day
    "financial_ratios": 86400,  # 1 day
    "technical_indicators": 1800,  # 30 minutes
}
USE_REDIS_CACHE = os.getenv("USE_REDIS_CACHE", "true").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "logs/finance_agent.log")

# Technical Analysis Configuration
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_FAST_PERIOD = 12
DEFAULT_MACD_SLOW_PERIOD = 26
DEFAULT_MACD_SIGNAL_PERIOD = 9
DEFAULT_ADX_WINDOW = 14
DEFAULT_BOLLINGER_WINDOW = 20
DEFAULT_BOLLINGER_STD = 2

# Visualization Configuration
DEFAULT_CHART_STYLE = "dark_background"
DEFAULT_CHART_FIGSIZE = (12, 8)
CHART_COLORS = {
    "price": "#2962FF",
    "volume": "#2E7D32",
    "sma50": "#FF6D00",
    "sma200": "#6200EA",
    "upper_band": "#00BFA5",
    "lower_band": "#00BFA5",
    "rsi": "#FFC107",
    "macd": "#E91E63",
    "signal": "#9C27B0",
    "histogram_positive": "#00C853",
    "histogram_negative": "#D50000",
}

# Portfolio Management Configuration
DEFAULT_RISK_FREE_RATE = 0.02  # 2%
DEFAULT_PORTFOLIO_OPTIMIZATION_METHOD = "max_sharpe"  # Options: "max_sharpe", "min_volatility", "efficient_risk", "efficient_return"

# Backtesting Configuration
DEFAULT_INITIAL_CAPITAL = 10000
DEFAULT_COMMISSION = 0.001  # 0.1%

# Output Configuration
DEFAULT_OUTPUT_DIR = "outputs"
SAVE_OUTPUTS = True

# Dashboard Configuration
DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8501
DASHBOARD_DEBUG = False

# System Configuration
MAX_THREADS = 4
REQUEST_TIMEOUT = 30  # seconds

# Feature Flags
ENABLE_REAL_TIME_DATA = True
ENABLE_PREDICTIVE_ANALYTICS = True
ENABLE_PORTFOLIO_MANAGEMENT = True
ENABLE_BACKTESTING = True
ENABLE_SENTIMENT_ANALYSIS = False  # Disabled by default as it requires additional dependencies

def get_log_level():
    """Get the log level from the environment or default to INFO."""
    level = LOG_LEVEL
    if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        return getattr(logging, level)
    return logging.INFO
