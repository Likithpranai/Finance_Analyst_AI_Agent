"""
Configuration settings for the Finance Analyst AI Agent
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Model settings
GEMINI_MODEL = "gemini-pro"

# Tool settings
DEFAULT_STOCK_HISTORY_PERIOD = "1y"
DEFAULT_STOCK_HISTORY_INTERVAL = "1d"
MAX_NEWS_RESULTS = 5

# Data source settings
DATA_SOURCES = [
    "yahoo_finance", 
    "alpha_vantage", 
    "financial_modeling_prep", 
    "fred_economic", 
    "news_api"
]
PRIMARY_DATA_SOURCE = "yahoo_finance"  # Default source

# Risk management settings
DEFAULT_RISK_FREE_RATE = 0.03  # 3% annual risk-free rate

# Forecasting settings
DEFAULT_FORECAST_PERIOD = 30  # Default forecast days
DEFAULT_FORECAST_MODEL = "arima"  # Default forecasting model

# Agent settings
VERBOSE = True
TEMPERATURE = 0.2
