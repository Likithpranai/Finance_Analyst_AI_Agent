"""
Economic Indicators Tools for the Finance Analyst AI Agent.
Fetches data on GDP, inflation, unemployment, interest rates, and forex rates.
"""

import os
import requests
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain.tools import BaseTool
import matplotlib.pyplot as plt
import io
import base64

try:
    import fredapi
    from fredapi import Fred
except ImportError:
    fredapi = None

from config import FRED_API_KEY

class FREDEconomicDataTool(BaseTool):
    name = "fred_economic_data"
    description = """
    Fetches economic indicators from Federal Reserve Economic Data (FRED) database.
    Provides data on GDP, inflation, unemployment, interest rates, etc.
    
    Args:
        indicator: FRED series ID or common name (e.g., 'GDP', 'UNRATE', 'CPIAUCSL', 'T10Y2Y', 'DFF')
        start_date: Optional start date in format 'YYYY-MM-DD' (defaults to 1 year ago)
        end_date: Optional end date in format 'YYYY-MM-DD' (defaults to today)
        
    Returns:
        A dictionary with economic indicator data, description, and historical values.
    """
    
    # Mapping of common names to FRED series IDs
    INDICATOR_MAPPING = {
        "GDP": "GDP",  # Gross Domestic Product
        "RGDP": "GDPC1",  # Real Gross Domestic Product
        "INFLATION": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers
        "CORE_INFLATION": "CPILFESL",  # CPI Less Food and Energy
        "UNEMPLOYMENT": "UNRATE",  # Unemployment Rate
        "FED_FUNDS_RATE": "DFF",  # Federal Funds Effective Rate
        "YIELD_CURVE": "T10Y2Y",  # 10-Year Treasury Constant Maturity Minus 2-Year
        "RETAIL_SALES": "RSXFS",  # Retail Sales
        "INDUSTRIAL_PRODUCTION": "INDPRO",  # Industrial Production Index
        "HOUSING_STARTS": "HOUST",  # Housing Starts
        "CONSUMER_SENTIMENT": "UMCSENT",  # University of Michigan Consumer Sentiment
        "PCE": "PCE",  # Personal Consumption Expenditures
        "MONEY_SUPPLY": "M2SL",  # M2 Money Supply
    }
    
    def _run(self, indicator: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        if not FRED_API_KEY:
            return {"error": "FRED API key is not configured. Please set the FRED_API_KEY in your .env file."}
            
        if fredapi is None:
            return {"error": "fredapi package is not installed. Please install with 'pip install fredapi'."}
        
        try:
            # Get FRED series ID if common name is used
            series_id = self.INDICATOR_MAPPING.get(indicator.upper(), indicator)
            
            # Setup dates
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            if start_date is None:
                # Default to 1 year ago
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            # Initialize FRED API
            fred = Fred(api_key=FRED_API_KEY)
            
            # Fetch data
            series_data = fred.get_series(series_id, start_date, end_date)
            
            if series_data.empty:
                return {"error": f"No data found for indicator {indicator} (series ID: {series_id})"}
            
            # Get series information
            series_info = fred.get_series_info(series_id)
            
            # Format the data for return
            data_points = []
            for date, value in series_data.items():
                data_points.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "value": float(value) if not pd.isna(value) else None
                })
            
            # Calculate basic statistics
            latest_value = float(series_data.iloc[-1]) if not pd.isna(series_data.iloc[-1]) else None
            avg_value = float(series_data.mean()) if not series_data.empty else None
            min_value = float(series_data.min()) if not series_data.empty else None
            max_value = float(series_data.max()) if not series_data.empty else None
            
            # Calculate recent trend (last 3 data points)
            recent_data = series_data.tail(3)
            if len(recent_data) > 1 and not recent_data.isna().all():
                if recent_data.iloc[-1] > recent_data.iloc[0]:
                    trend = "increasing"
                elif recent_data.iloc[-1] < recent_data.iloc[0]:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "unknown"
                
            # Format result
            result = {
                "indicator_name": series_info.title,
                "indicator_id": series_id,
                "frequency": series_info.frequency,
                "units": series_info.units,
                "latest_value": latest_value,
                "latest_date": series_data.index[-1].strftime("%Y-%m-%d") if not series_data.empty else None,
                "average_value": avg_value,
                "min_value": min_value,
                "max_value": max_value,
                "recent_trend": trend,
                "data_points": data_points,
                "notes": series_info.notes,
                "source": "Federal Reserve Economic Data (FRED)"
            }
            
            # Add economic context based on the indicator
            context = self._get_economic_context(indicator.upper(), latest_value, trend)
            if context:
                result["economic_context"] = context
                
            return result
            
        except Exception as e:
            return {"error": f"Error fetching economic data for {indicator}: {str(e)}"}
            
    def _get_economic_context(self, indicator: str, value: float, trend: str) -> str:
        """Provides economic context for the indicator values."""
        if indicator == "GDP" or indicator == "RGDP":
            if trend == "increasing":
                return "GDP growth indicates economic expansion. Sustained growth above 2% suggests a healthy economy."
            elif trend == "decreasing":
                if value < 0:
                    return "Negative GDP growth for two consecutive quarters technically defines a recession."
                else:
                    return "Slowing GDP growth may indicate economic deceleration, which could affect corporate earnings."
            else:
                return "Stable GDP suggests steady economic conditions."
                
        elif indicator == "INFLATION" or indicator == "CPIAUCSL":
            if value is None:
                return None
                
            if value > 4:
                return "Inflation significantly above the Fed's 2% target may prompt interest rate increases."
            elif 1.5 <= value <= 3:
                return "Inflation near the Federal Reserve's 2% target indicates a balanced economy."
            else:
                return "Low inflation may indicate weak demand or economic slack."
                
        elif indicator == "UNEMPLOYMENT" or indicator == "UNRATE":
            if value is None:
                return None
                
            if value < 4:
                return "Unemployment below 4% is typically considered full employment and may lead to wage pressures."
            elif 4 <= value <= 6:
                return "Moderate unemployment indicates a reasonably healthy labor market."
            else:
                return "High unemployment suggests economic weakness and reduced consumer spending power."
                
        elif indicator == "FED_FUNDS_RATE" or indicator == "DFF":
            if trend == "increasing":
                return "Rising interest rates typically aim to control inflation, but can slow economic growth and pressure stock valuations."
            elif trend == "decreasing":
                return "Falling interest rates typically stimulate economic growth and can boost stock valuations."
            else:
                return "Stable interest rates suggest the Federal Reserve sees balanced economic conditions."
                
        elif indicator == "YIELD_CURVE" or indicator == "T10Y2Y":
            if value is None:
                return None
                
            if value < 0:
                return "A negative yield curve (inverted) has historically preceded economic recessions."
            elif 0 <= value <= 1:
                return "A flat yield curve may suggest slowing economic growth expectations."
            else:
                return "A steep positive yield curve typically indicates expectations of economic growth."
                
        return None
    
    async def _arun(self, indicator: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(indicator, start_date, end_date)


class ForexDataTool(BaseTool):
    name = "forex_data"
    description = """
    Fetches foreign exchange rate data for currency pairs.
    Provides current and historical exchange rates.
    
    Args:
        from_currency: Base currency code (e.g., 'USD', 'EUR')
        to_currency: Quote currency code (e.g., 'JPY', 'GBP')
        time_period: Optional time period for historical data ('daily', 'weekly', 'monthly')
        
    Returns:
        A dictionary with exchange rate data and historical values.
    """
    
    def _run(self, from_currency: str, to_currency: str, time_period: str = "daily") -> Dict[str, Any]:
        try:
            # Use Alpha Vantage for forex data
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()
            
            # Import here to avoid circular imports
            from tools.alpha_vantage import AlphaVantageForexTool
            
            # Create and call the Alpha Vantage forex tool
            forex_tool = AlphaVantageForexTool()
            result = forex_tool._run(from_currency=from_currency, to_currency=to_currency, interval=time_period)
            
            # Add additional context about the currency pair
            if "error" not in result:
                result["context"] = self._get_currency_context(from_currency, to_currency)
                
            return result
            
        except Exception as e:
            return {"error": f"Error fetching forex data for {from_currency}/{to_currency}: {str(e)}"}
            
    def _get_currency_context(self, base: str, quote: str) -> str:
        """Provides context about currency pairs."""
        currency_contexts = {
            "USD": "US Dollar - The world's primary reserve currency",
            "EUR": "Euro - The official currency of 19 of the 27 member states of the EU",
            "GBP": "British Pound Sterling - The oldest currency still in use",
            "JPY": "Japanese Yen - The third most traded currency",
            "CHF": "Swiss Franc - Known as a safe-haven currency",
            "CAD": "Canadian Dollar - Strongly influenced by commodity prices",
            "AUD": "Australian Dollar - Commodity currency tied to exports",
            "NZD": "New Zealand Dollar - Agricultural export-dependent currency",
            "CNY": "Chinese Yuan - The currency of the world's second-largest economy"
        }
        
        pair_contexts = {
            "EURUSD": "The most traded currency pair, often reflecting global economic sentiment",
            "USDJPY": "Sensitive to interest rate differentials and risk sentiment",
            "GBPUSD": "Known as 'Cable', sensitive to UK economic data and Brexit developments",
            "USDCHF": "Affected by safe-haven flows during market uncertainty",
            "AUDUSD": "Known as the 'Aussie', sensitive to commodity prices and China's economy",
            "USDCAD": "Known as 'Loonie', closely tied to oil prices and US-Canada trade",
            "EURJPY": "A proxy for global risk appetite",
            "EURGBP": "Reflects European and UK economic divergence"
        }
        
        pair = f"{base}{quote}"
        context = []
        
        if base in currency_contexts:
            context.append(f"Base currency: {currency_contexts[base]}")
        
        if quote in currency_contexts:
            context.append(f"Quote currency: {currency_contexts[quote]}")
        
        if pair in pair_contexts:
            context.append(f"Pair characteristics: {pair_contexts[pair]}")
        elif pair[::-1] in pair_contexts:  # Check reversed pair
            context.append(f"Pair characteristics: {pair_contexts[pair[::-1]]} (inverse relationship)")
        
        return " ".join(context) if context else "No specific context available for this currency pair."
    
    async def _arun(self, from_currency: str, to_currency: str, time_period: str = "daily") -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(from_currency, to_currency, time_period)


class GlobalMarketIndicesTool(BaseTool):
    name = "global_market_indices"
    description = """
    Fetches data on global market indices like S&P 500, NASDAQ, FTSE, Nikkei, etc.
    Provides current levels, historical performance, and comparative analysis.
    
    Args:
        index: Index symbol or name (e.g., '^GSPC' for S&P 500, '^IXIC' for NASDAQ)
        period: Optional time period (e.g., '1d', '1mo', '1y', '5y')
        
    Returns:
        A dictionary with index data, performance metrics, and context.
    """
    
    # Mapping of common names to index symbols
    INDEX_MAPPING = {
        "S&P500": "^GSPC",
        "SP500": "^GSPC",
        "DOW": "^DJI",
        "NASDAQ": "^IXIC",
        "RUSSELL2000": "^RUT",
        "VIX": "^VIX",
        "FTSE": "^FTSE",  # UK
        "DAX": "^GDAXI",  # Germany
        "CAC40": "^FCHI",  # France
        "NIKKEI": "^N225",  # Japan
        "HANGSENG": "^HSI",  # Hong Kong
        "SHANGHAI": "^SSEC",  # China
        "ASX": "^AXJO",  # Australia
        "TSX": "^GSPTSE",  # Canada
    }
    
    def _run(self, index: str, period: str = "1y") -> Dict[str, Any]:
        try:
            # Map common names to symbols
            index_symbol = self.INDEX_MAPPING.get(index.upper(), index)
            
            # Use yfinance to get index data
            import yfinance as yf
            
            idx = yf.Ticker(index_symbol)
            info = idx.info
            hist = idx.history(period=period)
            
            if hist.empty:
                return {"error": f"No data found for index {index} (symbol: {index_symbol})"}
            
            # Calculate performance metrics
            latest_close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else None
            day_change = ((latest_close / prev_close) - 1) * 100 if prev_close else None
            
            start_price = hist['Close'].iloc[0]
            period_change = ((latest_close / start_price) - 1) * 100
            
            # Calculate other metrics
            high_52wk = hist['High'].max() if len(hist) >= 252 else hist['High'].max()
            low_52wk = hist['Low'].min() if len(hist) >= 252 else hist['Low'].min()
            
            # Volatility (annualized)
            daily_returns = hist['Close'].pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized and in percentage
            
            # Format the data for return
            result = {
                "index_name": info.get('shortName', index),
                "index_symbol": index_symbol,
                "current_level": round(latest_close, 2),
                "previous_close": round(prev_close, 2) if prev_close else None,
                "day_change_percent": round(day_change, 2) if day_change else None,
                "period_change_percent": round(period_change, 2),
                "period": period,
                "52wk_high": round(high_52wk, 2),
                "52wk_low": round(low_52wk, 2),
                "volatility_percent": round(volatility, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "Yahoo Finance"
            }
            
            # Add market context
            result["market_context"] = self._get_market_context(index_symbol, result)
            
            return result
            
        except Exception as e:
            return {"error": f"Error fetching data for index {index}: {str(e)}"}
            
    def _get_market_context(self, index_symbol: str, data: dict) -> str:
        """Provides context about the market index."""
        context = ""
        
        # Add description based on the index
        if index_symbol == "^GSPC":  # S&P 500
            context = "The S&P 500 is a stock market index tracking the stock performance of 500 large companies listed on US exchanges. It is considered the best representation of the US stock market and the American economy."
        elif index_symbol == "^DJI":  # Dow Jones
            context = "The Dow Jones Industrial Average is a price-weighted average of 30 significant stocks traded on the NYSE and Nasdaq. It is one of the oldest and most-watched indices in the world."
        elif index_symbol == "^IXIC":  # NASDAQ
            context = "The NASDAQ Composite Index includes all Nasdaq domestic and international based common stocks. It is heavily weighted towards technology companies."
        elif index_symbol == "^VIX":  # VIX
            context = "The VIX (CBOE Volatility Index) measures market expectations of near-term volatility. Often called the 'fear index', high values indicate increased market uncertainty."
        else:
            # Generic context for other indices
            name = data.get("index_name", "This index")
            context = f"{name} is a stock market index tracking the performance of selected stocks in its respective market."
            
        # Add performance context
        day_change = data.get("day_change_percent")
        if day_change:
            if day_change > 1:
                context += f" It has shown significant strength today (up {round(day_change, 2)}%)."
            elif day_change > 0:
                context += f" It is slightly up today (up {round(day_change, 2)}%)."
            elif day_change > -1:
                context += f" It is slightly down today (down {round(abs(day_change), 2)}%)."
            else:
                context += f" It has shown significant weakness today (down {round(abs(day_change), 2)}%)."
                
        # Add period performance
        period_change = data.get("period_change_percent")
        period = data.get("period", "the selected period")
        if period_change:
            if period_change > 20:
                context += f" It has performed very strongly over {period} (up {round(period_change, 2)}%)."
            elif period_change > 0:
                context += f" It has performed positively over {period} (up {round(period_change, 2)}%)."
            elif period_change > -20:
                context += f" It has underperformed over {period} (down {round(abs(period_change), 2)}%)."
            else:
                context += f" It has performed very poorly over {period} (down {round(abs(period_change), 2)}%)."
                
        return context
    
    async def _arun(self, index: str, period: str = "1y") -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(index, period)
