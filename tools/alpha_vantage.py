"""
Tools for fetching stock data from Alpha Vantage API
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from langchain.tools import BaseTool
from typing import Optional, Dict, Any, List

from config import ALPHA_VANTAGE_API_KEY

class AlphaVantageStockTool(BaseTool):
    """Tool for getting stock data from Alpha Vantage"""
    
    name: str = "alpha_vantage_stock_data"
    description: str = """Gets stock data from Alpha Vantage API. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - function (optional): The Alpha Vantage function to use (default: TIME_SERIES_DAILY)
    - outputsize (optional): 'compact' for last 100 data points, 'full' for full history
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            function = kwargs.get("function", "TIME_SERIES_DAILY")
            outputsize = kwargs.get("outputsize", "compact")
            
            if not ticker:
                return "Error: No ticker symbol provided"
                
            if not ALPHA_VANTAGE_API_KEY:
                return "Error: Alpha Vantage API key not configured"
                
            # Construct the API URL
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": ticker,
                "outputsize": outputsize,
                "apikey": ALPHA_VANTAGE_API_KEY,
                "datatype": "json"
            }
            
            # Make the API request
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Check for errors
            if "Error Message" in data:
                return f"API Error: {data['Error Message']}"
                
            if "Information" in data:
                return f"API Information: {data['Information']}"
                
            # Format the response based on the function type
            formatted_data = {
                "source": "Alpha Vantage",
                "symbol": ticker,
                "function": function,
            }
            
            # Process different function types differently
            if function in ["TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY", "TIME_SERIES_WEEKLY", "TIME_SERIES_MONTHLY"]:
                # Determine the time series key
                time_series_key = None
                for key in data.keys():
                    if "Time Series" in key:
                        time_series_key = key
                        break
                
                if not time_series_key:
                    return "Error: Unexpected API response format"
                
                # Convert time series data to a more accessible format
                time_series = data[time_series_key]
                prices = []
                
                for date, values in time_series.items():
                    price_data = {
                        "date": date,
                        "open": float(values.get("1. open", 0)),
                        "high": float(values.get("2. high", 0)),
                        "low": float(values.get("3. low", 0)),
                        "close": float(values.get("4. close", 0)),
                        "volume": float(values.get("5. volume", 0)),
                    }
                    prices.append(price_data)
                
                # Sort by date
                prices.sort(key=lambda x: x["date"])
                
                # Add to formatted data
                formatted_data["prices"] = prices
                
                # Add summary statistics
                if prices:
                    closes = [price["close"] for price in prices]
                    formatted_data["summary"] = {
                        "min_close": min(closes),
                        "max_close": max(closes),
                        "avg_close": sum(closes) / len(closes),
                        "latest_close": closes[-1],
                        "start_date": prices[0]["date"],
                        "end_date": prices[-1]["date"],
                        "total_return": ((closes[-1] / closes[0]) - 1) * 100 if closes[0] > 0 else None
                    }
            
            elif function == "GLOBAL_QUOTE":
                # Handle global quote response
                quote = data.get("Global Quote", {})
                if quote:
                    formatted_data["quote"] = {
                        "symbol": quote.get("01. symbol"),
                        "price": float(quote.get("05. price", 0)),
                        "change": float(quote.get("09. change", 0)),
                        "change_percent": quote.get("10. change percent", "0%").rstrip("%"),
                        "volume": int(quote.get("06. volume", 0)),
                        "latest_trading_day": quote.get("07. latest trading day")
                    }
                
            elif function == "OVERVIEW":
                # Handle company overview
                formatted_data["company_info"] = data
                
            else:
                # For other functions, just return the raw data
                formatted_data["data"] = data
                
            return formatted_data
            
        except Exception as e:
            return f"Error fetching data from Alpha Vantage: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class AlphaVantageForexTool(BaseTool):
    """Tool for getting forex data from Alpha Vantage"""
    
    name: str = "alpha_vantage_forex_data"
    description: str = """Gets forex (currency exchange) data from Alpha Vantage API. 
    Input should be a JSON object with:
    - from_currency (required): The currency to convert from (e.g., USD)
    - to_currency (required): The currency to convert to (e.g., EUR)
    - function (optional): The Alpha Vantage function to use (default: CURRENCY_EXCHANGE_RATE for current rate or FX_DAILY for historical)
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            from_currency = kwargs.get("from_currency")
            to_currency = kwargs.get("to_currency")
            function = kwargs.get("function", "CURRENCY_EXCHANGE_RATE")
            
            if not from_currency or not to_currency:
                return "Error: Both from_currency and to_currency must be provided"
                
            if not ALPHA_VANTAGE_API_KEY:
                return "Error: Alpha Vantage API key not configured"
                
            # Construct the API URL
            base_url = "https://www.alphavantage.co/query"
            
            if function == "CURRENCY_EXCHANGE_RATE":
                # Real-time exchange rate
                params = {
                    "function": function,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "apikey": ALPHA_VANTAGE_API_KEY
                }
            else:
                # Historical forex data
                params = {
                    "function": function,
                    "from_symbol": from_currency,
                    "to_symbol": to_currency,
                    "outputsize": "compact",
                    "apikey": ALPHA_VANTAGE_API_KEY
                }
            
            # Make the API request
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Check for errors
            if "Error Message" in data:
                return f"API Error: {data['Error Message']}"
                
            if "Information" in data:
                return f"API Information: {data['Information']}"
                
            # Format the response based on the function
            formatted_data = {
                "source": "Alpha Vantage",
                "function": function,
                "from_currency": from_currency,
                "to_currency": to_currency
            }
            
            if function == "CURRENCY_EXCHANGE_RATE":
                exchange_data = data.get("Realtime Currency Exchange Rate", {})
                if exchange_data:
                    formatted_data["exchange_rate"] = {
                        "from": exchange_data.get("1. From_Currency Code"),
                        "to": exchange_data.get("3. To_Currency Code"),
                        "rate": float(exchange_data.get("5. Exchange Rate", 0)),
                        "timestamp": exchange_data.get("6. Last Refreshed"),
                        "timezone": exchange_data.get("7. Time Zone")
                    }
            else:
                # Process time series data for FX_DAILY, FX_WEEKLY, FX_MONTHLY
                time_series_key = None
                for key in data.keys():
                    if "Time Series" in key:
                        time_series_key = key
                        break
                
                if time_series_key:
                    time_series = data[time_series_key]
                    rates = []
                    
                    for date, values in time_series.items():
                        rate_data = {
                            "date": date,
                            "open": float(values.get("1. open", 0)),
                            "high": float(values.get("2. high", 0)),
                            "low": float(values.get("3. low", 0)),
                            "close": float(values.get("4. close", 0))
                        }
                        rates.append(rate_data)
                    
                    # Sort by date
                    rates.sort(key=lambda x: x["date"])
                    
                    # Add to formatted data
                    formatted_data["rates"] = rates
                    
                    # Add summary
                    if rates:
                        closes = [rate["close"] for rate in rates]
                        formatted_data["summary"] = {
                            "min_rate": min(closes),
                            "max_rate": max(closes),
                            "avg_rate": sum(closes) / len(closes),
                            "latest_rate": closes[-1],
                            "start_date": rates[0]["date"],
                            "end_date": rates[-1]["date"],
                            "change_percent": ((closes[-1] / closes[0]) - 1) * 100 if closes[0] > 0 else None
                        }
                else:
                    # For other formats, return the raw data
                    formatted_data["data"] = data
                
            return formatted_data
            
        except Exception as e:
            return f"Error fetching forex data from Alpha Vantage: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)
