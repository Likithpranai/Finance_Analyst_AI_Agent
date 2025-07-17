"""
Alpha Vantage integration for real-time financial data
Provides ultra-real-time data through Alpha Vantage API for the Finance Analyst AI Agent
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time
from typing import Dict, List, Any, Optional, Union

# Load environment variables
load_dotenv()

# Get Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")

class AlphaVantageTools:
    """Tools for accessing real-time financial data from Alpha Vantage API"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    @staticmethod
    def get_intraday_data(symbol: str, interval: str = "1min", output_size: str = "compact") -> Dict:
        """
        Get real-time intraday data for a stock with sub-second latency
        
        Args:
            symbol: Stock symbol
            interval: Time interval between data points (1min, 5min, 15min, 30min, 60min)
            output_size: compact (latest 100 data points) or full (up to 30 days of data)
            
        Returns:
            Dictionary with intraday data and metadata
        """
        try:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "outputsize": output_size,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            response = requests.get(AlphaVantageTools.BASE_URL, params=params)
            data = response.json()
            
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            
            if "Note" in data:
                return {"warning": data["Note"], "data": data}
                
            # Format the response for better readability
            time_series_key = f"Time Series ({interval})"
            if time_series_key in data:
                latest_timestamp = max(data[time_series_key].keys())
                latest_data = data[time_series_key][latest_timestamp]
                
                formatted_data = {
                    "symbol": symbol,
                    "last_refreshed": data["Meta Data"]["3. Last Refreshed"],
                    "interval": interval,
                    "latest_price": latest_data["4. close"],
                    "latest_timestamp": latest_timestamp,
                    "open": latest_data["1. open"],
                    "high": latest_data["2. high"],
                    "low": latest_data["3. low"],
                    "volume": latest_data["5. volume"],
                    "full_data": data
                }
                
                return formatted_data
            
            return data
            
        except Exception as e:
            return {"error": f"Error fetching intraday data: {str(e)}"}
    
    @staticmethod
    def get_real_time_quote(symbol: str) -> Dict:
        """
        Get real-time quote for a stock with minimal latency
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with real-time quote data
        """
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            response = requests.get(AlphaVantageTools.BASE_URL, params=params)
            data = response.json()
            
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            
            if "Note" in data:
                return {"warning": data["Note"], "data": data}
                
            if "Global Quote" in data and data["Global Quote"]:
                quote = data["Global Quote"]
                
                formatted_data = {
                    "symbol": quote["01. symbol"],
                    "price": quote["05. price"],
                    "change": quote["09. change"],
                    "change_percent": quote["10. change percent"],
                    "volume": quote["06. volume"],
                    "latest_trading_day": quote["07. latest trading day"],
                    "previous_close": quote["08. previous close"],
                    "open": quote["02. open"],
                    "high": quote["03. high"],
                    "low": quote["04. low"]
                }
                
                return formatted_data
            
            return {"error": "No quote data available"}
            
        except Exception as e:
            return {"error": f"Error fetching real-time quote: {str(e)}"}
    
    @staticmethod
    def get_crypto_data(symbol: str, market: str = "USD", interval: str = "1min") -> Dict:
        """
        Get real-time cryptocurrency data
        
        Args:
            symbol: Crypto symbol (e.g., BTC)
            market: Market to get data for (e.g., USD)
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            
        Returns:
            Dictionary with crypto data
        """
        try:
            # Determine the function based on the interval
            if interval in ["1min", "5min", "15min", "30min", "60min"]:
                function = "CRYPTO_INTRADAY"
            else:
                function = "DIGITAL_CURRENCY_" + interval.upper()
            
            params = {
                "function": function,
                "symbol": symbol,
                "market": market,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            if interval in ["1min", "5min", "15min", "30min", "60min"]:
                params["interval"] = interval
            
            response = requests.get(AlphaVantageTools.BASE_URL, params=params)
            data = response.json()
            
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            
            if "Note" in data:
                return {"warning": data["Note"], "data": data}
                
            # Format the response based on the interval
            if interval in ["1min", "5min", "15min", "30min", "60min"]:
                time_series_key = f"Time Series Crypto ({interval})"
                if time_series_key in data:
                    latest_timestamp = max(data[time_series_key].keys())
                    latest_data = data[time_series_key][latest_timestamp]
                    
                    formatted_data = {
                        "symbol": f"{symbol}/{market}",
                        "last_refreshed": data["Meta Data"]["6. Last Refreshed"],
                        "interval": interval,
                        "latest_price": latest_data["4. close"],
                        "latest_timestamp": latest_timestamp,
                        "open": latest_data["1. open"],
                        "high": latest_data["2. high"],
                        "low": latest_data["3. low"],
                        "volume": latest_data["5. volume"],
                        "full_data": data
                    }
                    
                    return formatted_data
            else:
                time_series_key = f"Time Series (Digital Currency {interval.capitalize()})"
                if time_series_key in data:
                    latest_timestamp = max(data[time_series_key].keys())
                    latest_data = data[time_series_key][latest_timestamp]
                    
                    formatted_data = {
                        "symbol": f"{symbol}/{market}",
                        "last_refreshed": latest_timestamp,
                        "interval": interval,
                        "latest_price": latest_data[f"4a. close ({market})"],
                        "open": latest_data[f"1a. open ({market})"],
                        "high": latest_data[f"2a. high ({market})"],
                        "low": latest_data[f"3a. low ({market})"],
                        "volume": latest_data[f"5. volume"],
                        "market_cap": latest_data.get(f"6. market cap ({market})", "N/A"),
                        "full_data": data
                    }
                    
                    return formatted_data
            
            return data
            
        except Exception as e:
            return {"error": f"Error fetching crypto data: {str(e)}"}
    
    @staticmethod
    def get_forex_data(from_currency: str, to_currency: str, interval: str = "1min") -> Dict:
        """
        Get real-time forex data
        
        Args:
            from_currency: From currency (e.g., EUR)
            to_currency: To currency (e.g., USD)
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            
        Returns:
            Dictionary with forex data
        """
        try:
            # Determine the function based on the interval
            if interval in ["1min", "5min", "15min", "30min", "60min"]:
                function = "FX_INTRADAY"
            elif interval == "daily":
                function = "FX_DAILY"
            elif interval == "weekly":
                function = "FX_WEEKLY"
            elif interval == "monthly":
                function = "FX_MONTHLY"
            else:
                return {"error": f"Invalid interval: {interval}"}
            
            params = {
                "function": function,
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            if interval in ["1min", "5min", "15min", "30min", "60min"]:
                params["interval"] = interval
            
            response = requests.get(AlphaVantageTools.BASE_URL, params=params)
            data = response.json()
            
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            
            if "Note" in data:
                return {"warning": data["Note"], "data": data}
                
            # Format the response based on the interval
            if interval in ["1min", "5min", "15min", "30min", "60min"]:
                time_series_key = f"Time Series FX ({interval})"
            else:
                time_series_key = f"Time Series FX ({interval.capitalize()})"
                
            if time_series_key in data:
                latest_timestamp = max(data[time_series_key].keys())
                latest_data = data[time_series_key][latest_timestamp]
                
                formatted_data = {
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "last_refreshed": data["Meta Data"]["5. Last Refreshed"],
                    "interval": interval,
                    "exchange_rate": latest_data["4. close"],
                    "open": latest_data["1. open"],
                    "high": latest_data["2. high"],
                    "low": latest_data["3. low"],
                    "full_data": data
                }
                
                return formatted_data
            
            return data
            
        except Exception as e:
            return {"error": f"Error fetching forex data: {str(e)}"}

    @staticmethod
    def get_economic_indicator(function: str) -> Dict:
        """
        Get economic indicator data
        
        Args:
            function: Economic indicator function (GDP, INFLATION, UNEMPLOYMENT, etc.)
            
        Returns:
            Dictionary with economic indicator data
        """
        try:
            # Map user-friendly names to Alpha Vantage function names
            function_map = {
                "GDP": "REAL_GDP",
                "INFLATION": "CPI",
                "UNEMPLOYMENT": "UNEMPLOYMENT",
                "RETAIL_SALES": "RETAIL_SALES",
                "TREASURY_YIELD": "TREASURY_YIELD",
                # Alpha Vantage doesn't have a direct CONSUMER_SENTIMENT endpoint
                # Using Consumer Price Index as an alternative
                "CONSUMER_SENTIMENT": "CPI",
                "NONFARM_PAYROLL": "NONFARM_PAYROLL"
            }
            
            if function in function_map:
                function = function_map[function]
            
            params = {
                "function": function,
                "interval": "monthly" if function != "REAL_GDP" else "quarterly",
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            response = requests.get(AlphaVantageTools.BASE_URL, params=params)
            data = response.json()
            
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            
            if "Note" in data:
                return {"warning": data["Note"], "data": data}
                
            if "data" in data:
                latest_data = data["data"][0]  # Most recent data point
                
                # Calculate trend (positive or negative)
                recent_values = [float(item["value"]) for item in data["data"][:6]]
                trend = recent_values[0] - recent_values[-1] if len(recent_values) > 1 else 0
                
                formatted_data = {
                    "indicator": function,
                    "latest_value": latest_data["value"],
                    "latest_date": latest_data["date"],
                    "recent_values": recent_values,  # Last 6 data points
                    "trend": trend,
                    "full_data": data
                }
                
                return formatted_data
            
            return data
            
        except Exception as e:
            return {"error": f"Error fetching economic indicator: {str(e)}"}

    @staticmethod
    def format_real_time_data_for_display(data: Dict) -> str:
        """
        Format real-time data for display
        
        Args:
            data: Dictionary with real-time data
            
        Returns:
            Formatted string for display
        """
        if "error" in data:
            return f"Error: {data['error']}"
        
        if "warning" in data:
            return f"Warning: {data['warning']}"
        
        # Format based on data structure
        if "symbol" in data:
            if "latest_price" in data:
                # Intraday data format
                return f"Real-Time Data for {data['symbol']}:\n" + \
                       f"Price: ${data['latest_price']}\n" + \
                       f"Last Updated: {data['latest_timestamp']}\n" + \
                       f"Open: ${data['open']} | High: ${data['high']} | Low: ${data['low']}\n" + \
                       f"Volume: {data['volume']}"
            elif "price" in data:
                # Quote data format
                return f"Real-Time Quote for {data['symbol']}:\n" + \
                       f"Price: ${data['price']}\n" + \
                       f"Change: {data['change']} ({data['change_percent']})\n" + \
                       f"Volume: {data['volume']}\n" + \
                       f"Trading Day: {data['latest_trading_day']}\n" + \
                       f"Previous Close: ${data['previous_close']}\n" + \
                       f"Open: ${data['open']} | High: ${data['high']} | Low: ${data['low']}"
            elif "exchange_rate" in data:
                # Forex data format
                return f"Real-Time Exchange Rate for {data['from_currency']}/{data['to_currency']}:\n" + \
                       f"Rate: {data['exchange_rate']}\n" + \
                       f"Last Updated: {data['last_refreshed']}\n" + \
                       f"Open: {data['open']} | High: {data['high']} | Low: {data['low']}"
        
        # Economic indicator format
        if "indicator" in data:
            trend_symbol = "↑" if data.get('trend', 0) >= 0 else "↓"
            trend_data = ", ".join([str(val) for val in data.get('recent_values', [])])
            
            return f"Economic Indicator: {data['indicator']}\n" + \
                   f"Latest Value: {data['latest_value']} (as of {data['latest_date']})\n" + \
                   f"Recent Trend: {trend_symbol} {trend_data}"
        
        # Default format for other data types
        return str(data)
