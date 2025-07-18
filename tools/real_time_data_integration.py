"""
Real-Time Data Integration Module for Finance Analyst AI Agent

This module integrates various real-time data sources including Polygon.io,
with caching and WebSocket streaming capabilities for professional-grade
financial analysis.
"""

import os
import json
import time
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dotenv import load_dotenv

# Import our custom modules
from tools.polygon_integration import PolygonIntegrationTools
from tools.websocket_manager import websocket_manager
# Import cache manager with fallback
try:
    from tools.cache_manager import cache_result, CacheManager
    redis_available = CacheManager.is_redis_available()
except Exception:
    # Define a no-op cache decorator if Redis is not available
    redis_available = False
    def cache_result(expiry_seconds=60):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

# Import Alpha Vantage for fallback
from tools.alpha_vantage_tools import AlphaVantageTools

# Load environment variables
load_dotenv()

# Check for API keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Track active WebSocket streams
active_streams = {}


class RealTimeDataTools:
    """Tools for real-time financial data with redundancy and caching"""
    
    @staticmethod
    @cache_result(expiry_seconds=5)  # Very short cache for real-time quotes
    def get_real_time_quote(symbol):
        """
        Get real-time stock quote with sub-second latency and fallback options
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with real-time quote data
        """
        try:
            # Try Polygon.io first (primary source)
            if POLYGON_API_KEY:
                result = PolygonIntegrationTools.get_real_time_quote(symbol)
                if result and not result.get("error"):
                    return {
                        "data": result,
                        "source": "Polygon.io",
                        "latency": "sub-second"
                    }
            
            # Fallback to Alpha Vantage
            if ALPHA_VANTAGE_API_KEY:
                result = AlphaVantageTools.get_real_time_quote(symbol)
                if result and not result.get("error"):
                    return {
                        "data": result,
                        "source": "Alpha Vantage",
                        "latency": "1-2 seconds"
                    }
            
            # If all else fails, return an error
            return {
                "error": "Could not retrieve real-time quote from any data source",
                "symbol": symbol
            }
        except Exception as e:
            return {
                "error": f"Error fetching real-time quote: {str(e)}",
                "symbol": symbol
            }
    
    @staticmethod
    @cache_result(expiry_seconds=60)  # 1 minute cache for intraday data
    def get_intraday_data(symbol, interval="1min", limit=100):
        """
        Get intraday data with specified interval and fallback options
        
        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            limit: Number of data points to retrieve
            
        Returns:
            DataFrame with intraday price data
        """
        try:
            # Try Polygon.io first (primary source)
            if POLYGON_API_KEY:
                result = PolygonIntegrationTools.get_intraday_data(symbol, interval, limit)
                if not result.empty:
                    result.attrs["source"] = "Polygon.io"
                    return result
            
            # Fallback to Alpha Vantage
            if ALPHA_VANTAGE_API_KEY:
                result = AlphaVantageTools.get_intraday_data(symbol, interval)
                if not result.empty:
                    result.attrs["source"] = "Alpha Vantage"
                    return result
            
            # If all else fails, return an empty DataFrame
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching intraday data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @cache_result(expiry_seconds=3600)  # 1 hour cache for historical data
    def get_historical_data(symbol, period="1y"):
        """
        Get historical daily data with fallback options
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (e.g., 1y, 2y, 5y)
            
        Returns:
            DataFrame with historical price data
        """
        try:
            # Calculate from_date based on period
            now = datetime.datetime.now()
            if period.endswith("d"):
                days = int(period[:-1])
                from_date = (now - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
            elif period.endswith("mo"):
                months = int(period[:-2])
                from_date = (now - datetime.timedelta(days=months*30)).strftime("%Y-%m-%d")
            elif period.endswith("y"):
                years = int(period[:-1])
                from_date = (now - datetime.timedelta(days=years*365)).strftime("%Y-%m-%d")
            else:
                from_date = (now - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            
            to_date = now.strftime("%Y-%m-%d")
            
            # Try Polygon.io first (primary source)
            if POLYGON_API_KEY:
                result = PolygonIntegrationTools.get_historical_data(symbol, from_date, to_date)
                if not result.empty:
                    result.attrs["source"] = "Polygon.io"
                    return result
            
            # Fallback to Alpha Vantage
            if ALPHA_VANTAGE_API_KEY:
                result = AlphaVantageTools.get_daily_adjusted(symbol, "full")
                if not result.empty:
                    # Filter by date range
                    result = result[result.index >= from_date]
                    result.attrs["source"] = "Alpha Vantage"
                    return result
            
            # If all else fails, return an empty DataFrame
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @cache_result(expiry_seconds=3600)  # 1 hour cache for crypto data
    def get_crypto_data(symbol, market="USD", interval="day"):
        """
        Get cryptocurrency data with fallback options
        
        Args:
            symbol: Crypto symbol (e.g., BTC)
            market: Market currency (e.g., USD)
            interval: Time interval (day, hour, minute)
            
        Returns:
            DataFrame with crypto price data
        """
        try:
            # Format symbol for Polygon
            poly_symbol = f"{symbol}{market}"
            
            # Calculate dates
            now = datetime.datetime.now()
            from_date = (now - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            to_date = now.strftime("%Y-%m-%d")
            
            # Try Polygon.io first (primary source)
            if POLYGON_API_KEY:
                result = PolygonIntegrationTools.get_crypto_data(poly_symbol, from_date, to_date)
                if not result.empty:
                    result.attrs["source"] = "Polygon.io"
                    return result
            
            # Fallback to Alpha Vantage
            if ALPHA_VANTAGE_API_KEY:
                result = AlphaVantageTools.get_crypto_data(symbol, market, interval)
                if not result.empty:
                    result.attrs["source"] = "Alpha Vantage"
                    return result
            
            # If all else fails, return an empty DataFrame
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching crypto data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @cache_result(expiry_seconds=3600)  # 1 hour cache for forex data
    def get_forex_data(from_currency, to_currency, interval="day"):
        """
        Get forex exchange rate data with fallback options
        
        Args:
            from_currency: Base currency code (e.g., EUR)
            to_currency: Quote currency code (e.g., USD)
            interval: Time interval (day, hour, minute)
            
        Returns:
            DataFrame with forex price data
        """
        try:
            # Calculate dates
            now = datetime.datetime.now()
            from_date = (now - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            to_date = now.strftime("%Y-%m-%d")
            
            # Try Polygon.io first (primary source)
            if POLYGON_API_KEY:
                result = PolygonIntegrationTools.get_forex_data(from_currency, to_currency, from_date, to_date)
                if not result.empty:
                    result.attrs["source"] = "Polygon.io"
                    return result
            
            # Fallback to Alpha Vantage
            if ALPHA_VANTAGE_API_KEY:
                result = AlphaVantageTools.get_forex_data(from_currency, to_currency, interval)
                if not result.empty:
                    result.attrs["source"] = "Alpha Vantage"
                    return result
            
            # If all else fails, return an empty DataFrame
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching forex data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @cache_result(expiry_seconds=86400)  # 24 hour cache for company details
    def get_company_details(symbol):
        """
        Get detailed company information with fallback options
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company details
        """
        try:
            # Try Polygon.io first (primary source)
            if POLYGON_API_KEY:
                result = PolygonIntegrationTools.get_company_details(symbol)
                if result and not result.get("error"):
                    return {
                        "data": result,
                        "source": "Polygon.io"
                    }
            
            # Fallback to Alpha Vantage or other sources
            # (Implementation depends on what's available in AlphaVantageTools)
            
            # If all else fails, return an error
            return {
                "error": "Could not retrieve company details from any data source",
                "symbol": symbol
            }
        except Exception as e:
            return {
                "error": f"Error fetching company details: {str(e)}",
                "symbol": symbol
            }
    
    @staticmethod
    @cache_result(expiry_seconds=900)  # 15 minute cache for news
    def get_market_news(symbol=None, limit=10):
        """
        Get latest news for a symbol or general market news with fallback options
        
        Args:
            symbol: Stock ticker symbol (optional)
            limit: Maximum number of news items to retrieve
            
        Returns:
            List of news items
        """
        try:
            # Try Polygon.io first (primary source)
            if POLYGON_API_KEY:
                result = PolygonIntegrationTools.get_news(symbol, limit)
                if result and not isinstance(result, dict):  # Check if it's not an error dict
                    return {
                        "data": result,
                        "source": "Polygon.io"
                    }
            
            # Fallback to Alpha Vantage or other sources
            # (Implementation depends on what's available in AlphaVantageTools)
            
            # If all else fails, return an error
            return {
                "error": "Could not retrieve news from any data source",
                "symbol": symbol if symbol else "market"
            }
        except Exception as e:
            return {
                "error": f"Error fetching news: {str(e)}",
                "symbol": symbol if symbol else "market"
            }
    
    @staticmethod
    def start_real_time_stream(symbols, callback, asset_type="stocks"):
        """
        Start a WebSocket stream for real-time updates
        
        Args:
            symbols: List of symbols to stream
            callback: Function to call with each update
            asset_type: Type of asset (stocks, crypto, forex)
            
        Returns:
            Stream ID for the WebSocket connection
        """
        try:
            # Choose the appropriate stream based on asset type
            if asset_type.lower() == "stocks":
                stream_id = websocket_manager.start_polygon_stock_stream(symbols, callback)
            elif asset_type.lower() == "crypto":
                stream_id = websocket_manager.start_polygon_crypto_stream(symbols, callback)
            elif asset_type.lower() == "forex":
                stream_id = websocket_manager.start_polygon_forex_stream(symbols, callback)
            else:
                return {"error": f"Unsupported asset type: {asset_type}"}
            
            # Store the stream ID
            active_streams[stream_id] = {
                "symbols": symbols,
                "asset_type": asset_type,
                "started_at": datetime.datetime.now().isoformat()
            }
            
            return {
                "stream_id": stream_id,
                "status": "started",
                "symbols": symbols,
                "asset_type": asset_type
            }
        except Exception as e:
            return {"error": f"Error starting real-time stream: {str(e)}"}
    
    @staticmethod
    def stop_real_time_stream(stream_id):
        """
        Stop a WebSocket stream
        
        Args:
            stream_id: ID of the stream to stop
            
        Returns:
            Status of the operation
        """
        try:
            result = websocket_manager.stop_stream(stream_id)
            if result:
                if stream_id in active_streams:
                    del active_streams[stream_id]
                return {"status": "stopped", "stream_id": stream_id}
            else:
                return {"error": f"Failed to stop stream: {stream_id}"}
        except Exception as e:
            return {"error": f"Error stopping stream: {str(e)}"}
    
    @staticmethod
    def get_active_streams():
        """
        Get information about active streams
        
        Returns:
            Dictionary with active stream information
        """
        return {
            "active_streams": len(active_streams),
            "streams": active_streams
        }
    
    @staticmethod
    def clear_cache(pattern=None):
        """
        Clear cache entries
        
        Args:
            pattern: Redis key pattern to match for deletion (optional)
            
        Returns:
            Number of keys deleted or status message if Redis is not available
        """
        if not redis_available:
            return {"status": "Redis cache not available", "cleared_entries": 0}
            
        try:
            pattern = pattern or "finance_agent:*"
            count = CacheManager.clear_cache(pattern)
            return {"cleared_entries": count, "pattern": pattern}
        except Exception as e:
            return {"error": str(e), "cleared_entries": 0}
    
    @staticmethod
    def get_cache_stats():
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics or status message if Redis is not available
        """
        if not redis_available:
            return {"status": "Redis cache not available", "keys": 0, "memory": "0 MB"}
            
        try:
            return CacheManager.get_cache_stats()
        except Exception as e:
            return {"error": str(e), "keys": 0, "memory": "0 MB"}


# Example usage
if __name__ == "__main__":
    # Test real-time quote
    symbol = "AAPL"
    print(f"\nGetting real-time quote for {symbol}...")
    quote = RealTimeDataTools.get_real_time_quote(symbol)
    print(json.dumps(quote, indent=2))
    
    # Test intraday data
    print(f"\nGetting intraday data for {symbol}...")
    intraday = RealTimeDataTools.get_intraday_data(symbol, interval="5min", limit=10)
    print(intraday.head())
    
    # Test historical data
    print(f"\nGetting historical data for {symbol}...")
    historical = RealTimeDataTools.get_historical_data(symbol, period="1y")
    print(historical.head())
    
    # Test cache stats
    print("\nCache statistics:")
    stats = RealTimeDataTools.get_cache_stats()
    print(json.dumps(stats, indent=2))
