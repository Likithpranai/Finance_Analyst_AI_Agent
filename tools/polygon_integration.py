"""
Polygon.io Integration Module for Finance Analyst AI Agent

This module provides real-time and historical financial data using Polygon.io's
premium API services. It includes WebSocket streaming for real-time data and
Redis caching to reduce API calls and latency.

Features:
- Real-time stock quotes with sub-second latency
- WebSocket streaming for continuous data updates
- Historical data with deep market depth
- Support for stocks, options, forex, and crypto
- Redis caching for frequently queried data
- Fallback mechanisms to ensure data reliability
"""

import os
import json
import time
import datetime
import pandas as pd
import numpy as np
import websocket
import threading
import redis
from polygon import RESTClient, WebSocketClient
from polygon.websocket.models import WebSocketMessage
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Load environment variables
load_dotenv()

# Get Polygon.io API key from environment variables
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    print("Warning: POLYGON_API_KEY not found in environment variables")
    print("Polygon.io integration will not work without an API key")

# Initialize Redis client for caching
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis cache connected successfully")
except Exception as e:
    print(f"Redis cache not available: {str(e)}")
    REDIS_AVAILABLE = False


class PolygonIntegrationTools:
    """Tools for integrating with Polygon.io API for real-time and historical financial data"""
    
    @staticmethod
    def _get_rest_client():
        """Get a Polygon REST client instance"""
        if not POLYGON_API_KEY:
            raise ValueError("Polygon.io API key not found")
        return RESTClient(api_key=POLYGON_API_KEY)
    
    @staticmethod
    def _get_cache_key(function_name, **kwargs):
        """Generate a cache key based on function name and parameters"""
        # Convert kwargs to a sorted string representation for consistent keys
        params_str = json.dumps(kwargs, sort_keys=True)
        return f"polygon:{function_name}:{params_str}"
    
    @staticmethod
    def _get_from_cache(cache_key):
        """Get data from Redis cache if available"""
        if not REDIS_AVAILABLE:
            return None
        
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            print(f"Cache retrieval error: {str(e)}")
            return None
    
    @staticmethod
    def _save_to_cache(cache_key, data, expiry_seconds=300):
        """Save data to Redis cache with expiration"""
        if not REDIS_AVAILABLE:
            return
        
        try:
            redis_client.setex(
                cache_key,
                expiry_seconds,
                json.dumps(data)
            )
        except Exception as e:
            print(f"Cache saving error: {str(e)}")
    
    @classmethod
    def get_real_time_quote(cls, symbol):
        """
        Get real-time stock quote with sub-second latency
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with real-time quote data
        """
        # Check cache first (very short expiry for real-time data)
        cache_key = cls._get_cache_key("get_real_time_quote", symbol=symbol)
        cached_data = cls._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            client = cls._get_rest_client()
            # Get last trade
            last_trade = client.get_last_trade(symbol)
            
            # Get current quote (bid/ask)
            quote = client.get_last_quote(symbol)
            
            # Format the response
            result = {
                "symbol": symbol,
                "last_price": last_trade.price,
                "last_size": last_trade.size,
                "last_exchange": last_trade.exchange,
                "last_timestamp": last_trade.sip_timestamp.isoformat(),
                "bid_price": quote.bid_price,
                "bid_size": quote.bid_size,
                "ask_price": quote.ask_price,
                "ask_size": quote.ask_size,
                "spread": quote.ask_price - quote.bid_price,
                "timestamp": quote.sip_timestamp.isoformat(),
                "data_source": "Polygon.io (Real-Time)"
            }
            
            # Cache for a very short time (5 seconds for real-time data)
            cls._save_to_cache(cache_key, result, expiry_seconds=5)
            
            return result
        except Exception as e:
            error_msg = f"Error fetching real-time quote for {symbol}: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    @classmethod
    def get_intraday_data(cls, symbol, interval="1min", limit=100):
        """
        Get intraday data with specified interval
        
        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            limit: Number of data points to retrieve
            
        Returns:
            DataFrame with intraday price data
        """
        # Map interval to Polygon.io timespan
        timespan_map = {
            "1min": "minute",
            "5min": "minute",
            "15min": "minute",
            "30min": "minute",
            "60min": "hour"
        }
        
        # Map interval to multiplier
        multiplier_map = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "60min": 1
        }
        
        if interval not in timespan_map:
            return {"error": f"Invalid interval: {interval}. Supported intervals: 1min, 5min, 15min, 30min, 60min"}
        
        # Check cache (longer expiry for intraday data)
        cache_key = cls._get_cache_key("get_intraday_data", symbol=symbol, interval=interval, limit=limit)
        cached_data = cls._get_from_cache(cache_key)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            client = cls._get_rest_client()
            timespan = timespan_map[interval]
            multiplier = multiplier_map[interval]
            
            # Get aggregated bars
            now = datetime.datetime.now()
            # Calculate start time based on limit and interval
            if interval == "1min":
                start_time = now - datetime.timedelta(minutes=limit)
            elif interval == "5min":
                start_time = now - datetime.timedelta(minutes=5*limit)
            elif interval == "15min":
                start_time = now - datetime.timedelta(minutes=15*limit)
            elif interval == "30min":
                start_time = now - datetime.timedelta(minutes=30*limit)
            else:  # 60min
                start_time = now - datetime.timedelta(hours=limit)
                
            # Format dates for API
            from_date = start_time.strftime("%Y-%m-%d")
            to_date = now.strftime("%Y-%m-%d")
            
            # Get aggregated bars
            aggs = client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                limit=limit
            )
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    "timestamp": datetime.datetime.fromtimestamp(agg.timestamp / 1000),
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                    "vwap": agg.vwap,
                    "transactions": agg.transactions
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
            
            # Cache for a moderate time (1 minute for intraday data)
            cls._save_to_cache(cache_key, data, expiry_seconds=60)
            
            return df
        except Exception as e:
            error_msg = f"Error fetching intraday data for {symbol}: {str(e)}"
            print(error_msg)
            return pd.DataFrame()
    
    @classmethod
    def get_historical_data(cls, symbol, from_date, to_date=None, timespan="day"):
        """
        Get historical daily data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD), defaults to today
            timespan: Time interval (day, hour, minute, etc.)
            
        Returns:
            DataFrame with historical price data
        """
        if to_date is None:
            to_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        # Check cache (longer expiry for historical data)
        cache_key = cls._get_cache_key("get_historical_data", symbol=symbol, from_date=from_date, to_date=to_date, timespan=timespan)
        cached_data = cls._get_from_cache(cache_key)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            client = cls._get_rest_client()
            
            # Get aggregated bars
            aggs = client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                limit=50000  # Maximum allowed by Polygon
            )
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    "timestamp": datetime.datetime.fromtimestamp(agg.timestamp / 1000),
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                    "vwap": agg.vwap,
                    "transactions": agg.transactions
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
                
                # Add Adj Close column for compatibility with yfinance
                df["Adj Close"] = df["close"]
                
                # Rename columns to match yfinance format
                df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                }, inplace=True)
            
            # Cache for a longer time (1 hour for historical data)
            cls._save_to_cache(cache_key, data, expiry_seconds=3600)
            
            return df
        except Exception as e:
            error_msg = f"Error fetching historical data for {symbol}: {str(e)}"
            print(error_msg)
            return pd.DataFrame()
    
    @classmethod
    def get_crypto_data(cls, symbol, from_date, to_date=None):
        """
        Get cryptocurrency data
        
        Args:
            symbol: Crypto symbol (e.g., X:BTCUSD)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with crypto price data
        """
        # Ensure symbol has X: prefix for crypto
        if not symbol.startswith("X:"):
            symbol = f"X:{symbol}"
            
        return cls.get_historical_data(symbol, from_date, to_date)
    
    @classmethod
    def get_forex_data(cls, from_currency, to_currency, from_date, to_date=None):
        """
        Get forex exchange rate data
        
        Args:
            from_currency: Base currency code (e.g., EUR)
            to_currency: Quote currency code (e.g., USD)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with forex price data
        """
        # Format forex symbol for Polygon
        symbol = f"C:{from_currency}{to_currency}"
        
        return cls.get_historical_data(symbol, from_date, to_date)
    
    @classmethod
    def get_company_details(cls, symbol):
        """
        Get detailed company information
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company details
        """
        # Check cache (can be cached longer as company details don't change often)
        cache_key = cls._get_cache_key("get_company_details", symbol=symbol)
        cached_data = cls._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            client = cls._get_rest_client()
            
            # Get ticker details
            ticker_details = client.get_ticker_details(symbol)
            
            # Format the response
            result = {
                "symbol": symbol,
                "name": ticker_details.name,
                "market": ticker_details.market,
                "locale": ticker_details.locale,
                "primary_exchange": ticker_details.primary_exchange,
                "type": ticker_details.type,
                "active": ticker_details.active,
                "currency_name": ticker_details.currency_name,
                "cik": ticker_details.cik,
                "composite_figi": ticker_details.composite_figi,
                "share_class_figi": ticker_details.share_class_figi,
                "description": ticker_details.description,
                "sic_code": ticker_details.sic_code,
                "sic_description": ticker_details.sic_description,
                "ticker_root": ticker_details.ticker_root,
                "homepage_url": ticker_details.homepage_url,
                "total_employees": ticker_details.total_employees,
                "list_date": ticker_details.list_date,
                "data_source": "Polygon.io"
            }
            
            # Cache for a day (company details don't change often)
            cls._save_to_cache(cache_key, result, expiry_seconds=86400)
            
            return result
        except Exception as e:
            error_msg = f"Error fetching company details for {symbol}: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    @classmethod
    def get_market_status(cls):
        """
        Get current market status (open/closed)
        
        Returns:
            Dictionary with market status information
        """
        # Check cache (short expiry for market status)
        cache_key = cls._get_cache_key("get_market_status")
        cached_data = cls._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            client = cls._get_rest_client()
            
            # Get market status
            market_status = client.get_market_status()
            
            # Format the response
            result = {
                "market": market_status.market,
                "server_time": market_status.server_time.isoformat(),
                "exchanges": {}
            }
            
            # Add exchange statuses
            for exchange in market_status.exchanges:
                result["exchanges"][exchange.name] = {
                    "name": exchange.name,
                    "type": exchange.type,
                    "market": exchange.market,
                    "status": exchange.status,
                    "session_start": exchange.session_start.isoformat() if exchange.session_start else None,
                    "session_end": exchange.session_end.isoformat() if exchange.session_end else None,
                }
            
            # Cache for 5 minutes (market status can change)
            cls._save_to_cache(cache_key, result, expiry_seconds=300)
            
            return result
        except Exception as e:
            error_msg = f"Error fetching market status: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    @classmethod
    def get_market_holidays(cls):
        """
        Get market holidays for the current year
        
        Returns:
            List of market holidays
        """
        # Check cache (can be cached longer as holidays don't change often)
        cache_key = cls._get_cache_key("get_market_holidays")
        cached_data = cls._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            client = cls._get_rest_client()
            
            # Get market holidays
            holidays = client.get_market_holidays()
            
            # Format the response
            result = []
            for holiday in holidays:
                result.append({
                    "exchange": holiday.exchange,
                    "name": holiday.name,
                    "date": holiday.date.isoformat(),
                    "status": holiday.status,
                    "open": holiday.open.isoformat() if holiday.open else None,
                    "close": holiday.close.isoformat() if holiday.close else None
                })
            
            # Cache for a month (holidays don't change often)
            cls._save_to_cache(cache_key, result, expiry_seconds=2592000)  # 30 days
            
            return result
        except Exception as e:
            error_msg = f"Error fetching market holidays: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    @classmethod
    def get_news(cls, symbol=None, limit=10):
        """
        Get latest news for a symbol or general market news
        
        Args:
            symbol: Stock ticker symbol (optional)
            limit: Maximum number of news items to retrieve
            
        Returns:
            List of news items
        """
        # Check cache (short expiry for news)
        cache_key = cls._get_cache_key("get_news", symbol=symbol, limit=limit)
        cached_data = cls._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            client = cls._get_rest_client()
            
            # Get news
            if symbol:
                news = client.get_ticker_news(symbol, limit=limit)
            else:
                news = client.get_ticker_news("SPY", limit=limit)  # General market news
            
            # Format the response
            result = []
            for item in news:
                result.append({
                    "id": item.id,
                    "publisher": item.publisher.name,
                    "title": item.title,
                    "author": item.author,
                    "published_utc": item.published_utc.isoformat(),
                    "article_url": item.article_url,
                    "tickers": item.tickers,
                    "amp_url": item.amp_url,
                    "image_url": item.image_url,
                    "description": item.description
                })
            
            # Cache for 15 minutes (news can change)
            cls._save_to_cache(cache_key, result, expiry_seconds=900)
            
            return result
        except Exception as e:
            error_msg = f"Error fetching news for {symbol if symbol else 'market'}: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    # WebSocket streaming implementation
    @staticmethod
    def start_stock_stream(symbols, callback_function):
        """
        Start a WebSocket stream for real-time stock updates
        
        Args:
            symbols: List of stock symbols to stream
            callback_function: Function to call with each update
            
        Returns:
            WebSocket client instance that can be used to stop the stream
        """
        if not POLYGON_API_KEY:
            raise ValueError("Polygon.io API key not found")
        
        def on_message(ws_client, message):
            """Handle incoming WebSocket messages"""
            try:
                # Parse the message
                msg_dict = json.loads(message)
                # Call the callback function with the parsed message
                callback_function(msg_dict)
            except Exception as e:
                print(f"Error processing WebSocket message: {str(e)}")
        
        def on_error(ws_client, error):
            """Handle WebSocket errors"""
            print(f"WebSocket error: {str(error)}")
        
        def on_close(ws_client, close_status_code, close_msg):
            """Handle WebSocket connection close"""
            print(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
        
        def on_open(ws_client):
            """Handle WebSocket connection open"""
            print("WebSocket connection opened")
            # Subscribe to the specified symbols
            auth_message = {"action": "auth", "params": POLYGON_API_KEY}
            ws_client.send(json.dumps(auth_message))
            
            # Subscribe to trades for the specified symbols
            symbols_str = ",".join([f"T.{symbol}" for symbol in symbols])
            subscribe_message = {"action": "subscribe", "params": symbols_str}
            ws_client.send(json.dumps(subscribe_message))
        
        # Create and start WebSocket client
        ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/stocks",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start the WebSocket connection in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        return ws  # Return the WebSocket client for later use (e.g., to stop the stream)
    
    @staticmethod
    def stop_stream(ws_client):
        """
        Stop a WebSocket stream
        
        Args:
            ws_client: WebSocket client instance to stop
        """
        if ws_client:
            ws_client.close()


# Example usage of WebSocket streaming
def example_callback(message):
    """Example callback function for WebSocket messages"""
    print(f"Received real-time update: {message}")


if __name__ == "__main__":
    # Example usage
    print("Testing Polygon.io integration...")
    
    # Test real-time quote
    symbol = "AAPL"
    print(f"\nGetting real-time quote for {symbol}...")
    quote = PolygonIntegrationTools.get_real_time_quote(symbol)
    print(json.dumps(quote, indent=2))
    
    # Test intraday data
    print(f"\nGetting intraday data for {symbol}...")
    intraday = PolygonIntegrationTools.get_intraday_data(symbol, interval="5min", limit=10)
    print(intraday.head())
    
    # Test historical data
    print(f"\nGetting historical data for {symbol}...")
    from_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    to_date = datetime.datetime.now().strftime("%Y-%m-%d")
    historical = PolygonIntegrationTools.get_historical_data(symbol, from_date, to_date)
    print(historical.head())
    
    # Test company details
    print(f"\nGetting company details for {symbol}...")
    details = PolygonIntegrationTools.get_company_details(symbol)
    print(json.dumps(details, indent=2))
    
    # Test market status
    print("\nGetting market status...")
    status = PolygonIntegrationTools.get_market_status()
    print(json.dumps(status, indent=2))
    
    # Test WebSocket streaming (uncomment to test)
    # print("\nStarting WebSocket stream for AAPL, MSFT, GOOGL...")
    # ws = PolygonIntegrationTools.start_stock_stream(["AAPL", "MSFT", "GOOGL"], example_callback)
    # print("WebSocket stream started. Press Ctrl+C to stop.")
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Stopping WebSocket stream...")
    #     PolygonIntegrationTools.stop_stream(ws)
    #     print("WebSocket stream stopped.")
