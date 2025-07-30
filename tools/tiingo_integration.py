"""
Tiingo API Integration for Finance Analyst AI Agent

This module provides high-quality financial data using Tiingo's API services.
Includes real-time quotes, historical data, news, and alternative datasets.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Tiingo API key
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

class TiingoTools:
    """Tools for accessing high-quality financial data from Tiingo API"""
    
    BASE_URL = "https://api.tiingo.com/tiingo"
    
    @staticmethod
    def get_stock_metadata(symbol: str) -> Dict:
        """
        Get stock metadata and company information
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock metadata
        """
        if not TIINGO_API_KEY:
            return {"error": "Tiingo API key not found"}
        
        try:
            url = f"{TiingoTools.BASE_URL}/daily/{symbol}"
            headers = {"Content-Type": "application/json"}
            params = {"token": TIINGO_API_KEY}
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return {"error": f"No metadata found for {symbol}"}
            
            return {
                "symbol": symbol,
                "name": data.get("name", "N/A"),
                "description": data.get("description", "N/A"),
                "start_date": data.get("startDate", "N/A"),
                "end_date": data.get("endDate", "N/A"),
                "exchange_code": data.get("exchangeCode", "N/A"),
                "ticker": data.get("ticker", symbol),
                "data_source": "Tiingo"
            }
            
        except Exception as e:
            return {"error": f"Error fetching stock metadata: {str(e)}"}
    
    @staticmethod
    def get_historical_prices(symbol: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        Get historical stock prices
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with historical price data
        """
        if not TIINGO_API_KEY:
            return {"error": "Tiingo API key not found"}
        
        try:
            # Default to last 30 days if no dates provided
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            url = f"{TiingoTools.BASE_URL}/daily/{symbol}/prices"
            headers = {"Content-Type": "application/json"}
            params = {
                "token": TIINGO_API_KEY,
                "startDate": start_date,
                "endDate": end_date,
                "format": "json"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return {"error": f"No historical data found for {symbol}"}
            
            # Format the data
            prices = []
            for item in data:
                prices.append({
                    "date": item.get("date", ""),
                    "open": item.get("open", 0),
                    "high": item.get("high", 0),
                    "low": item.get("low", 0),
                    "close": item.get("close", 0),
                    "volume": item.get("volume", 0),
                    "adj_close": item.get("adjClose", 0),
                    "adj_high": item.get("adjHigh", 0),
                    "adj_low": item.get("adjLow", 0),
                    "adj_open": item.get("adjOpen", 0),
                    "adj_volume": item.get("adjVolume", 0),
                    "dividend_cash": item.get("divCash", 0),
                    "split_factor": item.get("splitFactor", 1)
                })
            
            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "prices": prices,
                "data_source": "Tiingo"
            }
            
        except Exception as e:
            return {"error": f"Error fetching historical prices: {str(e)}"}
    
    @staticmethod
    def get_latest_prices(symbols: List[str]) -> Dict:
        """
        Get latest prices for multiple symbols
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary with latest price data
        """
        if not TIINGO_API_KEY:
            return {"error": "Tiingo API key not found"}
        
        try:
            # Join symbols with comma
            symbols_str = ",".join(symbols)
            
            url = f"{TiingoTools.BASE_URL}/daily/{symbols_str}/prices"
            headers = {"Content-Type": "application/json"}
            params = {
                "token": TIINGO_API_KEY,
                "format": "json"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return {"error": "No price data found"}
            
            # Format the data
            prices = {}
            for item in data:
                symbol = item.get("ticker", "")
                if symbol:
                    prices[symbol] = {
                        "date": item.get("date", ""),
                        "open": item.get("open", 0),
                        "high": item.get("high", 0),
                        "low": item.get("low", 0),
                        "close": item.get("close", 0),
                        "volume": item.get("volume", 0),
                        "adj_close": item.get("adjClose", 0)
                    }
            
            return {
                "symbols": symbols,
                "prices": prices,
                "data_source": "Tiingo"
            }
            
        except Exception as e:
            return {"error": f"Error fetching latest prices: {str(e)}"}
    
    @staticmethod
    def get_crypto_data(symbol: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        Get cryptocurrency data
        
        Args:
            symbol: Crypto symbol (e.g., 'btcusd')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with crypto data
        """
        if not TIINGO_API_KEY:
            return {"error": "Tiingo API key not found"}
        
        try:
            # Default to last 7 days if no dates provided
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            url = f"https://api.tiingo.com/tiingo/crypto/prices"
            headers = {"Content-Type": "application/json"}
            params = {
                "token": TIINGO_API_KEY,
                "tickers": symbol,
                "startDate": start_date,
                "endDate": end_date,
                "resampleFreq": "1day"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return {"error": f"No crypto data found for {symbol}"}
            
            # Format the data
            crypto_data = []
            for item in data:
                if "priceData" in item:
                    for price_item in item["priceData"]:
                        crypto_data.append({
                            "date": price_item.get("date", ""),
                            "open": price_item.get("open", 0),
                            "high": price_item.get("high", 0),
                            "low": price_item.get("low", 0),
                            "close": price_item.get("close", 0),
                            "volume": price_item.get("volume", 0),
                            "volume_notional": price_item.get("volumeNotional", 0)
                        })
            
            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "data": crypto_data,
                "data_source": "Tiingo"
            }
            
        except Exception as e:
            return {"error": f"Error fetching crypto data: {str(e)}"}
    
    @staticmethod
    def get_news(symbols: List[str] = None, limit: int = 10, offset: int = 0) -> Dict:
        """
        Get financial news
        
        Args:
            symbols: List of stock symbols to filter news (optional)
            limit: Maximum number of news items
            offset: Number of items to skip
            
        Returns:
            Dictionary with news data
        """
        if not TIINGO_API_KEY:
            return {"error": "Tiingo API key not found"}
        
        try:
            url = "https://api.tiingo.com/tiingo/news"
            headers = {"Content-Type": "application/json"}
            params = {
                "token": TIINGO_API_KEY,
                "limit": limit,
                "offset": offset
            }
            
            if symbols:
                params["tickers"] = ",".join(symbols)
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return {"error": "No news data found"}
            
            # Format news items
            news_items = []
            for item in data:
                news_items.append({
                    "id": item.get("id", ""),
                    "title": item.get("title", "No title"),
                    "description": item.get("description", "No description"),
                    "url": item.get("url", ""),
                    "published_date": item.get("publishedDate", ""),
                    "crawl_date": item.get("crawlDate", ""),
                    "source": item.get("source", "Tiingo"),
                    "tickers": item.get("tickers", []),
                    "tags": item.get("tags", [])
                })
            
            return {
                "symbols": symbols if symbols else "general",
                "news": news_items,
                "limit": limit,
                "offset": offset,
                "data_source": "Tiingo"
            }
            
        except Exception as e:
            return {"error": f"Error fetching news: {str(e)}"}
    
    @staticmethod
    def get_fundamentals(symbol: str) -> Dict:
        """
        Get fundamental data for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with fundamental data
        """
        if not TIINGO_API_KEY:
            return {"error": "Tiingo API key not found"}
        
        try:
            url = f"https://api.tiingo.com/tiingo/fundamentals/{symbol}/statements"
            headers = {"Content-Type": "application/json"}
            params = {"token": TIINGO_API_KEY}
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return {"error": f"No fundamental data found for {symbol}"}
            
            # Get the most recent data
            latest_data = data[0] if data else {}
            
            return {
                "symbol": symbol,
                "year": latest_data.get("year", 0),
                "quarter": latest_data.get("quarter", 0),
                "statement_type": latest_data.get("statementType", ""),
                "data_code": latest_data.get("dataCode", ""),
                "report_type": latest_data.get("reportType", ""),
                "fundamental_data": latest_data,
                "data_source": "Tiingo"
            }
            
        except Exception as e:
            return {"error": f"Error fetching fundamental data: {str(e)}"}
