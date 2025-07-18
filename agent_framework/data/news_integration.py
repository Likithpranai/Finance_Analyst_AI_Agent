"""
News Data Integration Module for Finance Analyst AI Agent
Provides integration with financial news APIs and data sources
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import requests
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class NewsIntegration:
    """
    Integration with financial news APIs and data sources
    """
    
    def __init__(self):
        """Initialize the news integration module"""
        self.api_keys = {
            "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "polygon": os.getenv("POLYGON_API_KEY"),
            "finnhub": os.getenv("FINNHUB_API_KEY"),
            "newsapi": os.getenv("NEWSAPI_API_KEY")
        }
        
        # Check for available APIs
        self.available_apis = []
        for api, key in self.api_keys.items():
            if key:
                self.available_apis.append(api)
        
        if not self.available_apis:
            logger.warning("No news API keys found. Using fallback news sources.")
    
    def get_company_news(self, symbol: str, days: int = 7, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get news articles for a company
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        # Try different APIs in order of preference
        for api in self.available_apis:
            try:
                if api == "alpha_vantage":
                    return self._get_alpha_vantage_news(symbol, days, max_items)
                elif api == "polygon":
                    return self._get_polygon_news(symbol, days, max_items)
                elif api == "finnhub":
                    return self._get_finnhub_news(symbol, days, max_items)
                elif api == "newsapi":
                    return self._get_newsapi_news(symbol, days, max_items)
            except Exception as e:
                logger.warning(f"Error getting news from {api}: {str(e)}")
        
        # Fallback to Yahoo Finance news via yfinance
        return self._get_yfinance_news(symbol, max_items)
    
    def _get_alpha_vantage_news(self, symbol: str, days: int = 7, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get news from Alpha Vantage API
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        api_key = self.api_keys["alpha_vantage"]
        if not api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if "feed" not in data:
            raise ValueError(f"Invalid response from Alpha Vantage: {data}")
        
        articles = []
        for item in data["feed"][:max_items]:
            article = {
                "title": item.get("title", ""),
                "description": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": item.get("time_published", ""),
                "sentiment": item.get("overall_sentiment_score", 0)
            }
            articles.append(article)
        
        return articles
    
    def _get_polygon_news(self, symbol: str, days: int = 7, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get news from Polygon.io API
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        api_key = self.api_keys["polygon"]
        if not api_key:
            raise ValueError("Polygon API key not found")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&published_utc.gte={start_date_str}&published_utc.lte={end_date_str}&limit={max_items}&apiKey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if "results" not in data:
            raise ValueError(f"Invalid response from Polygon: {data}")
        
        articles = []
        for item in data["results"]:
            article = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "url": item.get("article_url", ""),
                "source": item.get("publisher", {}).get("name", ""),
                "published_at": item.get("published_utc", ""),
                "keywords": item.get("keywords", []),
                "image_url": item.get("image_url", "")
            }
            articles.append(article)
        
        return articles
    
    def _get_finnhub_news(self, symbol: str, days: int = 7, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get news from Finnhub API
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        api_key = self.api_keys["finnhub"]
        if not api_key:
            raise ValueError("Finnhub API key not found")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date_str}&to={end_date_str}&token={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if not isinstance(data, list):
            raise ValueError(f"Invalid response from Finnhub: {data}")
        
        articles = []
        for item in data[:max_items]:
            article = {
                "title": item.get("headline", ""),
                "description": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "category": item.get("category", ""),
                "image_url": item.get("image", "")
            }
            articles.append(article)
        
        return articles
    
    def _get_newsapi_news(self, symbol: str, days: int = 7, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get news from News API
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        api_key = self.api_keys["newsapi"]
        if not api_key:
            raise ValueError("News API key not found")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Get company name from symbol for better search results
        company_name = self._get_company_name(symbol)
        query = f"{company_name} OR {symbol} stock"
        
        url = f"https://newsapi.org/v2/everything?q={query}&from={start_date_str}&sortBy=publishedAt&apiKey={api_key}&pageSize={max_items}"
        
        response = requests.get(url)
        data = response.json()
        
        if "articles" not in data:
            raise ValueError(f"Invalid response from News API: {data}")
        
        articles = []
        for item in data["articles"]:
            article = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
                "source": item.get("source", {}).get("name", ""),
                "published_at": item.get("publishedAt", ""),
                "author": item.get("author", ""),
                "image_url": item.get("urlToImage", "")
            }
            articles.append(article)
        
        return articles
    
    def _get_yfinance_news(self, symbol: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get news from Yahoo Finance via yfinance
        
        Args:
            symbol: Stock symbol
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        try:
            import yfinance as yf
            
            # Get ticker
            ticker = yf.Ticker(symbol)
            
            # Get news
            news_items = ticker.news
            
            articles = []
            for item in news_items[:max_items]:
                # Convert timestamp to ISO format
                published_at = datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat()
                
                article = {
                    "title": item.get("title", ""),
                    "description": item.get("summary", ""),
                    "url": item.get("link", ""),
                    "source": item.get("publisher", ""),
                    "published_at": published_at,
                    "type": item.get("type", ""),
                    "related_tickers": item.get("relatedTickers", [])
                }
                articles.append(article)
            
            return articles
        
        except Exception as e:
            logger.error(f"Error getting news from yfinance: {str(e)}")
            return self._get_fallback_news(symbol, max_items)
    
    def _get_fallback_news(self, symbol: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get fallback news when all APIs fail
        
        Args:
            symbol: Stock symbol
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        logger.warning(f"Using fallback news for {symbol}")
        
        # Return empty list with warning
        return [{
            "title": "News data unavailable",
            "description": "Unable to retrieve news data. Please check API keys or try again later.",
            "url": "",
            "source": "System",
            "published_at": datetime.now().isoformat(),
            "error": True
        }]
    
    def _get_company_name(self, symbol: str) -> str:
        """
        Get company name from symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company name
        """
        try:
            import yfinance as yf
            
            # Get ticker
            ticker = yf.Ticker(symbol)
            
            # Get company info
            info = ticker.info
            
            # Return company name
            return info.get("shortName", symbol)
        
        except Exception:
            return symbol
    
    def get_market_news(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get general market news
        
        Args:
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        # Try different APIs in order of preference
        for api in self.available_apis:
            try:
                if api == "alpha_vantage":
                    return self._get_alpha_vantage_market_news(max_items)
                elif api == "finnhub":
                    return self._get_finnhub_market_news(max_items)
                elif api == "newsapi":
                    return self._get_newsapi_market_news(max_items)
            except Exception as e:
                logger.warning(f"Error getting market news from {api}: {str(e)}")
        
        # Fallback to Yahoo Finance market news
        return self._get_yfinance_market_news(max_items)
    
    def _get_alpha_vantage_market_news(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get market news from Alpha Vantage API
        
        Args:
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        api_key = self.api_keys["alpha_vantage"]
        if not api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=financial_markets&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if "feed" not in data:
            raise ValueError(f"Invalid response from Alpha Vantage: {data}")
        
        articles = []
        for item in data["feed"][:max_items]:
            article = {
                "title": item.get("title", ""),
                "description": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": item.get("time_published", ""),
                "sentiment": item.get("overall_sentiment_score", 0)
            }
            articles.append(article)
        
        return articles
    
    def _get_finnhub_market_news(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get market news from Finnhub API
        
        Args:
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        api_key = self.api_keys["finnhub"]
        if not api_key:
            raise ValueError("Finnhub API key not found")
        
        url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if not isinstance(data, list):
            raise ValueError(f"Invalid response from Finnhub: {data}")
        
        articles = []
        for item in data[:max_items]:
            article = {
                "title": item.get("headline", ""),
                "description": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "category": item.get("category", ""),
                "image_url": item.get("image", "")
            }
            articles.append(article)
        
        return articles
    
    def _get_newsapi_market_news(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get market news from News API
        
        Args:
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        api_key = self.api_keys["newsapi"]
        if not api_key:
            raise ValueError("News API key not found")
        
        url = f"https://newsapi.org/v2/everything?q=stock+market+finance&sortBy=publishedAt&apiKey={api_key}&pageSize={max_items}"
        
        response = requests.get(url)
        data = response.json()
        
        if "articles" not in data:
            raise ValueError(f"Invalid response from News API: {data}")
        
        articles = []
        for item in data["articles"]:
            article = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
                "source": item.get("source", {}).get("name", ""),
                "published_at": item.get("publishedAt", ""),
                "author": item.get("author", ""),
                "image_url": item.get("urlToImage", "")
            }
            articles.append(article)
        
        return articles
    
    def _get_yfinance_market_news(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get market news from Yahoo Finance
        
        Args:
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        # Use SPY as proxy for market news
        return self._get_yfinance_news("SPY", max_items)
