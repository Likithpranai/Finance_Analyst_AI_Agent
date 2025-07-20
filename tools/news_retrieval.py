"""
News Retrieval Module for Finance Analyst AI Agent

This module provides functions to retrieve financial news using Alpha Vantage API.
"""

import os
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class NewsRetrievalTools:
    """Tools for retrieving financial news"""
    
    @staticmethod
    def get_alpha_vantage_news(symbol=None, limit=10):
        """
        Get news from Alpha Vantage API
        
        Args:
            symbol: Stock ticker symbol (optional)
            limit: Maximum number of news items to retrieve
            
        Returns:
            List of news items or error message
        """
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return {"error": "Alpha Vantage API key not found"}
        
        try:
            # Build the URL based on whether we're getting news for a specific symbol or general market news
            if symbol:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
            else:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=financial_markets&apikey={api_key}"
            
            # Make the request
            response = requests.get(url)
            data = response.json()
            
            # Check for error
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            
            if "feed" not in data:
                return {"error": "No news data found"}
            
            # Format the response
            result = []
            for item in data["feed"][:limit]:
                # Format the date
                published_date = datetime.strptime(item["time_published"], "%Y%m%dT%H%M%S")
                
                # Create a news item
                news_item = {
                    "id": item.get("title", "")[:20].replace(" ", "_"),
                    "publisher": item.get("source", "Alpha Vantage"),
                    "title": item.get("title", "No title"),
                    "author": item.get("authors", ["Unknown"])[0] if item.get("authors") else "Unknown",
                    "published_utc": published_date.isoformat(),
                    "article_url": item.get("url", ""),
                    "tickers": [ticker["ticker"] for ticker in item.get("ticker_sentiment", [])],
                    "image_url": item.get("banner_image", ""),
                    "description": item.get("summary", "No description"),
                    "sentiment": item.get("overall_sentiment_score", 0)
                }
                result.append(news_item)
            
            return result
        
        except Exception as e:
            return {"error": f"Error fetching news: {str(e)}"}
    
    @staticmethod
    def get_market_news(symbol=None, limit=10):
        """
        Get latest news for a symbol or general market news
        
        Args:
            symbol: Stock ticker symbol (optional)
            limit: Maximum number of news items to retrieve
            
        Returns:
            List of news items or error message
        """
        # Try Alpha Vantage
        result = NewsRetrievalTools.get_alpha_vantage_news(symbol, limit)
        
        # Check if we got valid results
        if isinstance(result, list) and len(result) > 0:
            return {
                "data": result,
                "source": "Alpha Vantage"
            }
        
        # If we get here, all sources failed
        return {
            "error": "Could not retrieve news from any data source",
            "symbol": symbol if symbol else "market"
        }
