"""
NewsAPI Integration for Finance Analyst AI Agent

This module provides comprehensive news coverage using NewsAPI's services.
Includes global news aggregation, sentiment analysis, and source diversity.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get NewsAPI key
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

class NewsAPITools:
    """Tools for accessing comprehensive news data from NewsAPI"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    @staticmethod
    def get_business_news(q: str = None, limit: int = 10, language: str = "en") -> Dict:
        """
        Get business and financial news
        
        Args:
            q: Search query (optional)
            limit: Maximum number of articles
            language: Language code (default: en)
            
        Returns:
            Dictionary with news data
        """
        if not NEWSAPI_API_KEY:
            return {"error": "NewsAPI key not found"}
        
        try:
            url = f"{NewsAPITools.BASE_URL}/top-headlines"
            headers = {"X-API-Key": NEWSAPI_API_KEY}
            params = {
                "category": "business",
                "language": language,
                "pageSize": min(limit, 100)  # NewsAPI max is 100
            }
            
            if q:
                params["q"] = q
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "ok":
                return {"error": f"NewsAPI error: {data.get('message', 'Unknown error')}"}
            
            # Format articles
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", "No title"),
                    "description": article.get("description", "No description"),
                    "url": article.get("url", ""),
                    "url_to_image": article.get("urlToImage", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "author": article.get("author", "Unknown")
                })
            
            return {
                "query": q if q else "business",
                "total_results": data.get("totalResults", 0),
                "articles": articles,
                "data_source": "NewsAPI"
            }
            
        except Exception as e:
            return {"error": f"Error fetching business news: {str(e)}"}
    
    @staticmethod
    def search_financial_news(query: str, limit: int = 10, sort_by: str = "publishedAt") -> Dict:
        """
        Search for specific financial news
        
        Args:
            query: Search query
            limit: Maximum number of articles
            sort_by: Sort by (relevancy, popularity, publishedAt)
            
        Returns:
            Dictionary with news data
        """
        if not NEWSAPI_API_KEY:
            return {"error": "NewsAPI key not found"}
        
        try:
            url = f"{NewsAPITools.BASE_URL}/everything"
            headers = {"X-API-Key": NEWSAPI_API_KEY}
            
            # Search in the last 7 days
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            params = {
                "q": f"{query} AND (finance OR stock OR market OR trading OR investment)",
                "from": from_date,
                "sortBy": sort_by,
                "language": "en",
                "pageSize": min(limit, 100)
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "ok":
                return {"error": f"NewsAPI error: {data.get('message', 'Unknown error')}"}
            
            # Format articles
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", "No title"),
                    "description": article.get("description", "No description"),
                    "url": article.get("url", ""),
                    "url_to_image": article.get("urlToImage", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "author": article.get("author", "Unknown"),
                    "content": article.get("content", "")
                })
            
            return {
                "query": query,
                "total_results": data.get("totalResults", 0),
                "articles": articles,
                "from_date": from_date,
                "sort_by": sort_by,
                "data_source": "NewsAPI"
            }
            
        except Exception as e:
            return {"error": f"Error searching financial news: {str(e)}"}
    
    @staticmethod
    def get_company_news(company_name: str, limit: int = 10) -> Dict:
        """
        Get news specifically about a company
        
        Args:
            company_name: Company name or ticker
            limit: Maximum number of articles
            
        Returns:
            Dictionary with company news
        """
        if not NEWSAPI_API_KEY:
            return {"error": "NewsAPI key not found"}
        
        try:
            # Search for company-specific news
            search_query = f'"{company_name}" OR "{company_name} stock" OR "{company_name} earnings"'
            
            url = f"{NewsAPITools.BASE_URL}/everything"
            headers = {"X-API-Key": NEWSAPI_API_KEY}
            
            # Search in the last 30 days
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            params = {
                "q": search_query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": min(limit, 100)
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "ok":
                return {"error": f"NewsAPI error: {data.get('message', 'Unknown error')}"}
            
            # Format articles
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", "No title"),
                    "description": article.get("description", "No description"),
                    "url": article.get("url", ""),
                    "url_to_image": article.get("urlToImage", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "author": article.get("author", "Unknown"),
                    "content": article.get("content", "")
                })
            
            return {
                "company": company_name,
                "total_results": data.get("totalResults", 0),
                "articles": articles,
                "from_date": from_date,
                "data_source": "NewsAPI"
            }
            
        except Exception as e:
            return {"error": f"Error fetching company news: {str(e)}"}
    
    @staticmethod
    def get_market_sentiment_news(limit: int = 20) -> Dict:
        """
        Get news for market sentiment analysis
        
        Args:
            limit: Maximum number of articles
            
        Returns:
            Dictionary with sentiment-focused news
        """
        if not NEWSAPI_API_KEY:
            return {"error": "NewsAPI key not found"}
        
        try:
            # Search for market sentiment keywords
            sentiment_keywords = [
                "market crash", "bull market", "bear market", "market rally",
                "recession", "economic growth", "inflation", "interest rates",
                "fed policy", "market volatility", "investor sentiment"
            ]
            
            query = " OR ".join([f'"{keyword}"' for keyword in sentiment_keywords])
            
            url = f"{NewsAPITools.BASE_URL}/everything"
            headers = {"X-API-Key": NEWSAPI_API_KEY}
            
            # Search in the last 3 days for recent sentiment
            from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": min(limit, 100)
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "ok":
                return {"error": f"NewsAPI error: {data.get('message', 'Unknown error')}"}
            
            # Format articles with sentiment indicators
            articles = []
            for article in data.get("articles", []):
                title = article.get("title", "").lower()
                description = article.get("description", "").lower() if article.get("description") else ""
                
                # Simple sentiment analysis based on keywords
                positive_words = ["rally", "growth", "gain", "rise", "bull", "optimistic", "positive"]
                negative_words = ["crash", "fall", "decline", "bear", "recession", "pessimistic", "negative"]
                
                sentiment_score = 0
                for word in positive_words:
                    if word in title or word in description:
                        sentiment_score += 1
                for word in negative_words:
                    if word in title or word in description:
                        sentiment_score -= 1
                
                sentiment = "neutral"
                if sentiment_score > 0:
                    sentiment = "positive"
                elif sentiment_score < 0:
                    sentiment = "negative"
                
                articles.append({
                    "title": article.get("title", "No title"),
                    "description": article.get("description", "No description"),
                    "url": article.get("url", ""),
                    "url_to_image": article.get("urlToImage", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "author": article.get("author", "Unknown"),
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score
                })
            
            return {
                "query": "market sentiment",
                "total_results": data.get("totalResults", 0),
                "articles": articles,
                "from_date": from_date,
                "data_source": "NewsAPI"
            }
            
        except Exception as e:
            return {"error": f"Error fetching sentiment news: {str(e)}"}
