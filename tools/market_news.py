"""
Tools for fetching and analyzing market news
"""
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from typing import Dict, Any, List, Optional
import re

from config import MAX_NEWS_RESULTS


class GetStockNewsTool(BaseTool):
    """Tool for getting recent news about a stock"""
    
    name = "get_stock_news"
    description = """Gets recent news articles about a company or stock.
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - max_results (optional): Maximum number of news articles to return (default: 5)
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            max_results = int(kwargs.get("max_results", MAX_NEWS_RESULTS))
            
            if not ticker:
                return "Error: No ticker symbol provided"
            
            # Get stock information and news
            stock = yf.Ticker(ticker)
            
            # Get company name for better context
            company_name = stock.info.get("longName", stock.info.get("shortName", ticker))
            
            # Get news items
            news_items = stock.news
            
            if not news_items:
                return f"No recent news found for {company_name} ({ticker})"
            
            # Format news items
            formatted_news = []
            for item in news_items[:max_results]:
                # Convert timestamp to datetime
                pub_time = datetime.fromtimestamp(item.get("providerPublishTime", 0))
                
                # Format article
                article = {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "published_time": pub_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": item.get("summary", ""),
                    "url": item.get("link", ""),
                }
                formatted_news.append(article)
            
            # Create result
            result = {
                "symbol": ticker,
                "company_name": company_name,
                "news_count": len(formatted_news),
                "news": formatted_news
            }
            
            return result
            
        except Exception as e:
            return f"Error fetching news for {ticker}: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class AnalyzeMarketSentimentTool(BaseTool):
    """Tool for analyzing market sentiment from news articles"""
    
    name = "analyze_market_sentiment"
    description = """Analyzes the general market sentiment for a stock based on recent news.
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - max_results (optional): Maximum number of news articles to analyze (default: 5)
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            max_results = int(kwargs.get("max_results", MAX_NEWS_RESULTS))
            
            if not ticker:
                return "Error: No ticker symbol provided"
            
            # Get stock information and news
            stock = yf.Ticker(ticker)
            
            # Get company name for better context
            company_name = stock.info.get("longName", stock.info.get("shortName", ticker))
            
            # Get news items
            news_items = stock.news
            
            if not news_items:
                return f"No recent news found for {company_name} ({ticker})"
            
            # Perform simple sentiment analysis
            positive_keywords = [
                'surge', 'jump', 'rise', 'gain', 'profit', 'bull', 'beat', 'grow', 'positive',
                'success', 'up', 'high', 'strong', 'climb', 'rally', 'outperform', 'exceed',
                'upgrade', 'buy', 'recommend', 'opportunity'
            ]
            
            negative_keywords = [
                'drop', 'fall', 'sink', 'loss', 'bear', 'miss', 'decline', 'negative', 'fail',
                'down', 'low', 'weak', 'plunge', 'sell', 'underperform', 'downgrade', 'warning',
                'risk', 'bankruptcy', 'lawsuit', 'investigation', 'probe', 'concern', 'short'
            ]
            
            neutral_keywords = [
                'hold', 'maintain', 'neutral', 'mixed', 'steady', 'stable', 'flat', 'unchanged',
                'expected', 'in-line', 'meet', 'normal', 'balance', 'wait', 'watch'
            ]
            
            # Analyze each news item
            news_analysis = []
            overall_sentiment_score = 0
            
            for item in news_items[:max_results]:
                title = item.get("title", "").lower()
                summary = item.get("summary", "").lower()
                combined_text = title + " " + summary
                
                # Count sentiment keywords
                positive_count = sum(1 for word in positive_keywords if re.search(r'\b' + word + r'\b', combined_text))
                negative_count = sum(1 for word in negative_keywords if re.search(r'\b' + word + r'\b', combined_text))
                neutral_count = sum(1 for word in neutral_keywords if re.search(r'\b' + word + r'\b', combined_text))
                
                # Calculate sentiment score (-1 to 1)
                total_keywords = positive_count + negative_count + neutral_count
                if total_keywords > 0:
                    sentiment_score = (positive_count - negative_count) / total_keywords
                else:
                    sentiment_score = 0
                
                # Determine sentiment
                if sentiment_score > 0.3:
                    sentiment = "positive"
                elif sentiment_score < -0.3:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                # Convert timestamp to datetime
                pub_time = datetime.fromtimestamp(item.get("providerPublishTime", 0))
                
                # Add to analysis
                analysis = {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "published_time": pub_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                }
                news_analysis.append(analysis)
                overall_sentiment_score += sentiment_score
            
            # Calculate overall sentiment
            if news_analysis:
                overall_sentiment_score /= len(news_analysis)
                
                if overall_sentiment_score > 0.3:
                    overall_sentiment = "positive"
                elif overall_sentiment_score < -0.3:
                    overall_sentiment = "negative"
                else:
                    overall_sentiment = "neutral"
            else:
                overall_sentiment = "unknown"
                overall_sentiment_score = 0
            
            # Create result
            result = {
                "symbol": ticker,
                "company_name": company_name,
                "overall_sentiment": overall_sentiment,
                "overall_sentiment_score": overall_sentiment_score,
                "news_count": len(news_analysis),
                "news_analysis": news_analysis
            }
            
            return result
            
        except Exception as e:
            return f"Error analyzing sentiment for {ticker}: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)
