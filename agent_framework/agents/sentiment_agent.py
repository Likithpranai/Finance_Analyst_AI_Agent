"""
Sentiment Analysis Agent for Finance Analyst AI Agent Framework
Specializes in analyzing sentiment from financial news and social media
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import base agent
from agent_framework.agents.base_agent import BaseAgent

# Import sentiment analysis module
from agent_framework.ml.sentiment_analysis import SentimentAnalysis

# Import news integration module
from agent_framework.data.news_integration import NewsIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SentimentAgent(BaseAgent):
    """
    Sentiment Analysis Agent specializing in analyzing sentiment from financial news and social media
    """
    
    def __init__(self):
        """Initialize the Sentiment Analysis Agent"""
        super().__init__()
        
        # Initialize sentiment analysis module
        self.sentiment_analyzer = SentimentAnalysis()
        
        # Initialize news integration module
        self.news_integration = NewsIntegration()
        
        # Register tools
        self._register_tools()
        
        # Set agent-specific prompt
        self._set_agent_prompt()
    
    def _register_tools(self):
        """Register sentiment analysis tools"""
        self.register_tool("get_company_news", self.get_company_news)
        self.register_tool("get_market_news", self.get_market_news)
        self.register_tool("analyze_news_sentiment", self.analyze_news_sentiment)
        self.register_tool("get_sentiment_summary", self.get_sentiment_summary)
        self.register_tool("visualize_sentiment_trends", self.visualize_sentiment_trends)
        self.register_tool("analyze_text_sentiment", self.analyze_text_sentiment)
    
    def _set_agent_prompt(self):
        """Set agent-specific prompt"""
        self.system_prompt = """
        You are a Sentiment Analysis Agent specializing in analyzing sentiment from financial news and social media.
        Your goal is to provide insights into market sentiment and help users understand how news and social media
        are affecting market perceptions of stocks and the overall market.
        
        Follow the ReAct pattern (Reason → Act → Observe → Loop) to analyze sentiment:
        
        1. REASON: Analyze what sentiment information is needed based on the user's query
        2. ACT: Execute appropriate sentiment analysis tools
        3. OBSERVE: Analyze results from the tools
        4. LOOP: Use additional tools if needed for a comprehensive analysis
        
        Available tools:
        - get_company_news(symbol, days=7, max_items=10): Get news articles for a specific company
        - get_market_news(max_items=10): Get general market news
        - analyze_news_sentiment(articles): Analyze sentiment of news articles
        - get_sentiment_summary(articles): Get summary of sentiment analysis
        - visualize_sentiment_trends(articles): Visualize sentiment trends over time
        - analyze_text_sentiment(text): Analyze sentiment of a specific text
        
        When analyzing sentiment, consider:
        - Overall sentiment (positive, negative, neutral)
        - Sentiment trends over time
        - Key positive and negative factors
        - Potential impact on stock price or market
        - Correlation with price movements
        
        Provide clear, concise, and actionable insights based on sentiment analysis.
        """
    
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
        try:
            articles = self.news_integration.get_company_news(symbol, days, max_items)
            return articles
        except Exception as e:
            logger.error(f"Error getting company news: {str(e)}")
            return [{
                "title": "Error retrieving news",
                "description": f"Error: {str(e)}",
                "url": "",
                "source": "System",
                "published_at": datetime.now().isoformat(),
                "error": True
            }]
    
    def get_market_news(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get general market news
        
        Args:
            max_items: Maximum number of items to return
            
        Returns:
            List of news articles
        """
        try:
            articles = self.news_integration.get_market_news(max_items)
            return articles
        except Exception as e:
            logger.error(f"Error getting market news: {str(e)}")
            return [{
                "title": "Error retrieving market news",
                "description": f"Error: {str(e)}",
                "url": "",
                "source": "System",
                "published_at": datetime.now().isoformat(),
                "error": True
            }]
    
    def analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of news articles
        
        Args:
            articles: List of news articles
            
        Returns:
            List of articles with sentiment analysis
        """
        try:
            articles_with_sentiment = self.sentiment_analyzer.analyze_news_articles(articles)
            return articles_with_sentiment
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return articles
    
    def get_sentiment_summary(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of sentiment analysis
        
        Args:
            articles: List of news articles with sentiment analysis
            
        Returns:
            Dictionary with sentiment summary
        """
        try:
            summary = self.sentiment_analyzer.get_sentiment_summary(articles)
            return summary
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {str(e)}")
            return {
                "overall_sentiment": "unknown",
                "overall_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_articles": len(articles),
                "error": str(e)
            }
    
    def visualize_sentiment_trends(self, articles: List[Dict[str, Any]]) -> str:
        """
        Visualize sentiment trends over time
        
        Args:
            articles: List of news articles with sentiment analysis and timestamps
            
        Returns:
            Path to saved visualization
        """
        try:
            visualization_path = self.sentiment_analyzer.visualize_sentiment_trends(articles)
            return visualization_path
        except Exception as e:
            logger.error(f"Error visualizing sentiment trends: {str(e)}")
            return f"Error visualizing sentiment trends: {str(e)}"
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a specific text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            sentiment = self.sentiment_analyzer.analyze_text(text)
            return sentiment
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return {
                "sentiment": "unknown",
                "score": 0.0,
                "model": "error",
                "error": str(e)
            }
    
    def process_query(self, query: str) -> str:
        """
        Process a user query using sentiment analysis
        
        Args:
            query: User query string
            
        Returns:
            Response with sentiment analysis
        """
        # Extract symbols from query
        symbols = self.extract_symbols(query)
        
        # Determine if this is a market-wide or company-specific query
        is_market_query = self._is_market_query(query)
        
        # Process based on query type
        if is_market_query:
            return self._process_market_query(query)
        elif symbols:
            return self._process_company_query(query, symbols[0])
        else:
            return self._process_text_query(query)
    
    def _is_market_query(self, query: str) -> bool:
        """
        Determine if this is a market-wide query
        
        Args:
            query: User query string
            
        Returns:
            True if market query, False otherwise
        """
        market_keywords = [
            "market", "index", "indices", "s&p", "dow", "nasdaq", "russell",
            "overall", "economy", "economic", "macro", "global", "sector"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in market_keywords)
    
    def _process_market_query(self, query: str) -> str:
        """
        Process a market-wide sentiment query
        
        Args:
            query: User query string
            
        Returns:
            Response with market sentiment analysis
        """
        try:
            # Get market news
            articles = self.get_market_news(max_items=15)
            
            # Analyze sentiment
            articles_with_sentiment = self.analyze_news_sentiment(articles)
            
            # Get sentiment summary
            summary = self.get_sentiment_summary(articles_with_sentiment)
            
            # Visualize sentiment trends
            visualization_path = self.visualize_sentiment_trends(articles_with_sentiment)
            
            # Generate response using Gemini
            prompt = f"""
            Analyze the following market sentiment data and provide insights:
            
            Query: {query}
            
            Sentiment Summary:
            - Overall Sentiment: {summary['overall_sentiment']}
            - Overall Score: {summary['overall_score']:.2f}
            - Positive Articles: {summary['positive_count']}
            - Negative Articles: {summary['negative_count']}
            - Neutral Articles: {summary['neutral_count']}
            - Total Articles: {summary['total_articles']}
            
            Recent Market News with Sentiment:
            {self._format_articles_for_prompt(articles_with_sentiment[:5])}
            
            Visualization: {'Created successfully' if 'Error' not in visualization_path else 'Failed to create visualization'}
            
            Provide a comprehensive analysis of market sentiment based on this data.
            Include insights on overall market sentiment, key factors driving sentiment,
            and potential implications for market direction.
            """
            
            response = self.generate_response(prompt)
            
            # Add visualization path if successful
            if 'Error' not in visualization_path:
                response += f"\n\nA visualization of sentiment trends has been saved to: {visualization_path}"
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing market query: {str(e)}")
            return f"Error analyzing market sentiment: {str(e)}"
    
    def _process_company_query(self, query: str, symbol: str) -> str:
        """
        Process a company-specific sentiment query
        
        Args:
            query: User query string
            symbol: Stock symbol
            
        Returns:
            Response with company sentiment analysis
        """
        try:
            # Get company news
            articles = self.get_company_news(symbol, days=14, max_items=20)
            
            # Analyze sentiment
            articles_with_sentiment = self.analyze_news_sentiment(articles)
            
            # Get sentiment summary
            summary = self.get_sentiment_summary(articles_with_sentiment)
            
            # Visualize sentiment trends
            visualization_path = self.visualize_sentiment_trends(articles_with_sentiment)
            
            # Generate response using Gemini
            prompt = f"""
            Analyze the following company sentiment data and provide insights:
            
            Query: {query}
            Symbol: {symbol}
            
            Sentiment Summary:
            - Overall Sentiment: {summary['overall_sentiment']}
            - Overall Score: {summary['overall_score']:.2f}
            - Positive Articles: {summary['positive_count']}
            - Negative Articles: {summary['negative_count']}
            - Neutral Articles: {summary['neutral_count']}
            - Total Articles: {summary['total_articles']}
            
            Recent Company News with Sentiment:
            {self._format_articles_for_prompt(articles_with_sentiment[:5])}
            
            Visualization: {'Created successfully' if 'Error' not in visualization_path else 'Failed to create visualization'}
            
            Provide a comprehensive analysis of sentiment for {symbol} based on this data.
            Include insights on overall company sentiment, key factors driving sentiment,
            and potential implications for stock price movement.
            """
            
            response = self.generate_response(prompt)
            
            # Add visualization path if successful
            if 'Error' not in visualization_path:
                response += f"\n\nA visualization of sentiment trends has been saved to: {visualization_path}"
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing company query: {str(e)}")
            return f"Error analyzing sentiment for {symbol}: {str(e)}"
    
    def _process_text_query(self, query: str) -> str:
        """
        Process a text sentiment query
        
        Args:
            query: User query string
            
        Returns:
            Response with text sentiment analysis
        """
        try:
            # Extract text to analyze
            text_to_analyze = self._extract_text_to_analyze(query)
            
            # Analyze sentiment
            sentiment = self.analyze_text_sentiment(text_to_analyze)
            
            # Generate response using Gemini
            prompt = f"""
            Analyze the following text sentiment data and provide insights:
            
            Query: {query}
            
            Text: "{text_to_analyze}"
            
            Sentiment Analysis:
            - Sentiment: {sentiment['sentiment']}
            - Score: {sentiment['score']:.2f}
            - Model: {sentiment['model']}
            
            Provide an analysis of the sentiment of this text in the context of financial markets.
            Explain what this sentiment might mean for the subject of the text and any potential
            financial implications.
            """
            
            response = self.generate_response(prompt)
            return response
        
        except Exception as e:
            logger.error(f"Error processing text query: {str(e)}")
            return f"Error analyzing text sentiment: {str(e)}"
    
    def _extract_text_to_analyze(self, query: str) -> str:
        """
        Extract text to analyze from query
        
        Args:
            query: User query string
            
        Returns:
            Text to analyze
        """
        # Check for quoted text
        import re
        quoted_text = re.findall(r'"([^"]*)"', query)
        
        if quoted_text:
            return quoted_text[0]
        
        # Check for specific patterns
        analyze_patterns = [
            r"analyze[:\s]+(.+)",
            r"sentiment[:\s]+(.+)",
            r"analyze sentiment[:\s]+(.+)",
            r"sentiment of[:\s]+(.+)",
            r"analyze the sentiment of[:\s]+(.+)"
        ]
        
        for pattern in analyze_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Default to the entire query
        return query
    
    def _format_articles_for_prompt(self, articles: List[Dict[str, Any]]) -> str:
        """
        Format articles for inclusion in prompt
        
        Args:
            articles: List of articles with sentiment
            
        Returns:
            Formatted string
        """
        formatted = ""
        
        for i, article in enumerate(articles):
            sentiment = article.get("sentiment", {})
            sentiment_label = sentiment.get("sentiment", "unknown")
            sentiment_score = sentiment.get("score", 0.0)
            
            formatted += f"{i+1}. {article.get('title', 'No title')}\n"
            formatted += f"   Source: {article.get('source', 'Unknown')}\n"
            formatted += f"   Date: {article.get('published_at', 'Unknown')}\n"
            formatted += f"   Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})\n\n"
        
        return formatted
    
    def extract_symbols(self, query: str) -> List[str]:
        """
        Extract stock symbols from query
        
        Args:
            query: User query string
            
        Returns:
            List of stock symbols
        """
        import re
        
        # Look for ticker symbols (1-5 uppercase letters)
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query)
        
        # Filter out common words that might be mistaken for symbols
        common_words = {"A", "I", "AT", "BE", "DO", "GO", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE"}
        symbols = [symbol for symbol in symbols if symbol not in common_words]
        
        return symbols
