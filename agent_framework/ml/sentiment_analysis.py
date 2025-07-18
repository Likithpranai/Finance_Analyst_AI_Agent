"""
Sentiment Analysis Module for Finance Analyst AI Agent
Provides sentiment analysis capabilities for financial news and social media
"""
import os
import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SentimentAnalysis:
    """
    Sentiment Analysis for financial news and social media
    """
    
    def __init__(self):
        """Initialize the sentiment analysis module"""
        self.nlp_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models for sentiment analysis"""
        try:
            # Try to import transformers
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers import pipeline
            
            # Initialize FinBERT model for financial sentiment analysis
            try:
                model_name = "ProsusAI/finbert"
                self.nlp_models["finbert"] = pipeline("sentiment-analysis", model=model_name)
                logger.info(f"Successfully loaded FinBERT model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT model: {str(e)}")
            
            # Initialize RoBERTa model as fallback
            try:
                model_name = "cardiffnlp/twitter-roberta-base-sentiment"
                self.nlp_models["roberta"] = pipeline("sentiment-analysis", model=model_name)
                logger.info(f"Successfully loaded RoBERTa model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load RoBERTa model: {str(e)}")
                
        except ImportError:
            logger.warning("Transformers library not available. Using fallback sentiment analysis.")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback sentiment analysis using NLTK"""
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            # Download VADER lexicon if not already downloaded
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon')
            
            # Initialize VADER sentiment analyzer
            self.nlp_models["vader"] = SentimentIntensityAnalyzer()
            logger.info("Successfully loaded VADER sentiment analyzer as fallback")
        except ImportError:
            logger.warning("NLTK library not available. Using simple rule-based sentiment analysis.")
            self.nlp_models["simple"] = None
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Use FinBERT if available
        if "finbert" in self.nlp_models:
            try:
                result = self.nlp_models["finbert"](cleaned_text)
                return {
                    "sentiment": result[0]["label"],
                    "score": result[0]["score"],
                    "model": "finbert"
                }
            except Exception as e:
                logger.warning(f"FinBERT analysis failed: {str(e)}")
        
        # Use RoBERTa if available
        if "roberta" in self.nlp_models:
            try:
                result = self.nlp_models["roberta"](cleaned_text)
                # Map RoBERTa labels to positive/negative/neutral
                label_map = {
                    "LABEL_0": "negative",
                    "LABEL_1": "neutral",
                    "LABEL_2": "positive"
                }
                return {
                    "sentiment": label_map.get(result[0]["label"], result[0]["label"]),
                    "score": result[0]["score"],
                    "model": "roberta"
                }
            except Exception as e:
                logger.warning(f"RoBERTa analysis failed: {str(e)}")
        
        # Use VADER if available
        if "vader" in self.nlp_models:
            try:
                scores = self.nlp_models["vader"].polarity_scores(cleaned_text)
                sentiment = "neutral"
                if scores["compound"] >= 0.05:
                    sentiment = "positive"
                elif scores["compound"] <= -0.05:
                    sentiment = "negative"
                
                return {
                    "sentiment": sentiment,
                    "score": scores["compound"],
                    "model": "vader",
                    "details": scores
                }
            except Exception as e:
                logger.warning(f"VADER analysis failed: {str(e)}")
        
        # Use simple rule-based analysis as last resort
        return self._simple_sentiment_analysis(cleaned_text)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for sentiment analysis
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _simple_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Simple positive and negative word lists
        positive_words = [
            'up', 'rise', 'rising', 'rose', 'high', 'higher', 'gain', 'gains', 'positive',
            'bull', 'bullish', 'buy', 'buying', 'growth', 'growing', 'grew', 'increase',
            'increasing', 'increased', 'profit', 'profitable', 'strong', 'strength',
            'opportunity', 'opportunities', 'success', 'successful', 'good', 'great',
            'excellent', 'outperform', 'outperforming', 'beat', 'beating', 'exceeded'
        ]
        
        negative_words = [
            'down', 'fall', 'falling', 'fell', 'low', 'lower', 'loss', 'losses', 'negative',
            'bear', 'bearish', 'sell', 'selling', 'decline', 'declining', 'declined',
            'decrease', 'decreasing', 'decreased', 'risk', 'risky', 'weak', 'weakness',
            'threat', 'threats', 'fail', 'failing', 'failed', 'failure', 'bad', 'poor',
            'terrible', 'underperform', 'underperforming', 'miss', 'missing', 'missed'
        ]
        
        # Count positive and negative words
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            sentiment = "neutral"
            score = 0.0
        else:
            score = (positive_count - negative_count) / total_count
            if score > 0.1:
                sentiment = "positive"
            elif score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": score,
            "model": "simple",
            "details": {
                "positive_count": positive_count,
                "negative_count": negative_count
            }
        }
    
    def analyze_news_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of news articles
        
        Args:
            articles: List of news articles with 'title' and 'description' fields
            
        Returns:
            List of articles with sentiment analysis results
        """
        results = []
        
        for article in articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            # Analyze sentiment
            sentiment = self.analyze_text(text)
            
            # Add sentiment to article
            article_with_sentiment = article.copy()
            article_with_sentiment["sentiment"] = sentiment
            
            results.append(article_with_sentiment)
        
        return results
    
    def calculate_news_sentiment_score(self, articles: List[Dict[str, Any]]) -> float:
        """
        Calculate overall sentiment score for news articles
        
        Args:
            articles: List of news articles with sentiment analysis
            
        Returns:
            Overall sentiment score (-1.0 to 1.0)
        """
        if not articles:
            return 0.0
        
        # Extract sentiment scores
        scores = []
        for article in articles:
            if "sentiment" in article and "score" in article["sentiment"]:
                scores.append(article["sentiment"]["score"])
        
        # Calculate weighted average
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def get_sentiment_summary(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get sentiment summary for news articles
        
        Args:
            articles: List of news articles with sentiment analysis
            
        Returns:
            Dictionary with sentiment summary
        """
        if not articles:
            return {
                "overall_sentiment": "neutral",
                "overall_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0
            }
        
        # Count sentiments
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            if "sentiment" in article and "sentiment" in article["sentiment"]:
                sentiment = article["sentiment"]["sentiment"]
                if sentiment == "positive":
                    positive_count += 1
                elif sentiment == "negative":
                    negative_count += 1
                else:
                    neutral_count += 1
        
        # Calculate overall sentiment score
        overall_score = self.calculate_news_sentiment_score(articles)
        
        # Determine overall sentiment
        if overall_score > 0.1:
            overall_sentiment = "positive"
        elif overall_score < -0.1:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "overall_score": overall_score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total_articles": len(articles)
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
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import MaxNLocator
            
            # Extract timestamps and sentiment scores
            timestamps = []
            scores = []
            
            for article in articles:
                if "sentiment" in article and "score" in article["sentiment"] and "published_at" in article:
                    try:
                        timestamp = datetime.fromisoformat(article["published_at"].replace('Z', '+00:00'))
                        timestamps.append(timestamp)
                        scores.append(article["sentiment"]["score"])
                    except (ValueError, TypeError):
                        continue
            
            if not timestamps:
                return "No valid timestamps found for visualization"
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'score': scores
            })
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Calculate rolling average
            df['rolling_avg'] = df['score'].rolling(window=5, min_periods=1).mean()
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot individual scores
            plt.scatter(df['timestamp'], df['score'], alpha=0.5, label='Individual Articles')
            
            # Plot rolling average
            plt.plot(df['timestamp'], df['rolling_avg'], 'r-', label='5-Article Rolling Average')
            
            # Add horizontal lines
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=0.1, color='g', linestyle='--', alpha=0.3)
            plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.3)
            
            # Format plot
            plt.title('Sentiment Trend Over Time')
            plt.xlabel('Date')
            plt.ylabel('Sentiment Score')
            plt.legend()
            
            # Format x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(MaxNLocator(10))
            plt.gcf().autofmt_xdate()
            
            # Set y-axis limits
            plt.ylim(-1.1, 1.1)
            
            # Save plot
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'sentiment_trend_{timestamp}.png')
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error visualizing sentiment trends: {str(e)}")
            return f"Error visualizing sentiment trends: {str(e)}"
