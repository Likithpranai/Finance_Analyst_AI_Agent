"""
Finnhub API Integration for Finance Analyst AI Agent

This module provides comprehensive financial data using Finnhub's API services.
Includes real-time quotes, company fundamentals, earnings data, analyst recommendations,
and insider trading information.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Finnhub API key
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

class FinnhubTools:
    """Tools for accessing comprehensive financial data from Finnhub API"""
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    @staticmethod
    def get_company_profile(symbol: str) -> Dict:
        """
        Get detailed company profile and metrics
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company profile data
        """
        if not FINNHUB_API_KEY:
            return {"error": "Finnhub API key not found"}
        
        try:
            # Get company profile
            profile_url = f"{FinnhubTools.BASE_URL}/stock/profile2"
            params = {"symbol": symbol, "token": FINNHUB_API_KEY}
            
            response = requests.get(profile_url, params=params)
            response.raise_for_status()
            
            profile_data = response.json()
            
            if not profile_data:
                return {"error": f"No profile data found for {symbol}"}
            
            # Format the response
            return {
                "symbol": symbol,
                "name": profile_data.get("name", "N/A"),
                "country": profile_data.get("country", "N/A"),
                "currency": profile_data.get("currency", "N/A"),
                "exchange": profile_data.get("exchange", "N/A"),
                "industry": profile_data.get("finnhubIndustry", "N/A"),
                "ipo_date": profile_data.get("ipo", "N/A"),
                "market_cap": profile_data.get("marketCapitalization", 0),
                "shares_outstanding": profile_data.get("shareOutstanding", 0),
                "website": profile_data.get("weburl", "N/A"),
                "logo": profile_data.get("logo", "N/A"),
                "phone": profile_data.get("phone", "N/A")
            }
            
        except Exception as e:
            return {"error": f"Error fetching company profile: {str(e)}"}
    
    @staticmethod
    def get_earnings_data(symbol: str) -> Dict:
        """
        Get earnings data and estimates
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with earnings data
        """
        if not FINNHUB_API_KEY:
            return {"error": "Finnhub API key not found"}
        
        try:
            # Get earnings data
            earnings_url = f"{FinnhubTools.BASE_URL}/stock/earnings"
            params = {"symbol": symbol, "token": FINNHUB_API_KEY}
            
            response = requests.get(earnings_url, params=params)
            response.raise_for_status()
            
            earnings_data = response.json()
            
            if not earnings_data:
                return {"error": f"No earnings data found for {symbol}"}
            
            # Format recent earnings
            recent_earnings = []
            for earning in earnings_data[:4]:  # Last 4 quarters
                recent_earnings.append({
                    "period": earning.get("period", "N/A"),
                    "year": earning.get("year", "N/A"),
                    "quarter": earning.get("quarter", "N/A"),
                    "actual_eps": earning.get("actual", 0),
                    "estimate_eps": earning.get("estimate", 0),
                    "surprise": earning.get("surprise", 0),
                    "surprise_percent": earning.get("surprisePercent", 0)
                })
            
            return {
                "symbol": symbol,
                "recent_earnings": recent_earnings,
                "data_source": "Finnhub"
            }
            
        except Exception as e:
            return {"error": f"Error fetching earnings data: {str(e)}"}
    
    @staticmethod
    def get_analyst_recommendations(symbol: str) -> Dict:
        """
        Get analyst recommendations and price targets
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with analyst recommendations
        """
        if not FINNHUB_API_KEY:
            return {"error": "Finnhub API key not found"}
        
        try:
            # Get recommendation trends
            rec_url = f"{FinnhubTools.BASE_URL}/stock/recommendation"
            params = {"symbol": symbol, "token": FINNHUB_API_KEY}
            
            response = requests.get(rec_url, params=params)
            response.raise_for_status()
            
            rec_data = response.json()
            
            if not rec_data:
                return {"error": f"No recommendation data found for {symbol}"}
            
            # Get the most recent recommendation
            latest_rec = rec_data[0] if rec_data else {}
            
            # Get price target
            target_url = f"{FinnhubTools.BASE_URL}/stock/price-target"
            target_response = requests.get(target_url, params=params)
            target_response.raise_for_status()
            target_data = target_response.json()
            
            return {
                "symbol": symbol,
                "period": latest_rec.get("period", "N/A"),
                "strong_buy": latest_rec.get("strongBuy", 0),
                "buy": latest_rec.get("buy", 0),
                "hold": latest_rec.get("hold", 0),
                "sell": latest_rec.get("sell", 0),
                "strong_sell": latest_rec.get("strongSell", 0),
                "target_high": target_data.get("targetHigh", 0),
                "target_low": target_data.get("targetLow", 0),
                "target_mean": target_data.get("targetMean", 0),
                "target_median": target_data.get("targetMedian", 0),
                "last_updated": target_data.get("lastUpdated", "N/A"),
                "data_source": "Finnhub"
            }
            
        except Exception as e:
            return {"error": f"Error fetching analyst recommendations: {str(e)}"}
    
    @staticmethod
    def get_insider_trading(symbol: str, days: int = 30) -> Dict:
        """
        Get insider trading data
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with insider trading data
        """
        if not FINNHUB_API_KEY:
            return {"error": "Finnhub API key not found"}
        
        try:
            # Calculate date range
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Get insider trading data
            insider_url = f"{FinnhubTools.BASE_URL}/stock/insider-transactions"
            params = {
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_API_KEY
            }
            
            response = requests.get(insider_url, params=params)
            response.raise_for_status()
            
            insider_data = response.json()
            
            if not insider_data or "data" not in insider_data:
                return {"error": f"No insider trading data found for {symbol}"}
            
            # Format insider transactions
            transactions = []
            for transaction in insider_data["data"][:10]:  # Last 10 transactions
                transactions.append({
                    "name": transaction.get("name", "N/A"),
                    "share": transaction.get("share", 0),
                    "change": transaction.get("change", 0),
                    "filing_date": transaction.get("filingDate", "N/A"),
                    "transaction_date": transaction.get("transactionDate", "N/A"),
                    "transaction_code": transaction.get("transactionCode", "N/A"),
                    "transaction_price": transaction.get("transactionPrice", 0)
                })
            
            return {
                "symbol": symbol,
                "transactions": transactions,
                "period_days": days,
                "data_source": "Finnhub"
            }
            
        except Exception as e:
            return {"error": f"Error fetching insider trading data: {str(e)}"}
    
    @staticmethod
    def get_financial_metrics(symbol: str) -> Dict:
        """
        Get comprehensive financial metrics
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with financial metrics
        """
        if not FINNHUB_API_KEY:
            return {"error": "Finnhub API key not found"}
        
        try:
            # Get basic financials
            metrics_url = f"{FinnhubTools.BASE_URL}/stock/metric"
            params = {"symbol": symbol, "metric": "all", "token": FINNHUB_API_KEY}
            
            response = requests.get(metrics_url, params=params)
            response.raise_for_status()
            
            metrics_data = response.json()
            
            if not metrics_data or "metric" not in metrics_data:
                return {"error": f"No financial metrics found for {symbol}"}
            
            metrics = metrics_data["metric"]
            
            return {
                "symbol": symbol,
                "valuation_metrics": {
                    "pe_ratio": metrics.get("peBasicExclExtraTTM", 0),
                    "pe_forward": metrics.get("peNormalizedAnnual", 0),
                    "peg_ratio": metrics.get("pegRatio", 0),
                    "price_to_book": metrics.get("pbAnnual", 0),
                    "price_to_sales": metrics.get("psAnnual", 0),
                    "ev_to_ebitda": metrics.get("evToEbitdaTTM", 0)
                },
                "profitability_metrics": {
                    "roe": metrics.get("roeTTM", 0),
                    "roa": metrics.get("roaTTM", 0),
                    "roi": metrics.get("roiTTM", 0),
                    "gross_margin": metrics.get("grossMarginTTM", 0),
                    "operating_margin": metrics.get("operatingMarginTTM", 0),
                    "net_margin": metrics.get("netProfitMarginTTM", 0)
                },
                "growth_metrics": {
                    "revenue_growth_3y": metrics.get("revenueGrowth3Y", 0),
                    "revenue_growth_5y": metrics.get("revenueGrowth5Y", 0),
                    "eps_growth_3y": metrics.get("epsGrowth3Y", 0),
                    "eps_growth_5y": metrics.get("epsGrowth5Y", 0)
                },
                "financial_strength": {
                    "current_ratio": metrics.get("currentRatioAnnual", 0),
                    "debt_to_equity": metrics.get("totalDebtToEquityAnnual", 0),
                    "debt_to_assets": metrics.get("totalDebtToTotalCapitalAnnual", 0),
                    "interest_coverage": metrics.get("interestCoverageAnnual", 0)
                },
                "data_source": "Finnhub"
            }
            
        except Exception as e:
            return {"error": f"Error fetching financial metrics: {str(e)}"}
    
    @staticmethod
    def get_market_news(symbol: str = None, limit: int = 10) -> Dict:
        """
        Get market news from Finnhub
        
        Args:
            symbol: Stock ticker symbol (optional)
            limit: Maximum number of news items
            
        Returns:
            Dictionary with news data
        """
        if not FINNHUB_API_KEY:
            return {"error": "Finnhub API key not found"}
        
        try:
            if symbol:
                # Get company news
                news_url = f"{FinnhubTools.BASE_URL}/company-news"
                from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                to_date = datetime.now().strftime("%Y-%m-%d")
                
                params = {
                    "symbol": symbol,
                    "from": from_date,
                    "to": to_date,
                    "token": FINNHUB_API_KEY
                }
            else:
                # Get general market news
                news_url = f"{FinnhubTools.BASE_URL}/news"
                params = {
                    "category": "general",
                    "token": FINNHUB_API_KEY
                }
            
            response = requests.get(news_url, params=params)
            response.raise_for_status()
            
            news_data = response.json()
            
            if not news_data:
                return {"error": "No news data found"}
            
            # Format news items
            news_items = []
            for item in news_data[:limit]:
                news_items.append({
                    "headline": item.get("headline", "No headline"),
                    "summary": item.get("summary", "No summary"),
                    "url": item.get("url", ""),
                    "image": item.get("image", ""),
                    "datetime": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                    "source": item.get("source", "Finnhub"),
                    "category": item.get("category", "general")
                })
            
            return {
                "symbol": symbol if symbol else "market",
                "news": news_items,
                "data_source": "Finnhub"
            }
            
        except Exception as e:
            return {"error": f"Error fetching news: {str(e)}"}
