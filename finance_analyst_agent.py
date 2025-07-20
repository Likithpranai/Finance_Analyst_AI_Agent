import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import base64
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import re
import traceback
import time
from typing import Dict, List, Any, Tuple, Optional, Union

# Import existing tools
from tools.alpha_vantage_tools import AlphaVantageTools
from tools.technical_analysis import TechnicalAnalysisTools
from tools.fundamental_analysis import FundamentalAnalysisTools
from tools.alpha_vantage_tools import AlphaVantageTools
from tools.real_time_data_integration import RealTimeDataTools
from tools.combined_analysis import CombinedAnalysisTools
from tools.portfolio_management import PortfolioManagementTools
from tools.predictive_analytics import PredictiveAnalyticsTools
from tools.polygon_integration import PolygonIntegrationTools
from tools.news_retrieval import NewsRetrievalTools
from tools.portfolio_integration import PortfolioIntegrationTools
from tools.enhanced_visualization import EnhancedVisualizationTools
from tools.backtesting import BacktestingTools
from tools.backtesting_visualization import BacktestVisualizationTools

# Import new professional-grade tools
from tools.polygon_integration import PolygonIntegrationTools
from tools.websocket_manager import websocket_manager
from tools.cache_manager import CacheManager
from tools.real_time_data_integration import RealTimeDataTools

load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model - try multiple model names and versions
model = None
model_names = ['gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro']

for model_name in model_names:
    try:
        model = genai.GenerativeModel(model_name)
        print(f"Successfully initialized Gemini model: {model_name}")
        break  # Exit the loop if successful
    except Exception as e:
        print(f"Could not initialize model {model_name}: {str(e)}")

if model is None:
    print("Warning: Could not initialize any Gemini model. Will use raw data without AI analysis.")
    print("Please check your API key and ensure you have access to Gemini models.")
    print("You can still use the agent's tools for financial analysis.")


# Define stock analysis tools
class StockTools:
    @staticmethod
    def get_stock_price(symbol):
        """Get current stock price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if data.empty:
                return f"Could not find data for {symbol}"
            
            current_price = data['Close'].iloc[-1]
            return f"{symbol} current price: ${current_price:.2f}"
        except Exception as e:
            return f"Error fetching stock price for {symbol}: {str(e)}"
    
    @staticmethod
    def get_stock_history(symbol, period="1mo", return_df=False):
        """Get historical stock data
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            return_df: If True, returns the DataFrame directly instead of formatted string
            
        Returns:
            Formatted string with historical data summary or DataFrame if return_df=True
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                return f"Could not find historical data for {symbol}" if not return_df else None
            
            # Return DataFrame if requested
            if return_df:
                return data
            
            # Format the data for display
            result = f"Historical data for {symbol} (last {period}):\n"
            result += f"Start: ${data['Close'].iloc[0]:.2f}, End: ${data['Close'].iloc[-1]:.2f}\n"
            result += f"High: ${data['High'].max():.2f}, Low: ${data['Low'].min():.2f}\n"
            result += f"Volume: {data['Volume'].mean():.0f} (avg)"
            return result
        except Exception as e:
            error_msg = f"Error fetching historical data for {symbol}: {str(e)}"
            return error_msg if not return_df else None
    
    @staticmethod
    def calculate_rsi(symbol, period=14):
        """Calculate RSI for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo")
            if data.empty:
                return f"Could not find data for {symbol}"
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # Interpret RSI
            if current_rsi > 70:
                interpretation = "overbought"
            elif current_rsi < 30:
                interpretation = "oversold"
            else:
                interpretation = "neutral"
                
            return f"RSI(14) for {symbol}: {current_rsi:.2f} - {interpretation}"
        except Exception as e:
            return f"Error calculating RSI for {symbol}: {str(e)}"
            
    @staticmethod
    def calculate_macd(symbol):
        """Calculate MACD for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            if data.empty:
                return f"Could not find data for {symbol}"
            
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # Get the latest values
            latest_macd = macd.iloc[-1]
            latest_signal = signal.iloc[-1]
            latest_histogram = histogram.iloc[-1]
            
            # Interpret MACD
            if latest_macd > latest_signal:
                trend = "bullish"
            else:
                trend = "bearish"
                
            if latest_histogram > 0 and latest_histogram > histogram.iloc[-2]:
                strength = "strengthening"
            elif latest_histogram > 0 and latest_histogram < histogram.iloc[-2]:
                strength = "weakening but still positive"
            elif latest_histogram < 0 and latest_histogram < histogram.iloc[-2]:
                strength = "weakening"
            else:
                strength = "improving but still negative"
                
            result = f"MACD Analysis for {symbol}:\n"
            result += f"MACD Line: {latest_macd:.4f}\n"
            result += f"Signal Line: {latest_signal:.4f}\n"
            result += f"Histogram: {latest_histogram:.4f}\n"
            result += f"Trend: {trend.capitalize()} ({strength})"
            
            return result
        except Exception as e:
            return f"Error calculating MACD for {symbol}: {str(e)}"
            
    @staticmethod
    def calculate_obv(symbol):
        """Calculate On-Balance Volume (OBV) for a stock"""
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            # Basic validation
            if data is None or data.empty:
                return f"Could not find data for {symbol}"
            
            # Check if data has required columns
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                return f"Missing required price or volume data for {symbol}"
            
            # Ensure data is clean
            data = data.dropna(subset=['Close', 'Volume'])
            
            # Calculate OBV using the TechnicalAnalysisTools
            obv = TechnicalAnalysisTools.calculate_obv(data)
            
            # Validate OBV result
            if obv is None or len(obv) == 0:
                return f"Failed to calculate OBV for {symbol}"
            
            # Get the latest values and calculate change
            latest_obv = obv.iloc[-1]
            obv_change = 0
            if len(obv) >= 20 and obv.iloc[-20] != 0:
                obv_change = ((obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20])) * 100
            
            # Interpret OBV
            if obv_change > 5:
                trend = "strong bullish"
            elif obv_change > 0:
                trend = "moderately bullish"
            elif obv_change < -5:
                trend = "strong bearish"
            else:
                trend = "moderately bearish"
                
            result = f"On-Balance Volume (OBV) Analysis for {symbol}:\n"
            result += f"Current OBV: {latest_obv:.0f}\n"
            result += f"OBV 20-day Change: {obv_change:.2f}%\n"
            result += f"Volume Trend: {trend.capitalize()}"
            
            return result
        except Exception as e:
            return f"Error calculating OBV for {symbol}: {str(e)}"
            
    @staticmethod
    def calculate_adline(symbol):
        """Calculate Accumulation/Distribution Line for a stock"""
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            # Basic validation
            if data is None or data.empty:
                return f"Could not find data for {symbol}"
            
            # Check if data has required columns
            required_columns = ['High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                return f"Missing required price or volume data for {symbol}"
            
            # Ensure data is clean
            data = data.dropna(subset=required_columns)
            
            # Calculate A/D Line using the TechnicalAnalysisTools
            ad_line = TechnicalAnalysisTools.calculate_adline(data)
            
            # Validate A/D Line result
            if ad_line is None or len(ad_line) == 0:
                return f"Failed to calculate A/D Line for {symbol}"
            
            # Get the latest values and calculate change
            latest_ad = ad_line.iloc[-1]
            ad_change = 0
            if len(ad_line) >= 20 and ad_line.iloc[-20] != 0:
                ad_change = ((ad_line.iloc[-1] - ad_line.iloc[-20]) / abs(ad_line.iloc[-20])) * 100
            
            # Interpret A/D Line
            if ad_change > 5:
                trend = "strong accumulation"
            elif ad_change > 0:
                trend = "moderate accumulation"
            elif ad_change < -5:
                trend = "strong distribution"
            else:
                trend = "moderate distribution"
                
            result = f"Accumulation/Distribution Line Analysis for {symbol}:\n"
            result += f"Current A/D Line: {latest_ad:.2f}\n"
            result += f"A/D Line 20-day Change: {ad_change:.2f}%\n"
            result += f"Money Flow Trend: {trend.capitalize()}"
            
            return result
        except Exception as e:
            return f"Error calculating A/D Line for {symbol}: {str(e)}"
            
    @staticmethod
    def calculate_adx(symbol, window=14):
        """Calculate Average Directional Index (ADX) for a stock"""
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            # Basic validation
            if data is None or data.empty:
                return f"Could not find data for {symbol}"
                
            # Check if we have enough data points for calculation
            if len(data) < window * 2:
                return f"Insufficient data points for ADX calculation for {symbol}. Need at least {window * 2} data points."
            
            # Check if data has required columns
            required_columns = ['High', 'Low', 'Close']
            if not all(col in data.columns for col in required_columns):
                return f"Missing required price data for {symbol}"
            
            # Ensure data is clean
            data = data.dropna(subset=required_columns)
            
            # Calculate ADX using the TechnicalAnalysisTools
            adx_result = TechnicalAnalysisTools.calculate_adx(data, window)
            
            # Check if ADX calculation was successful
            if not isinstance(adx_result, dict) or not all(key in adx_result for key in ['ADX', '+DI', '-DI']):
                return f"Error calculating ADX for {symbol}: invalid result format"
                
            adx = adx_result['ADX']
            plus_di = adx_result['+DI']
            minus_di = adx_result['-DI']
            
            # Get the latest values (with null checks)
            if adx.empty or plus_di.empty or minus_di.empty:
                return f"Insufficient data to calculate ADX for {symbol}"
                
            # Handle NaN values
            latest_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
            latest_plus_di = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0
            latest_minus_di = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0
            
            # Interpret ADX
            if latest_adx > 25:
                strength = "strong"
            elif latest_adx > 20:
                strength = "moderate"
            else:
                strength = "weak"
                
            if latest_plus_di > latest_minus_di:
                trend = "bullish"
            else:
                trend = "bearish"
                
            result = f"Average Directional Index (ADX) Analysis for {symbol}:\n"
            result += f"ADX: {latest_adx:.2f} - {strength} trend\n"
            result += f"+DI: {latest_plus_di:.2f}\n"
            result += f"-DI: {latest_minus_di:.2f}\n"
            result += f"Trend Direction: {trend.capitalize()}"
            
            return result
        except Exception as e:
            return f"Error calculating ADX for {symbol}: {str(e)}"
    
    @staticmethod
    def get_company_info(symbol):
        """Get company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            name = info.get('longName', 'N/A')
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                market_cap = f"${market_cap/1e9:.2f}B"
            
            pe_ratio = info.get('trailingPE', 'N/A')
            if pe_ratio != 'N/A':
                pe_ratio = f"{pe_ratio:.2f}"
                
            dividend_yield = info.get('dividendYield', 'N/A')
            if dividend_yield != 'N/A':
                dividend_yield = f"{dividend_yield*100:.2f}%"
            
            result = f"Company Information for {symbol} ({name}):\n"
            result += f"Sector: {sector}\n"
            result += f"Industry: {industry}\n"
            result += f"Market Cap: {market_cap}\n"
            result += f"P/E Ratio: {pe_ratio}\n"
            result += f"Dividend Yield: {dividend_yield}"
            return result
        except Exception as e:
            return f"Error fetching company info for {symbol}: {str(e)}"
    
    @staticmethod
    def get_stock_news(symbol, max_items=5):
        """Get latest news for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return f"No recent news found for {symbol}"
            
            # Format the news items
            result = f"Latest News for {symbol}:\n\n"
            
            # Limit to max_items
            for i, item in enumerate(news[:max_items], 1):
                title = item.get('title', 'No title')
                publisher = item.get('publisher', 'Unknown source')
                link = item.get('link', '#')
                publish_time = item.get('providerPublishTime', 0)
                
                # Convert timestamp to readable date
                if publish_time:
                    date = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                else:
                    date = 'Unknown date'
                
                result += f"{i}. {title}\n"
                result += f"   Source: {publisher} | {date}\n"
                result += f"   Link: {link}\n\n"
            
            return result
        except Exception as e:
            return f"Error fetching news for {symbol}: {str(e)}"
    
    @staticmethod
    def visualize_stock(symbol, period="6mo", indicators=None):
        """Create a visualization of stock data with technical indicators"""
        try:
            if indicators is None:
                indicators = ["sma", "rsi"]  # Default indicators
                
            # Get stock data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return f"Could not find data for {symbol}"
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(f"{symbol} Stock Analysis - {period}", fontsize=16)
            
            # Main price plot
            axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
            axes[0].set_ylabel('Price')
            axes[0].grid(True, alpha=0.3)
            
            # Add volume to bottom subplot
            axes[1].bar(data.index, data['Volume'], label='Volume', color='gray', alpha=0.5)
            axes[1].set_ylabel('Volume')
            axes[1].grid(True, alpha=0.3)
            
            # Add technical indicators
            if "sma" in indicators:
                # Add 50-day and 200-day SMA
                sma50 = data['Close'].rolling(window=50).mean()
                sma200 = data['Close'].rolling(window=200).mean()
                axes[0].plot(data.index, sma50, label='50-day SMA', color='orange', alpha=0.7)
                axes[0].plot(data.index, sma200, label='200-day SMA', color='red', alpha=0.7)
            
            if "bollinger" in indicators:
                # Add Bollinger Bands
                sma20 = data['Close'].rolling(window=20).mean()
                std20 = data['Close'].rolling(window=20).std()
                upper_band = sma20 + (std20 * 2)
                lower_band = sma20 - (std20 * 2)
                axes[0].plot(data.index, upper_band, label='Upper BB', color='green', linestyle='--', alpha=0.7)
                axes[0].plot(data.index, sma20, label='20-day SMA', color='purple', alpha=0.7)
                axes[0].plot(data.index, lower_band, label='Lower BB', color='green', linestyle='--', alpha=0.7)
                axes[0].fill_between(data.index, lower_band, upper_band, color='green', alpha=0.05)
            
            if "rsi" in indicators:
                # Calculate and plot RSI in a separate subplot
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Add RSI as a line on the volume subplot
                ax_rsi = axes[1].twinx()
                ax_rsi.plot(data.index, rsi, label='RSI', color='purple', alpha=0.7)
                ax_rsi.set_ylabel('RSI')
                ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax_rsi.set_ylim(0, 100)
            
            # Add legends
            axes[0].legend(loc='upper left')
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            # Convert to base64 for display
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Return a message with the chart data
            return f"Generated visualization for {symbol} with {', '.join(indicators)} indicators.\n[Chart data available but not displayed in text format]\n\nAnalysis period: {period}\nLatest price: ${data['Close'].iloc[-1]:.2f}\nPrice change: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}%"
        except Exception as e:
            return f"Error visualizing data for {symbol}: {str(e)}"

# Define the main agent class using ReAct pattern
class FinanceAnalystReActAgent:
    def __init__(self):
        self.tools = {
            # Standard yfinance tools
            "get_stock_price": StockTools.get_stock_price,
            "get_stock_history": StockTools.get_stock_history,
            "calculate_rsi": StockTools.calculate_rsi,
            "calculate_macd": StockTools.calculate_macd,
            "get_company_info": StockTools.get_company_info,
            "get_stock_news": StockTools.get_stock_news,
            "visualize_stock": StockTools.visualize_stock,
            
            # Professional Real-Time Data Tools (Polygon.io with fallback)
            "get_real_time_quote": RealTimeDataTools.get_real_time_quote,
            "get_intraday_data": RealTimeDataTools.get_intraday_data,
            "get_historical_data": RealTimeDataTools.get_historical_data,
            "get_crypto_data": RealTimeDataTools.get_crypto_data,
            "get_forex_data": RealTimeDataTools.get_forex_data,
            "get_company_details": RealTimeDataTools.get_company_details,
            "get_market_news": NewsRetrievalTools.get_market_news,
            "start_real_time_stream": RealTimeDataTools.start_real_time_stream,
            "stop_real_time_stream": RealTimeDataTools.stop_real_time_stream,
            "get_active_streams": RealTimeDataTools.get_active_streams,
            "clear_cache": RealTimeDataTools.clear_cache,
            "get_cache_stats": RealTimeDataTools.get_cache_stats,
            
            # Legacy Alpha Vantage tools (kept for compatibility)
            "get_alpha_vantage_quote": AlphaVantageTools.get_real_time_quote,
            "get_alpha_vantage_intraday": AlphaVantageTools.get_intraday_data,
            "get_alpha_vantage_crypto": AlphaVantageTools.get_crypto_data,
            "get_alpha_vantage_forex": AlphaVantageTools.get_forex_data,
            "get_economic_indicator": AlphaVantageTools.get_economic_indicator,
            
            # Predictive Analytics tools
            "forecast_with_prophet": PredictiveAnalyticsTools.forecast_with_prophet,
            "forecast_with_lstm": PredictiveAnalyticsTools.forecast_with_lstm,
            "detect_anomalies": PredictiveAnalyticsTools.detect_anomalies,
            "calculate_volatility": PredictiveAnalyticsTools.calculate_volatility,
            "scenario_analysis": PredictiveAnalyticsTools.scenario_analysis,
            "check_stationarity": PredictiveAnalyticsTools.check_stationarity,
            
            # Additional Technical Indicators
            "calculate_obv": StockTools.calculate_obv,
            "calculate_adline": StockTools.calculate_adline,
            "calculate_adx": StockTools.calculate_adx,
            
            # Fundamental Analysis Tools
            "get_financial_ratios": FundamentalAnalysisTools.get_financial_ratios,
            "get_income_statement": FundamentalAnalysisTools.get_income_statement,
            "get_balance_sheet": FundamentalAnalysisTools.get_balance_sheet,
            "get_cash_flow": FundamentalAnalysisTools.get_cash_flow,
            "get_income_statement_summary": FundamentalAnalysisTools.get_income_statement_summary,
            "get_balance_sheet_summary": FundamentalAnalysisTools.get_balance_sheet_summary,
            "format_financial_ratios_for_display": FundamentalAnalysisTools.format_financial_ratios_for_display,
            "get_industry_comparison": FundamentalAnalysisTools.get_industry_comparison,
            
            # Enhanced Visualization tools
            "visualize_financial_trends": EnhancedVisualizationTools.visualize_financial_trends,
            "create_correlation_matrix": EnhancedVisualizationTools.create_correlation_matrix,
            "compare_performance": EnhancedVisualizationTools.visualize_performance_comparison,
            "visualize_financial_ratios": EnhancedVisualizationTools.visualize_financial_ratios,
            
            # Combined Analysis tools
            "create_combined_analysis": CombinedAnalysisTools.create_combined_analysis,
            "format_combined_analysis": CombinedAnalysisTools.format_combined_analysis,
            
            # Portfolio Management tools
            "calculate_risk_metrics": PortfolioManagementTools.calculate_risk_metrics,
            "optimize_portfolio": PortfolioManagementTools.optimize_portfolio,
            "generate_efficient_frontier": PortfolioManagementTools.generate_efficient_frontier,
            "visualize_portfolio": PortfolioManagementTools.visualize_portfolio,
            
            # Backtesting tools
            "backtest_sma_crossover": BacktestingTools.backtest_sma_crossover,
            "backtest_rsi_strategy": BacktestingTools.backtest_rsi_strategy,
            "backtest_macd_strategy": BacktestingTools.backtest_macd_strategy,
            "visualize_backtest_results": BacktestVisualizationTools.visualize_backtest_results,
            "paper_trading_simulation": BacktestVisualizationTools.paper_trading_simulation,
            
            # Portfolio Integration tools
            "analyze_portfolio": PortfolioIntegrationTools.analyze_portfolio,
            "backtest_strategy": PortfolioIntegrationTools.backtest_strategy,
            "run_paper_trading": PortfolioIntegrationTools.run_paper_trading
        }
        
        self.system_prompt = """
        You are a Finance Analyst AI that helps users analyze stocks and provide financial insights.
        You follow the ReAct pattern: Reason → Act → Observe → Loop.
        
        You have access to the following tools:
        
        STANDARD TOOLS (YFINANCE):
        1. get_stock_price(symbol): Get the current price of a stock
        2. get_stock_history(symbol, period): Get historical data for a stock (period can be 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        3. calculate_rsi(symbol, period): Calculate the RSI technical indicator (default period is 14)
        4. calculate_macd(symbol): Calculate the MACD technical indicator
        5. get_company_info(symbol): Get company information including sector, industry, market cap, P/E ratio, and dividend yield
        6. get_stock_news(symbol, max_items): Get the latest news articles about a stock
        7. visualize_stock(symbol, period, indicators): Create a visualization of stock data with technical indicators
        
        PROFESSIONAL REAL-TIME DATA TOOLS (POLYGON.IO WITH FALLBACK):
        8. get_real_time_quote(symbol): Get real-time stock quote with sub-second latency
        9. get_intraday_data(symbol, interval, limit): Get intraday data with low latency (intervals: 1min, 5min, 15min, 30min, 60min)
        10. get_historical_data(symbol, period): Get historical daily data with optimized performance
        11. get_crypto_data(symbol, market, interval): Get real-time cryptocurrency data (e.g., BTC/USD)
        12. get_forex_data(from_currency, to_currency, interval): Get real-time forex exchange rate data
        13. get_company_details(symbol): Get detailed company information including financials and metrics
        14. get_market_news(symbol, limit): Get latest news for a symbol or general market news
        15. start_real_time_stream(symbols, callback, asset_type): Start a WebSocket stream for real-time updates
        16. stop_real_time_stream(stream_id): Stop a WebSocket stream
        17. get_active_streams(): Get information about active streams
        18. clear_cache(pattern): Clear cache entries to refresh data
        19. get_cache_stats(): Get cache statistics
        
        ECONOMIC INDICATORS:
        20. get_economic_indicator(indicator): Get economic indicators (GDP, INFLATION, UNEMPLOYMENT, etc.)
        
        PREDICTIVE ANALYTICS TOOLS:
        13. forecast_with_prophet(historical_data, periods): Generate time series forecasts using Facebook Prophet
        14. forecast_with_lstm(historical_data, target_column, sequence_length, forecast_periods): Generate forecasts using LSTM neural networks
        15. detect_anomalies(time_series, contamination): Detect anomalies in time series data using Isolation Forest
        16. calculate_volatility(historical_data, price_column, window_size): Calculate historical volatility and forecast future volatility
        17. scenario_analysis(historical_data, price_column, scenarios, forecast_periods): Perform scenario analysis for different market conditions
        18. check_stationarity(time_series): Check if a time series is stationary using the Augmented Dickey-Fuller test
        
        ADDITIONAL TECHNICAL INDICATORS:
        19. calculate_obv(data): Calculate On-Balance Volume (OBV) - measures buying/selling pressure by adding volume on up days and subtracting on down days
        20. calculate_adline(data): Calculate Accumulation/Distribution Line - tracks money flow into or out of a security
        21. calculate_adx(data, window): Calculate Average Directional Index (ADX) - quantifies trend strength on a scale of 0-100
        
        FUNDAMENTAL ANALYSIS TOOLS:
        22. get_financial_ratios(symbol): Get key financial ratios including P/E, PEG, P/S, P/B, D/E, ROE, and EPS
        23. get_income_statement(symbol, period): Get income statement data (revenue, expenses, profits)
        24. get_balance_sheet(symbol, period): Get balance sheet data (assets, liabilities, equity)
        25. get_cash_flow(symbol, period): Get cash flow statement data (operating, investing, financing activities)
        26. format_financial_ratios(ratios): Format financial ratios for display
        27. get_industry_comparison(symbol, ratios): Compare a stock's financial ratios to industry averages
        
        ENHANCED VISUALIZATION TOOLS:
        28. visualize_financial_trends(symbol, period, chart_types): Create interactive visualizations of financial trends
        29. create_correlation_matrix(symbols, period): Generate correlation matrix heatmap for multiple stocks
        30. compare_performance(symbols, benchmark, period): Compare stock performance against benchmarks
        31. visualize_financial_ratios(symbol, peer_symbols): Create visualizations of financial ratios with peer comparisons
        
        COMBINED ANALYSIS TOOLS:
        32. create_combined_analysis(symbol, period): Generate comprehensive analysis combining technical and fundamental data
        33. format_combined_analysis(analysis): Format combined analysis results for display
        
        PORTFOLIO MANAGEMENT TOOLS:
        34. calculate_risk_metrics(symbols, weights, period): Calculate portfolio risk metrics including VaR, Sharpe ratio, beta, etc.
        35. optimize_portfolio(symbols, objective, constraints, period): Optimize portfolio weights based on objective (sharpe, min_volatility, max_return)
        36. generate_efficient_frontier(symbols, num_portfolios, period): Generate efficient frontier by simulating multiple portfolios
        37. visualize_portfolio(portfolio_data): Create visualizations of portfolio allocation, risk metrics, efficient frontier, etc.
        
        BACKTESTING TOOLS:
        38. backtest_sma_crossover(symbol, short_window, long_window, period, initial_capital): Backtest SMA crossover strategy
        39. backtest_rsi_strategy(symbol, rsi_period, overbought, oversold, period, initial_capital): Backtest RSI strategy
        40. backtest_macd_strategy(symbol, fast_period, slow_period, signal_period, period, initial_capital): Backtest MACD strategy
        41. visualize_backtest_results(backtest_data): Create visualizations of backtest results including performance charts
        42. paper_trading_simulation(symbol, strategy, parameters, initial_capital, start_date, end_date): Run paper trading simulation
        
        PORTFOLIO INTEGRATION TOOLS:
        43. analyze_portfolio(query): Analyze portfolio based on user query (extract symbols, weights, and perform analysis)
        44. backtest_strategy(query): Backtest trading strategy based on user query (extract symbol, strategy type, and parameters)
        45. run_paper_trading(query): Run paper trading simulation based on user query
        
        When a user asks a question about a stock, you should follow the ReAct pattern:
        1. REASON: Think about what information is needed to answer the query
        2. ACT: Select and execute the appropriate tool(s)
        3. OBSERVE: Analyze the results from the tool(s)
        4. LOOP: If needed, use additional tools based on initial observations
        
        IMPORTANT GUIDELINES:
        - For professional real-time trading analysis, use Polygon.io tools (8-19) for sub-second latency
        - For streaming real-time data, use start_real_time_stream to create WebSocket connections
        - For cryptocurrency queries, use get_crypto_data with Polygon.io for professional-grade data
        - For forex exchange rates, use get_forex_data with Polygon.io for low-latency quotes
        - For economic context, use get_economic_indicator
        - For standard historical analysis, use get_historical_data with caching for optimized performance
        - For price predictions and forecasting, use forecast_with_prophet or forecast_with_lstm (13-14)
        - For volatility analysis and risk assessment, use calculate_volatility (16)
        - For detecting unusual price movements, use detect_anomalies (15)
        - For multiple future scenarios (bull/bear/base cases), use scenario_analysis (17)
        - For time series statistical properties, use check_stationarity (18)
        - For advanced financial trend visualizations, use visualize_financial_trends (25)
        - For analyzing relationships between multiple stocks, use create_correlation_matrix (26)
        - For benchmark comparisons, use compare_performance (27)
        - For visualizing financial ratios with peer comparisons, use visualize_financial_ratios (28)
        - For comprehensive stock analysis combining technical and fundamental factors, use create_combined_analysis (29)
        - For portfolio analysis and risk metrics, use analyze_portfolio (40)
        - For portfolio optimization and efficient frontier, use analyze_portfolio with optimization keywords (40)
        - For backtesting trading strategies, use backtest_strategy (41)
        - For paper trading simulation, use run_paper_trading (42)
        
        Always format your response in a clear, professional manner with sections for:
        - Summary: A brief overview of the findings
        - Analysis: Detailed explanation of the data
        - Recommendation: Suggested actions or insights based on the data
        
        DO NOT make up information. Only use the data provided by the tools.
        """
    
    def extract_stock_symbol(self, query):
        """Extract stock symbol from user query"""
        # Detect asset type
        asset_type = self._detect_asset_type(query)
        query_lower = query.lower()
        
        # Handle crypto symbols (BTC, ETH, etc.)
        if asset_type == "crypto":
            crypto_pattern = r'\b(BTC|ETH|XRP|LTC|BCH|ADA|DOT|LINK|XLM|DOGE|SOL|USDT|BNB)\b'
            crypto_matches = re.findall(crypto_pattern, query.upper())
            if crypto_matches:
                return crypto_matches[0]
                
            # Check for written out crypto names
            crypto_names = {
                "bitcoin": "BTC",
                "ethereum": "ETH",
                "ripple": "XRP",
                "litecoin": "LTC",
                "cardano": "ADA",
                "polkadot": "DOT",
                "chainlink": "LINK",
                "stellar": "XLM",
                "dogecoin": "DOGE",
                "solana": "SOL",
                "tether": "USDT",
                "binance": "BNB"
            }
            
            for name, symbol in crypto_names.items():
                if name in query_lower:
                    return symbol
            
            # Default crypto
            return "BTC"
        
        # Handle forex pairs (EUR/USD, etc.)
        elif asset_type == "forex":
            # Try to find currency pairs
            forex_pattern = r'\b([A-Z]{3})/([A-Z]{3})\b'
            forex_matches = re.findall(forex_pattern, query.upper())
            if forex_matches:
                return forex_matches[0]  # Returns tuple (from_currency, to_currency)
                
            # Check for common currency names
            currency_dict = {
                "dollar": "USD",
                "euro": "EUR",
                "pound": "GBP",
                "yen": "JPY",
                "yuan": "CNY",
                "franc": "CHF",
                "canadian": "CAD",
                "aussie": "AUD",
                "australian": "AUD",
                "kiwi": "NZD",
                "rupee": "INR"
            }
            
            found_currencies = []
            for name, code in currency_dict.items():
                if name in query_lower:
                    found_currencies.append(code)
            
            if len(found_currencies) >= 2:
                return (found_currencies[0], found_currencies[1])  # Return as tuple
            
            # Default forex pair
            return ("EUR", "USD")
        
        # Handle stocks (default)
        else:
            # Try to find stock symbols using regex pattern for uppercase 1-5 letter words
            pattern = r'\b([A-Z]{1,5})\b'
            matches = re.findall(pattern, query)
            
            # Filter out common non-ticker uppercase words
            common_words = ['I', 'A', 'THE', 'FOR', 'AND', 'OR', 'RSI', 'MACD', 'EMA', 'SMA', 'GDP', 'CPI', 'USD', 'EUR', 'OBV', 'ADX', 'AD']
            ticker_candidates = [word for word in matches if word not in common_words]
            
            if ticker_candidates:
                return ticker_candidates[0]
            
            # Check for company names
            company_names = {
                "apple": "AAPL",
                "microsoft": "MSFT",
                "amazon": "AMZN",
                "google": "GOOGL",
                "alphabet": "GOOGL",
                "facebook": "META",
                "meta": "META",
                "tesla": "TSLA",
                "nvidia": "NVDA",
                "netflix": "NFLX"
            }
            
            for name, symbol in company_names.items():
                if name in query_lower:
                    return symbol
            
            # Default to AAPL if no symbol found
            return "AAPL"

    def determine_tools_needed(self, query):
        """Determine which financial tools are needed based on the query"""
        query = query.lower()
        tools_needed = []
        
        # Detect asset type and whether real-time data is needed
        asset_type = self._detect_asset_type(query)
        needs_real_time = self._needs_real_time_data(query)
        
        # Check for portfolio management needs
        if any(term in query for term in ["portfolio", "allocation", "risk metrics", "sharpe", "var", "efficient frontier", "optimize"]):
            tools_needed.append("analyze_portfolio")
            return tools_needed
        
        # Check for backtesting needs
        if any(term in query for term in ["backtest", "strategy", "sma crossover", "rsi strategy", "macd strategy"]):
            tools_needed.append("backtest_strategy")
            return tools_needed
        
        # Check for paper trading simulation needs
        if any(term in query for term in ["paper trading", "simulate trading", "trading simulation"]):
            tools_needed.append("run_paper_trading")
            return tools_needed
        
        # Check for real-time stock price information with professional-grade data
        if any(term in query for term in ["price", "worth", "value", "cost", "current", "how much", "quote", "tick", "bid", "ask"]):
            if asset_type == "stock":
                # Always use professional real-time quotes for current prices
                tools_needed.append("get_real_time_quote")
            elif asset_type == "crypto":
                tools_needed.append("get_crypto_data")
            elif asset_type == "forex":
                tools_needed.append("get_forex_data")
        
        # Check for intraday/real-time analysis needs
        if asset_type == "stock" and any(term in query for term in ["intraday", "minute", "hourly", "today's", "day trading", "scalping", "high frequency", "tick data", "level 2", "order book"]):
            tools_needed.append("get_intraday_data")
            
        # Check for real-time streaming needs
        if any(term in query for term in ["stream", "real-time updates", "live data", "continuous updates", "websocket", "streaming"]):
            tools_needed.append("start_real_time_stream")
        
        # Check for technical analysis needs
        if any(term in query for term in ["technical", "indicator", "analysis"]):
            if not needs_real_time:
                tools_needed.append("calculate_rsi")
                tools_needed.append("calculate_macd")
        
        # Check for specific technical indicators
        if any(term in query for term in ["rsi", "relative strength", "overbought", "oversold"]):
            tools_needed.append("calculate_rsi")
        
        if any(term in query for term in ["macd", "moving average convergence", "divergence"]):
            tools_needed.append("calculate_macd")
            
        # Check for new technical indicators
        if any(term in query for term in ["obv", "on-balance volume", "on balance volume", "volume price"]):
            tools_needed.append("calculate_obv")
            
        if any(term in query for term in ["a/d line", "adline", "accumulation distribution", "accumulation/distribution"]):
            tools_needed.append("calculate_adline")
            
        if any(term in query for term in ["adx", "average directional", "directional movement", "trend strength"]):
            tools_needed.append("calculate_adx")
        
        # Check for historical data needs with optimized performance
        if any(term in query for term in ["history", "historical", "trend", "past", "performance", "chart", "graph"]):
            if asset_type == "stock":
                # Use the new professional-grade historical data tool with caching
                tools_needed.append("get_historical_data")
            elif asset_type == "crypto":
                tools_needed.append("get_crypto_data")
            elif asset_type == "forex":
                tools_needed.append("get_forex_data")
        
        # Check for company information needs with enhanced details
        if asset_type == "stock" and any(term in query for term in ["info", "information", "about", "company", "details", "fundamentals", "profile", "executives", "market cap", "sector"]):
            # Use the new professional-grade company details tool
            tools_needed.append("get_company_details")
            
        # Check for fundamental analysis needs
        if asset_type == "stock" and any(term in query for term in ["fundamental", "financials", "ratios", "pe", "eps", "revenue", "earnings", "profit", "margin", "debt", "equity", "book value", "roe", "roa", "comprehensive"]):
            # Add fundamental analysis tools
            tools_needed.append("get_financial_ratios")
            tools_needed.append("get_income_statement_summary")
            tools_needed.append("get_balance_sheet_summary")
        
        # Check for news needs with professional-grade sources
        if asset_type == "stock" and any(term in query for term in ["news", "headlines", "articles", "press", "announcement", "media", "release"]):
            # Use the new professional-grade news tool
            tools_needed.append("get_market_news")
        
        # Check for visualization needs
        if asset_type == "stock" and any(term in query for term in ["chart", "graph", "plot", "visual", "picture", "show me"]):
            tools_needed.append("visualize_stock")
        
        # If nothing specific was detected, get appropriate default info based on asset type
        if not tools_needed:
            if asset_type == "stock":
                if needs_real_time:
                    tools_needed = ["get_real_time_quote", "get_company_info"]
                else:
                    # For stocks, include comprehensive analysis by default
                    tools_needed = ["get_stock_price", "get_company_info", "calculate_rsi", "calculate_macd", "get_financial_ratios", "get_market_news"]
            elif asset_type == "crypto":
                tools_needed = ["get_crypto_data"]
            elif asset_type == "forex":
                tools_needed = ["get_forex_data"]
            else:
                tools_needed = ["get_stock_price", "get_company_info"]
        
        return tools_needed
    
    def _detect_asset_type(self, query):
        """Detect the type of asset being queried (stock, crypto, forex)"""
        query = query.lower()
        
        # Check for cryptocurrency keywords
        crypto_keywords = ["crypto", "bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "token", "coin"]
        if any(keyword in query for keyword in crypto_keywords):
            return "crypto"
        
        # Check for forex keywords
        forex_keywords = ["forex", "currency", "exchange rate", "eur/usd", "usd/jpy", "gbp", "aud", "cad", "fx"]
        if any(keyword in query for keyword in forex_keywords):
            return "forex"
        
        # Default to stock
        return "stock"
        
    def _needs_real_time_data(self, query):
        """Determine if real-time data is needed based on query"""
        query = query.lower()
        
        # Keywords suggesting real-time data needs
        real_time_keywords = [
            "real-time", "realtime", "real time", "live", "current", "right now", "latest", 
            "up to the minute", "intraday", "minute", "hourly", "today's", "day trading", 
            "scalping", "tick", "second", "instant", "immediate", "now"
        ]
        
        return any(keyword in query for keyword in real_time_keywords)
    
    def process_query(self, query):
        """Process a user query using the ReAct pattern and return a response"""
        # REASON: Determine asset type and real-time needs
        asset_type = self._detect_asset_type(query)
        needs_real_time = self._needs_real_time_data(query)
        print(f"REACT - REASON: Analyzing query for asset type: {asset_type}")
        print(f"REACT - REASON: Real-time data needed: {needs_real_time}")
        
        # REASON: Determine which tools to use based on the query
        tools_needed = self.determine_tools_needed(query)
        print(f"REACT - ACT: Selected tools: {tools_needed}")
        
        # Check if portfolio integration tools are needed
        if "analyze_portfolio" in tools_needed:
            print(f"REACT - ACT: Using portfolio analysis tool with full query")
            result = self.tools["analyze_portfolio"](query)
            return result
            
        elif "backtest_strategy" in tools_needed:
            print(f"REACT - ACT: Using backtesting tool with full query")
            result = self.tools["backtest_strategy"](query)
            return result
            
        elif "run_paper_trading" in tools_needed:
            print(f"REACT - ACT: Using paper trading simulation tool with full query")
            result = self.tools["run_paper_trading"](query)
            return result
        
        # For standard single-stock analysis, continue with normal flow
        # ACT: Extract stock symbol from query
        symbol = self.extract_stock_symbol(query)
        
        # ACT: Execute tools
        results = {}
        for tool_name in tools_needed:
            if tool_name in self.tools:
                print(f"REACT - ACT: Executing {tool_name} for {symbol}")
                try:
                    if tool_name == "get_economic_indicator":
                        # Economic indicators don't need a symbol
                        results[tool_name] = self.tools[tool_name](query)
                    elif tool_name in ["calculate_obv", "calculate_adline", "calculate_adx"]:
                        # Technical indicators need to fetch their own data
                        # The wrapper methods already handle fetching data for the symbol
                        results[tool_name] = self.tools[tool_name](symbol)
                    else:
                        # Most tools need a symbol
                        results[tool_name] = self.tools[tool_name](symbol)
                    print(f"REACT - OBSERVE: Got result from {tool_name}")
                except Exception as e:
                    print(f"REACT - ERROR: Tool '{tool_name}' failed with error: {str(e)}")
                    results[tool_name] = f"Error: {str(e)}"
            else:
                print(f"REACT - ERROR: Tool '{tool_name}' not found")
        
        # Format results into a response
        response = self._format_response(query, symbol, results)
        
        return response
    
    def _format_response(self, query, symbol, results):
        """Format the results into a structured response"""
        # Create a structured response with sections
        response = f"# Financial Analysis for {symbol}\n\n"
        
        # Add sections based on available results
        for tool_name, result in results.items():
            if isinstance(result, str) and result.startswith("Error"):
                response += f"## {tool_name.replace('_', ' ').title()}\n{result}\n\n"
            else:
                response += f"## {tool_name.replace('_', ' ').title()}\n{result}\n\n"
        
        return response
        
    def _detect_asset_type(self, query):
        """Detect the type of asset being queried (stock, crypto, forex)"""
        query = query.lower()
        
        # Check for cryptocurrency keywords
        crypto_keywords = ["crypto", "bitcoin", "btc", "ethereum", "eth", "cryptocurrency", "token", "coin"]
        if any(keyword in query for keyword in crypto_keywords):
            return "crypto"
        
        # Check for forex keywords
        forex_keywords = ["forex", "currency", "exchange rate", "eur/usd", "usd/jpy", "gbp", "aud", "cad", "fx"]
        if any(keyword in query for keyword in forex_keywords):
            return "forex"
        
        # Default to stock
        return "stock"
    
    def _needs_real_time_data(self, query):
        """Determine if real-time data is needed based on query"""
        query = query.lower()
        
        # Keywords suggesting real-time data needs
        real_time_keywords = [
            "real-time", "realtime", "real time", "live", "current", "right now", "latest", 
            "up to the minute", "intraday", "minute", "hourly", "today's", "day trading", 
            "scalping", "tick", "second", "instant", "immediate", "now"
        ]
        
        return any(keyword in query for keyword in real_time_keywords)
    
    def process_query(self, query):
        """Process a user query using the ReAct pattern and return a response"""
        try:
            # REASON: Determine which tools are needed based on the query
            tools_needed = self.determine_tools_needed(query)
            
            # Handle portfolio integration tools first (these take the full query as input)
            if "analyze_portfolio" in tools_needed:
                print(f"REACT - ACT: Using analyze_portfolio integration tool with full query")
                result = self.tools["analyze_portfolio"](query)
                return result
                
            elif "backtest_strategy" in tools_needed:
                print(f"REACT - ACT: Using backtest_strategy integration tool with full query")
                result = self.tools["backtest_strategy"](query)
                return result
                
            elif "run_paper_trading" in tools_needed:
                print(f"REACT - ACT: Using run_paper_trading integration tool with full query")
                result = self.tools["run_paper_trading"](query)
                return result
            
            # For standard single-asset analysis, continue with normal flow
            # REASON: Extract symbol and determine asset type
            asset_type = self._detect_asset_type(query)
            symbol = self.extract_stock_symbol(query)
            needs_real_time = self._needs_real_time_data(query)
            
            print(f"\nREACT - REASON: Analyzing query for asset type: {asset_type}, symbol: {symbol}")
            print(f"REACT - REASON: Real-time data needed: {needs_real_time}")
            print(f"REACT - ACT: Selected tools: {tools_needed}")
            
            # ACT & OBSERVE: Execute each tool and collect results
            results = {}
            for tool_name in tools_needed:
                print(f"REACT - ACT: Executing {tool_name} for {symbol}")
                
                tool_function = self.tools[tool_name]
                
                # Handle special parameters for different tools
                if tool_name == "get_stock_history":
                    # Extract period if specified, otherwise use default
                    period = "1mo"  # default
                    if "year" in query.lower() or "1y" in query.lower():
                        period = "1y"
                    elif "month" in query.lower() or "1mo" in query.lower():
                        period = "1mo"
                    elif "week" in query.lower() or "1w" in query.lower():
                        period = "1w"
                    elif "day" in query.lower() or "1d" in query.lower():
                        period = "1d"
                    
                    results[tool_name] = tool_function(symbol, period)
                
                elif tool_name == "visualize_stock":
                    # Determine which indicators to include
                    indicators = ["sma", "rsi"]  # Default
                    if "bollinger" in query.lower() or "bands" in query.lower():
                        indicators.append("bollinger")
                    if "macd" in query.lower():
                        indicators.append("macd")
                    
                    # Determine period
                    period = "6mo"  # Default
                    if "year" in query.lower() or "1y" in query.lower():
                        period = "1y"
                    elif "month" in query.lower() or "3mo" in query.lower():
                        period = "3mo"
                    
                    results[tool_name] = tool_function(symbol, period, indicators)
                
                elif tool_name == "get_stock_news":
                    # Determine how many news items to include
                    max_items = 5  # Default
                    if "more news" in query.lower() or "detailed news" in query.lower():
                        max_items = 10
                    
                    results[tool_name] = tool_function(symbol, max_items)
                    
                elif tool_name == "get_intraday_data":
                    # Determine interval for intraday data
                    interval = "1min"  # Default to 1-minute data
                    if "5 minute" in query.lower() or "5min" in query.lower():
                        interval = "5min"
                    elif "15 minute" in query.lower() or "15min" in query.lower():
                        interval = "15min"
                    elif "30 minute" in query.lower() or "30min" in query.lower():
                        interval = "30min"
                    elif "60 minute" in query.lower() or "hourly" in query.lower() or "1hour" in query.lower():
                        interval = "60min"
                    
                    # Determine output size
                    output_size = "compact"  # Default (latest 100 data points)
                    if "full" in query.lower() or "complete" in query.lower() or "all data" in query.lower():
                        output_size = "full"  # Full data (up to 30 days)
                    
                    data = tool_function(symbol, interval, output_size)
                    results[tool_name] = AlphaVantageTools.format_real_time_data_for_display(data)
                
                elif tool_name == "get_real_time_quote":
                    data = tool_function(symbol)
                    results[tool_name] = AlphaVantageTools.format_real_time_data_for_display(data)
                
                elif tool_name == "get_crypto_data":
                    # Handle crypto data
                    market = "USD"  # Default market
                    if isinstance(symbol, tuple) and len(symbol) == 2:
                        # If symbol is actually a currency pair
                        from_currency, to_currency = symbol
                        # Use forex tool instead
                        data = self.tools["get_forex_data"](from_currency, to_currency)
                        results[tool_name] = AlphaVantageTools.format_real_time_data_for_display(data)
                    else:
                        # Determine interval
                        interval = "1min" if needs_real_time else "daily"
                        if "weekly" in query.lower():
                            interval = "weekly"
                        elif "monthly" in query.lower():
                            interval = "monthly"
                        
                        data = tool_function(symbol, market, interval)
                        results[tool_name] = AlphaVantageTools.format_real_time_data_for_display(data)
                
                elif tool_name == "get_forex_data":
                    # Handle forex data
                    if isinstance(symbol, tuple) and len(symbol) == 2:
                        from_currency, to_currency = symbol
                    else:
                        # Default to EUR/USD if not properly specified
                        from_currency, to_currency = "EUR", "USD"
                    
                    # Determine interval
                    interval = "1min" if needs_real_time else "daily"
                    if "weekly" in query.lower():
                        interval = "weekly"
                    elif "monthly" in query.lower():
                        interval = "monthly"
                    
                    data = tool_function(from_currency, to_currency, interval)
                    results[tool_name] = AlphaVantageTools.format_real_time_data_for_display(data)
                
                elif tool_name == "get_economic_indicator":
                    # Determine which economic indicator to fetch
                    indicator = "GDP"  # Default
                    if "inflation" in query.lower() or "cpi" in query.lower():
                        indicator = "INFLATION"
                    elif "unemployment" in query.lower():
                        indicator = "UNEMPLOYMENT"
                    elif "retail" in query.lower() or "sales" in query.lower():
                        indicator = "RETAIL_SALES"
                    elif "treasury" in query.lower() or "yield" in query.lower():
                        indicator = "TREASURY_YIELD"
                    elif "consumer" in query.lower() or "sentiment" in query.lower():
                        indicator = "CONSUMER_SENTIMENT"
                    elif "nonfarm" in query.lower() or "payroll" in query.lower() or "jobs" in query.lower():
                        indicator = "NONFARM_PAYROLL"
                    
                    data = tool_function(indicator)
                    results[tool_name] = AlphaVantageTools.format_real_time_data_for_display(data)
                
                # Handle predictive analytics tools
                elif tool_name in ["forecast_with_prophet", "forecast_with_lstm", "detect_anomalies", 
                                  "calculate_volatility", "scenario_analysis", "check_stationarity"]:
                    # First, we need historical data to work with
                    print(f"REACT - ACT: Getting historical data for {symbol} for predictive analytics")
                    
                    # Determine period based on query
                    period = "1y"  # Default to 1 year of data for predictions
                    if "short term" in query.lower() or "near term" in query.lower():
                        period = "3mo"
                    elif "long term" in query.lower() or "long run" in query.lower():
                        period = "5y"
                    
                    # Get historical data using yfinance
                    if asset_type == "stock":
                        historical_data = StockTools.get_stock_history(symbol, period, return_df=True)
                    elif asset_type == "crypto":
                        # For crypto, we'll use Alpha Vantage data
                        crypto_data = AlphaVantageTools.get_crypto_data(symbol, "USD", "daily")
                        if "error" in crypto_data:
                            results[tool_name] = {"error": f"Could not get historical data for {symbol}: {crypto_data['error']}"}
                            continue
                        # Convert Alpha Vantage data to DataFrame
                        historical_data = pd.DataFrame(crypto_data['time_series'])
                    else:
                        results[tool_name] = {"error": f"Predictive analytics not supported for asset type: {asset_type}"}
                        continue
                    
                    # Check if we have valid historical data
                    if historical_data is None or historical_data.empty:
                        results[tool_name] = {"error": f"No historical data available for {symbol}"}
                        continue
                    
                    # Process based on specific predictive tool
                    if tool_name == "forecast_with_prophet":
                        # Determine forecast periods
                        periods = 30  # Default to 30 days
                        if "week" in query.lower():
                            periods = 7
                        elif "month" in query.lower():
                            periods = 30
                        elif "quarter" in query.lower():
                            periods = 90
                        elif "year" in query.lower():
                            periods = 365
                        
                        # Ensure historical_data is properly formatted for Prophet
                        try:
                            # Make a copy to avoid modifying the original data
                            prophet_data = historical_data.copy()
                            
                            # Check if we have a DatetimeIndex
                            if isinstance(prophet_data.index, pd.DatetimeIndex):
                                prophet_data.reset_index(inplace=True)
                                prophet_data.rename(columns={'index': 'ds', 'Close': 'y'}, inplace=True)
                            
                            # If we already have 'Date' column, rename it
                            if 'Date' in prophet_data.columns:
                                prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
                            
                            # Ensure we have 'ds' and 'y' columns for Prophet
                            if 'ds' not in prophet_data.columns or 'y' not in prophet_data.columns:
                                # Create them from scratch if needed
                                if isinstance(historical_data.index, pd.DatetimeIndex):
                                    prophet_data = pd.DataFrame({
                                        'ds': historical_data.index,
                                        'y': historical_data['Close']
                                    })
                            
                            results[tool_name] = tool_function(prophet_data, periods=periods)
                        except Exception as e:
                            results[tool_name] = {"error": f"Error preparing data for Prophet: {str(e)}"}
                        
                    elif tool_name == "forecast_with_lstm":
                        # Configure LSTM parameters
                        target_column = 'Close'  # Default to Close price
                        sequence_length = 10  # Default lookback window
                        forecast_periods = 30  # Default forecast horizon
                        
                        if "week" in query.lower():
                            forecast_periods = 7
                        elif "month" in query.lower():
                            forecast_periods = 30
                        elif "quarter" in query.lower():
                            forecast_periods = 90
                            
                        results[tool_name] = tool_function(historical_data, target_column, sequence_length, forecast_periods)
                        
                    elif tool_name == "detect_anomalies":
                        # Configure anomaly detection parameters
                        contamination = 0.05  # Default contamination rate
                        if "high sensitivity" in query.lower() or "detect more" in query.lower():
                            contamination = 0.1
                        elif "low sensitivity" in query.lower() or "strict" in query.lower():
                            contamination = 0.01
                            
                        results[tool_name] = tool_function(historical_data['Close'], contamination)
                        
                    elif tool_name == "calculate_volatility":
                        # Configure volatility calculation parameters
                        window_size = 20  # Default window size (20 trading days ~ 1 month)
                        if "short term" in query.lower():
                            window_size = 10
                        elif "long term" in query.lower():
                            window_size = 60
                            
                        results[tool_name] = tool_function(historical_data, price_column='Close', window_size=window_size)
                        
                    elif tool_name == "scenario_analysis":
                        # Configure scenario analysis parameters
                        forecast_periods = 30  # Default to 30 days
                        if "week" in query.lower():
                            forecast_periods = 7
                        elif "month" in query.lower():
                            forecast_periods = 30
                        elif "quarter" in query.lower():
                            forecast_periods = 90
                        elif "year" in query.lower():
                            forecast_periods = 252  # Trading days in a year
                            
                        # Define scenarios based on query
                        scenarios = ['base', 'bull', 'bear']  # Use explicit default scenarios
                        
                        try:
                            # Ensure historical_data has the required format
                            if 'Close' not in historical_data.columns:
                                # Try to find an appropriate price column
                                price_columns = [col for col in historical_data.columns 
                                                if col.lower() in ['close', 'price', 'value', 'adjclose']]
                                
                                if price_columns:
                                    price_column = price_columns[0]
                                else:
                                    # If no suitable column found, use the first numeric column
                                    numeric_cols = [col for col in historical_data.columns 
                                                  if pd.api.types.is_numeric_dtype(historical_data[col])]
                                    if numeric_cols:
                                        price_column = numeric_cols[0]
                                    else:
                                        results[tool_name] = {"error": "No suitable price column found in data"}
                                        continue
                            else:
                                price_column = 'Close'
                                
                            # Ensure we have a proper datetime index
                            if not isinstance(historical_data.index, pd.DatetimeIndex):
                                if 'Date' in historical_data.columns:
                                    historical_data = historical_data.set_index('Date')
                                else:
                                    # Create a synthetic date index
                                    historical_data = historical_data.reset_index(drop=True)
                                    end_date = datetime.now()
                                    start_date = end_date - timedelta(days=len(historical_data))
                                    historical_data.index = pd.date_range(start=start_date, periods=len(historical_data))
                            
                            results[tool_name] = tool_function(historical_data, price_column=price_column, 
                                                            scenarios=scenarios, forecast_periods=forecast_periods)
                        except Exception as e:
                            results[tool_name] = {"error": f"Error in scenario analysis: {str(e)}"}
                        
                    elif tool_name == "check_stationarity":
                        # Check stationarity of the time series
                        results[tool_name] = tool_function(historical_data['Close'])
                
                else:
                    # Standard execution for other tools
                    results[tool_name] = tool_function(symbol)
                
                print(f"REACT - OBSERVE: Got result from {tool_name}")
                
            # Format asset type and symbol for display
            if asset_type == "crypto":
                asset_display = f"cryptocurrency {symbol}"
            elif asset_type == "forex" and isinstance(symbol, tuple) and len(symbol) == 2:
                asset_display = f"forex pair {symbol[0]}/{symbol[1]}"
            else:
                asset_display = f"stock {symbol}"
            
            # LOOP: Analyze all results together
            print("REACT - LOOP: Analyzing all results together")
            
            # Combine all results into a comprehensive analysis
            combined_result = f"Financial Analysis for {asset_display}:\n\n"
            
            for tool_name, result in results.items():
                combined_result += f"--- {tool_name.replace('_', ' ').title()} Results ---\n"
                combined_result += f"{result}\n\n"
            
            # Use Gemini to analyze the combined results and provide insights if available
            if model is not None:
                try:
                    # Create appropriate prompt based on asset type
                    if asset_type == "crypto":
                        prompt = f"""
                        As a financial analyst following the ReAct pattern (Reason → Act → Observe → Loop),
                        analyze the following comprehensive data about {symbol} cryptocurrency:
                        
                        {combined_result}
                        
                        Provide a thorough analysis with:
                        1. Summary: Brief overview of key findings
                        2. Technical Analysis: Insights from price action and indicators
                        3. Market Sentiment: Current market sentiment and trends
                        4. Recommendation: Suggested actions or insights based on all data
                        
                        Keep your response professional and fact-based.
                        """
                    elif asset_type == "forex":
                        from_currency = symbol[0] if isinstance(symbol, tuple) else "EUR"
                        to_currency = symbol[1] if isinstance(symbol, tuple) else "USD"
                        prompt = f"""
                        As a financial analyst following the ReAct pattern (Reason → Act → Observe → Loop),
                        analyze the following comprehensive data about {from_currency}/{to_currency} forex pair:
                        
                        {combined_result}
                        
                        Provide a thorough analysis with:
                        1. Summary: Brief overview of key findings
                        2. Technical Analysis: Insights from exchange rate movements
                        3. Economic Factors: Relevant economic factors affecting the currencies
                        4. Recommendation: Suggested actions or insights based on all data
                        
                        Keep your response professional and fact-based.
                        """
                    elif "get_economic_indicator" in tools_needed:
                        prompt = f"""
                        As a financial analyst following the ReAct pattern (Reason → Act → Observe → Loop),
                        analyze the following comprehensive economic data:
                        
                        {combined_result}
                        
                        Provide a thorough analysis with:
                        1. Summary: Brief overview of key findings
                        2. Economic Analysis: Insights from economic indicators
                        3. Market Impact: How these economic factors might affect financial markets
                        4. Outlook: Economic outlook based on the data
                        
                        Keep your response professional and fact-based.
                        """
                    else:  # Default stock analysis
                        prompt = f"""
                        As a financial analyst following the ReAct pattern (Reason → Act → Observe → Loop),
                        analyze the following comprehensive data about {symbol} stock:
                        
                        {combined_result}
                        
                        Provide a thorough analysis with:
                        1. Summary: Brief overview of key findings
                        2. Technical Analysis: Insights from technical indicators
                        3. Fundamental Analysis: Insights from company information
                        4. News Impact: How recent news might affect the stock
                        5. Recommendation: Suggested actions or insights based on all data
                        
                        Keep your response professional and fact-based.
                        """
                    
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    print(f"Warning: Gemini API error: {str(e)}")
                    print("Falling back to raw data")
                    # Fall back to raw data with a simple analysis
                    return self._generate_simple_analysis(symbol, results, asset_type)
            else:
                # If no model is available, generate a simple analysis
                return self._generate_simple_analysis(symbol, results, asset_type)
            
        except Exception as e:
            return f"Error processing your query: {str(e)}"
    
    def _generate_simple_analysis(self, symbol, results, asset_type="stock"):
        """Generate a simple analysis when Gemini is not available"""
        # Format asset display name
        if asset_type == "crypto":
            asset_display = f"cryptocurrency {symbol}"
        elif asset_type == "forex" and isinstance(symbol, tuple) and len(symbol) == 2:
            asset_display = f"forex pair {symbol[0]}/{symbol[1]}"
        else:
            asset_display = f"stock {symbol}"
            
        analysis = f"Financial Analysis for {asset_display}:\n\n"
        
        # Summary section
        analysis += "SUMMARY\n-------\n"
        
        # Handle different asset types
        if asset_type == "stock":
            # Stock price information
            if "get_stock_price" in results:
                analysis += f"{results['get_stock_price']}\n"
            elif "get_real_time_quote" in results:
                analysis += f"{results['get_real_time_quote']}\n"
                
            # Technical Analysis for stocks
            has_technical = any(tool in results for tool in ["calculate_rsi", "calculate_macd", "get_intraday_data"])
            if has_technical:
                analysis += "\nTECHNICAL ANALYSIS\n-----------------\n"
                if "calculate_rsi" in results:
                    analysis += f"{results['calculate_rsi']}\n"
                if "calculate_macd" in results:
                    analysis += f"{results['calculate_macd']}\n"
                if "get_intraday_data" in results:
                    analysis += f"{results['get_intraday_data']}\n"
            
            # Fundamental Analysis for stocks
            if "get_company_info" in results:
                analysis += "\nFUNDAMENTAL ANALYSIS\n-------------------\n"
                analysis += f"{results['get_company_info']}\n"
            
            # News Analysis for stocks
            if "get_stock_news" in results:
                analysis += "\nRECENT NEWS\n-----------\n"
                analysis += f"{results['get_stock_news']}\n"
            
            # Historical Performance for stocks
            if "get_stock_history" in results:
                analysis += "\nHISTORICAL PERFORMANCE\n--------------------\n"
                analysis += f"{results['get_stock_history']}\n"
            
            # Visualization for stocks
            if "visualize_stock" in results:
                analysis += "\nVISUALIZATION\n-------------\n"
                analysis += f"{results['visualize_stock']}\n"
                
        elif asset_type == "crypto":
            # Crypto data
            if "get_crypto_data" in results:
                analysis += f"{results['get_crypto_data']}\n"
                
            analysis += "\nCRYPTO MARKET ANALYSIS\n---------------------\n"
            analysis += "Based on the data retrieved, here is a simple analysis of the cryptocurrency.\n"
            analysis += "For more detailed insights, please check the data above.\n"
            
        elif asset_type == "forex":
            # Forex data
            if "get_forex_data" in results:
                analysis += f"{results['get_forex_data']}\n"
                
            analysis += "\nFOREX MARKET ANALYSIS\n--------------------\n"
            analysis += "Based on the exchange rate data retrieved, here is a simple analysis of the forex pair.\n"
            analysis += "For more detailed insights, please check the data above.\n"
            
        # Economic indicators (can apply to any asset type)
        if "get_economic_indicator" in results:
            analysis += "\nECONOMIC INDICATORS\n-------------------\n"
            analysis += f"{results['get_economic_indicator']}\n"
            analysis += "\nThese economic indicators provide context for market conditions and may impact asset prices.\n"
        
        return analysis

# Main function to run the agent
def main():
    print("=" * 80)
    print("Financial Analyst ReAct Agent".center(80))
    print("=" * 80)
    
    agent = FinanceAnalystReActAgent()
    print("Agent initialized! Ask me about any stock.")
    print("\nExample queries:")
    print("1. What's the current price of AAPL?")
    print("2. Tell me about MSFT company information")
    print("3. Is TSLA overbought or oversold based on RSI?")
    print("4. Show me the MACD analysis for NVDA")
    print("5. What are the latest news about AMZN?")
    print("6. Create a visualization of GOOG with bollinger bands")
    print("7. Give me a complete technical analysis of META")
    
    while True:
        try:
            query = input("\nEnter your question (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("Thank you for using the Financial Analyst ReAct Agent. Goodbye!")
                break
                
            if not query.strip():
                continue
                
            print("\nAnalyzing using ReAct pattern (Reason → Act → Observe → Loop)...")
            response = agent.process_query(query)
            print("\nResponse:")
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
