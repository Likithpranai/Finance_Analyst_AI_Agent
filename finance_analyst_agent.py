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
        You are a Senior Financial Analyst with extensive experience in equity research, technical analysis, and portfolio management.
        You provide institutional-grade financial analysis and investment recommendations following rigorous analytical frameworks.
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
        
        RESPONSE FORMATTING REQUIREMENTS:
        You must structure ALL responses using the following professional format:
        
        ## EXECUTIVE SUMMARY
        • Key findings and investment thesis in 2-3 bullet points
        • Current price and price target (if applicable)
        • Overall recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        
        ## TECHNICAL ANALYSIS
        • Price action and trend analysis
        • Key technical indicators (RSI, MACD, moving averages)
        • Support and resistance levels
        • Volume analysis and momentum indicators
        
        ## FUNDAMENTAL ANALYSIS
        • Financial metrics and ratios (P/E, P/B, ROE, etc.)
        • Revenue and earnings trends
        • Balance sheet strength and debt levels
        • Industry comparison and competitive positioning
        
        ## MARKET CONTEXT
        • Recent news and market developments
        • Sector performance and economic factors
        • Risk factors and catalysts
        
        ## INVESTMENT RECOMMENDATION
        • Clear buy/sell/hold recommendation with rationale
        • Price targets and time horizon
        • Risk assessment and position sizing guidance
        • Key metrics to monitor going forward
        
        PROFESSIONAL STANDARDS:
        • Use precise financial terminology and industry-standard metrics
        • Provide quantitative analysis with specific numbers and percentages
        • Include confidence levels and risk assessments
        • Maintain objectivity and avoid emotional language
        • Always cite data sources and time periods
        • DO NOT make up information - only use data provided by tools
        """
    
    def determine_tools_needed(self, query):
        """Determine which tools are needed based on the query"""
        query_lower = query.lower()
        tools_needed = []
        
        # Portfolio integration tools (these handle the full query)
        if "portfolio" in query_lower or "holdings" in query_lower:
            tools_needed.append("analyze_portfolio")
            return tools_needed
            
        if "backtest" in query_lower or "strategy" in query_lower:
            tools_needed.append("backtest_strategy")
            return tools_needed
            
        if "paper trading" in query_lower or "simulation" in query_lower:
            tools_needed.append("run_paper_trading")
            return tools_needed
        
        # Check if this is a general financial query
        if self._is_general_financial_query(query):
            # For general queries, we'll use market news and economic indicators
            if "market" in query_lower or "trend" in query_lower:
                tools_needed.append("get_market_news")
            if "economic" in query_lower or "inflation" in query_lower or "interest rate" in query_lower:
                tools_needed.append("get_economic_indicator")
            # If no specific tools were selected, use a default set
            if not tools_needed:
                tools_needed = ["get_market_news"]
            return tools_needed
        
        # For specific asset analysis, determine appropriate tools
        # Always get basic price information
        if self._needs_real_time_data(query):
            tools_needed.append("get_real_time_quote")
        else:
            tools_needed.append("get_stock_price")
            tools_needed.append("get_stock_history")
        
        # Check for technical analysis keywords
        if any(term in query_lower for term in ["technical", "rsi", "relative strength", "momentum", "overbought", "oversold"]):
            tools_needed.append("calculate_rsi")
            
        if any(term in query_lower for term in ["macd", "moving average", "convergence", "divergence", "signal", "crossover"]):
            tools_needed.append("calculate_macd")
            
        if any(term in query_lower for term in ["obv", "on balance volume", "volume"]):
            tools_needed.append("calculate_obv")
            
        if any(term in query_lower for term in ["adline", "accumulation distribution", "money flow"]):
            tools_needed.append("calculate_adline")
            
        if any(term in query_lower for term in ["adx", "directional", "trend strength", "dmi"]):
            tools_needed.append("calculate_adx")
            
        # Check for fundamental analysis keywords
        if any(term in query_lower for term in ["fundamental", "company", "financials", "ratios", "p/e", "eps", "revenue"]):
            tools_needed.append("get_company_info")
            tools_needed.append("get_financial_ratios")
            
        if any(term in query_lower for term in ["income", "earnings", "profit", "revenue", "expenses"]):
            tools_needed.append("get_income_statement_summary")
            
        if any(term in query_lower for term in ["balance sheet", "assets", "liabilities", "equity", "debt"]):
            tools_needed.append("get_balance_sheet_summary")
            
        # Check for news keywords
        if any(term in query_lower for term in ["news", "headlines", "announcement", "press", "release"]):
            tools_needed.append("get_stock_news")
            
        # Check for visualization keywords
        if any(term in query_lower for term in ["chart", "graph", "plot", "visual", "picture", "show"]):
            tools_needed.append("visualize_stock")
            
        # Check for comprehensive analysis keywords
        if any(term in query_lower for term in ["comprehensive", "complete", "detailed", "full", "thorough", "analysis", "evaluate", "assess"]):
            # For comprehensive analysis, include a broad set of tools
            tools_needed = [
                "get_stock_price", 
                "get_stock_history", 
                "calculate_rsi", 
                "calculate_macd", 
                "get_company_info", 
                "get_financial_ratios",
                "get_stock_news", 
                "visualize_stock"
            ]
            
            # Add combined analysis if available
            if "create_combined_analysis" in self.tools:
                tools_needed.append("create_combined_analysis")
                
        # If no specific tools were identified, use a default set
        if not tools_needed:
            tools_needed = ["get_stock_price", "get_stock_history", "get_company_info", "get_stock_news"]
            
        return tools_needed
    
    def _needs_real_time_data(self, query):
        """Determine if the query requires real-time data"""
        query_lower = query.lower()
        
        # Keywords that suggest real-time data is needed
        real_time_keywords = [
            "current", "now", "today", "real-time", "realtime", "live", "latest", 
            "up to date", "right now", "at this moment", "current price", "trading at",
            "intraday", "minute", "hourly", "today's", "current session"
        ]
        
        # Check if any real-time keywords are in the query
        return any(keyword in query_lower for keyword in real_time_keywords)
    
    def _is_general_financial_query(self, query):
        """Detect if this is a general financial query rather than a specific stock analysis"""
        query_lower = query.lower()
        
        # Common financial terms that shouldn't be treated as stock symbols
        financial_terms = [
            "roi", "irr", "wacc", "capm", "cagr", "ebitda", "eps", "pe", "pb", "ps",
            "dcf", "fcf", "ocf", "roa", "roe", "roic", "npm", "gpm", "opm", "ltv",
            "cac", "arpu", "cogs", "sga", "r&d", "capex", "opex", "cpi", "gdp", "ppi",
            "ipo", "m&a", "p&l", "yoy", "qoq", "ttm", "ytd", "mtd", "etf", "reit",
            "hedge", "mutual", "index", "bond", "stock", "share", "equity", "debt",
            "asset", "liability", "dividend", "yield", "growth", "value", "beta",
            "finance", "market", "trend", "economy", "recession", "inflation", "interest rate",
            "fed", "bull", "bear", "sector", "industry", "treasury",
            "portfolio", "diversification", "asset allocation", "risk", "return", "volatility",
            "market cap", "capitalization", "ratio", "ratios", "financial ratio", "financial ratios",
            "valuation", "fundamental", "fundamental analysis", "technical analysis", "investment",
            "investing", "trader", "trading", "investor", "financial statement", "balance sheet",
            "income statement", "cash flow", "profit margin", "gross margin", "operating margin"
        ]
        
        # Technical analysis terms that shouldn't be treated as stock symbols
        technical_terms = [
            "rsi", "macd", "sma", "ema", "bollinger", "fibonacci", "stochastic", "momentum",
            "oscillator", "indicator", "chart pattern", "support", "resistance", "trend line",
            "volume", "obv", "adx", "dmi", "atr", "vwap", "ichimoku", "candlestick",
            "relative strength index", "moving average", "convergence divergence", "bands",
            "technical indicator", "chart", "pattern", "price action", "overbought", "oversold"
        ]
        
        # General question indicators
        question_indicators = [
            "what", "how", "why", "when", "which", "where", "can you", "tell me", "explain",
            "describe", "overview", "summary", "trends", "outlook", "forecast", "prediction",
            "define", "meaning", "concept", "understand", "difference between", "compare",
            "list", "what are", "key", "important", "best", "top", "main", "primary", "essential",
            "fundamental", "basic", "beginner", "learn", "guide", "tutorial", "introduction"
        ]
        
        # Educational query patterns
        educational_patterns = [
            r'what\s+(?:are|is)\s+(?:the\s+)?(?:key|main|important|essential)\s+(?:financial\s+)?(?:ratios|indicators|metrics|factors)',
            r'how\s+(?:do|can|to)\s+(?:calculate|compute|determine|evaluate|assess)',
            r'explain\s+(?:how|what|why)\s+(?:to|is|are)',
            r'(?:list|tell\s+me)\s+(?:the|some|all)\s+(?:key|main|important)'
        ]
        
        # Check if query contains financial or technical terms
        has_financial_terms = any(term in query_lower for term in financial_terms)
        has_technical_terms = any(term in query_lower for term in technical_terms)
        
        # Check if query is phrased as a general question
        is_general_question = any(indicator in query_lower for indicator in question_indicators)
        
        # Check if query is asking about general market trends or conditions
        about_general_trends = "trend" in query_lower or "market" in query_lower or "current" in query_lower or "recent" in query_lower
        
        # Check for specific patterns that indicate general questions
        is_definition_question = re.search(r'what\s+is\s+[a-z\s]+', query_lower) is not None
        is_how_to_question = re.search(r'how\s+to\s+[a-z\s]+', query_lower) is not None
        is_explain_question = re.search(r'explain\s+[a-z\s]+', query_lower) is not None
        
        # Check for educational query patterns
        is_educational_query = any(re.search(pattern, query_lower) is not None for pattern in educational_patterns)
        
        # If it looks like a general financial query and doesn't have a clear stock ticker pattern
        # (most tickers are 1-5 uppercase letters not preceded by text)
        has_clear_ticker = bool(re.search(r'\b[A-Z]{1,5}\b', query))
        
        # If query contains "for [TICKER]" or "of [TICKER]", it's likely about a specific stock
        specific_stock_pattern = re.search(r'for\s+[A-Z]{1,5}\b|of\s+[A-Z]{1,5}\b', query) is not None
        
        # Return True if it's likely a general query
        return ((has_financial_terms or has_technical_terms or is_definition_question or 
                is_how_to_question or is_explain_question or is_educational_query or
                (is_general_question and about_general_trends)) and 
                not specific_stock_pattern and not (has_clear_ticker and "current" in query_lower))
    
    def _detect_asset_type(self, query, symbol=None):
        """Detect the type of asset being queried (stock, crypto, forex)"""
        query_lower = query.lower()
        
        # Check for cryptocurrency keywords
        crypto_keywords = ["crypto", "bitcoin", "ethereum", "btc", "eth", "ltc", "xrp", "doge", "blockchain", "coin", "token"]
        crypto_symbols = ["BTC", "ETH", "LTC", "XRP", "BCH", "ADA", "DOT", "LINK", "XLM", "DOGE", "UNI", "USDT", "USDC"]
        
        # Check for forex keywords
        forex_keywords = ["forex", "currency", "exchange rate", "fx", "foreign exchange", "currency pair", "usd", "eur", "jpy", "gbp"]
        forex_pairs = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
        
        # Check if the symbol is a known crypto symbol
        if symbol and symbol in crypto_symbols:
            return "crypto"
            
        # Check if the symbol looks like a forex pair (XXX/YYY format)
        if symbol and "/" in symbol and len(symbol) == 7:
            return "forex"
            
        # Check for crypto keywords in the query
        if any(keyword in query_lower for keyword in crypto_keywords):
            return "crypto"
            
        # Check for forex keywords in the query
        if any(keyword in query_lower for keyword in forex_keywords) or any(pair.lower() in query_lower for pair in forex_pairs):
            return "forex"
            
        # Default to stock if no other asset type is detected
        return "stock"

    def _is_technical_indicator(self, potential_symbol):
        """Check if a potential symbol is actually a technical indicator"""
        # List of common technical indicators that might be mistaken for stock symbols
        technical_indicators = [
            "RSI", "MACD", "SMA", "EMA", "ADX", "ATR", "OBV", "CCI", "MFI", "ROC", 
            "DMI", "SAR", "TSI", "CMF", "PPO", "DPO", "KST", "PVT", "ADL", "MOM"
        ]
        
        return potential_symbol in technical_indicators
    
    def extract_stock_symbol(self, query):
        """Extract stock symbol from query"""
        # Common terms that might be mistaken for stock symbols
        exclude_terms = ["CEO", "CFO", "CTO", "COO", "CIO", "IPO", "ETF", "REIT", "GAAP", "SEC"]
        
        # Look for ticker pattern: 1-5 uppercase letters
        matches = re.findall(r'\b[A-Z]{1,5}\b', query)
        
        # Filter out common financial terms and technical indicators
        valid_symbols = []
        for match in matches:
            # Skip if it's in the exclude list
            if match in exclude_terms:
                continue
                
            # Skip if it's a technical indicator
            if self._is_technical_indicator(match):
                continue
                
            valid_symbols.append(match)
        
        # Check for specific patterns like "RSI for AAPL" or "TSLA's RSI"
        # In these cases, we want to extract AAPL or TSLA, not RSI
        rsi_pattern = re.search(r'RSI\s+(?:for|of)\s+([A-Z]{1,5})\b|([A-Z]{1,5})(?:[\s\'"]s)?\s+RSI', query)
        if rsi_pattern:
            symbol = rsi_pattern.group(1) or rsi_pattern.group(2)
            if symbol and symbol not in exclude_terms:
                return symbol
        
        if valid_symbols:
            return valid_symbols[0]
        else:
            # Check if this is likely a general query
            if self._is_general_financial_query(query):
                return None
            # Default to a common stock if no symbol found
            return "AAPL"

    def process_query(self, query):
        """Process a user query using the ReAct pattern and return a response"""
        try:
            # First, check if this is a general financial query rather than a specific stock analysis
            is_general_query = self._is_general_financial_query(query)
            
            # For general financial queries, use a conversational approach
            if is_general_query:
                print(f"REACT - REASON: Detected general financial query: '{query}'")
                
                # Use the AI model to generate a direct, conversational response if available
                if 'model' in globals() and model is not None:
                    try:
                        print(f"REACT - ACT: Using AI model for conversational response to general query")
                        prompt = f"""You are a professional financial analyst assistant. Provide a helpful, conversational, and directly relevant answer to this financial question: 
                        
                        {query}
                        
                        IMPORTANT: Your response MUST include specific financial data, numbers, statistics, and concrete information. For example:
                        - If discussing ROI, include actual percentage ranges and calculation examples
                        - If discussing market trends, include specific index values, percentage changes, and time periods
                        - If discussing financial ratios, include actual numerical ranges considered healthy/concerning
                        - If discussing economic indicators, include recent actual values and historical comparisons
                        
                        Make your answer clear, concise, and informative. Use a natural, conversational tone while ensuring you provide concrete financial data and numbers. Don't use a structured format with headers or sections. Don't mention that you're using any particular framework or methodology. Answer the question directly as a knowledgeable finance professional would in conversation, but ensure you include specific numbers and data points."""
                        
                        response = model.generate_content(prompt)
                        return response.text
                    except Exception as e:
                        print(f"Error using AI model for general query: {str(e)}")
                        # Fall back to standard processing if AI model fails
                
                # If AI model isn't available or fails, use relevant financial tools
                # but format the response in a more conversational way
                tools_needed = []
                
                # Select appropriate tools based on the query content
                query_lower = query.lower()
                
                # Always include market data for context
                tools_needed.append("get_market_news")
                
                # Add more specific tools based on query content
                if "market" in query_lower or "trend" in query_lower or "outlook" in query_lower:
                    tools_needed.append("get_market_news")
                if any(term in query_lower for term in ["economic", "economy", "inflation", "interest rate", "fed", "gdp", "unemployment", "cpi"]):
                    tools_needed.append("get_economic_indicator")
                
                # Add financial ratio tools for ratio-related queries
                if any(term in query_lower for term in ["ratio", "pe", "p/e", "eps", "dividend", "yield", "margin", "profit", "revenue", "income", "balance", "cash flow"]):
                    tools_needed.append("get_financial_ratios")
                if any(term in query_lower for term in ["sector", "industry", "performance", "compare"]):
                    tools_needed.append("get_sector_performance")
                if any(term in query_lower for term in ["dividend", "yield", "income", "payout"]):
                    tools_needed.append("get_dividend_info")
                
                # If no specific tools were selected, use a default set
                if not tools_needed:
                    tools_needed = ["get_market_news"]
                
                # Execute the selected tools
                results = {}
                for tool_name in tools_needed:
                    if tool_name in self.tools:
                        print(f"REACT - ACT: Executing {tool_name} for general query")
                        try:
                            if tool_name == "get_market_news":
                                # For general market news, don't specify a symbol
                                results[tool_name] = self.tools[tool_name]("MARKET")
                            elif tool_name == "get_economic_indicator":
                                # Determine which economic indicator to fetch based on the query
                                indicator = "GDP"  # Default
                                if "inflation" in query_lower or "cpi" in query_lower:
                                    indicator = "INFLATION"
                                elif "unemployment" in query_lower:
                                    indicator = "UNEMPLOYMENT"
                                elif "retail" in query_lower or "sales" in query_lower:
                                    indicator = "RETAIL_SALES"
                                elif "treasury" in query_lower or "yield" in query_lower:
                                    indicator = "TREASURY_YIELD"
                                elif "consumer" in query_lower or "sentiment" in query_lower:
                                    indicator = "CONSUMER_SENTIMENT"
                                elif "nonfarm" in query_lower or "payroll" in query_lower:
                                    indicator = "NONFARM_PAYROLL"
                                results[tool_name] = self.tools[tool_name](indicator)
                            elif tool_name == "get_sector_performance":
                                results[tool_name] = self.tools[tool_name]()
                            elif tool_name == "get_dividend_info":
                                # Use a diversified ETF for general dividend info
                                results[tool_name] = self.tools[tool_name]("SPY")
                            else:
                                # For other tools that might need a symbol
                                results[tool_name] = self.tools[tool_name]("SPY")  # Use S&P 500 ETF as default
                        except Exception as e:
                            print(f"REACT - ERROR: Tool '{tool_name}' failed with error: {str(e)}")
                            results[tool_name] = f"Error: {str(e)}"
                
                # Format the response in a conversational way
                return self._format_response(query, None, results, is_general_query=True)
            
            # For specific asset analysis, continue with normal flow
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
            
            # ACT: Execute the selected tools
            results = {}
            for tool_name in tools_needed:
                if tool_name in self.tools:
                    print(f"REACT - ACT: Executing {tool_name} for {symbol}")
                    try:
                        if tool_name == "get_real_time_quote":
                            results[tool_name] = self.tools[tool_name](symbol)
                        elif tool_name == "get_intraday_data":
                            results[tool_name] = self.tools[tool_name](symbol, needs_real_time)
                        elif tool_name == "get_historical_data":
                            results[tool_name] = self.tools[tool_name](symbol, needs_real_time)
                        elif tool_name == "get_crypto_data":
                            results[tool_name] = self.tools[tool_name](symbol, needs_real_time)
                        elif tool_name == "get_forex_data":
                            results[tool_name] = self.tools[tool_name](symbol, needs_real_time)
                        elif tool_name == "get_economic_indicator":
                            results[tool_name] = self.tools[tool_name](symbol)
                        else:
                            results[tool_name] = self.tools[tool_name](symbol)
                    except Exception as e:
                        print(f"REACT - ERROR: Tool '{tool_name}' failed with error: {str(e)}")
                        results[tool_name] = f"Error: {str(e)}"
            
            # LOOP: Analyze all results together
            print(f"REACT - LOOP: Analyzing all results together")
            
            # Check if this is a general financial query
            is_general_query = self._is_general_financial_query(query)
            
            # For general queries, use a direct response format
            if is_general_query:
                # If we have AI model results, use those directly
                if 'model' in globals() and model is not None:
                    try:
                        # Create a summary of all the tool results
                        summary = ""
                        for tool_name, result in results.items():
                            if isinstance(result, str):
                                summary += f"{result}\n\n"
                        
                        # Ask the AI model to generate a direct response based on the tools' output
                        prompt = f"Based on the following financial data and the query '{query}', provide a direct, concise answer without using a structured format or mentioning the ReAct framework:\n\n{summary}"
                        response = model.generate_content(prompt)
                        return response.text
                    except Exception as e:
                        print(f"Error using AI model for response formatting: {str(e)}")
                        # Fall back to standard formatting
                
                # If AI model isn't available or fails, use simplified formatting
                return self._format_response(query, symbol, results, is_general_query=True)
            
            # Format the final response for specific asset types
            if asset_type == "stock":
                return self._format_response(query, symbol, results)
            elif asset_type == "crypto":
                return self._format_response(query, symbol, results)
            elif asset_type == "forex":
                return self._format_response(query, symbol, results)
            else:
                return self._format_response(query, symbol, results)
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"

    def _format_response(self, query, symbol, results, is_general_query=False):
        """Format the results into a structured professional financial analyst response"""
        
        # Use Gemini AI to create a professional structured response
        if 'model' in globals() and model is not None:
            try:
                # Extract key information from results
                data_points = []
                for tool_name, result in results.items():
                    if isinstance(result, str) and not result.startswith("Error"):
                        data_points.append(f"{tool_name}: {result}")
                    elif not isinstance(result, str) and isinstance(result, dict) and "text" in result:
                        data_points.append(f"{tool_name}: {result['text']}")
                
                # Create a context-rich prompt for the model
                data_context = "\n\n".join(data_points)
                
                # Determine asset type for context
                asset_context = ""
                if symbol:
                    if isinstance(symbol, tuple) and len(symbol) == 2:
                        asset_context = f"analyzing forex pair {symbol[0]}/{symbol[1]}"
                    elif self._detect_asset_type(query) == "crypto":
                        asset_context = f"analyzing cryptocurrency {symbol}"
                    else:
                        asset_context = f"analyzing stock {symbol}"
                else:
                    asset_context = "providing general market analysis"
                
                prompt = f"""You are a Senior Financial Analyst providing institutional-grade analysis. You are {asset_context} based on the query: '{query}'
                
DATA AVAILABLE:
{data_context}

You MUST structure your response using this EXACT professional format:

## EXECUTIVE SUMMARY
• [Key findings and investment thesis in 2-3 bullet points]
• [Current price and price target if applicable]
• [Overall recommendation: Strong Buy/Buy/Hold/Sell/Strong Sell]

## TECHNICAL ANALYSIS
• [Price action and trend analysis]
• [Key technical indicators (RSI, MACD, moving averages)]
• [Support and resistance levels]
• [Volume analysis and momentum indicators]

## FUNDAMENTAL ANALYSIS
• [Financial metrics and ratios (P/E, P/B, ROE, etc.)]
• [Revenue and earnings trends]
• [Balance sheet strength and debt levels]
• [Industry comparison and competitive positioning]

## MARKET CONTEXT
• [Recent news and market developments]
• [Sector performance and economic factors]
• [Risk factors and catalysts]

## INVESTMENT RECOMMENDATION
• [Clear buy/sell/hold recommendation with rationale]
• [Price targets and time horizon]
• [Risk assessment and position sizing guidance]
• [Key metrics to monitor going forward]

PROFESSIONAL REQUIREMENTS:
- Use precise financial terminology and industry-standard metrics
- Provide quantitative analysis with specific numbers and percentages
- Include confidence levels and risk assessments
- Maintain objectivity and avoid emotional language
- Always cite data sources and time periods
- Only use the data provided - DO NOT make up information
- If data is missing for a section, state "Data not available" rather than fabricating"""
                
                response = model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                print(f"Error using AI model for response formatting: {str(e)}")
                # Fall back to manual formatting
        
        # Format asset display name
        if symbol is None:
            asset_display = "financial markets"
        elif isinstance(symbol, tuple) and len(symbol) == 2:
            asset_display = f"forex pair {symbol[0]}/{symbol[1]}"
        elif self._detect_asset_type(query) == "crypto":
            asset_display = f"cryptocurrency {symbol}"
        else:
            asset_display = f"stock {symbol}"
            
        analysis = f"Financial Analysis for {asset_display}:\n\n"
        
        # Summary section
        analysis += "SUMMARY\n-------\n"
        
        # Handle different asset types
        asset_type = self._detect_asset_type(query)
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
