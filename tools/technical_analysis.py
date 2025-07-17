"""
Technical Analysis Tools for Finance Analyst AI Agent

This module provides technical analysis tools for stock data including:
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
- Volatility indicators (Bollinger Bands)
- Support and resistance levels
- Chart pattern recognition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalAnalysisTools:
    """Technical analysis tools for stock data"""
    
    @staticmethod
    def get_stock_data(symbol, period="1y", interval="1d"):
        """
        Get historical stock data using yfinance
        
        Args:
            symbol (str): Stock symbol
            period (str): Period of data to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            dict: Dictionary with stock data and metadata
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return {"error": f"No data found for {symbol}"}
            
            # Get company info
            info = ticker.info
            company_name = info.get('longName', symbol)
            
            # Calculate price change
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = ((current_price / prev_close) - 1) * 100
            
            return {
                "symbol": symbol,
                "company_name": company_name,
                "data": hist,
                "current_price": current_price,
                "price_change": price_change,
                "period": period,
                "interval": interval
            }
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return {"error": f"Error fetching stock data: {str(e)}"}
    
    @staticmethod
    def calculate_moving_averages(data, windows=[20, 50, 200]):
        """
        Calculate Simple Moving Averages for given windows
        
        Args:
            data (DataFrame): Stock price data with 'Close' column
            windows (list): List of window sizes for moving averages
            
        Returns:
            dict: Dictionary with moving averages
        """
        result = {}
        
        for window in windows:
            ma_name = f"SMA{window}"
            result[ma_name] = data['Close'].rolling(window=window).mean()
            
        return result
    
    @staticmethod
    def calculate_exponential_moving_averages(data, windows=[12, 26, 50]):
        """
        Calculate Exponential Moving Averages for given windows
        
        Args:
            data (DataFrame): Stock price data with 'Close' column
            windows (list): List of window sizes for moving averages
            
        Returns:
            dict: Dictionary with exponential moving averages
        """
        result = {}
        
        for window in windows:
            ema_name = f"EMA{window}"
            result[ema_name] = data['Close'].ewm(span=window, adjust=False).mean()
            
        return result
    
    @staticmethod
    def calculate_rsi(data, window=14):
        """
        Calculate Relative Strength Index
        
        Args:
            data (DataFrame): Stock price data with 'Close' column
            window (int): RSI calculation window
            
        Returns:
            Series: RSI values
        """
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """
        Calculate Moving Average Convergence Divergence
        
        Args:
            data (DataFrame): Stock price data with 'Close' column
            fast (int): Fast EMA window
            slow (int): Slow EMA window
            signal (int): Signal line window
            
        Returns:
            dict: Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Args:
            data (DataFrame): Stock price data with 'Close' column
            window (int): Moving average window
            num_std (int): Number of standard deviations
            
        Returns:
            dict: Dictionary with middle band, upper band, and lower band
        """
        middle_band = data['Close'].rolling(window=window).mean()
        std_dev = data['Close'].rolling(window=window).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return {
            "middle_band": middle_band,
            "upper_band": upper_band,
            "lower_band": lower_band
        }
    
    @staticmethod
    def calculate_volatility(data, window=20):
        """
        Calculate historical volatility
        
        Args:
            data (DataFrame): Stock price data with 'Close' column
            window (int): Window for volatility calculation
            
        Returns:
            float: Annualized volatility percentage
        """
        # Calculate daily returns
        returns = data['Close'].pct_change().dropna()
        
        # Calculate rolling standard deviation
        std_dev = returns.rolling(window=window).std()
        
        # Annualize (multiply by sqrt of trading days)
        annualized_volatility = std_dev.iloc[-1] * np.sqrt(252) * 100
        
        return annualized_volatility
    
    @staticmethod
    def identify_support_resistance(data, window=10):
        """
        Identify support and resistance levels
        
        Args:
            data (DataFrame): Stock price data with 'High' and 'Low' columns
            window (int): Window for local minima/maxima detection
            
        Returns:
            dict: Dictionary with support and resistance levels
        """
        # Find local maxima for resistance
        resistance_levels = []
        for i in range(window, len(data) - window):
            if all(data['High'].iloc[i] > data['High'].iloc[i-window:i]) and \
               all(data['High'].iloc[i] > data['High'].iloc[i+1:i+window+1]):
                resistance_levels.append(data['High'].iloc[i])
        
        # Find local minima for support
        support_levels = []
        for i in range(window, len(data) - window):
            if all(data['Low'].iloc[i] < data['Low'].iloc[i-window:i]) and \
               all(data['Low'].iloc[i] < data['Low'].iloc[i+1:i+window+1]):
                support_levels.append(data['Low'].iloc[i])
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Filter levels near current price
        active_resistance = [level for level in resistance_levels if level > current_price]
        active_support = [level for level in support_levels if level < current_price]
        
        # Sort and take closest levels
        active_resistance.sort()
        active_support.sort(reverse=True)
        
        return {
            "support_levels": active_support[:3] if active_support else [],
            "resistance_levels": active_resistance[:3] if active_resistance else []
        }
    
    @staticmethod
    def get_technical_signals(symbol, period="1y"):
        """
        Get comprehensive technical analysis signals
        
        Args:
            symbol (str): Stock symbol
            period (str): Period of data to fetch
            
        Returns:
            dict: Dictionary with technical analysis results
        """
        try:
            # Get stock data
            stock_data = TechnicalAnalysisTools.get_stock_data(symbol, period)
            
            if "error" in stock_data:
                return stock_data
            
            data = stock_data["data"]
            
            # Calculate indicators
            sma = TechnicalAnalysisTools.calculate_moving_averages(data)
            ema = TechnicalAnalysisTools.calculate_exponential_moving_averages(data)
            rsi = TechnicalAnalysisTools.calculate_rsi(data)
            macd = TechnicalAnalysisTools.calculate_macd(data)
            bollinger = TechnicalAnalysisTools.calculate_bollinger_bands(data)
            volatility = TechnicalAnalysisTools.calculate_volatility(data)
            support_resistance = TechnicalAnalysisTools.identify_support_resistance(data)
            
            # Get latest values
            current_price = stock_data["current_price"]
            latest_rsi = rsi.iloc[-1]
            
            # Determine signals
            # Moving Average signals
            ma_signal = "Bullish" if current_price > sma["SMA50"].iloc[-1] and sma["SMA50"].iloc[-1] > sma["SMA200"].iloc[-1] else \
                       "Bearish" if current_price < sma["SMA50"].iloc[-1] and sma["SMA50"].iloc[-1] < sma["SMA200"].iloc[-1] else \
                       "Neutral"
            
            # RSI signals
            rsi_signal = "Overbought" if latest_rsi > 70 else \
                        "Oversold" if latest_rsi < 30 else \
                        "Neutral"
            
            # MACD signals
            macd_signal = "Bullish" if macd["histogram"].iloc[-1] > 0 and macd["histogram"].iloc[-2] <= 0 else \
                         "Bearish" if macd["histogram"].iloc[-1] < 0 and macd["histogram"].iloc[-2] >= 0 else \
                         "Neutral"
            
            # Bollinger Band signals
            bb_signal = "Upper Band Touch" if data["Close"].iloc[-1] > bollinger["upper_band"].iloc[-1] else \
                       "Lower Band Touch" if data["Close"].iloc[-1] < bollinger["lower_band"].iloc[-1] else \
                       "Within Bands"
            
            # Overall technical score (simple version)
            technical_score = 0
            
            # Add to score based on MA
            if ma_signal == "Bullish":
                technical_score += 1
            elif ma_signal == "Bearish":
                technical_score -= 1
                
            # Add to score based on RSI
            if rsi_signal == "Oversold":
                technical_score += 1
            elif rsi_signal == "Overbought":
                technical_score -= 1
                
            # Add to score based on MACD
            if macd_signal == "Bullish":
                technical_score += 1
            elif macd_signal == "Bearish":
                technical_score -= 1
            
            # Determine overall signal
            if technical_score >= 2:
                overall_signal = "Strong Buy"
            elif technical_score == 1:
                overall_signal = "Buy"
            elif technical_score == 0:
                overall_signal = "Hold"
            elif technical_score == -1:
                overall_signal = "Sell"
            else:
                overall_signal = "Strong Sell"
            
            return {
                "symbol": symbol,
                "company_name": stock_data["company_name"],
                "current_price": current_price,
                "price_change": stock_data["price_change"],
                "sma50": sma["SMA50"].iloc[-1],
                "sma200": sma["SMA200"].iloc[-1],
                "ema12": ema["EMA12"].iloc[-1],
                "ema26": ema["EMA26"].iloc[-1],
                "rsi": latest_rsi,
                "macd": macd["macd_line"].iloc[-1],
                "macd_signal": macd_signal,
                "macd_histogram": macd["histogram"].iloc[-1],
                "bollinger_upper": bollinger["upper_band"].iloc[-1],
                "bollinger_middle": bollinger["middle_band"].iloc[-1],
                "bollinger_lower": bollinger["lower_band"].iloc[-1],
                "bollinger_signal": bb_signal,
                "volatility": volatility,
                "support_levels": support_resistance["support_levels"],
                "resistance_levels": support_resistance["resistance_levels"],
                "ma_signal": ma_signal,
                "rsi_signal": rsi_signal,
                "technical_score": technical_score,
                "overall_signal": overall_signal
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {str(e)}")
            return {"error": f"Error in technical analysis: {str(e)}"}
    
    @staticmethod
    def visualize_technical_analysis(symbol, period="1y", save_path=None):
        """
        Create and save technical analysis visualization
        
        Args:
            symbol (str): Stock symbol
            period (str): Period of data to fetch
            save_path (str): Path to save the visualization
            
        Returns:
            dict: Dictionary with visualization path and analysis results
        """
        try:
            # Get technical signals
            signals = TechnicalAnalysisTools.get_technical_signals(symbol, period)
            
            if "error" in signals:
                return signals
            
            # Get stock data
            stock_data = TechnicalAnalysisTools.get_stock_data(symbol, period)
            data = stock_data["data"]
            
            # Create figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot price and moving averages
            axs[0].plot(data.index, data['Close'], label='Close Price')
            axs[0].plot(data.index, signals['sma50'], label='SMA 50', linestyle='--')
            axs[0].plot(data.index, signals['sma200'], label='SMA 200', linestyle='--')
            
            # Plot Bollinger Bands
            sma = TechnicalAnalysisTools.calculate_moving_averages(data, [20])
            bollinger = TechnicalAnalysisTools.calculate_bollinger_bands(data)
            axs[0].plot(data.index, bollinger['upper_band'], 'g--', alpha=0.3)
            axs[0].plot(data.index, bollinger['middle_band'], 'g-', alpha=0.3)
            axs[0].plot(data.index, bollinger['lower_band'], 'g--', alpha=0.3)
            axs[0].fill_between(data.index, bollinger['upper_band'], bollinger['lower_band'], alpha=0.1, color='green')
            
            # Add support and resistance levels if available
            for level in signals['support_levels']:
                axs[0].axhline(y=level, color='g', linestyle='-', alpha=0.7)
            
            for level in signals['resistance_levels']:
                axs[0].axhline(y=level, color='r', linestyle='-', alpha=0.7)
            
            # Plot volume
            axs[1].bar(data.index, data['Volume'], color='blue', alpha=0.5)
            axs[1].set_ylabel('Volume')
            
            # Plot RSI
            rsi = TechnicalAnalysisTools.calculate_rsi(data)
            axs[2].plot(data.index, rsi, label='RSI', color='purple')
            axs[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axs[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
            axs[2].set_ylim(0, 100)
            axs[2].set_ylabel('RSI')
            
            # Set titles and labels
            fig.suptitle(f"{signals['company_name']} ({symbol}) - Technical Analysis", fontsize=16)
            axs[0].set_title(f"Current Price: ${signals['current_price']:.2f} ({signals['price_change']:+.2f}%) - Signal: {signals['overall_signal']}")
            axs[0].set_ylabel('Price ($)')
            
            # Add legend
            axs[0].legend()
            axs[2].legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path is None:
                # Create directory if it doesn't exist
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'technical_analysis')
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"technical_analysis_{symbol}_{timestamp}.png"
                save_path = os.path.join(output_dir, filename)
            
            plt.savefig(save_path)
            plt.close()
            
            # Add visualization path to results
            signals['visualization_path'] = save_path
            
            return signals
            
        except Exception as e:
            logger.error(f"Error creating technical analysis visualization for {symbol}: {str(e)}")
            return {"error": f"Error creating visualization: {str(e)}"}
