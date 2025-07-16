"""
Tools for calculating technical indicators for stock analysis
"""
import yfinance as yf
import pandas as pd
# Fix numpy NaN import issue for pandas_ta
from . import ta_fix
import pandas_ta as ta
import numpy as np
from langchain.tools import BaseTool
from typing import Dict, Any, List, Optional

from config import DEFAULT_STOCK_HISTORY_PERIOD, DEFAULT_STOCK_HISTORY_INTERVAL


class CalculateRSITool(BaseTool):
    """Tool for calculating Relative Strength Index (RSI)"""
    
    name: str = "calculate_rsi"
    description: str = """Calculates RSI (Relative Strength Index) for a stock. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - period (optional): Time period for historical data
    - interval (optional): Time interval for historical data
    - window (optional): RSI calculation window (default: 14)
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            period = kwargs.get("period", DEFAULT_STOCK_HISTORY_PERIOD)
            interval = kwargs.get("interval", DEFAULT_STOCK_HISTORY_INTERVAL)
            window = int(kwargs.get("window", 14))
            
            if not ticker:
                return "Error: No ticker symbol provided"
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                return f"No historical data available for {ticker}"
            
            # Calculate RSI
            rsi = ta.rsi(hist['Close'], length=window)
            
            # Get the latest values
            latest_rsi = rsi.iloc[-1] if not rsi.empty else None
            latest_close = hist['Close'].iloc[-1] if not hist.empty else None
            
            # Prepare interpretation
            interpretation = None
            if latest_rsi is not None:
                if latest_rsi < 30:
                    interpretation = "Potentially oversold"
                elif latest_rsi > 70:
                    interpretation = "Potentially overbought"
                else:
                    interpretation = "Neutral"
            
            # Format the result
            result = {
                "symbol": ticker,
                "latest_close": latest_close,
                "latest_rsi": latest_rsi,
                "window": window,
                "timestamp": hist.index[-1].strftime("%Y-%m-%d %H:%M:%S") if not hist.empty else None,
                "interpretation": interpretation,
                "historical_rsi": rsi.dropna().tolist()[-10:],  # Last 10 RSI values
                "historical_dates": [date.strftime("%Y-%m-%d") for date in rsi.dropna().index[-10:]]
            }
            
            return result
        
        except Exception as e:
            return f"Error calculating RSI for {ticker}: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class CalculateMovingAveragesTool(BaseTool):
    """Tool for calculating Moving Averages"""
    
    name: str = "calculate_moving_averages"
    description: str = """Calculates simple and exponential moving averages for a stock. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - period (optional): Time period for historical data
    - interval (optional): Time interval for historical data
    - ma_windows (optional): List of windows for MA calculation (default: [20, 50, 200])
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            period = kwargs.get("period", DEFAULT_STOCK_HISTORY_PERIOD)
            interval = kwargs.get("interval", DEFAULT_STOCK_HISTORY_INTERVAL)
            ma_windows = kwargs.get("ma_windows", [20, 50, 200])
            
            if isinstance(ma_windows, str):
                try:
                    ma_windows = [int(x) for x in ma_windows.replace('[', '').replace(']', '').split(',')]
                except:
                    ma_windows = [20, 50, 200]
            
            if not ticker:
                return "Error: No ticker symbol provided"
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                return f"No historical data available for {ticker}"
            
            # Calculate moving averages
            result = {
                "symbol": ticker,
                "latest_close": hist['Close'].iloc[-1],
                "timestamp": hist.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "sma": {},
                "ema": {},
                "signals": [],
            }
            
            # Calculate SMA and EMA for each window
            for window in ma_windows:
                # Simple Moving Average
                sma = ta.sma(hist['Close'], length=window)
                result["sma"][f"SMA_{window}"] = sma.iloc[-1] if not sma.empty else None
                
                # Exponential Moving Average
                ema = ta.ema(hist['Close'], length=window)
                result["ema"][f"EMA_{window}"] = ema.iloc[-1] if not ema.empty else None
                
                # Generate signals based on price vs MA
                if not sma.empty and result["latest_close"] is not None and sma.iloc[-1] is not None:
                    if result["latest_close"] > sma.iloc[-1]:
                        result["signals"].append(f"Price above SMA_{window} - potential bullish signal")
                    elif result["latest_close"] < sma.iloc[-1]:
                        result["signals"].append(f"Price below SMA_{window} - potential bearish signal")
                
                if not ema.empty and result["latest_close"] is not None and ema.iloc[-1] is not None:
                    if result["latest_close"] > ema.iloc[-1]:
                        result["signals"].append(f"Price above EMA_{window} - potential bullish signal")
                    elif result["latest_close"] < ema.iloc[-1]:
                        result["signals"].append(f"Price below EMA_{window} - potential bearish signal")
            
            # Golden Cross / Death Cross detection (if we have SMA 50 and 200)
            if "SMA_50" in result["sma"] and "SMA_200" in result["sma"]:
                if len(sma) > 2 and len(hist) > 200:  # Make sure we have enough data
                    # Check current position
                    if result["sma"]["SMA_50"] > result["sma"]["SMA_200"]:
                        result["signals"].append("SMA 50 above SMA 200 - bullish configuration")
                    else:
                        result["signals"].append("SMA 50 below SMA 200 - bearish configuration")
                    
                    # Check if we just had a cross (using data from yesterday and today)
                    sma_50_yesterday = ta.sma(hist['Close'], length=50).iloc[-2]
                    sma_200_yesterday = ta.sma(hist['Close'], length=200).iloc[-2]
                    sma_50_today = result["sma"]["SMA_50"]
                    sma_200_today = result["sma"]["SMA_200"]
                    
                    if sma_50_yesterday < sma_200_yesterday and sma_50_today > sma_200_today:
                        result["signals"].append("GOLDEN CROSS DETECTED: SMA 50 crossed above SMA 200 - strong bullish signal")
                    elif sma_50_yesterday > sma_200_yesterday and sma_50_today < sma_200_today:
                        result["signals"].append("DEATH CROSS DETECTED: SMA 50 crossed below SMA 200 - strong bearish signal")
            
            return result
        
        except Exception as e:
            return f"Error calculating Moving Averages for {ticker}: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class CalculateMACDTool(BaseTool):
    """Tool for calculating MACD (Moving Average Convergence Divergence)"""
    
    name: str = "calculate_macd"
    description: str = """Calculates MACD (Moving Average Convergence Divergence) for a stock. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - period (optional): Time period for historical data
    - interval (optional): Time interval for historical data
    - fast (optional): Fast EMA period (default: 12)
    - slow (optional): Slow EMA period (default: 26)
    - signal (optional): Signal period (default: 9)
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            period = kwargs.get("period", DEFAULT_STOCK_HISTORY_PERIOD)
            interval = kwargs.get("interval", DEFAULT_STOCK_HISTORY_INTERVAL)
            fast = int(kwargs.get("fast", 12))
            slow = int(kwargs.get("slow", 26))
            signal = int(kwargs.get("signal", 9))
            
            if not ticker:
                return "Error: No ticker symbol provided"
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                return f"No historical data available for {ticker}"
            
            # Calculate MACD
            macd_result = ta.macd(hist['Close'], fast=fast, slow=slow, signal=signal)
            
            # Get the latest values
            macd_line = macd_result[f"MACD_{fast}_{slow}_{signal}"].iloc[-1] if not macd_result.empty else None
            signal_line = macd_result[f"MACDs_{fast}_{slow}_{signal}"].iloc[-1] if not macd_result.empty else None
            histogram = macd_result[f"MACDh_{fast}_{slow}_{signal}"].iloc[-1] if not macd_result.empty else None
            
            # Prepare interpretation
            interpretation = None
            signal_strength = "weak"
            
            if macd_line is not None and signal_line is not None:
                if macd_line > signal_line:
                    interpretation = "Bullish"
                    if abs(macd_line - signal_line) > 0.5:  # Adjust threshold as needed
                        signal_strength = "strong"
                elif macd_line < signal_line:
                    interpretation = "Bearish"
                    if abs(macd_line - signal_line) > 0.5:  # Adjust threshold as needed
                        signal_strength = "strong"
                else:
                    interpretation = "Neutral"
            
            # Check for recent crossover
            recent_macd = macd_result[f"MACD_{fast}_{slow}_{signal}"].iloc[-3:].tolist()
            recent_signal = macd_result[f"MACDs_{fast}_{slow}_{signal}"].iloc[-3:].tolist()
            
            crossover = None
            if len(recent_macd) == 3 and len(recent_signal) == 3:
                if recent_macd[0] < recent_signal[0] and recent_macd[-1] > recent_signal[-1]:
                    crossover = "Bullish crossover (MACD crossed above Signal Line)"
                elif recent_macd[0] > recent_signal[0] and recent_macd[-1] < recent_signal[-1]:
                    crossover = "Bearish crossover (MACD crossed below Signal Line)"
            
            # Format the result
            result = {
                "symbol": ticker,
                "latest_close": hist['Close'].iloc[-1],
                "timestamp": hist.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "interpretation": interpretation,
                "signal_strength": signal_strength,
                "crossover": crossover,
                "parameters": {
                    "fast": fast,
                    "slow": slow,
                    "signal": signal
                }
            }
            
            return result
        
        except Exception as e:
            return f"Error calculating MACD for {ticker}: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class CalculateBollingerBandsTool(BaseTool):
    """Tool for calculating Bollinger Bands"""
    
    name: str = "calculate_bollinger_bands"
    description: str = """Calculates Bollinger Bands for a stock. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - period (optional): Time period for historical data
    - interval (optional): Time interval for historical data
    - window (optional): Bollinger Bands window (default: 20)
    - std_dev (optional): Standard deviation factor (default: 2)
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            period = kwargs.get("period", DEFAULT_STOCK_HISTORY_PERIOD)
            interval = kwargs.get("interval", DEFAULT_STOCK_HISTORY_INTERVAL)
            window = int(kwargs.get("window", 20))
            std_dev = float(kwargs.get("std_dev", 2.0))
            
            if not ticker:
                return "Error: No ticker symbol provided"
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                return f"No historical data available for {ticker}"
            
            # Calculate Bollinger Bands
            bbands = ta.bbands(hist['Close'], length=window, std=std_dev)
            
            # Get the latest values
            latest_close = hist['Close'].iloc[-1]
            upper_band = bbands[f"BBU_{window}_{std_dev}"].iloc[-1]
            middle_band = bbands[f"BBM_{window}_{std_dev}"].iloc[-1]
            lower_band = bbands[f"BBL_{window}_{std_dev}"].iloc[-1]
            
            # Calculate bandwidth and %B
            bandwidth = (upper_band - lower_band) / middle_band
            percent_b = (latest_close - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else None
            
            # Prepare interpretation
            interpretation = None
            if latest_close > upper_band:
                interpretation = "Price above upper band - potentially overbought"
            elif latest_close < lower_band:
                interpretation = "Price below lower band - potentially oversold"
            else:
                interpretation = "Price within bands - neutral"
            
            # Format the result
            result = {
                "symbol": ticker,
                "latest_close": latest_close,
                "timestamp": hist.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "upper_band": upper_band,
                "middle_band": middle_band,
                "lower_band": lower_band,
                "bandwidth": bandwidth,
                "percent_b": percent_b,
                "interpretation": interpretation,
                "parameters": {
                    "window": window,
                    "std_dev": std_dev
                }
            }
            
            return result
        
        except Exception as e:
            return f"Error calculating Bollinger Bands for {ticker}: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)
