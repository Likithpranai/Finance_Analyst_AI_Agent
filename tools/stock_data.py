import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from langchain.tools import BaseTool
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from config import DEFAULT_STOCK_HISTORY_PERIOD, DEFAULT_STOCK_HISTORY_INTERVAL


class GetStockPriceTool(BaseTool):
    """Tool for getting current stock price data"""
    
    name: str = "get_stock_price"
    description: str = "Gets the current price of a stock. Input should be a valid stock ticker symbol."
    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """Run the tool"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get the current price data
            data = {
                "symbol": ticker,
                "currentPrice": info.get("currentPrice", info.get("regularMarketPrice", None)),
                "previousClose": info.get("previousClose", None),
                "open": info.get("open", None),
                "dayHigh": info.get("dayHigh", None),
                "dayLow": info.get("dayLow", None),
                "volume": info.get("volume", None),
                "marketCap": info.get("marketCap", None),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Calculate price change and percentage
            if data["currentPrice"] and data["previousClose"]:
                data["change"] = data["currentPrice"] - data["previousClose"]
                data["changePercent"] = (data["change"] / data["previousClose"]) * 100
            
            return data
            
        except Exception as e:
            return f"Error fetching stock data for {ticker}: {str(e)}"
    
    def _arun(self, ticker: str):
        """Run the tool asynchronously"""
        return self._run(ticker)


class GetStockHistoryTool(BaseTool):
    """Tool for getting historical stock price data"""
    
    name: str = "get_stock_history"
    description: str = """Gets historical stock price data. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - period (optional): Time period, e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    - interval (optional): Time interval, e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            period = kwargs.get("period", DEFAULT_STOCK_HISTORY_PERIOD)
            interval = kwargs.get("interval", DEFAULT_STOCK_HISTORY_INTERVAL)
            
            if not ticker:
                return "Error: No ticker symbol provided"
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            # Format the data for return
            data = {
                "symbol": ticker,
                "period": period,
                "interval": interval,
                "history": hist.reset_index().to_dict(orient="records"),
                "start_date": hist.index[0].strftime("%Y-%m-%d"),
                "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                "latest_price": hist["Close"][-1] if not hist.empty else None,
            }
            
            # Add summary statistics
            if not hist.empty:
                data["summary"] = {
                    "min": hist["Close"].min(),
                    "max": hist["Close"].max(),
                    "mean": hist["Close"].mean(),
                    "std": hist["Close"].std(),
                    "total_return": ((hist["Close"][-1] / hist["Close"][0]) - 1) * 100 if hist["Close"][0] > 0 else None
                }
            
            return data
            
        except Exception as e:
            return f"Error fetching historical stock data: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class PlotStockPriceTool(BaseTool):
    """Tool for plotting stock price data"""
    
    name: str = "plot_stock_price"
    description: str = """Generates a plot of historical stock prices. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - period (optional): Time period, e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    - interval (optional): Time interval, e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
    - plot_type (optional): Type of plot, 'line' (default) or 'candle'
    """
    
    def _run(self, **kwargs) -> str:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            period = kwargs.get("period", DEFAULT_STOCK_HISTORY_PERIOD)
            interval = kwargs.get("interval", DEFAULT_STOCK_HISTORY_INTERVAL)
            plot_type = kwargs.get("plot_type", "line")
            
            if not ticker:
                return "Error: No ticker symbol provided"
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                return f"No historical data available for {ticker} with the specified parameters"
            
            plt.figure(figsize=(10, 6))
            
            if plot_type.lower() == "candle":
                # Create a candlestick-like chart
                plt.bar(hist.index, 
                       hist['Close'] - hist['Open'], 
                       bottom=hist['Open'], 
                       width=0.6, 
                       color=["green" if close > open else "red" 
                              for close, open in zip(hist['Close'], hist['Open'])])
                plt.bar(hist.index, 
                       hist['High'] - hist['Close'], 
                       bottom=hist['Close'], 
                       width=0.1, 
                       color='black')
                plt.bar(hist.index, 
                       hist['Low'] - hist['Open'], 
                       bottom=hist['Open'], 
                       width=0.1, 
                       color='black')
                plt.title(f"{ticker} Stock Price (Candlestick)")
            else:
                # Default line chart
                plt.plot(hist.index, hist['Close'], label='Close Price', color='blue')
                plt.title(f"{ticker} Stock Price")
                
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to a base64 encoded string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return f"Chart generated for {ticker} over {period} period with {interval} intervals."
            
        except Exception as e:
            return f"Error generating stock price plot: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class GetCompanyInfoTool(BaseTool):
    """Tool for getting company information"""
    
    name: str = "get_company_info"
    description: str = "Gets basic information about a company. Input should be a valid stock ticker symbol."
    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """Run the tool"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant company information
            data = {
                "symbol": ticker,
                "name": info.get("longName", info.get("shortName", None)),
                "industry": info.get("industry", None),
                "sector": info.get("sector", None),
                "website": info.get("website", None),
                "description": info.get("longBusinessSummary", None),
                "employees": info.get("fullTimeEmployees", None),
                "country": info.get("country", None),
                "city": info.get("city", None),
                "state": info.get("state", None),
                "exchange": info.get("exchange", None),
                "currency": info.get("currency", None),
            }
            
            # Add financial metrics if available
            metrics = {
                "marketCap": info.get("marketCap", None),
                "forwardPE": info.get("forwardPE", None),
                "trailingPE": info.get("trailingPE", None),
                "dividend_yield": info.get("dividendYield", None) * 100 if info.get("dividendYield") else None,
                "earnings_growth": info.get("earningsGrowth", None) * 100 if info.get("earningsGrowth") else None,
                "revenue_growth": info.get("revenueGrowth", None) * 100 if info.get("revenueGrowth") else None,
                "profit_margins": info.get("profitMargins", None) * 100 if info.get("profitMargins") else None,
                "beta": info.get("beta", None),
            }
            
            data["metrics"] = metrics
            
            return data
            
        except Exception as e:
            return f"Error fetching company information for {ticker}: {str(e)}"
    
    def _arun(self, ticker: str):
        """Run the tool asynchronously"""
        return self._run(ticker)
