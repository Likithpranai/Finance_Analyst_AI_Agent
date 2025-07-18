"""
Trading Agent for Finance Analyst AI Agent Framework
"""
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

from agent_framework.agents.base_agent import BaseAgent
from tools.technical_analysis import StockTools
from tools.real_time_data_integration import RealTimeDataTools
from tools.backtesting import BacktestingTools

class TradingAgent(BaseAgent):
    """
    Specialized agent for generating trading signals and executing trades
    """
    
    def __init__(self):
        """Initialize the Trading Agent with specialized tools"""
        # Initialize with trading tools
        tools = {
            # Technical analysis tools for signal generation
            "get_stock_price": StockTools.get_stock_price,
            "get_stock_history": StockTools.get_stock_history,
            "calculate_rsi": StockTools.calculate_rsi,
            "calculate_macd": StockTools.calculate_macd,
            
            # Real-time data tools
            "get_real_time_quote": RealTimeDataTools.get_real_time_quote,
            "get_intraday_data": RealTimeDataTools.get_intraday_data,
            "start_real_time_stream": RealTimeDataTools.start_real_time_stream,
            "stop_real_time_stream": RealTimeDataTools.stop_real_time_stream,
            
            # Backtesting tools for strategy validation
            "backtest_strategy": BacktestingTools.backtest_strategy,
            "calculate_strategy_metrics": BacktestingTools.calculate_strategy_metrics,
            "visualize_backtest_results": BacktestingTools.visualize_backtest_results,
            
            # Trading execution tools (to be implemented with actual broker APIs)
            "place_market_order": lambda symbol, quantity, side: f"SIMULATED: Placed {side} market order for {quantity} shares of {symbol}",
            "place_limit_order": lambda symbol, quantity, price, side: f"SIMULATED: Placed {side} limit order for {quantity} shares of {symbol} at ${price}",
            "get_account_balance": lambda: {"cash": 100000, "equity": 150000, "buying_power": 200000},
            "get_positions": lambda: [{"symbol": "AAPL", "quantity": 100, "avg_price": 150.0, "current_price": 175.0, "market_value": 17500.0}],
        }
        
        # Initialize the base agent
        super().__init__(
            name="Trading Agent",
            description="Specialized agent for generating trading signals and executing trades",
            tools=tools
        )
        
        # Add specialized system prompt for trading
        self.system_prompt = """
        You are a Trading Agent specializing in generating trading signals and executing trades based on technical analysis and market conditions.
        
        Follow the ReAct pattern: Reason → Act → Observe → Loop.
        
        When analyzing trading opportunities:
        1. REASON: Determine what technical indicators and price patterns would be most relevant
        2. ACT: Use appropriate tools to gather price data and calculate technical indicators
        3. OBSERVE: Analyze the results and identify potential trading signals
        4. LOOP: If needed, gather additional data or validate signals with other indicators
        
        Focus on:
        - Trend identification and confirmation
        - Support and resistance levels
        - Technical indicator signals (RSI, MACD, moving averages)
        - Entry and exit points with specific price levels
        - Position sizing and risk management
        - Stop-loss and take-profit levels
        - Trade execution timing
        
        Always provide:
        - Clear trading recommendation (Buy, Sell, or Hold)
        - Specific entry price or price range
        - Stop-loss level with rationale
        - Take-profit target with rationale
        - Position sizing recommendation
        - Risk-reward ratio calculation
        
        DO NOT make up information. Only use the data provided by the tools.
        IMPORTANT: Clearly label all trading recommendations as simulated and not financial advice.
        """
    
    def extract_symbol(self, query: str) -> str:
        """Extract stock symbol from user query"""
        # Look for stock symbols in the query (typically 1-5 uppercase letters)
        matches = re.findall(r'\b[A-Z]{1,5}\b', query)
        if matches:
            return matches[0]
        
        # Look for common phrases like "for AAPL" or "of MSFT"
        symbol_match = re.search(r'(?:for|of|on|about)\s+([A-Z]{1,5})', query, re.IGNORECASE)
        if symbol_match:
            return symbol_match.group(1).upper()
        
        return ""
    
    def determine_timeframe(self, query: str) -> str:
        """Determine the appropriate timeframe from the query"""
        if any(term in query.lower() for term in ["day", "daily", "today", "intraday", "short-term"]):
            return "1d"
        elif any(term in query.lower() for term in ["week", "weekly"]):
            return "1wk"
        elif any(term in query.lower() for term in ["month", "monthly", "medium-term"]):
            return "1mo"
        elif any(term in query.lower() for term in ["year", "yearly", "long-term"]):
            return "1y"
        else:
            return "1d"  # Default to daily for trading
    
    def determine_strategy(self, query: str) -> str:
        """Determine the trading strategy based on the query"""
        if "trend" in query.lower() or "following" in query.lower():
            return "trend_following"
        elif "reversal" in query.lower() or "contrarian" in query.lower():
            return "reversal"
        elif "breakout" in query.lower():
            return "breakout"
        elif "swing" in query.lower():
            return "swing"
        elif "scalp" in query.lower():
            return "scalping"
        else:
            return "technical"  # Default to general technical analysis
    
    def process_query(self, query: str) -> str:
        """
        Process a trading query
        
        Args:
            query: User query string
            
        Returns:
            Trading analysis and recommendation
        """
        # Extract symbol from query
        symbol = self.extract_symbol(query)
        if not symbol:
            return "I couldn't identify a stock symbol in your query. Please specify a valid stock symbol (e.g., AAPL, MSFT, GOOGL)."
        
        # Determine timeframe and strategy
        timeframe = self.determine_timeframe(query)
        strategy = self.determine_strategy(query)
        
        # Log the analysis plan
        print(f"REACT - REASON: Trading analysis for {symbol} on {timeframe} timeframe using {strategy} strategy")
        
        # Gather data
        results = {}
        
        try:
            # Get real-time quote
            print(f"REACT - ACT: Getting real-time quote for {symbol}")
            results["quote"] = self.execute_tool("get_real_time_quote", symbol=symbol)
            print(f"REACT - OBSERVE: Got real-time quote for {symbol}")
            
            # Get historical data
            print(f"REACT - ACT: Getting historical data for {symbol}")
            results["history"] = self.execute_tool("get_stock_history", symbol=symbol, period=timeframe)
            print(f"REACT - OBSERVE: Got historical data for {symbol}")
            
            # Calculate technical indicators
            print(f"REACT - ACT: Calculating RSI for {symbol}")
            results["rsi"] = self.execute_tool("calculate_rsi", symbol=symbol)
            print(f"REACT - OBSERVE: Calculated RSI for {symbol}")
            
            print(f"REACT - ACT: Calculating MACD for {symbol}")
            results["macd"] = self.execute_tool("calculate_macd", symbol=symbol)
            print(f"REACT - OBSERVE: Calculated MACD for {symbol}")
            
            # Backtest the strategy if requested
            if "backtest" in query.lower():
                print(f"REACT - ACT: Backtesting {strategy} strategy for {symbol}")
                results["backtest"] = self.execute_tool("backtest_strategy", 
                                                       symbol=symbol, 
                                                       strategy_type=strategy,
                                                       period="1y")
                print(f"REACT - OBSERVE: Backtested {strategy} strategy for {symbol}")
                
                print(f"REACT - ACT: Calculating strategy metrics")
                results["strategy_metrics"] = self.execute_tool("calculate_strategy_metrics", 
                                                              backtest_results=results["backtest"])
                print(f"REACT - OBSERVE: Calculated strategy metrics")
            
            # Get account information if requested
            if "account" in query.lower() or "position" in query.lower():
                print(f"REACT - ACT: Getting account balance")
                results["account"] = self.execute_tool("get_account_balance")
                print(f"REACT - OBSERVE: Got account balance")
                
                print(f"REACT - ACT: Getting current positions")
                results["positions"] = self.execute_tool("get_positions")
                print(f"REACT - OBSERVE: Got current positions")
            
            # Generate trading signal using Gemini
            print(f"REACT - LOOP: Generating trading signal for {symbol}")
            
            # Create prompt for Gemini
            prompt = f"""
            Provide a trading analysis and recommendation for {symbol} based on the following data:
            
            Real-time quote: {results.get('quote', 'Not available')}
            
            Historical data: {results.get('history', 'Not available')}
            
            RSI: {results.get('rsi', 'Not available')}
            
            MACD: {results.get('macd', 'Not available')}
            
            Backtest results: {results.get('backtest', 'Not available')}
            
            Strategy metrics: {results.get('strategy_metrics', 'Not available')}
            
            Account information: {results.get('account', 'Not available')}
            
            Current positions: {results.get('positions', 'Not available')}
            
            Follow the ReAct pattern (Reason → Act → Observe → Loop) and include:
            1. Current market context and price action analysis
            2. Technical indicator signals and interpretation
            3. Support and resistance levels
            4. Clear trading recommendation (Buy, Sell, or Hold)
            5. Specific entry price or price range
            6. Stop-loss level with rationale
            7. Take-profit target with rationale
            8. Position sizing recommendation
            9. Risk-reward ratio calculation
            
            Format your response in markdown with clear sections.
            
            IMPORTANT: Clearly label all trading recommendations as simulated and not financial advice.
            """
            
            trading_signal = self.generate_response(prompt)
            
            # Add visualization if backtest was performed
            if "backtest" in results and "visualize" in query.lower():
                print(f"REACT - ACT: Generating backtest visualization")
                visualization = self.execute_tool("visualize_backtest_results", 
                                                backtest_results=results["backtest"])
                print(f"REACT - OBSERVE: Generated backtest visualization")
                trading_signal += f"\n\n{visualization}"
            
            return trading_signal
            
        except Exception as e:
            error_message = f"Error generating trading signal for {symbol}: {str(e)}"
            print(error_message)
            return error_message
