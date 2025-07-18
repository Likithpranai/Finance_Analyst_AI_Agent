"""
Technical Analysis Agent for Finance Analyst AI Agent Framework
"""
import re
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd

from agent_framework.agents.base_agent import BaseAgent
from tools.technical_analysis import StockTools
from tools.real_time_data_integration import RealTimeDataTools

class TechnicalAnalysisAgent(BaseAgent):
    """
    Specialized agent for technical analysis of financial instruments
    """
    
    def __init__(self):
        """Initialize the Technical Analysis Agent with specialized tools"""
        # Initialize with technical analysis tools
        tools = {
            # Standard technical analysis tools
            "get_stock_price": StockTools.get_stock_price,
            "get_stock_history": StockTools.get_stock_history,
            "calculate_rsi": StockTools.calculate_rsi,
            "calculate_macd": StockTools.calculate_macd,
            "visualize_stock": StockTools.visualize_stock,
            
            # Professional real-time data tools
            "get_real_time_quote": RealTimeDataTools.get_real_time_quote,
            "get_intraday_data": RealTimeDataTools.get_intraday_data,
            "get_historical_data": RealTimeDataTools.get_historical_data,
        }
        
        # Initialize the base agent
        super().__init__(
            name="Technical Analysis Agent",
            description="Specialized agent for technical analysis of financial instruments",
            tools=tools
        )
        
        # Add specialized system prompt for technical analysis
        self.system_prompt = """
        You are a Technical Analysis Agent specializing in analyzing financial markets using technical indicators and chart patterns.
        
        Follow the ReAct pattern: Reason → Act → Observe → Loop.
        
        When analyzing a financial instrument:
        1. REASON: Determine what technical indicators and chart patterns would be most relevant
        2. ACT: Use appropriate tools to gather price data and calculate technical indicators
        3. OBSERVE: Analyze the results and identify significant patterns or signals
        4. LOOP: If needed, gather additional data or calculate other indicators
        
        Focus on:
        - Price action and volume analysis
        - Trend identification (bullish, bearish, or sideways)
        - Support and resistance levels
        - Technical indicators (RSI, MACD, moving averages, Bollinger Bands)
        - Chart patterns (head and shoulders, double tops/bottoms, triangles, flags)
        - Divergences between price and indicators
        - Potential entry and exit points
        
        Always provide:
        - Clear identification of the current trend
        - Key support and resistance levels
        - Relevant technical indicator readings and their interpretations
        - Potential trade setups with risk/reward considerations
        - Time frame context (short-term, medium-term, long-term outlook)
        
        DO NOT make up information. Only use the data provided by the tools.
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
            return "3mo"  # Default to 3 months
    
    def determine_indicators(self, query: str) -> List[str]:
        """Determine which technical indicators to use based on the query"""
        indicators = []
        
        if "rsi" in query.lower():
            indicators.append("rsi")
        if "macd" in query.lower():
            indicators.append("macd")
        if "moving average" in query.lower() or "ma" in query.lower():
            indicators.append("sma")
        if "bollinger" in query.lower():
            indicators.append("bollinger")
        if "volume" in query.lower():
            indicators.append("volume")
        
        # If no specific indicators mentioned, use a default set
        if not indicators:
            indicators = ["sma", "rsi", "macd"]
            
        return indicators
    
    def process_query(self, query: str) -> str:
        """
        Process a technical analysis query
        
        Args:
            query: User query string
            
        Returns:
            Technical analysis response
        """
        # Extract symbol from query
        symbol = self.extract_symbol(query)
        if not symbol:
            return "I couldn't identify a stock symbol in your query. Please specify a valid stock symbol (e.g., AAPL, MSFT, GOOGL)."
        
        # Determine timeframe and indicators
        timeframe = self.determine_timeframe(query)
        indicators = self.determine_indicators(query)
        
        # Log the analysis plan
        print(f"REACT - REASON: Technical analysis for {symbol} over {timeframe} timeframe with indicators: {', '.join(indicators)}")
        
        # Gather data
        results = {}
        
        try:
            # Get real-time quote
            print(f"REACT - ACT: Getting real-time quote for {symbol}")
            results["quote"] = self.execute_tool("get_real_time_quote", symbol=symbol)
            print(f"REACT - OBSERVE: Got real-time quote for {symbol}")
            
            # Get historical data
            print(f"REACT - ACT: Getting historical data for {symbol}")
            results["history"] = self.execute_tool("get_historical_data", symbol=symbol, period=timeframe)
            print(f"REACT - OBSERVE: Got historical data for {symbol}")
            
            # Calculate technical indicators
            for indicator in indicators:
                if indicator == "rsi":
                    print(f"REACT - ACT: Calculating RSI for {symbol}")
                    results["rsi"] = self.execute_tool("calculate_rsi", symbol=symbol)
                    print(f"REACT - OBSERVE: Calculated RSI for {symbol}")
                elif indicator == "macd":
                    print(f"REACT - ACT: Calculating MACD for {symbol}")
                    results["macd"] = self.execute_tool("calculate_macd", symbol=symbol)
                    print(f"REACT - OBSERVE: Calculated MACD for {symbol}")
            
            # Generate visualization if requested
            if "chart" in query.lower() or "visual" in query.lower():
                print(f"REACT - ACT: Generating visualization for {symbol}")
                results["visualization"] = self.execute_tool("visualize_stock", symbol=symbol, period=timeframe, indicators=indicators)
                print(f"REACT - OBSERVE: Generated visualization for {symbol}")
            
            # Generate analysis using Gemini
            print(f"REACT - LOOP: Generating technical analysis for {symbol}")
            
            # Create prompt for Gemini
            prompt = f"""
            Provide a comprehensive technical analysis for {symbol} based on the following data:
            
            Real-time quote: {results.get('quote', 'Not available')}
            
            Historical data: {results.get('history', 'Not available')}
            
            RSI: {results.get('rsi', 'Not calculated')}
            
            MACD: {results.get('macd', 'Not calculated')}
            
            Follow the ReAct pattern (Reason → Act → Observe → Loop) and include:
            1. Summary of current price action and trend
            2. Key support and resistance levels
            3. Technical indicator analysis
            4. Chart pattern identification
            5. Trading outlook (bullish, bearish, or neutral)
            
            Format your response in markdown with clear sections.
            """
            
            analysis = self.generate_response(prompt)
            
            # Add visualization if available
            if "visualization" in results:
                analysis += f"\n\n{results['visualization']}"
            
            return analysis
            
        except Exception as e:
            error_message = f"Error performing technical analysis for {symbol}: {str(e)}"
            print(error_message)
            return error_message
