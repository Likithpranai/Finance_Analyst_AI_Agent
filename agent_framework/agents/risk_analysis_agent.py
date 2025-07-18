"""
Risk Analysis Agent for Finance Analyst AI Agent Framework
"""
import re
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

from agent_framework.agents.base_agent import BaseAgent
from tools.portfolio_management import PortfolioTools
from tools.backtesting import BacktestingTools
from tools.real_time_data_integration import RealTimeDataTools

class RiskAnalysisAgent(BaseAgent):
    """
    Specialized agent for risk analysis of financial instruments and portfolios
    """
    
    def __init__(self):
        """Initialize the Risk Analysis Agent with specialized tools"""
        # Initialize with risk analysis tools
        tools = {
            # Portfolio risk tools
            "calculate_portfolio_risk": PortfolioTools.calculate_portfolio_risk,
            "calculate_var": PortfolioTools.calculate_var,
            "calculate_sharpe_ratio": PortfolioTools.calculate_sharpe_ratio,
            "calculate_sortino_ratio": PortfolioTools.calculate_sortino_ratio,
            "calculate_max_drawdown": PortfolioTools.calculate_max_drawdown,
            "calculate_beta": PortfolioTools.calculate_beta,
            "calculate_correlation_matrix": PortfolioTools.calculate_correlation_matrix,
            "visualize_portfolio_risk": PortfolioTools.visualize_portfolio_risk,
            
            # Backtesting tools for risk assessment
            "backtest_strategy": BacktestingTools.backtest_strategy,
            "calculate_strategy_metrics": BacktestingTools.calculate_strategy_metrics,
            "visualize_backtest_results": BacktestingTools.visualize_backtest_results,
            
            # Real-time data for volatility analysis
            "get_historical_data": RealTimeDataTools.get_historical_data,
            "get_real_time_quote": RealTimeDataTools.get_real_time_quote,
        }
        
        # Initialize the base agent
        super().__init__(
            name="Risk Analysis Agent",
            description="Specialized agent for risk analysis of financial instruments and portfolios",
            tools=tools
        )
        
        # Add specialized system prompt for risk analysis
        self.system_prompt = """
        You are a Risk Analysis Agent specializing in evaluating market risks, volatility, and portfolio risk metrics.
        
        Follow the ReAct pattern: Reason → Act → Observe → Loop.
        
        When analyzing risk:
        1. REASON: Determine what risk metrics and analysis would be most relevant
        2. ACT: Use appropriate tools to calculate risk metrics and perform stress tests
        3. OBSERVE: Analyze the results and identify key risk factors and exposures
        4. LOOP: If needed, gather additional data or calculate other risk metrics
        
        Focus on:
        - Volatility analysis (historical and implied)
        - Value at Risk (VaR) calculations
        - Maximum drawdown assessment
        - Risk-adjusted return metrics (Sharpe, Sortino, Calmar ratios)
        - Beta and correlation analysis
        - Stress testing and scenario analysis
        - Portfolio diversification assessment
        - Tail risk evaluation
        
        Always provide:
        - Clear quantification of risk exposures
        - Comparative risk metrics against benchmarks
        - Diversification recommendations
        - Risk mitigation strategies
        - Potential hedging approaches
        
        DO NOT make up information. Only use the data provided by the tools.
        """
    
    def extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from user query"""
        # Look for stock symbols in the query (typically 1-5 uppercase letters)
        matches = re.findall(r'\b[A-Z]{1,5}\b', query)
        if matches:
            return matches
        
        return []
    
    def determine_risk_metrics(self, query: str) -> List[str]:
        """Determine which risk metrics to calculate based on the query"""
        risk_metrics = []
        
        if any(term in query.lower() for term in ["volatility", "standard deviation", "std"]):
            risk_metrics.append("volatility")
        
        if any(term in query.lower() for term in ["var", "value at risk"]):
            risk_metrics.append("var")
        
        if any(term in query.lower() for term in ["sharpe", "risk-adjusted"]):
            risk_metrics.append("sharpe")
        
        if any(term in query.lower() for term in ["sortino"]):
            risk_metrics.append("sortino")
        
        if any(term in query.lower() for term in ["drawdown", "maximum drawdown", "max drawdown"]):
            risk_metrics.append("max_drawdown")
        
        if any(term in query.lower() for term in ["beta", "market risk"]):
            risk_metrics.append("beta")
        
        if any(term in query.lower() for term in ["correlation", "diversification"]):
            risk_metrics.append("correlation")
        
        # If no specific metrics mentioned, use a comprehensive set
        if not risk_metrics:
            risk_metrics = ["volatility", "var", "sharpe", "max_drawdown", "beta"]
            
        return risk_metrics
    
    def process_query(self, query: str) -> str:
        """
        Process a risk analysis query
        
        Args:
            query: User query string
            
        Returns:
            Risk analysis response
        """
        # Extract symbols from query
        symbols = self.extract_symbols(query)
        if not symbols:
            return "I couldn't identify any stock symbols in your query. Please specify valid stock symbols (e.g., AAPL, MSFT, GOOGL)."
        
        # Determine risk metrics to calculate
        risk_metrics = self.determine_risk_metrics(query)
        
        # Log the analysis plan
        print(f"REACT - REASON: Risk analysis for {', '.join(symbols)} with metrics: {', '.join(risk_metrics)}")
        
        # Gather data
        results = {}
        
        try:
            # Get historical data for all symbols
            print(f"REACT - ACT: Getting historical data for {', '.join(symbols)}")
            historical_data = {}
            for symbol in symbols:
                historical_data[symbol] = self.execute_tool("get_historical_data", symbol=symbol, period="1y")
            results["historical_data"] = historical_data
            print(f"REACT - OBSERVE: Got historical data for {', '.join(symbols)}")
            
            # Calculate portfolio risk metrics
            if len(symbols) > 1:
                print(f"REACT - ACT: Calculating portfolio risk for {', '.join(symbols)}")
                # Create equal-weighted portfolio for analysis
                weights = [1.0/len(symbols)] * len(symbols)
                results["portfolio_risk"] = self.execute_tool("calculate_portfolio_risk", symbols=symbols, weights=weights)
                print(f"REACT - OBSERVE: Calculated portfolio risk for {', '.join(symbols)}")
                
                # Calculate correlation matrix
                if "correlation" in risk_metrics:
                    print(f"REACT - ACT: Calculating correlation matrix for {', '.join(symbols)}")
                    results["correlation_matrix"] = self.execute_tool("calculate_correlation_matrix", symbols=symbols)
                    print(f"REACT - OBSERVE: Calculated correlation matrix for {', '.join(symbols)}")
            
            # Calculate individual risk metrics for each symbol
            for symbol in symbols:
                symbol_results = {}
                
                for metric in risk_metrics:
                    if metric == "volatility":
                        # Volatility is calculated as part of portfolio risk
                        continue
                    elif metric == "var":
                        print(f"REACT - ACT: Calculating VaR for {symbol}")
                        symbol_results["var"] = self.execute_tool("calculate_var", symbol=symbol, confidence_level=0.95)
                        print(f"REACT - OBSERVE: Calculated VaR for {symbol}")
                    elif metric == "sharpe":
                        print(f"REACT - ACT: Calculating Sharpe ratio for {symbol}")
                        symbol_results["sharpe"] = self.execute_tool("calculate_sharpe_ratio", symbol=symbol)
                        print(f"REACT - OBSERVE: Calculated Sharpe ratio for {symbol}")
                    elif metric == "sortino":
                        print(f"REACT - ACT: Calculating Sortino ratio for {symbol}")
                        symbol_results["sortino"] = self.execute_tool("calculate_sortino_ratio", symbol=symbol)
                        print(f"REACT - OBSERVE: Calculated Sortino ratio for {symbol}")
                    elif metric == "max_drawdown":
                        print(f"REACT - ACT: Calculating maximum drawdown for {symbol}")
                        symbol_results["max_drawdown"] = self.execute_tool("calculate_max_drawdown", symbol=symbol)
                        print(f"REACT - OBSERVE: Calculated maximum drawdown for {symbol}")
                    elif metric == "beta":
                        print(f"REACT - ACT: Calculating beta for {symbol}")
                        symbol_results["beta"] = self.execute_tool("calculate_beta", symbol=symbol, benchmark="SPY")
                        print(f"REACT - OBSERVE: Calculated beta for {symbol}")
                
                results[symbol] = symbol_results
            
            # Generate visualization if requested
            if "chart" in query.lower() or "visual" in query.lower() or "graph" in query.lower():
                if len(symbols) > 1:
                    print(f"REACT - ACT: Generating portfolio risk visualization")
                    results["visualization"] = self.execute_tool("visualize_portfolio_risk", symbols=symbols, weights=weights)
                    print(f"REACT - OBSERVE: Generated portfolio risk visualization")
                else:
                    # For single symbol, visualize risk metrics over time
                    symbol = symbols[0]
                    print(f"REACT - ACT: Generating risk visualization for {symbol}")
                    results["visualization"] = self.execute_tool("visualize_portfolio_risk", symbols=[symbol], weights=[1.0])
                    print(f"REACT - OBSERVE: Generated risk visualization for {symbol}")
            
            # Generate analysis using Gemini
            print(f"REACT - LOOP: Generating risk analysis for {', '.join(symbols)}")
            
            # Create prompt for Gemini
            prompt = f"""
            Provide a comprehensive risk analysis for {', '.join(symbols)} based on the following data:
            
            Historical data: {results.get('historical_data', 'Not available')}
            
            Portfolio risk metrics: {results.get('portfolio_risk', 'Not available')}
            
            Correlation matrix: {results.get('correlation_matrix', 'Not available')}
            
            Individual risk metrics:
            """
            
            for symbol in symbols:
                prompt += f"\n{symbol}: {results.get(symbol, 'Not available')}"
            
            prompt += """
            
            Follow the ReAct pattern (Reason → Act → Observe → Loop) and include:
            1. Volatility assessment (historical and relative to market)
            2. Value at Risk (VaR) interpretation
            3. Risk-adjusted return analysis (Sharpe, Sortino ratios)
            4. Maximum drawdown analysis
            5. Beta and market risk exposure
            6. Correlation and diversification benefits (for multiple symbols)
            7. Risk mitigation recommendations
            
            Format your response in markdown with clear sections.
            """
            
            analysis = self.generate_response(prompt)
            
            # Add visualization if available
            if "visualization" in results:
                analysis += f"\n\n{results['visualization']}"
            
            return analysis
            
        except Exception as e:
            error_message = f"Error performing risk analysis for {', '.join(symbols)}: {str(e)}"
            print(error_message)
            return error_message
