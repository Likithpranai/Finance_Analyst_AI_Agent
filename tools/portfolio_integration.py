"""
Portfolio Management and Backtesting Integration for Finance Analyst AI Agent

This module integrates portfolio management and backtesting capabilities into the
ReAct pattern of the Finance Analyst AI Agent, allowing for multi-symbol analysis,
portfolio optimization, risk metrics calculation, and strategy backtesting.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from tools.portfolio_management import PortfolioManagementTools
from tools.backtesting import BacktestingTools
from tools.backtesting_visualization import BacktestVisualizationTools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioIntegrationTools:
    """Tools for integrating portfolio management and backtesting into the Finance Analyst AI Agent"""
    
    @staticmethod
    def extract_portfolio_symbols(query: str) -> List[str]:
        """
        Extract portfolio symbols from a query string
        
        Args:
            query: The user query string
            
        Returns:
            List of stock symbols
        """
        # Look for portfolio-specific patterns
        portfolio_patterns = [
            r"portfolio\s+(?:with|containing|of|including)?\s+([A-Z]{1,5}(?:,\s*[A-Z]{1,5})*)",
            r"stocks?\s+([A-Z]{1,5}(?:,\s*[A-Z]{1,5})*)",
            r"symbols?\s+([A-Z]{1,5}(?:,\s*[A-Z]{1,5})*)",
            r"tickers?\s+([A-Z]{1,5}(?:,\s*[A-Z]{1,5})*)",
            r"companies?\s+([A-Z]{1,5}(?:,\s*[A-Z]{1,5})*)",
            r"basket\s+(?:of)?\s+([A-Z]{1,5}(?:,\s*[A-Z]{1,5})*)",
            r"([A-Z]{1,5}(?:,\s*[A-Z]{1,5})+)"  # Fallback pattern for comma-separated symbols
        ]
        
        for pattern in portfolio_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Extract and clean symbols
                symbols_str = match.group(1)
                symbols = [s.strip() for s in symbols_str.split(',')]
                return symbols
        
        # If no portfolio pattern is found, look for individual stock symbols
        stock_pattern = r'\b[A-Z]{1,5}\b'
        symbols = re.findall(stock_pattern, query)
        
        # Filter out common words that might be mistaken for symbols
        common_words = {'A', 'I', 'ME', 'MY', 'IT', 'FOR', 'TO', 'IN', 'IS', 'AND', 'OR', 'THE', 'ON'}
        symbols = [s for s in symbols if s not in common_words]
        
        return symbols
    
    @staticmethod
    def extract_weights(query: str, symbols: List[str]) -> Optional[List[float]]:
        """
        Extract portfolio weights from a query string
        
        Args:
            query: The user query string
            symbols: List of stock symbols
            
        Returns:
            List of weights or None if not found
        """
        # Look for weight patterns
        weight_patterns = [
            r"weights?\s+(?:of)?\s+([\d.]+%?(?:,\s*[\d.]+%?)*)",
            r"allocations?\s+(?:of)?\s+([\d.]+%?(?:,\s*[\d.]+%?)*)",
            r"proportions?\s+(?:of)?\s+([\d.]+%?(?:,\s*[\d.]+%?)*)"
        ]
        
        for pattern in weight_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Extract and clean weights
                weights_str = match.group(1)
                weights_raw = [w.strip() for w in weights_str.split(',')]
                
                # Convert percentages to decimals
                weights = []
                for w in weights_raw:
                    if '%' in w:
                        weights.append(float(w.replace('%', '')) / 100)
                    else:
                        weights.append(float(w))
                
                # Validate weights
                if len(weights) != len(symbols):
                    return None
                
                # Normalize weights if they don't sum to 1
                if abs(sum(weights) - 1.0) > 1e-6:
                    weights = [w / sum(weights) for w in weights]
                
                return weights
        
        return None
    
    @staticmethod
    def extract_strategy_parameters(query: str, strategy_type: str) -> Dict:
        """
        Extract strategy parameters from a query string
        
        Args:
            query: The user query string
            strategy_type: Type of strategy ('sma_crossover', 'rsi', 'macd')
            
        Returns:
            Dictionary of strategy parameters
        """
        parameters = {}
        
        if strategy_type.lower() == 'sma_crossover':
            # Extract SMA parameters
            short_window_match = re.search(r'short\s+(?:window|period|ma|sma)\s+(?:of)?\s+(\d+)', query, re.IGNORECASE)
            if short_window_match:
                parameters['short_window'] = int(short_window_match.group(1))
            else:
                parameters['short_window'] = 50  # Default
            
            long_window_match = re.search(r'long\s+(?:window|period|ma|sma)\s+(?:of)?\s+(\d+)', query, re.IGNORECASE)
            if long_window_match:
                parameters['long_window'] = int(long_window_match.group(1))
            else:
                parameters['long_window'] = 200  # Default
                
        elif strategy_type.lower() == 'rsi':
            # Extract RSI parameters
            period_match = re.search(r'rsi\s+(?:period|window)\s+(?:of)?\s+(\d+)', query, re.IGNORECASE)
            if period_match:
                parameters['rsi_period'] = int(period_match.group(1))
            else:
                parameters['rsi_period'] = 14  # Default
            
            overbought_match = re.search(r'overbought\s+(?:level|threshold)?\s+(?:of)?\s+(\d+)', query, re.IGNORECASE)
            if overbought_match:
                parameters['overbought'] = int(overbought_match.group(1))
            else:
                parameters['overbought'] = 70  # Default
            
            oversold_match = re.search(r'oversold\s+(?:level|threshold)?\s+(?:of)?\s+(\d+)', query, re.IGNORECASE)
            if oversold_match:
                parameters['oversold'] = int(oversold_match.group(1))
            else:
                parameters['oversold'] = 30  # Default
                
        elif strategy_type.lower() == 'macd':
            # Extract MACD parameters
            fast_match = re.search(r'fast\s+(?:period|window|ema)\s+(?:of)?\s+(\d+)', query, re.IGNORECASE)
            if fast_match:
                parameters['fast_period'] = int(fast_match.group(1))
            else:
                parameters['fast_period'] = 12  # Default
            
            slow_match = re.search(r'slow\s+(?:period|window|ema)\s+(?:of)?\s+(\d+)', query, re.IGNORECASE)
            if slow_match:
                parameters['slow_period'] = int(slow_match.group(1))
            else:
                parameters['slow_period'] = 26  # Default
            
            signal_match = re.search(r'signal\s+(?:period|window|line)\s+(?:of)?\s+(\d+)', query, re.IGNORECASE)
            if signal_match:
                parameters['signal_period'] = int(signal_match.group(1))
            else:
                parameters['signal_period'] = 9  # Default
        
        # Extract common parameters
        period_match = re.search(r'(?:for|over|using|with)\s+(?:the\s+)?(?:last\s+)?(\d+)\s+(?:years?|months?|days?)', query, re.IGNORECASE)
        if period_match:
            value = int(period_match.group(1))
            unit = period_match.group(2).lower()
            
            if 'year' in unit:
                parameters['period'] = f"{value}y"
            elif 'month' in unit:
                parameters['period'] = f"{value}mo"
            elif 'day' in unit:
                parameters['period'] = f"{value}d"
            else:
                parameters['period'] = "5y"  # Default
        else:
            parameters['period'] = "5y"  # Default
        
        # Extract initial capital
        capital_match = re.search(r'(?:with|using)\s+(?:an?\s+)?(?:initial\s+)?(?:capital|investment)\s+(?:of)?\s+\$?(\d+(?:,\d+)*(?:\.\d+)?)', query, re.IGNORECASE)
        if capital_match:
            capital_str = capital_match.group(1).replace(',', '')
            parameters['initial_capital'] = float(capital_str)
        else:
            parameters['initial_capital'] = 10000.0  # Default
        
        return parameters
    
    @staticmethod
    def analyze_portfolio(query: str) -> Dict:
        """
        Analyze a portfolio based on the query
        
        Args:
            query: The user query string
            
        Returns:
            Dictionary with portfolio analysis results
        """
        try:
            # Extract symbols from query
            symbols = PortfolioIntegrationTools.extract_portfolio_symbols(query)
            
            if not symbols:
                return {"error": "No stock symbols found in the query"}
            
            # Extract weights if provided
            weights = PortfolioIntegrationTools.extract_weights(query, symbols)
            
            # Determine time period
            period_match = re.search(r'(?:for|over|using|with)\s+(?:the\s+)?(?:last\s+)?(\d+)\s+(years?|months?|days?)', query, re.IGNORECASE)
            if period_match:
                value = int(period_match.group(1))
                unit = period_match.group(2).lower()
                
                if 'year' in unit:
                    period = f"{value}y"
                elif 'month' in unit:
                    period = f"{value}mo"
                elif 'day' in unit:
                    period = f"{value}d"
                else:
                    period = "1y"  # Default
            else:
                period = "1y"  # Default
            
            # Determine analysis type
            if re.search(r'optimize|optimization|efficient\s+frontier', query, re.IGNORECASE):
                # Portfolio optimization
                if re.search(r'sharpe|maximum\s+sharpe', query, re.IGNORECASE):
                    objective = "sharpe"
                elif re.search(r'minimum\s+(?:risk|volatility)', query, re.IGNORECASE):
                    objective = "min_volatility"
                elif re.search(r'maximum\s+(?:return|performance)', query, re.IGNORECASE):
                    objective = "max_return"
                else:
                    objective = "sharpe"  # Default
                
                # Check for target constraints
                constraints = {}
                target_return_match = re.search(r'target\s+return\s+(?:of)?\s+([\d.]+%?)', query, re.IGNORECASE)
                if target_return_match:
                    target_return = target_return_match.group(1)
                    if '%' in target_return:
                        constraints['target_return'] = float(target_return.replace('%', '')) / 100
                    else:
                        constraints['target_return'] = float(target_return)
                
                target_vol_match = re.search(r'target\s+(?:risk|volatility)\s+(?:of)?\s+([\d.]+%?)', query, re.IGNORECASE)
                if target_vol_match:
                    target_vol = target_vol_match.group(1)
                    if '%' in target_vol:
                        constraints['target_volatility'] = float(target_vol.replace('%', '')) / 100
                    else:
                        constraints['target_volatility'] = float(target_vol)
                
                # Generate efficient frontier if requested
                if re.search(r'efficient\s+frontier', query, re.IGNORECASE):
                    num_portfolios_match = re.search(r'(\d+)\s+portfolios', query, re.IGNORECASE)
                    num_portfolios = int(num_portfolios_match.group(1)) if num_portfolios_match else 1000
                    
                    result = PortfolioManagementTools.generate_efficient_frontier(
                        symbols=symbols,
                        num_portfolios=num_portfolios,
                        period=period
                    )
                    
                    # Create visualizations
                    if "success" in result and result["success"]:
                        visualizations = PortfolioManagementTools.visualize_portfolio(result)
                        if "success" in visualizations and visualizations["success"]:
                            result["visualizations"] = visualizations["visualizations"]
                    
                    return result
                else:
                    # Optimize portfolio
                    result = PortfolioManagementTools.optimize_portfolio(
                        symbols=symbols,
                        objective=objective,
                        constraints=constraints if constraints else None,
                        period=period
                    )
                    
                    # Create visualizations
                    if "success" in result and result["success"]:
                        visualizations = PortfolioManagementTools.visualize_portfolio(result)
                        if "success" in visualizations and visualizations["success"]:
                            result["visualizations"] = visualizations["visualizations"]
                    
                    return result
            else:
                # Calculate risk metrics
                result = PortfolioManagementTools.calculate_risk_metrics(
                    symbols=symbols,
                    weights=weights,
                    period=period
                )
                
                # Create visualizations
                if "success" in result and result["success"]:
                    visualizations = PortfolioManagementTools.visualize_portfolio(result)
                    if "success" in visualizations and visualizations["success"]:
                        result["visualizations"] = visualizations["visualizations"]
                
                return result
                
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            return {"error": f"Error analyzing portfolio: {str(e)}"}
    
    @staticmethod
    def backtest_strategy(query: str) -> Dict:
        """
        Backtest a trading strategy based on the query
        
        Args:
            query: The user query string
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Extract symbol from query
            symbols = PortfolioIntegrationTools.extract_portfolio_symbols(query)
            
            if not symbols:
                return {"error": "No stock symbol found in the query"}
            
            # For backtesting, we typically use a single symbol
            symbol = symbols[0]
            
            # Determine strategy type
            if re.search(r'sma\s+crossover|moving\s+average\s+crossover|ma\s+crossover', query, re.IGNORECASE):
                strategy = 'sma_crossover'
            elif re.search(r'\brsi\b', query, re.IGNORECASE):
                strategy = 'rsi'
            elif re.search(r'\bmacd\b', query, re.IGNORECASE):
                strategy = 'macd'
            else:
                strategy = 'sma_crossover'  # Default
            
            # Extract strategy parameters
            parameters = PortfolioIntegrationTools.extract_strategy_parameters(query, strategy)
            
            # Run backtest based on strategy
            if strategy == 'sma_crossover':
                result = BacktestingTools.backtest_sma_crossover(
                    symbol=symbol,
                    short_window=parameters.get('short_window', 50),
                    long_window=parameters.get('long_window', 200),
                    period=parameters.get('period', '5y'),
                    initial_capital=parameters.get('initial_capital', 10000.0)
                )
            elif strategy == 'rsi':
                result = BacktestingTools.backtest_rsi_strategy(
                    symbol=symbol,
                    rsi_period=parameters.get('rsi_period', 14),
                    overbought=parameters.get('overbought', 70),
                    oversold=parameters.get('oversold', 30),
                    period=parameters.get('period', '5y'),
                    initial_capital=parameters.get('initial_capital', 10000.0)
                )
            elif strategy == 'macd':
                result = BacktestingTools.backtest_macd_strategy(
                    symbol=symbol,
                    fast_period=parameters.get('fast_period', 12),
                    slow_period=parameters.get('slow_period', 26),
                    signal_period=parameters.get('signal_period', 9),
                    period=parameters.get('period', '5y'),
                    initial_capital=parameters.get('initial_capital', 10000.0)
                )
            
            # Create visualizations
            if "success" in result and result["success"]:
                visualizations = BacktestVisualizationTools.visualize_backtest_results(result)
                if "success" in visualizations and visualizations["success"]:
                    result["visualizations"] = visualizations["visualizations"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error backtesting strategy: {str(e)}")
            return {"error": f"Error backtesting strategy: {str(e)}"}
    
    @staticmethod
    def run_paper_trading(query: str) -> Dict:
        """
        Run a paper trading simulation based on the query
        
        Args:
            query: The user query string
            
        Returns:
            Dictionary with paper trading results
        """
        try:
            # Extract symbol from query
            symbols = PortfolioIntegrationTools.extract_portfolio_symbols(query)
            
            if not symbols:
                return {"error": "No stock symbol found in the query"}
            
            # For paper trading, we typically use a single symbol
            symbol = symbols[0]
            
            # Determine strategy type
            if re.search(r'sma\s+crossover|moving\s+average\s+crossover|ma\s+crossover', query, re.IGNORECASE):
                strategy = 'sma_crossover'
            elif re.search(r'\brsi\b', query, re.IGNORECASE):
                strategy = 'rsi'
            elif re.search(r'\bmacd\b', query, re.IGNORECASE):
                strategy = 'macd'
            else:
                strategy = 'sma_crossover'  # Default
            
            # Extract strategy parameters
            parameters = PortfolioIntegrationTools.extract_strategy_parameters(query, strategy)
            
            # Extract date range if provided
            start_date_match = re.search(r'from\s+(\d{4}-\d{2}-\d{2})', query, re.IGNORECASE)
            start_date = start_date_match.group(1) if start_date_match else None
            
            end_date_match = re.search(r'to\s+(\d{4}-\d{2}-\d{2})', query, re.IGNORECASE)
            end_date = end_date_match.group(1) if end_date_match else None
            
            # Run paper trading simulation
            result = BacktestVisualizationTools.paper_trading_simulation(
                symbol=symbol,
                strategy=strategy,
                parameters=parameters,
                initial_capital=parameters.get('initial_capital', 10000.0),
                start_date=start_date,
                end_date=end_date
            )
            
            # Create visualizations
            if "success" in result and result["success"]:
                visualizations = BacktestVisualizationTools.visualize_backtest_results(result)
                if "success" in visualizations and visualizations["success"]:
                    result["visualizations"] = visualizations["visualizations"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error running paper trading simulation: {str(e)}")
            return {"error": f"Error running paper trading simulation: {str(e)}"}
