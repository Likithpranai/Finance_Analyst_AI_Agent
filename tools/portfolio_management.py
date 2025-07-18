"""
Portfolio Management and Risk Analysis Tools for Finance Analyst AI Agent

This module provides portfolio management capabilities including:
- Portfolio optimization
- Risk metrics calculation (VaR, Sharpe ratio, beta)
- Asset allocation optimization
- Backtesting of trading strategies
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import yfinance as yf
import scipy.optimize as sco
from scipy import stats
import warnings
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioManagementTools:
    """Tools for portfolio management and risk analysis"""
    
    @staticmethod
    def get_portfolio_data(symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict:
        """
        Get historical price data for a portfolio of stocks
        
        Args:
            symbols: List of stock symbols
            period: Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary with portfolio data
        """
        try:
            # Validate inputs
            if not symbols:
                return {"error": "No symbols provided"}
            
            # Download data for all symbols
            data = yf.download(symbols, period=period, interval=interval, group_by='ticker')
            
            if len(symbols) == 1:
                # Handle single symbol case
                symbol = symbols[0]
                prices_df = pd.DataFrame(data['Close'])
                prices_df.columns = [symbol]
            else:
                # Extract closing prices
                prices_df = pd.DataFrame()
                for symbol in symbols:
                    if (symbol, 'Close') in data.columns:
                        prices_df[symbol] = data[(symbol, 'Close')]
            
            # Calculate daily returns
            returns_df = prices_df.pct_change().dropna()
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns_df).cumprod()
            
            # Calculate portfolio statistics
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            
            # Get current prices and market caps
            current_prices = {}
            market_caps = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                current_prices[symbol] = ticker.history(period="1d")['Close'].iloc[-1]
                
                # Get market cap if available
                try:
                    market_caps[symbol] = ticker.info.get('marketCap', None)
                except:
                    market_caps[symbol] = None
            
            # Create equal weight portfolio by default
            weights = np.array([1/len(symbols)] * len(symbols))
            
            return {
                "success": True,
                "symbols": symbols,
                "period": period,
                "interval": interval,
                "prices": prices_df.to_dict(),
                "returns": returns_df.to_dict(),
                "cumulative_returns": cumulative_returns.to_dict(),
                "mean_returns": mean_returns.to_dict(),
                "cov_matrix": cov_matrix.to_dict(),
                "current_prices": current_prices,
                "market_caps": market_caps,
                "weights": dict(zip(symbols, weights.tolist()))
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio data: {str(e)}")
            return {"error": f"Error getting portfolio data: {str(e)}"}
    
    @staticmethod
    def calculate_risk_metrics(symbols: List[str], weights: Optional[List[float]] = None, 
                              period: str = "1y", risk_free_rate: float = 0.03) -> Dict:
        """
        Calculate risk metrics for a portfolio
        
        Args:
            symbols: List of stock symbols
            weights: List of portfolio weights (default: equal weight)
            period: Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            risk_free_rate: Annual risk-free rate (default: 3%)
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            # Validate inputs
            if not symbols:
                return {"error": "No symbols provided"}
            
            # Set default weights if not provided
            if weights is None:
                weights = [1/len(symbols)] * len(symbols)
            
            # Validate weights
            if len(weights) != len(symbols):
                return {"error": "Number of weights must match number of symbols"}
            
            if abs(sum(weights) - 1.0) > 1e-6:
                return {"error": "Weights must sum to 1.0"}
            
            # Get portfolio data
            portfolio_data = PortfolioManagementTools.get_portfolio_data(symbols, period)
            
            if "error" in portfolio_data:
                return portfolio_data
            
            # Convert returns to DataFrame
            returns_df = pd.DataFrame(portfolio_data["returns"])
            
            # Convert weights to numpy array
            weights = np.array(weights)
            
            # Calculate portfolio return and volatility
            portfolio_return = np.sum(returns_df.mean() * weights) * 252  # Annualized
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
            
            # Calculate Sharpe Ratio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            # Calculate Value at Risk (VaR)
            # Using parametric method with 95% confidence
            var_95 = -1.645 * portfolio_volatility / np.sqrt(252)
            # Using parametric method with 99% confidence
            var_99 = -2.326 * portfolio_volatility / np.sqrt(252)
            
            # Calculate portfolio beta relative to market (using SPY as proxy)
            try:
                market_data = yf.download("SPY", period=period)
                market_returns = market_data['Close'].pct_change().dropna()
                
                # Calculate beta for each stock
                betas = {}
                for i, symbol in enumerate(symbols):
                    if symbol in returns_df.columns:
                        stock_returns = returns_df[symbol]
                        covariance = stock_returns.cov(market_returns)
                        market_variance = market_returns.var()
                        beta = covariance / market_variance
                        betas[symbol] = beta
                
                # Calculate portfolio beta
                portfolio_beta = sum(betas.get(symbol, 0) * weight for symbol, weight in zip(symbols, weights))
            except:
                betas = {symbol: None for symbol in symbols}
                portfolio_beta = None
            
            # Calculate Maximum Drawdown
            cumulative_returns = (1 + returns_df).cumprod()
            portfolio_cumulative_returns = np.sum(cumulative_returns * weights, axis=1)
            rolling_max = np.maximum.accumulate(portfolio_cumulative_returns)
            drawdowns = (portfolio_cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Calculate Sortino Ratio (downside risk only)
            negative_returns = returns_df.copy()
            for col in negative_returns.columns:
                negative_returns[col] = negative_returns[col].apply(lambda x: min(x, 0))
            
            downside_deviation = np.sqrt(np.dot(weights.T, np.dot(negative_returns.cov() * 252, weights)))
            sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
            
            # Calculate Treynor Ratio
            treynor_ratio = (portfolio_return - risk_free_rate) / portfolio_beta if portfolio_beta else None
            
            return {
                "success": True,
                "symbols": symbols,
                "weights": dict(zip(symbols, weights.tolist())),
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "var_95": var_95,
                "var_99": var_99,
                "betas": betas,
                "portfolio_beta": portfolio_beta,
                "treynor_ratio": treynor_ratio,
                "max_drawdown": max_drawdown,
                "risk_free_rate": risk_free_rate
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {"error": f"Error calculating risk metrics: {str(e)}"}
    
    @staticmethod
    def optimize_portfolio(symbols: List[str], objective: str = "sharpe", 
                          constraints: Dict = None, period: str = "1y",
                          risk_free_rate: float = 0.03) -> Dict:
        """
        Optimize portfolio weights based on different objectives
        
        Args:
            symbols: List of stock symbols
            objective: Optimization objective ('sharpe', 'min_volatility', 'max_return', 'target_volatility', 'target_return')
            constraints: Dictionary with constraints (e.g., {'target_return': 0.2, 'target_volatility': 0.15})
            period: Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            risk_free_rate: Annual risk-free rate (default: 3%)
            
        Returns:
            Dictionary with optimized portfolio
        """
        try:
            # Validate inputs
            if not symbols:
                return {"error": "No symbols provided"}
            
            if objective not in ['sharpe', 'min_volatility', 'max_return', 'target_volatility', 'target_return']:
                return {"error": "Invalid objective. Choose from 'sharpe', 'min_volatility', 'max_return', 'target_volatility', 'target_return'"}
            
            # Set default constraints if not provided
            if constraints is None:
                constraints = {}
            
            # Get portfolio data
            portfolio_data = PortfolioManagementTools.get_portfolio_data(symbols, period)
            
            if "error" in portfolio_data:
                return portfolio_data
            
            # Convert returns to DataFrame
            returns_df = pd.DataFrame(portfolio_data["returns"])
            
            # Calculate mean returns and covariance matrix
            mean_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            num_assets = len(symbols)
            
            # Define optimization functions
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def portfolio_return(weights):
                return np.sum(mean_returns * weights)
            
            def negative_sharpe_ratio(weights):
                p_return = portfolio_return(weights)
                p_volatility = portfolio_volatility(weights)
                return -(p_return - risk_free_rate) / p_volatility
            
            # Define constraints
            constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
            
            # Add target return constraint if specified
            if 'target_return' in constraints:
                target_return = constraints['target_return']
                constraints_list.append({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return})
            
            # Add target volatility constraint if specified
            if 'target_volatility' in constraints:
                target_volatility = constraints['target_volatility']
                constraints_list.append({'type': 'eq', 'fun': lambda x: portfolio_volatility(x) - target_volatility})
            
            # Bounds for weights (0 to 1)
            bounds = tuple((0, 1) for asset in range(num_assets))
            
            # Initial guess (equal weights)
            initial_weights = np.array([1/num_assets] * num_assets)
            
            # Perform optimization based on objective
            if objective == 'sharpe':
                # Maximize Sharpe ratio
                result = sco.minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', 
                                     bounds=bounds, constraints=constraints_list)
            elif objective == 'min_volatility':
                # Minimize volatility
                result = sco.minimize(portfolio_volatility, initial_weights, method='SLSQP', 
                                     bounds=bounds, constraints=constraints_list)
            elif objective == 'max_return':
                # Maximize return
                result = sco.minimize(lambda x: -portfolio_return(x), initial_weights, method='SLSQP', 
                                     bounds=bounds, constraints=constraints_list)
            elif objective == 'target_volatility' or objective == 'target_return':
                # Maximize Sharpe with target constraints
                result = sco.minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', 
                                     bounds=bounds, constraints=constraints_list)
            
            # Check if optimization was successful
            if not result['success']:
                return {"error": f"Optimization failed: {result['message']}"}
            
            # Get optimized weights
            optimized_weights = result['x']
            
            # Calculate portfolio metrics with optimized weights
            opt_portfolio_return = portfolio_return(optimized_weights)
            opt_portfolio_volatility = portfolio_volatility(optimized_weights)
            opt_sharpe_ratio = (opt_portfolio_return - risk_free_rate) / opt_portfolio_volatility
            
            # Calculate VaR with optimized weights
            var_95 = -1.645 * opt_portfolio_volatility / np.sqrt(252)
            var_99 = -2.326 * opt_portfolio_volatility / np.sqrt(252)
            
            return {
                "success": True,
                "symbols": symbols,
                "objective": objective,
                "constraints": constraints,
                "optimized_weights": dict(zip(symbols, optimized_weights.tolist())),
                "portfolio_return": opt_portfolio_return,
                "portfolio_volatility": opt_portfolio_volatility,
                "sharpe_ratio": opt_sharpe_ratio,
                "var_95": var_95,
                "var_99": var_99,
                "risk_free_rate": risk_free_rate
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return {"error": f"Error optimizing portfolio: {str(e)}"}
    
    @staticmethod
    def generate_efficient_frontier(symbols: List[str], num_portfolios: int = 10000,
                                  period: str = "1y", risk_free_rate: float = 0.03) -> Dict:
        """
        Generate the efficient frontier for a set of assets
        
        Args:
            symbols: List of stock symbols
            num_portfolios: Number of random portfolios to generate
            period: Time period for data
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with efficient frontier data
        """
        try:
            # Validate inputs
            if not symbols:
                return {"error": "No symbols provided"}
            
            # Get portfolio data
            portfolio_data = PortfolioManagementTools.get_portfolio_data(symbols, period)
            
            if "error" in portfolio_data:
                return portfolio_data
            
            # Convert returns to DataFrame
            returns_df = pd.DataFrame(portfolio_data["returns"])
            
            # Calculate mean returns and covariance matrix
            mean_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            num_assets = len(symbols)
            
            # Generate random portfolios
            results = np.zeros((3, num_portfolios))
            weights_record = []
            
            for i in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                weights_record.append(weights)
                
                # Portfolio return
                portfolio_return = np.sum(mean_returns * weights)
                # Portfolio volatility
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                # Sharpe ratio
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                results[0, i] = portfolio_return
                results[1, i] = portfolio_volatility
                results[2, i] = sharpe_ratio
            
            # Find portfolio with maximum Sharpe ratio
            max_sharpe_idx = np.argmax(results[2])
            max_sharpe_return = results[0, max_sharpe_idx]
            max_sharpe_volatility = results[1, max_sharpe_idx]
            max_sharpe_ratio = results[2, max_sharpe_idx]
            max_sharpe_weights = weights_record[max_sharpe_idx]
            
            # Find portfolio with minimum volatility
            min_vol_idx = np.argmin(results[1])
            min_vol_return = results[0, min_vol_idx]
            min_vol_volatility = results[1, min_vol_idx]
            min_vol_sharpe = results[2, min_vol_idx]
            min_vol_weights = weights_record[min_vol_idx]
            
            # Convert results to lists for JSON serialization
            returns_list = results[0].tolist()
            volatility_list = results[1].tolist()
            sharpe_list = results[2].tolist()
            
            # Create efficient frontier portfolios
            efficient_portfolios = []
            for i in range(num_portfolios):
                efficient_portfolios.append({
                    "return": returns_list[i],
                    "volatility": volatility_list[i],
                    "sharpe_ratio": sharpe_list[i]
                })
            
            return {
                "success": True,
                "symbols": symbols,
                "num_portfolios": num_portfolios,
                "efficient_frontier": efficient_portfolios,
                "max_sharpe_portfolio": {
                    "return": max_sharpe_return,
                    "volatility": max_sharpe_volatility,
                    "sharpe_ratio": max_sharpe_ratio,
                    "weights": dict(zip(symbols, max_sharpe_weights.tolist()))
                },
                "min_volatility_portfolio": {
                    "return": min_vol_return,
                    "volatility": min_vol_volatility,
                    "sharpe_ratio": min_vol_sharpe,
                    "weights": dict(zip(symbols, min_vol_weights.tolist()))
                },
                "risk_free_rate": risk_free_rate
            }
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
            return {"error": f"Error generating efficient frontier: {str(e)}"}
    
    @staticmethod
    def visualize_portfolio(portfolio_data: Dict, output_dir: str = None) -> Dict:
        """
        Create visualizations for portfolio analysis
        
        Args:
            portfolio_data: Portfolio data dictionary
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with visualization file paths
        """
        try:
            # Validate inputs
            if "success" not in portfolio_data or not portfolio_data["success"]:
                return {"error": "Invalid portfolio data"}
            
            # Create output directory if not provided
            if output_dir is None:
                output_dir = os.path.join(os.getcwd(), "outputs", "portfolio")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract data
            symbols = portfolio_data.get("symbols", [])
            weights = portfolio_data.get("weights", {})
            
            # Create visualizations
            visualizations = {}
            
            # 1. Portfolio Allocation Pie Chart
            if weights:
                plt.figure(figsize=(10, 6))
                plt.pie(list(weights.values()), labels=list(weights.keys()), autopct='%1.1f%%')
                plt.title('Portfolio Allocation')
                
                # Save figure
                allocation_file = os.path.join(output_dir, f"portfolio_allocation_{timestamp}.png")
                plt.savefig(allocation_file)
                plt.close()
                
                visualizations["allocation_chart"] = allocation_file
            
            # 2. Risk-Return Scatter Plot (if efficient frontier data is available)
            if "efficient_frontier" in portfolio_data:
                efficient_frontier = portfolio_data["efficient_frontier"]
                max_sharpe = portfolio_data["max_sharpe_portfolio"]
                min_vol = portfolio_data["min_volatility_portfolio"]
                
                plt.figure(figsize=(12, 8))
                
                # Plot random portfolios
                returns = [p["return"] for p in efficient_frontier]
                volatilities = [p["volatility"] for p in efficient_frontier]
                sharpe_ratios = [p["sharpe_ratio"] for p in efficient_frontier]
                
                plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.5)
                plt.colorbar(label='Sharpe Ratio')
                
                # Plot max Sharpe and min volatility portfolios
                plt.scatter(max_sharpe["volatility"], max_sharpe["return"], marker='*', color='r', s=300, label='Maximum Sharpe')
                plt.scatter(min_vol["volatility"], min_vol["return"], marker='*', color='g', s=300, label='Minimum Volatility')
                
                # Plot individual assets
                if "mean_returns" in portfolio_data and "cov_matrix" in portfolio_data:
                    mean_returns = portfolio_data["mean_returns"]
                    cov_matrix = portfolio_data["cov_matrix"]
                    
                    for i, symbol in enumerate(symbols):
                        asset_return = mean_returns[symbol]
                        asset_volatility = np.sqrt(cov_matrix[symbol][symbol])
                        plt.scatter(asset_volatility, asset_return, marker='o', s=100, label=symbol)
                
                plt.xlabel('Volatility (Standard Deviation)')
                plt.ylabel('Expected Return')
                plt.title('Efficient Frontier')
                plt.legend()
                
                # Save figure
                frontier_file = os.path.join(output_dir, f"efficient_frontier_{timestamp}.png")
                plt.savefig(frontier_file)
                plt.close()
                
                visualizations["efficient_frontier"] = frontier_file
            
            # 3. Correlation Heatmap
            if "cov_matrix" in portfolio_data:
                cov_matrix = pd.DataFrame(portfolio_data["cov_matrix"])
                correlation_matrix = cov_matrix.corr()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Asset Correlation Matrix')
                
                # Save figure
                correlation_file = os.path.join(output_dir, f"correlation_matrix_{timestamp}.png")
                plt.savefig(correlation_file)
                plt.close()
                
                visualizations["correlation_matrix"] = correlation_file
            
            # 4. Performance Comparison
            if "cumulative_returns" in portfolio_data:
                cumulative_returns = pd.DataFrame(portfolio_data["cumulative_returns"])
                
                plt.figure(figsize=(12, 6))
                cumulative_returns.plot()
                plt.title('Cumulative Returns')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Return')
                plt.legend(loc='best')
                plt.grid(True)
                
                # Save figure
                performance_file = os.path.join(output_dir, f"performance_comparison_{timestamp}.png")
                plt.savefig(performance_file)
                plt.close()
                
                visualizations["performance_comparison"] = performance_file
            
            return {
                "success": True,
                "visualizations": visualizations,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Error visualizing portfolio: {str(e)}")
            return {"error": f"Error visualizing portfolio: {str(e)}"}
