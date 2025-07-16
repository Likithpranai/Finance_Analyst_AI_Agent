"""
Portfolio Analysis Tools for the Finance Analyst AI Agent.
Includes risk management, portfolio optimization, and performance analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple, Union
from langchain.tools import BaseTool
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

# Try importing optional dependencies
try:
    import scipy.optimize as sco
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pypfopt
    from pypfopt import expected_returns, risk_models, objective_functions
    from pypfopt.efficient_frontier import EfficientFrontier
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False


class PortfolioRiskTool(BaseTool):
    name = "portfolio_risk_analyzer"
    description = """
    Analyzes risk metrics for a portfolio of stocks including volatility, beta, Value at Risk (VaR), 
    Sharpe ratio, and correlation matrix.
    
    Args:
        tickers: Comma-separated list of ticker symbols (e.g., 'AAPL,MSFT,GOOGL')
        weights: Optional comma-separated list of portfolio weights (e.g., '0.4,0.3,0.3')
                If not provided, equal weights will be assumed.
        period: Optional time period for historical data (e.g., '1y', '3y', '5y', defaults to '1y')
        risk_free_rate: Optional annual risk-free rate for Sharpe ratio calculation (defaults to 0.03 or 3%)
        
    Returns:
        A dictionary with portfolio risk metrics, analysis, and visualizations.
    """
    
    def _run(self, tickers: str, weights: str = None, period: str = "1y", risk_free_rate: float = 0.03) -> Dict[str, Any]:
        try:
            # Parse tickers and weights
            ticker_list = [ticker.strip() for ticker in tickers.split(',')]
            
            if weights:
                weight_list = [float(w.strip()) for w in weights.split(',')]
                # Normalize weights to sum to 1
                weight_list = [w / sum(weight_list) for w in weight_list]
            else:
                # Equal weights if not provided
                weight_list = [1.0 / len(ticker_list) for _ in ticker_list]
                
            if len(ticker_list) != len(weight_list):
                return {"error": "The number of tickers must match the number of weights"}
                
            # Fetch historical data
            data = yf.download(ticker_list, period=period, progress=False)['Adj Close']
            
            # If only one ticker is provided, convert Series to DataFrame
            if len(ticker_list) == 1:
                data = pd.DataFrame(data, columns=ticker_list)
                
            if data.empty:
                return {"error": f"Could not fetch price data for the provided tickers: {tickers}"}
                
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Calculate risk metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weight_list, risk_free_rate)
            
            # Calculate stock-specific metrics
            stock_metrics = self._calculate_stock_metrics(returns, data)
            
            # Generate correlation matrix
            correlation_matrix = returns.corr().round(3).to_dict()
            
            # Create result
            result = {
                "portfolio": {
                    "tickers": ticker_list,
                    "weights": [round(w, 4) for w in weight_list],
                    "metrics": portfolio_metrics
                },
                "individual_stocks": stock_metrics,
                "correlation_matrix": correlation_matrix,
                "analysis_period": period,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add risk interpretation
            result["risk_interpretation"] = self._interpret_risk_metrics(portfolio_metrics, stock_metrics)
            
            return result
            
        except Exception as e:
            return {"error": f"Error analyzing portfolio risk: {str(e)}"}
    
    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: List[float], risk_free_rate: float) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics."""
        # Convert annual risk-free rate to daily
        daily_rf = ((1 + risk_free_rate) ** (1/252)) - 1
        
        # Calculate weighted returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Daily VaR in percentage
        daily_var_95 = -var_95 * 100
        
        # Annualized VaR (approximate)
        annual_var_95 = daily_var_95 * np.sqrt(252)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Beta to market (using S&P 500 as proxy)
        try:
            spy = yf.download("^GSPC", start=returns.index[0], end=returns.index[-1], progress=False)['Adj Close']
            market_returns = spy.pct_change().dropna()
            
            # Align dates
            aligned_returns = pd.DataFrame({'portfolio': portfolio_returns, 'market': market_returns})
            aligned_returns = aligned_returns.dropna()
            
            if not aligned_returns.empty:
                # Calculate beta using covariance
                covariance = aligned_returns['portfolio'].cov(aligned_returns['market'])
                market_variance = aligned_returns['market'].var()
                beta = covariance / market_variance if market_variance > 0 else 1
            else:
                beta = 1  # Default if data alignment fails
        except:
            beta = 1  # Default if market data fetch fails
        
        return {
            "annualized_return": round(annual_return * 100, 2),  # in percentage
            "annualized_volatility": round(annual_volatility * 100, 2),  # in percentage
            "sharpe_ratio": round(sharpe_ratio, 2),
            "daily_var_95": round(daily_var_95, 2),  # in percentage
            "annual_var_95": round(annual_var_95, 2),  # in percentage
            "conditional_var_95": round(-cvar_95 * 100, 2),  # in percentage
            "max_drawdown": round(max_drawdown * 100, 2),  # in percentage
            "beta": round(beta, 2)
        }
    
    def _calculate_stock_metrics(self, returns: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate risk metrics for individual stocks."""
        result = {}
        
        # Calculate metrics for each stock
        for ticker in returns.columns:
            stock_returns = returns[ticker]
            
            # Calculate metrics
            annual_return = stock_returns.mean() * 252
            annual_volatility = stock_returns.std() * np.sqrt(252)
            
            # Value at Risk (VaR)
            var_95 = np.percentile(stock_returns, 5)
            
            # Daily VaR in percentage
            daily_var_95 = -var_95 * 100
            
            # Maximum Drawdown
            stock_prices = prices[ticker]
            max_drawdown = ((stock_prices / stock_prices.cummax()) - 1).min()
            
            # Beta to market (using S&P 500 as proxy)
            try:
                spy = yf.download("^GSPC", start=returns.index[0], end=returns.index[-1], progress=False)['Adj Close']
                market_returns = spy.pct_change().dropna()
                
                # Align dates
                aligned_returns = pd.DataFrame({ticker: stock_returns, 'market': market_returns})
                aligned_returns = aligned_returns.dropna()
                
                if not aligned_returns.empty:
                    # Calculate beta using covariance
                    covariance = aligned_returns[ticker].cov(aligned_returns['market'])
                    market_variance = aligned_returns['market'].var()
                    beta = covariance / market_variance if market_variance > 0 else 1
                else:
                    beta = 1  # Default if data alignment fails
            except:
                beta = 1  # Default if market data fetch fails
                
            result[ticker] = {
                "annualized_return": round(annual_return * 100, 2),  # in percentage
                "annualized_volatility": round(annual_volatility * 100, 2),  # in percentage
                "daily_var_95": round(daily_var_95, 2),  # in percentage
                "max_drawdown": round(max_drawdown * 100, 2),  # in percentage
                "beta": round(beta, 2)
            }
            
        return result
    
    def _interpret_risk_metrics(self, portfolio_metrics: Dict[str, float], stock_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Provide interpretations of the risk metrics."""
        insights = []
        
        # Portfolio-level insights
        port_return = portfolio_metrics["annualized_return"]
        port_vol = portfolio_metrics["annualized_volatility"]
        port_sharpe = portfolio_metrics["sharpe_ratio"]
        port_var = portfolio_metrics["annual_var_95"]
        port_beta = portfolio_metrics["beta"]
        port_drawdown = portfolio_metrics["max_drawdown"]
        
        # Return interpretation
        if port_return > 20:
            insights.append(f"The portfolio has an excellent annualized return of {port_return}%, which is significantly above average market returns.")
        elif port_return > 10:
            insights.append(f"The portfolio has a good annualized return of {port_return}%, above the long-term average market return.")
        elif port_return > 0:
            insights.append(f"The portfolio has a positive but modest annualized return of {port_return}%.")
        else:
            insights.append(f"The portfolio has a negative annualized return of {port_return}%, indicating poor performance.")
            
        # Volatility interpretation
        if port_vol > 25:
            insights.append(f"The portfolio exhibits high volatility ({port_vol}%), suggesting higher risk.")
        elif port_vol > 15:
            insights.append(f"The portfolio has moderate volatility ({port_vol}%), typical of a balanced portfolio.")
        else:
            insights.append(f"The portfolio has low volatility ({port_vol}%), suggesting conservative positioning.")
            
        # Sharpe ratio interpretation
        if port_sharpe > 1:
            insights.append(f"The Sharpe ratio of {port_sharpe} indicates good risk-adjusted returns.")
        elif port_sharpe > 0.5:
            insights.append(f"The Sharpe ratio of {port_sharpe} suggests acceptable risk-adjusted performance.")
        elif port_sharpe > 0:
            insights.append(f"The Sharpe ratio of {port_sharpe} indicates poor risk-adjusted performance.")
        else:
            insights.append(f"The negative Sharpe ratio of {port_sharpe} suggests returns don't compensate for the risk taken.")
            
        # VaR interpretation
        insights.append(f"The 95% Value at Risk (VaR) indicates a potential {port_var}% loss or worse in a year, with 5% probability.")
            
        # Beta interpretation
        if port_beta > 1.2:
            insights.append(f"The portfolio beta of {port_beta} is high, indicating greater sensitivity to market movements.")
        elif port_beta < 0.8:
            insights.append(f"The portfolio beta of {port_beta} is low, suggesting less sensitivity to market fluctuations.")
        else:
            insights.append(f"The portfolio beta of {port_beta} indicates market-like sensitivity.")
            
        # Drawdown interpretation
        if port_drawdown > 30:
            insights.append(f"The maximum drawdown of {port_drawdown}% reveals significant historical losses.")
        elif port_drawdown > 15:
            insights.append(f"The maximum drawdown of {port_drawdown}% indicates moderate historical losses.")
        else:
            insights.append(f"The maximum drawdown of {port_drawdown}% suggests relatively contained historical losses.")
            
        # Add diversification insight based on correlation
        if len(stock_metrics) > 1:
            insights.append("Review the correlation matrix to assess diversification benefits. Lower correlation between assets typically improves portfolio efficiency.")
            
        return insights
    
    async def _arun(self, tickers: str, weights: str = None, period: str = "1y", risk_free_rate: float = 0.03) -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(tickers, weights, period, risk_free_rate)


class PortfolioOptimizationTool(BaseTool):
    name = "portfolio_optimizer"
    description = """
    Optimizes portfolio asset allocation for maximum return-to-risk ratio or other objectives.
    Implements Modern Portfolio Theory to find the efficient frontier.
    
    Args:
        tickers: Comma-separated list of ticker symbols (e.g., 'AAPL,MSFT,GOOGL')
        objective: Optimization objective ('sharpe', 'min_volatility', 'max_return', defaults to 'sharpe')
        period: Optional time period for historical data (e.g., '1y', '3y', '5y', defaults to '1y')
        risk_free_rate: Optional annual risk-free rate for Sharpe ratio calculation (defaults to 0.03 or 3%)
        
    Returns:
        A dictionary with optimal portfolio weights, expected returns, risk metrics, and visualizations.
    """
    
    def _run(self, tickers: str, objective: str = "sharpe", period: str = "1y", risk_free_rate: float = 0.03) -> Dict[str, Any]:
        if not PYPFOPT_AVAILABLE:
            return {"error": "PyPortfolioOpt is not installed. Please install with 'pip install PyPortfolioOpt'."}
            
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy is not installed. Please install with 'pip install scipy'."}
            
        try:
            # Parse tickers
            ticker_list = [ticker.strip() for ticker in tickers.split(',')]
            
            # Fetch historical data
            data = yf.download(ticker_list, period=period, progress=False)['Adj Close']
            
            # If only one ticker is provided, convert Series to DataFrame
            if len(ticker_list) == 1:
                data = pd.DataFrame(data, columns=ticker_list)
                
            if data.empty:
                return {"error": f"Could not fetch price data for the provided tickers: {tickers}"}
                
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            
            # Initialize Efficient Frontier
            ef = EfficientFrontier(mu, S)
            
            # Optimize based on objective
            if objective == "min_volatility":
                weights = ef.min_volatility()
                optimization_type = "Minimum Volatility"
            elif objective == "max_return":
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                optimization_type = "Maximum Return"
            else:  # default to sharpe
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                optimization_type = "Maximum Sharpe Ratio"
                
            # Get cleaned weights
            cleaned_weights = ef.clean_weights()
            
            # Performance metrics
            performance = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            expected_annual_return, annual_volatility, sharpe_ratio = performance
            
            # Format output
            result = {
                "portfolio": {
                    "optimization_type": optimization_type,
                    "tickers": ticker_list,
                    "weights": {ticker: round(cleaned_weights[ticker], 4) for ticker in cleaned_weights if cleaned_weights[ticker] > 0.001},  # Filter out tiny allocations
                },
                "performance": {
                    "expected_annual_return": round(expected_annual_return * 100, 2),  # in percentage
                    "annual_volatility": round(annual_volatility * 100, 2),  # in percentage
                    "sharpe_ratio": round(sharpe_ratio, 2)
                },
                "analysis_period": period,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add optimization insights
            result["optimization_insights"] = self._get_optimization_insights(result, ticker_list)
            
            return result
            
        except Exception as e:
            return {"error": f"Error optimizing portfolio: {str(e)}"}
    
    def _get_optimization_insights(self, result: Dict[str, Any], original_tickers: List[str]) -> List[str]:
        """Generate insights based on optimization results."""
        insights = []
        
        # Check portfolio concentration
        weights = result["portfolio"]["weights"]
        
        # Count tickers with significant allocation
        significant_allocation_count = sum(1 for w in weights.values() if w >= 0.05)  # 5% or more
        original_count = len(original_tickers)
        
        # Check if any tickers were removed in optimization
        removed_tickers = [ticker for ticker in original_tickers if ticker not in weights or weights[ticker] < 0.001]
        
        if removed_tickers:
            insights.append(f"The optimizer excluded {len(removed_tickers)} assets from the portfolio: {', '.join(removed_tickers)}. These may have unfavorable risk-return profiles or high correlation with other holdings.")
            
        if significant_allocation_count <= len(weights) / 3:
            insights.append(f"The optimal portfolio is concentrated in {significant_allocation_count} out of {len(weights)} assets with significant allocations (≥5%), suggesting these assets provide the most efficient risk-return tradeoff.")
        else:
            insights.append(f"The optimal portfolio is well-diversified across {significant_allocation_count} assets with significant allocations (≥5%).")
            
        # Performance insights
        expected_return = result["performance"]["expected_annual_return"]
        volatility = result["performance"]["annual_volatility"]
        sharpe = result["performance"]["sharpe_ratio"]
        
        insights.append(f"The optimized portfolio has an expected annual return of {expected_return}% with {volatility}% volatility.")
        
        if sharpe > 1:
            insights.append(f"Sharpe ratio of {sharpe} indicates strong risk-adjusted returns, well above the risk-free rate.")
        elif sharpe > 0.5:
            insights.append(f"Sharpe ratio of {sharpe} indicates acceptable risk-adjusted returns.")
        else:
            insights.append(f"Sharpe ratio of {sharpe} is relatively low, suggesting limited excess return over the risk-free rate.")
            
        # Add implementation considerations
        insights.append("Consider transaction costs and tax implications when rebalancing your portfolio to match these optimal weights.")
        insights.append("Periodic reoptimization is recommended as market conditions and asset correlations change over time.")
        
        return insights
    
    async def _arun(self, tickers: str, objective: str = "sharpe", period: str = "1y", risk_free_rate: float = 0.03) -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(tickers, objective, period, risk_free_rate)
