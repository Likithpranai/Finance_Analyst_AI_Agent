"""
Test script for portfolio management and backtesting capabilities of the Finance Analyst AI Agent.

This script demonstrates:
1. Portfolio data retrieval and risk metrics calculation
2. Portfolio optimization with different objectives
3. Efficient frontier generation
4. Backtesting of trading strategies (SMA Crossover, RSI, MACD)
5. Paper trading simulation
6. Visualization of portfolio and backtest results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tools.portfolio_management import PortfolioManagementTools
from tools.backtesting import BacktestingTools
from tools.backtesting_visualization import BacktestVisualizationTools

def test_portfolio_management():
    """Test portfolio management capabilities"""
    print("\n===== Testing Portfolio Management =====")
    
    # Define a portfolio of stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    print(f"Portfolio symbols: {symbols}")
    
    # 1. Get portfolio data
    print("\n1. Getting portfolio data...")
    portfolio_data = PortfolioManagementTools.get_portfolio_data(symbols, period="1y")
    
    if "error" in portfolio_data:
        print(f"Error: {portfolio_data['error']}")
        return
    
    print("Portfolio data retrieved successfully")
    print(f"Data period: {portfolio_data['period']}")
    print(f"Current prices: {portfolio_data['current_prices']}")
    print(f"Default weights: {portfolio_data['weights']}")
    
    # 2. Calculate risk metrics with equal weights
    print("\n2. Calculating risk metrics with equal weights...")
    risk_metrics = PortfolioManagementTools.calculate_risk_metrics(symbols, period="1y")
    
    if "error" in risk_metrics:
        print(f"Error: {risk_metrics['error']}")
    else:
        print("Portfolio risk metrics:")
        print(f"Annual Return: {risk_metrics['portfolio_return']:.2%}")
        print(f"Annual Volatility: {risk_metrics['portfolio_volatility']:.2%}")
        print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print(f"VaR (95%): {risk_metrics['var_95']:.2%}")
        print(f"VaR (99%): {risk_metrics['var_99']:.2%}")
        print(f"Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
        print(f"Portfolio Beta: {risk_metrics['portfolio_beta']:.2f}")
    
    # 3. Optimize portfolio for maximum Sharpe ratio
    print("\n3. Optimizing portfolio for maximum Sharpe ratio...")
    optimized_portfolio = PortfolioManagementTools.optimize_portfolio(symbols, objective="sharpe", period="1y")
    
    if "error" in optimized_portfolio:
        print(f"Error: {optimized_portfolio['error']}")
    else:
        print("Optimized portfolio weights:")
        for symbol, weight in optimized_portfolio["optimized_weights"].items():
            print(f"{symbol}: {weight:.2%}")
        
        print("\nOptimized portfolio metrics:")
        print(f"Annual Return: {optimized_portfolio['portfolio_return']:.2%}")
        print(f"Annual Volatility: {optimized_portfolio['portfolio_volatility']:.2%}")
        print(f"Sharpe Ratio: {optimized_portfolio['sharpe_ratio']:.2f}")
    
    # 4. Generate efficient frontier
    print("\n4. Generating efficient frontier...")
    efficient_frontier = PortfolioManagementTools.generate_efficient_frontier(symbols, num_portfolios=1000, period="1y")
    
    if "error" in efficient_frontier:
        print(f"Error: {efficient_frontier['error']}")
    else:
        print("Efficient frontier generated successfully")
        print("\nMaximum Sharpe Ratio Portfolio:")
        print(f"Return: {efficient_frontier['max_sharpe_portfolio']['return']:.2%}")
        print(f"Volatility: {efficient_frontier['max_sharpe_portfolio']['volatility']:.2%}")
        print(f"Sharpe Ratio: {efficient_frontier['max_sharpe_portfolio']['sharpe_ratio']:.2f}")
        print("Weights:")
        for symbol, weight in efficient_frontier['max_sharpe_portfolio']['weights'].items():
            print(f"{symbol}: {weight:.2%}")
        
        print("\nMinimum Volatility Portfolio:")
        print(f"Return: {efficient_frontier['min_volatility_portfolio']['return']:.2%}")
        print(f"Volatility: {efficient_frontier['min_volatility_portfolio']['volatility']:.2%}")
        print(f"Sharpe Ratio: {efficient_frontier['min_volatility_portfolio']['sharpe_ratio']:.2f}")
        print("Weights:")
        for symbol, weight in efficient_frontier['min_volatility_portfolio']['weights'].items():
            print(f"{symbol}: {weight:.2%}")
    
    # 5. Visualize portfolio
    print("\n5. Creating portfolio visualizations...")
    visualizations = PortfolioManagementTools.visualize_portfolio(efficient_frontier)
    
    if "error" in visualizations:
        print(f"Error: {visualizations['error']}")
    else:
        print("Portfolio visualizations created successfully")
        print("Visualization files:")
        for name, path in visualizations["visualizations"].items():
            print(f"{name}: {path}")

def test_backtesting():
    """Test backtesting capabilities"""
    print("\n===== Testing Backtesting =====")
    
    # Define a stock symbol
    symbol = 'AAPL'
    print(f"Testing backtesting for {symbol}")
    
    # 1. Backtest SMA Crossover strategy
    print("\n1. Backtesting SMA Crossover strategy...")
    sma_backtest = BacktestingTools.backtest_sma_crossover(
        symbol=symbol, 
        short_window=50, 
        long_window=200, 
        period="5y", 
        initial_capital=10000.0
    )
    
    if "error" in sma_backtest:
        print(f"Error: {sma_backtest['error']}")
    else:
        print("SMA Crossover backtest completed successfully")
        print("Performance metrics:")
        print(f"Total Return: {sma_backtest['performance']['total_return']:.2%}")
        print(f"Annual Return: {sma_backtest['performance']['annual_return']:.2%}")
        print(f"Sharpe Ratio: {sma_backtest['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {sma_backtest['performance']['max_drawdown']:.2%}")
        print(f"Number of Trades: {sma_backtest['performance']['num_trades']:.0f}")
        print(f"Win Rate: {sma_backtest['performance'].get('win_rate', 0):.2%}")
        print(f"Buy & Hold Return: {sma_backtest['performance']['buy_hold_return']:.2%}")
    
    # 2. Backtest RSI strategy
    print("\n2. Backtesting RSI strategy...")
    rsi_backtest = BacktestingTools.backtest_rsi_strategy(
        symbol=symbol, 
        rsi_period=14, 
        overbought=70, 
        oversold=30, 
        period="5y", 
        initial_capital=10000.0
    )
    
    if "error" in rsi_backtest:
        print(f"Error: {rsi_backtest['error']}")
    else:
        print("RSI strategy backtest completed successfully")
        print("Performance metrics:")
        print(f"Total Return: {rsi_backtest['performance']['total_return']:.2%}")
        print(f"Annual Return: {rsi_backtest['performance']['annual_return']:.2%}")
        print(f"Sharpe Ratio: {rsi_backtest['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {rsi_backtest['performance']['max_drawdown']:.2%}")
        print(f"Number of Trades: {rsi_backtest['performance']['num_trades']:.0f}")
        print(f"Win Rate: {rsi_backtest['performance'].get('win_rate', 0):.2%}")
        print(f"Buy & Hold Return: {rsi_backtest['performance']['buy_hold_return']:.2%}")
    
    # 3. Backtest MACD strategy
    print("\n3. Backtesting MACD strategy...")
    macd_backtest = BacktestingTools.backtest_macd_strategy(
        symbol=symbol, 
        fast_period=12, 
        slow_period=26, 
        signal_period=9, 
        period="5y", 
        initial_capital=10000.0
    )
    
    if "error" in macd_backtest:
        print(f"Error: {macd_backtest['error']}")
    else:
        print("MACD strategy backtest completed successfully")
        print("Performance metrics:")
        print(f"Total Return: {macd_backtest['performance']['total_return']:.2%}")
        print(f"Annual Return: {macd_backtest['performance']['annual_return']:.2%}")
        print(f"Sharpe Ratio: {macd_backtest['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {macd_backtest['performance']['max_drawdown']:.2%}")
        print(f"Number of Trades: {macd_backtest['performance']['num_trades']:.0f}")
        print(f"Buy & Hold Return: {macd_backtest['performance']['buy_hold_return']:.2%}")
    
    # 4. Visualize backtest results
    print("\n4. Creating backtest visualizations...")
    visualizations = BacktestVisualizationTools.visualize_backtest_results(sma_backtest)
    
    if "error" in visualizations:
        print(f"Error: {visualizations['error']}")
    else:
        print("Backtest visualizations created successfully")
        print("Visualization files:")
        for name, path in visualizations["visualizations"].items():
            print(f"{name}: {path}")
    
    # 5. Run paper trading simulation
    print("\n5. Running paper trading simulation...")
    simulation = BacktestVisualizationTools.paper_trading_simulation(
        symbol=symbol,
        strategy='sma_crossover',
        parameters={'short_window': 50, 'long_window': 200},
        initial_capital=10000.0,
        start_date=(datetime.now().replace(year=datetime.now().year-1)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    if "error" in simulation:
        print(f"Error: {simulation['error']}")
    else:
        print("Paper trading simulation completed successfully")
        print(f"Initial Capital: ${simulation['initial_capital']:.2f}")
        print(f"Final Portfolio Value: ${simulation['final_portfolio_value']:.2f}")
        print(f"Total Return: {simulation['performance']['total_return']:.2%}")
        print(f"Number of Trades: {simulation['performance']['num_trades']:.0f}")
        print(f"Buy & Hold Return: {simulation['performance']['buy_hold_return']:.2%}")
        
        print("\nCurrent Position:")
        print(f"Position: {'Long' if simulation['current_position']['position'] == 1 else 'None'}")
        print(f"Shares: {simulation['current_position']['shares']:.2f}")
        print(f"Holdings Value: ${simulation['current_position']['holdings_value']:.2f}")
        print(f"Cash: ${simulation['current_position']['cash']:.2f}")

if __name__ == "__main__":
    # Create output directories
    os.makedirs("outputs/portfolio", exist_ok=True)
    os.makedirs("outputs/backtests", exist_ok=True)
    
    # Run tests
    test_portfolio_management()
    test_backtesting()
    
    print("\nAll tests completed. Check the 'outputs' directory for visualization files.")
