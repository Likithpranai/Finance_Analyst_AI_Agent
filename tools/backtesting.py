"""
Backtesting and Strategy Simulation Tools for Finance Analyst AI Agent

This module provides backtesting capabilities including:
- Historical strategy testing
- Performance metrics calculation
- Paper trading simulation
- Strategy visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import yfinance as yf
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestingTools:
    """Tools for backtesting trading strategies"""
    
    @staticmethod
    def backtest_sma_crossover(symbol: str, short_window: int = 50, long_window: int = 200, 
                              period: str = "5y", initial_capital: float = 10000.0) -> Dict:
        """
        Backtest a simple moving average crossover strategy
        
        Args:
            symbol: Stock symbol
            short_window: Short moving average window
            long_window: Long moving average window
            period: Time period for data
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Download historical data
            data = yf.download(symbol, period=period)
            
            if data.empty:
                return {"success": False, "error": f"No data found for {symbol}"}
            
            # Create signals dataframe
            signals = pd.DataFrame(index=data.index)
            # Use 'Adj Close' if available, otherwise use 'Close'
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            signals['price'] = data[price_col]
            signals['short_mavg'] = signals['price'].rolling(window=short_window, min_periods=1).mean()
            signals['long_mavg'] = signals['price'].rolling(window=long_window, min_periods=1).mean()
            signals['signal'] = 0.0
            
            # Get the index values starting from short_window position
            idx = signals.index[short_window:]
            signals.loc[idx, 'signal'] = np.where(
                signals.loc[idx, 'short_mavg'] > signals.loc[idx, 'long_mavg'], 1.0, 0.0)
            
            # Generate trading orders
            signals['position'] = signals['signal'].diff()
            
            # Create portfolio dataframe
            portfolio = pd.DataFrame(index=signals.index)
            portfolio['holdings'] = 0.0
            portfolio['cash'] = initial_capital
            
            # Initialize first row
            portfolio.loc[portfolio.index[0], 'holdings'] = 0.0
            portfolio.loc[portfolio.index[0], 'cash'] = initial_capital
            
            # Simulate trading
            for i in range(1, len(portfolio)):
                # If we have a buy signal
                if signals['position'].iloc[i] == 1.0:
                    # Buy as many shares as possible with current cash
                    shares = portfolio['cash'].iloc[i-1] / signals['price'].iloc[i]
                    portfolio.loc[portfolio.index[i], 'holdings'] = shares * signals['price'].iloc[i]
                    portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - portfolio['holdings'].iloc[i]
                
                # If we have a sell signal
                elif signals['position'].iloc[i] == -1.0:
                    # Sell all shares
                    portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + portfolio['holdings'].iloc[i-1]
                    portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                
                # No signal, maintain position but update holdings value
                else:
                    if signals['signal'].iloc[i] == 1.0:  # We're holding shares
                        # Calculate number of shares based on previous holdings value and price
                        if signals['price'].iloc[i-1] != 0:
                            shares = portfolio['holdings'].iloc[i-1] / signals['price'].iloc[i-1]
                            portfolio.loc[portfolio.index[i], 'holdings'] = shares * signals['price'].iloc[i]
                            portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                        else:
                            portfolio.loc[portfolio.index[i], 'holdings'] = portfolio['holdings'].iloc[i-1]
                            portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                    else:  # We're not holding shares
                        portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Calculate total portfolio value and returns
            portfolio['total'] = portfolio['holdings'] + portfolio['cash']
            portfolio['returns'] = portfolio['total'].pct_change()
            
            # Calculate strategy metrics
            total_return = (portfolio['total'].iloc[-1] / initial_capital) - 1.0
            annual_return = ((1 + total_return) ** (252 / len(portfolio))) - 1
            sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std())
            
            # Calculate drawdown
            portfolio['cum_returns'] = (1 + portfolio['returns']).cumprod()
            portfolio['cum_max'] = portfolio['cum_returns'].cummax()
            portfolio['drawdown'] = (portfolio['cum_returns'] / portfolio['cum_max']) - 1
            max_drawdown = portfolio['drawdown'].min()
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot price and moving averages
            ax1.plot(signals['price'], label=f'{symbol} Price')
            ax1.plot(signals['short_mavg'], label=f'{short_window} Day MA')
            ax1.plot(signals['long_mavg'], label=f'{long_window} Day MA')
            
            # Add buy/sell markers
            ax1.plot(signals.loc[signals['position'] == 1.0].index, 
                    signals['price'][signals['position'] == 1.0], 
                    '^', markersize=10, color='g', label='Buy')
            ax1.plot(signals.loc[signals['position'] == -1.0].index, 
                    signals['price'][signals['position'] == -1.0], 
                    'v', markersize=10, color='r', label='Sell')
            
            ax1.set_title(f'SMA Crossover Strategy: {symbol}')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            
            # Plot portfolio value
            ax2.plot(portfolio['total'], label='Portfolio Value')
            ax2.set_ylabel('Portfolio Value ($)')
            ax2.legend()
            ax2.grid(True)
            
            # Plot drawdown
            ax3.fill_between(portfolio.index, portfolio['drawdown'], 0, color='red', alpha=0.3)
            ax3.set_ylabel('Drawdown')
            ax3.set_xlabel('Date')
            ax3.grid(True)
            
            # Save visualization
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / f"sma_crossover_{symbol}.png")
            plt.close()
            
            # Return results
            return {
                "success": True,
                "symbol": symbol,
                "strategy": "SMA Crossover",
                "parameters": {
                    "short_window": short_window,
                    "long_window": long_window,
                    "period": period,
                    "initial_capital": initial_capital
                },
                "metrics": {
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "final_portfolio_value": portfolio['total'].iloc[-1]
                },
                "visualization_path": str(output_dir / f"sma_crossover_{symbol}.png")
            }
        
        except Exception as e:
            logger.error(f"Error in SMA crossover backtest: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def backtest_rsi_strategy(symbol: str, rsi_period: int = 14, overbought: int = 70, 
                             oversold: int = 30, period: str = "5y", 
                             initial_capital: float = 10000.0) -> Dict:
        """
        Backtest an RSI strategy
        
        Args:
            symbol: Stock symbol
            rsi_period: Period for RSI calculation
            overbought: Overbought threshold
            oversold: Oversold threshold
            period: Time period for data
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Download historical data
            data = yf.download(symbol, period=period)
            
            if data.empty:
                return {"success": False, "error": f"No data found for {symbol}"}
            
            # Create signals dataframe
            signals = pd.DataFrame(index=data.index)
            # Use 'Adj Close' if available, otherwise use 'Close'
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            signals['price'] = data[price_col]
            
            # Calculate RSI
            delta = signals['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            
            # Calculate RS and RSI
            rs = gain / loss
            signals['rsi'] = 100 - (100 / (1 + rs))
            signals['rsi'] = signals['rsi'].fillna(50)  # Fill NaN values with neutral RSI
            
            # Generate signals
            signals['signal'] = 0
            signals['position'] = 0
            
            # Buy when RSI crosses below oversold and then back up
            signals['oversold'] = signals['rsi'] < oversold
            signals['buy_signal'] = (signals['oversold'] != signals['oversold'].shift(1)) & (signals['oversold'] == False) & (signals['oversold'].shift(1) == True)
            
            # Sell when RSI crosses above overbought and then back down
            signals['overbought'] = signals['rsi'] > overbought
            signals['sell_signal'] = (signals['overbought'] != signals['overbought'].shift(1)) & (signals['overbought'] == False) & (signals['overbought'].shift(1) == True)
            
            # Convert signals to position changes
            signals.loc[signals['buy_signal'], 'position'] = 1
            signals.loc[signals['sell_signal'], 'position'] = -1
            
            # Create portfolio dataframe
            portfolio = pd.DataFrame(index=signals.index)
            portfolio['holdings'] = 0.0
            portfolio['cash'] = 0.0
            
            # Initialize first row
            portfolio.loc[portfolio.index[0], 'holdings'] = 0.0
            portfolio.loc[portfolio.index[0], 'cash'] = initial_capital
            
            # Simulate trading
            position = 0
            for i in range(1, len(signals)):
                if signals['position'].iloc[i] == 1:  # Buy
                    if position == 0:  # Only buy if not already in position
                        shares = portfolio['cash'].iloc[i-1] / signals['price'].iloc[i]
                        portfolio.loc[portfolio.index[i], 'holdings'] = shares * signals['price'].iloc[i]
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - portfolio['holdings'].iloc[i]
                        position = 1
                    else:
                        # Update holdings value based on current price
                        shares = portfolio['holdings'].iloc[i-1] / signals['price'].iloc[i-1]
                        portfolio.loc[portfolio.index[i], 'holdings'] = shares * signals['price'].iloc[i]
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                elif signals['position'].iloc[i] == -1:  # Sell
                    if position == 1:  # Only sell if in position
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + portfolio['holdings'].iloc[i-1]
                        portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                        position = 0
                    else:
                        portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                else:
                    # No trade
                    if position == 1:
                        # Update holdings value based on current price
                        shares = portfolio['holdings'].iloc[i-1] / signals['price'].iloc[i-1]
                        portfolio.loc[portfolio.index[i], 'holdings'] = shares * signals['price'].iloc[i]
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                    else:
                        portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Calculate total portfolio value and returns
            portfolio['total'] = portfolio['holdings'] + portfolio['cash']
            portfolio['returns'] = portfolio['total'].pct_change()
            
            # Calculate strategy metrics
            total_return = (portfolio['total'].iloc[-1] / initial_capital) - 1.0
            annual_return = ((1 + total_return) ** (252 / len(portfolio))) - 1
            sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std())
            
            # Calculate drawdown
            portfolio['cum_returns'] = (1 + portfolio['returns']).cumprod()
            portfolio['cum_max'] = portfolio['cum_returns'].cummax()
            portfolio['drawdown'] = (portfolio['cum_returns'] / portfolio['cum_max']) - 1
            max_drawdown = portfolio['drawdown'].min()
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot price and RSI
            ax1.plot(signals['price'], label=f'{symbol} Price')
            ax1.set_title(f'RSI Strategy: {symbol}')
            ax1.set_ylabel('Price')
            
            # Add buy/sell markers
            ax1.plot(signals.loc[signals['buy_signal']].index, 
                    signals['price'][signals['buy_signal']], 
                    '^', markersize=10, color='g', label='Buy')
            ax1.plot(signals.loc[signals['sell_signal']].index, 
                    signals['price'][signals['sell_signal']], 
                    'v', markersize=10, color='r', label='Sell')
            
            ax1.legend()
            ax1.grid(True)
            
            # Plot RSI
            ax2.plot(signals['rsi'], label='RSI', color='purple')
            ax2.axhline(y=overbought, color='r', linestyle='--', label=f'Overbought ({overbought})')
            ax2.axhline(y=oversold, color='g', linestyle='--', label=f'Oversold ({oversold})')
            ax2.set_ylabel('RSI')
            ax2.legend()
            ax2.grid(True)
            
            # Plot portfolio value
            ax3.plot(portfolio['total'], label='Portfolio Value')
            ax3.set_ylabel('Portfolio Value ($)')
            ax3.set_xlabel('Date')
            ax3.legend()
            ax3.grid(True)
            
            # Save visualization
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / f"rsi_strategy_{symbol}.png")
            plt.close()
            
            # Return results
            return {
                "success": True,
                "symbol": symbol,
                "strategy": "RSI",
                "parameters": {
                    "rsi_period": rsi_period,
                    "overbought": overbought,
                    "oversold": oversold,
                    "period": period,
                    "initial_capital": initial_capital
                },
                "metrics": {
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "final_portfolio_value": portfolio['total'].iloc[-1]
                },
                "visualization_path": str(output_dir / f"rsi_strategy_{symbol}.png")
            }
        
        except Exception as e:
            logger.error(f"Error in RSI strategy backtest: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def backtest_macd_strategy(symbol: str, fast_period: int = 12, slow_period: int = 26, 
                              signal_period: int = 9, period: str = "5y", 
                              initial_capital: float = 10000.0) -> Dict:
        """
        Backtest a MACD strategy
        
        Args:
            symbol: Stock symbol
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            period: Time period for data
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Download historical data
            data = yf.download(symbol, period=period)
            
            if data.empty:
                return {"success": False, "error": f"No data found for {symbol}"}
            
            # Create signals dataframe
            signals = pd.DataFrame(index=data.index)
            # Use 'Adj Close' if available, otherwise use 'Close'
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            signals['price'] = data[price_col]
            
            # Calculate MACD
            exp1 = signals['price'].ewm(span=fast_period, adjust=False).mean()
            exp2 = signals['price'].ewm(span=slow_period, adjust=False).mean()
            signals['macd'] = exp1 - exp2
            signals['signal'] = signals['macd'].ewm(span=signal_period, adjust=False).mean()
            signals['histogram'] = signals['macd'] - signals['signal']
            
            # Generate trading signals
            signals['position'] = 0
            
            # Buy when MACD crosses above signal line
            signals['position'] = np.where(
                (signals['macd'] > signals['signal']) & (signals['macd'].shift(1) <= signals['signal'].shift(1)),
                1, signals['position'])
            
            # Sell when MACD crosses below signal line
            signals['position'] = np.where(
                (signals['macd'] < signals['signal']) & (signals['macd'].shift(1) >= signals['signal'].shift(1)),
                -1, signals['position'])
            
            # Create portfolio dataframe
            portfolio = pd.DataFrame(index=signals.index)
            portfolio['holdings'] = 0.0
            portfolio['cash'] = 0.0
            
            # Initialize first row
            portfolio.loc[portfolio.index[0], 'holdings'] = 0.0
            portfolio.loc[portfolio.index[0], 'cash'] = initial_capital
            
            # Simulate trading
            position = 0
            for i in range(1, len(signals)):
                if signals['position'].iloc[i] == 1:  # Buy
                    if position == 0:  # Only buy if not already in position
                        shares = portfolio['cash'].iloc[i-1] / signals['price'].iloc[i]
                        portfolio.loc[portfolio.index[i], 'holdings'] = shares * signals['price'].iloc[i]
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - portfolio['holdings'].iloc[i]
                        position = 1
                    else:
                        # Update holdings value based on current price
                        shares = portfolio['holdings'].iloc[i-1] / signals['price'].iloc[i-1]
                        portfolio.loc[portfolio.index[i], 'holdings'] = shares * signals['price'].iloc[i]
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                elif signals['position'].iloc[i] == -1:  # Sell
                    if position == 1:  # Only sell if in position
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + portfolio['holdings'].iloc[i-1]
                        portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                        position = 0
                    else:
                        portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                else:
                    # No trade
                    if position == 1:
                        # Update holdings value based on current price
                        shares = portfolio['holdings'].iloc[i-1] / signals['price'].iloc[i-1]
                        portfolio.loc[portfolio.index[i], 'holdings'] = shares * signals['price'].iloc[i]
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
                    else:
                        portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                        portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Calculate total portfolio value and returns
            portfolio['total'] = portfolio['holdings'] + portfolio['cash']
            portfolio['returns'] = portfolio['total'].pct_change()
            
            # Calculate strategy metrics
            total_return = (portfolio['total'].iloc[-1] / initial_capital) - 1.0
            annual_return = ((1 + total_return) ** (252 / len(portfolio))) - 1
            sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std())
            
            # Calculate drawdown
            portfolio['cum_returns'] = (1 + portfolio['returns']).cumprod()
            portfolio['cum_max'] = portfolio['cum_returns'].cummax()
            portfolio['drawdown'] = (portfolio['cum_returns'] / portfolio['cum_max']) - 1
            max_drawdown = portfolio['drawdown'].min()
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot price
            ax1.plot(signals['price'], label=f'{symbol} Price')
            ax1.set_title(f'MACD Strategy: {symbol}')
            ax1.set_ylabel('Price')
            
            # Add buy/sell markers
            buy_signals = signals[signals['position'] == 1]
            sell_signals = signals[signals['position'] == -1]
            
            ax1.plot(buy_signals.index, signals.loc[buy_signals.index, 'price'], 
                    '^', markersize=10, color='g', label='Buy')
            ax1.plot(sell_signals.index, signals.loc[sell_signals.index, 'price'], 
                    'v', markersize=10, color='r', label='Sell')
            
            ax1.legend()
            ax1.grid(True)
            
            # Plot MACD
            ax2.plot(signals['macd'], label='MACD', color='blue')
            ax2.plot(signals['signal'], label='Signal', color='red')
            ax2.bar(signals.index, signals['histogram'], color='green', label='Histogram', alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--')
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True)
            
            # Plot portfolio value
            ax3.plot(portfolio['total'], label='Portfolio Value')
            ax3.set_ylabel('Portfolio Value ($)')
            ax3.set_xlabel('Date')
            ax3.legend()
            ax3.grid(True)
            
            # Save visualization
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / f"macd_strategy_{symbol}.png")
            plt.close()
            
            # Return results
            return {
                "success": True,
                "symbol": symbol,
                "strategy": "MACD",
                "parameters": {
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period,
                    "period": period,
                    "initial_capital": initial_capital
                },
                "metrics": {
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "final_portfolio_value": portfolio['total'].iloc[-1]
                },
                "visualization_path": str(output_dir / f"macd_strategy_{symbol}.png")
            }
        
        except Exception as e:
            logger.error(f"Error in MACD strategy backtest: {str(e)}")
            return {"success": False, "error": str(e)}
