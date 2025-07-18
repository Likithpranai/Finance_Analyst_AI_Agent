"""
Backtesting Visualization and Paper Trading Simulation Tools for Finance Analyst AI Agent

This module provides visualization and paper trading capabilities including:
- Backtest result visualization
- Performance metrics charts
- Paper trading simulation
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestVisualizationTools:
    """Tools for visualizing backtest results and paper trading"""
    
    @staticmethod
    def visualize_backtest_results(backtest_results: Dict, output_dir: str = None) -> Dict:
        """
        Create visualizations for backtest results
        
        Args:
            backtest_results: Backtest results dictionary
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with visualization file paths
        """
        try:
            # Validate inputs
            if "success" not in backtest_results or not backtest_results["success"]:
                return {"error": "Invalid backtest results"}
            
            # Create output directory if not provided
            if output_dir is None:
                output_dir = os.path.join(os.getcwd(), "outputs", "backtests")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract data
            symbol = backtest_results.get("symbol", "")
            strategy = backtest_results.get("strategy", "")
            signals = pd.DataFrame(backtest_results.get("signals", {}))
            portfolio = pd.DataFrame(backtest_results.get("portfolio", {}))
            
            # Create visualizations
            visualizations = {}
            
            # 1. Portfolio Performance Chart
            if not portfolio.empty and 'total' in portfolio.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio.index, portfolio['total'], label='Portfolio Value')
                
                # Add buy and sell markers if available
                if not signals.empty and 'position' in signals.columns:
                    buy_signals = signals[signals['position'] == 1]
                    sell_signals = signals[signals['position'] == -1]
                    
                    if not buy_signals.empty and 'price' in signals.columns:
                        plt.scatter(buy_signals.index, buy_signals['price'], 
                                   marker='^', color='g', s=100, label='Buy Signal')
                    
                    if not sell_signals.empty and 'price' in signals.columns:
                        plt.scatter(sell_signals.index, sell_signals['price'], 
                                   marker='v', color='r', s=100, label='Sell Signal')
                
                plt.title(f'{symbol} - {strategy} Performance')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.legend()
                plt.grid(True)
                
                # Save figure
                performance_file = os.path.join(output_dir, f"{symbol}_{strategy}_performance_{timestamp}.png")
                plt.savefig(performance_file)
                plt.close()
                
                visualizations["performance_chart"] = performance_file
            
            # 2. Strategy Signals Chart
            if not signals.empty and 'price' in signals.columns:
                plt.figure(figsize=(12, 8))
                
                # Plot price
                plt.subplot(2, 1, 1)
                plt.plot(signals.index, signals['price'], label='Price')
                
                # Add buy and sell markers
                if 'position' in signals.columns:
                    buy_signals = signals[signals['position'] == 1]
                    sell_signals = signals[signals['position'] == -1]
                    
                    if not buy_signals.empty:
                        plt.scatter(buy_signals.index, buy_signals['price'], 
                                   marker='^', color='g', s=100, label='Buy Signal')
                    
                    if not sell_signals.empty:
                        plt.scatter(sell_signals.index, sell_signals['price'], 
                                   marker='v', color='r', s=100, label='Sell Signal')
                
                plt.title(f'{symbol} - {strategy} Signals')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid(True)
                
                # Plot strategy-specific indicators
                plt.subplot(2, 1, 2)
                
                if strategy == "SMA Crossover" and 'short_mavg' in signals.columns and 'long_mavg' in signals.columns:
                    plt.plot(signals.index, signals['short_mavg'], label='Short MA')
                    plt.plot(signals.index, signals['long_mavg'], label='Long MA')
                    plt.title('Moving Averages')
                    
                elif strategy == "RSI Strategy" and 'rsi' in signals.columns:
                    plt.plot(signals.index, signals['rsi'], label='RSI')
                    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
                    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
                    plt.title('RSI Indicator')
                    
                elif strategy == "MACD Strategy" and 'macd' in signals.columns and 'signal' in signals.columns:
                    plt.plot(signals.index, signals['macd'], label='MACD')
                    plt.plot(signals.index, signals['signal'], label='Signal')
                    
                    if 'histogram' in signals.columns:
                        plt.bar(signals.index, signals['histogram'], color='gray', alpha=0.3, label='Histogram')
                    
                    plt.title('MACD Indicator')
                
                plt.xlabel('Date')
                plt.legend()
                plt.grid(True)
                
                # Save figure
                signals_file = os.path.join(output_dir, f"{symbol}_{strategy}_signals_{timestamp}.png")
                plt.savefig(signals_file)
                plt.close()
                
                visualizations["signals_chart"] = signals_file
            
            # 3. Performance Metrics Chart
            performance = backtest_results.get("performance", {})
            if performance:
                # Create bar chart for key metrics
                metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']
                values = [performance.get(metric, 0) for metric in metrics]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2%}' if bar.get_height() != performance.get('sharpe_ratio', 0) else f'{height:.2f}',
                            ha='center', va='bottom')
                
                plt.title(f'{symbol} - {strategy} Performance Metrics')
                plt.ylabel('Value')
                plt.grid(True, axis='y')
                
                # Save figure
                metrics_file = os.path.join(output_dir, f"{symbol}_{strategy}_metrics_{timestamp}.png")
                plt.savefig(metrics_file)
                plt.close()
                
                visualizations["metrics_chart"] = metrics_file
                
                # 4. Strategy vs Buy & Hold Comparison
                if 'buy_hold_return' in performance:
                    labels = ['Strategy', 'Buy & Hold']
                    returns = [performance.get('total_return', 0), performance.get('buy_hold_return', 0)]
                    
                    plt.figure(figsize=(8, 6))
                    bars = plt.bar(labels, returns, color=['blue', 'gray'])
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2%}',
                                ha='center', va='bottom')
                    
                    plt.title(f'{symbol} - Strategy vs Buy & Hold')
                    plt.ylabel('Total Return')
                    plt.grid(True, axis='y')
                    
                    # Save figure
                    comparison_file = os.path.join(output_dir, f"{symbol}_{strategy}_comparison_{timestamp}.png")
                    plt.savefig(comparison_file)
                    plt.close()
                    
                    visualizations["comparison_chart"] = comparison_file
            
            # 5. Create interactive Plotly chart
            if not signals.empty and not portfolio.empty and 'price' in signals.columns and 'total' in portfolio.columns:
                # Create subplots
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, 
                                   subplot_titles=(f'{symbol} Price and Signals', 'Portfolio Value'))
                
                # Add price trace
                fig.add_trace(
                    go.Scatter(x=signals.index, y=signals['price'], name='Price', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Add buy signals
                if 'position' in signals.columns:
                    buy_signals = signals[signals['position'] == 1]
                    if not buy_signals.empty:
                        fig.add_trace(
                            go.Scatter(x=buy_signals.index, y=buy_signals['price'], 
                                      mode='markers', name='Buy Signal',
                                      marker=dict(color='green', size=10, symbol='triangle-up')),
                            row=1, col=1
                        )
                
                    # Add sell signals
                    sell_signals = signals[signals['position'] == -1]
                    if not sell_signals.empty:
                        fig.add_trace(
                            go.Scatter(x=sell_signals.index, y=sell_signals['price'], 
                                      mode='markers', name='Sell Signal',
                                      marker=dict(color='red', size=10, symbol='triangle-down')),
                            row=1, col=1
                        )
                
                # Add strategy-specific indicators
                if strategy == "SMA Crossover" and 'short_mavg' in signals.columns and 'long_mavg' in signals.columns:
                    fig.add_trace(
                        go.Scatter(x=signals.index, y=signals['short_mavg'], name='Short MA', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=signals.index, y=signals['long_mavg'], name='Long MA', line=dict(color='purple')),
                        row=1, col=1
                    )
                    
                elif strategy == "RSI Strategy" and 'rsi' in signals.columns:
                    fig.add_trace(
                        go.Scatter(x=signals.index, y=signals['rsi'], name='RSI', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=signals.index, y=[70] * len(signals), name='Overbought', 
                                  line=dict(color='red', dash='dash')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=signals.index, y=[30] * len(signals), name='Oversold', 
                                  line=dict(color='green', dash='dash')),
                        row=1, col=1
                    )
                    
                elif strategy == "MACD Strategy" and 'macd' in signals.columns and 'signal' in signals.columns:
                    fig.add_trace(
                        go.Scatter(x=signals.index, y=signals['macd'], name='MACD', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=signals.index, y=signals['signal'], name='Signal', line=dict(color='purple')),
                        row=1, col=1
                    )
                
                # Add portfolio value trace
                fig.add_trace(
                    go.Scatter(x=portfolio.index, y=portfolio['total'], name='Portfolio Value', line=dict(color='green')),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=f'{symbol} - {strategy} Backtest Results',
                    height=800,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Update y-axis labels
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
                
                # Update x-axis
                fig.update_xaxes(title_text="Date", row=2, col=1)
                
                # Save as HTML
                interactive_file = os.path.join(output_dir, f"{symbol}_{strategy}_interactive_{timestamp}.html")
                fig.write_html(interactive_file)
                
                visualizations["interactive_chart"] = interactive_file
            
            return {
                "success": True,
                "visualizations": visualizations,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Error visualizing backtest results: {str(e)}")
            return {"error": f"Error visualizing backtest results: {str(e)}"}
    
    @staticmethod
    def paper_trading_simulation(symbol: str, strategy: str, parameters: Dict, 
                               initial_capital: float = 10000.0, 
                               start_date: str = None, end_date: str = None) -> Dict:
        """
        Run a paper trading simulation for a given strategy
        
        Args:
            symbol: Stock symbol
            strategy: Strategy name ('sma_crossover', 'rsi', 'macd')
            parameters: Strategy parameters
            initial_capital: Initial capital
            start_date: Start date for simulation (format: 'YYYY-MM-DD')
            end_date: End date for simulation (format: 'YYYY-MM-DD')
            
        Returns:
            Dictionary with simulation results
        """
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Download historical data
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if data.empty:
                return {"error": f"No data found for symbol {symbol} in the specified date range"}
            
            # Initialize results
            signals = pd.DataFrame(index=data.index)
            signals['price'] = data['Close']
            
            # Generate signals based on strategy
            if strategy.lower() == 'sma_crossover':
                # Extract parameters
                short_window = parameters.get('short_window', 50)
                long_window = parameters.get('long_window', 200)
                
                # Calculate moving averages
                signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
                signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
                
                # Generate signals
                signals['signal'] = 0.0
                signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, 0.0)
                signals['position'] = signals['signal'].diff()
                
            elif strategy.lower() == 'rsi':
                # Extract parameters
                rsi_period = parameters.get('rsi_period', 14)
                overbought = parameters.get('overbought', 70)
                oversold = parameters.get('oversold', 30)
                
                # Calculate RSI
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
                rs = gain / loss
                signals['rsi'] = 100 - (100 / (1 + rs))
                
                # Generate signals
                signals['signal'] = 0.0
                signals['signal'] = np.where(signals['rsi'] < oversold, 1.0, 0.0)
                signals['signal'] = np.where(signals['rsi'] > overbought, -1.0, signals['signal'])
                signals['position'] = signals['signal'].diff()
                
            elif strategy.lower() == 'macd':
                # Extract parameters
                fast_period = parameters.get('fast_period', 12)
                slow_period = parameters.get('slow_period', 26)
                signal_period = parameters.get('signal_period', 9)
                
                # Calculate MACD
                exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
                exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
                signals['macd'] = exp1 - exp2
                signals['signal_line'] = signals['macd'].ewm(span=signal_period, adjust=False).mean()
                signals['histogram'] = signals['macd'] - signals['signal_line']
                
                # Generate signals
                signals['signal'] = 0.0
                signals['signal'] = np.where(signals['macd'] > signals['signal_line'], 1.0, 0.0)
                signals['position'] = signals['signal'].diff()
                
            else:
                return {"error": f"Unsupported strategy: {strategy}"}
            
            # Simulate trading
            portfolio = pd.DataFrame(index=signals.index)
            portfolio['holdings'] = 0.0
            portfolio['cash'] = initial_capital
            portfolio['total'] = initial_capital
            
            position = 0
            shares = 0
            
            for i in range(1, len(signals)):
                # Update portfolio value based on current position
                if position == 1:
                    portfolio['holdings'].iloc[i] = shares * signals['price'].iloc[i]
                else:
                    portfolio['holdings'].iloc[i] = 0
                
                # Process buy signal
                if signals['position'].iloc[i] == 1:
                    if position == 0:  # Buy only if not already in position
                        shares = portfolio['cash'].iloc[i-1] / signals['price'].iloc[i]
                        portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1] - (shares * signals['price'].iloc[i])
                        portfolio['holdings'].iloc[i] = shares * signals['price'].iloc[i]
                        position = 1
                    else:
                        portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1]
                
                # Process sell signal
                elif signals['position'].iloc[i] == -1:
                    if position == 1:  # Sell only if in position
                        portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1] + portfolio['holdings'].iloc[i-1]
                        portfolio['holdings'].iloc[i] = 0
                        position = 0
                        shares = 0
                    else:
                        portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1]
                
                # No signal
                else:
                    portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1]
                
                # Update total portfolio value
                portfolio['total'].iloc[i] = portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]
            
            # Calculate performance metrics
            portfolio['returns'] = portfolio['total'].pct_change()
            
            total_return = (portfolio['total'].iloc[-1] / initial_capital) - 1
            annual_return = ((1 + total_return) ** (252 / len(portfolio))) - 1
            sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std())
            
            # Calculate drawdown
            portfolio['cum_returns'] = (1 + portfolio['returns']).cumprod()
            portfolio['cum_max'] = portfolio['cum_returns'].cummax()
            portfolio['drawdown'] = (portfolio['cum_returns'] / portfolio['cum_max']) - 1
            max_drawdown = portfolio['drawdown'].min()
            
            # Calculate number of trades
            num_trades = signals['position'].abs().sum() / 2
            
            # Calculate buy and hold return
            buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
            
            # Prepare results
            results = {
                "success": True,
                "symbol": symbol,
                "strategy": strategy,
                "parameters": parameters,
                "simulation_period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "initial_capital": initial_capital,
                "final_portfolio_value": portfolio['total'].iloc[-1],
                "performance": {
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "num_trades": num_trades,
                    "buy_hold_return": buy_hold_return
                },
                "current_position": {
                    "position": position,
                    "shares": shares,
                    "holdings_value": portfolio['holdings'].iloc[-1],
                    "cash": portfolio['cash'].iloc[-1]
                },
                "signals": signals.to_dict(),
                "portfolio": portfolio.to_dict()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in paper trading simulation: {str(e)}")
            return {"error": f"Error in paper trading simulation: {str(e)}"}
