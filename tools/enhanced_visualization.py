"""
Enhanced Visualization Tools for Finance Analyst AI Agent

This module provides advanced visualization capabilities for financial data including:
- Interactive financial trend visualizations
- Comparative ratio charts
- Combined technical and fundamental analysis visualizations
- Correlation heatmaps
- Performance benchmarking
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import base64
from io import BytesIO
import yfinance as yf
import warnings

# Try importing plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Using matplotlib for visualizations.")

# Import local modules
from tools.fundamental_analysis import FundamentalAnalysisTools

class EnhancedVisualizationTools:
    """Tools for enhanced financial data visualization"""
    
    @staticmethod
    def visualize_financial_trends(symbol: str, period: str = "1y", metrics: List[str] = None) -> Dict:
        """
        Create advanced visualization of financial trends over time
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            metrics: List of metrics to visualize (price, volume, volatility, rsi, macd)
            
        Returns:
            Dictionary with visualization results and paths
        """
        try:
            if metrics is None:
                metrics = ["price", "volume", "volatility", "rsi"]
                
            # Get stock data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return {"error": f"Could not find data for {symbol}"}
            
            # Calculate additional metrics if needed
            if "volatility" in metrics:
                # Calculate rolling volatility (20-day standard deviation of returns)
                data['Returns'] = data['Close'].pct_change()
                data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252) * 100  # Annualized volatility in %
            
            if "rsi" in metrics:
                # Calculate RSI
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
            
            if "macd" in metrics:
                # Calculate MACD
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['MACD_Histogram'] = data['MACD'] - data['Signal']
            
            # Create interactive visualization if Plotly is available
            if PLOTLY_AVAILABLE:
                # Determine number of rows based on metrics
                num_rows = sum(1 for m in metrics if m in ["price", "volume", "volatility", "rsi", "macd"])
                
                # Create subplot layout
                fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    subplot_titles=[m.upper() for m in metrics if m in ["price", "volume", "volatility", "rsi", "macd"]])
                
                row = 1
                # Add price chart
                if "price" in metrics:
                    # Add candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name='Price'
                        ),
                        row=row, col=1
                    )
                    
                    # Add moving averages
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Close'].rolling(window=50).mean(),
                            line=dict(color='orange', width=1),
                            name='50-day MA'
                        ),
                        row=row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Close'].rolling(window=200).mean(),
                            line=dict(color='red', width=1),
                            name='200-day MA'
                        ),
                        row=row, col=1
                    )
                    row += 1
                
                # Add volume chart
                if "volume" in metrics:
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name='Volume',
                            marker=dict(color='rgba(100, 100, 255, 0.5)')
                        ),
                        row=row, col=1
                    )
                    row += 1
                
                # Add volatility chart
                if "volatility" in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Volatility'],
                            line=dict(color='purple', width=1),
                            name='Volatility (20-day)'
                        ),
                        row=row, col=1
                    )
                    row += 1
                
                # Add RSI chart
                if "rsi" in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['RSI'],
                            line=dict(color='blue', width=1),
                            name='RSI (14-day)'
                        ),
                        row=row, col=1
                    )
                    
                    # Add RSI reference lines
                    fig.add_shape(
                        type="line", line_color="red", line_dash="dash",
                        x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
                        row=row, col=1
                    )
                    
                    fig.add_shape(
                        type="line", line_color="green", line_dash="dash",
                        x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
                        row=row, col=1
                    )
                    row += 1
                
                # Add MACD chart
                if "macd" in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['MACD'],
                            line=dict(color='blue', width=1),
                            name='MACD'
                        ),
                        row=row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Signal'],
                            line=dict(color='red', width=1),
                            name='Signal'
                        ),
                        row=row, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['MACD_Histogram'],
                            name='Histogram',
                            marker=dict(color='green', opacity=0.5)
                        ),
                        row=row, col=1
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"{symbol} Financial Trends - {period}",
                    height=250 * num_rows,
                    width=1000,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                # Save as HTML
                os.makedirs('outputs/financial_trends', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_path = f"outputs/financial_trends/trends_{symbol}_{timestamp}.html"
                
                with open(html_path, 'w') as f:
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                
                # Also save as PNG for compatibility
                png_path = f"outputs/financial_trends/trends_{symbol}_{timestamp}.png"
                fig.write_image(png_path)
                
                return {
                    "symbol": symbol,
                    "period": period,
                    "metrics": metrics,
                    "html_path": html_path,
                    "png_path": png_path,
                    "current_price": data['Close'].iloc[-1],
                    "price_change": ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                }
            
            else:
                # Fallback to matplotlib
                # Determine number of subplots needed
                num_plots = sum(1 for m in metrics if m in ["price", "volume", "volatility", "rsi", "macd"])
                
                # Create figure with subplots
                fig = plt.figure(figsize=(12, 4 * num_plots))
                gs = GridSpec(num_plots, 1, figure=fig)
                
                plot_idx = 0
                
                # Price plot
                if "price" in metrics:
                    ax_price = fig.add_subplot(gs[plot_idx])
                    ax_price.plot(data.index, data['Close'], label='Close Price', color='blue')
                    ax_price.set_title(f"{symbol} Price")
                    ax_price.set_ylabel('Price')
                    
                    # Add moving averages
                    ax_price.plot(data.index, data['Close'].rolling(window=50).mean(), 
                                 label='50-day MA', color='orange', alpha=0.7)
                    ax_price.plot(data.index, data['Close'].rolling(window=200).mean(), 
                                 label='200-day MA', color='red', alpha=0.7)
                    
                    ax_price.grid(True, alpha=0.3)
                    ax_price.legend(loc='upper left')
                    plot_idx += 1
                
                # Volume plot
                if "volume" in metrics:
                    ax_volume = fig.add_subplot(gs[plot_idx])
                    ax_volume.bar(data.index, data['Volume'], label='Volume', color='blue', alpha=0.5)
                    ax_volume.set_title(f"{symbol} Volume")
                    ax_volume.set_ylabel('Volume')
                    ax_volume.grid(True, alpha=0.3)
                    plot_idx += 1
                
                # Volatility plot
                if "volatility" in metrics:
                    ax_vol = fig.add_subplot(gs[plot_idx])
                    ax_vol.plot(data.index, data['Volatility'], label='Volatility (20-day)', color='purple')
                    ax_vol.set_title(f"{symbol} Volatility")
                    ax_vol.set_ylabel('Volatility (%)')
                    ax_vol.grid(True, alpha=0.3)
                    plot_idx += 1
                
                # RSI plot
                if "rsi" in metrics:
                    ax_rsi = fig.add_subplot(gs[plot_idx])
                    ax_rsi.plot(data.index, data['RSI'], label='RSI (14-day)', color='blue')
                    ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                    ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                    ax_rsi.set_title(f"{symbol} RSI")
                    ax_rsi.set_ylabel('RSI')
                    ax_rsi.set_ylim(0, 100)
                    ax_rsi.grid(True, alpha=0.3)
                    plot_idx += 1
                
                # MACD plot
                if "macd" in metrics:
                    ax_macd = fig.add_subplot(gs[plot_idx])
                    ax_macd.plot(data.index, data['MACD'], label='MACD', color='blue')
                    ax_macd.plot(data.index, data['Signal'], label='Signal', color='red')
                    ax_macd.bar(data.index, data['MACD_Histogram'], label='Histogram', color='green', alpha=0.5)
                    ax_macd.set_title(f"{symbol} MACD")
                    ax_macd.set_ylabel('MACD')
                    ax_macd.grid(True, alpha=0.3)
                    ax_macd.legend(loc='upper left')
                
                # Format x-axis dates
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save plot
                os.makedirs('outputs/financial_trends', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                png_path = f"outputs/financial_trends/trends_{symbol}_{timestamp}.png"
                plt.savefig(png_path, dpi=100)
                plt.close()
                
                return {
                    "symbol": symbol,
                    "period": period,
                    "metrics": metrics,
                    "png_path": png_path,
                    "current_price": data['Close'].iloc[-1],
                    "price_change": ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                }
                
        except Exception as e:
            return {"error": f"Error visualizing financial trends for {symbol}: {str(e)}"}
    
    @staticmethod
    def create_correlation_matrix(symbols: List[str], period: str = "1y") -> Dict:
        """
        Create a correlation matrix visualization for multiple stocks
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with correlation matrix results and path
        """
        try:
            # Get data for all symbols
            data = pd.DataFrame()
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close']
            
            if data.empty:
                return {"error": "Could not retrieve data for any of the provided symbols"}
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Create heatmap visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                       linewidths=0.5, fmt=".2f")
            plt.title(f"Correlation Matrix - {period}")
            plt.tight_layout()
            
            # Save plot
            os.makedirs('outputs/correlation_matrix', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = f"outputs/correlation_matrix/corr_matrix_{timestamp}.png"
            plt.savefig(png_path, dpi=100)
            plt.close()
            
            # Format correlation data for return
            corr_data = []
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    if i != j:
                        corr_data.append({
                            "symbol1": symbols[i],
                            "symbol2": symbols[j],
                            "correlation": float(corr_matrix.iloc[i, j]),
                            "relationship": "Strong Positive" if corr_matrix.iloc[i, j] > 0.7 else
                                           "Moderate Positive" if corr_matrix.iloc[i, j] > 0.3 else
                                           "Weak Positive" if corr_matrix.iloc[i, j] > 0 else
                                           "Weak Negative" if corr_matrix.iloc[i, j] > -0.3 else
                                           "Moderate Negative" if corr_matrix.iloc[i, j] > -0.7 else
                                           "Strong Negative"
                        })
            
            return {
                "symbols": symbols,
                "period": period,
                "correlation_data": corr_data,
                "png_path": png_path
            }
            
        except Exception as e:
            return {"error": f"Error creating correlation matrix: {str(e)}"}
    
    @staticmethod
    def visualize_performance_comparison(symbols: List[str], period: str = "1y", benchmark: str = "^GSPC") -> Dict:
        """
        Create a performance comparison visualization for multiple stocks against a benchmark
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            benchmark: Benchmark symbol (default: S&P 500)
            
        Returns:
            Dictionary with performance comparison results and path
        """
        try:
            # Get data for all symbols and benchmark
            all_symbols = symbols + [benchmark]
            data = pd.DataFrame()
            
            for symbol in all_symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close']
            
            if data.empty:
                return {"error": "Could not retrieve data for any of the provided symbols"}
            
            # Normalize to 100 at the start
            normalized_data = data.div(data.iloc[0]) * 100
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot symbols
            for symbol in symbols:
                if symbol in normalized_data.columns:
                    plt.plot(normalized_data.index, normalized_data[symbol], label=symbol)
            
            # Plot benchmark
            if benchmark in normalized_data.columns:
                plt.plot(normalized_data.index, normalized_data[benchmark], 'k--', label=benchmark)
            
            plt.title(f"Performance Comparison - {period}")
            plt.xlabel('Date')
            plt.ylabel('Normalized Price (Base 100)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            os.makedirs('outputs/performance_comparison', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = f"outputs/performance_comparison/perf_comp_{timestamp}.png"
            plt.savefig(png_path, dpi=100)
            plt.close()
            
            # Calculate performance metrics
            performance_data = []
            for symbol in all_symbols:
                if symbol in normalized_data.columns:
                    start_price = data[symbol].iloc[0]
                    end_price = data[symbol].iloc[-1]
                    total_return = ((end_price / start_price) - 1) * 100
                    
                    # Calculate annualized return
                    days = (data.index[-1] - data.index[0]).days
                    if days > 0:
                        annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100
                    else:
                        annualized_return = total_return
                    
                    # Calculate volatility
                    daily_returns = data[symbol].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252) * 100
                    
                    # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
                    sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility != 0 else 0
                    
                    performance_data.append({
                        "symbol": symbol,
                        "total_return": float(total_return),
                        "annualized_return": float(annualized_return),
                        "volatility": float(volatility),
                        "sharpe_ratio": float(sharpe_ratio),
                        "is_benchmark": symbol == benchmark
                    })
            
            return {
                "symbols": symbols,
                "benchmark": benchmark,
                "period": period,
                "performance_data": performance_data,
                "png_path": png_path
            }
            
        except Exception as e:
            return {"error": f"Error creating performance comparison: {str(e)}"}
    
    @staticmethod
    def visualize_financial_ratios(symbol: str, comparison_symbols: List[str] = None) -> Dict:
        """
        Create a visualization of financial ratios for a stock with optional peer comparison
        
        Args:
            symbol: Stock ticker symbol
            comparison_symbols: Optional list of peer companies for comparison
            
        Returns:
            Dictionary with financial ratio visualization results and path
        """
        try:
            # Get financial ratios for the main symbol
            ratios = FundamentalAnalysisTools.get_financial_ratios(symbol)
            if "error" in ratios:
                return ratios
            
            # Extract company name
            company_name = ratios["company_name"]
            
            # Extract ratio values
            main_ratios = {}
            for ratio_name, ratio_data in ratios["ratios"].items():
                if ratio_data["value"] is not None:
                    main_ratios[ratio_name] = ratio_data["value"]
            
            # Get comparison data if requested
            comparison_data = {}
            if comparison_symbols:
                for comp_symbol in comparison_symbols:
                    comp_ratios = FundamentalAnalysisTools.get_financial_ratios(comp_symbol)
                    if "error" not in comp_ratios:
                        comparison_data[comp_symbol] = {
                            "company_name": comp_ratios["company_name"],
                            "ratios": {}
                        }
                        for ratio_name, ratio_data in comp_ratios["ratios"].items():
                            if ratio_data["value"] is not None:
                                comparison_data[comp_symbol]["ratios"][ratio_name] = ratio_data["value"]
            
            # Create visualization
            # Determine which ratios to plot based on available data
            available_ratios = list(main_ratios.keys())
            
            if not available_ratios:
                return {"error": f"No valid financial ratios available for {symbol}"}
            
            # Create bar chart for each ratio
            num_ratios = len(available_ratios)
            fig, axes = plt.subplots(num_ratios, 1, figsize=(10, 4 * num_ratios))
            
            # Handle case with only one ratio
            if num_ratios == 1:
                axes = [axes]
            
            for i, ratio_name in enumerate(available_ratios):
                ax = axes[i]
                
                # Format ratio name for display
                if ratio_name == "pe_ratio":
                    display_name = "P/E Ratio"
                elif ratio_name == "forward_pe_ratio":
                    display_name = "Forward P/E"
                elif ratio_name == "peg_ratio":
                    display_name = "PEG Ratio"
                elif ratio_name == "ps_ratio":
                    display_name = "P/S Ratio"
                elif ratio_name == "pb_ratio":
                    display_name = "P/B Ratio"
                elif ratio_name == "de_ratio":
                    display_name = "D/E Ratio"
                elif ratio_name == "roe":
                    display_name = "ROE"
                elif ratio_name == "eps":
                    display_name = "EPS"
                else:
                    display_name = ratio_name.replace("_", " ").title()
                
                # Create list of companies and values
                companies = [company_name]
                values = [main_ratios[ratio_name]]
                
                # Add comparison companies
                for comp_symbol, comp_data in comparison_data.items():
                    if ratio_name in comp_data["ratios"]:
                        companies.append(comp_data["company_name"])
                        values.append(comp_data["ratios"][ratio_name])
                
                # Create bar chart
                bars = ax.bar(companies, values)
                ax.set_title(f"{display_name}")
                ax.set_ylabel(display_name)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    if ratio_name in ["roe"]:
                        # Format as percentage
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f"{height*100:.1f}%", ha='center', va='bottom')
                    elif ratio_name in ["eps"]:
                        # Format as currency
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f"${height:.2f}", ha='center', va='bottom')
                    else:
                        # Format as decimal
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f"{height:.2f}", ha='center', va='bottom')
                
                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs('outputs/financial_ratios', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = f"outputs/financial_ratios/ratios_{symbol}_{timestamp}.png"
            plt.savefig(png_path, dpi=100)
            plt.close()
            
            return {
                "symbol": symbol,
                "company_name": company_name,
                "comparison_symbols": comparison_symbols,
                "ratios": main_ratios,
                "comparison_data": comparison_data,
                "png_path": png_path
            }
            
        except Exception as e:
            return {"error": f"Error visualizing financial ratios for {symbol}: {str(e)}"}
