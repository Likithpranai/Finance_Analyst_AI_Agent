"""
Interactive Visualization Tools for Finance Analyst AI Agent

This module provides advanced interactive visualization capabilities including:
- Interactive financial charts with zoom, pan, and hover capabilities
- Multi-timeframe analysis views
- Custom technical indicator overlays
- Volume profiles and advanced chart patterns
- Real-time updates via websockets (when available)

Built with Plotly and Bokeh for TradingView-like functionality
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import yfinance as yf
import warnings

# Import Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import Bokeh as an alternative
try:
    from bokeh.plotting import figure
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool
    from bokeh.models.widgets import Tabs, Panel
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    warnings.warn("Bokeh not available. Using Plotly for all visualizations.")

# Import local modules
from tools.fundamental_analysis import FundamentalAnalysisTools


class InteractiveVisualizationTools:
    """Tools for interactive financial data visualization"""
    
    @staticmethod
    def create_interactive_chart(symbol: str, period: str = "1y", 
                               indicators: List[str] = None,
                               comparison_symbols: List[str] = None) -> Dict:
        """
        Create an interactive financial chart with customizable indicators
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            indicators: List of technical indicators to include
                Options: ["sma", "ema", "bollinger", "rsi", "macd", "volume_profile"]
            comparison_symbols: List of symbols to compare with the main symbol
            
        Returns:
            Dictionary with HTML path and metadata
        """
        try:
            if indicators is None:
                indicators = ["sma", "ema", "rsi"]
                
            # Get stock data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return {"error": f"Could not find data for {symbol}"}
            
            # Calculate indicators
            df = data.copy()
            
            # Calculate SMA
            if "sma" in indicators:
                df['SMA20'] = df['Close'].rolling(window=20).mean()
                df['SMA50'] = df['Close'].rolling(window=50).mean()
                df['SMA200'] = df['Close'].rolling(window=200).mean()
            
            # Calculate EMA
            if "ema" in indicators:
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # Calculate Bollinger Bands
            if "bollinger" in indicators:
                df['SMA20'] = df['Close'].rolling(window=20).mean()
                df['Upper'] = df['SMA20'] + (df['Close'].rolling(window=20).std() * 2)
                df['Lower'] = df['SMA20'] - (df['Close'].rolling(window=20).std() * 2)
            
            # Calculate RSI
            if "rsi" in indicators:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            if "macd" in indicators:
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Histogram'] = df['MACD'] - df['Signal']
            
            # Determine number of rows based on indicators
            has_rsi = "rsi" in indicators
            has_macd = "macd" in indicators
            has_volume = True  # Always include volume
            
            num_rows = 1 + has_rsi + has_macd + has_volume
            row_heights = [0.5] + [0.15] * (num_rows - 1)
            
            # Create subplot layout
            fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, row_heights=row_heights,
                                subplot_titles=[f"{symbol} Price"] + 
                                              (["Volume"] if has_volume else []) +
                                              (["RSI"] if has_rsi else []) + 
                                              (["MACD"] if has_macd else []))
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=symbol,
                increasing_line_color='#26A69A', 
                decreasing_line_color='#EF5350'
            ), row=1, col=1)
            
            # Add SMA lines
            if "sma" in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['SMA20'],
                    line=dict(color='blue', width=1),
                    name='SMA20'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['SMA50'],
                    line=dict(color='orange', width=1),
                    name='SMA50'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['SMA200'],
                    line=dict(color='red', width=1),
                    name='SMA200'
                ), row=1, col=1)
            
            # Add EMA lines
            if "ema" in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['EMA12'],
                    line=dict(color='purple', width=1, dash='dash'),
                    name='EMA12'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['EMA26'],
                    line=dict(color='magenta', width=1, dash='dash'),
                    name='EMA26'
                ), row=1, col=1)
            
            # Add Bollinger Bands
            if "bollinger" in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Upper'],
                    line=dict(color='rgba(0,0,255,0.5)', width=1),
                    name='Upper Band',
                    fill=None
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Lower'],
                    line=dict(color='rgba(0,0,255,0.5)', width=1),
                    name='Lower Band',
                    fill='tonexty',
                    fillcolor='rgba(0,0,255,0.1)'
                ), row=1, col=1)
            
            # Add comparison symbols
            if comparison_symbols:
                for comp_symbol in comparison_symbols:
                    try:
                        comp_ticker = yf.Ticker(comp_symbol)
                        comp_data = comp_ticker.history(period=period)
                        
                        if not comp_data.empty:
                            # Normalize to percentage change for fair comparison
                            first_price = comp_data['Close'].iloc[0]
                            norm_data = (comp_data['Close'] / first_price - 1) * 100
                            
                            # Also normalize the main symbol data for comparison
                            if comp_symbol == comparison_symbols[0]:  # Only do this once
                                first_main_price = df['Close'].iloc[0]
                                norm_main_data = (df['Close'] / first_main_price - 1) * 100
                                
                                fig.add_trace(go.Scatter(
                                    x=df.index, y=norm_main_data,
                                    line=dict(color='blue', width=2),
                                    name=f"{symbol} %",
                                    visible="legendonly"
                                ), row=1, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=comp_data.index, y=norm_data,
                                line=dict(width=1.5),
                                name=f"{comp_symbol} %",
                                visible="legendonly"
                            ), row=1, col=1)
                    except Exception as e:
                        print(f"Error adding comparison symbol {comp_symbol}: {str(e)}")
            
            # Add volume chart
            current_row = 2
            
            if has_volume:
                colors = ['#26A69A' if row['Close'] >= row['Open'] else '#EF5350' for _, row in df.iterrows()]
                
                fig.add_trace(go.Bar(
                    x=df.index, y=df['Volume'],
                    marker_color=colors,
                    name='Volume'
                ), row=current_row, col=1)
                current_row += 1
            
            # Add RSI
            if has_rsi:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['RSI'],
                    line=dict(color='purple', width=1),
                    name='RSI'
                ), row=current_row, col=1)
                
                # Add RSI overbought/oversold lines
                fig.add_shape(
                    type="line", line_color="red", line_width=1, opacity=0.5, line_dash="dash",
                    x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
                    xref=f"x{current_row}", yref=f"y{current_row}"
                )
                
                fig.add_shape(
                    type="line", line_color="green", line_width=1, opacity=0.5, line_dash="dash",
                    x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
                    xref=f"x{current_row}", yref=f"y{current_row}"
                )
                
                current_row += 1
            
            # Add MACD
            if has_macd:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['MACD'],
                    line=dict(color='blue', width=1),
                    name='MACD'
                ), row=current_row, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Signal'],
                    line=dict(color='red', width=1),
                    name='Signal'
                ), row=current_row, col=1)
                
                colors = ['#26A69A' if val >= 0 else '#EF5350' for val in df['MACD_Histogram']]
                
                fig.add_trace(go.Bar(
                    x=df.index, y=df['MACD_Histogram'],
                    marker_color=colors,
                    name='Histogram'
                ), row=current_row, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"{ticker.info.get('longName', symbol)} Interactive Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=800,
                width=1200,
                legend_title="Legend",
                template="plotly_white",
                xaxis_rangeslider_visible=False,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Add range selector
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                row=1, col=1
            )
            
            # Add crosshair and hover tools
            fig.update_layout(
                hovermode="x unified",
                hoverdistance=100,
                spikedistance=1000,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            fig.update_xaxes(
                showspikes=True,
                spikesnap="cursor",
                spikemode="across",
                spikethickness=1,
                spikedash="solid"
            )
            
            fig.update_yaxes(
                showspikes=True,
                spikesnap="cursor",
                spikemode="across",
                spikethickness=1,
                spikedash="solid"
            )
            
            # Save as interactive HTML
            os.makedirs('outputs/interactive_charts', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = f"outputs/interactive_charts/chart_{symbol}_{timestamp}.html"
            fig.write_html(html_path, include_plotlyjs='cdn')
            
            # Create a static image version as well
            png_path = f"outputs/interactive_charts/chart_{symbol}_{timestamp}.png"
            fig.write_image(png_path)
            
            return {
                "symbol": symbol,
                "period": period,
                "indicators": indicators,
                "html_path": html_path,
                "png_path": png_path,
                "current_price": df['Close'].iloc[-1],
                "price_change": ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            }
            
        except Exception as e:
            return {"error": f"Error creating interactive chart for {symbol}: {str(e)}"}
    
    @staticmethod
    def create_financial_dashboard(symbols: List[str], period: str = "1y") -> Dict:
        """
        Create a financial dashboard with multiple charts and KPIs
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period for analysis
            
        Returns:
            Dictionary with HTML path and metadata
        """
        try:
            if not symbols:
                return {"error": "No symbols provided"}
            
            # Create a dashboard with multiple subplots
            fig = make_subplots(
                rows=len(symbols), cols=1,
                subplot_titles=[f"{symbol} Price" for symbol in symbols],
                vertical_spacing=0.1,
                specs=[[{"secondary_y": True}] for _ in symbols]
            )
            
            # Add data for each symbol
            for i, symbol in enumerate(symbols):
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    
                    if data.empty:
                        continue
                    
                    # Add candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name=symbol,
                            increasing_line_color='#26A69A', 
                            decreasing_line_color='#EF5350'
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add volume as bar chart on secondary y-axis
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name=f"{symbol} Volume",
                            marker_color='rgba(100, 100, 100, 0.5)',
                            opacity=0.5
                        ),
                        row=i+1, col=1,
                        secondary_y=True
                    )
                    
                    # Add 50-day moving average
                    sma50 = data['Close'].rolling(window=50).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=sma50,
                            line=dict(color='orange', width=1),
                            name=f"{symbol} SMA50"
                        ),
                        row=i+1, col=1
                    )
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
            
            # Update layout
            fig.update_layout(
                title="Financial Dashboard",
                height=300 * len(symbols),
                width=1200,
                template="plotly_white",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Disable range sliders for all but the last plot
            for i in range(len(symbols)-1):
                fig.update_xaxes(rangeslider_visible=False, row=i+1, col=1)
            
            # Add range selector to the last plot
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                row=len(symbols), col=1
            )
            
            # Save as interactive HTML
            os.makedirs('outputs/dashboards', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = f"outputs/dashboards/dashboard_{timestamp}.html"
            fig.write_html(html_path, include_plotlyjs='cdn')
            
            # Create a static image version as well
            png_path = f"outputs/dashboards/dashboard_{timestamp}.png"
            fig.write_image(png_path)
            
            return {
                "symbols": symbols,
                "period": period,
                "html_path": html_path,
                "png_path": png_path
            }
            
        except Exception as e:
            return {"error": f"Error creating financial dashboard: {str(e)}"}
    
    @staticmethod
    def create_market_heatmap(sector: str = None, market_cap_min: float = None) -> Dict:
        """
        Create a market heatmap showing performance by sector or industry
        
        Args:
            sector: Filter by specific sector
            market_cap_min: Minimum market cap in billions
            
        Returns:
            Dictionary with HTML path and metadata
        """
        try:
            # For a real implementation, you would use a market data API
            # Here we'll use a sample of major stocks
            symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "PG", "JNJ",
                      "WMT", "HD", "BAC", "PFE", "DIS", "NFLX", "CSCO", "INTC", "VZ", "KO"]
            
            # Get data for all symbols
            data = {}
            sectors = {}
            market_caps = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    
                    if not hist.empty:
                        # Calculate daily returns
                        daily_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                        
                        # Get sector info
                        info = ticker.info
                        sector_name = info.get('sector', 'Unknown')
                        market_cap = info.get('marketCap', 0) / 1e9  # Convert to billions
                        
                        # Filter by sector if specified
                        if sector and sector_name != sector:
                            continue
                            
                        # Filter by market cap if specified
                        if market_cap_min and market_cap < market_cap_min:
                            continue
                        
                        data[symbol] = daily_return
                        sectors[symbol] = sector_name
                        market_caps[symbol] = market_cap
                        
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
            
            if not data:
                return {"error": "No data available for the specified filters"}
            
            # Create DataFrame for heatmap
            df = pd.DataFrame({
                'Symbol': list(data.keys()),
                'Return': list(data.values()),
                'Sector': [sectors.get(s, 'Unknown') for s in data.keys()],
                'MarketCap': [market_caps.get(s, 0) for s in data.keys()]
            })
            
            # Sort by sector and market cap
            df = df.sort_values(['Sector', 'MarketCap'], ascending=[True, False])
            
            # Create heatmap using Plotly
            fig = px.treemap(
                df,
                path=['Sector', 'Symbol'],
                values='MarketCap',
                color='Return',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                title='Market Performance Heatmap'
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            # Update hover information
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>Return: %{color:.2f}%<br>Market Cap: $%{value:.2f}B<extra></extra>'
            )
            
            # Save as interactive HTML
            os.makedirs('outputs/heatmaps', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = f"outputs/heatmaps/market_heatmap_{timestamp}.html"
            fig.write_html(html_path, include_plotlyjs='cdn')
            
            # Create a static image version as well
            png_path = f"outputs/heatmaps/market_heatmap_{timestamp}.png"
            fig.write_image(png_path)
            
            return {
                "sector": sector,
                "market_cap_min": market_cap_min,
                "html_path": html_path,
                "png_path": png_path,
                "symbols_count": len(data)
            }
            
        except Exception as e:
            return {"error": f"Error creating market heatmap: {str(e)}"}
