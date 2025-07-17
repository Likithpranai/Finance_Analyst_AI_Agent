"""
Finance Analyst AI Agent Dashboard

This Streamlit application provides an interactive dashboard interface for the
Finance Analyst AI Agent with real-time data visualization and analysis tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our tools
from tools.interactive_visualization import InteractiveVisualizationTools
from tools.combined_analysis import CombinedAnalysisTools
from tools.fundamental_analysis import FundamentalAnalysisTools
from tools.technical_analysis import TechnicalAnalysisTools

# Page configuration
st.set_page_config(
    page_title="Finance Analyst Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .positive {
        color: #26a69a;
    }
    .negative {
        color: #ef5350;
    }
    .neutral {
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## Finance Analyst AI Dashboard")
st.sidebar.markdown("---")

# Symbol input
default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
symbols_input = st.sidebar.text_input("Enter stock symbols (comma-separated)", 
                                     value="AAPL,MSFT")
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

if not symbols:
    symbols = ["AAPL"]  # Default

# Time period selection
period_options = {
    "1 Day": "1d", 
    "5 Days": "5d", 
    "1 Month": "1mo", 
    "3 Months": "3mo", 
    "6 Months": "6mo", 
    "1 Year": "1y", 
    "2 Years": "2y", 
    "5 Years": "5y"
}
selected_period = st.sidebar.selectbox("Select time period", 
                                      list(period_options.keys()))
period = period_options[selected_period]

# Analysis type selection
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Interactive Charts", "Combined Analysis", "Market Heatmap", "Multi-Stock Dashboard"]
)

# Technical indicators selection
tech_indicators = st.sidebar.multiselect(
    "Select Technical Indicators",
    ["sma", "ema", "bollinger", "rsi", "macd", "volume_profile"],
    default=["sma", "rsi"]
)

# Main content
st.markdown('<h1 class="main-header">Finance Analyst AI Dashboard</h1>', unsafe_allow_html=True)

# Function to format numbers with appropriate suffixes
def format_large_number(num):
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    else:
        return f"${num:.2f}"

# Function to get color based on value
def get_color(val):
    if val > 0:
        return "positive"
    elif val < 0:
        return "negative"
    else:
        return "neutral"

# Function to create a metric card
def metric_card(title, value, change=None, prefix="", suffix=""):
    color_class = get_color(change) if change is not None else "neutral"
    change_html = f"<span class='{color_class}'>({change:+.2f}%)</span>" if change is not None else ""
    
    return f"""
    <div class="metric-container">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{prefix}{value}{suffix} {change_html}</div>
    </div>
    """

# Load data for the selected symbols
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_stock_data(symbol, period):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    info = ticker.info
    return data, info

# Display based on selected analysis type
if analysis_type == "Interactive Charts":
    st.markdown('<h2 class="sub-header">Interactive Stock Charts</h2>', unsafe_allow_html=True)
    
    # Create tabs for each symbol
    tabs = st.tabs(symbols)
    
    for i, symbol in enumerate(symbols):
        with tabs[i]:
            try:
                with st.spinner(f"Loading data for {symbol}..."):
                    # Get stock data
                    data, info = load_stock_data(symbol, period)
                    
                    if data.empty:
                        st.error(f"No data available for {symbol}")
                        continue
                    
                    # Display company info
                    col1, col2, col3 = st.columns(3)
                    
                    company_name = info.get('longName', symbol)
                    current_price = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    price_change = ((current_price / prev_close) - 1) * 100
                    
                    with col1:
                        st.markdown(metric_card("Current Price", f"{current_price:.2f}", price_change, "$"), 
                                   unsafe_allow_html=True)
                    
                    with col2:
                        market_cap = info.get('marketCap', 0)
                        st.markdown(metric_card("Market Cap", format_large_number(market_cap)), 
                                   unsafe_allow_html=True)
                    
                    with col3:
                        volume = data['Volume'].iloc[-1]
                        avg_volume = data['Volume'].mean()
                        volume_change = ((volume / avg_volume) - 1) * 100
                        st.markdown(metric_card("Volume", format_large_number(volume), volume_change), 
                                   unsafe_allow_html=True)
                    
                    # Create interactive chart
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                       vertical_spacing=0.03, row_heights=[0.7, 0.3],
                                       subplot_titles=[f"{company_name} Price", "Volume"])
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=symbol,
                        increasing_line_color='#26A69A', 
                        decreasing_line_color='#EF5350'
                    ), row=1, col=1)
                    
                    # Add technical indicators
                    if "sma" in tech_indicators:
                        sma20 = data['Close'].rolling(window=20).mean()
                        sma50 = data['Close'].rolling(window=50).mean()
                        sma200 = data['Close'].rolling(window=200).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=data.index, y=sma20,
                            line=dict(color='blue', width=1),
                            name='SMA20'
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index, y=sma50,
                            line=dict(color='orange', width=1),
                            name='SMA50'
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index, y=sma200,
                            line=dict(color='red', width=1),
                            name='SMA200'
                        ), row=1, col=1)
                    
                    if "bollinger" in tech_indicators:
                        sma20 = data['Close'].rolling(window=20).mean()
                        std20 = data['Close'].rolling(window=20).std()
                        upper_band = sma20 + (std20 * 2)
                        lower_band = sma20 - (std20 * 2)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index, y=upper_band,
                            line=dict(color='rgba(0,0,255,0.5)', width=1),
                            name='Upper Band',
                            fill=None
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index, y=lower_band,
                            line=dict(color='rgba(0,0,255,0.5)', width=1),
                            name='Lower Band',
                            fill='tonexty',
                            fillcolor='rgba(0,0,255,0.1)'
                        ), row=1, col=1)
                    
                    # Add volume chart
                    colors = ['#26A69A' if row['Close'] >= row['Open'] else '#EF5350' 
                             for _, row in data.iterrows()]
                    
                    fig.add_trace(go.Bar(
                        x=data.index, y=data['Volume'],
                        marker_color=colors,
                        name='Volume'
                    ), row=2, col=1)
                    
                    # Update layout
                    fig.update_layout(
                        height=600,
                        template="plotly_white",
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=50, r=50, t=50, b=50)
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
                        row=2, col=1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display key statistics
                    st.markdown('<h3 class="sub-header">Key Statistics</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        pe_ratio = info.get('trailingPE', 'N/A')
                        pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio
                        st.markdown(metric_card("P/E Ratio", pe_ratio_str), unsafe_allow_html=True)
                    
                    with col2:
                        eps = info.get('trailingEps', 'N/A')
                        eps_str = f"{eps:.2f}" if isinstance(eps, (int, float)) else eps
                        st.markdown(metric_card("EPS", eps_str, prefix="$"), unsafe_allow_html=True)
                    
                    with col3:
                        div_yield = info.get('dividendYield', 'N/A')
                        div_yield_str = f"{div_yield*100:.2f}" if isinstance(div_yield, (int, float)) else div_yield
                        st.markdown(metric_card("Dividend Yield", div_yield_str, suffix="%"), unsafe_allow_html=True)
                    
                    with col4:
                        beta = info.get('beta', 'N/A')
                        beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else beta
                        st.markdown(metric_card("Beta", beta_str), unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")

elif analysis_type == "Combined Analysis":
    st.markdown('<h2 class="sub-header">Combined Technical & Fundamental Analysis</h2>', 
               unsafe_allow_html=True)
    
    # Only analyze the first symbol for combined analysis
    symbol = symbols[0]
    
    try:
        with st.spinner(f"Generating combined analysis for {symbol}..."):
            # Get combined analysis
            analysis = CombinedAnalysisTools.create_combined_analysis(symbol, period)
            
            if "error" in analysis:
                st.error(f"Error: {analysis['error']}")
            else:
                # Display analysis summary
                st.markdown(f"### {analysis['company_name']} ({symbol})")
                
                # Display rating
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="card">
                        <h4>Overall Rating</h4>
                        <h2 class="{get_color(1 if analysis['rating'] in ['Strong Buy', 'Buy'] else -1 if analysis['rating'] in ['Strong Sell', 'Sell'] else 0)}">
                            {analysis['rating']}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="card">
                        <h4>Score</h4>
                        <p>Technical: {analysis['technical_score']}/3</p>
                        <p>Fundamental: {analysis['fundamental_score']}/3</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Current price and metrics
                    current_price = analysis['current_price']
                    price_change = analysis['price_change']
                    
                    st.markdown(f"""
                    <div class="card">
                        <h4>Current Price</h4>
                        <h2>${current_price:.2f} <span class="{get_color(price_change)}">({price_change:+.2f}%)</span></h2>
                        <p>Volatility: {analysis['volatility']:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display technical and fundamental analysis tabs
                tab1, tab2 = st.tabs(["Technical Analysis", "Fundamental Analysis"])
                
                with tab1:
                    st.markdown("### Technical Indicators")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("RSI (14)", f"{analysis['rsi']:.2f}", 
                                 delta="Overbought" if analysis['rsi'] > 70 else "Oversold" if analysis['rsi'] < 30 else "Neutral")
                    
                    with col2:
                        st.metric("MACD Signal", analysis['macd_signal'])
                    
                    with col3:
                        st.metric("Moving Averages", analysis['ma_signal'])
                    
                    # Display the technical chart
                    if 'technical_chart_path' in analysis:
                        st.image(analysis['technical_chart_path'])
                    else:
                        # Create a basic chart
                        data, _ = load_stock_data(symbol, period)
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Close Price'
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.markdown("### Fundamental Analysis")
                    
                    # Display key ratios
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pb_ratio = analysis.get('pb_ratio', 'N/A')
                        pb_str = f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else pb_ratio
                        st.metric("P/B Ratio", pb_str)
                    
                    with col2:
                        ps_ratio = analysis.get('ps_ratio', 'N/A')
                        ps_str = f"{ps_ratio:.2f}" if isinstance(ps_ratio, (int, float)) else ps_ratio
                        st.metric("P/S Ratio", ps_str)
                    
                    with col3:
                        pe_ratio = analysis.get('pe_ratio', 'N/A')
                        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio
                        st.metric("P/E Ratio", pe_str)
                    
                    # Display peer comparison if available
                    if 'peer_comparison' in analysis and analysis['peer_comparison']:
                        st.markdown("### Industry Comparison")
                        
                        peer_data = analysis['peer_comparison']
                        peer_symbols = list(peer_data.keys())
                        
                        if peer_symbols:
                            # Create comparison chart
                            ratios = ['P/E', 'P/S', 'P/B']
                            available_ratios = [r for r in ratios if any(peer_data[s].get(r.lower() + '_ratio') is not None for s in peer_symbols)]
                            
                            if available_ratios:
                                fig = go.Figure()
                                
                                for ratio in available_ratios:
                                    ratio_key = ratio.lower() + '_ratio'
                                    values = []
                                    labels = []
                                    
                                    # Add the main symbol first
                                    if ratio_key in analysis:
                                        values.append(analysis[ratio_key])
                                        labels.append(symbol)
                                    
                                    # Add peer symbols
                                    for peer in peer_symbols:
                                        if peer_data[peer].get(ratio_key) is not None:
                                            values.append(peer_data[peer][ratio_key])
                                            labels.append(peer)
                                    
                                    fig.add_trace(go.Bar(
                                        x=labels,
                                        y=values,
                                        name=ratio
                                    ))
                                
                                fig.update_layout(
                                    title="Ratio Comparison with Industry Peers",
                                    xaxis_title="Companies",
                                    yaxis_title="Ratio Value",
                                    barmode='group'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                # Display the combined visualization
                st.markdown("### Combined Analysis Visualization")
                if 'visualization_path' in analysis:
                    st.image(analysis['visualization_path'])
                
                # Display formatted analysis
                st.markdown("### Analysis Summary")
                formatted_analysis = CombinedAnalysisTools.format_combined_analysis(analysis)
                st.markdown(f"```\n{formatted_analysis}\n```")
    
    except Exception as e:
        st.error(f"Error generating combined analysis: {str(e)}")

elif analysis_type == "Market Heatmap":
    st.markdown('<h2 class="sub-header">Market Performance Heatmap</h2>', unsafe_allow_html=True)
    
    # Sector filter
    sectors = ["All Sectors", "Technology", "Healthcare", "Financial Services", 
              "Consumer Cyclical", "Communication Services", "Industrials", 
              "Consumer Defensive", "Energy", "Basic Materials", "Real Estate", "Utilities"]
    
    selected_sector = st.selectbox("Filter by Sector", sectors)
    sector_filter = None if selected_sector == "All Sectors" else selected_sector
    
    # Market cap filter
    market_cap_options = ["All", ">$500B", ">$100B", ">$50B", ">$10B", ">$1B"]
    selected_market_cap = st.selectbox("Filter by Market Cap", market_cap_options)
    
    market_cap_map = {
        "All": None,
        ">$500B": 500,
        ">$100B": 100,
        ">$50B": 50,
        ">$10B": 10,
        ">$1B": 1
    }
    market_cap_filter = market_cap_map[selected_market_cap]
    
    try:
        with st.spinner("Generating market heatmap..."):
            # Create heatmap
            heatmap = InteractiveVisualizationTools.create_market_heatmap(
                sector=sector_filter,
                market_cap_min=market_cap_filter
            )
            
            if "error" in heatmap:
                st.error(f"Error: {heatmap['error']}")
            else:
                # Display heatmap
                if 'html_path' in heatmap and os.path.exists(heatmap['html_path']):
                    # Read HTML content
                    with open(heatmap['html_path'], 'r') as f:
                        html_content = f.read()
                    
                    # Display using components.html
                    st.components.v1.html(html_content, height=800)
                elif 'png_path' in heatmap and os.path.exists(heatmap['png_path']):
                    # Display static image as fallback
                    st.image(heatmap['png_path'])
                else:
                    st.error("Heatmap visualization not available")
    
    except Exception as e:
        st.error(f"Error generating market heatmap: {str(e)}")

elif analysis_type == "Multi-Stock Dashboard":
    st.markdown('<h2 class="sub-header">Multi-Stock Dashboard</h2>', unsafe_allow_html=True)
    
    try:
        with st.spinner("Loading dashboard..."):
            # Create dashboard
            dashboard = InteractiveVisualizationTools.create_financial_dashboard(
                symbols=symbols,
                period=period
            )
            
            if "error" in dashboard:
                st.error(f"Error: {dashboard['error']}")
            else:
                # Display dashboard
                if 'html_path' in dashboard and os.path.exists(dashboard['html_path']):
                    # Read HTML content
                    with open(dashboard['html_path'], 'r') as f:
                        html_content = f.read()
                    
                    # Display using components.html
                    st.components.v1.html(html_content, height=800)
                elif 'png_path' in dashboard and os.path.exists(dashboard['png_path']):
                    # Display static image as fallback
                    st.image(dashboard['png_path'])
                else:
                    st.error("Dashboard visualization not available")
                    
                # Add performance comparison table
                st.markdown("### Performance Comparison")
                
                # Get performance data for each symbol
                performance_data = []
                
                for symbol in symbols:
                    try:
                        data, info = load_stock_data(symbol, period)
                        
                        if not data.empty:
                            start_price = data['Close'].iloc[0]
                            end_price = data['Close'].iloc[-1]
                            price_change = ((end_price / start_price) - 1) * 100
                            
                            # Calculate volatility
                            returns = data['Close'].pct_change().dropna()
                            volatility = returns.std() * np.sqrt(252) * 100
                            
                            # Get additional info
                            market_cap = info.get('marketCap', 0)
                            pe_ratio = info.get('trailingPE', None)
                            
                            performance_data.append({
                                'Symbol': symbol,
                                'Company': info.get('longName', symbol),
                                'Current Price': f"${end_price:.2f}",
                                'Change (%)': price_change,
                                'Volatility (%)': volatility,
                                'Market Cap': format_large_number(market_cap),
                                'P/E Ratio': f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else 'N/A'
                            })
                    except Exception as e:
                        st.error(f"Error processing {symbol}: {str(e)}")
                
                if performance_data:
                    df = pd.DataFrame(performance_data)
                    
                    # Style the dataframe
                    def color_change(val):
                        try:
                            val = float(val)
                            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                            return f'color: {color}'
                        except:
                            return 'color: black'
                    
                    styled_df = df.style.applymap(color_change, subset=['Change (%)'])
                    
                    st.dataframe(styled_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error generating dashboard: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Finance Analyst AI Dashboard | Data provided by Yahoo Finance")
