"""
Test script for interactive visualization and dashboard features

This script demonstrates the new interactive visualization capabilities
of the Finance Analyst AI Agent including:
- Interactive charts with customizable indicators
- Market heatmaps
- Multi-stock dashboards
"""

import os
import sys
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our tools
from tools.interactive_visualization import InteractiveVisualizationTools

def test_interactive_chart():
    """Test the interactive chart creation with various indicators"""
    print("\n" + "="*80)
    print("TESTING INTERACTIVE CHART CREATION".center(80))
    print("="*80 + "\n")
    
    # Test with a single symbol
    symbol = "AAPL"
    print(f"Creating interactive chart for {symbol}...")
    
    try:
        # Create chart with default indicators
        result = InteractiveVisualizationTools.create_interactive_chart(
            symbol=symbol,
            period="1y",
            indicators=["sma", "ema", "rsi", "macd"]
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Interactive chart created successfully!")
            print(f"HTML chart saved to: {result['html_path']}")
            print(f"PNG version saved to: {result['png_path']}")
            print(f"Current price: ${result['current_price']:.2f}")
            print(f"Price change: {result['price_change']:.2f}%")
    except Exception as e:
        print(f"Error creating interactive chart: {str(e)}")
    
    # Test with comparison symbols
    print("\nCreating interactive chart with peer comparison...")
    
    try:
        # Create chart with comparison symbols
        result = InteractiveVisualizationTools.create_interactive_chart(
            symbol=symbol,
            period="1y",
            indicators=["sma", "bollinger"],
            comparison_symbols=["MSFT", "GOOGL"]
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Interactive comparison chart created successfully!")
            print(f"HTML chart saved to: {result['html_path']}")
    except Exception as e:
        print(f"Error creating comparison chart: {str(e)}")

def test_market_heatmap():
    """Test the market heatmap creation"""
    print("\n" + "="*80)
    print("TESTING MARKET HEATMAP CREATION".center(80))
    print("="*80 + "\n")
    
    print("Creating market heatmap...")
    
    try:
        # Create market heatmap
        result = InteractiveVisualizationTools.create_market_heatmap()
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Market heatmap created successfully!")
            print(f"HTML heatmap saved to: {result['html_path']}")
            print(f"PNG version saved to: {result['png_path']}")
            print(f"Number of symbols included: {result['symbols_count']}")
    except Exception as e:
        print(f"Error creating market heatmap: {str(e)}")
    
    # Test with sector filter
    print("\nCreating technology sector heatmap...")
    
    try:
        # Create sector-specific heatmap
        result = InteractiveVisualizationTools.create_market_heatmap(
            sector="Technology",
            market_cap_min=100  # Only companies >$100B
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Technology sector heatmap created successfully!")
            print(f"HTML heatmap saved to: {result['html_path']}")
            print(f"Number of symbols included: {result['symbols_count']}")
    except Exception as e:
        print(f"Error creating sector heatmap: {str(e)}")

def test_financial_dashboard():
    """Test the multi-stock dashboard creation"""
    print("\n" + "="*80)
    print("TESTING FINANCIAL DASHBOARD CREATION".center(80))
    print("="*80 + "\n")
    
    # Test with multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    print(f"Creating financial dashboard for {', '.join(symbols)}...")
    
    try:
        # Create financial dashboard
        result = InteractiveVisualizationTools.create_financial_dashboard(
            symbols=symbols,
            period="6mo"
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Financial dashboard created successfully!")
            print(f"HTML dashboard saved to: {result['html_path']}")
            print(f"PNG version saved to: {result['png_path']}")
    except Exception as e:
        print(f"Error creating financial dashboard: {str(e)}")

def test_streamlit_dashboard():
    """Provide instructions for running the Streamlit dashboard"""
    print("\n" + "="*80)
    print("STREAMLIT DASHBOARD INSTRUCTIONS".center(80))
    print("="*80 + "\n")
    
    print("To run the Streamlit dashboard, execute the following command:")
    print("\nstreamlit run dashboard.py\n")
    print("This will start a local web server and open the dashboard in your browser.")
    print("You can then interact with the dashboard to explore different visualizations.")

def main():
    """Main function to run all tests"""
    print("\n" + "="*80)
    print("INTERACTIVE VISUALIZATION TESTS".center(80))
    print("="*80 + "\n")
    
    test_interactive_chart()
    test_market_heatmap()
    test_financial_dashboard()
    test_streamlit_dashboard()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED".center(80))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
