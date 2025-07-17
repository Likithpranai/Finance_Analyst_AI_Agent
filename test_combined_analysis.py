"""
Test script for combined fundamental and technical analysis
This demonstrates how the Finance Analyst AI Agent can use both types of analysis together
using the new enhanced visualization and combined analysis tools.
"""

import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tools.fundamental_analysis import FundamentalAnalysisTools
from tools.alpha_vantage_tools import AlphaVantageTools
from tools.enhanced_visualization import EnhancedVisualizationTools
from tools.combined_analysis import CombinedAnalysisTools

# Load environment variables
load_dotenv()

def perform_combined_analysis(symbol):
    """
    Perform a combined fundamental and technical analysis for a stock
    using the new CombinedAnalysisTools module
    """
    print("=" * 80)
    print(f"COMBINED ANALYSIS FOR {symbol}".center(80))
    print("=" * 80)
    
    # REASON: Determine what information we need for a comprehensive analysis
    print("\n[REASON] Planning analysis approach...")
    print("1. We need current price and technical indicators for short-term analysis")
    print("2. We need fundamental ratios for long-term value assessment")
    print("3. We need to combine both for a holistic view")
    
    # ACT: Use the combined analysis tool
    print("\n[ACT] Generating combined analysis...")
    try:
        # Create combined analysis
        analysis = CombinedAnalysisTools.create_combined_analysis(symbol, period="1y")
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
            
        # Format the analysis for display
        formatted_analysis = CombinedAnalysisTools.format_combined_analysis(analysis)
        
        # Print the formatted analysis
        print(formatted_analysis)
        
        # Print the path to the visualization
        print(f"\nVisualization saved to: {analysis['visualization_path']}")
        
    except Exception as e:
        print(f"Error in combined analysis: {str(e)}")

def test_enhanced_visualization(symbol):
    """
    Test the enhanced visualization tools
    """
    print("=" * 80)
    print(f"ENHANCED VISUALIZATION FOR {symbol}".center(80))
    print("=" * 80)
    
    # Test financial trends visualization
    print("\n[TEST] Creating financial trends visualization...")
    try:
        trends_result = EnhancedVisualizationTools.visualize_financial_trends(
            symbol, 
            period="1y", 
            metrics=["price", "volume", "rsi", "macd"]
        )
        
        if "error" in trends_result:
            print(f"Error in financial trends visualization: {trends_result['error']}")
        else:
            print(f"Financial trends visualization created successfully!")
            print(f"Visualization saved to: {trends_result['png_path']}")
    except Exception as e:
        print(f"Error in financial trends visualization: {str(e)}")
    
    # Test financial ratio visualization
    print("\n[TEST] Creating financial ratio visualization...")
    try:
        # Get some peer symbols based on sector
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # Find a few peers
        peer_symbols = []
        if 'symbol' in info and len(info.get('symbol', '')) > 0:
            if symbol == "AAPL":
                peer_symbols = ["MSFT", "GOOGL", "AMZN"]
            elif symbol == "MSFT":
                peer_symbols = ["AAPL", "GOOGL", "AMZN"]
            else:
                # Try to get peers from info
                peers = info.get('peerSet', [])
                if peers and len(peers) > 0:
                    peer_symbols = list(peers)[:3]
        
        if peer_symbols:
            ratios_result = EnhancedVisualizationTools.visualize_financial_ratios(
                symbol,
                comparison_symbols=peer_symbols
            )
            
            if "error" in ratios_result:
                print(f"Error in financial ratio visualization: {ratios_result['error']}")
            else:
                print(f"Financial ratio visualization created successfully!")
                print(f"Visualization saved to: {ratios_result['png_path']}")
        else:
            print(f"No peer symbols found for {symbol}")
    except Exception as e:
        print(f"Error in financial ratio visualization: {str(e)}")

def main():
    """Run the test script"""
    symbols = ["AAPL", "MSFT"]
    
    # Test combined analysis
    for symbol in symbols:
        perform_combined_analysis(symbol)
        print("\n" + "-" * 80 + "\n")
    
    # Test enhanced visualization
    for symbol in symbols:
        test_enhanced_visualization(symbol)
        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()
