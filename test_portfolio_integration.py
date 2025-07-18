"""
Test script for Portfolio Integration Tools

This script tests the functionality of the PortfolioIntegrationTools class
by running sample queries for portfolio analysis and strategy backtesting.
"""

import os
import sys
import json
from dotenv import load_dotenv
from tools.portfolio_integration import PortfolioIntegrationTools

# Load environment variables
load_dotenv()

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_result(result):
    """Pretty print a result dictionary"""
    if isinstance(result, dict):
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return
            
        print("SUCCESS: True" if result.get("success", False) else "SUCCESS: False")
        
        # Print non-visualization keys
        for key, value in result.items():
            if key != "visualizations" and key != "success":
                if isinstance(value, dict):
                    print(f"\n{key.upper()}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
        
        # Print visualization paths if available
        if "visualizations" in result:
            print("\nVISUALIZATIONS:")
            for viz in result["visualizations"]:
                print(f"  - {viz}")
    else:
        print(result)

def test_portfolio_analysis():
    """Test portfolio analysis functionality"""
    print_section("Testing Portfolio Analysis")
    
    # Test case 1: Basic portfolio risk metrics
    query1 = "Calculate risk metrics for a portfolio with AAPL, MSFT, GOOGL with equal weights"
    print(f"QUERY: {query1}")
    result1 = PortfolioIntegrationTools.analyze_portfolio(query1)
    print_result(result1)
    
    # Test case 2: Portfolio optimization
    query2 = "Optimize a portfolio with AAPL, MSFT, GOOGL, AMZN, TSLA for maximum Sharpe ratio"
    print(f"\nQUERY: {query2}")
    result2 = PortfolioIntegrationTools.analyze_portfolio(query2)
    print_result(result2)
    
    # Test case 3: Efficient frontier
    query3 = "Generate efficient frontier for portfolio with AAPL, MSFT, GOOGL using 100 portfolios"
    print(f"\nQUERY: {query3}")
    result3 = PortfolioIntegrationTools.analyze_portfolio(query3)
    print_result(result3)
    
    # Test case 4: Portfolio with specific weights
    query4 = "Calculate risk metrics for portfolio with AAPL, MSFT, GOOGL with weights 0.5, 0.3, 0.2"
    print(f"\nQUERY: {query4}")
    result4 = PortfolioIntegrationTools.analyze_portfolio(query4)
    print_result(result4)

def test_backtest_strategy():
    """Test strategy backtesting functionality"""
    print_section("Testing Strategy Backtesting")
    
    # Test case 1: SMA Crossover strategy
    query1 = "Backtest SMA crossover strategy for AAPL with short window 50 and long window 200"
    print(f"QUERY: {query1}")
    result1 = PortfolioIntegrationTools.backtest_strategy(query1)
    print_result(result1)
    
    # Test case 2: RSI strategy
    query2 = "Backtest RSI strategy for TSLA with RSI period 14, overbought 70, oversold 30"
    print(f"\nQUERY: {query2}")
    result2 = PortfolioIntegrationTools.backtest_strategy(query2)
    print_result(result2)
    
    # Test case 3: MACD strategy
    query3 = "Backtest MACD strategy for the stock MSFT with fast period 12, slow period 26, signal period 9"
    print(f"\nQUERY: {query3}")
    result3 = PortfolioIntegrationTools.backtest_strategy(query3)
    print_result(result3)
    
    # Test case 4: Strategy with initial capital
    query4 = "Backtest SMA crossover strategy for GOOGL with initial capital $50000"
    print(f"\nQUERY: {query4}")
    result4 = PortfolioIntegrationTools.backtest_strategy(query4)
    print_result(result4)

def main():
    """Main test function"""
    print_section("Portfolio Integration Tools Test")
    
    try:
        # Test portfolio analysis
        test_portfolio_analysis()
        
        # Test strategy backtesting
        test_backtest_strategy()
        
        print_section("All Tests Completed")
        
    except Exception as e:
        import traceback
        print(f"Error during testing: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
