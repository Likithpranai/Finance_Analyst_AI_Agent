#!/usr/bin/env python3
"""
Test script for Alpha Vantage integration in Finance Analyst AI Agent.
This script tests all the Alpha Vantage functions to ensure they're working properly.
"""

import os
import sys
from dotenv import load_dotenv
from tools.alpha_vantage_tools import AlphaVantageTools

# Load environment variables
load_dotenv()

def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def test_real_time_quote():
    print_section("Testing Real-Time Stock Quote")
    try:
        symbol = "AAPL"
        print(f"Getting real-time quote for {symbol}...")
        result = AlphaVantageTools.get_real_time_quote(symbol)
        formatted = AlphaVantageTools.format_real_time_data_for_display(result)
        print(formatted)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_intraday_data():
    print_section("Testing Intraday Data")
    try:
        symbol = "MSFT"
        interval = "5min"
        output_size = "compact"
        print(f"Getting {interval} intraday data for {symbol}...")
        result = AlphaVantageTools.get_intraday_data(symbol, interval, output_size)
        formatted = AlphaVantageTools.format_real_time_data_for_display(result)
        print(formatted)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_crypto_data():
    print_section("Testing Cryptocurrency Data")
    try:
        symbol = "BTC"
        market = "USD"
        interval = "daily"
        print(f"Getting {interval} data for {symbol}/{market}...")
        result = AlphaVantageTools.get_crypto_data(symbol, market, interval)
        formatted = AlphaVantageTools.format_real_time_data_for_display(result)
        print(formatted)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_forex_data():
    print_section("Testing Forex Data")
    try:
        from_currency = "EUR"
        to_currency = "USD"
        interval = "daily"
        print(f"Getting {interval} data for {from_currency}/{to_currency}...")
        result = AlphaVantageTools.get_forex_data(from_currency, to_currency, interval)
        formatted = AlphaVantageTools.format_real_time_data_for_display(result)
        print(formatted)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_economic_indicators():
    print_section("Testing Economic Indicators")
    indicators = ["GDP", "INFLATION", "UNEMPLOYMENT", "RETAIL_SALES", 
                  "TREASURY_YIELD", "CONSUMER_SENTIMENT", "NONFARM_PAYROLL"]
    
    success = True
    for indicator in indicators:
        print(f"\nTesting indicator: {indicator}")
        try:
            result = AlphaVantageTools.get_economic_indicator(indicator)
            formatted = AlphaVantageTools.format_real_time_data_for_display(result)
            print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
        except Exception as e:
            print(f"Error with {indicator}: {str(e)}")
            success = False
    
    return success

def main():
    print_section("ALPHA VANTAGE API TESTS")
    
    # Check if API key exists
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("Error: ALPHA_VANTAGE_API_KEY not found in .env file")
        return False
    
    print(f"API Key found: {api_key[:4]}...{api_key[-4:]}")
    
    # Run tests
    tests = [
        ("Real-Time Stock Quote", test_real_time_quote),
        ("Intraday Data", test_intraday_data),
        ("Cryptocurrency Data", test_crypto_data),
        ("Forex Data", test_forex_data),
        ("Economic Indicators", test_economic_indicators)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        results[name] = test_func()
    
    # Print summary
    print_section("TEST RESULTS SUMMARY")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed! Alpha Vantage integration is working correctly.")
    else:
        print("\nSome tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
