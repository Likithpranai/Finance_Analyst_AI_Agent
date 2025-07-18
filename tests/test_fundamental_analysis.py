"""
Test script for the Fundamental Analysis Tools
"""

import os
from dotenv import load_dotenv
from tools.fundamental_analysis import FundamentalAnalysisTools

# Load environment variables
load_dotenv()

def test_financial_ratios():
    """Test the financial ratios functionality"""
    print("\n===== TESTING FINANCIAL RATIOS =====")
    
    # Test with a well-known stock
    symbol = "AAPL"
    print(f"Getting financial ratios for {symbol}...")
    
    # Get the financial ratios
    ratios = FundamentalAnalysisTools.get_financial_ratios(symbol)
    
    # Format and display the ratios
    formatted_ratios = FundamentalAnalysisTools.format_financial_ratios_for_display(ratios)
    print(formatted_ratios)
    
    return ratios

def test_financial_statements():
    """Test the financial statements functionality"""
    print("\n===== TESTING FINANCIAL STATEMENTS =====")
    
    # Test with a well-known stock
    symbol = "MSFT"
    print(f"Getting financial statements for {symbol}...")
    
    # Get the income statement
    print("\n--- Income Statement ---")
    income_stmt = FundamentalAnalysisTools.get_income_statement(symbol)
    if "error" in income_stmt:
        print(f"Error: {income_stmt['error']}")
    else:
        print(f"Successfully retrieved income statement for {symbol}")
        print(f"Period: {income_stmt['period']}")
        print(f"Currency: {income_stmt['currency']}")
        print(f"Number of periods: {len(income_stmt['data'])}")
    
    # Get the balance sheet
    print("\n--- Balance Sheet ---")
    balance_sheet = FundamentalAnalysisTools.get_balance_sheet(symbol)
    if "error" in balance_sheet:
        print(f"Error: {balance_sheet['error']}")
    else:
        print(f"Successfully retrieved balance sheet for {symbol}")
        print(f"Period: {balance_sheet['period']}")
        print(f"Currency: {balance_sheet['currency']}")
        print(f"Number of periods: {len(balance_sheet['data'])}")
    
    # Get the cash flow statement
    print("\n--- Cash Flow Statement ---")
    cash_flow = FundamentalAnalysisTools.get_cash_flow(symbol)
    if "error" in cash_flow:
        print(f"Error: {cash_flow['error']}")
    else:
        print(f"Successfully retrieved cash flow statement for {symbol}")
        print(f"Period: {cash_flow['period']}")
        print(f"Currency: {cash_flow['currency']}")
        print(f"Number of periods: {len(cash_flow['data'])}")
    
    return {
        "income_statement": income_stmt,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow
    }

def test_industry_comparison():
    """Test the industry comparison functionality"""
    print("\n===== TESTING INDUSTRY COMPARISON =====")
    
    # Test with a well-known stock
    symbol = "GOOGL"
    print(f"Getting industry comparison for {symbol}...")
    
    # Get the industry comparison
    comparison = FundamentalAnalysisTools.get_industry_comparison(symbol)
    
    if "error" in comparison:
        print(f"Error: {comparison['error']}")
    else:
        print(f"Company: {comparison['company_name']} ({symbol})")
        print(f"Industry: {comparison['industry']}")
        print(f"Sector: {comparison['sector']}")
        print("\nRatio Comparisons:")
        
        for ratio, data in comparison["comparisons"].items():
            print(f"\n{ratio.replace('_', ' ').title()}:")
            print(f"  Company Value: {data['stock_value']:.2f}")
            print(f"  Industry Average: {data['industry_average']:.2f}")
            print(f"  Difference: {data['percentage_difference']:.2f}%")
            print(f"  Position: {data['relative_position']}")
    
    return comparison

def main():
    """Main function to run all tests"""
    print("=" * 80)
    print("FUNDAMENTAL ANALYSIS TOOLS TEST".center(80))
    print("=" * 80)
    
    # Test financial ratios
    ratios = test_financial_ratios()
    
    # Test financial statements
    statements = test_financial_statements()
    
    # Test industry comparison
    comparison = test_industry_comparison()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED".center(80))
    print("=" * 80)

if __name__ == "__main__":
    main()
