"""
Enhanced test script for fundamental analysis tools
This script demonstrates the improved financial statement formatting and comprehensive analysis
"""

import os
from dotenv import load_dotenv
import yfinance as yf
from tools.fundamental_analysis import FundamentalAnalysisTools

# Load environment variables
load_dotenv()

def test_comprehensive_analysis(symbol):
    """Test comprehensive fundamental analysis for a stock"""
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE FUNDAMENTAL ANALYSIS FOR {symbol}".center(80))
    print("=" * 80)
    
    # Get ticker info
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    print(f"\nCompany: {info.get('longName', symbol)} ({symbol})")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")
    print(f"Current Price: ${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}")
    
    # Get financial ratios
    print("\n" + "-" * 80)
    print("FINANCIAL RATIOS ANALYSIS".center(80))
    print("-" * 80)
    
    ratios = FundamentalAnalysisTools.get_financial_ratios(symbol)
    if "error" in ratios:
        print(f"Error retrieving financial ratios: {ratios['error']}")
    else:
        formatted_ratios = FundamentalAnalysisTools.format_financial_ratios_for_display(ratios)
        print(formatted_ratios)
    
    # Get income statement
    print("\n" + "-" * 80)
    print("INCOME STATEMENT ANALYSIS".center(80))
    print("-" * 80)
    
    income_stmt = FundamentalAnalysisTools.get_income_statement(symbol)
    if "error" in income_stmt:
        print(f"Error retrieving income statement: {income_stmt['error']}")
    else:
        formatted_income = FundamentalAnalysisTools.format_financial_statement(income_stmt, "income_statement")
        print(formatted_income)
    
    # Get balance sheet
    print("\n" + "-" * 80)
    print("BALANCE SHEET ANALYSIS".center(80))
    print("-" * 80)
    
    balance_sheet = FundamentalAnalysisTools.get_balance_sheet(symbol)
    if "error" in balance_sheet:
        print(f"Error retrieving balance sheet: {balance_sheet['error']}")
    else:
        formatted_balance = FundamentalAnalysisTools.format_financial_statement(balance_sheet, "balance_sheet")
        print(formatted_balance)
    
    # Get cash flow statement
    print("\n" + "-" * 80)
    print("CASH FLOW ANALYSIS".center(80))
    print("-" * 80)
    
    cash_flow = FundamentalAnalysisTools.get_cash_flow(symbol)
    if "error" in cash_flow:
        print(f"Error retrieving cash flow statement: {cash_flow['error']}")
    else:
        formatted_cash_flow = FundamentalAnalysisTools.format_financial_statement(cash_flow, "cash_flow")
        print(formatted_cash_flow)
    
    # Get industry comparison
    print("\n" + "-" * 80)
    print("INDUSTRY COMPARISON".center(80))
    print("-" * 80)
    
    comparison = FundamentalAnalysisTools.get_industry_comparison(symbol)
    if "error" in comparison:
        print(f"Error retrieving industry comparison: {comparison['error']}")
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
    
    # Provide a comprehensive analysis summary
    print("\n" + "=" * 80)
    print("FUNDAMENTAL ANALYSIS SUMMARY".center(80))
    print("=" * 80)
    
    print("\nThis comprehensive fundamental analysis provides:")
    print("1. Key financial ratios showing valuation metrics and profitability")
    print("2. Income statement analysis showing revenue and profit trends")
    print("3. Balance sheet analysis showing assets, liabilities, and equity")
    print("4. Cash flow analysis showing operational, investing, and financing activities")
    print("5. Industry comparison to benchmark performance against peers")
    
    print("\nThese tools enable the Finance Analyst AI Agent to provide in-depth")
    print("fundamental analysis alongside technical indicators for comprehensive")
    print("stock evaluation and investment decision support.")

def main():
    """Main function to run the test"""
    print("=" * 80)
    print("ENHANCED FUNDAMENTAL ANALYSIS TEST".center(80))
    print("=" * 80)
    
    # Test with a few major stocks
    stocks = ["AAPL", "MSFT"]
    
    for symbol in stocks:
        test_comprehensive_analysis(symbol)

if __name__ == "__main__":
    main()
