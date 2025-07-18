"""
Test script specifically for fundamental analysis ratios
This script focuses on testing the financial ratio calculations we added
"""

import os
from dotenv import load_dotenv
import yfinance as yf
from tools.fundamental_analysis import FundamentalAnalysisTools

# Load environment variables
load_dotenv()

def test_financial_ratios_detailed(symbol):
    """Test the financial ratios functionality with detailed output"""
    print("\n" + "=" * 80)
    print(f"FUNDAMENTAL ANALYSIS FOR {symbol}".center(80))
    print("=" * 80)
    
    # Get ticker info directly first
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    print(f"\nCompany: {info.get('longName', 'N/A')} ({symbol})")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")
    
    # Get raw data for key metrics
    market_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
    eps = info.get('trailingEPS', 'N/A')
    book_value = info.get('bookValue', 'N/A')
    revenue_per_share = info.get('revenuePerShare', 'N/A')
    
    print("\n--- Raw Financial Data ---")
    print(f"Current Price: ${market_price if market_price != 'N/A' else 'N/A'}")
    print(f"EPS (Trailing): ${eps if eps != 'N/A' else 'N/A'}")
    print(f"Book Value per Share: ${book_value if book_value != 'N/A' else 'N/A'}")
    print(f"Revenue per Share: ${revenue_per_share if revenue_per_share != 'N/A' else 'N/A'}")
    
    # Calculate basic ratios manually for verification
    print("\n--- Manual Ratio Calculations ---")
    
    # P/E Ratio
    if market_price != 'N/A' and eps != 'N/A' and eps != 0:
        pe_ratio = market_price / eps
        print(f"P/E Ratio: {pe_ratio:.2f}")
    else:
        print("P/E Ratio: N/A (insufficient data or negative earnings)")
    
    # P/B Ratio
    if market_price != 'N/A' and book_value != 'N/A' and book_value != 0:
        pb_ratio = market_price / book_value
        print(f"P/B Ratio: {pb_ratio:.2f}")
    else:
        print("P/B Ratio: N/A (insufficient data)")
    
    # P/S Ratio
    if market_price != 'N/A' and revenue_per_share != 'N/A' and revenue_per_share != 0:
        ps_ratio = market_price / revenue_per_share
        print(f"P/S Ratio: {ps_ratio:.2f}")
    else:
        print("P/S Ratio: N/A (insufficient data)")
    
    # Now use our FundamentalAnalysisTools class
    print("\n--- Using FundamentalAnalysisTools ---")
    ratios = FundamentalAnalysisTools.get_financial_ratios(symbol)
    
    if "error" in ratios:
        print(f"Error: {ratios['error']}")
        return
    
    # Print formatted ratios
    formatted_ratios = FundamentalAnalysisTools.format_financial_ratios_for_display(ratios)
    print(formatted_ratios)
    
    # Get financial statements
    print("\n--- Financial Statements ---")
    
    # Income Statement
    income_stmt = FundamentalAnalysisTools.get_income_statement(symbol)
    if "error" in income_stmt:
        print(f"Income Statement Error: {income_stmt['error']}")
    else:
        print("Income Statement: Available")
        # Get the latest period
        latest_period = list(income_stmt['data'].keys())[0] if income_stmt['data'] else None
        if latest_period:
            print(f"Latest Period: {latest_period}")
            # Print key metrics from income statement
            try:
                total_revenue = income_stmt['data'].get('Total Revenue', {}).get(latest_period, 'N/A')
                net_income = income_stmt['data'].get('Net Income', {}).get(latest_period, 'N/A')
                
                print(f"Total Revenue: ${total_revenue/1e9:.2f}B" if total_revenue != 'N/A' else "Total Revenue: N/A")
                print(f"Net Income: ${net_income/1e9:.2f}B" if net_income != 'N/A' else "Net Income: N/A")
            except:
                print("Could not extract key metrics from income statement")
    
    # Balance Sheet
    balance_sheet = FundamentalAnalysisTools.get_balance_sheet(symbol)
    if "error" in balance_sheet:
        print(f"Balance Sheet Error: {balance_sheet['error']}")
    else:
        print("\nBalance Sheet: Available")
        # Get the latest period
        latest_period = list(balance_sheet['data'].keys())[0] if balance_sheet['data'] else None
        if latest_period:
            print(f"Latest Period: {latest_period}")
            # Print key metrics from balance sheet
            try:
                total_assets = balance_sheet['data'].get('Total Assets', {}).get(latest_period, 'N/A')
                total_liabilities = balance_sheet['data'].get('Total Liabilities Net Minority Interest', {}).get(latest_period, 'N/A')
                total_equity = balance_sheet['data'].get('Total Equity Gross Minority Interest', {}).get(latest_period, 'N/A')
                
                print(f"Total Assets: ${total_assets/1e9:.2f}B" if total_assets != 'N/A' else "Total Assets: N/A")
                print(f"Total Liabilities: ${total_liabilities/1e9:.2f}B" if total_liabilities != 'N/A' else "Total Liabilities: N/A")
                print(f"Total Equity: ${total_equity/1e9:.2f}B" if total_equity != 'N/A' else "Total Equity: N/A")
            except:
                print("Could not extract key metrics from balance sheet")
    
    # Cash Flow
    cash_flow = FundamentalAnalysisTools.get_cash_flow(symbol)
    if "error" in cash_flow:
        print(f"Cash Flow Error: {cash_flow['error']}")
    else:
        print("\nCash Flow Statement: Available")
        # Get the latest period
        latest_period = list(cash_flow['data'].keys())[0] if cash_flow['data'] else None
        if latest_period:
            print(f"Latest Period: {latest_period}")
            # Print key metrics from cash flow statement
            try:
                operating_cash_flow = cash_flow['data'].get('Operating Cash Flow', {}).get(latest_period, 'N/A')
                free_cash_flow = cash_flow['data'].get('Free Cash Flow', {}).get(latest_period, 'N/A')
                
                print(f"Operating Cash Flow: ${operating_cash_flow/1e9:.2f}B" if operating_cash_flow != 'N/A' else "Operating Cash Flow: N/A")
                print(f"Free Cash Flow: ${free_cash_flow/1e9:.2f}B" if free_cash_flow != 'N/A' else "Free Cash Flow: N/A")
            except:
                print("Could not extract key metrics from cash flow statement")
    
    # Industry comparison
    print("\n--- Industry Comparison ---")
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

def main():
    """Main function to run the test"""
    print("=" * 80)
    print("FUNDAMENTAL ANALYSIS RATIO TEST".center(80))
    print("=" * 80)
    
    # Test with different stocks to demonstrate versatility
    stocks = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in stocks:
        test_financial_ratios_detailed(symbol)

if __name__ == "__main__":
    main()
