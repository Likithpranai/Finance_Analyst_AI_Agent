#!/usr/bin/env python3
"""
Test script to retrieve fundamental data for a stock symbol
"""

import os
import yfinance as yf
import pandas as pd
from pprint import pprint

def get_financial_ratios(symbol):
    """
    Get key financial ratios for a stock
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with financial ratios
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract key financial data
        market_price = info.get('currentPrice', info.get('regularMarketPrice', None))
        if not market_price:
            return {"error": f"Could not retrieve current price for {symbol}"}
        
        # Get earnings per share
        eps = info.get('trailingEPS', None)
        forward_eps = info.get('forwardEPS', None)
        
        # Get book value per share
        book_value = info.get('bookValue', None)
        
        # Get revenue per share
        revenue_per_share = info.get('revenuePerShare', None)
        
        # Get total debt and shareholders' equity
        total_debt = info.get('totalDebt', None)
        shareholders_equity = info.get('totalStockholderEquity', None)
        
        # Get net income and earnings growth
        net_income = info.get('netIncomeToCommon', None)
        earnings_growth = info.get('earningsGrowth', None)
        
        # Calculate key ratios
        pe_ratio = market_price / eps if eps and eps != 0 else None
        pb_ratio = market_price / book_value if book_value and book_value != 0 else None
        ps_ratio = market_price / revenue_per_share if revenue_per_share and revenue_per_share != 0 else None
        debt_to_equity = total_debt / shareholders_equity if shareholders_equity and shareholders_equity != 0 and total_debt else None
        
        # Get additional ratios directly from info
        forward_pe = info.get('forwardPE', None)
        peg_ratio = info.get('pegRatio', None)
        profit_margins = info.get('profitMargins', None)
        return_on_equity = info.get('returnOnEquity', None)
        return_on_assets = info.get('returnOnAssets', None)
        
        # Format the results
        ratios = {
            "symbol": symbol,
            "current_price": market_price,
            "pe_ratio": pe_ratio,
            "forward_pe": forward_pe,
            "peg_ratio": peg_ratio,
            "pb_ratio": pb_ratio,
            "ps_ratio": ps_ratio,
            "debt_to_equity": debt_to_equity,
            "profit_margins": profit_margins,
            "return_on_equity": return_on_equity,
            "return_on_assets": return_on_assets,
            "earnings_per_share": eps,
            "forward_eps": forward_eps,
            "earnings_growth": earnings_growth,
            "book_value": book_value,
            "revenue_per_share": revenue_per_share
        }
        
        return ratios
    
    except Exception as e:
        return {"error": f"Error retrieving financial ratios for {symbol}: {str(e)}"}

def get_income_statement(symbol, periods=4):
    """
    Get income statement data for a stock
    
    Args:
        symbol: Stock ticker symbol
        periods: Number of periods to retrieve
        
    Returns:
        DataFrame with income statement data
    """
    try:
        ticker = yf.Ticker(symbol)
        income_stmt = ticker.income_stmt
        
        if income_stmt.empty:
            return {"error": f"Could not retrieve income statement for {symbol}"}
        
        # Get the last N periods
        income_stmt = income_stmt.iloc[:, :periods]
        
        # Format the results
        result = {
            "symbol": symbol,
            "income_statement": income_stmt.to_dict()
        }
        
        return result
    
    except Exception as e:
        return {"error": f"Error retrieving income statement for {symbol}: {str(e)}"}

def get_balance_sheet(symbol, periods=4):
    """
    Get balance sheet data for a stock
    
    Args:
        symbol: Stock ticker symbol
        periods: Number of periods to retrieve
        
    Returns:
        DataFrame with balance sheet data
    """
    try:
        ticker = yf.Ticker(symbol)
        balance_sheet = ticker.balance_sheet
        
        if balance_sheet.empty:
            return {"error": f"Could not retrieve balance sheet for {symbol}"}
        
        # Get the last N periods
        balance_sheet = balance_sheet.iloc[:, :periods]
        
        # Format the results
        result = {
            "symbol": symbol,
            "balance_sheet": balance_sheet.to_dict()
        }
        
        return result
    
    except Exception as e:
        return {"error": f"Error retrieving balance sheet for {symbol}: {str(e)}"}

def main():
    """Main function to test fundamental data retrieval"""
    symbol = "MSFT"
    
    print(f"\n=== Financial Ratios for {symbol} ===")
    ratios = get_financial_ratios(symbol)
    pprint(ratios)
    
    print(f"\n=== Income Statement for {symbol} ===")
    income_stmt = get_income_statement(symbol)
    if "error" not in income_stmt:
        print(f"Retrieved {len(income_stmt['income_statement'])} income statement items")
    else:
        print(income_stmt["error"])
    
    print(f"\n=== Balance Sheet for {symbol} ===")
    balance_sheet = get_balance_sheet(symbol)
    if "error" not in balance_sheet:
        print(f"Retrieved {len(balance_sheet['balance_sheet'])} balance sheet items")
    else:
        print(balance_sheet["error"])

if __name__ == "__main__":
    main()
