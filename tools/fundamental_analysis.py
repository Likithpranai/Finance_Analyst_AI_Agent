"""
Fundamental Analysis Tools for the Finance Analyst AI Agent.
Includes financial ratios, valuation metrics, and company financial data analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from langchain.tools import BaseTool
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

from config import FMP_API_KEY, ALPHA_VANTAGE_API_KEY

# Helper functions for ratio calculations
def calculate_ratios(ticker: str) -> Dict[str, Any]:
    """Calculate key financial ratios for a given stock."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial data
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        if income_stmt.empty or balance_sheet.empty:
            return {"error": f"Could not fetch financial data for {ticker}"}
        
        # Get current market data
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
        market_cap = info.get('marketCap', None)
        shares_outstanding = info.get('sharesOutstanding', None)
        
        # Calculate ratios
        ratios = {}
        
        # P/E Ratio - Price to Earnings
        if 'Net Income' in income_stmt.index and shares_outstanding and current_price:
            latest_net_income = income_stmt.loc['Net Income'].iloc[0]
            eps = latest_net_income / shares_outstanding
            pe_ratio = current_price / eps if eps > 0 else None
            ratios["PE_Ratio"] = pe_ratio
            
        # Price to Sales (P/S)
        if 'Total Revenue' in income_stmt.index and market_cap:
            latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
            ps_ratio = market_cap / latest_revenue if latest_revenue > 0 else None
            ratios["PS_Ratio"] = ps_ratio
            
        # Price to Book (P/B)
        if 'Total Stockholder Equity' in balance_sheet.index and market_cap:
            latest_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            pb_ratio = market_cap / latest_equity if latest_equity > 0 else None
            ratios["PB_Ratio"] = pb_ratio
            
        # Debt to Equity (D/E)
        if 'Total Stockholder Equity' in balance_sheet.index and 'Total Liabilities Net Minority Interest' in balance_sheet.index:
            latest_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            latest_debt = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
            de_ratio = latest_debt / latest_equity if latest_equity > 0 else None
            ratios["DE_Ratio"] = de_ratio
            
        # Return on Equity (ROE)
        if 'Net Income' in income_stmt.index and 'Total Stockholder Equity' in balance_sheet.index:
            latest_net_income = income_stmt.loc['Net Income'].iloc[0]
            latest_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            roe = latest_net_income / latest_equity if latest_equity > 0 else None
            ratios["ROE"] = roe
            
        # PEG Ratio - need growth rate estimate
        # Try to get analysts' growth rate estimate
        growth_rate = info.get('earningsGrowth', info.get('revenueGrowth', None))
        if growth_rate and 'PE_Ratio' in ratios and ratios['PE_Ratio'] and growth_rate > 0:
            peg_ratio = ratios['PE_Ratio'] / (growth_rate * 100)  # Convert growth to percentage
            ratios["PEG_Ratio"] = peg_ratio
            
        # Current Ratio
        if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
            current_assets = balance_sheet.loc['Current Assets'].iloc[0]
            current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0]
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else None
            ratios["Current_Ratio"] = current_ratio
            
        # Dividend Yield
        dividend_yield = info.get('dividendYield', None)
        if dividend_yield:
            ratios["Dividend_Yield"] = dividend_yield
            
        # Format values for better readability
        formatted_ratios = {}
        for k, v in ratios.items():
            if v is not None:
                formatted_ratios[k] = round(v, 3)
            else:
                formatted_ratios[k] = None
                
        # Add interpretation guidelines
        interpretations = {
            "PE_Ratio": interpret_pe_ratio(formatted_ratios.get("PE_Ratio"), ticker),
            "PS_Ratio": interpret_ps_ratio(formatted_ratios.get("PS_Ratio"), ticker),
            "PB_Ratio": interpret_pb_ratio(formatted_ratios.get("PB_Ratio")),
            "DE_Ratio": interpret_de_ratio(formatted_ratios.get("DE_Ratio")),
            "ROE": interpret_roe(formatted_ratios.get("ROE")),
            "PEG_Ratio": interpret_peg_ratio(formatted_ratios.get("PEG_Ratio")),
            "Current_Ratio": interpret_current_ratio(formatted_ratios.get("Current_Ratio")),
            "Dividend_Yield": interpret_dividend_yield(formatted_ratios.get("Dividend_Yield"), ticker)
        }
        
        return {
            "ticker": ticker,
            "ratios": formatted_ratios,
            "interpretations": interpretations,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": f"Error calculating ratios for {ticker}: {str(e)}"}

# Interpretation helper functions
def interpret_pe_ratio(pe_ratio, ticker):
    if pe_ratio is None:
        return "No data available or negative earnings"
        
    if pe_ratio < 0:
        return "Negative P/E indicates the company is not profitable"
    elif pe_ratio < 10:
        return "Low P/E ratio might indicate an undervalued stock or market concerns"
    elif 10 <= pe_ratio <= 20:
        return "Moderate P/E ratio, generally considered reasonable valuation"
    elif 20 < pe_ratio <= 30:
        return "Above average P/E ratio, suggesting high growth expectations"
    else:
        return "Very high P/E ratio, indicating high growth expectations or potential overvaluation"

def interpret_ps_ratio(ps_ratio, ticker):
    if ps_ratio is None:
        return "No data available"
        
    if ps_ratio < 1:
        return "Low P/S ratio, potentially undervalued relative to sales"
    elif 1 <= ps_ratio <= 3:
        return "Moderate P/S ratio, generally considered reasonable"
    else:
        return "High P/S ratio, indicating high price premium relative to sales"

def interpret_pb_ratio(pb_ratio):
    if pb_ratio is None:
        return "No data available"
        
    if pb_ratio < 1:
        return "P/B ratio below 1 suggests potential undervaluation relative to book value"
    elif 1 <= pb_ratio <= 3:
        return "Moderate P/B ratio, generally considered reasonable"
    else:
        return "High P/B ratio, indicating stock trades at premium to book value"

def interpret_de_ratio(de_ratio):
    if de_ratio is None:
        return "No data available"
        
    if de_ratio < 0.5:
        return "Low debt-to-equity ratio, indicating conservative financing"
    elif 0.5 <= de_ratio <= 1.5:
        return "Moderate debt-to-equity ratio, balanced financing"
    else:
        return "High debt-to-equity ratio, potentially higher financial risk"

def interpret_roe(roe):
    if roe is None:
        return "No data available"
        
    if roe < 0:
        return "Negative ROE indicates the company is not generating returns from shareholder equity"
    elif 0 <= roe <= 0.10:
        return "Low ROE, generating modest returns on shareholder equity"
    elif 0.10 < roe <= 0.20:
        return "Good ROE, efficiently generating returns on shareholder equity"
    else:
        return "Excellent ROE, very efficiently generating returns on shareholder equity"

def interpret_peg_ratio(peg_ratio):
    if peg_ratio is None:
        return "No data available"
        
    if peg_ratio < 1:
        return "PEG ratio below 1 suggests the stock may be undervalued relative to growth"
    elif 1 <= peg_ratio <= 2:
        return "Moderate PEG ratio, generally considered reasonable valuation relative to growth"
    else:
        return "High PEG ratio, potentially overvalued relative to growth"

def interpret_current_ratio(current_ratio):
    if current_ratio is None:
        return "No data available"
        
    if current_ratio < 1:
        return "Current ratio below 1 indicates potential liquidity issues"
    elif 1 <= current_ratio <= 2:
        return "Healthy current ratio, good short-term liquidity"
    else:
        return "Very high current ratio, excellent liquidity but potentially inefficient asset utilization"

def interpret_dividend_yield(dividend_yield, ticker):
    if dividend_yield is None:
        return "No dividend data available or company does not pay dividends"
        
    if dividend_yield < 0.01:
        return "Very low dividend yield, focus is likely on growth"
    elif 0.01 <= dividend_yield <= 0.03:
        return "Moderate dividend yield"
    else:
        return "High dividend yield, may indicate income-focused stock or potential concerns about sustainability"

class FinancialRatiosTool(BaseTool):
    name: str = "financial_ratios_analyzer"
    description: str = """
    Calculate and analyze key financial ratios for a given stock ticker. 
    Provides metrics such as P/E ratio, P/S ratio, P/B ratio, D/E ratio, ROE, PEG ratio, 
    current ratio, and dividend yield, along with interpretations.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        A dictionary with financial ratios, their interpretations, and timestamp.
    """
    
    def _run(self, ticker: str) -> Dict[str, Any]:
        return calculate_ratios(ticker)
        
    async def _arun(self, ticker: str) -> Dict[str, Any]:
        # Async implementation if needed
        return calculate_ratios(ticker)

class CompetitiveAnalysisTool(BaseTool):
    name: str = "competitive_analysis"
    description: str = """
    Perform competitive analysis by comparing financial metrics of a company with its peers or sector averages.
    Compares key metrics like P/E ratio, ROE, revenue growth, and profit margins.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        peers: Optional comma-separated list of peer company tickers (e.g., 'AAPL,MSFT,GOOGL')
        
    Returns:
        Comparative analysis of the company against its peers or sector.
    """
    
    def _run(self, ticker: str, peers: str = None) -> Dict[str, Any]:
        try:
            # Get the company's sector if peers not specified
            if not peers:
                stock = yf.Ticker(ticker)
                sector = stock.info.get('sector', None)
                if sector:
                    # Get companies in the same sector
                    if FMP_API_KEY:
                        url = f"https://financialmodelingprep.com/api/v3/stock-screener?sector={sector}&apikey={FMP_API_KEY}"
                        response = requests.get(url)
                        if response.status_code == 200:
                            sector_companies = response.json()
                            # Take top 5 by market cap
                            peer_list = [comp['symbol'] for comp in sector_companies[:5]]
                            if ticker in peer_list:
                                peer_list.remove(ticker)
                            peers = ','.join(peer_list)
                        else:
                            # Fallback to a default list based on sector
                            sector_peers = {
                                "Technology": "AAPL,MSFT,GOOGL,META,AMZN",
                                "Healthcare": "JNJ,PFE,MRK,UNH,ABT",
                                "Financial Services": "JPM,BAC,WFC,C,GS",
                                "Consumer Cyclical": "AMZN,HD,MCD,NKE,SBUX",
                                "Communication Services": "GOOGL,META,VZ,T,DIS",
                                "Industrials": "HON,UPS,UNP,CAT,BA",
                                "Energy": "XOM,CVX,COP,SLB,EOG"
                            }
                            peers = sector_peers.get(sector, "AAPL,MSFT,GOOGL,AMZN,META")
                            if ticker in peers.split(','):
                                peers_list = peers.split(',')
                                peers_list.remove(ticker)
                                peers = ','.join(peers_list)
            
            if not peers:
                return {"error": f"Could not determine peers for {ticker}, please specify manually"}
                
            # Convert peers string to list
            peer_list = [p.strip() for p in peers.split(',')]
            if ticker not in peer_list:
                peer_list.append(ticker)
                
            # Collect data for all companies
            results = {}
            metrics = ["PE_Ratio", "PS_Ratio", "PB_Ratio", "DE_Ratio", "ROE", "Current_Ratio", "Dividend_Yield"]
            metric_data = {metric: {} for metric in metrics}
            
            for company in peer_list:
                ratio_data = calculate_ratios(company)
                if "error" not in ratio_data:
                    results[company] = ratio_data
                    # Extract metrics for comparison
                    for metric in metrics:
                        if metric in ratio_data["ratios"]:
                            metric_data[metric][company] = ratio_data["ratios"][metric]
            
            # Calculate averages and comparisons
            averages = {metric: np.mean([v for v in values.values() if v is not None]) 
                      for metric, values in metric_data.items() 
                      if any(v is not None for v in values.values())}
            
            # Percentile rankings
            percentiles = {}
            for metric, values in metric_data.items():
                if ticker in values and values[ticker] is not None:
                    valid_values = [v for v in values.values() if v is not None]
                    if valid_values:
                        # Different metrics have different interpretations
                        if metric in ["ROE", "Current_Ratio", "Dividend_Yield"]:
                            # Higher is better
                            percentile = sum(1 for v in valid_values if v < values[ticker]) / len(valid_values) * 100
                        else:
                            # Lower is better for most valuation ratios
                            percentile = sum(1 for v in valid_values if v > values[ticker]) / len(valid_values) * 100
                        percentiles[metric] = round(percentile, 1)
            
            # Create comparison insights
            insights = []
            for metric, percentile in percentiles.items():
                if metric == "PE_Ratio":
                    if percentile > 80:
                        insights.append(f"{ticker} has a more favorable P/E ratio than {percentile}% of peers, suggesting potential undervaluation")
                    elif percentile < 20:
                        insights.append(f"{ticker} has a less favorable P/E ratio than {100-percentile}% of peers, suggesting potential overvaluation")
                    
                elif metric == "ROE":
                    if percentile > 80:
                        insights.append(f"{ticker} has better Return on Equity than {percentile}% of peers, indicating efficient use of capital")
                    elif percentile < 20:
                        insights.append(f"{ticker} has lower Return on Equity than {100-percentile}% of peers, suggesting room for improvement")
                        
                elif metric == "DE_Ratio":
                    if percentile > 80:
                        insights.append(f"{ticker} has a more favorable debt-to-equity ratio than {percentile}% of peers, indicating lower financial risk")
                    elif percentile < 20:
                        insights.append(f"{ticker} has a higher debt-to-equity ratio than {100-percentile}% of peers, suggesting higher financial risk")
                        
            # Prepare summary
            summary = {
                "company": ticker,
                "peer_companies": [p for p in peer_list if p != ticker],
                "metrics_comparison": {
                    metric: {
                        "company_value": metric_data[metric].get(ticker),
                        "peer_average": round(averages[metric], 3) if metric in averages else None,
                        "percentile_rank": percentiles.get(metric)
                    } for metric in metrics if metric in metric_data
                },
                "insights": insights,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Error performing competitive analysis for {ticker}: {str(e)}"}
    
    async def _arun(self, ticker: str, peers: str = None) -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(ticker, peers)

class DCFValuationTool(BaseTool):
    name: str = "dcf_valuation"
    description: str = """
    Performs a simplified Discounted Cash Flow (DCF) valuation for a stock.
    Estimates future free cash flows based on growth rates and discounts them to present value.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        growth_rate: Optional annual growth rate estimate for the next 5 years (e.g., 0.15 for 15%)
        terminal_rate: Optional long-term growth rate after year 5 (e.g., 0.03 for 3%)
        discount_rate: Optional discount rate/required return (e.g., 0.10 for 10%)
        
    Returns:
        A dictionary with DCF valuation results including fair value estimate per share.
    """
    
    def _run(self, ticker: str, growth_rate: float = None, terminal_rate: float = None, 
             discount_rate: float = None) -> Dict[str, Any]:
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            cash_flow = stock.cashflow
            balance_sheet = stock.balance_sheet
            info = stock.info
            
            if cash_flow.empty or balance_sheet.empty:
                return {"error": f"Could not fetch financial data for {ticker}"}
            
            # Current price and shares outstanding
            current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
            shares_outstanding = info.get('sharesOutstanding', None)
            
            if not current_price or not shares_outstanding:
                return {"error": f"Could not fetch current price or shares outstanding for {ticker}"}
            
            # Default parameters if not provided
            if growth_rate is None:
                # Try to get analysts' growth rate estimate or use a default
                growth_rate = info.get('earningsGrowth', info.get('revenueGrowth', 0.10))
            
            if terminal_rate is None:
                # Long-term growth typically around inflation rate
                terminal_rate = 0.025  # 2.5% default
                
            if discount_rate is None:
                # Default discount rate based on typical market return
                discount_rate = 0.10  # 10% default
            
            # Free Cash Flow (FCF) calculation
            # FCF = Operating Cash Flow - Capital Expenditures
            try:
                operating_cash_flow = cash_flow.loc['Cash From Operations'].iloc[0]
                capital_expenditures = cash_flow.loc['Capital Expenditure'].iloc[0]
                current_fcf = operating_cash_flow - abs(capital_expenditures)
                
                if current_fcf <= 0:
                    # Try using net income as a fallback
                    current_fcf = cash_flow.loc['Net Income'].iloc[0] * 0.8  # Approximate FCF
            except:
                # Fallback to a rough estimate based on net income
                try:
                    current_fcf = cash_flow.loc['Net Income'].iloc[0] * 0.8
                except:
                    return {"error": f"Could not calculate free cash flow for {ticker}"}
            
            # Forecast FCF for 5 years
            forecast_fcf = []
            for i in range(1, 6):
                fcf = current_fcf * ((1 + growth_rate) ** i)
                forecast_fcf.append(fcf)
            
            # Terminal value calculation (Gordon Growth Model)
            terminal_value = forecast_fcf[-1] * (1 + terminal_rate) / (discount_rate - terminal_rate)
            
            # Discount all cash flows to present value
            present_values = []
            for i, fcf in enumerate(forecast_fcf):
                pv = fcf / ((1 + discount_rate) ** (i + 1))
                present_values.append(pv)
                
            # Discount terminal value
            terminal_value_pv = terminal_value / ((1 + discount_rate) ** 5)
            
            # Sum up the present values
            total_enterprise_value = sum(present_values) + terminal_value_pv
            
            # Adjustments for cash and debt
            try:
                cash_and_equivalents = balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[0]
            except:
                cash_and_equivalents = 0
                
            try:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
            except:
                total_debt = 0
                
            # Equity value
            equity_value = total_enterprise_value + cash_and_equivalents - total_debt
            
            # Value per share
            value_per_share = equity_value / shares_outstanding
            
            # Comparison with current price
            upside_downside = (value_per_share - current_price) / current_price * 100
            
            # Format results
            results = {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "estimated_fair_value": round(value_per_share, 2),
                "upside_downside_percent": round(upside_downside, 2),
                "assumptions": {
                    "current_fcf_millions": round(current_fcf / 1000000, 2),
                    "growth_rate_percent": round(growth_rate * 100, 2),
                    "terminal_rate_percent": round(terminal_rate * 100, 2),
                    "discount_rate_percent": round(discount_rate * 100, 2)
                },
                "forecast_fcf_millions": [round(fcf / 1000000, 2) for fcf in forecast_fcf],
                "terminal_value_billions": round(terminal_value / 1000000000, 2),
                "total_enterprise_value_billions": round(total_enterprise_value / 1000000000, 2),
                "equity_value_billions": round(equity_value / 1000000000, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add interpretation
            if upside_downside > 20:
                results["interpretation"] = f"{ticker} appears significantly undervalued based on DCF analysis, with potential upside of {round(upside_downside, 1)}%"
            elif upside_downside > 0:
                results["interpretation"] = f"{ticker} appears moderately undervalued based on DCF analysis, with potential upside of {round(upside_downside, 1)}%"
            elif upside_downside > -20:
                results["interpretation"] = f"{ticker} appears fairly valued based on DCF analysis, with estimated value within 20% of current price"
            else:
                results["interpretation"] = f"{ticker} appears overvalued based on DCF analysis, with potential downside of {round(abs(upside_downside), 1)}%"
                
            results["notes"] = [
                "DCF is highly sensitive to input assumptions, particularly growth and discount rates",
                "This is a simplified DCF model and should be supplemented with other valuation methods",
                "Long-term forecasts have inherent uncertainty"
            ]
            
            return results
            
        except Exception as e:
            return {"error": f"Error performing DCF valuation for {ticker}: {str(e)}"}
    
    async def _arun(self, ticker: str, growth_rate: float = None, terminal_rate: float = None, 
                  discount_rate: float = None) -> Dict[str, Any]:
        # Async implementation if needed
        return self._run(ticker, growth_rate, terminal_rate, discount_rate)
