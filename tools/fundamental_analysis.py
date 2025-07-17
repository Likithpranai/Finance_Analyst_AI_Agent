"""
Fundamental Analysis Tools for Finance Analyst AI Agent
Provides key financial ratios and fundamental analysis capabilities
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union

class FundamentalAnalysisTools:
    """Tools for fundamental analysis of stocks"""
    
    @staticmethod
    def get_financial_ratios(symbol: str) -> Dict:
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
            earnings_growth = info.get('earningsGrowth', info.get('earningsQuarterlyGrowth', None))
            
            # Calculate financial ratios
            ratios = {
                "symbol": symbol,
                "company_name": info.get('longName', 'N/A'),
                "market_price": market_price,
                "ratios": {}
            }
            
            # Price-to-Earnings (P/E) Ratio
            if eps and eps != 0:
                ratios["ratios"]["pe_ratio"] = {
                    "value": market_price / eps,
                    "formula": "Market Price per Share / Earnings per Share (EPS)",
                    "description": "Measures how much investors pay per dollar of earnings"
                }
            else:
                ratios["ratios"]["pe_ratio"] = {
                    "value": None,
                    "formula": "Market Price per Share / Earnings per Share (EPS)",
                    "description": "Not available (EPS may be zero or negative)"
                }
            
            # Forward P/E Ratio
            if forward_eps and forward_eps != 0:
                ratios["ratios"]["forward_pe_ratio"] = {
                    "value": market_price / forward_eps,
                    "formula": "Market Price per Share / Forward EPS",
                    "description": "P/E ratio using projected earnings for next fiscal year"
                }
            
            # Price/Earnings-to-Growth (PEG) Ratio
            if eps and eps != 0 and earnings_growth and earnings_growth != 0:
                pe_ratio = market_price / eps
                ratios["ratios"]["peg_ratio"] = {
                    "value": pe_ratio / earnings_growth,
                    "formula": "P/E Ratio / Earnings Growth Rate",
                    "description": "Adjusts P/E for growth; ideal <1 for growth stocks"
                }
            
            # Price-to-Sales (P/S) Ratio
            if revenue_per_share and revenue_per_share != 0:
                ratios["ratios"]["ps_ratio"] = {
                    "value": market_price / revenue_per_share,
                    "formula": "Market Price per Share / Revenue per Share",
                    "description": "Useful for unprofitable companies; compares sales efficiency"
                }
            
            # Price-to-Book (P/B) Ratio
            if book_value and book_value != 0:
                ratios["ratios"]["pb_ratio"] = {
                    "value": market_price / book_value,
                    "formula": "Market Price per Share / Book Value per Share",
                    "description": "Compares market value to net assets; <1 suggests undervaluation"
                }
            
            # Debt-to-Equity (D/E) Ratio
            if total_debt is not None and shareholders_equity and shareholders_equity != 0:
                ratios["ratios"]["de_ratio"] = {
                    "value": total_debt / shareholders_equity,
                    "formula": "Total Debt / Shareholders' Equity",
                    "description": "Assesses leverage; higher values indicate more debt risk"
                }
            
            # Return on Equity (ROE)
            if net_income and shareholders_equity and shareholders_equity != 0:
                ratios["ratios"]["roe"] = {
                    "value": net_income / shareholders_equity,
                    "formula": "Net Income / Shareholders' Equity",
                    "description": "Measures profitability from equity; higher is better"
                }
            
            # Earnings per Share (EPS)
            if eps is not None:
                ratios["ratios"]["eps"] = {
                    "value": eps,
                    "formula": "Net Income / Outstanding Shares",
                    "description": "Indicates profitability per share"
                }
            
            return ratios
            
        except Exception as e:
            return {"error": f"Error calculating financial ratios for {symbol}: {str(e)}"}
    
    @staticmethod
    def get_income_statement(symbol: str, period: str = "annual") -> Dict:
        """
        Get income statement for a stock
        
        Args:
            symbol: Stock ticker symbol
            period: 'annual' or 'quarterly'
            
        Returns:
            Dictionary with income statement data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if period.lower() == "annual":
                income_stmt = ticker.income_stmt
            else:
                income_stmt = ticker.quarterly_income_stmt
            
            if income_stmt.empty:
                return {"error": f"Could not retrieve income statement for {symbol}"}
            
            # Convert to dictionary format
            income_dict = {
                "symbol": symbol,
                "period": period,
                "currency": ticker.info.get('currency', 'USD'),
                "data": income_stmt.to_dict()
            }
            
            return income_dict
            
        except Exception as e:
            return {"error": f"Error retrieving income statement for {symbol}: {str(e)}"}
    
    @staticmethod
    def get_balance_sheet(symbol: str, period: str = "annual") -> Dict:
        """
        Get balance sheet for a stock
        
        Args:
            symbol: Stock ticker symbol
            period: 'annual' or 'quarterly'
            
        Returns:
            Dictionary with balance sheet data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if period.lower() == "annual":
                balance_sheet = ticker.balance_sheet
            else:
                balance_sheet = ticker.quarterly_balance_sheet
            
            if balance_sheet.empty:
                return {"error": f"Could not retrieve balance sheet for {symbol}"}
            
            # Convert to dictionary format
            balance_dict = {
                "symbol": symbol,
                "period": period,
                "currency": ticker.info.get('currency', 'USD'),
                "data": balance_sheet.to_dict()
            }
            
            return balance_dict
            
        except Exception as e:
            return {"error": f"Error retrieving balance sheet for {symbol}: {str(e)}"}
    
    @staticmethod
    def get_cash_flow(symbol: str, period: str = "annual") -> Dict:
        """
        Get cash flow statement for a stock
        
        Args:
            symbol: Stock ticker symbol
            period: 'annual' or 'quarterly'
            
        Returns:
            Dictionary with cash flow data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if period.lower() == "annual":
                cash_flow = ticker.cashflow
            else:
                cash_flow = ticker.quarterly_cashflow
            
            if cash_flow.empty:
                return {"error": f"Could not retrieve cash flow statement for {symbol}"}
            
            # Convert to dictionary format
            cash_flow_dict = {
                "symbol": symbol,
                "period": period,
                "currency": ticker.info.get('currency', 'USD'),
                "data": cash_flow.to_dict()
            }
            
            return cash_flow_dict
            
        except Exception as e:
            return {"error": f"Error retrieving cash flow statement for {symbol}: {str(e)}"}
    
    @staticmethod
    def format_financial_ratios_for_display(ratios: Dict) -> str:
        """
        Format financial ratios for display
        
        Args:
            ratios: Dictionary with financial ratios
            
        Returns:
            Formatted string for display
        """
        if "error" in ratios:
            return f"Error: {ratios['error']}"
        
        result = f"Financial Ratios for {ratios['symbol']} ({ratios['company_name']}):\n\n"
        
        for ratio_name, ratio_data in ratios["ratios"].items():
            if ratio_data["value"] is not None:
                # Format the value based on the ratio type
                if ratio_name in ["pe_ratio", "forward_pe_ratio", "peg_ratio", "ps_ratio", "pb_ratio"]:
                    value = f"{ratio_data['value']:.2f}x"
                elif ratio_name in ["de_ratio"]:
                    value = f"{ratio_data['value']:.2f}"
                elif ratio_name in ["roe"]:
                    value = f"{ratio_data['value']*100:.2f}%"
                elif ratio_name in ["eps"]:
                    value = f"${ratio_data['value']:.2f}"
                else:
                    value = f"{ratio_data['value']}"
                
                # Format the ratio name for display
                if ratio_name == "pe_ratio":
                    display_name = "Price-to-Earnings (P/E) Ratio"
                elif ratio_name == "forward_pe_ratio":
                    display_name = "Forward P/E Ratio"
                elif ratio_name == "peg_ratio":
                    display_name = "Price/Earnings-to-Growth (PEG) Ratio"
                elif ratio_name == "ps_ratio":
                    display_name = "Price-to-Sales (P/S) Ratio"
                elif ratio_name == "pb_ratio":
                    display_name = "Price-to-Book (P/B) Ratio"
                elif ratio_name == "de_ratio":
                    display_name = "Debt-to-Equity (D/E) Ratio"
                elif ratio_name == "roe":
                    display_name = "Return on Equity (ROE)"
                elif ratio_name == "eps":
                    display_name = "Earnings per Share (EPS)"
                else:
                    display_name = ratio_name.replace("_", " ").title()
                
                result += f"{display_name}: {value}\n"
                result += f"  • {ratio_data['description']}\n"
                result += f"  • Formula: {ratio_data['formula']}\n\n"
            else:
                result += f"{ratio_name.replace('_', ' ').title()}: Not available\n\n"
        
        # Add interpretation
        result += "Interpretation Guide:\n"
        result += "• P/E Ratio: Lower values may indicate undervaluation\n"
        result += "• PEG Ratio: Values below 1 may indicate an undervalued stock relative to growth\n"
        result += "• P/S Ratio: Lower values suggest better value relative to sales\n"
        result += "• P/B Ratio: Values below 1 may suggest undervaluation\n"
        result += "• D/E Ratio: Lower values indicate less leverage and potentially less risk\n"
        result += "• ROE: Higher values indicate better profitability from equity\n"
        
        return result
    
    @staticmethod
    def get_industry_comparison(symbol: str, ratios: List[str] = None) -> Dict:
        """
        Get industry comparison for a stock's financial ratios
        
        Args:
            symbol: Stock ticker symbol
            ratios: List of ratios to compare (default: all available)
            
        Returns:
            Dictionary with industry comparison data
        """
        try:
            if ratios is None:
                ratios = ["pe_ratio", "ps_ratio", "pb_ratio", "de_ratio", "roe"]
                
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get industry and sector
            industry = info.get('industry', None)
            sector = info.get('sector', None)
            
            if not industry:
                return {"error": f"Could not retrieve industry information for {symbol}"}
            
            # Get stock's ratios
            stock_ratios = FundamentalAnalysisTools.get_financial_ratios(symbol)
            if "error" in stock_ratios:
                return stock_ratios
            
            # This would ideally fetch industry averages from a database or API
            # For now, we'll use placeholder values
            # In a real implementation, you would fetch actual industry averages
            industry_averages = {
                "pe_ratio": 20.5,
                "ps_ratio": 2.3,
                "pb_ratio": 3.1,
                "de_ratio": 1.2,
                "roe": 0.15
            }
            
            # Compare stock's ratios to industry averages
            comparison = {
                "symbol": symbol,
                "company_name": info.get('longName', 'N/A'),
                "industry": industry,
                "sector": sector,
                "comparisons": {}
            }
            
            for ratio in ratios:
                if ratio in stock_ratios["ratios"] and stock_ratios["ratios"][ratio]["value"] is not None:
                    stock_value = stock_ratios["ratios"][ratio]["value"]
                    industry_value = industry_averages.get(ratio, None)
                    
                    if industry_value:
                        # Calculate percentage difference
                        pct_diff = ((stock_value - industry_value) / industry_value) * 100
                        
                        comparison["comparisons"][ratio] = {
                            "stock_value": stock_value,
                            "industry_average": industry_value,
                            "percentage_difference": pct_diff,
                            "relative_position": "above average" if pct_diff > 0 else "below average"
                        }
            
            return comparison
            
        except Exception as e:
            return {"error": f"Error comparing industry ratios for {symbol}: {str(e)}"}
