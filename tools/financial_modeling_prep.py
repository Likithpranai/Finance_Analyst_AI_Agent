"""
Tools for fetching financial data from Financial Modeling Prep API
"""
import requests
import pandas as pd
from datetime import datetime
from langchain.tools import BaseTool
from typing import Optional, Dict, Any, List

from config import FMP_API_KEY


class CompanyFinancialsTool(BaseTool):
    """Tool for getting financial statements from Financial Modeling Prep"""
    
    name = "company_financials"
    description = """Gets company financial statements from Financial Modeling Prep API. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - statement (required): Type of statement ('income', 'balance', 'cash', 'metrics', 'ratios', 'growth')
    - period (optional): Period of report ('annual' or 'quarter', default: 'annual')
    - limit (optional): Number of periods to fetch (default: 5)
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            statement_type = kwargs.get("statement")
            period = kwargs.get("period", "annual")
            limit = int(kwargs.get("limit", 5))
            
            if not ticker or not statement_type:
                return "Error: Both ticker and statement type must be provided"
                
            if not FMP_API_KEY:
                return "Error: Financial Modeling Prep API key not configured"
            
            # Map statement types to API endpoints
            endpoint_map = {
                "income": f"income-statement/{ticker}",
                "balance": f"balance-sheet-statement/{ticker}",
                "cash": f"cash-flow-statement/{ticker}",
                "metrics": f"key-metrics/{ticker}",
                "ratios": f"ratios/{ticker}",
                "growth": f"financial-growth/{ticker}"
            }
            
            if statement_type.lower() not in endpoint_map:
                return f"Error: Invalid statement type. Choose from: {', '.join(endpoint_map.keys())}"
            
            endpoint = endpoint_map[statement_type.lower()]
            
            # Construct API URL
            base_url = "https://financialmodelingprep.com/api/v3"
            period_param = "?period=quarter" if period.lower() == "quarter" else ""
            url = f"{base_url}/{endpoint}{period_param}&limit={limit}&apikey={FMP_API_KEY}"
            
            # Make the request
            response = requests.get(url)
            
            # Check if request was successful
            if response.status_code != 200:
                return f"API Error: Status code {response.status_code}"
            
            data = response.json()
            
            if not data:
                return f"No financial data found for {ticker}"
                
            # Format the response
            formatted_data = {
                "source": "Financial Modeling Prep",
                "symbol": ticker,
                "statement_type": statement_type,
                "period": period,
                "data": data,
            }
            
            # Extract some key metrics based on statement type
            if statement_type.lower() == "income":
                key_metrics = {}
                for i, period_data in enumerate(data):
                    date = period_data.get("date", "")
                    key_metrics[date] = {
                        "revenue": period_data.get("revenue"),
                        "gross_profit": period_data.get("grossProfit"),
                        "net_income": period_data.get("netIncome"),
                        "eps": period_data.get("eps")
                    }
                formatted_data["key_metrics"] = key_metrics
                
            elif statement_type.lower() == "balance":
                key_metrics = {}
                for i, period_data in enumerate(data):
                    date = period_data.get("date", "")
                    key_metrics[date] = {
                        "total_assets": period_data.get("totalAssets"),
                        "total_liabilities": period_data.get("totalLiabilities"),
                        "total_equity": period_data.get("totalStockholdersEquity"),
                        "cash_and_equivalents": period_data.get("cashAndCashEquivalents")
                    }
                formatted_data["key_metrics"] = key_metrics
                
            elif statement_type.lower() == "ratios":
                key_metrics = {}
                for i, period_data in enumerate(data):
                    date = period_data.get("date", "")
                    key_metrics[date] = {
                        "pe_ratio": period_data.get("peRatio"),
                        "price_to_book": period_data.get("priceToBookRatio"),
                        "debt_to_equity": period_data.get("debtToEquity"),
                        "dividend_yield": period_data.get("dividendYield")
                    }
                formatted_data["key_metrics"] = key_metrics
                
            return formatted_data
            
        except Exception as e:
            return f"Error fetching financial data from Financial Modeling Prep: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class CompanyValuationTool(BaseTool):
    """Tool for getting company valuation data from Financial Modeling Prep"""
    
    name = "company_valuation"
    description = """Gets company valuation metrics from Financial Modeling Prep API. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - metric (optional): Specific valuation metric ('dcf', 'rating', 'price-target', 'all')
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            metric = kwargs.get("metric", "all").lower()
            
            if not ticker:
                return "Error: Ticker symbol must be provided"
                
            if not FMP_API_KEY:
                return "Error: Financial Modeling Prep API key not configured"
            
            # Base URL for API
            base_url = "https://financialmodelingprep.com/api/v3"
            
            valuation_data = {
                "source": "Financial Modeling Prep",
                "symbol": ticker,
            }
            
            # Fetch DCF (Discounted Cash Flow) valuation
            if metric in ["dcf", "all"]:
                dcf_url = f"{base_url}/dcf/{ticker}?apikey={FMP_API_KEY}"
                dcf_response = requests.get(dcf_url)
                
                if dcf_response.status_code == 200:
                    dcf_data = dcf_response.json()
                    if dcf_data and isinstance(dcf_data, list) and len(dcf_data) > 0:
                        valuation_data["dcf"] = dcf_data[0]
                
            # Fetch company rating
            if metric in ["rating", "all"]:
                rating_url = f"{base_url}/rating/{ticker}?apikey={FMP_API_KEY}"
                rating_response = requests.get(rating_url)
                
                if rating_response.status_code == 200:
                    rating_data = rating_response.json()
                    if rating_data and isinstance(rating_data, list) and len(rating_data) > 0:
                        valuation_data["rating"] = rating_data[0]
                
            # Fetch price targets
            if metric in ["price-target", "all"]:
                price_target_url = f"{base_url}/price-target/{ticker}?apikey={FMP_API_KEY}"
                pt_response = requests.get(price_target_url)
                
                if pt_response.status_code == 200:
                    pt_data = pt_response.json()
                    if pt_data and isinstance(pt_data, list) and len(pt_data) > 0:
                        valuation_data["price_target"] = pt_data[0]
            
            # Generate summary insights
            summary = {}
            
            # DCF comparison with current price
            if "dcf" in valuation_data:
                dcf = valuation_data["dcf"].get("dcf")
                price = valuation_data["dcf"].get("price")
                
                if dcf and price:
                    dcf = float(dcf)
                    price = float(price)
                    difference = ((dcf / price) - 1) * 100
                    
                    if difference > 15:
                        summary["dcf_insight"] = f"DCF valuation suggests stock may be undervalued by {difference:.1f}%"
                    elif difference < -15:
                        summary["dcf_insight"] = f"DCF valuation suggests stock may be overvalued by {abs(difference):.1f}%"
                    else:
                        summary["dcf_insight"] = "DCF valuation suggests stock is fairly valued"
            
            # Rating summary
            if "rating" in valuation_data:
                rating = valuation_data["rating"].get("ratingScore")
                recommendation = valuation_data["rating"].get("recommendation")
                
                if rating:
                    summary["rating_insight"] = f"Overall rating: {rating}/5, Recommendation: {recommendation}"
            
            # Price target
            if "price_target" in valuation_data:
                price = valuation_data["price_target"].get("priceTarget")
                current = valuation_data["price_target"].get("price")
                
                if price and current:
                    price = float(price)
                    current = float(current)
                    potential = ((price / current) - 1) * 100
                    
                    if potential > 0:
                        summary["target_insight"] = f"Analysts suggest upside potential of {potential:.1f}%"
                    else:
                        summary["target_insight"] = f"Analysts suggest downside risk of {abs(potential):.1f}%"
            
            valuation_data["summary"] = summary
            
            return valuation_data
            
        except Exception as e:
            return f"Error fetching valuation data from Financial Modeling Prep: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)


class IndustryComparisonTool(BaseTool):
    """Tool for comparing a company with its industry peers"""
    
    name = "industry_comparison"
    description = """Compares a company with its industry peers using data from Financial Modeling Prep. 
    Input should be a JSON object with:
    - ticker (required): A valid stock ticker symbol
    - metric (optional): Comparison metric ('pe', 'pb', 'ps', 'debt-to-equity', 'dividend-yield')
    """
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool"""
        try:
            ticker = kwargs.get("ticker")
            metric = kwargs.get("metric", "pe").lower()
            
            if not ticker:
                return "Error: Ticker symbol must be provided"
                
            if not FMP_API_KEY:
                return "Error: Financial Modeling Prep API key not configured"
            
            # First get company profile to determine the industry
            profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
            profile_response = requests.get(profile_url)
            
            if profile_response.status_code != 200:
                return f"API Error: Status code {profile_response.status_code}"
                
            profile_data = profile_response.json()
            
            if not profile_data or not isinstance(profile_data, list) or len(profile_data) == 0:
                return f"No company profile found for {ticker}"
                
            industry = profile_data[0].get("industry")
            sector = profile_data[0].get("sector")
            
            if not industry:
                return f"Could not determine industry for {ticker}"
                
            # Now get peers in the same industry
            peers_url = f"https://financialmodelingprep.com/api/v3/stock-screener?industry={industry}&limit=20&apikey={FMP_API_KEY}"
            peers_response = requests.get(peers_url)
            
            if peers_response.status_code != 200:
                return f"API Error: Status code {peers_response.status_code}"
                
            peers_data = peers_response.json()
            
            if not peers_data:
                return f"No industry peers found for {ticker} in {industry}"
                
            # Map metrics to API fields
            metric_map = {
                "pe": "pe",
                "pb": "pb",
                "ps": "price/salesTTM",
                "debt-to-equity": "debtToEquity",
                "dividend-yield": "dividendYield"
            }
            
            api_field = metric_map.get(metric)
            if not api_field:
                return f"Invalid metric. Choose from: {', '.join(metric_map.keys())}"
                
            # Collect comparison data
            comparison_data = []
            target_data = None
            
            for company in peers_data:
                company_symbol = company.get("symbol")
                company_name = company.get("companyName")
                company_metric = company.get(api_field)
                
                if company_metric:
                    peer_data = {
                        "symbol": company_symbol,
                        "name": company_name,
                        "value": company_metric
                    }
                    
                    if company_symbol == ticker:
                        target_data = peer_data
                    else:
                        comparison_data.append(peer_data)
            
            # Sort the comparison data
            comparison_data.sort(key=lambda x: x["value"])
            
            # Calculate industry average
            values = [peer["value"] for peer in comparison_data if peer["value"] is not None]
            industry_avg = sum(values) / len(values) if values else None
            
            # Format the response
            result = {
                "source": "Financial Modeling Prep",
                "company": ticker,
                "industry": industry,
                "sector": sector,
                "metric": metric,
                "target_value": target_data["value"] if target_data else None,
                "industry_average": industry_avg,
                "peers": comparison_data[:10],  # Limit to top 10 peers
            }
            
            # Add percentile rank if target data exists
            if target_data and industry_avg is not None:
                rank = sum(1 for peer in comparison_data if peer["value"] < target_data["value"]) / len(comparison_data) * 100
                result["percentile_rank"] = rank
                
                # Add interpretation based on the metric
                if metric in ["pe", "pb", "ps"]:
                    if rank < 25:
                        result["interpretation"] = f"{ticker} has a lower {metric.upper()} ratio than most industry peers, suggesting it may be undervalued relative to its industry."
                    elif rank > 75:
                        result["interpretation"] = f"{ticker} has a higher {metric.upper()} ratio than most industry peers, suggesting it may be overvalued relative to its industry."
                    else:
                        result["interpretation"] = f"{ticker}'s {metric.upper()} ratio is in line with industry peers."
                elif metric == "debt-to-equity":
                    if rank < 25:
                        result["interpretation"] = f"{ticker} has lower debt relative to equity than most industry peers, suggesting lower financial risk."
                    elif rank > 75:
                        result["interpretation"] = f"{ticker} has higher debt relative to equity than most industry peers, suggesting higher financial risk."
                    else:
                        result["interpretation"] = f"{ticker}'s debt level is in line with industry peers."
                elif metric == "dividend-yield":
                    if rank < 25:
                        result["interpretation"] = f"{ticker} has a lower dividend yield than most industry peers."
                    elif rank > 75:
                        result["interpretation"] = f"{ticker} has a higher dividend yield than most industry peers."
                    else:
                        result["interpretation"] = f"{ticker}'s dividend yield is in line with industry peers."
            
            return result
            
        except Exception as e:
            return f"Error performing industry comparison: {str(e)}"
    
    def _arun(self, **kwargs):
        """Run the tool asynchronously"""
        return self._run(**kwargs)
