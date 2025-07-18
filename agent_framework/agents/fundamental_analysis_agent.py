"""
Fundamental Analysis Agent for Finance Analyst AI Agent Framework
"""
import re
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd

from agent_framework.agents.base_agent import BaseAgent
from tools.fundamental_analysis import FundamentalAnalysisTools
from tools.real_time_data_integration import RealTimeDataTools

class FundamentalAnalysisAgent(BaseAgent):
    """
    Specialized agent for fundamental analysis of companies and financial instruments
    """
    
    def __init__(self):
        """Initialize the Fundamental Analysis Agent with specialized tools"""
        # Initialize with fundamental analysis tools
        tools = {
            # Fundamental analysis tools
            "get_income_statement": FundamentalAnalysisTools.get_income_statement,
            "get_balance_sheet": FundamentalAnalysisTools.get_balance_sheet,
            "get_cash_flow": FundamentalAnalysisTools.get_cash_flow,
            "get_financial_ratios": FundamentalAnalysisTools.get_financial_ratios,
            "get_company_profile": FundamentalAnalysisTools.get_company_profile,
            "get_earnings_calendar": FundamentalAnalysisTools.get_earnings_calendar,
            "get_industry_comparison": FundamentalAnalysisTools.get_industry_comparison,
            "visualize_financial_metrics": FundamentalAnalysisTools.visualize_financial_metrics,
            
            # Real-time data tools for company information
            "get_company_details": RealTimeDataTools.get_company_details,
            "get_market_news": RealTimeDataTools.get_market_news,
        }
        
        # Initialize the base agent
        super().__init__(
            name="Fundamental Analysis Agent",
            description="Specialized agent for fundamental analysis of companies and financial instruments",
            tools=tools
        )
        
        # Add specialized system prompt for fundamental analysis
        self.system_prompt = """
        You are a Fundamental Analysis Agent specializing in analyzing companies based on financial statements, valuation metrics, and business fundamentals.
        
        Follow the ReAct pattern: Reason → Act → Observe → Loop.
        
        When analyzing a company:
        1. REASON: Determine what financial data and metrics would be most relevant
        2. ACT: Use appropriate tools to gather financial statements and calculate key ratios
        3. OBSERVE: Analyze the results and identify strengths, weaknesses, and trends
        4. LOOP: If needed, gather additional data or calculate other metrics
        
        Focus on:
        - Revenue growth and profitability trends
        - Balance sheet strength and liquidity
        - Cash flow generation and quality
        - Valuation metrics (P/E, P/S, P/B, EV/EBITDA)
        - Return on capital (ROE, ROIC)
        - Industry comparisons and competitive positioning
        - Business model analysis and economic moat
        - Management quality and capital allocation
        
        Always provide:
        - Clear assessment of financial health
        - Key growth drivers and risks
        - Valuation analysis with fair value estimate
        - Competitive positioning within industry
        - Long-term business outlook
        
        DO NOT make up information. Only use the data provided by the tools.
        """
    
    def extract_symbol(self, query: str) -> str:
        """Extract stock symbol from user query"""
        # Look for stock symbols in the query (typically 1-5 uppercase letters)
        matches = re.findall(r'\b[A-Z]{1,5}\b', query)
        if matches:
            return matches[0]
        
        # Look for common phrases like "for AAPL" or "of MSFT"
        symbol_match = re.search(r'(?:for|of|on|about)\s+([A-Z]{1,5})', query, re.IGNORECASE)
        if symbol_match:
            return symbol_match.group(1).upper()
        
        return ""
    
    def determine_analysis_type(self, query: str) -> List[str]:
        """Determine what type of fundamental analysis to perform"""
        analysis_types = []
        
        if any(term in query.lower() for term in ["income", "revenue", "earnings", "profit", "eps"]):
            analysis_types.append("income_statement")
        
        if any(term in query.lower() for term in ["balance", "assets", "liabilities", "debt"]):
            analysis_types.append("balance_sheet")
        
        if any(term in query.lower() for term in ["cash flow", "cash", "dividend", "capex"]):
            analysis_types.append("cash_flow")
        
        if any(term in query.lower() for term in ["ratio", "pe", "ps", "pb", "valuation"]):
            analysis_types.append("financial_ratios")
        
        if any(term in query.lower() for term in ["profile", "about", "business", "company"]):
            analysis_types.append("company_profile")
        
        if any(term in query.lower() for term in ["industry", "sector", "competitor", "compare"]):
            analysis_types.append("industry_comparison")
        
        # If no specific analysis types mentioned, use a comprehensive set
        if not analysis_types:
            analysis_types = ["company_profile", "financial_ratios", "income_statement"]
            
        return analysis_types
    
    def process_query(self, query: str) -> str:
        """
        Process a fundamental analysis query
        
        Args:
            query: User query string
            
        Returns:
            Fundamental analysis response
        """
        # Extract symbol from query
        symbol = self.extract_symbol(query)
        if not symbol:
            return "I couldn't identify a company symbol in your query. Please specify a valid stock symbol (e.g., AAPL, MSFT, GOOGL)."
        
        # Determine analysis types
        analysis_types = self.determine_analysis_type(query)
        
        # Log the analysis plan
        print(f"REACT - REASON: Fundamental analysis for {symbol} with focus on: {', '.join(analysis_types)}")
        
        # Gather data
        results = {}
        
        try:
            # Get company details
            print(f"REACT - ACT: Getting company details for {symbol}")
            results["company_details"] = self.execute_tool("get_company_details", symbol=symbol)
            print(f"REACT - OBSERVE: Got company details for {symbol}")
            
            # Get relevant financial data based on analysis types
            for analysis_type in analysis_types:
                if analysis_type == "income_statement":
                    print(f"REACT - ACT: Getting income statement for {symbol}")
                    results["income_statement"] = self.execute_tool("get_income_statement", symbol=symbol)
                    print(f"REACT - OBSERVE: Got income statement for {symbol}")
                
                elif analysis_type == "balance_sheet":
                    print(f"REACT - ACT: Getting balance sheet for {symbol}")
                    results["balance_sheet"] = self.execute_tool("get_balance_sheet", symbol=symbol)
                    print(f"REACT - OBSERVE: Got balance sheet for {symbol}")
                
                elif analysis_type == "cash_flow":
                    print(f"REACT - ACT: Getting cash flow statement for {symbol}")
                    results["cash_flow"] = self.execute_tool("get_cash_flow", symbol=symbol)
                    print(f"REACT - OBSERVE: Got cash flow statement for {symbol}")
                
                elif analysis_type == "financial_ratios":
                    print(f"REACT - ACT: Getting financial ratios for {symbol}")
                    results["financial_ratios"] = self.execute_tool("get_financial_ratios", symbol=symbol)
                    print(f"REACT - OBSERVE: Got financial ratios for {symbol}")
                
                elif analysis_type == "company_profile":
                    print(f"REACT - ACT: Getting company profile for {symbol}")
                    results["company_profile"] = self.execute_tool("get_company_profile", symbol=symbol)
                    print(f"REACT - OBSERVE: Got company profile for {symbol}")
                
                elif analysis_type == "industry_comparison":
                    print(f"REACT - ACT: Getting industry comparison for {symbol}")
                    results["industry_comparison"] = self.execute_tool("get_industry_comparison", symbol=symbol)
                    print(f"REACT - OBSERVE: Got industry comparison for {symbol}")
            
            # Get recent news
            print(f"REACT - ACT: Getting market news for {symbol}")
            results["market_news"] = self.execute_tool("get_market_news", symbol=symbol, limit=5)
            print(f"REACT - OBSERVE: Got market news for {symbol}")
            
            # Generate visualization if requested
            if "chart" in query.lower() or "visual" in query.lower() or "graph" in query.lower():
                print(f"REACT - ACT: Generating financial metrics visualization for {symbol}")
                metrics = ["revenue", "net_income", "eps", "pe_ratio"] if "financial_ratios" in results else ["revenue", "net_income"]
                results["visualization"] = self.execute_tool("visualize_financial_metrics", symbol=symbol, metrics=metrics)
                print(f"REACT - OBSERVE: Generated financial metrics visualization for {symbol}")
            
            # Generate analysis using Gemini
            print(f"REACT - LOOP: Generating fundamental analysis for {symbol}")
            
            # Create prompt for Gemini
            prompt = f"""
            Provide a comprehensive fundamental analysis for {symbol} based on the following data:
            
            Company details: {results.get('company_details', 'Not available')}
            
            Company profile: {results.get('company_profile', 'Not available')}
            
            Income statement: {results.get('income_statement', 'Not available')}
            
            Balance sheet: {results.get('balance_sheet', 'Not available')}
            
            Cash flow statement: {results.get('cash_flow', 'Not available')}
            
            Financial ratios: {results.get('financial_ratios', 'Not available')}
            
            Industry comparison: {results.get('industry_comparison', 'Not available')}
            
            Recent news: {results.get('market_news', 'Not available')}
            
            Follow the ReAct pattern (Reason → Act → Observe → Loop) and include:
            1. Business overview and competitive positioning
            2. Financial health assessment
            3. Growth trends and profitability analysis
            4. Valuation analysis with fair value estimate
            5. Key risks and opportunities
            6. Investment recommendation (Buy, Hold, or Sell)
            
            Format your response in markdown with clear sections.
            """
            
            analysis = self.generate_response(prompt)
            
            # Add visualization if available
            if "visualization" in results:
                analysis += f"\n\n{results['visualization']}"
            
            return analysis
            
        except Exception as e:
            error_message = f"Error performing fundamental analysis for {symbol}: {str(e)}"
            print(error_message)
            return error_message
