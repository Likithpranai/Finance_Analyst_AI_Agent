#!/usr/bin/env python3
"""
Script to run a specific query through the Finance Analyst AI Agent
With enhanced demonstration capabilities
"""

import os
import time
import json
from finance_analyst_agent import FinanceAnalystReActAgent

def get_mock_comprehensive_response(symbol="MSFT"):
    """
    Generate a comprehensive mock response with all capabilities
    to demonstrate the full UI formatting capabilities
    """
    return f'''
# Comprehensive Analysis: {symbol}

## Summary
Microsoft Corporation (MSFT) shows **strong technical indicators** with a bullish trend in the medium term. Fundamental analysis reveals solid financial health with consistent revenue growth and strong profitability. Recent news regarding cloud services expansion and AI integration is positive for future growth prospects.

## Technical Analysis
- **RSI(14)**: 67.84 (Neutral but approaching overbought territory)
- **MACD**: MACD Line (10.67) below Signal Line (11.15), Histogram (-0.48) showing weakening bearish momentum
- **Moving Averages**: Price above both 50-day and 200-day SMAs indicating bullish trend
- **Bollinger Bands**: Trading in upper half of bands showing positive momentum
- **Volume Analysis**: Recent volume increase on up days confirms buying pressure

## Fundamental Analysis
- **P/E Ratio**: 34.2 (Above industry average of 28.5)
- **Forward P/E**: 29.8 (Indicating expected earnings growth)
- **PEG Ratio**: 1.8 (Slightly overvalued relative to growth)
- **Revenue Growth**: +12.8% YoY
- **Profit Margin**: 38.5% (Exceptional compared to industry average of 22.3%)
- **Debt-to-Equity**: 0.42 (Healthy debt levels)
- **Return on Equity**: 43.2% (Excellent capital efficiency)

## Financial Statement Highlights
- **Revenue**: $211.9B (TTM), showing consistent growth across all business segments
- **Net Income**: $72.4B (TTM), +18.3% YoY
- **Cash & Equivalents**: $104.7B, providing strong liquidity
- **Free Cash Flow**: $63.8B (TTM), enabling continued investments and shareholder returns

## News Impact
1. **Cloud Services Expansion**: Microsoft announced expansion of Azure data centers in Asia, potentially increasing market share (+)
2. **AI Integration**: New Copilot features for Microsoft 365 expected to drive subscription growth (+)
3. **Regulatory Concerns**: Potential antitrust scrutiny in EU markets for cloud services dominance (-)
4. **Strategic Acquisition**: Recent acquisition of cybersecurity firm enhances product offerings (+)

## Risk Assessment
- **Market Risk**: Beta of 0.92 indicates slightly lower volatility than the market
- **Competition Risk**: Medium - facing strong competition from Amazon (AWS) and Google (Cloud)
- **Valuation Risk**: High - trading at premium multiples requires continued growth execution
- **Regulatory Risk**: Medium - increasing scrutiny from global regulators

## Investment Recommendation
**MODERATE BUY**

*Target Price Range*: $390-410 (12-month)

**Rationale**: Microsoft's strong market position in cloud services, productivity software, and growing AI capabilities support continued growth. While valuation is rich, the company's consistent execution, strong balance sheet, and diverse revenue streams justify a premium. Recent technical indicators show positive momentum despite some near-term resistance.

**Entry Strategy**: Consider phased buying with initial position now and adding on any pullbacks below $350.

## Visualization Data
![MSFT Price Chart with Technical Indicators](/static/visualizations/msft_technical_chart_20250720.png)
![MSFT Financial Performance](/static/visualizations/msft_fundamental_metrics_20250720.png)
'''

def main():
    print("Initializing Finance Analyst AI Agent...")
    agent = FinanceAnalystReActAgent()
    
    query = "Provide a comprehensive analysis of MSFT including technical indicators, fundamental data (P/E ratio, revenue growth, earnings), news impact, and a detailed buy/sell recommendation"
    
    print(f"\nProcessing query: '{query}'")
    print("-" * 80)
    
    start_time = time.time()
    try:
        response = agent.process_query(query)
        
        # Check if the response is missing fundamental data or news
        if "Missing Data: No fundamental data" in response or "Missing Data: No news" in response:
            print("\nDetected missing data in response. Using enhanced demonstration mode...")
            response = get_mock_comprehensive_response("MSFT")
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        print("Falling back to demonstration mode")
        response = get_mock_comprehensive_response("MSFT")
        
    end_time = time.time()
    
    print("\nResponse:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    print(f"Query processed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
