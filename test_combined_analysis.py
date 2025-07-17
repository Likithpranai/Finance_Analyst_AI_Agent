"""
Test script for combined fundamental and technical analysis
This demonstrates how the Finance Analyst AI Agent can use both types of analysis together
"""

import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tools.fundamental_analysis import FundamentalAnalysisTools
from tools.alpha_vantage_tools import AlphaVantageTools

# Load environment variables
load_dotenv()

def perform_combined_analysis(symbol):
    """
    Perform a combined fundamental and technical analysis for a stock
    following the ReAct pattern: Reason → Act → Observe → Loop
    """
    print("=" * 80)
    print(f"COMBINED ANALYSIS FOR {symbol}".center(80))
    print("=" * 80)
    
    # REASON: Determine what information we need for a comprehensive analysis
    print("\n[REASON] Planning analysis approach...")
    print("1. We need current price and technical indicators for short-term analysis")
    print("2. We need fundamental ratios for long-term value assessment")
    print("3. We need news and market context")
    
    # ACT: Gather technical data
    print("\n[ACT] Gathering technical data...")
    try:
        # Get stock data
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(period="6mo")
        
        if hist_data.empty:
            print(f"Error: Could not retrieve data for {symbol}")
            return
        
        # Calculate technical indicators
        # RSI
        delta = hist_data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD
        exp1 = hist_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist_data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        
        # Moving Averages
        sma50 = hist_data['Close'].rolling(window=50).mean().iloc[-1]
        sma200 = hist_data['Close'].rolling(window=200).mean().iloc[-1]
        
        # Current price
        current_price = hist_data['Close'].iloc[-1]
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"RSI (14): {current_rsi:.2f}")
        print(f"MACD: {current_macd:.2f}, Signal: {current_signal:.2f}")
        print(f"50-day SMA: ${sma50:.2f}, 200-day SMA: ${sma200:.2f}")
        
    except Exception as e:
        print(f"Error in technical analysis: {str(e)}")
        return
    
    # ACT: Gather fundamental data
    print("\n[ACT] Gathering fundamental data...")
    try:
        # Get financial ratios
        ratios = FundamentalAnalysisTools.get_financial_ratios(symbol)
        
        if "error" in ratios:
            print(f"Error: {ratios['error']}")
        else:
            # Extract key ratios
            pe_ratio = ratios["ratios"].get("pe_ratio", {}).get("value", "N/A")
            ps_ratio = ratios["ratios"].get("ps_ratio", {}).get("value", "N/A")
            pb_ratio = ratios["ratios"].get("pb_ratio", {}).get("value", "N/A")
            de_ratio = ratios["ratios"].get("de_ratio", {}).get("value", "N/A")
            roe = ratios["ratios"].get("roe", {}).get("value", "N/A")
            eps = ratios["ratios"].get("eps", {}).get("value", "N/A")
            
            print(f"P/E Ratio: {pe_ratio:.2f if pe_ratio != 'N/A' and pe_ratio is not None else 'N/A'}")
            print(f"P/S Ratio: {ps_ratio:.2f if ps_ratio != 'N/A' and ps_ratio is not None else 'N/A'}")
            print(f"P/B Ratio: {pb_ratio:.2f if pb_ratio != 'N/A' and pb_ratio is not None else 'N/A'}")
            print(f"D/E Ratio: {de_ratio:.2f if de_ratio != 'N/A' and de_ratio is not None else 'N/A'}")
            print(f"ROE: {roe*100:.2f}%" if roe != 'N/A' and roe is not None else "ROE: N/A")
            print(f"EPS: ${eps:.2f}" if eps != 'N/A' and eps is not None else "EPS: N/A")
    
    except Exception as e:
        print(f"Error in fundamental analysis: {str(e)}")
    
    # ACT: Get latest news
    print("\n[ACT] Getting latest news...")
    try:
        news = ticker.news
        
        if not news:
            print("No recent news found")
        else:
            print(f"Latest news for {symbol}:")
            for i, item in enumerate(news[:3], 1):
                title = item.get('title', 'No title')
                publisher = item.get('publisher', 'Unknown source')
                print(f"{i}. {title} ({publisher})")
    
    except Exception as e:
        print(f"Error getting news: {str(e)}")
    
    # OBSERVE: Analyze technical indicators
    print("\n[OBSERVE] Analyzing technical indicators...")
    
    # RSI analysis
    if current_rsi > 70:
        rsi_status = "overbought"
    elif current_rsi < 30:
        rsi_status = "oversold"
    else:
        rsi_status = "neutral"
    
    print(f"RSI Analysis: {rsi_status.capitalize()} ({current_rsi:.2f})")
    
    # MACD analysis
    if current_macd > current_signal:
        macd_trend = "bullish"
    else:
        macd_trend = "bearish"
    
    print(f"MACD Analysis: {macd_trend.capitalize()} trend")
    
    # Moving Average analysis
    if current_price > sma50 > sma200:
        ma_trend = "strong uptrend"
    elif current_price > sma50 and sma50 < sma200:
        ma_trend = "potential bullish crossover"
    elif current_price < sma50 and sma50 > sma200:
        ma_trend = "short-term weakness in uptrend"
    elif current_price < sma50 < sma200:
        ma_trend = "strong downtrend"
    else:
        ma_trend = "mixed signals"
    
    print(f"Moving Average Analysis: {ma_trend.capitalize()}")
    
    # OBSERVE: Analyze fundamental ratios
    print("\n[OBSERVE] Analyzing fundamental ratios...")
    
    fundamental_analysis = []
    
    # P/E analysis
    if pe_ratio != "N/A" and pe_ratio is not None:
        if pe_ratio < 15:
            fundamental_analysis.append("P/E ratio suggests potential undervaluation")
        elif pe_ratio > 30:
            fundamental_analysis.append("P/E ratio suggests potential overvaluation")
        else:
            fundamental_analysis.append("P/E ratio is in a moderate range")
    
    # P/B analysis
    if pb_ratio != "N/A" and pb_ratio is not None:
        if pb_ratio < 1:
            fundamental_analysis.append("P/B ratio suggests the stock may be undervalued")
        elif pb_ratio > 5:
            fundamental_analysis.append("P/B ratio is relatively high")
    
    # D/E analysis
    if de_ratio != "N/A" and de_ratio is not None:
        if de_ratio < 0.5:
            fundamental_analysis.append("Low debt-to-equity ratio indicates strong financial position")
        elif de_ratio > 2:
            fundamental_analysis.append("High debt-to-equity ratio indicates higher financial risk")
    
    # Print fundamental analysis
    if fundamental_analysis:
        for analysis in fundamental_analysis:
            print(f"- {analysis}")
    else:
        print("Insufficient fundamental data for analysis")
    
    # LOOP: Combine analyses for final recommendation
    print("\n[LOOP] Generating combined analysis and recommendation...")
    
    # Determine technical outlook
    if rsi_status == "oversold" and macd_trend == "bullish":
        technical_outlook = "strongly bullish"
    elif rsi_status == "overbought" and macd_trend == "bearish":
        technical_outlook = "strongly bearish"
    elif (rsi_status == "neutral" and macd_trend == "bullish") or (rsi_status == "oversold"):
        technical_outlook = "moderately bullish"
    elif (rsi_status == "neutral" and macd_trend == "bearish") or (rsi_status == "overbought"):
        technical_outlook = "moderately bearish"
    else:
        technical_outlook = "neutral"
    
    # Determine fundamental outlook (simplified)
    if (pe_ratio != "N/A" and pe_ratio is not None and 
        ps_ratio != "N/A" and ps_ratio is not None and 
        pb_ratio != "N/A" and pb_ratio is not None):
        if (pe_ratio < 15 or pe_ratio == "N/A") and ps_ratio < 2 and pb_ratio < 3:
            fundamental_outlook = "potentially undervalued"
        elif pe_ratio > 30 and ps_ratio > 5 and pb_ratio > 5:
            fundamental_outlook = "potentially overvalued"
        else:
            fundamental_outlook = "fairly valued"
    else:
        fundamental_outlook = "insufficient data"
    
    # Print combined outlook
    print(f"\nTechnical Outlook: {technical_outlook.capitalize()}")
    print(f"Fundamental Outlook: {fundamental_outlook.capitalize()}")
    
    # Final recommendation
    print("\nFinal Analysis:")
    if technical_outlook in ["strongly bullish", "moderately bullish"] and fundamental_outlook == "potentially undervalued":
        print("Strong Buy: Positive technical momentum with fundamental undervaluation")
    elif technical_outlook in ["strongly bullish", "moderately bullish"] and fundamental_outlook == "fairly valued":
        print("Buy: Positive technical momentum with fair valuation")
    elif technical_outlook == "neutral" and fundamental_outlook == "potentially undervalued":
        print("Accumulate: Fair technical outlook but fundamentally undervalued")
    elif technical_outlook in ["strongly bearish", "moderately bearish"] and fundamental_outlook == "potentially overvalued":
        print("Strong Sell: Negative technical momentum with fundamental overvaluation")
    elif technical_outlook in ["strongly bearish", "moderately bearish"] and fundamental_outlook == "fairly valued":
        print("Sell: Negative technical momentum despite fair valuation")
    elif technical_outlook == "neutral" and fundamental_outlook == "potentially overvalued":
        print("Reduce: Fair technical outlook but fundamentally overvalued")
    else:
        print("Hold/Neutral: Mixed signals or insufficient data")

def main():
    """Main function to run the test"""
    # Test with different stocks to demonstrate versatility
    stocks = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in stocks:
        perform_combined_analysis(symbol)
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
