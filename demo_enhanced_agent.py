#!/usr/bin/env python3
"""
Demo script for the Enhanced Finance Analyst AI Agent
Demonstrates the capabilities of the multi-agent architecture
"""
import os
import sys
import argparse
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv
import time
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Import enhanced agent
from enhanced_finance_analyst import EnhancedFinanceAnalystAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FinanceAnalystAI-Demo")

# Load environment variables
load_dotenv()

class EnhancedAgentDemo:
    """Demo for the Enhanced Finance Analyst AI Agent"""
    
    def __init__(self):
        """Initialize the demo"""
        self.agent = EnhancedFinanceAnalystAI()
        self.demo_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "=" * 80)
        print(f" {title} ".center(80, "="))
        print("=" * 80 + "\n")
    
    def print_section(self, title: str):
        """Print a formatted section header"""
        print("\n" + "-" * 80)
        print(f" {title} ".center(80, "-"))
        print("-" * 80 + "\n")
    
    def run_demo_query(self, query: str):
        """Run a demo query and print the response"""
        self.print_section(f"QUERY: {query}")
        
        print("Processing query...")
        start_time = time.time()
        response = self.agent.process_query(query)
        elapsed_time = time.time() - start_time
        
        print(f"\nRESPONSE (processed in {elapsed_time:.2f} seconds):")
        print(response)
        print("\n")
        
        return response
    
    def demo_technical_analysis(self):
        """Demo technical analysis capabilities"""
        self.print_header("TECHNICAL ANALYSIS DEMO")
        
        # Simple technical analysis
        self.run_demo_query(f"Show me the technical indicators for {self.demo_symbols[0]} including RSI, MACD, and Bollinger Bands")
        
        # Chart pattern analysis
        self.run_demo_query(f"Identify any chart patterns for {self.demo_symbols[1]} in the last 3 months")
        
        # Multi-timeframe analysis
        self.run_demo_query(f"Compare the technical outlook for {self.demo_symbols[2]} across daily, weekly, and monthly timeframes")
    
    def demo_fundamental_analysis(self):
        """Demo fundamental analysis capabilities"""
        self.print_header("FUNDAMENTAL ANALYSIS DEMO")
        
        # Financial ratios
        self.run_demo_query(f"What are the key financial ratios for {self.demo_symbols[0]}?")
        
        # Financial statement analysis
        self.run_demo_query(f"Analyze the income statement trends for {self.demo_symbols[1]} over the past 3 years")
        
        # Industry comparison
        self.run_demo_query(f"Compare {self.demo_symbols[2]} with its industry peers based on fundamental metrics")
    
    def demo_risk_analysis(self):
        """Demo risk analysis capabilities"""
        self.print_header("RISK ANALYSIS DEMO")
        
        # Volatility analysis
        self.run_demo_query(f"Calculate the volatility and Value at Risk for {self.demo_symbols[0]}")
        
        # Portfolio risk
        portfolio = ", ".join(self.demo_symbols[:3])
        self.run_demo_query(f"Analyze the risk profile of a portfolio containing {portfolio}")
        
        # Risk-adjusted returns
        self.run_demo_query(f"Calculate the Sharpe and Sortino ratios for {self.demo_symbols[1]}")
    
    def demo_trading_strategies(self):
        """Demo trading strategy capabilities"""
        self.print_header("TRADING STRATEGY DEMO")
        
        # Trading signals
        self.run_demo_query(f"Generate trading signals for {self.demo_symbols[0]} based on technical indicators")
        
        # Strategy backtesting
        self.run_demo_query(f"Backtest a simple moving average crossover strategy for {self.demo_symbols[1]}")
        
        # Trade execution simulation
        self.run_demo_query(f"Simulate a trade execution for buying 100 shares of {self.demo_symbols[2]}")
    
    def demo_multi_agent_collaboration(self):
        """Demo multi-agent collaboration capabilities"""
        self.print_header("MULTI-AGENT COLLABORATION DEMO")
        
        # Comprehensive analysis
        self.run_demo_query(f"Provide a comprehensive analysis of {self.demo_symbols[0]} including technical, fundamental, and risk factors")
        
        # Investment recommendation
        self.run_demo_query(f"Should I invest in {self.demo_symbols[1]} right now? Consider all relevant factors")
        
        # Portfolio optimization
        portfolio = ", ".join(self.demo_symbols)
        self.run_demo_query(f"Optimize a portfolio containing {portfolio} for maximum Sharpe ratio")
    
    def demo_sentiment_analysis(self):
        """Demo sentiment analysis capabilities"""
        self.print_header("SENTIMENT ANALYSIS DEMO")
        
        # Company-specific sentiment analysis
        self.run_demo_query(f"Analyze the news sentiment for {self.demo_symbols[0]} over the past week")
        
        # Market sentiment analysis
        self.run_demo_query("What is the overall market sentiment today based on financial news?")
        
        # Sentiment comparison
        self.run_demo_query(f"Compare the sentiment between {self.demo_symbols[0]} and {self.demo_symbols[1]} based on recent news")
        
        # Sentiment impact analysis
        self.run_demo_query(f"How is the current news sentiment likely to affect {self.demo_symbols[2]}'s stock price?")
        
        # Social media sentiment
        self.run_demo_query(f"Analyze social media sentiment for {self.demo_symbols[3]}")
    
    def demo_memory_capabilities(self):
        """Demo memory capabilities"""
        self.print_header("MEMORY CAPABILITIES DEMO")
        
        # Add watched symbols
        for symbol in self.demo_symbols[:3]:
            self.agent.add_watched_symbol(symbol)
        
        # Show watched symbols
        watched_symbols = self.agent.get_watched_symbols()
        self.print_section("WATCHED SYMBOLS")
        print(", ".join(watched_symbols))
        
        # Run queries that reference previous analysis
        self.run_demo_query(f"What was my last analysis of {self.demo_symbols[0]}?")
        
        # Show conversation history
        conversations = self.agent.get_recent_conversations()
        self.print_section("RECENT CONVERSATIONS")
        
        if conversations:
            for i, conv in enumerate(conversations):
                print(f"{i+1}. Query: {conv['user_query']}")
                print(f"   Symbols: {', '.join(conv['symbols']) if conv['symbols'] else 'None'}")
                print(f"   Analysis Types: {', '.join(conv['analysis_types']) if conv['analysis_types'] else 'None'}")
                print(f"   Time: {conv['timestamp']}")
                print()
        else:
            print("No conversation history")
    
    def run_full_demo(self):
        """Run the full demonstration"""
        self.print_header("ENHANCED FINANCE ANALYST AI AGENT DEMO")
        print("This demo showcases the capabilities of the Enhanced Finance Analyst AI Agent")
        print("with its new multi-agent architecture for financial analysis and decision-making.")
        
        # Check for API keys
        if not os.getenv("GEMINI_API_KEY"):
            print("\nWARNING: GEMINI_API_KEY not found in environment variables.")
            print("The demo will not work without a valid API key.")
            return
        
        # Run individual demos
        self.demo_technical_analysis()
        self.demo_fundamental_analysis()
        self.demo_risk_analysis()
        self.demo_trading_strategies()
        self.demo_sentiment_analysis()
        self.demo_multi_agent_collaboration()
        self.demo_memory_capabilities()
        
        self.print_header("DEMO COMPLETED")
        print("The Enhanced Finance Analyst AI Agent demo has completed successfully.")
        print("You can now use the agent for your own financial analysis and decision-making.")

def main():
    """Main entry point for the demo"""
    parser = argparse.ArgumentParser(description="Enhanced Finance Analyst AI Agent Demo")
    parser.add_argument("--technical", action="store_true", help="Run technical analysis demo")
    parser.add_argument("--fundamental", action="store_true", help="Run fundamental analysis demo")
    parser.add_argument("--risk", action="store_true", help="Run risk analysis demo")
    parser.add_argument("--trading", action="store_true", help="Run trading strategy demo")
    parser.add_argument("--sentiment", action="store_true", help="Run sentiment analysis demo")
    parser.add_argument("--multi-agent", action="store_true", help="Run multi-agent collaboration demo")
    parser.add_argument("--memory", action="store_true", help="Run memory capabilities demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize the demo
    demo = EnhancedAgentDemo()
    
    # Run selected demos
    if args.all or (not any([args.technical, args.fundamental, args.risk, args.trading, args.sentiment, args.multi_agent, args.memory])):
        demo.run_full_demo()
    else:
        if args.technical:
            demo.demo_technical_analysis()
        if args.fundamental:
            demo.demo_fundamental_analysis()
        if args.risk:
            demo.demo_risk_analysis()
        if args.trading:
            demo.demo_trading_strategies()
        if args.sentiment:
            demo.demo_sentiment_analysis()
        if args.multi_agent:
            demo.demo_multi_agent_collaboration()
        if args.memory:
            demo.demo_memory_capabilities()

if __name__ == "__main__":
    main()
