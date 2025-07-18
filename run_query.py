#!/usr/bin/env python3
"""
Script to run a specific query through the Finance Analyst AI Agent
"""

import os
import time
from finance_analyst_agent import FinanceAnalystReActAgent

def main():
    """Run a specific query through the agent"""
    print("Initializing Finance Analyst AI Agent...")
    agent = FinanceAnalystReActAgent()
    
    # Define your query here - using features available in Polygon.io's free tier
    query = "Compare company details for AAPL, MSFT, and GOOGL and recommend which has the best growth potential"
    
    print(f"\nProcessing query: '{query}'")
    print("-" * 80)
    
    start_time = time.time()
    response = agent.process_query(query)
    end_time = time.time()
    
    print("\nResponse:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    print(f"Query processed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
