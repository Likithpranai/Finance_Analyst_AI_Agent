#!/usr/bin/env python3
"""
Finance Analyst AI Agent - Unified Entry Point

This script provides a unified interface to the Finance Analyst AI Agent,
combining all functionality including:
- ReAct-based AI analysis
- Interactive visualization
- Real-time data processing
- Technical and fundamental analysis
- Portfolio management
- Dashboard capabilities

Usage:
    python3 finance_agent.py [--dashboard] [--query "your query here"]
"""

import os
import sys
import time
import argparse
import subprocess
from finance_analyst_agent import FinanceAnalystReActAgent

def run_query(query=None):
    """Run a query through the Finance Analyst AI Agent"""
    print("Initializing Finance Analyst AI Agent...")
    agent = FinanceAnalystReActAgent()
    
    if not query:
        print("\nEnter your financial query (type 'exit' to quit):")
        while True:
            query = input("\n> ")
            if query.lower() in ['exit', 'quit']:
                break
                
            print("-" * 80)
            start_time = time.time()
            response = agent.process_query(query)
            end_time = time.time()
            
            print("\nResponse:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            print(f"Query processed in {end_time - start_time:.2f} seconds")
    else:
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

def run_dashboard():
    """Launch the interactive Streamlit dashboard"""
    print("Launching Finance Analyst Dashboard...")
    try:
        subprocess.run([sys.executable, "dashboard.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")

def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Finance Analyst AI Agent")
    parser.add_argument("--dashboard", action="store_true", help="Launch the interactive dashboard")
    parser.add_argument("--query", type=str, help="Run a specific query")
    
    args = parser.parse_args()
    
    if args.dashboard:
        run_dashboard()
    else:
        run_query(args.query)

if __name__ == "__main__":
    main()
