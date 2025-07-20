#!/usr/bin/env python3
"""
Run a query through the Finance Analyst AI Agent with valid API keys
"""

import os
import sys
import json
from finance_analyst_agent import FinanceAnalystReActAgent

# Set the API keys directly in environment variables
os.environ["ALPHA_VANTAGE_API_KEY"] = "I9H30JO7WPUD9ECS"
os.environ["GEMINI_API_KEY"] = "AIzaSyDcl9MVfMuzATS6PVQTuqbCDbPlFoOKiJ8"

def main():
    """Run a query through the Finance Analyst AI Agent"""
    # Default query if none provided
    default_query = "Provide a comprehensive analysis of MSFT including technical indicators, fundamental data, and news impact"
    
    # Get query from command line arguments or use default
    query = sys.argv[1] if len(sys.argv) > 1 else default_query
    
    print(f"\nRunning query: '{query}'")
    print("\nInitializing Finance Analyst AI Agent...")
    
    try:
        # Initialize the agent
        agent = FinanceAnalystReActAgent()
        
        print("\nProcessing query...")
        # Process the query
        response = agent.process_query(query)
        
        # Print the response
        print("\n" + "="*80)
        print("RESPONSE:")
        print("="*80)
        
        if isinstance(response, dict):
            if 'text' in response:
                print(response['text'])
            else:
                print(json.dumps(response, indent=2))
        else:
            print(response)
            
        print("="*80)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
