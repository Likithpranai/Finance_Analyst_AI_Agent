"""
Finance Analyst AI Agent - Demo Script
"""
import os
import time
from dotenv import load_dotenv
from agent import create_agent_executor, run_agent

# Load environment variables from .env file
load_dotenv()

# Check for required API keys
if not os.getenv("GEMINI_API_KEY"):
    print("\033[91mError: GEMINI_API_KEY not found in environment variables.\033[0m")
    print("Please create a .env file with your Gemini API key:")
    print("GEMINI_API_KEY=your_key_here")
    exit(1)


def display_header(text):
    """Display a formatted header."""
    print("\n\033[1m\033[94m" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\033[0m\n")


def run_demo():
    """Run demo scenarios to showcase the Finance Analyst AI Agent."""
    display_header("Finance Analyst AI Agent Demo")
    
    # Initialize agent
    print("\033[92mInitializing agent...\033[0m")
    finance_agent = create_agent_executor()
    print("\033[92mAgent ready!\033[0m")
    
    # Demo scenarios
    scenarios = [
        {
            "title": "Basic Stock Price Information",
            "query": "What's the current price of AAPL and how has it performed over the past week?"
        },
        {
            "title": "Technical Indicator Analysis",
            "query": "Calculate RSI and MACD for TSLA and tell me if it's overbought or oversold."
        },
        {
            "title": "Market News and Sentiment",
            "query": "What's the latest news about NVDA and what's the overall market sentiment?"
        },
        {
            "title": "Investment Analysis",
            "query": "Compare AMZN and MSFT based on technical indicators and recent performance. Which looks better for a short-term investment?"
        }
    ]
    
    # Run each scenario
    for i, scenario in enumerate(scenarios, 1):
        display_header(f"Demo {i}: {scenario['title']}")
        print(f"\033[93mQuery: {scenario['query']}\033[0m\n")
        
        print("\033[90mAnalyzing...\033[0m")
        
        # Run the agent
        response = run_agent(finance_agent, scenario['query'])
        
        # Print the response
        print("\n\033[1m\033[94mResponse:\033[0m")
        print(f"\033[97m{response['answer']}\033[0m")
        
        if i < len(scenarios):
            print("\n\033[90mNext demo starting in 3 seconds...\033[0m")
            time.sleep(3)
    
    display_header("Demo Completed")
    print("\033[92mYou can now run 'python main.py' to interact with the agent directly!\033[0m")


if __name__ == "__main__":
    run_demo()
