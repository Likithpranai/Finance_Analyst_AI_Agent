"""
Finance Analyst AI Agent - Main Entry Point
"""
import os
import sys
from dotenv import load_dotenv
from agent import create_agent_executor, run_agent

# Load environment variables from .env file
load_dotenv()

# Check for required API keys
if not os.getenv("GEMINI_API_KEY"):
    print("\033[91mError: GEMINI_API_KEY not found in environment variables.\033[0m")
    print("Please create a .env file with your Gemini API key:")
    print("GEMINI_API_KEY=your_key_here")
    sys.exit(1)


def main():
    """Main function to run the finance analyst agent."""
    print("\033[1m\033[94m" + "=" * 80)
    print("Finance Analyst AI Agent".center(80))
    print("=" * 80 + "\033[0m")
    
    print("\033[92mInitializing agent...\033[0m")
    finance_agent = create_agent_executor()
    print("\033[92mAgent ready!\033[0m")
    
    # Sample queries to help users get started
    sample_queries = [
        "What's the current price of AAPL?",
        "Calculate the RSI for TSLA",
        "Compare the moving averages for MSFT",
        "What's the latest news about AMZN?",
        "Is NVDA overbought or oversold based on technical indicators?",
        "Plot GOOG stock price for the past month"
    ]
    
    print("\033[93m\nSample queries you can try:\033[0m")
    for i, query in enumerate(sample_queries, 1):
        print(f"\033[93m{i}. {query}\033[0m")
    
    chat_history = []
    
    while True:
        try:
            print("\n\033[1m\033[96mAsk a financial question (or type 'exit' to quit):\033[0m")
            query = input("> ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\033[92mThank you for using Finance Analyst AI Agent. Goodbye!\033[0m")
                break
                
            if not query.strip():
                continue
                
            print("\033[90mAnalyzing...\033[0m")
            
            # Run the agent
            response = run_agent(finance_agent, query, chat_history)
            
            # Print the response
            print("\n\033[1m\033[94mResponse:\033[0m")
            print(f"\033[97m{response['answer']}\033[0m")
            
            # Update chat history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response['answer']})
            
            # Limit chat history to last 10 exchanges
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
                
        except KeyboardInterrupt:
            print("\n\033[92mExiting gracefully...\033[0m")
            break
            
        except Exception as e:
            print(f"\033[91mAn error occurred: {str(e)}\033[0m")


if __name__ == "__main__":
    main()
